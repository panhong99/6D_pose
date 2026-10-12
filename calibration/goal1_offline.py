"""A3 candidate optimization / workspace mapping (offline only, no pipeline execution).

Checks sampled simulator robot/gripper geometry; physical adapter and carried dish remain unmodelled.
Reports describe model clearance, not a real-robot safety certificate.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calib_core import load_config, load_json, latest_t_base_cam
from plate_plans import library_checkers,nearest_direction_candidates
from robot_model import CollisionChecker, HOME_JOINTS_DEG, polyline_path, smooth_path, time_parameterize, joint_limits_check

ROOT=Path(__file__).resolve().parents[1]


class CombinedChecker(CollisionChecker):
    """Margins remain obstacle-specific; planner sees the minimum extra clearance beyond them."""
    def __init__(self,checks,extra=0.010):
        self.checks=checks;self.kin=next(iter(checks.values()))[0].kin
        self.margin=0.0;self.plan_margin=extra

    def clearance(self,q):
        return min(c.clearance(q)-margin for c,margin in self.checks.values())

    def path_clearance(self,path):
        """Batch the same sampled geometry/voxel query as CollisionChecker; no samples dropped."""
        minimum=float('inf')
        for offset in range(0,len(path),32):
            Q=np.asarray(path[offset:offset+32]);n=len(Q)
            T=np.broadcast_to(np.eye(4),(n,4,4)).copy();frames={}
            for j,(name,origin,axis,_,_) in enumerate(self.kin.joints):
                turn=np.broadcast_to(np.eye(4),(n,4,4)).copy()
                turn[:,:3,:3]=Rotation.from_rotvec(Q[:,j,None]*axis).as_matrix()
                T=T @ origin @ turn;frames[name]=T.copy()
            for checker,margin in self.checks.values():
                clouds=[]
                for name,points in checker.link_points.items():
                    F=frames[name]
                    clouds.append(np.einsum('njk,pk->npj',F[:,:3,:3],points)+F[:,None,:3,3])
                if checker.tool is not None:
                    F=frames['link_6']
                    clouds.append(np.einsum('njk,pk->npj',F[:,:3,:3],checker.tool)+F[:,None,:3,3])
                P=np.concatenate(clouds,axis=1).reshape(-1,3)
                idx=np.floor((P-checker.lo)/checker.res).astype(int)
                inside=np.all((idx>=0)&(idx<np.asarray(checker.distance.shape)),axis=1)
                if inside.any():
                    minimum=min(minimum,float(checker.distance[tuple(idx[inside].T)].min())-margin)
        return minimum


def checkers():
    q=dict(load_config()['plate_ik_bc'],tool_model='sim_open_gripper')
    doc=load_json(latest_t_base_cam())
    T=np.asarray(doc['T_base_cam'])
    checks=library_checkers(T,q)
    return checks


def evaluate_path(path,checker,dt=.05,return_frames=False):
    dense=smooth_path(path,.25)
    extra=float(checker.path_clearance(dense))
    speed_budget,acc_budget=6.0,6.0
    from grasp_candidates import tcp_transform
    tool_T=tcp_transform(ROOT/'rl/assets/m1013_2f85.urdf')
    for _ in range(8):
        frames=time_parameterize(dense,np.radians(speed_budget),np.radians(acc_budget),dt)
        padded=np.vstack([frames[0],frames,frames[-1]])
        velocity=np.diff(padded,axis=0)/dt
        speed=float(np.abs(velocity).max())
        acc=float(np.abs(np.diff(velocity,axis=0)/dt).max())
        transforms=np.broadcast_to(np.eye(4),(len(frames),4,4)).copy()
        for j,(_,origin,axis,_,_) in enumerate(checker.kin.joints):
            turn=np.broadcast_to(np.eye(4),transforms.shape).copy()
            turn[:,:3,:3]=Rotation.from_rotvec(frames[:,j,None]*axis).as_matrix()
            transforms=transforms @ origin @ turn
        transforms=transforms @ tool_T
        tcp_speed=float(np.linalg.norm(np.diff(transforms[:,:3,3],axis=0)/dt,axis=1).max())
        relative=np.swapaxes(transforms[:-1,:3,:3],1,2) @ transforms[1:,:3,:3]
        rot_speed=float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(),axis=1)/dt).max())
        if np.degrees(speed)<=6.001 and np.degrees(acc)<=6.001 and tcp_speed<=.05001 and rot_speed<=10.001:
            break
        # Retiming can overshoot a nominal acceleration budget after interpolation/filtering.
        # Reduce budgets and measure the resulting samples; never raise the acceptance limit.
        speed_budget*=min(.95,6/max(float(np.degrees(speed)),1e-9),.049/max(tcp_speed,1e-9),9.8/max(rot_speed,1e-9))
        acc_budget*=min(.8,5.8/max(float(np.degrees(acc)),1e-9))
    # The time filter changes the geometric path: validate that trajectory too.
    extra=min(extra,float(checker.path_clearance(frames)))
    metrics=dict(duration_s=(len(frames)-1)*dt,
                joint_travel_sum_deg=float(np.degrees(np.abs(np.diff(dense,axis=0))).sum()),
                waypoints=len(path),extra_clearance_mm=extra*1000,
                max_joint_speed_deg_s=float(np.degrees(speed)),max_joint_accel_deg_s2=float(np.degrees(acc)),
                max_model_tcp_speed_mps=tcp_speed,max_model_tool_rot_speed_deg_s=rot_speed)
    return (metrics,frames) if return_frames else metrics


def optimize(library,output,max_rrt=600):
    output.mkdir(parents=True,exist_ok=True)
    with np.load(library,allow_pickle=False) as data:L={k:data[k] for k in ('ok','q_pre_deg','radius_m')}
    checker=CombinedChecker(checkers())
    home=np.radians(HOME_JOINTS_DEG);rows=[];paths=[];indices=[];timed=[]
    def solve(i):
        pre=np.radians(L['q_pre_deg'][i])
        original=np.array([home,pre]);before,before_frames=evaluate_path(original,checker,return_frames=True)
        try:
            if before['extra_clearance_mm']>=10:
                selected=original;method='direct_already_shortest_joint_line'
            else:
                path,method=checker.plan(home,pre,max_nodes=max_rrt,seed=int(i))
                selected=np.asarray(path)
            if method=='direct_already_shortest_joint_line':
                after,frames=before.copy(),before_frames
            else:
                after,frames=evaluate_path(selected,checker,return_frames=True)
            accepted=(after['extra_clearance_mm']>=10-1e-6 and after['max_joint_speed_deg_s']<=6.001
                      and after['max_joint_accel_deg_s2']<=6.001 and after['max_model_tcp_speed_mps']<=.05001
                      and after['max_model_tool_rot_speed_deg_s']<=10.001)
            row=dict(index=int(i),radius_m=float(L['radius_m'][i]),before=before,after=after,
                     method=method,accepted=bool(accepted))
            return row,np.degrees(selected),np.degrees(frames)
        except RuntimeError as exc:
            return dict(index=int(i),radius_m=float(L['radius_m'][i]),before=before,accepted=False,reason=str(exc)),None,None
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row,path,frames in pool.map(solve,np.flatnonzero(L['ok'])):
            rows.append(row)
            if row['accepted']:paths.append(path);indices.append(row['index']);timed.append(frames)
            print(f"OPT {row['index']} r={row['radius_m']:.3f} {row.get('method',row.get('reason'))} accepted={row['accepted']}",flush=True)
    maxlen=max(map(len,paths),default=0);padded=np.zeros((len(paths),maxlen,6))
    for j,path in enumerate(paths):padded[j,:len(path)]=path;padded[j,len(path):]=path[-1]
    np.savez_compressed(output/'optimized_prepaths.npz',library_index=indices,q_deg=padded,length=[len(x) for x in paths])
    maxt=max(map(len,timed),default=0);frames_padded=np.zeros((len(timed),maxt,6))
    for j,path in enumerate(timed):frames_padded[j,:len(path)]=path;frames_padded[j,len(path):]=path[-1]
    np.savez_compressed(output/'retimed_prepaths.npz',library_index=indices,q_deg=frames_padded,
                        length=[len(x) for x in timed],dt_s=.05,candidate_only=True)
    report=dict(candidate_only=True,physical_adapter_modelled=False,carried_plate_modelled=False,
                library=str(library),extra_margin_mm=10,joint_speed_cap_deg_s=6,rows=rows,
                n_input=int(L['ok'].sum()),n_accepted=sum(x['accepted'] for x in rows),
                note='Only home-to-pre approach is optimized. BC samples are unchanged. Direct joint lines already '
                     'minimize joint travel; rerouting improves clearance but may lengthen time.')
    (output/'optimization.json').write_text(json.dumps(report,indent=2))
    return report


def workspace(library,output,step=.05,tolerance=.004):
    """5 cm Cartesian grid, choose the best surviving phi/pitch after full-path FK checks."""
    # NPZ access decompresses an entire array each time; retain it once for this offline scan.
    with np.load(library,allow_pickle=False) as archive:L={k:archive[k] for k in archive.files}
    checks=checkers();checker=CombinedChecker(checks)
    good=np.flatnonzero(L['ok']);rows=[]
    home=np.radians(HOME_JOINTS_DEG)
    from concurrent.futures import ThreadPoolExecutor
    def point_row(point):
        x,y=point
        r=float(np.hypot(x,y));az=float(np.arctan2(y,x))
        in_band=.85<=r<=1.32
        candidates=nearest_direction_candidates(L,r,tolerance) if in_band else np.array([],dtype=int)
        best=None;closest=float(np.min(np.abs(L['radius_m'][good]-r))) if len(good) else None
        for i in candidates:
            end=int(L['success_step'][i]);c=int(L['close_step'][i])
            Q=np.radians(L['q_deg'][:end+1,i].astype(float));Q[:,0]+=az
            # A subset's minimum is an upper bound on the full path minimum.
            # Skip a candidate only when even that bound cannot beat the fully checked best.
            bound=checker.path_clearance(np.vstack([home,Q[[0,c,end]]]))
            if best is not None and bound<=best['extra_clearance_mm']/1000:
                continue
            approach=polyline_path([home,Q[0]],.5)
            # Keep all recorded BC samples; no straight shortcuts in the contact phase.
            path=np.vstack([approach,Q])
            extra=checker.path_clearance(path)
            if best is None or extra>best['extra_clearance_mm']/1000:
                best=dict(index=int(i),radius_error_mm=float(abs(L['radius_m'][i]-r)*1000),
                          phi_deg=float(L['phi_deg'][i]) if 'phi_deg' in L else 0,
                          pitch_deg=float(L['pitch_deg'][i]) if 'pitch_deg' in L else 0,
                          extra_clearance_mm=float(extra*1000))
        accepted=best is not None and best['extra_clearance_mm']>=10
        return dict(x_m=float(x),y_m=float(y),radius_m=r,accepted=bool(accepted),best=best,
                         reason='outside_radius_band' if not in_band else ('candidate' if accepted else ('no_radius_candidate' if best is None else 'cell_clearance')),
                         nearest_radius_error_m=closest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for x in np.arange(.87,1.63+1e-9,step):
            rows.extend(pool.map(point_row,[(x,y) for y in np.arange(-.52,.52+1e-9,step)]))
            print(f'MAP x={x:.2f} accepted={sum(a["accepted"] for a in rows)}/{len(rows)}',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    result=dict(candidate_only=True,interpolated_radius_not_physics_validated=True,
                lookup_tolerance_m=tolerance,grid_step_m=step,physical_adapter_modelled=False,carried_plate_modelled=False,
                library=str(library),rows=rows,n=len(rows),n_accepted=sum(a['accepted'] for a in rows))
    (output/'workspace_map.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for good,label,color in [(True,'Model candidate (not physics validated)','tab:blue'),(False,'Rejected','tab:red')]:
        subset=[a for a in rows if a['accepted']==good]
        ax.scatter([a['x_m'] for a in subset],[a['y_m'] for a in subset],s=15,label=label,color=color)
    ax.set_xlabel('Base x (m)');ax.set_ylabel('Base y (m)');ax.set_aspect('equal')
    ax.spines[['top','right']].set_visible(False);ax.grid(False);ax.legend(frameon=False,bbox_to_anchor=(0,1.25),loc='upper left')
    fig.tight_layout();fig.savefig(output/'workspace_map.png',dpi=180,bbox_inches='tight');plt.close(fig)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['optimize','map'])
    p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--step',type=float,default=.05);p.add_argument('--rrt_nodes',type=int,default=600)
    p.add_argument('--radius_tolerance',type=float,default=.004)
    a=p.parse_args()
    if a.mode=='optimize':r=optimize(a.library,a.output,a.rrt_nodes)
    else:r=workspace(a.library,a.output,a.step,a.radius_tolerance)
    print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
