"""A3 candidate optimization / workspace mapping (offline only, no pipeline execution).

Checks sampled simulator robot/gripper geometry; physical adapter and carried dish remain unmodelled.
Reports describe model clearance, not a real-robot safety certificate.
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calib_core import load_config, load_json, latest_t_base_cam
from plate_plans import library_checkers,nearest_direction_candidates
from robot_model import CollisionChecker, HOME_JOINTS_DEG, polyline_path, smooth_path, time_parameterize, joint_limits_check

ROOT=Path(__file__).resolve().parents[1]
_BASKET_REACH = {}


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


def _basket_gap_ik(Tbb, qd, gap_index, face, start_deg):
    """Thirteen-point vertical insertion IK; deliberately excludes carry/RRT/collision checks."""
    G=_BASKET_REACH; pp=G['pp']; q=G['q']
    g=np.asarray(G['gaps'][qd][gap_index],dtype=float).copy()
    room=G['basket_size_y_m']/2-.012-q.get('basket_wall_margin_m',.01)-pp.PLATE_R-.005
    if abs(g[1])>room:g[1]+=float(np.clip(0.,-room-g[1],room-g[1]))
    g[0]+=float(q.get('basket_quadrant_x_trim_m',0.))
    gap_base=(Tbb@np.r_[g,1.])[:3]
    n=Tbb[:3,0];t=np.radians(0.)
    v=-n*np.sin(t)+np.array([0.,0.,1.])*np.cos(t)
    phis=sorted(range(-90,91,10),key=abs)
    for side in (face,):
        zt=side*(n*np.cos(t)+np.array([0.,0.,1.])*np.sin(t))
        for phi_deg in phis:
            ph=np.radians(phi_deg);x=v*np.cos(ph)+np.cross(zt,v)*np.sin(ph)
            Tp=np.eye(4);Tp[:3,:3]=np.stack([x,np.cross(zt,x),zt],axis=1)
            Tp[:3,3]=gap_base+v*G['centre_h']
            T6=Tp@G['H_inv']@G['tool_inv'];prev=None;ok=True
            for lift in np.linspace(G['above_m'],-G['extra_down_m'],13):
                target=T6.copy();target[2,3]+=lift
                seeds=[prev] if prev is not None else [np.radians(start_deg),np.radians(G['home_deg'])]
                sol=None
                for seed in seeds:
                    try:cand=G['kin'].ik(target,seed)
                    except RuntimeError:continue
                    deg=np.degrees(cand)
                    if (np.all(deg>np.asarray(G['lo_deg'])+3) and np.all(deg<np.asarray(G['hi_deg'])-3)
                            and (prev is None or np.max(np.degrees(np.abs(cand-prev)))<8)):
                        sol=cand;break
                if sol is None:ok=False;break
                prev=sol
            if ok:return True
    return False


def _basket_lift_ratio(Tbb,start_deg):
    """Held-plate clearance ratio immediately after lift, using static cell fields + basket OBB walls."""
    G=_BASKET_REACH;pp=G['pp'];qrad=np.radians(start_deg)
    key=tuple(np.round(Tbb.ravel(),5))
    if key not in G['basket_cache']:G['basket_cache'][key]=pp.basket_boxes(Tbb)
    points=G['frame_checker'].robot_points(qrad)
    # Cell checkers use the same voxelized obstacle fields and the full plate + gripper point cloud.
    ratios=[G['frame_checker'].clearance(qrad)/G['frame_checker'].margin,
            G['table_checker'].clearance(qrad)/G['table_checker'].margin]
    wall_min=float('inf')
    for box in G['basket_cache'][key]:
        _,centre,size,R=box;local=(points-np.asarray(centre))@np.asarray(R)
        d=np.abs(local)-np.asarray(size)/2
        outside=np.linalg.norm(np.maximum(d,0.),axis=1)
        outside[np.all(d<=0,axis=1)]=0.
        wall_min=min(wall_min,float(outside.min()))
    ratios.append(wall_min/float(G['q'].get('basket_wall_margin_m',.01)))
    return float(min(ratios))


def _basket_reach_job(job):
    dish_id,px,py,bx,by,yaw=job;G=_BASKET_REACH;pp=G['pp']
    # Lift pose depends on dish XY, not basket XY/yaw; cache it per worker.
    # Rotating J1 alone preserves the wrong source radius at shifted dish poses.
    key=(round(px,6),round(py,6));cache=G['lift_solution_cache']
    if key not in cache:
        source_q=np.radians(G['lift_deg'])
        Tplate=G['kin'].fk(source_q)@G['tool']@G['H']
        Tplate[:3,3]+=[px-G['source_plate_xy'][0],py-G['source_plate_xy'][1],0.]
        target_flange=Tplate@G['H_inv']@G['tool_inv'];start=None
        for seed in (source_q,np.radians(G['home_deg'])):
            try:cand=G['kin'].ik(target_flange,seed)
            except RuntimeError:continue
            d=np.degrees(cand)
            if np.all(d>np.asarray(G['lo_deg'])+3) and np.all(d<np.asarray(G['hi_deg'])-3):start=d;break
        cache[key]=start
    start=cache[key]
    if start is None:
        return {'dish_id':dish_id,'dish_xy_m':[px,py],'basket_xy_m':[bx,by],'yaw_deg':yaw,
                'plate_lift_ik_ok':False,
                'quadrants':{qd:{face:[0]*len(gs) for face in ('plus_x','minus_x')} for qd,gs in G['gaps'].items()},
                'lift_clearance_ratio':None}
    c,s=np.cos(np.radians(yaw)),np.sin(np.radians(yaw));Tbb=np.eye(4)
    Tbb[:2,:2]=[[c,-s],[s,c]];Tbb[:3,3]=[bx,by,G['basket_z']]
    out={}
    for qd,gaps in G['gaps'].items():
        out[qd]={}
        for face,name in ((1,'plus_x'),(-1,'minus_x')):
            out[qd][name]=[int(_basket_gap_ik(Tbb,qd,gi,face,start)) for gi in range(len(gaps))]
    clearance=_basket_lift_ratio(Tbb,start)
    return {'dish_id':dish_id,'dish_xy_m':[px,py],'basket_xy_m':[bx,by],'yaw_deg':yaw,
            'plate_lift_ik_ok':True,'quadrants':out,'lift_clearance_ratio':clearance}


def _basket_reach_init(context):
    global _BASKET_REACH
    _BASKET_REACH=context


def basket_reach_map(output,workers=12,smoke=False):
    """CPU-only reach-rate and post-lift clearance maps for the 2026-10-14 parallel layout."""
    import plate_plans as pp
    from calib_core import DATA,latest_t_base_cam,load_config,load_json
    from robot_model import HOME_JOINTS_DEG,Kinematics
    import yaml
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    H,tool,run=pp.plate_in_hand();run_dir=DATA/'real_pipeline'/run
    req=json.loads((run_dir/'realplan_req.json').read_text())
    event_rows=[json.loads(x) for x in (run_dir/'robot_state.jsonl').read_text().splitlines() if '"event"' in x]
    plan=Path(next(x['plan'] for x in event_rows if x.get('event')=='plan'))
    lift_deg=json.loads(plan.read_text())['legs']['lift']['waypoints_deg'][-1]
    q=load_config()['plate_ik_bc'];kin=Kinematics()
    basket_config=yaml.safe_load((ROOT/'rl'/'basket.yaml').read_text())['full']
    limits=np.asarray([[j[3],j[4]] for j in kin.joints]);lo_deg,hi_deg=np.degrees(limits[:,0]),np.degrees(limits[:,1])
    lo_deg[4],hi_deg[4]=-135.,135.
    Tbc=np.asarray(load_json(latest_t_base_cam())['T_base_cam'],float)
    checks=pp.library_checkers(Tbc,q)
    # Add the same plate point cloud used by quadrant_checkers to the collision tool samples.
    plate=pp.plate_cloud(H,tool)
    for key in ('frame','table'):
        checker=checks[key][0];checker.tool=np.vstack([checker.tool,plate])
    frame_checker,table_checker=checks['frame'][0],checks['table'][0]
    source=np.asarray(req['plate_root_fp'][:2],float)
    basket_z=.1442
    if smoke:
        # Small boundary smoke: both ends of basket/dish offsets and yaw signs.
        xs=[.707,.907];ys=[-.292,-.092];yaws=[-15.,15.]
        pxs=[.798,.948];pys=[.239,.359]
    else:
        xs=np.round(np.arange(.707,.907+1e-9,.025),3).tolist()
        ys=np.round(np.arange(-.292,-.092+1e-9,.025),3).tolist()
        yaws=[0.,-10.,10.,-15.,15.]
        pxs=np.round(.848+np.array([-.05,0.,.05,.10]),3).tolist()
        pys=np.round(.339+np.array([-.10,-.06,-.02,.02]),3).tolist()
    gaps={k:[np.asarray(g,float) for g in v] for k,v in pp.basket_gaps().items()}
    context={'pp':pp,'q':q,'kin':kin,'H_inv':np.linalg.inv(H),'tool_inv':np.linalg.inv(tool),'gaps':gaps,
             'lift_deg':lift_deg,'source_plate_xy':source.tolist(),'basket_z':basket_z,
             'H':H,'tool':tool,
             'centre_h':q.get('basket_quadrant_centre_h_m',.185),
             'basket_size_y_m':float(basket_config['size_xy'][1]),
             'above_m':.11,'extra_down_m':q.get('basket_policy_extra_down_m',0.),
             'lo_deg':lo_deg.tolist(),'hi_deg':hi_deg.tolist(),'home_deg':list(HOME_JOINTS_DEG),
             'frame_checker':frame_checker,'table_checker':table_checker,'basket_cache':{}}
    context['lift_solution_cache']={}
    jobs=[(di,float(px),float(py),float(bx),float(by),float(yaw))
          for di,(px,py) in enumerate(( (x,y) for x in pxs for y in pys))
          for yaw in yaws for by in ys for bx in xs]
    progress=output/'progress.jsonl';rows=[]
    with progress.open('w') as pf,mp.get_context('fork').Pool(workers,initializer=_basket_reach_init,initargs=(context,)) as pool:
        for i,row in enumerate(pool.imap_unordered(_basket_reach_job,jobs,chunksize=1),1):
            rows.append(row);pf.write(json.dumps({'done':i,'total':len(jobs)})+'\n');pf.flush()
            if i%20==0 or i==len(jobs):print(f'basket placement reach {i}/{len(jobs)}',flush=True)
    all_x=sorted(set(r['basket_xy_m'][0] for r in rows));all_y=sorted(set(r['basket_xy_m'][1] for r in rows))
    qnames=list(pp.QUADRANTS);faces=('plus_x','minus_x','either')
    maps={f:{qd:np.zeros((len(all_y),len(all_x))) for qd in qnames} for f in faces}
    lift_min=np.full((len(all_y),len(all_x)),np.nan);lift_med=np.full_like(lift_min,np.nan)
    lift_ok=np.zeros_like(lift_min);lift_pose_rate=np.zeros_like(lift_min)
    for yi,y in enumerate(all_y):
        for xi,x in enumerate(all_x):
            cell=[r for r in rows if abs(r['basket_xy_m'][0]-x)<1e-8 and abs(r['basket_xy_m'][1]-y)<1e-8]
            for qd in qnames:
                den=sum(len(r['quadrants'][qd]['plus_x']) for r in cell)
                for f in ('plus_x','minus_x'):
                    maps[f][qd][yi,xi]=sum(sum(r['quadrants'][qd][f]) for r in cell)/den if den else 0.
                maps['either'][qd][yi,xi]=sum(sum(int(a or b) for a,b in zip(r['quadrants'][qd]['plus_x'],r['quadrants'][qd]['minus_x']))
                                                for r in cell)/den if den else 0.
            cv=np.asarray([r['lift_clearance_ratio'] for r in cell if r['lift_clearance_ratio'] is not None],float)
            lift_pose_rate[yi,xi]=float(np.mean([r['plate_lift_ik_ok'] for r in cell])) if cell else 0.
            if len(cv):
                lift_min[yi,xi]=float(cv.min());lift_med[yi,xi]=float(np.median(cv));lift_ok[yi,xi]=float(np.mean(cv>=1.3))
    json_grid=lambda a:[[float(v) if np.isfinite(v) else None for v in row] for row in a]
    summary={}
    for f in faces:
        good=np.logical_and.reduce([maps[f][qd]>=.8 for qd in qnames]);ij=np.argwhere(good)
        summary[f]={'all4_ge80_cells':int(len(ij)),'x_bounds_m':[all_x[ij[:,1].min()],all_x[ij[:,1].max()]] if len(ij) else None,
                    'y_bounds_m':[all_y[ij[:,0].min()],all_y[ij[:,0].max()]] if len(ij) else None,
                    'quadrant_max_rate':{qd:float(maps[f][qd].max()) for qd in qnames}}
    result={'scope':'CPU IK reach only; carry/RRT and insertion collision excluded',
            'source_grasp_run':run,'dish_root_reference_m':[.848,.339],
            'basket_reference_m':[.807,-.192],'basket_xy_step_m':.025,'basket_x_grid_m':all_x,'basket_y_grid_m':all_y,
            'basket_yaw_deg':yaws,'dish_x_offsets_m':(np.asarray(pxs)-.848).tolist(),
            'dish_y_offsets_m':(np.asarray(pys)-.339).tolist(),
            'dish_root_xy_m':[[float(x),float(y)] for x in pxs for y in pys],
            'gap_count':{k:len(v) for k,v in gaps.items()},'lift_clearance_threshold_ratio':1.3,
            'lift_clearance_summary':{'lift_pose_ik_rate_grid':lift_pose_rate.tolist(),
                                      'min_ratio_grid':json_grid(lift_min),'median_ratio_grid':json_grid(lift_med),
                                      'fraction_ge_1_3_grid':lift_ok.tolist()},
            'ik_rate_maps':{f:{q:maps[f][q].tolist() for q in qnames} for f in faces},
            'all4_ge80_summary':summary,'rows':rows}
    (output/'basket_reach_map.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    extent=[all_x[0]-.025,all_x[-1]+.025,all_y[0]-.025,all_y[-1]+.025]
    for face in faces:
        fig,axs=plt.subplots(1,5,figsize=(16,3.5),sharey=True)
        for ax,qd in zip(axs[:4],qnames):
            m=maps[face][qd];ax.imshow(m,origin='lower',extent=extent,vmin=0,vmax=1,cmap='viridis',aspect='equal')
            ax.text(.03,.95,f'{qd} max {m.max():.0%}',transform=ax.transAxes,va='top',color='white');ax.set_xlabel('basket x (m)')
        axs[4].imshow(np.logical_and.reduce([maps[face][qd]>=.8 for qd in qnames]),origin='lower',extent=extent,vmin=0,vmax=1,cmap='Greens',aspect='equal')
        axs[4].set_xlabel('basket x (m)');axs[4].text(.03,.95,f'all Q ≥80%: {summary[face]["all4_ge80_cells"]}',transform=axs[4].transAxes,va='top')
        axs[0].set_ylabel('basket y (m)')
        for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(False)
        fig.tight_layout();fig.savefig(output/f'ik_rate_{face}.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    for ax,m,label in zip(axs,(lift_min,lift_med,lift_ok),('worst','median','fraction ≥1.3')):
        im=ax.imshow(m,origin='lower',extent=extent,cmap='magma',aspect='equal');ax.set_xlabel('basket x (m)')
        ax.text(.03,.95,label,transform=ax.transAxes,va='top',color='white')
        ax.spines[['top','right']].set_visible(False);ax.grid(False);fig.colorbar(im,ax=ax,fraction=.046)
    axs[0].set_ylabel('basket y (m)');fig.tight_layout();fig.savefig(output/'lift_collision_margin.png',dpi=160,bbox_inches='tight');plt.close(fig)
    return result


def _basket_lift_recheck_job(row):
    G=_BASKET_REACH;pp=G['pp'];px,py=row['dish_xy_m'];bx,by=row['basket_xy_m'];yaw=row['yaw_deg']
    source_q=np.radians(G['lift_deg']);Tplate=G['kin'].fk(source_q)@G['tool']@G['H']
    Tplate[:3,3]+=[px-G['source_plate_xy'][0],py-G['source_plate_xy'][1],0.]
    target=Tplate@G['H_inv']@G['tool_inv'];start=None
    for seed in (source_q,np.radians(G['home_deg'])):
        try:cand=G['kin'].ik(target,seed)
        except RuntimeError:continue
        deg=np.degrees(cand)
        if np.all(deg>np.asarray(G['lo_deg'])+3) and np.all(deg<np.asarray(G['hi_deg'])-3):start=deg;break
    if start is None:return False,None,None
    c,s=np.cos(np.radians(yaw)),np.sin(np.radians(yaw));Tbb=np.eye(4)
    Tbb[:2,:2]=[[c,-s],[s,c]];Tbb[:3,3]=[bx,by,G['basket_z']]
    quadrants={}
    for qd,gaps in G['gaps'].items():
        quadrants[qd]={name:[int(_basket_gap_ik(Tbb,qd,gi,face,start)) for gi in range(len(gaps))]
                       for face,name in ((1,'plus_x'),(-1,'minus_x'))}
    return True,_basket_lift_ratio(Tbb,start),quadrants


def correct_basket_lift_clearance(source_json,output,workers=12):
    """Recompute only the lifted-dish pose/collision metric in a preliminary IK-map JSON."""
    from calib_core import DATA,latest_t_base_cam,load_config,load_json
    from robot_model import HOME_JOINTS_DEG,Kinematics
    import plate_plans as pp
    import yaml
    source=json.loads(Path(source_json).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source_run_dir=DATA/'real_pipeline'/source['source_grasp_run']
    H,tool,run=pp.plate_in_hand(source_run_dir);req=json.loads((source_run_dir/'realplan_req.json').read_text())
    q=load_config()['plate_ik_bc'];kin=Kinematics();lims=np.asarray([[j[3],j[4]] for j in kin.joints])
    lo,hi=np.degrees(lims[:,0]),np.degrees(lims[:,1]);lo[4],hi[4]=-135.,135.
    checks=pp.library_checkers(np.asarray(load_json(latest_t_base_cam())['T_base_cam']),q);plate=pp.plate_cloud(H,tool)
    for name in ('frame','table'):checks[name][0].tool=np.vstack([checks[name][0].tool,plate])
    bc=yaml.safe_load((ROOT/'rl'/'basket.yaml').read_text())['full']
    global _BASKET_REACH
    gaps={k:[np.asarray(g,float) for g in v] for k,v in pp.basket_gaps().items()}
    _BASKET_REACH={'pp':pp,'q':q,'kin':kin,'H':H,'tool':tool,'H_inv':np.linalg.inv(H),'tool_inv':np.linalg.inv(tool),'gaps':gaps,
                   'lift_deg':json.loads(next(Path(e['plan']).read_text() for e in
                       [json.loads(x) for x in (DATA/'real_pipeline'/run/'robot_state.jsonl').read_text().splitlines() if '"event"' in x]
                       if e.get('event')=='plan'))['legs']['lift']['waypoints_deg'][-1],
                   'source_plate_xy':np.asarray(req['plate_root_fp'][:2],float).tolist(),'basket_z':.1442,
                   'lo_deg':lo.tolist(),'hi_deg':hi.tolist(),'home_deg':list(HOME_JOINTS_DEG),
                   'frame_checker':checks['frame'][0],'table_checker':checks['table'][0],'basket_cache':{},
                   'basket_size_y_m':float(bc['size_xy'][1]),'centre_h':q.get('basket_quadrant_centre_h_m',.185),
                   'above_m':.11,'extra_down_m':q.get('basket_policy_extra_down_m',0.)}
    with mp.get_context('fork').Pool(workers,initializer=_basket_reach_init,initargs=(_BASKET_REACH,)) as pool:
        results=pool.map(_basket_lift_recheck_job,source['rows'],chunksize=2)
    rows=source['rows']
    for row,(ok,ratio,quadrants) in zip(rows,results):
        row['plate_lift_ik_ok']=ok;row['lift_clearance_ratio']=ratio
        if quadrants is not None:row['quadrants']=quadrants
        if not ok:
            for qd in row['quadrants']:
                for face in row['quadrants'][qd]:row['quadrants'][qd][face]=[0]*len(row['quadrants'][qd][face])
    xs=source['basket_x_grid_m'];ys=source['basket_y_grid_m'];qnames=list(source['gap_count']);faces=('plus_x','minus_x','either')
    maps={f:{qd:np.zeros((len(ys),len(xs))) for qd in qnames} for f in faces}
    lift_min=np.full((len(ys),len(xs)),np.nan);lift_med=np.full_like(lift_min,np.nan);lift_ok=np.zeros_like(lift_min);lift_pose=np.zeros_like(lift_min)
    for yi,y in enumerate(ys):
        for xi,x in enumerate(xs):
            cell=[r for r in rows if abs(r['basket_xy_m'][0]-x)<1e-8 and abs(r['basket_xy_m'][1]-y)<1e-8]
            for qd in qnames:
                den=sum(len(r['quadrants'][qd]['plus_x']) for r in cell)
                for f in ('plus_x','minus_x'):
                    maps[f][qd][yi,xi]=sum(sum(r['quadrants'][qd][f]) for r in cell)/den if den else 0.
                maps['either'][qd][yi,xi]=sum(sum(int(a or b) for a,b in zip(r['quadrants'][qd]['plus_x'],r['quadrants'][qd]['minus_x'])) for r in cell)/den if den else 0.
            cv=np.asarray([r['lift_clearance_ratio'] for r in cell if r['lift_clearance_ratio'] is not None],float)
            lift_pose[yi,xi]=float(np.mean([r['plate_lift_ik_ok'] for r in cell])) if cell else 0.
            if len(cv):lift_min[yi,xi]=float(cv.min());lift_med[yi,xi]=float(np.median(cv));lift_ok[yi,xi]=float(np.mean(cv>=1.3))
    summary={}
    for f in faces:
        mask=np.logical_and.reduce([maps[f][qd]>=.8 for qd in qnames]);ij=np.argwhere(mask)
        summary[f]={'all4_ge80_cells':int(len(ij)),'x_bounds_m':[xs[ij[:,1].min()],xs[ij[:,1].max()]] if len(ij) else None,
                    'y_bounds_m':[ys[ij[:,0].min()],ys[ij[:,0].max()]] if len(ij) else None,
                    'quadrant_max_rate':{qd:float(maps[f][qd].max()) for qd in qnames}}
    finite=lambda a:[[float(v) if np.isfinite(v) else None for v in row] for row in a]
    source['scope']='13-point descent IK rates; post-lift plate+gripper model clearance rechecked at the varied dish XY. Carry/RRT excluded.'
    source['lift_clearance_summary']={'lift_pose_ik_rate_grid':lift_pose.tolist(),'min_ratio_grid':finite(lift_min),
                                     'median_ratio_grid':finite(lift_med),'fraction_ge_1_3_grid':lift_ok.tolist()}
    source['ik_rate_maps']={f:{qd:maps[f][qd].tolist() for qd in qnames} for f in faces};source['all4_ge80_summary']=summary
    source['clearance_correction_source']=str(source_json);(out/'basket_reach_map.json').write_text(json.dumps(source,indent=2))
    # Re-render corrected summary maps, keeping the user-facing plot style.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ex=[xs[0]-.025,xs[-1]+.025,ys[0]-.025,ys[-1]+.025]
    for face in faces:
        fig,axs=plt.subplots(1,5,figsize=(16,3.5),sharey=True)
        for ax,qd in zip(axs[:4],qnames):
            m=maps[face][qd];ax.imshow(m,origin='lower',extent=ex,vmin=0,vmax=1,cmap='viridis',aspect='equal')
            ax.text(.03,.95,f'{qd} max {m.max():.0%}',transform=ax.transAxes,va='top',color='white');ax.set_xlabel('basket x (m)')
        axs[4].imshow(np.logical_and.reduce([maps[face][qd]>=.8 for qd in qnames]),origin='lower',extent=ex,cmap='Greens',aspect='equal')
        axs[4].text(.03,.95,f'all Q ≥80%: {summary[face]["all4_ge80_cells"]}',transform=axs[4].transAxes,va='top');axs[4].set_xlabel('basket x (m)')
        axs[0].set_ylabel('basket y (m)')
        for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(False)
        fig.tight_layout();fig.savefig(out/f'ik_rate_{face}.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    for ax,m,label in zip(axs,(lift_min,lift_med,lift_ok),('worst','median','fraction ≥1.3')):
        im=ax.imshow(m,origin='lower',extent=ex,cmap='magma',aspect='equal');ax.set_xlabel('basket x (m)')
        ax.text(.03,.95,label,transform=ax.transAxes,va='top',color='white');ax.spines[['top','right']].set_visible(False);ax.grid(False);fig.colorbar(im,ax=ax,fraction=.046)
    axs[0].set_ylabel('basket y (m)');fig.tight_layout();fig.savefig(out/'lift_collision_margin.png',dpi=160,bbox_inches='tight');plt.close(fig)
    return source


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['optimize','map','basket_map','correct_basket_lift'])
    p.add_argument('--library',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--step',type=float,default=.05);p.add_argument('--rrt_nodes',type=int,default=600)
    p.add_argument('--radius_tolerance',type=float,default=.004)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--smoke',action='store_true')
    p.add_argument('--source_json',type=Path)
    a=p.parse_args()
    if a.mode=='basket_map':r=basket_reach_map(a.output,a.workers,a.smoke)
    elif a.mode=='correct_basket_lift':
        if not a.source_json:p.error('--source_json required for correct_basket_lift')
        r=correct_basket_lift_clearance(a.source_json,a.output,a.workers)
    elif not a.library:p.error('--library is required for optimize/map')
    elif a.mode=='optimize':r=optimize(a.library,a.output,a.rrt_nodes)
    else:r=workspace(a.library,a.output,a.step,a.radius_tolerance)
    print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
