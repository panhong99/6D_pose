"""Offline goal-1 IK candidates and path diagnostics. No hardware commands.

This is a simulation model (20 mm coupling + 2F-85), not the measured printed-adapter stack.
Generate a bank first; physics demos/gates must qualify candidates before deployment.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from calib_core import make_T
from robot_model import HOME_JOINTS_DEG, Kinematics

ROOT = Path(__file__).resolve().parents[1]


def tcp_transform(urdf):
    root = ET.parse(urdf).getroot()
    T = np.eye(4)
    for name in ('gripper_mount', 'tcp_joint'):
        origin = root.find(f"joint[@name='{name}']/origin")
        T = T @ make_T(Rotation.from_euler('xyz', np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix(),
                       np.fromstring(origin.get('xyz'),sep=' '))
    return T


def generate_bank(output, radii, phis, pitches, pre_m=0.10):
    if output.exists():
        raise RuntimeError('Use a new candidate bank path; existing banks are preserved')
    cfg = yaml.safe_load((ROOT/'rl/config.yaml').read_text())
    G = cfg['geometry']
    kin = Kinematics()
    tcp_T = tcp_transform(ROOT/'rl'/cfg['paths']['robot_urdf'])
    ready = json.loads((ROOT/'rl'/cfg['paths']['ready_joints']).read_text())['joints']
    seed0 = np.array([ready[f'joint_{i+1}'] for i in range(6)])
    R0 = (kin.fk(seed0) @ tcp_T)[:3,:3]
    rim0 = np.array([*G['rim_offset'],G['rim_dz']],float)
    rim0[:2] *= 1-G['grasp_inset']/np.linalg.norm(rim0[:2])
    rows, rejected = [], []
    for phi in phis:
        Z = Rotation.from_euler('z',phi,degrees=True).as_matrix()
        rim = Z @ rim0
        for pitch in pitches:
            # World pitch about the yaw-rotated lateral axis; preserves a defined approach orientation.
            R = Z @ Rotation.from_euler('y',pitch,degrees=True).as_matrix() @ R0
            seed = seed0.copy()
            for r in radii:
                point = np.array([r,0,G['table_top_z']+G['box_h']]) + rim
                pre = make_T(R, point-R[:,2]*pre_m)
                flange = pre @ np.linalg.inv(tcp_T)
                try:
                    q = kin.ik(flange,seed)
                    grasp_q = kin.ik(make_T(R,point) @ np.linalg.inv(tcp_T),q)
                    seed=q
                    rows.append(dict(radius_m=float(r),phi_deg=float(phi),pitch_deg=float(pitch),
                                     plate_xyz=[float(r),0.0,G['table_top_z']+G['box_h']],
                                     q_pre_deg=np.degrees(q).tolist(), q_grasp_ik_deg=np.degrees(grasp_q).tolist(),
                                     rim_offset=rim.tolist(),tool_z=R[:,2].tolist(),finger_y=R[:,1].tolist()))
                except RuntimeError:
                    rejected.append(dict(radius_m=float(r),phi_deg=float(phi),pitch_deg=float(pitch),reason='unreachable_ik'))
    output.parent.mkdir(parents=True,exist_ok=True)
    keys=['radius_m','phi_deg','pitch_deg','plate_xyz','q_pre_deg','q_grasp_ik_deg','rim_offset','tool_z','finger_y']
    np.savez_compressed(output,**{k:np.asarray([r[k] for r in rows]) for k in keys},
                        home_deg=np.asarray(HOME_JOINTS_DEG),pre_m=pre_m,tcp_transform=tcp_T)
    output.with_suffix('.json').write_text(json.dumps(dict(candidate_only=True,physical_tcp_calibrated=False,
        n=len(rows),n_attempted=len(rows)+len(rejected),rejected=rejected,radii=list(map(float,radii)),
        phis=list(map(float,phis)),pitches=list(map(float,pitches)),pre_m=pre_m),indent=2))
    print(f'BANK {output} {len(rows)}/{len(rows)+len(rejected)} reachable IK (not physics success)',flush=True)


def workspace_bank(output,samples,seed,phi=-10.0,pitch=-10.0):
    """Uniform Cartesian targets inside the requested frame and radius band; no collision rejection hidden here."""
    rng=np.random.default_rng(seed);xy=[];outside=0
    while len(xy)<samples:
        p=rng.uniform([.87,-.52],[1.63,.52])
        if not .85<=np.linalg.norm(p)<=1.32:
            outside+=1;continue
        xy.append(p)
    xy=np.asarray(xy);r=np.linalg.norm(xy,axis=1);az=np.arctan2(xy[:,1],xy[:,0])
    raw=output.with_name(output.stem+'_canonical.npz')
    generate_bank(raw,r,[phi],[pitch])
    with np.load(raw,allow_pickle=False) as bank:arrays={k:bank[k].copy() for k in bank.files}
    if len(arrays['radius_m'])!=samples:
        raise RuntimeError('Unreachable IK in uniform workspace; investigate rather than removing test targets')
    # The canonical solver returns rows in radius order (the input order here), preserving the evaluation sample.
    arrays['plate_xyz'][:,:2]=xy
    for key in ('rim_offset','tool_z','finger_y'):
        v=arrays[key].copy();c,s=np.cos(az),np.sin(az)
        arrays[key][:,0]=c*v[:,0]-s*v[:,1];arrays[key][:,1]=s*v[:,0]+c*v[:,1]
    for key in ('q_pre_deg','q_grasp_ik_deg'):arrays[key][:,0]+=np.degrees(az)
    arrays['base_azimuth_deg']=np.degrees(az)
    np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(dict(candidate_only=True,samples=samples,seed=seed,
        phi_deg=phi,pitch_deg=pitch,uniform_cartesian_bounds=[[.87,1.63],[-.52,.52]],
        radius_band=[.85,1.32],outside_radius_band_rejections=outside,
        collision_targets_not_removed=True),indent=2))
    print(f'WORKSPACE BANK {output} samples={samples} outside_radius_band={outside}',flush=True)


def select_workspace_bank(output,workspace,library,source_bank,full_paths=False,prefer_flat=False):
    """Rank directions at every target by three pose clearances. Selection only, never path qualification."""
    from goal1_offline import CombinedChecker,checkers
    if output.exists():raise RuntimeError('Use a new output bank')
    def read(path):
        with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}
    W,L,B=read(workspace),read(library),read(source_bank)
    checker=CombinedChecker(checkers())
    if len(B['radius_m'])!=len(L['radius_m']):raise ValueError('Library/source bank row mismatch')
    from concurrent.futures import ThreadPoolExecutor
    from robot_model import polyline_path
    home=np.radians(HOME_JOINTS_DEG)
    rows=[];scores=[];selection=[]
    def choose(point):
        radius=float(np.linalg.norm(point[:2]));az=np.arctan2(point[1],point[0])
        options=[]
        for phi,pitch in set(zip(L['phi_deg'],L['pitch_deg'])):
            ids=np.flatnonzero(L['ok']&(L['phi_deg']==phi)&(L['pitch_deg']==pitch))
            if not len(ids):continue
            i=int(ids[np.argmin(abs(L['radius_m'][ids]-radius))])
            if abs(L['radius_m'][i]-radius)>(.004 if full_paths else .0101):continue
            end=int(L['success_step'][i]);close=int(L['close_step'][i])
            first=L['q_deg'][0,i] if full_paths else L['q_pre_deg'][i]
            q=np.radians(np.array([first,L['q_deg'][close,i],L['q_deg'][end,i]],dtype=float));q[:,0]+=az
            if full_paths:q=np.vstack([home,q])
            options.append((checker.path_clearance(q),-round(abs(L['radius_m'][i]-radius),9),-abs(phi),-abs(pitch),i))
        if not options:raise RuntimeError('No direction candidate; test targets cannot be removed')
        winner=None;flat_winner=None
        def flat(option):return abs(float(B['finger_y'][option[-1],2]))>=np.cos(np.radians(10))
        ordered=sorted(options,key=lambda v:(bool(prefer_flat and flat(v)),v),reverse=True)
        for option in ordered:
            bound,*rest=option
            if prefer_flat and flat_winner is not None and not flat(option):continue
            compare=flat_winner if prefer_flat and flat(option) else winner
            if compare is not None and (bound<compare[0] or (bound==compare[0] and tuple(rest)<=compare[1:])):continue
            i=rest[-1]
            if full_paths:
                end=int(L['success_step'][i]);Q=np.radians(L['q_deg'][:end+1,i].astype(float));Q[:,0]+=az
                path=np.vstack([polyline_path([home,Q[0]],.5),Q])
                option=(checker.path_clearance(path),*rest)
            if winner is None or option>winner:winner=option
            if prefer_flat and flat(option) and option[0]>=.010:
                if flat_winner is None or option>flat_winner:flat_winner=option
        score,_,_,_,i=flat_winner or winner
        row={k:B[k][i].copy() for k in ('radius_m','phi_deg','pitch_deg','plate_xyz','q_pre_deg','q_grasp_ik_deg','rim_offset','tool_z','finger_y')}
        row['radius_m']=radius;row['plate_xyz']=point.copy()
        row['q_pre_deg']=L['q_pre_deg'][i].copy();row['q_pre_deg'][0]+=np.degrees(az)
        row['q_grasp_ik_deg'][0]+=np.degrees(az)
        Z=Rotation.from_euler('z',az).as_matrix()
        for k in ('rim_offset','tool_z','finger_y'):row[k]=Z@row[k]
        return row,score,i
    with ThreadPoolExecutor(max_workers=4 if full_paths else 1) as pool:
        for j,(row,score,i) in enumerate(pool.map(choose,W['plate_xyz'])):
            rows.append(row);scores.append(score);selection.append(i)
            if j%128==0:print(f'SELECT {j}/{len(W["plate_xyz"])}',flush=True)
    arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    arrays.update(home_deg=B['home_deg'],pre_m=B['pre_m'],tcp_transform=B['tcp_transform'],
                  source_library_index=selection,selection_extra_clearance_m=scores)
    np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(dict(candidate_only=True,samples=len(rows),
        source_workspace=str(workspace),source_library=str(library),
        selection_scope='full sampled home/BC path FK, physics/real validation still mandatory' if full_paths else '3 poses only; full path and physics gates mandatory',
        prefer_flat_jaws_when_full_extra_clearance_at_least_10mm=prefer_flat,
        no_collision_targets_removed=True,negative_selection_clearance=int((np.asarray(scores)<0).sum())),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--r_min',type=float,default=.85);p.add_argument('--r_max',type=float,default=1.32)
    p.add_argument('--r_step',type=float,default=.02)
    p.add_argument('--phi_deg',type=float,nargs='+',default=[-40,-20,0,20,40])
    p.add_argument('--pitch_deg',type=float,nargs='+',default=[-15,0,15])
    p.add_argument('--pre_m',type=float,default=.10)
    p.add_argument('--workspace_samples',type=int,default=0);p.add_argument('--seed',type=int,default=8401)
    p.add_argument('--select_workspace',type=Path);p.add_argument('--library',type=Path);p.add_argument('--source_bank',type=Path)
    p.add_argument('--full_paths',action='store_true',help='rank directions using every stored BC sample and dense home path')
    p.add_argument('--prefer_flat',action='store_true',help='prefer closing axis within 10deg of plate normal, only if full extra clearance >=10mm')
    a=p.parse_args()
    radii=np.unique(np.r_[np.arange(a.r_min,a.r_max+1e-9,a.r_step),a.r_max])
    if a.select_workspace:
        if not a.library or not a.source_bank:p.error('Selection requires --library and --source_bank')
        if a.prefer_flat and not a.full_paths:p.error('Flat preference requires full-path clearance')
        select_workspace_bank(a.output,a.select_workspace,a.library,a.source_bank,a.full_paths,a.prefer_flat)
    elif a.workspace_samples:
        workspace_bank(a.output,a.workspace_samples,a.seed,a.phi_deg[0],a.pitch_deg[0])
    else:generate_bank(a.output,radii,a.phi_deg,a.pitch_deg,a.pre_m)
