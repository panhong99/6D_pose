"""Opt-in goal-1 simulation variants. Bank is generated offline by calibration/grasp_candidates.py.

Existing environments/checkpoints keep their original reset and target definitions when no bank is installed.
"""
import numpy as np


def install(env, path, fixed=False, azimuth_deg=0.0):
    import torch
    bank = np.load(path,allow_pickle=False)
    keys=('radius_m','phi_deg','pitch_deg','plate_xyz','q_pre_deg','rim_offset','tool_z','finger_y')
    if not len(bank['radius_m']):
        raise ValueError('Empty candidate bank')
    if fixed and len(bank['radius_m']) != env.num_envs:
        raise ValueError('Fixed candidate bank requires one environment per row')
    env._grasp_bank={k:torch.as_tensor(bank[k],device=env.device,dtype=torch.float32) for k in keys}
    env._grasp_fixed=bool(fixed)
    env._grasp_azimuth_deg=float(azimuth_deg)
    env._grasp_bank_index=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    env._grasp_azimuth=torch.zeros(env.num_envs,device=env.device)
    for key in ('rim_offset','tool_z','finger_y'):
        setattr(env,'_grasp_'+key,torch.zeros(env.num_envs,3,device=env.device))


def reset_from_bank(env, ids):
    import torch
    import plate_env as pe
    bank=env._grasp_bank
    idx=ids if env._grasp_fixed else torch.randint(len(bank['radius_m']),(len(ids),),device=env.device)
    az=torch.zeros(len(ids),device=env.device) if env._grasp_fixed else (
        torch.rand(len(ids),device=env.device)*2-1)*np.radians(env._grasp_azimuth_deg)
    env._grasp_bank_index[ids]=idx;env._grasp_azimuth[ids]=az

    def rotate(x):
        y=x.clone();c,s=torch.cos(az),torch.sin(az)
        y[:,0]=c*x[:,0]-s*x[:,1];y[:,1]=s*x[:,0]+c*x[:,1]
        return y

    for key in ('rim_offset','tool_z','finger_y'):
        getattr(env,'_grasp_'+key)[ids]=rotate(bank[key][idx])
    plate=env.scene['plate']
    pose=plate.data.default_root_state[ids,:7].clone()
    pose[:,:3]=rotate(bank['plate_xyz'][idx])+env.scene.env_origins[ids]
    # Each simulation has one support height; bank IK starts may be a few mm above/below it.
    pose[:,2]=env.scene.env_origins[ids,2]+pe.REST_Z
    # Disk yaw is irrelevant; keep the reset orientation (normal still upright).
    pose[:,3:]=plate.data.root_quat_w[ids]
    plate.write_root_pose_to_sim(pose,env_ids=ids)
    plate.write_root_velocity_to_sim(torch.zeros(len(ids),6,device=env.device),env_ids=ids)
    pe.box_under_plate(env,ids)
    robot=env.scene['robot'];q=robot.data.default_joint_pos[ids].clone()
    arm_ids=[robot.joint_names.index(f'joint_{i+1}') for i in range(6)]
    q[:,arm_ids]=torch.deg2rad(bank['q_pre_deg'][idx]);q[:,arm_ids[0]]+=az
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
    robot.set_joint_position_target(q,env_ids=ids)
    pe.reset_task_state(env,ids)


def library_replay(env,bank_path,library_path):
    """Sim-only test of stored BC joint paths, never teleports the arm/plate while moving.

    Rate-limited joint targets pass through the existing voxel shield and physical drives.
    Cursor waits for measured tracking; at close it waits for physical held/jaw stall.
    This is trajectory replay, not a live-policy generalization score.
    """
    import torch
    import plate_env as pe
    if not env._grasp_fixed or getattr(env,'_arm_joint_speed_rad_s',None) is None:
        raise ValueError('Replay needs a fixed bank and explicit slow joint cap')
    with np.load(bank_path,allow_pickle=False) as B:
        source=B['source_library_index'].astype(int)
        az=np.arctan2(B['plate_xyz'][:,1],B['plate_xyz'][:,0])
        starts=B['q_pre_deg'].copy();radii=B['radius_m'].copy()
    with np.load(library_path,allow_pickle=False) as L:
        if len(source)!=env.num_envs or (source<0).any() or (source>=len(L['ok'])).any():raise ValueError('Replay bank/source mismatch')
        if not L['ok'][source].all():raise ValueError('Replay bank selected failed source paths')
        expected=L['q_pre_deg'][source].copy();expected[:,0]+=np.degrees(az)
        if not np.allclose(starts,expected,atol=1e-4,rtol=0) or (np.abs(radii-L['radius_m'][source])>.004).any():
            raise ValueError('Replay start/radius does not match stored library')
        Q=np.radians(L['q_deg'][:,source].transpose(1,0,2).copy())
        Q[:,:,0]+=az[:,None]
        end=L['success_step'][source].copy();close=L['close_step'][source].copy()
    if (close<0).any() or (end<close).any():raise ValueError('Invalid stored close/success indices')
    dev=env.device;n=env.num_envs;rows=torch.arange(n,device=dev)
    paths=torch.as_tensor(Q,device=dev,dtype=torch.float32)
    close=torch.as_tensor(close,device=dev);end=torch.as_tensor(end,device=dev)
    cursor=torch.zeros(n,dtype=torch.long,device=dev);settled=torch.zeros_like(cursor)
    passed_close=torch.zeros(n,dtype=torch.bool,device=dev)
    previous_fingers=torch.zeros(n,device=dev)
    arm=env.action_manager.get_term('arm_action')
    def act(obs):
        reset=env.episode_length_buf<=1
        cursor[reset]=0;settled[reset]=0;passed_close[reset]=False
        st=pe._state(env);target=paths[rows,cursor]
        tracked=(env.scene['robot'].data.joint_pos[:,arm._joint_ids]-target).abs().amax(1)<np.radians(.1)
        at_close=cursor>=close
        stalled=(st['fingers']-previous_fingers).abs()<1e-3
        settled.copy_(torch.where(at_close&st['held']&stalled,settled+1,torch.zeros_like(settled)))
        passed_close.logical_or_(settled>=5)
        advance=tracked&((cursor<close)|passed_close)&(cursor<end)
        cursor.add_(advance.long())
        env._candidate_joint_replay_target=paths[rows,cursor]
        previous_fingers.copy_(st['fingers'])
        action=torch.zeros(n,7,device=dev)
        action[:,6]=torch.where(cursor>=close,-1.,1.)
        return action
    return act
