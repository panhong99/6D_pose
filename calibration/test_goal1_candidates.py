"""Hardware-free checks of height selection and the new IK target-frame convention."""
import json
from pathlib import Path
import tempfile
import unittest
import ast
import math
from types import SimpleNamespace
import sys
from unittest.mock import patch

import numpy as np

from grasp_candidates import generate_bank, tcp_transform, ROOT
from plate_plans import height_library_config,nearest_direction_candidates
from robot_model import Kinematics,HOME_JOINTS_DEG


class Goal1Tests(unittest.TestCase):
    def test_library_replay_waits_for_physical_grasp_and_checks_source(self):
        import torch
        source=ast.parse((ROOT/'rl/grasp_variants.py').read_text())
        fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='library_replay')
        namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'replay-under-test','exec'),namespace)
        state=dict(fingers=torch.tensor([.1]),held=torch.tensor([False]))
        pe=SimpleNamespace(_state=lambda env:state)
        arm=SimpleNamespace(_joint_ids=list(range(6)))
        positions=torch.zeros(1,6)
        env=SimpleNamespace(_grasp_fixed=True,_arm_joint_speed_rad_s=.1,num_envs=1,device='cpu',
            episode_length_buf=torch.tensor([2]),scene={'robot':SimpleNamespace(data=SimpleNamespace(joint_pos=positions))},
            action_manager=SimpleNamespace(get_term=lambda name:arm))
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{'plate_env':pe}):
            bank=Path(d)/'bank.npz';library=Path(d)/'lib.npz'
            np.savez(bank,source_library_index=[0],plate_xyz=[[1.,0.,.1]],q_pre_deg=np.zeros((1,6)),radius_m=[1.])
            np.savez(library,ok=[True],q_pre_deg=np.zeros((1,6)),radius_m=[1.],
                     q_deg=np.arange(4)[:,None,None]*np.ones((4,1,6))*.01,close_step=[2],success_step=[3])
            act=namespace['library_replay'](env,bank,library)
            self.assertEqual(float(act(None)[0,6]),1.)
            positions.copy_(env._candidate_joint_replay_target)
            self.assertEqual(float(act(None)[0,6]),-1.)
            close=env._candidate_joint_replay_target.clone();positions.copy_(close)
            for _ in range(8):act(None);self.assertTrue(torch.equal(close,env._candidate_joint_replay_target))
            state['held'][:]=True;state['fingers'][:]=.85
            for _ in range(5):act(None);self.assertTrue(torch.equal(close,env._candidate_joint_replay_target))
            act(None);self.assertTrue(torch.all(env._candidate_joint_replay_target>close))
            env.episode_length_buf[:]=0;positions.zero_();state['held'][:]=False
            self.assertEqual(float(act(None)[0,6]),1.)
            np.savez(bank,source_library_index=[0],plate_xyz=[[1.,0.,.1]],q_pre_deg=np.ones((1,6)),radius_m=[1.])
            with self.assertRaises(ValueError):namespace['library_replay'](env,bank,library)

    def test_grasp_wait_uses_gripper_feedback_before_bc_lift_and_resets(self):
        import torch
        source=ast.parse((ROOT/'rl/eval.py').read_text())
        fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='approach_wrapper')
        state=dict(d_rim=torch.tensor([.004]),fingers=torch.tensor([.2]),held=torch.tensor([False]),
                   rim=torch.zeros(1,3),tcp=torch.zeros(1,3))
        pe=SimpleNamespace(A=dict(max_tcp_speed=.05,scale=.0025),_state=lambda e:state)
        arm=SimpleNamespace(hold_pos=torch.ones(1,3),_compute_frame_pose=lambda:(torch.zeros(1,3),None))
        env=SimpleNamespace(num_envs=1,device='cpu',step_dt=1/30,episode_length_buf=torch.tensor([10]),
            action_manager=SimpleNamespace(get_term=lambda name:arm))
        command=torch.tensor([[0.,0.,1.,0.,0.,0.,-1.]])
        args=SimpleNamespace(approach='policy',grasp_wait=True,latch_gripper=False,handoff_m=.04)
        namespace={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'sequencer-under-test','exec'),namespace)
        with patch.dict(sys.modules,{'plate_env':pe}):act=namespace['approach_wrapper'](args,env,lambda obs:command.clone())
        self.assertEqual(float(act(None)[0,2]),0)
        self.assertTrue(torch.all(arm.hold_pos==0))
        state['fingers'][:]=.7;state['held'][:]=True
        self.assertEqual(float(act(None)[0,2]),0)  # moving fingers do not satisfy stall
        for _ in range(4):self.assertEqual(float(act(None)[0,2]),0)
        self.assertEqual(float(act(None)[0,2]),1)
        command[:,6]=1
        self.assertEqual(float(act(None)[0,6]),-1)  # no unrequested opening after grasp
        env.episode_length_buf[:]=0;state['d_rim'][:]=.1
        self.assertEqual(float(act(None)[0,6]),1)

    def test_translation_cap_applies_during_reset_and_target_lead_is_bounded(self):
        # Execute the actual controller method with a recording IK base; no Isaac app or hardware.
        import torch
        source=ast.parse((ROOT/'rl/plate_env.py').read_text())
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='SafeDiffIKAction')
        class Base:
            def process_actions(self,a):self.received=a.clone()
        n=3;z=torch.zeros(n);quat=torch.zeros(n,4);quat[:,0]=1
        state=dict(d_rim=torch.ones(n),tool_match=z,finger_match=z,fingers=z)
        namespace=dict(torch=torch,math=math,DifferentialInverseKinematicsAction=Base,
            A=dict(max_tcp_speed=.05,hold_windup=.002,max_rot_speed_deg=10,
                   orientation_lock=dict(dist=.04,tool_deg=20,finger_deg=25)),
            _state=lambda env:state,_cos=lambda deg:math.cos(math.radians(deg)),
            quat_conjugate=lambda q:q,quat_mul=lambda a,b:a,
            axis_angle_from_quat=lambda q:torch.zeros(n,3))
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'controller-under-test','exec'),namespace)
        c=namespace['SafeDiffIKAction'].__new__(namespace['SafeDiffIKAction'])
        c.cfg=SimpleNamespace(scale=.03);c._env=SimpleNamespace(step_dt=1/30)
        c.blocked=torch.zeros(n,dtype=torch.bool);c.hold_init=torch.tensor([2,1,0])
        c.hold_pos=torch.zeros(n,3);c.orientation_locked=c.blocked.clone();c.orientation_target=quat.clone()
        c._compute_frame_pose=lambda:(torch.zeros(n,3),quat)
        action=torch.ones(n,6)*100
        c.process_actions(action)
        self.assertTrue(torch.all(c.received[:,:3].norm(dim=1)*.03<=.05/30+1e-8))
        self.assertTrue(torch.all(c.received[:,3:].norm(dim=1)*.03<=math.radians(10)/30+1e-8))
        c.hold_init[:]=0;c.hold_pos[:,0]=.1
        c.process_actions(torch.zeros(n,6))
        self.assertTrue(torch.all(c.received[:,:3].norm(dim=1)*.03<=.002+1e-8))
        c._env.physics_dt=1/120;c._env._arm_joint_speed_rad_s=math.radians(6)
        c._joint_ids=list(range(6));c._safe_target=torch.zeros(n,6)
        c._asset=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(n,6)),
            set_joint_position_target=lambda target,ids:setattr(c,'joint_target',target.clone()))
        c._ik_controller=SimpleNamespace(compute=lambda *args:torch.ones(n,6))
        c._compute_frame_jacobian=lambda:None
        c.shield=SimpleNamespace(safe=lambda q:torch.ones(n,dtype=torch.bool))
        c.apply_actions();first=c.joint_target.clone();c.apply_actions()
        cap=math.radians(6)/120
        self.assertTrue(torch.all(first.abs()<=cap+1e-8))
        self.assertTrue(torch.all((c.joint_target-first).abs()<=cap+1e-8))
        self.assertTrue(torch.all(c.joint_target>first))  # target continues despite mock servo lag

    def test_ik_start_and_contact_share_tcp_rim_and_orientation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bank.npz'
            generate_bank(path,[.90,1.20],[-40,40],[-15,0])
            bank=np.load(path);kin=Kinematics();tool=tcp_transform(ROOT/'rl/assets/m1013_2f85.urdf')
            self.assertAlmostEqual(tool[2,3],.1758,places=6)
            for i in range(len(bank['radius_m'])):
                pre=kin.fk(np.radians(bank['q_pre_deg'][i])) @ tool
                grasp=kin.fk(np.radians(bank['q_grasp_ik_deg'][i])) @ tool
                target=bank['plate_xyz'][i]+bank['rim_offset'][i]
                np.testing.assert_allclose(grasp[:3,3],target,atol=1e-4)
                np.testing.assert_allclose(pre[:3,3]+bank['tool_z'][i]*.10,target,atol=1e-4)
                np.testing.assert_allclose(pre[:3,2],bank['tool_z'][i],atol=4e-4)
                np.testing.assert_allclose(pre[:3,1],bank['finger_y'][i],atol=4e-4)

    def test_height_requires_measured_z_and_refuses_extrapolation(self):
        with tempfile.TemporaryDirectory() as d:
            index=Path(d)/'index.json'
            index.write_text(json.dumps(dict(height_tolerance_m=.0025,entries=[
                dict(plate_root_z_m=.0946,library='candidate0.npz',n_ok=8),
                dict(plate_root_z_m=.0986,library='candidate4.npz',n_ok=8)])))
            q=dict(library='original.npz',library_height_index=str(index))
            selected=height_library_config(q,.098)
            self.assertEqual(selected['library'],'candidate4.npz')
            self.assertEqual(q['library'],'original.npz')
            for z in (None,np.nan,.11):
                with self.assertRaises(RuntimeError):height_library_config(q,z)

    def test_existing_config_does_not_select_a_new_height(self):
        q=dict(library='original.npz')
        self.assertIs(height_library_config(q,None),q)

    def test_radius_match_is_nearest_valid_for_each_direction(self):
        lib=dict(radius_m=np.array([1.,1.002,1.004,1.,1.002]),ok=np.array([1,1,0,1,1],dtype=bool),
                 phi_deg=np.array([0,0,0,20,20]),pitch_deg=np.zeros(5))
        np.testing.assert_array_equal(nearest_direction_candidates(lib,1.0018),[1,4])
        self.assertEqual(len(nearest_direction_candidates(lib,1.2)),0)

    def test_batched_clearance_keeps_original_geometry_and_all_samples(self):
        from goal1_offline import CombinedChecker,checkers
        checker=CombinedChecker(checkers())
        rng=np.random.default_rng(811)
        path=np.radians(HOME_JOINTS_DEG)+rng.normal(0,.4,(37,6))
        self.assertAlmostEqual(checker.path_clearance(path),min(checker.clearance(q) for q in path),places=9)


if __name__=='__main__':unittest.main()
