"""(10/20 보스: parallel sim envs) batched collision-aware IK on the M1013 sphere model: N targets (reachable: FK of the
carry starts) in one call, N = 1..4096; time + GPU memory -> OUT_DIR/ik_batch.json.  Run in .envs/curobo-bm-20261020."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import plan_curobo as P  # noqa: E402


def main(out, sizes=(1, 64, 256, 1024, 4096), seeds=16):
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import GoalToolPose, JointState, Pose
    out = Path(out).resolve()
    spheres = json.loads((out / 'robot_spheres.json').read_text())
    scenes = json.loads((out / 'scenes.json').read_text())['scenes']
    world = dict(cuboid={c['name']: dict(dims=c['dims'], pose=c['pose']) for c in scenes[0]['cuboids']})
    rows = []
    for n in sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        ik = InverseKinematics(InverseKinematicsCfg.create(robot=P.robot_cfg(out, spheres), scene_model=world,
                                                           num_seeds=seeds, max_batch_size=n,
                                                           self_collision_check=False))
        q = torch.tensor([scenes[i % len(scenes)]['goal'] for i in range(n)], device='cuda', dtype=torch.float32)
        pose = ik.compute_kinematics(JointState.from_position(q, joint_names=P.JOINTS)).tool_poses.get_link_pose('link_6')
        goal = GoalToolPose.from_poses({'link_6': Pose(position=pose.position.reshape(n, 3).clone(),
                                                       quaternion=pose.quaternion.reshape(n, 4).clone())}, num_goalset=1)
        ik.solve_pose(goal)                                         # warm-up / CUDA graph capture
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = ik.solve_pose(goal)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        rows.append(dict(n=n, seeds=seeds, solved=int(r.success.reshape(n, -1).any(-1).sum()),
                         solve_ms=1e3 * float(np.median(times)),
                         peak_mb=(torch.cuda.max_memory_allocated() - base) / 2 ** 20))
        print(rows[-1], file=sys.stderr, flush=True)
        del ik
    (out / f'ik_batch_s{seeds}_{sizes[-1]}.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0), rows=rows), indent=1))


if __name__ == '__main__':
    kw = dict(a.split('=') for a in sys.argv[2:])
    main(sys.argv[1], sizes=tuple(int(x) for x in kw.get('--sizes', '1,64,256,1024,4096').split(',')),
         seeds=int(kw.get('--seeds', 16)))
