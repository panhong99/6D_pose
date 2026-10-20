# cuRobo V2 vs our RRT vs PyRoki: plate carry (10/20)

Offline only. The demo code (`calibration/`) is unchanged and nothing here commands the robot.

## Environment

- `.envs/curobo-bm-20261020`: Python 3.10, torch 2.11.0+cu128, cuRobo V2 (`.envs/src/curobo`, commit 78fd485, Apache 2.0), Warp 1.18. RTX 5070 (sm_120).
- Rebuild with `bash experiments/curobo_bm/setup_env.sh`. The first run compiles GPU kernels (about 35 s, cached in `~/.cache/warp`). Later runs set up in about 4 s.

## Steps

```bash
D=calibration/data/basket/stand_free_20261019/curobo_vs_rrt/pad04_act50
# 1. scenes (foundationpose env): same start/goal and boxes as the RRT plans of the stand sweep fine_v4 (133 plans)
~/miniconda3/envs/foundationpose/bin/python experiments/curobo_bm/scene.py export calibration/data/basket/stand_free_20261019/fine_v4 <new dir>
# 2. cuRobo plans (GPU)
.envs/curobo-bm-20261020/bin/python experiments/curobo_bm/plan_curobo.py $D --act=0.05
# 3. our exact gate (MultiChecker held + FloorCheck, joint limits, J4/J6 <= 100 deg)
~/miniconda3/envs/foundationpose/bin/python experiments/curobo_bm/scene.py gate $D
# parallel envs: one world per env in one call
.envs/curobo-bm-20261020/bin/python experiments/curobo_bm/plan_curobo.py $D --batch --act=0.05
.envs/curobo-bm-20261020/bin/python experiments/curobo_bm/ik_batch.py $D --sizes=1,64,256 --seeds=16
```

## Model

- **Robot**: the link_3..6 mesh samples of our checker, 24 k-means spheres per link. Each radius reaches the farthest sample. Plus the gripper cylinder (r 6 cm, 16.5 cm) on link_6.
- **Held plate**: 90 spheres on an `attached_object` link under link_6.
- **Self collision**: none, as in our checker. cuRobo needs at least one pair, so plate vs link_3 is kept.
- **World**:
  - every held-checker box, grown by part of that checker's margin: frame 0.5×5 cm, table 1×2 cm, basket/markers/standing plates 1×1 cm;
  - the floor over the basket as a box up to `z_floor`.
- **Gap to our rules**:
  - The floor rule is for the plate only, but a cuRobo obstacle stops every link. A box over the full 8 cm pad blocks 46/133 starts, because the arm is already there after the lift. So the box pad is 4 cm, and a job whose start is blocked is planned again with pad 0.
  - The frame margin is halved so that starts already near `upright_01` are not refused.
- **Settings**: `optimizer_collision_activation_distance` 0.05 m, `plan_cspace(max_attempts=5)` with graph seeding.

## Results (133 stand-sweep plans, same start/goal, same exact gate)

| | our RRT | PyRoki | cuRobo V2 |
|---|---|---|---|
| pass the exact gate | 133 | 109 | **118** |
| time per plan | 3–16 s (CPU) | median 4.1 s, max 9.7 s (CPU) | **median 0.22 s, max 0.39 s** (GPU) |
| first-call setup | none | ~14 s (JAX compile) | ~35 s once (kernel compile, cached), then ~4 s |
| GPU memory, one planner | — | — | 250 MB peak |

The 97 jobs that both PyRoki and cuRobo pass, as medians:

| | RRT | PyRoki | cuRobo |
|---|---|---|---|
| joint travel | 165.3° | 164.7° | 166.9° |
| jerk (path shape) | 1 | 0.58 | 0.61 |
| acceleration (path shape) | 1 | 0.58 | 1.95 |
| floor clearance (1 = at the limit) | 1.25 | 1.47 | 1.50 |

The pass sets differ: cuRobo only 21, PyRoki only 12, neither 3.

How the activation distance and floor pad move the result (passes / plans found):

| floor pad, activation | 5 mm | 3 cm | 5 cm |
|---|---|---|---|
| 0 cm | 27 / 124 | 96 / 124 | 107 / 124 |
| 4 cm | 34 / 118 | 106 / 118 | 114 / 118 |

Remaining failures:
- **Starts blocked by the sphere model**: 9 (pad 0) to 15 (pad 4 cm). These are far stand spots (r ≥ 1.2 m, arm over the basket pad) and Q4 front 0.85 m.
- **Floor**: 4 jobs where the plate went under `z_floor` in the 8 cm pad.

## Parallel envs (one world per env, RTX 5070 12 GB)

Full motion planning (`BatchMotionPlanner`, `multi_env`, 1 attempt, no graph seeding):

| envs | solved | time per call | peak GPU memory |
|---|---|---|---|
| 1 | 1 | 0.11 s | 177 MB |
| 4 | 4 | 0.19 s | 228 MB |
| 16 | 13 | 0.71 s | 1.16 GB |
| 32 | 22 | 1.43 s | 1.82 GB |
| 64 | 47 | 2.89 s | 3.64 GB |
| 128 | out of memory | | |

IK only (16 seeds, collision aware):

| targets | time per call | peak GPU memory |
|---|---|---|
| 1 | 3.0 ms | 21 MB |
| 64 | 3.5 ms | 765 MB |
| 256 | 4.8 ms | 2.33 GB |
| 1024 (4 seeds) | 8.6 ms | 3.43 GB |
| 1024 (16 seeds) | out of memory | |

There is one process and one robot model. Memory still grows with envs × seeds, by about 0.6–0.8 MB per env-seed for IK and about 57 MB per env for full planning. IK time stays almost flat, so IK runs truly in parallel. Full-planning time grows about linearly: 64 envs take 26× the time of 1, a 2.5× throughput gain.
