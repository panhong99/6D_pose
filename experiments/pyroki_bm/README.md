# PyRoki M1013 offline bridge and Claude handoff (2026-10-19)

The boss asked for PyRoki to replace current FK/IK/RRT. This directory is an
**opt-in offline prototype**; the 10/22 demo path has not been changed.

## Environment and tested commands

- PyRoki source: `https://github.com/chungmin99/pyroki` at `388e43e1fc0d0ee382968d3dd72970fd62a0450c`.
- Isolated env: `/home/pan/pan/6D_pose/.envs/pyroki-bm-20261019` (Python 3.11, PyRoki 0.0.0, JAX/JAXLIB 0.10.2, PyYAML). Source clone: `/tmp/pyroki-bm-20261019`.
- `JAX_PLATFORM_NAME=cpu .envs/pyroki-bm-20261019/bin/python experiments/pyroki_bm/compare_fk.py`
- `JAX_PLATFORM_NAME=cpu .envs/pyroki-bm-20261019/bin/python experiments/pyroki_bm/compare_ik.py`
- `bridge.PyrokiKinematics(DEFAULT_URDF)` implements `fk(q)` and seeded `ik(target, seed)` signatures. This uses upstream pose and joint-limit costs with a dynamic initial seed and checks final position/angle residuals. No robot interface is imported.

## Observed results (CPU, single run)

| Check | Existing code | PyRoki | Scope |
|---|---:|---:|---|
| 1000 random FK pose max position difference | reference | 1.03e-12 mm | Same URDF/link_6/joint order, x64 |
| 1000 random FK pose max angle difference | reference | 6.32e-14 deg | Same |
| 1000 FK elapsed | 32.6 ms | 238 ms first batch, 2.33 ms warm batch | Distinct JAX shapes compile separately; no robot loop claim |
| 20 FK-generated reachable target IK success | 20/20 | 20/20 | Existing seeded at joint-limit midpoint; PyRoki upstream example unseeded midpoint; position <=0.1 mm and angle <=0.1 deg for PyRoki |
| IK median per target | 13.7 ms | 1.30 ms warm, 1.55 s first compile | 20 samples, CPU only |
| Bridge seeded IK smoke | — | 20/20 | Seed within 0.03 rad of source q; 1.73 s first, 0.91 ms median warm; **not** a global IK success rate |

Machine-readable numbers: `fk_result.json`, `ik_result.json` in this directory (local artifacts; do not stage as datasets).

Additional upstream trajopt smoke: using a saved taught `to_above` start/end
link_6 pose (`20261019_185810_433_taught/plan.json`), M1013 collision meshes,
20 waypoints, and **no world obstacles**, upstream `solve_trajopt` returned a
20×6 trajectory in 14.14 s first call (`/tmp/pyroki_trajopt_smoke.log`). Its
endpoint IK chose different joint branches: stored start J3/J5 were
114.575°/−27.528°, optimized start J3/J5 were 104.333°/+21.756°.
This confirms the example cannot directly preserve the already chosen carry
start/end joint configurations. It is a functional smoke, not a collision
comparison with RRT or an executable replacement.

The official IK example's `_solve_ik_jax` wrapper passes an int64 link index when JAX x64 is enabled, while the analytic Jacobian asserts int32. `compare_ik.py` calls the same upstream JIT with an explicit int32 link index. `bridge.py` does the same. This was an integration detail, not an FK/IK accuracy failure.

## Why RRT has not been replaced

PyRoki supplies URDF FK/IK and gradient-based trajectory optimization with
collision costs/constraints; its README states it does not include a sampling
planner. The current `plate_plans.rrt_carry` runs `MultiChecker(held.checkers +
[floor])`, with a held plate cloud, basket wall/marker/standing plate checks,
floor-over-basket condition, and J4/J6 <=100 degree excursion. The official
PyRoki trajopt example approximates robot meshes as capsules, uses primitive
world obstacles, and initializes from start/end IK. It does not encode this
project's attached plate cloud or `FloorCheck` automatically. Replacing RRT
now would remove collision requirements; no equivalent-obstacle trajectory
comparison has been completed. Optimization also does not guarantee the
globally shortest path and may fail from a straight-line initial trajectory.

## Suggested integration for Claude

1. Keep the demo tag and existing production planner unchanged during the freeze.
2. If opting in later, use the bridge for FK/IK only behind a backend switch;
   compare seeded IK over actual saved targets and enforce joint limits and
   the existing final residual tolerance. Do not use a new solution if it flips
   wrist branches or violates segment-continuity checks.
3. Build a PyRoki collision model from M1013 collision meshes and **explicitly**
   represent the gripper/tool, attached plate cloud, cell/table, basket walls,
   marker boxes, standing plates, and `FloorCheck` rule. Validate *every*
   optimized segment with current `MultiChecker` before accepting it.
4. Benchmark stored `calibration/data/rl_plans/20261019_*_taught/plan.json`
   carry endpoints against `rrt_carry` with identical obstacles, then compare
   solve/compile time, path length, J4/J6 excursion and minimum clearance.
   Only after parity is demonstrated should trajopt be an opt-in alternative.

Official sources: https://github.com/chungmin99/pyroki and
https://chungmin99.github.io/pyroki/examples/07_trajopt/ .

## BN carry candidate (in progress)

`trajopt_backend.py` uses PyRoki FK and its M1013 collision capsules inside a
JAX-differentiated joint trajectory objective. It also penalizes the attached
plate point cloud against the current world boxes, a raised basket floor, and
J4/J6 excursion. It fixes the supplied joint endpoints exactly. Crucially,
`validate()` densifies every segment at <=2° and uses the existing held-plate
checker and `FloorCheck`; failed candidates are discarded. This is an offline
candidate generator in `experiments/`, with no production backend switch yet.

Reproduce a saved-path pilot (CPU):

`JAX_PLATFORM_NAME=cpu .envs/pyroki-bm-20261019/bin/python experiments/pyroki_bm/benchmark_saved.py calibration/data/rl_plans/20261019_183143_663_taught/plan.json --n 20 --maxiter 120 --output experiments/pyroki_bm/bench_q4_183143_best.json`

Pilot using current configuration and latest camera calibration, with the
historical RRT drop index inferred as its last floor-valid waypoint:

| Saved prefix | Existing prefix exact gate | PyRoki candidate exact gate | First-call wall time | Joint travel |
|---|---|---|---:|---|
| Q3 `174044_615` | pass | 0/3 seeds pass | 9.41 s | existing 174.2° |
| Q4 `183143_663` | pass | 1/3 seeds pass | 9.65 s | existing 167.0°, candidate 243.2° |

These are *not* a historical, equivalent-environment RRT comparison: the
stored file omits the original camera calibration path and drop-start index,
and the current configuration can differ. The accepted Q4 candidate is longer
and has a larger resampled acceleration/jerk norm; it does not justify a
production switch. JSON pilots are local result data and are not staged.
