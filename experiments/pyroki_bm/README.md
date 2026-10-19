# PyRoki M1013 offline bridge and Claude handoff (2026-10-19)

The boss asked for PyRoki to replace current FK/IK/RRT. This directory is an
**opt-in offline prototype**; the 10/22 demo path has not been changed.

`opt_in_backend.OfflinePyrokiBackend` is the single offline entry point for
PyRoki FK, seeded IK, and checked carry optimization. The caller supplies
the current scene's held `MultiChecker`, `FloorCheck`, attached plate cloud,
obstacle boxes, basket transform, and floor height. `carry()` rejects a
candidate unless the legacy dense gate passes and endpoints remain fixed.
The vertical drop is a separate existing-planner step; this wrapper does not
generate it, command a robot, or alter the demo default. The benchmark below
is the only evaluated use so far; no live pipeline integration is claimed.

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

## Why the demo still uses RRT

PyRoki supplies URDF FK/IK and gradient-based trajectory optimization with
collision costs/constraints; its README states it does not include a sampling
planner. The current `plate_plans.rrt_carry` runs `MultiChecker(held.checkers +
[floor])`, with a held plate cloud, basket wall/marker/standing plate checks,
floor-over-basket condition, and J4/J6 <=100 degree excursion. The official
PyRoki trajopt example approximates robot meshes as capsules, uses primitive
world obstacles, and initializes from start/end IK. It does not encode this
project's attached plate cloud or `FloorCheck` automatically. Replacing RRT
without the exact saved context would lose collision requirements. The offline
candidate generator below now models the attached plate and reconstructed
obstacles, then rejects any path failing the existing `MultiChecker` and
`FloorCheck`. This is a conditional same-gate comparison: saved plans omit the
original camera calibration, and the current configuration may differ.
`hand_plate()` returns the same identity-plus-z-length tool constructed by
this benchmark under the current `plate_ik_bc` config; the saved
`plate_in_tcp` already captures the grasp-derived plate transform. Missing
grasp/tool fields alone do **not** explain a floor discrepancy. Optimization does not guarantee the
globally shortest path and can fail from a straight-line initial trajectory.

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
4. Record camera calibration and standing-plate state alongside future plans;
   storing the tool and grasp would make provenance auditable. Recheck both candidate and saved trajectories
   under that exact state before considering an opt-in production alternative.

Official sources: https://github.com/chungmin99/pyroki and
https://chungmin99.github.io/pyroki/examples/07_trajopt/ .

## Stand-map comparison and its provenance limit

**Newer sweep after the dense floor fix:** Claude regenerated 133 plans after
commit `2163d36` (0.5° final carry floor check) and compared the same saved
endpoints in `calibration/data/basket/stand_free_20261019/pyroki_vs_rrt_v4/table.jsonl`.
The first benchmark report accepted 125/133 saved RRT paths and 101/133
PyRoki paths because it applied the carry-only 100° wrist limit to the shared
vertical drop. The planner does not apply that limit there. After correcting
the benchmark's drop gate, the same saved paths yield **133/133 RRT** and
**109/133 PyRoki** full-path passes. All 133 shared drops pass insertion
clearance and joint limits. In the corrected 109-case shared pass set,
median travel was 164.2° versus 164.7°, and 101-point acceleration/jerk L2
norms were 0.00428/0.00499 versus 0.00249/0.00302. Median exact held/floor
clearance ratios were 1.61/1.24 versus 1.80/1.47. PyRoki solve time median
was 4.1 s, maximum 9.7 s across all 133 cases (CPU).
These remain offline reconstructed results, without historical camera/config
provenance or robot execution. They supersede the older 111-plan performance
table for current-code comparison; retain that older table as a diagnostic of
the earlier floor-check gap.

Claude's offline `bench_sweep.py` ran the same saved carry endpoints across
111 stand-map plans. Under the reconstructed current checker, 91/111 saved
RRT carries and 89/111 PyRoki carries passed. Among cases where both passed,
median joint travel was 166.7° versus 166.3°, and 101-point acceleration /
jerk L2 norms were 0.004/0.005 versus 0.003/0.003. PyRoki solve time had a
4.2 s median and 10.5 s maximum (CPU). Results:
`calibration/data/basket/stand_free_20261019/pyroki_vs_rrt/table.jsonl`.

These are **not** 111 exact historical replays. The saved plan has
`plate_in_tcp` but not a separate `tool` field. However `hand_plate()` returns
exactly the same identity-plus-configured-z-length tool used by this benchmark,
and the saved `plate_in_tcp` is the grasp-derived transform. Therefore the
reported 4/16 floor discrepancies are real reconstruction discrepancies, but
cannot be attributed to a missing tool field alone. Investigate floor height,
standing plates, config and camera pose against the original sweep context.
The saved plans also omit original camera calibration/config provenance. The
91/111 and 89/111 figures describe this conditional reconstruction only;
they should not be used to rank planner reliability or safety. The benchmark
now reads a saved `drop_start` when present, and standing-plate pins use
`pin_gap()` (commit `92c10d0`).

## BN carry candidate (in progress)

`trajopt_backend.py` uses PyRoki FK in a JAX-differentiated joint trajectory
objective. It penalizes the attached plate point cloud against the current
world boxes, a raised basket floor, and J4/J6 excursion. It fixes the supplied
joint endpoints exactly. An optional robot-capsule cost is available for
diagnosis, but its default weight is zero: it penalized current-checker-valid
saved paths and pushed candidate trajectories into failures. Crucially,
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

For a second-plate offline reconstruction, pass each already standing plate
as a one-based pin, for example `--standing Q4:11`. This adds its seven-slice
geometry to the existing checker and PyRoki world boxes. In quadrants listed
by `basket_carry_rrt_over_plates`, the floor also rises above the standing
plate top by `basket_carry_plate_clear_m`, plus any configured RRT extra height.
The saved `plan.json` does not record which plates were present, so this input
must be supplied from an independently documented run; guessing it would not
create an equivalent historical comparison. Rejected seeds now retain their
individual exact-gate checks in the result JSON.

The four carry-library files also store an exact `drop_start` index and their
source `plan.json`. Pass `--library calibration/data/carry_library/Q3_red.json`
with that source plan to avoid inferring the carry prefix; the benchmark checks
that both waypoint arrays match. It still cannot restore historical camera
calibration or configuration. `--repeat 2` reruns the same input in one process
and records warm solve time and maximum accepted-path joint variation.

Current-config, latest-camera reconstruction using the four library paths
(20 knots, 120 L-BFGS iterations, three deterministic seeds per path):

| Library | Legacy prefix gate | PyRoki seed passes | PyRoki optimization time | Notes |
|---|---|---:|---:|---|
| Q3 red | pass | 0/3 | 3.54 s | Stored prefix is 155.8° joint travel. |
| Q3 white, `Q3:12` standing | fail (held 0) | 1/3 | 4.77 s | Candidate 265.9° vs saved 174.4°; saved path was valid under its original conditions, which are not fully recorded here. |
| Q4 red | pass | 0/3 | 4.47 s | Stored prefix is 143.7° joint travel. |
| Q4 white, `Q4:6` standing | fail (floor 0.612) | 0/3 | 4.89 s | Current Q4 standing-plate floor is 0.345 m. |

Local JSON: `bench_q3_red_library.json`, `bench_q3_white_library_standing.json`,
`bench_q4_red_library.json`, and `bench_q4_white_library_standing.json`. These
are conditional reconstructions. A legacy prefix failing the current gate
prevents a direct performance comparison; the PyRoki Q3 white candidate is
not evidence of improvement over the historical RRT. Q4 `183143_663` repeated
twice with identical input: 4.38/3.86 s optimization time, max joint path
variation 0°, but its candidate still travelled 243.2° vs legacy 167.0° and
had larger sampled acceleration and jerk. First-run Python/import/checker
setup is excluded from those optimization times.

An optional `--seed-from-legacy` experiment adds the saved RRT carry prefix
as a fourth initialization. It is a hybrid RRT postprocessor, not a PyRoki
replacement. On Q3 red, a 20-knot resampling of the valid 47-point RRT
prefix itself passed the exact gate (held ratio 1.414, floor 1.273), but the
unconstrained local solve moved away and failed (held ratio 0.566). A trial
joint-space trust penalty of 100 also failed (held ratio 0.283, floor 0.220),
so it was not retained. Results: `bench_q3_red_legacyseed_v2.json` and
`bench_q3_red_legacyseed_trust100.json`. The backend discards a failed candidate;
it does not silently return the original path as a PyRoki result.

## Corrected capsule-cost experiment (current offline default)

The approximate capsule/box cost disagreed with the current exact checker:
Q3 red's **valid** 20-knot RRT seed had unweighted robot penalty 0.1803
(weighted objective contribution 3606), and minimizing that term moved the
trajectory into the current checker collision. The default now leaves this
proxy cost off; the existing held-plate checker, joint/wrist limits, and
`FloorCheck` remain mandatory at <=2° interpolation. This is only as strong
as those existing checkers; no physical collision proof is claimed.

| Carry library | Current checker on saved prefix | PyRoki exact-gate seeds (proxy weight 0) | Selected travel vs saved | First optimization wall time |
|---|---|---:|---:|---:|
| Q3 red | pass | 4/4 (includes RRT seed) | 155.2° vs 155.8° | 3.74 s |
| Q3 white with `Q3:12` | fail under current reconstruction | 3/3 | 174.5° vs 174.4° | 6.3 s |
| Q4 red | pass | 4/4 (includes RRT seed) | 143.9° vs 143.7° | 4.87 s |
| Q4 white with `Q4:6` | fail under current reconstruction | 2/3 | 173.5° vs 173.0° | 4.3 s |

Q3/Q4 red used the extra RRT seed; all three independent seeds also passed.
The white saved prefixes fail the *current* reconstructed gate, so their
numbers are diagnostics rather than direct RRT-versus-PyRoki evidence. Q3 red
identical-input repeats were 2.32/1.88 s; Q4 red 3.00/2.69 s, with 0°
accepted-path variation in each pair. These times exclude Python/import and
checker setup. Local results: `bench_q3_red_norobotproxy.json`,
`bench_q3_red_default_repeat2.json`, `bench_q3_white_norobotproxy.json`,
`bench_q4_red_norobotproxy.json`, `bench_q4_red_default_repeat2.json`,
`bench_q4_white_norobotproxy.json`. No production backend was switched.

The carry-library `drop_start` also identifies the downstream vertical drop.
`benchmark_saved.py` now checks this shared saved drop with the current
insertion checker and joint limits at <=2° interpolation. The 100° J4/J6
excursion limit applies to the carry candidate only. In
the four reconstructed examples the drop gate passed, with minimum insertion
clearance ratios Q3 red/white 6.013/6.579 and Q4 red/white 2.400/2.530.
Combining that shared drop with the accepted PyRoki carry produced 4/4
current-checker-valid complete paths. The saved Q3/Q4 white carry prefixes
still fail the current reconstruction, and no new vertical descent was
generated by PyRoki. Results are `bench_{q3_red,q3_white,q4_red,q4_white}_fullgate.json`.
