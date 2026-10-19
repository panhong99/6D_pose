# PyRoki opt-in integration handoff (after the 10/22 demo)

This is a code map, not an instruction to change the frozen demo. The current
offline result is 109/133 PyRoki versus 133/133 saved RRT paths under a
reconstructed scene. Keep the existing planner as the default.

## Carry insertion point

In `calibration/plate_plans.py`, `quadrant_plan()` builds the held-plate
`MultiChecker`, `FloorCheck`, optional prior, start step-away, and vertical
`drop` before calling `rrt_carry()` (currently around line 1392). A future
`basket_planner_backend` switch belongs at that call site, after the prior
check and before assembling `carry = top + drop[::-1][1:]`.

Pass the exact `s0` and `drop[-1]` joints, `held`, `fl`, `cloud`, `Tbb`,
`z_mk`, and the obstacle boxes from the same `held.checkers` already built for
this scene into `OfflinePyrokiBackend.carry()`. Use the same basket footprint
half-width as `FloorCheck.half`, not a separately guessed margin. Preserve
`pre`, the existing vertical `drop`, and the current insertion gate. Tag the
returned path with `how='pyroki'` so logs identify the backend. The final
assembled carry must still pass `c_car` and `fl_ok`. Today `fl_ok` is enabled
only for `how` values beginning with `rrt` or `prior`; a future integration
must extend that condition to `pyroki` (or apply an equivalent unconditional
floor gate). Otherwise the final assembled path skips the floor check even
though the backend checked its candidate segment.

The PyRoki backend requires PyRoki/JAX/yourdfpy in its process. The current
isolated `.envs/pyroki-bm-20261019` is an experiment environment; it does
not make the production `foundationpose` environment ready. Before a live
opt-in, resolve that dependency boundary and measure cold compilation time.
Do not import PyRoki on the default RRT path.

## FK and IK

`experiments/pyroki_bm/bridge.py` exposes `fk(q_rad)` and seeded
`ik(target_T, seed_q_rad)`. `plate_plans.py` creates `Kinematics()` at several
planning, collision, hand-model, and insertion points; `pipeline.py` also
uses it for robot-state transforms. Replace those through one interface only
after checking every call site's frame, units, joint order, seed continuity,
and post-IK collision/limit contract. The 1000 FK and 20 reachable IK checks
are bridge parity tests, not validation of all production targets.

## Acceptance and rollback

1. The new backend is explicitly selected; absent selection retains today's
   RRT behavior and saved-path semantics.
2. A candidate must preserve start/end joints exactly and pass the existing
   `MultiChecker`, `FloorCheck`, joint limits, J4/J6 carry excursion, and the
   separate insertion checks. `CandidateRejected` must never reach execution.
3. If fallback is allowed, make it explicit in the result (`backend_used`
   and failure reason), then run the legacy planner and its full gate. Do not
   report a fallback path as PyRoki success.
4. Compare with newly captured plans that record camera/config/standing-plate
   provenance. The stored 133-plan table omits original camera/config, so it
   cannot establish a historical same-scene success rate.
5. Keep opt-in disabled until the exact-gate pass rate and latency satisfy a
   chosen deployment criterion and the boss reviews it. No robot command is
   part of this offline handoff.
