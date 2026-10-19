"""(10/19 night) How much does the RRT carry change when the plate on the stand moves by a few mm?  Same inputs give the
same path (fixed seeds); this measures the spread from realistic input noise.  Offline, no robot."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'calibration'))
sys.path.insert(0, str(ROOT / 'rl'))
sys.path.insert(0, str(ROOT / 'experiments/stand_free'))
import sweep  # noqa: E402

if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    step = sweep.STEPS[int(sys.argv[2]) if len(sys.argv) > 2 else 2]
    import plate_plans as pp
    paths, rows = [], []
    orig = pp.quadrant_plan

    def keep(*a, **k):
        e, rep = orig(*a, **k)
        if e is not None:
            paths.append(np.array(json.loads(Path(e).read_text())['legs']['to_above']['waypoints_deg']))
        return e, rep
    pp.quadrant_plan = keep
    rng = np.random.default_rng(0)
    for i in range(int(sys.argv[3]) if len(sys.argv) > 3 else 8):
        dr, daz = (0.0, 0.0) if i == 0 else (rng.normal(0, 0.002), rng.normal(0, 0.15))   # ~2 mm, 0.15 deg
        n0 = len(paths)
        t0 = time.monotonic()
        res = sweep.plan_step((step, (0.966 + dr, 21.0 + daz), False, str(out), {}))
        rows.append(dict(i=i, dr_mm=dr * 1000, daz_deg=daz, ok=res['ok'], pin=res.get('pin'), carry=res.get('carry'),
                         t=time.monotonic() - t0, got=len(paths) > n0))
    ref = paths[0]
    def resample(w, n=60):
        s = np.r_[0, np.cumsum(np.abs(np.diff(w, axis=0)).max(1))]
        t = np.linspace(0, s[-1], n)
        return np.stack([np.interp(t, s, w[:, j]) for j in range(6)], 1)
    R0 = resample(ref)
    for r_, w in zip([r for r in rows if r['got']], paths):
        Rw = resample(w)
        r_.update(max_joint_dev_deg=float(np.abs(Rw - R0).max()), travel_deg=float(np.abs(np.diff(w, axis=0)).max(1).sum()),
                  wrist_turn_deg=float(np.abs(w[:, [3, 5]] - w[0, [3, 5]]).max()))
    (out / 'repeat.json').write_text(json.dumps(rows, indent=1))
    for r_ in rows:
        print(r_)
