"""(10/19 night, Claude) PyRoki vs RRT carry over the stand-position sweep: for every planned job of
experiments/stand_free/sweep.py (results.jsonl with `plan`), run benchmark_saved.py (PyRoki env, CPU) with the job's
standing plates and collect one table.  Offline only.

    python experiments/pyroki_bm/bench_sweep.py SWEEP_DIR OUT_DIR [--workers 3]"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / '.envs/pyroki-bm-20261019/bin/python'


def run(job, out):
    tag = f"{job['quadrant']}_{job['region']}_{job['r']:.2f}_{job['az']:.0f}"
    dst = out / f'{tag}.json'
    cmd = [str(PY), str(ROOT / 'experiments/pyroki_bm/benchmark_saved.py'), job['plan'], '--output', str(dst)]
    for st in job.get('standing', []):
        cmd += ['--standing', st]
    env = dict(os.environ, JAX_PLATFORM_NAME='cpu')
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not dst.exists():
        return dict(job=tag, error=(p.stderr or p.stdout)[-300:])
    r = json.loads(dst.read_text())
    L, P = r['legacy'], r['pyroki']
    return dict(job=tag, legacy_valid=r['legacy_full_path_valid'], pyroki_valid=r['pyroki_full_path_valid'],
                pyroki_s=r['pyroki_wall_s'], legacy=L, pyroki=P)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('sweep')
    a.add_argument('out')
    a.add_argument('--workers', type=int, default=3)
    a = a.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    jobs = [j for j in map(json.loads, open(Path(a.sweep) / 'results.jsonl')) if j.get('ok') and j.get('plan')]
    with ThreadPoolExecutor(a.workers) as ex, open(out / 'table.jsonl', 'w') as f:
        for res in ex.map(lambda j: run(j, out), jobs):
            f.write(json.dumps(res) + '\n')
            f.flush()
    print(f'{len(jobs)} plans -> {out}', file=sys.stderr)
