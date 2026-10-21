"""Evaluate BR checkpoints after training, preserving all existing artifacts.

Only offline saved photographs are read. The fixed evaluation rules come from
eval.py; no label-based point corrections or lines appear in CNN-only overlays.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def proc_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--after-pid', type=int, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    state = out / 'state.json'
    state.write_text(json.dumps(dict(status='waiting', training_pid=a.after_pid)))
    started = proc_start(a.after_pid)
    deadline = time.monotonic() + 3 * 3600
    while started is not None and proc_start(a.after_pid) == started:
        if time.monotonic() > deadline:
            raise TimeoutError('Training did not exit within three hours; no process was terminated')
        time.sleep(15)
    run = Path(a.run)
    completed = sorted(run.glob('ep[0-9][0-9].pt'))
    if not completed:
        raise RuntimeError('Training exited without any completed epoch; inspect its terminal and train.jsonl')
    # Never compete with an active hardware pipeline when the training finishes.
    commands = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
    active = [s for s in commands if ('calibration/run.sh' in s or 'calibration/pipeline.py' in s)
              and not any(x in s for x in ('br_followup.py', 'rg ', 'bash -lc'))]
    if active:
        state.write_text(json.dumps(dict(status='deferred_robot_active', processes=active)))
        return
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    import cv2
    import torch
    import eval as E
    from train import TineNet
    torch.set_num_threads(2)
    E.TTA = True
    results = []
    paths = [Path('calibration/data/slot_cnn/run_v3/last.pt')]
    requested = [run / f'ep{i:02d}.pt' for i in (2, 5, 8, 11)]
    paths += [p for p in requested if p.exists()]
    if requested[-1] not in paths and completed[-1] not in paths:
        paths.append(completed[-1])
    state.write_text(json.dumps(dict(status='evaluating', checkpoints=[str(p) for p in paths],
                                     training_complete=(run / 'last.pt').exists())))
    for ckpt in paths:
        dst = out / f'{ckpt.parent.name}_{ckpt.stem}'
        state.write_text(json.dumps(dict(status='evaluating', checkpoint=str(ckpt))))
        with (out / f'{dst.name}.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name('eval.py')),
                            '--ckpt', str(ckpt), '--tta', '--out', str(dst)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        m = json.loads((dst / 'metrics.json').read_text())
        tip = m['total']['tip']
        q1 = next(r['tip']['found'] for r in m['photos'] if r['label_quadrant'] == 'Q1')
        wrong = sum(r['wrong'] for r in m['tine_numbers']['tip'].values())
        passed = tip['recall'] >= .97 and tip['precision'] >= .93 and q1 >= 13 and wrong == 0
        results.append(dict(checkpoint=str(ckpt), total=m['total'], q1_found=q1,
                            tip_number_wrong=wrong, gate_passed=passed, metrics=str(dst / 'metrics.json')))
        net = TineNet(pretrained=False).cuda().eval()
        net.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=True))
        for image, _, _, q in E.D.real_tests():
            bgr = cv2.imread(image)
            pr = E.predict(net, bgr, .3)
            for key, colour in [('root', (255, 120, 0)), ('tip', (0, 0, 255))]:
                for x, y, score in pr[key]:
                    cv2.circle(bgr, (round(float(x)), round(float(y))), 3, colour, -1)
            cv2.imwrite(str(dst / f'{q}_cnn_only.png'), bgr)
        del net
        torch.cuda.empty_cache()
        (out / 'comparison.json').write_text(json.dumps(results, indent=2))
    state.write_text(json.dumps(dict(status='evaluated', any_candidate_passed=any(
        r['gate_passed'] for r in results[1:]), comparison=str(out / 'comparison.json'))))


if __name__ == '__main__':
    main()
