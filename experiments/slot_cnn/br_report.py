"""BR saved-log plots and isolated batch-one inference measurements (no hardware commands)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def plots(run, evaluation, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False})
    records = [json.loads(s) for s in (run / 'train.jsonl').read_text().splitlines() if s]
    tr = [r for r in records if 'step' in r]
    va = [r for r in records if 'val_loss' in r]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    ax[0].plot([r['step'] for r in tr], [r['loss'] for r in tr], lw=.7, alpha=.6)
    ax[0].set(xlabel='Additional training step', ylabel='Training loss')
    ax[1].plot([r['epoch'] + 1 for r in va], [r['val_loss'] for r in va], 'o-')
    ax[1].set(xlabel='Additional epoch', ylabel='Synthetic validation loss')
    ax[2].plot([r['step'] for r in tr], [r['lr'] for r in tr])
    ax[2].set(xlabel='Additional training step', ylabel='Learning rate')
    fig.savefig(out / 'learning_curves.png', dpi=160)
    plt.close(fig)
    comp = json.loads((evaluation / 'comparison.json').read_text())
    labels = ['v3 baseline' if i == 0 else Path(r['checkpoint']).stem for i, r in enumerate(comp)]
    x = np.arange(len(comp))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for kind in ('tip', 'root'):
        for metric, style in [('recall', 'o-'), ('precision', 's--')]:
            ax[0].plot(x, [r['total'][kind][metric] * 100 for r in comp], style, label=f'{kind} {metric}')
    ax[0].set(ylabel='Detection (%) at 8 px, threshold 0.3')
    ax[0].legend(fontsize=8)
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        metrics = [json.loads(Path(r['metrics']).read_text()) for r in comp]
        ax[1].plot(x, [next(p['tip']['recall'] * 100 for p in m['photos'] if p['label_quadrant'] == q)
                       for m in metrics], 'o-', label=q)
    ax[1].set(ylabel='Tip recall (%) by quadrant')
    ax[1].legend()
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        values = []
        for m in metrics:
            n = m['tine_numbers']['tip'][q]
            gt = next(p['tip']['labels'] for p in m['photos'] if p['label_quadrant'] == q)
            values.append(100 * (n['right'] + n['wrong']) / (100 * gt))
        ax[2].plot(x, values, 'o-', label=q)
    ax[2].set(ylabel='Numbered matched-tip coverage (%)')
    ax[2].legend()
    for a in ax:
        a.set_xticks(x, labels, rotation=20)
    fig.savefig(out / 'evaluation_curves.png', dpi=160)
    plt.close(fig)
    # Include refusal and wrong counts directly; zero answered is not 100% accuracy.
    (out / 'metrics_summary.json').write_text(json.dumps(dict(comparison=comp, numbers=[
        dict(checkpoint=r['checkpoint'], tip=m['tine_numbers']['tip']) for r, m in zip(comp, metrics)],
        caveat='Four repeatedly evaluated photos; precision only inside annotated region. '
               'Coverage denominator: labelled tips x 100 pose perturbation trials. '
               'Number accuracy is conditional on matched detections; unmatched predictions are unscored.'), indent=2))
    return comp


def benchmark(ckpt, out):
    import cv2
    import torch
    import eval as E
    from train import TineNet, to_tensor
    torch.set_num_threads(2)
    net = TineNet(pretrained=False).cuda().eval()
    net.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=True))
    photos = [cv2.imread(p) for p, _, _, _ in E.D.real_tests()]
    image = cv2.resize(photos[0], (E.D.OW, E.D.OH - 4))
    image = np.r_[image, np.zeros((4, E.D.OW, 3), np.uint8)]
    x = to_tensor(image)[None].cuda()
    result = dict(checkpoint=str(ckpt), device=torch.cuda.get_device_name(), batch=1,
                  dtype='float32', warmup=20, repetitions=100, measurements={})
    with torch.inference_mode():
        for tta in (False, True):
            E.TTA = tta
            for name in ('forward', 'predict'):
                def call(i):
                    if name == 'predict':
                        return E.predict(net, photos[i % len(photos)], .3)
                    y = net(x)
                    if tta:
                        y = net(x.flip(-1))
                    return y
                for i in range(20):
                    call(i)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for i in range(100):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    call(i)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)
                key = name + ('_tta' if tta else '')
                result['measurements'][key] = dict(mean_ms=float(np.mean(times)),
                    median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                    peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20)
    result['gpu_process_memory_snapshot'] = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True).strip()
    result['pid'] = os.getpid()
    result['scope'] = 'Warm steady state. predict includes preprocessing/H2D/peak decode/D2H; '
    result['scope'] += 'excludes camera, slot numbering, model load. TTA forward means two forwards only. '
    result['scope'] += 'Allocator peaks exclude CUDA context; process snapshot is after all runs.'
    (out / 'inference_benchmark.json').write_text(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    deadline = time.monotonic() + 3 * 3600
    while True:
        statefile = a.evaluation / 'state.json'
        state = json.loads(statefile.read_text()) if statefile.exists() else {}
        if state.get('status') == 'evaluated':
            break
        if state.get('status') == 'deferred_robot_active' or time.monotonic() > deadline:
            raise RuntimeError('Evaluation unavailable; defer report and GPU measurement')
        time.sleep(15)
    a.out.mkdir(parents=True, exist_ok=False)
    comp = plots(a.run, a.evaluation, a.out)
    commands = subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines()
    if any(('calibration/pipeline.py' in s or 'calibration/run.sh' in s or 'slot_cnn/train.py' in s)
           and not any(x in s for x in ('br_report.py', 'bash -lc', 'rg ')) for s in commands):
        (a.out / 'benchmark_deferred.txt').write_text('Another training or hardware pipeline is active.')
        return
    best = max(comp, key=lambda r: (r['gate_passed'], r['total']['tip']['recall'], r['total']['tip']['precision']))
    benchmark(Path(best['checkpoint']), a.out)


if __name__ == '__main__':
    main()
