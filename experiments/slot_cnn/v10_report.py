"""Build the final run_v10_cross comparison, curves, and activation maps."""
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import eval as E  # noqa: E402
from train import TineNet, to_tensor  # noqa: E402


RUN = ROOT / 'calibration/data/slot_cnn/run_v10_cross'
OUT = ROOT / 'outputs/slot_cnn_run_v10_cross_report'
BASELINE_CROSS = ROOT / 'outputs/slot_cnn_cross_view_q3_q4_20261026/summary.json'


def number_totals(metrics):
    values = metrics['tine_numbers']['tip'].values()
    return {key: sum(item[key] for item in values)
            for key in ('right', 'wrong', 'refused')}


def read_rows():
    rows = []
    for epoch in range(12):
        metrics = json.loads((ROOT / f'calibration/data/slot_cnn/'
                                     f'fast_eval_run_v10_cross_ep{epoch:02d}/metrics.json').read_text())
        cross = json.loads((ROOT / f'outputs/slot_cnn_cross_view_run_v10_ep{epoch:02d}/'
                                   'summary.json').read_text())
        tip = metrics['total']['tip']
        root = metrics['total']['root']
        q1 = next(photo['tip'] for photo in metrics['photos']
                  if photo['label_quadrant'] == 'Q1')
        numbers = number_totals(metrics)
        total = cross['cross_total']
        rows.append({
            'epoch': epoch,
            'tip_recall': tip['recall'],
            'tip_precision': tip['precision'],
            'root_recall': root['recall'],
            'root_precision': root['precision'],
            'q1_found': q1['found'],
            'q1_labels': q1['labels'],
            'number_right': numbers['right'],
            'number_wrong': numbers['wrong'],
            'number_refused': numbers['refused'],
            'cross_found': total['found'],
            'cross_expected': total['expected'],
            'cross_strict': total['strict_answered'],
            'cross_whole_rows': total['whole_rows'],
            'cross_ambiguous_rows': total['ambiguous_rows'],
        })
    return rows


def valid(row):
    return (row['tip_recall'] >= .97 and row['tip_precision'] >= .93
            and row['q1_found'] >= 13 and row['number_wrong'] == 0
            and row['cross_found'] >= 258 and row['cross_strict'] >= 7)


def save_table(rows):
    with (OUT / 'checkpoint_summary.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def smooth(values, width=9):
    if len(values) < width:
        return np.asarray(values)
    return np.convolve(values, np.ones(width) / width, mode='same')


def training_curves():
    records = [json.loads(line) for line in (RUN / 'train.jsonl').read_text().splitlines()]
    batches = [record for record in records if 'step' in record]
    validation = [record for record in records if 'val_loss' in record]
    steps = [record['step'] for record in batches]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for key, label in [('loss', 'total'), ('heat', 'heatmap'), ('off', 'offset')]:
        axes[0].plot(steps, smooth([record[key] for record in batches]), label=label)
    axes[0].set(title='Training loss (9-point moving average)', xlabel='step', ylabel='loss')
    axes[0].grid(alpha=.25)
    axes[0].legend()
    axes[1].plot([record['epoch'] for record in validation],
                 [record['val_loss'] for record in validation], marker='o')
    axes[1].set(title='Synthetic validation loss', xlabel='epoch', ylabel='loss',
                xticks=range(12))
    axes[1].grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(OUT / 'training_curves.png', dpi=170)
    plt.close(figure)


def comparison_chart(rows, best_epoch):
    epochs = [row['epoch'] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(epochs, [row['tip_recall'] for row in rows], marker='o', label='recall')
    axes[0, 0].plot(epochs, [row['tip_precision'] for row in rows], marker='o', label='precision')
    axes[0, 0].axhline(.97, ls='--', lw=1, color='C0')
    axes[0, 0].axhline(.93, ls='--', lw=1, color='C1')
    axes[0, 0].set(title='Labelled v4 tip metrics', ylim=(.88, 1.01))
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [row['q1_found'] for row in rows], marker='o')
    axes[0, 1].axhline(13, ls='--', lw=1, color='black')
    axes[0, 1].set(title='Q1 tips found (of 14)', ylim=(10, 14.3))
    axes[1, 0].plot(epochs, [row['cross_found'] for row in rows], marker='o')
    axes[1, 0].axhline(258, ls='--', lw=1, color='black')
    axes[1, 0].set(title='Cross-view geometric coverage (of 260)', ylim=(248, 261))
    axes[1, 1].plot(epochs, [row['cross_strict'] for row in rows], marker='o', label='strict / 10')
    axes[1, 1].plot(epochs, [row['cross_ambiguous_rows'] for row in rows], marker='o',
                    label='ambiguous rows')
    axes[1, 1].axhline(7, ls='--', lw=1, color='C0')
    axes[1, 1].set(title='Cross-view numbering acceptance')
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.axvline(best_epoch, color='green', alpha=.25, lw=5)
        axis.set_xlabel('epoch')
        axis.set_xticks(range(12))
        axis.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(OUT / 'checkpoint_comparison.png', dpi=170)
    plt.close(figure)


def tip_heatmap(net, image, tta=True):
    resized = cv2.resize(image, (D.OW, D.OH - 4), interpolation=cv2.INTER_AREA)
    padded = np.r_[resized, np.zeros((4, D.OW, 3), np.uint8)]
    with torch.no_grad():
        logits = net(to_tensor(padded)[None].cuda()).float()[0]
        heat = torch.sigmoid(logits[1])
        if tta:
            flipped = net(to_tensor(np.ascontiguousarray(padded[:, ::-1]))[None].cuda()).float()[0]
            heat = (heat + torch.sigmoid(flipped[1]).flip(-1)) / 2
    heat = heat.cpu().numpy()
    heat = cv2.resize(heat, (D.OW, D.OH), interpolation=cv2.INTER_CUBIC)[:D.OH - 4]
    return cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)


def activation_maps(best_epoch):
    checkpoint = RUN / f'ep{best_epoch:02d}.pt'
    net = TineNet(pretrained=False).cuda().eval()
    net.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
    photos = []
    source = ROOT / 'calibration/data/basket/wrist_look'
    for photo in sorted(source.glob('20261026_*.raw.png')):
        match = re.search(r'_(Q[34])\.raw\.png$', photo.name)
        if match:
            photos.append((photo, match.group(1)))
    tiles_probability = []
    tiles_normalized = []
    for photo, view in photos:
        image = cv2.imread(str(photo))
        heat = np.clip(tip_heatmap(net, image), 0, 1)
        predictions = E.predict(net, image, .3)['tip']
        for normalized, target in ((False, tiles_probability), (True, tiles_normalized)):
            display_heat = heat / max(float(heat.max()), 1e-6) if normalized else heat
            colour = cv2.applyColorMap(np.uint8(np.clip(display_heat, 0, 1) * 255),
                                       cv2.COLORMAP_TURBO)
            alpha = np.clip(display_heat[..., None] * 0.75, 0, .75)
            overlay = np.uint8(image * (1 - alpha) + colour * alpha)
            for x, y, score in predictions:
                cv2.circle(overlay, (round(x), round(y)), 4, (255, 255, 255), 1, cv2.LINE_AA)
            bar = np.zeros((55, overlay.shape[1], 3), np.uint8)
            label = ('normalized activation' if normalized else 'raw tip probability')
            cv2.putText(bar, f'{view} | {label} | max={heat.max():.3f} | peaks={len(predictions)}',
                        (12, 35), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2, cv2.LINE_AA)
            tile = cv2.resize(np.vstack([bar, overlay]), (640, 388), interpolation=cv2.INTER_AREA)
            target.append(tile)
    for name, tiles in [('tip_probability_maps_Q3_Q4.jpg', tiles_probability),
                        ('activation_maps_Q3_Q4_normalized.jpg', tiles_normalized)]:
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[index:index + 2])
                          for index in range(0, len(tiles), 2)])
        cv2.imwrite(str(OUT / name), grid)


def baseline_summary():
    summary = json.loads(BASELINE_CROSS.read_text())
    items = [record['cross'] for record in summary['records']]
    return {
        'checkpoint': summary['checkpoint'],
        'found': sum(item['found'] for item in items),
        'expected': sum(item['expected'] for item in items),
        'strict_answered': sum(item['answer'] for item in items),
        'whole_rows': sum(item['whole_rows'] for item in items),
        'ambiguous_rows': sum(sum(bool(row.get('ambiguous'))
                                  for row in item['rows'].values()) for item in items),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    candidates = [row for row in rows if valid(row)]
    if not candidates:
        raise RuntimeError('no checkpoint satisfies the declared selection gates')
    best = sorted(candidates,
                  key=lambda row: (-row['cross_strict'], row['cross_ambiguous_rows'],
                                   -row['cross_found'], -row['tip_precision']))[0]
    save_table(rows)
    training_curves()
    comparison_chart(rows, best['epoch'])
    activation_maps(best['epoch'])
    result = {
        'selected_epoch': best['epoch'],
        'selected_checkpoint': str(RUN / f'ep{best["epoch"]:02d}.pt'),
        'selection_gates': {
            'tip_recall_min': .97,
            'tip_precision_min': .93,
            'q1_found_min': 13,
            'number_wrong_max': 0,
            'cross_found_min': 258,
            'cross_strict_min': 7,
        },
        'selected_metrics': best,
        'baseline_cross': baseline_summary(),
        'limitations': [
            'The v4 set contains only four repeatedly evaluated labelled photos.',
            'The 19 markerless photos have no hand labels; cross-view values are geometric consistency, not independent accuracy.',
            'The selected model does not reduce ambiguous rows versus the baseline (4 versus 3).',
        ],
    }
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
