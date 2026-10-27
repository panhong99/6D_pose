"""Targeted real-image hard-negative update with output distillation.

Only the last prediction layer is updated.  Away from reviewed false-tip
neighbourhoods, the original checkpoint logits are held by a distillation loss.
The output is a new checkpoint and trace; inputs are never modified.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import eval as E  # noqa: E402
from train import STRIDE, TineNet, to_tensor  # noqa: E402


def prepare(image, false_points, flip=False):
    x = cv2.resize(image, (D.OW, D.OH - 4), interpolation=cv2.INTER_AREA)
    x = np.r_[x, np.zeros((4, D.OW, 3), np.uint8)]
    points = np.asarray(false_points, np.float32) * E.SCALE
    if flip:
        x = np.ascontiguousarray(x[:, ::-1])
        points[:, 0] = D.OW - 1 - points[:, 0]
    cells = np.floor(points / STRIDE).astype(int)
    return to_tensor(x), cells


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--photos', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--distill', type=float, default=20.0)
    p.add_argument('--radius', type=int, default=1)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    review = json.loads(Path(a.labels).read_text())
    by_photo = {}
    for item in review['frames']:
        by_photo.setdefault(item['photo'], []).append(item['false_tip_uv'])

    batches, cells = [], []
    for name, false_points in sorted(by_photo.items()):
        image = cv2.imread(str(Path(a.photos) / name))
        if image is None:
            raise FileNotFoundError(Path(a.photos) / name)
        for flip in (False, True):
            x, c = prepare(image, false_points, flip)
            batches.append(x)
            cells.append(c)
    x = torch.stack(batches).cuda()

    teacher = TineNet(pretrained=False).cuda().eval()
    teacher.load_state_dict(torch.load(a.ckpt, map_location='cuda', weights_only=True))
    student = TineNet(pretrained=False).cuda().eval()
    student.load_state_dict(teacher.state_dict())
    for param in student.parameters():
        param.requires_grad_(False)
    for param in student.head[-1].parameters():
        param.requires_grad_(True)
    with torch.no_grad():
        target = teacher(x).detach()

    mask = torch.ones_like(target)
    negative = []
    for b, points in enumerate(cells):
        for ix, iy in points:
            y0, y1 = max(0, iy - a.radius), min(mask.shape[-2], iy + a.radius + 1)
            x0, x1 = max(0, ix - a.radius), min(mask.shape[-1], ix + a.radius + 1)
            mask[b, 1, y0:y1, x0:x1] = 0
            negative.append((b, y0, y1, x0, x1))

    opt = torch.optim.AdamW(student.head[-1].parameters(), lr=a.lr, weight_decay=1e-4)
    trace = []
    for step in range(a.steps):
        pred = student(x)
        keep = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum') / mask.sum()
        neg = torch.stack([F.softplus(pred[b, 1, y0:y1, x0:x1]).mean()
                           for b, y0, y1, x0, x1 in negative]).mean()
        loss = a.distill * keep + neg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == a.steps - 1:
            with torch.no_grad():
                probs = [torch.sigmoid(pred[b, 1, (y0 + y1 - 1)//2, (x0 + x1 - 1)//2]).item()
                         for b, y0, y1, x0, x1 in negative]
            rec = {'step': step, 'loss': float(loss), 'distill': float(keep),
                   'negative': float(neg), 'false_mean': float(np.mean(probs)),
                   'false_max': float(np.max(probs))}
            trace.append(rec)
            print(json.dumps(rec), flush=True)
    torch.save(student.state_dict(), out / 'last.pt')
    (out / 'trace.json').write_text(json.dumps(trace, indent=1))
    (out / 'args.json').write_text(json.dumps(vars(a), indent=1))


if __name__ == '__main__':
    main()
