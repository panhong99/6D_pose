"""(10/20 보스, BQ) tine root/tip keypoint network trained on synthetic renders only (experiments/slot_cnn/data.py).

    ~/miniconda3/envs/foundationpose/bin/python experiments/slot_cnn/train.py --data calibration/data/slot_cnn/sim_v1 \
        --out calibration/data/slot_cnn/run_v1 [--epochs 12]

Model: ResNet18 (ImageNet) encoder + U-Net style decoder (3 up stages, skips from strides 16/8/4) -> stride-4 output:
2 heatmaps (root, tip) + a sub-pixel offset pair for each.  Loss: CenterNet focal loss on the heatmaps + L1 on the offsets at the
labelled cells.  Input 960 x 544 (the wrist 1280 x 720 at 0.75, padded).  Real photos are never used for training."""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4
KEYS = ('root', 'tip')


class TineNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.d3, self.d2, self.d1 = block(512 + 256, 256), block(256 + 128, 128), block(128 + 64, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 6, 1))
        self.head[-1].bias.data[:2] = -2.19                         # CenterNet prior: p ~ 0.1

    def forward(self, x):
        c1 = self.l1(self.stem(x))
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode='bilinear', align_corners=False)
        d = self.d3(torch.cat([up(c4, c3), c3], 1))
        d = self.d2(torch.cat([up(d, c2), c2], 1))
        d = self.d1(torch.cat([up(d, c1), c1], 1))
        return self.head(d)                                         # (B, 6, H/4, W/4)


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(bgr):
    x = (bgr[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class SimSet(torch.utils.data.Dataset):
    def __init__(self, root, ids, train=True, sigma=1.5):
        self.root, self.train, self.sigma = Path(root), train, sigma
        lab = json.loads((self.root / 'labels.json').read_text())
        self.size = lab['size']
        self.samples = [lab['samples'][i] for i in ids]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        s = self.samples[k]
        img = cv2.imread(str(self.root / f"{s['i']:06d}.jpg"))
        h, w = img.shape[:2]
        pts = [[] for _ in KEYS]
        for t in s['tines']:
            for c, name in enumerate(KEYS):
                if t[name] is not None and t[name + '_vis']:
                    pts[c].append(t[name])
        pts = [np.asarray(p, np.float32).reshape(-1, 2) for p in pts]
        if self.train:
            rng = np.random
            sc = rng.uniform(0.85, 1.2)                             # scale / shift / small turn
            ang = rng.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w / 2 + rng.uniform(-60, 60), h / 2 + rng.uniform(-40, 40)), ang, sc)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            pts = [p @ M[:, :2].T + M[:, 2] if len(p) else p for p in pts]
            if rng.random() < 0.5:
                img = img[:, ::-1]
                pts = [np.c_[w - 1 - p[:, 0], p[:, 1]] if len(p) else p for p in pts]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= rng.uniform(0.6, 1.3)
            hsv[..., 2] = hsv[..., 2] * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        oh, ow = h // STRIDE, w // STRIDE
        hm = np.zeros((2, oh, ow), np.float32)
        off = np.zeros((2, 2, oh, ow), np.float32)
        msk = np.zeros((2, oh, ow), np.float32)
        yy, xx = np.mgrid[0:oh, 0:ow]
        for c, p in enumerate(pts):
            for x, y in p:
                cx, cy = x / STRIDE, y / STRIDE
                ix, iy = int(cx), int(cy)
                if not (0 <= ix < ow and 0 <= iy < oh):
                    continue
                y0, y1, x0, x1 = max(0, iy - 5), min(oh, iy + 6), max(0, ix - 5), min(ow, ix + 6)
                g = np.exp(-((xx[y0:y1, x0:x1] - ix) ** 2 + (yy[y0:y1, x0:x1] - iy) ** 2) / (2 * self.sigma ** 2))
                hm[c, y0:y1, x0:x1] = np.maximum(hm[c, y0:y1, x0:x1], g)
                off[c, :, iy, ix] = (cx - ix, cy - iy)
                msk[c, iy, ix] = 1.0
        return to_tensor(np.ascontiguousarray(img)), torch.from_numpy(hm), torch.from_numpy(off), torch.from_numpy(msk)


def focal(pred, gt):
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    lp = -(torch.log(p) * (1 - p) ** 2 * pos).sum()
    ln = -(torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg).sum()
    return (lp + ln) / pos.sum().clamp(min=1)


def loss_fn(out, hm, off, msk):
    lh = focal(out[:, :2], hm)
    lo = 0.0
    for c in range(2):
        o = torch.sigmoid(out[:, 2 + 2 * c:4 + 2 * c])
        m = msk[:, c:c + 1]
        lo = lo + (torch.abs(o - off[:, c]) * m).sum() / m.sum().clamp(min=1)
    return lh + lo, lh, lo


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--data', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--epochs', type=int, default=12)
    a.add_argument('--bs', type=int, default=12)
    a.add_argument('--lr', type=float, default=5e-4)
    a.add_argument('--workers', type=int, default=8)
    a.add_argument('--init', help='Initialize model weights from an existing checkpoint; optimizer/schedule start fresh')
    a = a.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    n = len(json.loads((Path(a.data) / 'labels.json').read_text())['samples'])
    ids = np.random.default_rng(0).permutation(n)
    nval = min(500, n // 10)
    tr, va = SimSet(a.data, ids[nval:]), SimSet(a.data, ids[:nval], train=False)
    dl = torch.utils.data.DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
    dv = torch.utils.data.DataLoader(va, a.bs, num_workers=4)
    net = TineNet(pretrained=not bool(a.init)).cuda()
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location='cuda', weights_only=True))
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    log = open(out / 'train.jsonl', 'w')
    (out / 'args.json').write_text(json.dumps(vars(a), indent=1))
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        net.train()
        for x, hm, off, msk in dl:
            x, hm, off, msk = x.cuda(non_blocking=True), hm.cuda(), off.cuda(), msk.cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = net(x)
            l, lh, lo = loss_fn(o.float(), hm, off, msk)
            opt.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                rec = dict(step=step, ep=ep, loss=float(l), heat=float(lh), off=float(lo), lr=sched.get_last_lr()[0],
                           min=round((time.time() - t0) / 60, 1))
                log.write(json.dumps(rec) + '\n')
                log.flush()
                print(rec, flush=True)
        net.eval()
        vl = []
        with torch.no_grad():
            for x, hm, off, msk in dv:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    o = net(x.cuda())
                vl.append(float(loss_fn(o.float(), hm.cuda(), off.cuda(), msk.cuda())[0]))
        rec = dict(epoch=ep, val_loss=float(np.mean(vl)), min=round((time.time() - t0) / 60, 1))
        log.write(json.dumps(rec) + '\n')
        log.flush()
        print(rec, flush=True)
        torch.save(net.state_dict(), out / f'ep{ep:02d}.pt')
    torch.save(net.state_dict(), out / 'last.pt')


if __name__ == '__main__':
    main()
