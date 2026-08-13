#!/usr/bin/env python3
"""The honest test: slide the classifier over testdata/.

The validation split shares its negatives with training (grass, soil, ordinary
clover from the same photos), so 91% there could just mean "clover vs grass".
testdata/n*.jpg are dense three-leaf mats with no four-leaf clover at all — any
confident hit on those is a false positive on the hard case that matters.
"""

import os, sys, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import TinyNet, SIZE
from eval2 import SCALE

W = 640


def load_model(path):
    ck = torch.load(path, weights_only=False)
    m = TinyNet(len(ck['classes']))
    m.load_state_dict(ck['state'])
    m.eval()
    return m, ck['four_idx']


def scan(model, four_idx, img, leaf_r, scales=(5.0, 6.5, 8.0), stride_frac=0.25):
    """Return (score, x, y, side) for every window, plus a per-pixel max map."""
    W_, H_ = img.size
    heat = np.zeros((H_, W_), np.float32)
    hits = []
    for sf in scales:
        side = leaf_r * sf
        if side < 24 or side > min(W_, H_):
            continue
        step = max(8, int(side * stride_frac))
        xs = list(range(0, max(1, W_ - int(side) + 1), step))
        ys = list(range(0, max(1, H_ - int(side) + 1), step))
        batch, coords = [], []
        for y in ys:
            for x in xs:
                c = img.crop((x, y, x + int(side), y + int(side))).resize(
                    (SIZE, SIZE), Image.LANCZOS)
                batch.append(torch.from_numpy(
                    np.asarray(c, np.float32).transpose(2, 0, 1) / 255.0))
                coords.append((x, y, side))
        if not batch:
            continue
        with torch.no_grad():
            out = []
            for i in range(0, len(batch), 128):
                t = torch.stack(batch[i:i+128])
                out.append(F.softmax(model(t), 1)[:, four_idx])
            p = torch.cat(out).numpy()
        for (x, y, s), v in zip(coords, p):
            hits.append((float(v), x, y, s))
            reg = heat[int(y):int(y+s), int(x):int(x+s)]
            np.maximum(reg, v, out=reg)
    hits.sort(reverse=True)
    return hits, heat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(BASE, 'data', 'tinynet.pt'))
    ap.add_argument('--thresh', type=float, default=0.9)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    model, four_idx = load_model(a.model)
    files = sorted(os.listdir(os.path.join(BASE, 'testdata')))
    files = [f for f in files if f.endswith('.jpg')]

    print(f'threshold {a.thresh}')
    print(f'{"file":30s} {"truth":>6s} {"windows":>8s} {"over":>6s} {"rate":>7s} {"max":>6s}')
    panels = []
    for f in files:
        name = f.rsplit('.', 1)[0]
        truth = '4-leaf' if name.startswith('p') else 'none'
        im = Image.open(os.path.join(BASE, 'testdata', f)).convert('RGB')
        h = round(W * im.height / im.width)
        img = im.resize((W, h), Image.LANCZOS)
        leaf_r = SCALE.get(name, 26) * W / 480.0

        hits, heat = scan(model, four_idx, img, leaf_r)
        over = [x for x in hits if x[0] >= a.thresh]
        rate = len(over) / max(1, len(hits))
        print(f'{name:30s} {truth:>6s} {len(hits):8d} {len(over):6d} '
              f'{rate*100:6.2f}% {hits[0][0]*100 if hits else 0:5.1f}%')

        if a.out:
            vis = img.convert('RGB').copy()
            ov = (np.clip(heat, 0, 1) ** 3 * 190).astype(np.uint8)
            amber = Image.new('RGB', vis.size, (242, 180, 65))
            vis = Image.composite(amber, vis, Image.fromarray(ov, 'L'))
            d = ImageDraw.Draw(vis)
            for v, x, y, s in over[:8]:
                d.rectangle([x, y, x + s, y + s], outline=(255, 255, 255), width=3)
                d.text((x + 4, y + 4), f'{v*100:.0f}%', fill=(255, 255, 255))
            d.rectangle([0, 0, vis.width, 20], fill=(10, 14, 11))
            d.text((5, 5), f'{name}  truth={truth}  over{a.thresh}={len(over)}/{len(hits)}'
                           f'  max={hits[0][0]*100:.0f}%', fill=(242, 180, 65))
            panels.append(vis)

    if a.out and panels:
        cols = 3
        cw = 430
        th = [p.resize((cw, int(cw * p.height / p.width)), Image.LANCZOS) for p in panels]
        rh = max(p.height for p in th)
        rows = (len(th) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * cw, rows * rh), (10, 14, 11))
        for i, p in enumerate(th):
            sheet.paste(p, ((i % cols) * cw, (i // cols) * rh))
        sheet.save(a.out)
        print('saved', a.out)


if __name__ == '__main__':
    main()
