#!/usr/bin/env python3
"""Hard-negative mining in the deployment mode.

The previous miner cropped windows one at a time. Now that the network is
fully convolutional, a whole training photo is scanned in a single pass, so
mining runs over every position at several scales — and the negatives it finds
are exactly the ones that will light up on screen during a whole-frame sweep.
"""

import os, sys, argparse
import numpy as np
import torch
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense import load, dense_scores, to_tensor, CELL
from train import SIZE
from make_crops import boxes_for, square_crop, green_fraction

DATA = os.path.join(BASE, 'data')
CROPS = os.path.join(DATA, 'crops')
SCALES = (160, 200, 260, 340)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(DATA, 'valid1.pt'))
    ap.add_argument('--thresh', type=float, default=0.50)
    ap.add_argument('--per-image', type=int, default=8)
    ap.add_argument('--tag', default='d')
    a = ap.parse_args()

    model, fi = load(a.model)

    train_stems = set()
    for f in os.listdir(os.path.join(CROPS, 'train', 'four')):
        train_stems.add(f.rsplit('_', 2)[0])

    out_dir = os.path.join(CROPS, 'train', 'other')
    kept = 0
    for n, stem in enumerate(sorted(train_stems)):
        ip = os.path.join(DATA, 'img', stem + '.jpg')
        xp = os.path.join(DATA, 'xml', stem + '.xml')
        if not (os.path.exists(ip) and os.path.exists(xp)):
            continue
        im = Image.open(ip).convert('RGB')
        W0, H0 = im.size
        bxs = boxes_for(xp)
        if not bxs:
            continue

        found = []
        for w in SCALES:
            h = round(w * H0 / W0)
            if min(w, h) < SIZE + CELL:
                continue
            m = dense_scores(model, fi, to_tensor(im.resize((w, h), Image.LANCZOS)))
            k = W0 / w
            side = SIZE * k
            ys, xs = np.nonzero(m >= a.thresh)
            for j, i in zip(ys, xs):
                cx = (i * CELL + SIZE / 2) * k
                cy = (j * CELL + SIZE / 2) * k
                # skip anything touching an annotated four-leaf
                if any(cx > x0 - side*0.35 and cx < x1 + side*0.35 and
                       cy > y0 - side*0.35 and cy < y1 + side*0.35
                       for (x0, y0, x1, y1) in bxs):
                    continue
                found.append((float(m[j, i]), cx, cy, side))

        found.sort(reverse=True)
        taken = []
        for s, cx, cy, side in found:
            if len(taken) >= a.per_image:
                break
            if any(abs(cx-ax) < side*0.6 and abs(cy-ay) < side*0.6 for ax, ay in taken):
                continue
            c = square_crop(im, cx, cy, side)
            if green_fraction(np.asarray(c)) < 0.30:
                continue
            c.save(os.path.join(out_dir, f'{stem}_{a.tag}{len(taken)}.jpg'), quality=92)
            taken.append((cx, cy)); kept += 1
        if (n + 1) % 60 == 0:
            print(f'  {n+1}/{len(train_stems)} images, {kept} mined')

    print(f'mined {kept} dense-mode hard negatives')
    print('train/other now:', len(os.listdir(out_dir)))


if __name__ == '__main__':
    main()
