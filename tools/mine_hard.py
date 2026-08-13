#!/usr/bin/env python3
"""Hard-negative mining.

The first round learned "clover vs grass" because almost every negative was
grass or soil. Here the current model is run back over the training photos and
every confident window that is NOT an annotated four-leaf is kept as a
negative — mostly ordinary three-leaf heads, which is precisely the comparison
the classifier has to make.

Only images from the training split are mined, so the validation photos stay
untouched.
"""

import os, sys, json, argparse
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import TinyNet, SIZE
from make_crops import boxes_for, square_crop, green_fraction, overlaps

DATA = os.path.join(BASE, 'data')
CROPS = os.path.join(DATA, 'crops')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(DATA, 'tinynet.pt'))
    ap.add_argument('--thresh', type=float, default=0.55)
    ap.add_argument('--per-image', type=int, default=6)
    a = ap.parse_args()

    ck = torch.load(a.model, weights_only=False)
    model = TinyNet(len(ck['classes'])); model.load_state_dict(ck['state']); model.eval()
    four_idx = ck['four_idx']

    # which photos are in the training split? read it back off the crop names
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
        W, H = im.size
        bxs = boxes_for(xp)
        if not bxs:
            continue
        sides = [max(x1-x0, y1-y0) * 1.7 for (x0, y0, x1, y1) in bxs]
        med = sorted(sides)[len(sides)//2]

        cand, coords = [], []
        for sf in (0.8, 1.0, 1.3):
            side = med * sf
            if side < 24 or side >= min(W, H):
                continue
            step = max(10, int(side * 0.30))
            for y in range(0, int(H - side) + 1, step):
                for x in range(0, int(W - side) + 1, step):
                    box = (x, y, x + side, y + side)
                    if overlaps(box, bxs, pad=side * 0.30):
                        continue
                    c = im.crop((x, y, int(x+side), int(y+side))).resize(
                        (SIZE, SIZE), Image.LANCZOS)
                    arr = np.asarray(c)
                    if green_fraction(arr) < 0.55:
                        continue
                    cand.append(torch.from_numpy(
                        arr.astype(np.float32).transpose(2, 0, 1) / 255.0))
                    coords.append((x, y, side))
        if not cand:
            continue
        with torch.no_grad():
            ps = []
            for i in range(0, len(cand), 256):
                ps.append(F.softmax(model(torch.stack(cand[i:i+256])), 1)[:, four_idx])
            p = torch.cat(ps).numpy()

        order = np.argsort(-p)
        taken = 0
        for k in order:
            if p[k] < a.thresh or taken >= a.per_image:
                break
            x, y, side = coords[k]
            c = square_crop(im, x + side/2, y + side/2, side)
            c.save(os.path.join(out_dir, f'{stem}_h{taken}.jpg'), quality=92)
            taken += 1; kept += 1
        if (n + 1) % 50 == 0:
            print(f'  {n+1}/{len(train_stems)} images, {kept} hard negatives')

    print(f'mined {kept} hard negatives into {out_dir}')
    print('train/other now:', len(os.listdir(out_dir)))


if __name__ == '__main__':
    main()
