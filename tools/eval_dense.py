#!/usr/bin/env python3
"""Measure the whole-frame detector so its threshold can be set from evidence.

With valid convolutions the dense pass is exactly the sliding window, so what
is measured here is what ships: how many spots does it light up on frames
containing no four-leaf clover at all, and does it light up the right spot
on frames that do.
"""

import os, sys, glob, time, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense import load, dense_scores, to_tensor, CELL
from train import SIZE
from eval2 import SCALE

# the training crops framed a head at roughly 56 px, so analyse the frame at a
# few sizes and let whichever matches the plant win
SCALES = (192, 256, 340)


def scan_frame(model, fi, img, scales=SCALES):
    """Returns list of (score, cx, cy, side) in original-image coordinates."""
    W0, H0 = img.size
    hits = []
    maps = []
    for w in scales:
        h = round(w * H0 / W0)
        if min(w, h) < SIZE + CELL:
            continue
        small = img.resize((w, h), Image.LANCZOS)
        m = dense_scores(model, fi, to_tensor(small))
        maps.append((w, m))
        k = W0 / w                      # back to original pixels
        for j in range(m.shape[0]):
            for i in range(m.shape[1]):
                cx = (i * CELL + SIZE / 2) * k
                cy = (j * CELL + SIZE / 2) * k
                hits.append((float(m[j, i]), cx, cy, SIZE * k))
    hits.sort(reverse=True)
    return hits, maps


def peak_pick(hits, thresh, max_n=12):
    kept = []
    for s, cx, cy, side in hits:
        if s < thresh:
            break
        if any(abs(cx - a) < side * 0.55 and abs(cy - b) < side * 0.55
               for _, a, b, _ in kept):
            continue
        kept.append((s, cx, cy, side))
        if len(kept) >= max_n:
            break
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--model', default=os.path.join(BASE,'data','valid2.pt'))
    ap.add_argument('--thresholds', default='0.90,0.95,0.98,0.995')
    a = ap.parse_args()
    THR = [float(t) for t in a.thresholds.split(',')]

    model, fi = load(a.model)
    files = sorted(glob.glob(os.path.join(BASE, 'testdata', '*.jpg')))

    hdr = ' '.join(f'{t:>7.3f}' for t in THR)
    print(f'{"file":28s} {"truth":>6s} {"cells":>6s} {"max":>6s}   marks at {hdr}')
    panels = []
    t_total = 0.0
    for f in files:
        name = os.path.basename(f).rsplit('.', 1)[0]
        truth = '4-leaf' if name.startswith('p') else 'none'
        img = Image.open(f).convert('RGB')
        t0 = time.time()
        hits, maps = scan_frame(model, fi, img)
        dt = time.time() - t0
        t_total += dt
        counts = [len(peak_pick(hits, t)) for t in THR]
        best = hits[0][0] if hits else 0
        cs = ' '.join(f'{c:7d}' for c in counts)
        print(f'{name:28s} {truth:>6s} {len(hits):6d} {best*100:5.1f}%   {cs}   ({dt:.2f}s)')

        if a.out:
            vis = img.copy()
            vis.thumbnail((520, 520), Image.LANCZOS)
            sc = vis.width / img.width
            d = ImageDraw.Draw(vis)
            for s, cx, cy, side in peak_pick(hits, THR[1]):
                r = side * sc / 2
                d.rectangle([cx*sc-r, cy*sc-r, cx*sc+r, cy*sc+r],
                            outline=(242, 180, 65), width=3)
                d.text((cx*sc-r+3, cy*sc-r+3), f'{s*100:.0f}', fill=(242, 180, 65))
            d.rectangle([0, 0, vis.width, 18], fill=(10, 14, 11))
            d.text((4, 4), f'{name}  truth={truth}  max={best*100:.0f}%',
                   fill=(242, 180, 65))
            panels.append(vis)

    print(f'\ntotal scan time {t_total:.1f}s for {len(files)} frames '
          f'({t_total/len(files):.2f}s each, PyTorch CPU, {len(SCALES)} scales)')

    negs = [f for f in files if os.path.basename(f).startswith('n')]
    print(f'\nfalse marks per frame on the {len(negs)} clean negatives:')
    for t in THR:
        tot = 0
        for f in negs:
            img = Image.open(f).convert('RGB')
            hits, _ = scan_frame(model, fi, img)
            tot += len(peak_pick(hits, t))
        print(f'  threshold {t:.3f} -> {tot/len(negs):.1f} marks per frame')

    if a.out and panels:
        cols = 3
        cw = max(p.width for p in panels)
        rh = max(p.height for p in panels)
        rows = (len(panels) + cols - 1) // cols
        sheet = Image.new('RGB', (cols*cw, rows*rh), (10, 14, 11))
        for i, p in enumerate(panels):
            sheet.paste(p, ((i % cols)*cw, (i // cols)*rh))
        sheet.save(a.out)
        print('saved', a.out)


if __name__ == '__main__':
    main()
