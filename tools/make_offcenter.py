#!/usr/bin/env python3
"""Teach the classifier that a half-caught clover is not a hit.

Every training crop so far was centred on something — a four-leaf, a
three-leaf, or plain background. A dense scan, though, spends most of its
windows partly overlapping objects, and the network had never been shown one.
That is why a window clipping the stem below a four-leaf came back at 95%.

So: for each four-leaf box, emit windows displaced far enough that the clover
sits at the edge, and label them 'other'. Standard practice for sliding-window
detectors, and it is the closest thing to a localisation loss that a plain
classifier can have.

Window side S = 1.7 * box, so the box is fully inside while the offset is
below 0.20*S. Offsets of 0.30-0.70*S leave it clipped by the frame edge.
"""

import os, sys, glob, random, math
import numpy as np
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_crops2 import (parse, square_crop, sources, FOUR, THREE, SKIP,
                         SIZE, CTX, OUT)


def main():
    rng = random.Random(4242)
    made = {'train': 0, 'val': 0}
    for ip, xp, split in sources():
        try:
            objs = parse(xp)
        except Exception:
            continue
        fours = [b for n, b in objs if n in FOUR]
        if not fours:
            continue
        im = Image.open(ip).convert('RGB')
        stem = os.path.splitext(os.path.basename(ip))[0][:60]
        reps = 2 if split == 'train' else 1

        for idx, (x0, y0, x1, y1) in enumerate(fours):
            side = max(x1-x0, y1-y0) * CTX
            bx, by = (x0+x1)/2, (y0+y1)/2
            for j in range(reps):
                # push the window off the clover, in a random direction
                d = side * rng.uniform(0.30, 0.70)
                a = rng.uniform(0, 2*math.pi)
                cx, cy = bx + math.cos(a)*d, by + math.sin(a)*d
                # ...but not so far that it lands centred on another four-leaf
                if any(abs(cx-(p0+p1)/2) < side*0.25 and abs(cy-(q0+q1)/2) < side*0.25
                       for (p0, q0, p1, q1) in fours):
                    continue
                c = square_crop(im, cx, cy, side)
                c.save(os.path.join(OUT, split, 'other',
                                    f'{stem}_off{idx}_{j}.jpg'), quality=92)
                made[split] += 1

    print(f'off-centre negatives added: {made}')
    for s in ('train', 'val'):
        for cl in ('four', 'other'):
            d = os.path.join(OUT, s, cl)
            print(f'  {s:5s} {cl:6s} {len(os.listdir(d)):6d}')


if __name__ == '__main__':
    main()
