#!/usr/bin/env python3
"""Build the 3-vs-4 crop dataset from deton/detect4clover (CC BY 4.0).

Only four-leaf clovers are annotated there, but every photo is a lawn full of
ordinary clover, so the negatives are mined from the SAME images. That matters:
if negatives came from other photos the network could pass by recognising the
camera, the lawn or the light instead of the leaflets.

Split is by source image, never by crop, so no photo appears in both sets.
"""

import os, sys, math, random, json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(DATA, 'crops')
SIZE = 96
CTX = 1.7          # crop this much wider than the annotated box


def boxes_for(xml_path):
    root = ET.parse(xml_path).getroot()
    out = []
    for o in root.findall('object'):
        bb = o.find('bndbox')
        out.append((int(bb.find('xmin').text), int(bb.find('ymin').text),
                    int(bb.find('xmax').text), int(bb.find('ymax').text)))
    return out


def square_crop(im, cx, cy, side):
    """Keep the window inside the picture — a box that runs off the edge comes
    back padded with black, which would teach the net that black means clover."""
    W, H = im.size
    side = min(side, W, H)
    half = side / 2
    cx = min(max(cx, half), W - half)
    cy = min(max(cy, half), H - half)
    box = (int(round(cx - half)), int(round(cy - half)),
           int(round(cx - half + side)), int(round(cy - half + side)))
    return im.crop(box).resize((SIZE, SIZE), Image.LANCZOS)


def green_fraction(arr):
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    exg = (2*g - r - b) / (r + g + b + 1)
    return float((exg > 0.06).mean())


def overlaps(box, others, pad):
    x0, y0, x1, y1 = box
    for (a0, b0, a1, b1) in others:
        if x0 < a1 + pad and a0 - pad < x1 and y0 < b1 + pad and b0 - pad < y1:
            return True
    return False


def main():
    rng = random.Random(20260813)
    img_dir, xml_dir = os.path.join(DATA, 'img'), os.path.join(DATA, 'xml')
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(xml_dir)
                   if f.endswith('.xml'))
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * 0.2))
    val_stems = set(stems[:n_val])

    for split in ('train', 'val'):
        for cls in ('four', 'other'):
            os.makedirs(os.path.join(OUT, split, cls), exist_ok=True)

    counts = {('train','four'):0, ('train','other'):0,
              ('val','four'):0, ('val','other'):0}
    manifest = []

    for stem in stems:
        xp = os.path.join(xml_dir, stem + '.xml')
        ip = os.path.join(img_dir, stem + '.jpg')
        if not os.path.exists(ip):
            continue
        im = Image.open(ip).convert('RGB')
        W, H = im.size
        bxs = boxes_for(xp)
        if not bxs:
            continue
        split = 'val' if stem in val_stems else 'train'

        sides = []
        for k, (x0, y0, x1, y1) in enumerate(bxs):
            side = max(x1-x0, y1-y0) * CTX
            sides.append(side)
            cx, cy = (x0+x1)/2, (y0+y1)/2
            # small jitter so the clover is not always dead centre
            for j in range(3 if split == 'train' else 1):
                jx = rng.uniform(-0.10, 0.10) * side if j else 0
                jy = rng.uniform(-0.10, 0.10) * side if j else 0
                js = side * (rng.uniform(0.88, 1.15) if j else 1.0)
                c = square_crop(im, cx+jx, cy+jy, js)
                fn = f'{stem}_{k}_{j}.jpg'
                c.save(os.path.join(OUT, split, 'four', fn), quality=92)
                counts[(split,'four')] += 1
                manifest.append(dict(file=f'{split}/four/{fn}', stem=stem, cls='four'))

        # negatives: same photo, same scale, away from every annotated clover
        med_side = sorted(sides)[len(sides)//2]
        want = len(bxs) * (4 if split == 'train' else 2)
        tries = 0
        got = 0
        while got < want and tries < want * 60:
            tries += 1
            side = med_side * rng.uniform(0.85, 1.20)
            if side >= min(W, H):
                break
            cx = rng.uniform(side/2, W - side/2)
            cy = rng.uniform(side/2, H - side/2)
            box = (cx-side/2, cy-side/2, cx+side/2, cy+side/2)
            if overlaps(box, bxs, pad=side*0.25):
                continue
            c = square_crop(im, cx, cy, side)
            if green_fraction(np.asarray(c)) < 0.45:
                continue                      # want clover/grass, not soil or sky
            fn = f'{stem}_n{got}.jpg'
            c.save(os.path.join(OUT, split, 'other', fn), quality=92)
            counts[(split,'other')] += 1
            manifest.append(dict(file=f'{split}/other/{fn}', stem=stem, cls='other'))
            got += 1

    with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
        json.dump(dict(counts={f'{a}/{b}': v for (a,b), v in counts.items()},
                       val_images=len(val_stems), train_images=len(stems)-len(val_stems),
                       size=SIZE, context=CTX,
                       source='deton/detect4clover v1.0.0, CC BY 4.0'),
                  f, indent=2, ensure_ascii=False)
    for k, v in counts.items():
        print(f'{k[0]:5s} {k[1]:6s} {v:5d}')
    print(f'images: {len(stems)-len(val_stems)} train / {len(val_stems)} val')


if __name__ == '__main__':
    main()
