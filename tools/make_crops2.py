#!/usr/bin/env python3
"""Build the 3-vs-4 crop set from all three sources.

The Roboflow "Hunting for four-leaf clovers" set is the important addition: it
labels 5,400 ordinary three-leaf heads as well as the four-leaf ones. Round 1
had to guess its negatives by mining, which taught the network "clover vs
grass"; here the hard negatives are human-labelled three-leaf clovers, so the
only thing separating the classes is the leaflet count.

Splits come from each dataset's own train/valid/test folders where they exist
(Roboflow splits before augmenting, so augmented copies never straddle a
split); the deton set is split by source photograph.

  four  <- 4-leaf boxes
  other <- labelled 3-leaf boxes, plus a little plain background
  5-leaf boxes are skipped entirely: not the target, and not a fair negative
"""

import os, sys, glob, json, random
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(DATA, 'crops2')
SIZE = 96
CTX = 1.7

FOUR = {'4leaf', '4-leaf-clovers', 'clover-4'}
THREE = {'clover-3', '3-leaf-clovers'}
SKIP = {'5-leaf-clovers', 'clover-5'}


def parse(xml_path):
    root = ET.parse(xml_path).getroot()
    out = []
    for o in root.findall('object'):
        name = (o.find('name').text or '').strip().lower()
        bb = o.find('bndbox')
        box = (float(bb.find('xmin').text), float(bb.find('ymin').text),
               float(bb.find('xmax').text), float(bb.find('ymax').text))
        out.append((name, box))
    return out


def square_crop(im, cx, cy, side):
    W, H = im.size
    side = min(side, W, H)
    half = side / 2
    cx = min(max(cx, half), W - half)
    cy = min(max(cy, half), H - half)
    b = (int(round(cx-half)), int(round(cy-half)),
         int(round(cx-half+side)), int(round(cy-half+side)))
    return im.crop(b).resize((SIZE, SIZE), Image.LANCZOS)


def green_fraction(arr):
    r, g, b = (arr[:, :, i].astype(np.float32) for i in range(3))
    return float((((2*g - r - b) / (r + g + b + 1)) > 0.06).mean())


def sources():
    """(image_path, xml_path, split) for every annotated image we have."""
    items = []

    # deton: flat img/ + xml/, split by photograph
    img_dir, xml_dir = os.path.join(DATA, 'img'), os.path.join(DATA, 'xml')
    if os.path.isdir(xml_dir):
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(xml_dir)
                       if f.endswith('.xml'))
        rng = random.Random(20260813)
        rng.shuffle(stems)
        n_val = max(1, int(len(stems) * 0.2))
        for i, s in enumerate(stems):
            ip = os.path.join(img_dir, s + '.jpg')
            if os.path.exists(ip):
                items.append((ip, os.path.join(xml_dir, s + '.xml'),
                              'val' if i < n_val else 'train'))

    # roboflow: train/ valid/ test/ subfolders, images and xml side by side
    for root in sorted(glob.glob(os.path.join(DATA, 'roboflow', '*'))):
        for sub, split in (('train', 'train'), ('valid', 'val'), ('test', 'val')):
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                continue
            for x in sorted(glob.glob(os.path.join(d, '*.xml'))):
                stem = os.path.splitext(x)[0]
                for ext in ('.jpg', '.jpeg', '.png'):
                    if os.path.exists(stem + ext):
                        items.append((stem + ext, x, split))
                        break
    return items


def main():
    rng = random.Random(20260813)
    for split in ('train', 'val'):
        for cls in ('four', 'other'):
            os.makedirs(os.path.join(OUT, split, cls), exist_ok=True)

    counts = {}
    kinds = {}
    for ip, xp, split in sources():
        try:
            objs = parse(xp)
        except Exception:
            continue
        if not objs:
            continue
        im = Image.open(ip).convert('RGB')
        W, H = im.size
        stem = os.path.splitext(os.path.basename(ip))[0][:60]
        skip_boxes = [b for n, b in objs if n in SKIP]

        sides = []
        for idx, (name, (x0, y0, x1, y1)) in enumerate(objs):
            if name in SKIP:
                continue
            if name in FOUR:
                cls = 'four'
            elif name in THREE:
                cls = 'other'
            else:
                kinds[name] = kinds.get(name, 0) + 1
                continue
            side = max(x1-x0, y1-y0) * CTX
            sides.append(side)
            cx, cy = (x0+x1)/2, (y0+y1)/2
            # a second jittered copy of each four-leaf: they are the rarer class
            reps = (2 if cls == 'four' else 1) if split == 'train' else 1
            for j in range(reps):
                jx = rng.uniform(-0.10, 0.10) * side if j else 0
                jy = rng.uniform(-0.10, 0.10) * side if j else 0
                js = side * (rng.uniform(0.88, 1.15) if j else 1.0)
                c = square_crop(im, cx+jx, cy+jy, js)
                c.save(os.path.join(OUT, split, cls, f'{stem}_{idx}_{j}.jpg'), quality=92)
                counts[(split, cls)] = counts.get((split, cls), 0) + 1

        # a little plain background so the net still knows grass and soil
        if sides and split == 'train':
            med = sorted(sides)[len(sides)//2]
            all_boxes = [b for _, b in objs]
            got, tries = 0, 0
            while got < 2 and tries < 60:
                tries += 1
                side = med * rng.uniform(0.85, 1.2)
                if side >= min(W, H):
                    break
                cx = rng.uniform(side/2, W - side/2)
                cy = rng.uniform(side/2, H - side/2)
                if any(cx > x0-side*0.3 and cx < x1+side*0.3 and
                       cy > y0-side*0.3 and cy < y1+side*0.3
                       for (x0, y0, x1, y1) in all_boxes + skip_boxes):
                    continue
                c = square_crop(im, cx, cy, side)
                if green_fraction(np.asarray(c)) < 0.40:
                    continue
                c.save(os.path.join(OUT, split, 'other', f'{stem}_bg{got}.jpg'), quality=92)
                counts[(split, 'other')] = counts.get((split, 'other'), 0) + 1
                got += 1

    for k in sorted(counts):
        print(f'{k[0]:5s} {k[1]:6s} {counts[k]:6d}')
    if kinds:
        print('unrecognised labels (ignored):', kinds)
    with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
        json.dump({f'{a}/{b}': v for (a, b), v in counts.items()}, f, indent=2)


if __name__ == '__main__':
    main()
