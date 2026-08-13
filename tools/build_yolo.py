#!/usr/bin/env python3
"""Merge the three sources into one YOLO dataset with a shared class list.

  0 clover-3   ordinary three-leaf heads
  1 clover-4   the target

Keeping three-leaf as its own class rather than as background is deliberate:
the detector then has to tell the two apart directly, and the app can show
what it judged to be three-leaf too, so a wrong call is visible rather than
silent.

Five-leaf boxes are dropped — not the target, and not a fair negative either.
Splits follow each source's own train/valid/test folders; the deton photos are
split by photograph.
"""

import os, sys, glob, shutil, random
import xml.etree.ElementTree as ET
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(DATA, 'yolo')
NAMES = ['clover-3', 'clover-4']

# how each source's own labels map onto ours; None means drop the box
MAPS = {
    'adam-fonagy__hunting-for-four-leaf-clovers-yolo': {0: 0, 1: 1},
    'test-ara07__4-leaf-clover-detect-yolo':           {0: 1, 1: None},
}
VOC_MAP = {'4leaf': 1}


def ensure():
    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            os.makedirs(os.path.join(OUT, split, kind), exist_ok=True)


def copy_yolo(root, mapping, counts):
    for sub, split in (('train', 'train'), ('valid', 'val'), ('test', 'val')):
        img_dir = os.path.join(root, sub, 'images')
        lab_dir = os.path.join(root, sub, 'labels')
        if not os.path.isdir(img_dir):
            continue
        tag = os.path.basename(root)[:14]
        for ip in sorted(glob.glob(os.path.join(img_dir, '*'))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(lab_dir, stem + '.txt')
            if not os.path.exists(lp):
                continue
            lines = []
            for ln in open(lp):
                parts = ln.split()
                if len(parts) < 5:
                    continue
                new = mapping.get(int(parts[0]))
                if new is None:
                    continue
                lines.append(' '.join([str(new)] + parts[1:5]))
                counts[new] = counts.get(new, 0) + 1
            if not lines:
                continue
            name = f'{tag}_{stem}'
            shutil.copy(ip, os.path.join(OUT, split, 'images',
                                         name + os.path.splitext(ip)[1]))
            open(os.path.join(OUT, split, 'labels', name + '.txt'), 'w').write(
                '\n'.join(lines) + '\n')
            counts['img_' + split] = counts.get('img_' + split, 0) + 1


def copy_voc(counts):
    img_dir, xml_dir = os.path.join(DATA, 'img'), os.path.join(DATA, 'xml')
    if not os.path.isdir(xml_dir):
        return
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(xml_dir)
                   if f.endswith('.xml'))
    rng = random.Random(20260813)
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * 0.2))
    for i, s in enumerate(stems):
        ip = os.path.join(img_dir, s + '.jpg')
        if not os.path.exists(ip):
            continue
        split = 'val' if i < n_val else 'train'
        root = ET.parse(os.path.join(xml_dir, s + '.xml')).getroot()
        sz = root.find('size')
        W, H = int(sz.find('width').text), int(sz.find('height').text)
        lines = []
        for o in root.findall('object'):
            cls = VOC_MAP.get((o.find('name').text or '').strip().lower())
            if cls is None:
                continue
            bb = o.find('bndbox')
            x0, y0 = float(bb.find('xmin').text), float(bb.find('ymin').text)
            x1, y1 = float(bb.find('xmax').text), float(bb.find('ymax').text)
            lines.append(f'{cls} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} '
                         f'{(x1-x0)/W:.6f} {(y1-y0)/H:.6f}')
            counts[cls] = counts.get(cls, 0) + 1
        if not lines:
            continue
        name = f'deton_{s}'
        shutil.copy(ip, os.path.join(OUT, split, 'images', name + '.jpg'))
        open(os.path.join(OUT, split, 'labels', name + '.txt'), 'w').write(
            '\n'.join(lines) + '\n')
        counts['img_' + split] = counts.get('img_' + split, 0) + 1


def copy_bg(counts):
    """Photographs with no clover in them, carried in with no label file.

    Ultralytics reads an unlabelled image as background: everything in it is a
    negative. Without these, every training image is a photograph of clover and
    the detector is never given a reason to answer "not a plant" — pointed at a
    desk it reported four-leaf at 82% on a computer monitor.
    """
    src = os.path.join(DATA, 'bg')
    if not os.path.isdir(src):
        print('no background set — run tools/fetch_bg.py')
        return
    files = sorted(glob.glob(os.path.join(src, '*.jpg')))
    rng = random.Random(20260814)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * 0.15))
    for i, f in enumerate(files):
        split = 'val' if i < n_val else 'train'
        name = 'bg_' + os.path.basename(f)
        shutil.copy(f, os.path.join(OUT, split, 'images', name))
        # no label file at all: that is what marks it as background
        counts['bg_' + split] = counts.get('bg_' + split, 0) + 1


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    ensure()
    counts = {}
    for d, mapping in MAPS.items():
        root = os.path.join(DATA, 'roboflow', d)
        if os.path.isdir(root):
            copy_yolo(root, mapping, counts)
        else:
            print('missing', root)
    copy_voc(counts)
    copy_bg(counts)

    with open(os.path.join(OUT, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(OUT)}\n'
                f'train: train/images\nval: val/images\n'
                f'nc: {len(NAMES)}\nnames: {NAMES}\n')

    print(f'images  train {counts.get("img_train",0)}  val {counts.get("img_val",0)}')
    print(f'背景    train {counts.get("bg_train",0)}  val {counts.get("bg_val",0)}'
          f'   ({counts.get("bg_train",0)/max(1,counts.get("img_train",0)+counts.get("bg_train",0))*100:.0f}% of train)')
    for i, n in enumerate(NAMES):
        print(f'boxes   {n:9s} {counts.get(i,0)}')
    print('->', os.path.join(OUT, 'data.yaml'))


if __name__ == '__main__':
    main()
