#!/usr/bin/env python3
"""Judge the detector on testdata/ — the same bar six earlier methods failed.

What matters is not mAP on the validation split but whether a box lands on the
actual four-leaf clover in photographs the model has never seen. Two frames
have caught every previous attempt:

  p1  a four-leaf held over paving — the dense classifier marked the paving
  p5  a four-leaf filling the frame — it marked a different plant at bottom right

If those two are not right, nothing else is worth reading.
"""

import os, sys, glob, time, argparse
import numpy as np
from PIL import Image, ImageDraw

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
COLOR = {'clover-4': (242, 180, 65), 'clover-3': (127, 179, 213)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=os.path.join(
        BASE, 'data', 'yolo_runs', 'v1', 'weights', 'best.pt'))
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--out')
    ap.add_argument('--show-three', action='store_true',
                    help='also draw the three-leaf detections')
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    names = model.names
    four_id = [k for k, v in names.items() if v == 'clover-4']
    four_id = four_id[0] if four_id else 1

    files = sorted(glob.glob(os.path.join(BASE, 'testdata', '*.jpg')))
    print(f'weights {os.path.relpath(a.weights, BASE)}   conf {a.conf}   imgsz {a.imgsz}')
    print(f'{"file":28s} {"truth":>6s} {"4-leaf":>7s} {"3-leaf":>7s} {"best4":>7s}   time')

    panels, t_all = [], 0.0
    n_pos_hit = n_neg_marks = 0
    for f in files:
        name = os.path.basename(f).rsplit('.', 1)[0]
        truth = '4-leaf' if name.startswith('p') else 'none'
        t0 = time.time()
        r = model.predict(f, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        dt = time.time() - t0
        t_all += dt

        boxes = r.boxes
        cls = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.zeros(0, int)
        conf = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))

        f4 = cls == four_id
        best4 = conf[f4].max() if f4.any() else 0.0
        print(f'{name:28s} {truth:>6s} {int(f4.sum()):7d} {int((~f4).sum()):7d} '
              f'{best4*100:6.1f}%   {dt:.2f}s')
        if truth == '4-leaf' and f4.any():
            n_pos_hit += 1
        if truth == 'none':
            n_neg_marks += int(f4.sum())

        if a.out:
            vis = Image.open(f).convert('RGB')
            vis.thumbnail((560, 560), Image.LANCZOS)
            sc = vis.width / r.orig_shape[1]
            d = ImageDraw.Draw(vis)
            order = np.argsort(conf)
            for i in order:
                is4 = cls[i] == four_id
                if not is4 and not a.show_three:
                    continue
                col = COLOR['clover-4'] if is4 else COLOR['clover-3']
                x0, y0, x1, y1 = xyxy[i] * sc
                d.rectangle([x0, y0, x1, y1], outline=col, width=4 if is4 else 2)
                if is4:
                    d.text((x0+4, y0+4), f'{conf[i]*100:.0f}', fill=col)
            d.rectangle([0, 0, vis.width, 18], fill=(10, 14, 11))
            d.text((4, 4), f'{name}  truth={truth}  4-leaf={int(f4.sum())} '
                           f'best={best4*100:.0f}%', fill=COLOR['clover-4'])
            panels.append(vis)

    negs = sum(1 for f in files if os.path.basename(f).startswith('n'))
    poss = len(files) - negs
    print(f'\npositives with a four-leaf box : {n_pos_hit}/{poss}')
    print(f'false four-leaf boxes per clean negative frame : {n_neg_marks/negs:.1f}')
    print(f'mean inference time : {t_all/len(files):.2f}s per frame (GPU)')

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
