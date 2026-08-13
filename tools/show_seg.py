#!/usr/bin/env python3
"""Render the v2 segmentation so the cell borders can be eyeballed."""

import os, sys, math
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg2 import segment

W = 480


def load(path, w=W):
    im = Image.open(path).convert('RGB')
    h = round(w * im.height / im.width)
    return im.resize((w, h), Image.LANCZOS)


def draw(img, res, scale=2):
    lab = res['labels']
    h, w = lab.shape
    out = img.resize((w * scale, h * scale), Image.LANCZOS).convert('RGB')

    # cell borders in amber
    border = np.zeros((h, w), bool)
    border[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    border[:-1, :] |= lab[:-1, :] != lab[1:, :]
    border &= lab > 0
    bimg = Image.fromarray((border * 255).astype(np.uint8)).resize(
        (w * scale, h * scale), Image.NEAREST)
    amber = Image.new('RGB', out.size, (242, 180, 65))
    out = Image.composite(amber, out, bimg)

    d = ImageDraw.Draw(out)
    for c in res['cells']:
        d.ellipse([(c['x'] - 2) * scale, (c['y'] - 2) * scale,
                   (c['x'] + 2) * scale, (c['y'] + 2) * scale],
                  fill=(127, 179, 213))
    return out


if __name__ == '__main__':
    jobs = [(sys.argv[i], int(sys.argv[i + 1])) for i in range(1, len(sys.argv), 2)]
    outs = []
    for path, leaf_r in jobs:
        img = load(path)
        rgb = np.asarray(img)
        res = segment(rgb, leaf_r)
        big = [c for c in res['cells'] if c['area'] > 40]
        print(f'{os.path.basename(path):30s} leafR={leaf_r:3d} seeds={res["n_seeds"]:4d} '
              f'cells={len(res["cells"]):4d} (>40px: {len(big)})')
        im = draw(img, res)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 20], fill=(10, 14, 11))
        d.text((6, 5), f'{os.path.basename(path)}  leafR={leaf_r}  cells={len(res["cells"])}',
               fill=(242, 180, 65))
        outs.append(im)

    Wd = 780
    th = [o.resize((Wd, int(Wd * o.height / o.width)), Image.LANCZOS) for o in outs]
    sheet = Image.new('RGB', (Wd, sum(i.height for i in th) + 8 * len(th)), (10, 14, 11))
    y = 0
    for i in th:
        sheet.paste(i, (0, y)); y += i.height + 8
    sheet.save(os.environ.get('OUT', '/tmp/seg2.png'))
    print('saved', os.environ.get('OUT', '/tmp/seg2.png'))
