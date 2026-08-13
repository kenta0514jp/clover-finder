#!/usr/bin/env python3
"""Score every test image with the v2 pipeline and report the separation
between images that contain a four-leaf clover and images that do not."""

import os, sys, glob, math, time
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg2 import segment
from classify2 import heads

W = 480

# leaf radius in px at W=480, judged from the photos; in the app this is the
# "小葉の大きさ" slider that the user sets for their camera distance
SCALE = {
    'n1-lawn-backyard': 16,
    'n5-white-clover': 26,
    'n6-clover-leaf-nc': 30,
    'p1-four-in-grass-poland': 34,
    'p2-four-trifolium-repens': 16,
    'p3-four-in-hand': 30,
    'p4-four-poland': 30,
    'p5-four-closeup': 45,
    'p6-three-four-clovers': 26,
}
HAS_FOUR = lambda name: name.startswith('p')


def run(path, leaf_r, out_dir=None, merge_t=26.0):
    img = Image.open(path).convert('RGB')
    h = round(W * img.height / img.width)
    small = img.resize((W, h), Image.LANCZOS)
    res = segment(np.asarray(small), leaf_r, merge_t=merge_t)
    hs = heads(res['labels'], res['cells'], leaf_r)
    quads = [d for d in hs if d['n'] == 4]
    trios = [d for d in hs if d['n'] == 3]

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        sc = 2
        vis = small.resize((W * sc, h * sc), Image.LANCZOS).convert('RGB')
        lab = res['labels']
        b = np.zeros(lab.shape, bool)
        b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        b[:-1, :] |= lab[:-1, :] != lab[1:, :]
        b &= lab > 0
        bimg = Image.fromarray((b * 255).astype(np.uint8)).resize(vis.size, Image.NEAREST)
        vis = Image.composite(Image.new('RGB', vis.size, (90, 120, 100)), vis, bimg)
        d = ImageDraw.Draw(vis)
        for t in trios[:40]:
            d.ellipse([(t['x'] - 4) * sc, (t['y'] - 4) * sc,
                       (t['x'] + 4) * sc, (t['y'] + 4) * sc], outline=(127, 179, 213), width=2)
        for q in quads:
            r = q['r'] * sc
            d.ellipse([q['x'] * sc - r, q['y'] * sc - r, q['x'] * sc + r, q['y'] * sc + r],
                      outline=(242, 180, 65), width=4)
            d.text((q['x'] * sc - r, q['y'] * sc - r - 12),
                   f"4x {q['score']*100:.0f}%", fill=(242, 180, 65))
        vis.save(os.path.join(out_dir, os.path.basename(path).rsplit('.', 1)[0] + '.png'))

    return quads, trios, res


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(glob.glob('testdata/*.jpg'))
    print(f'{"file":30s} {"leafR":>5s} {"cells":>5s} {"3-leaf":>6s} {"4-leaf":>6s} '
          f'{"best4":>6s} {"top-3":>18s}')
    rows = []
    for f in files:
        name = os.path.basename(f).rsplit('.', 1)[0]
        r = SCALE.get(name, 26)
        t0 = time.time()
        quads, trios, res = run(f, r, out)
        best = quads[0]['score'] if quads else 0.0
        top = ' '.join(f'{q["score"]*100:.0f}' for q in quads[:3])
        rows.append((name, HAS_FOUR(name), best, len(quads)))
        print(f'{name:30s} {r:5d} {len(res["cells"]):5d} {len(trios):6d} {len(quads):6d} '
              f'{best*100:5.0f}% {top:>18s}   ({time.time()-t0:.0f}s)')

    print()
    pos = [b for _, h, b, _ in rows if h]
    neg = [b for _, h, b, _ in rows if not h]
    print(f'four-leaf present : best scores {[f"{v*100:.0f}" for v in sorted(pos, reverse=True)]}')
    print(f'four-leaf absent  : best scores {[f"{v*100:.0f}" for v in sorted(neg, reverse=True)]}')
    print(f'separation        : min(pos)={min(pos)*100:.0f}%  max(neg)={max(neg)*100:.0f}%  '
          f'{"SEPARABLE" if min(pos) > max(neg) else "OVERLAP"}')
