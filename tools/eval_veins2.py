#!/usr/bin/env python3
"""Phase 1b evaluation on testdata/: go/no-go for the vein-convergence method."""
import os, sys, glob, math
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg2 import segment, vegetation
from classify2 import junctions
from veins2 import cell_axis_modes, solve_head, head_score
from eval2 import SCALE

W = 480

def evaluate(path, leaf_r, green=55):
    im = Image.open(path).convert('RGB')
    h = round(W * im.height / im.width)
    small = im.resize((W, h), Image.LANCZOS)
    rgb = np.asarray(small)
    res = segment(rgb, leaf_r, green=green)
    lum, veg = vegetation(rgb, green)
    modes = cell_axis_modes(lum, veg, res['labels'], res['cells'], leaf_r,
                            max(0.9, leaf_r / 26.0))
    byid = {c['id']: c for c in res['cells']}
    ideal = math.pi * leaf_r * leaf_r
    quads, trios = [], []
    for j in junctions(res['labels'], leaf_r):
        mem = [byid[i] for i in j['labels'] if i in byid and i in modes]
        mem = [c for c in mem if ideal*0.20 < c['area'] < ideal*3.0 and c['aspect'] < 2.6]
        n = len(mem)
        if n not in (3, 4):
            continue
        fit = solve_head(mem, modes, leaf_r)
        if fit is None:
            continue
        s, parts = head_score(mem, fit, leaf_r, n)
        rec = dict(n=n, x=float(fit['x'][0]), y=float(fit['x'][1]),
                   score=s, parts=parts, members=mem, fit=fit)
        (quads if n == 4 else trios).append(rec)
    quads.sort(key=lambda d: -d['score'])
    trios.sort(key=lambda d: -d['score'])
    return quads, trios, res, modes

if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    files = sorted(glob.glob(os.path.join(base, 'testdata', '*.jpg')))
    print(f'{"file":30s} {"leafR":>5s} {"cells":>5s} {"3":>4s} {"4":>4s} {"best4":>6s}  top (resid/gaps)')
    rows = []
    for f in files:
        name = os.path.basename(f).rsplit('.', 1)[0]
        r = SCALE.get(name, 26)
        quads, trios, res, modes = evaluate(f, r)
        best = quads[0]['score'] if quads else 0.0
        det = ' '.join(f"{q['score']*100:.0f}%(r{q['parts']['resid']:.2f},{q['parts']['gaps']})"
                       for q in quads[:2])
        rows.append((name, name.startswith('p'), best))
        print(f'{name:30s} {r:5d} {len(res["cells"]):5d} {len(trios):4d} {len(quads):4d} '
              f'{best*100:5.0f}%  {det}')
    pos = [b for _, h, b in rows if h]; neg = [b for _, h, b in rows if not h]
    print()
    print('positive:', [f'{v*100:.0f}' for v in sorted(pos, reverse=True)])
    print('negative:', [f'{v*100:.0f}' for v in sorted(neg, reverse=True)])
    print(f'min(pos)={min(pos)*100:.0f}  max(neg)={max(neg)*100:.0f}  '
          f'{"SEPARABLE" if min(pos) > max(neg) else "OVERLAP"}')
