#!/usr/bin/env python3
"""Phase 1 evaluation: can midrib angles separate three-leaf from four-leaf?

Two independent readings of the same vein axes:

  * as undirected lines, a four-leaf's four leaflets fall on only TWO
    directions (a perpendicular pair) while a three-leaf falls on THREE,
    60 degrees apart. Counting 2-vs-3 needs no sign for the axis.
  * the axis lines of one plant should all pass near its petiole. The
    residual of the best common intersection says whether the leaflets
    really belong together.

Go/no-go on the same benchmark every earlier method failed:
min(positive) > max(negative).
"""

import os, sys, glob, math
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg2 import segment, vegetation
from veins import ridges
from classify2 import junctions
from eval2 import SCALE

W = 480


# ------------------------------------------------------------------ axes

def cell_axes(lum, veg, labels, cells, leaf_r, sigma):
    """Dominant vein direction inside each cell, averaged with double angles
    so that opposite directions do not cancel."""
    s, dx, dy = ridges(lum, sigma, dark_lines=False)

    inner = np.ones(labels.shape, bool)
    inner[:, :-1] &= labels[:, :-1] == labels[:, 1:]
    inner[:-1, :] &= labels[:-1, :] == labels[1:, :]
    inner[:, 1:] &= labels[:, 1:] == labels[:, :-1]
    inner[1:, :] &= labels[1:, :] == labels[:-1, :]
    for _ in range(max(1, int(leaf_r * 0.10))):
        e = inner.copy()
        e[:, :-1] &= inner[:, 1:]; e[:, 1:] &= inner[:, :-1]
        e[:-1, :] &= inner[1:, :]; e[1:, :] &= inner[:-1, :]
        inner = e

    ok = inner & veg & (s > s.max() * 0.10)
    th = np.arctan2(dy, dx)
    K = int(labels.max()) + 1
    L = labels.ravel(); m = ok.ravel()
    sc = np.bincount(L[m], weights=(np.cos(2 * th) * s).ravel()[m], minlength=K)
    ss = np.bincount(L[m], weights=(np.sin(2 * th) * s).ravel()[m], minlength=K)
    sw = np.bincount(L[m], weights=s.ravel()[m], minlength=K)
    cnt = np.bincount(L[m], minlength=K)

    axes = {}
    for c in cells:
        i = c['id']
        if cnt[i] < 15:
            continue
        ang = 0.5 * math.atan2(ss[i], sc[i])          # mod pi
        coh = math.hypot(sc[i], ss[i]) / max(sw[i], 1e-6)
        axes[i] = (ang, coh)
    return axes


# ------------------------------------------------------------------ scoring

def circ_modes_pi(angles, weights, nbins=36, min_frac=0.40):
    """Modes of a distribution of undirected directions (period pi)."""
    hist = np.zeros(nbins, np.float32)
    for a, w in zip(angles, weights):
        hist[int((a % math.pi) / math.pi * nbins) % nbins] += w
    k = np.array([1, 2, 3, 2, 1], np.float32); k /= k.sum()
    hist = np.convolve(np.r_[hist, hist, hist], k, 'same')[nbins:2 * nbins]
    if hist.max() <= 0:
        return [], hist
    thr = hist.max() * min_frac
    idx = [i for i in range(nbins)
           if hist[i] >= thr and hist[i] >= hist[(i - 1) % nbins]
           and hist[i] > hist[(i + 1) % nbins]]
    return [i * 180.0 / nbins for i in idx], hist


def intersect_residual(members, axes, leaf_r):
    """Least-squares common intersection of the axis lines, and its residual."""
    A = np.zeros((2, 2)); b = np.zeros(2)
    pts = []
    for c in members:
        ang, _ = axes[c['id']]
        u = np.array([math.cos(ang), math.sin(ang)])
        P = np.eye(2) - np.outer(u, u)
        p = np.array([c['x'], c['y']])
        A += P; b += P @ p
        pts.append((p, u))
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, 1e9
    res = 0.0
    for p, u in pts:
        d = (x - p) - np.dot(x - p, u) * u
        res += float(d @ d)
    return x, math.sqrt(res / len(pts)) / leaf_r


def spacing_score(mode_angles, expect_gap):
    if len(mode_angles) < 2:
        return 0.0
    gaps = []
    for i, a in enumerate(mode_angles):
        nxt = mode_angles[0] + 180 if i == len(mode_angles) - 1 else mode_angles[i + 1]
        gaps.append(nxt - a)
    err = sum(abs(g - expect_gap) for g in gaps) / len(gaps)
    return max(0.0, 1.0 - err / (expect_gap * 0.6))


def evaluate(path, leaf_r, green=55):
    im = Image.open(path).convert('RGB')
    h = round(W * im.height / im.width)
    small = im.resize((W, h), Image.LANCZOS)
    rgb = np.asarray(small)

    res = segment(rgb, leaf_r, green=green)
    lum, veg = vegetation(rgb, green)
    axes = cell_axes(lum, veg, res['labels'], res['cells'], leaf_r,
                     max(0.9, leaf_r / 26.0))
    byid = {c['id']: c for c in res['cells']}

    ideal = math.pi * leaf_r * leaf_r
    quads, trios = [], []
    for j in junctions(res['labels'], leaf_r):
        mem = [byid[i] for i in j['labels'] if i in byid and i in axes]
        mem = [c for c in mem if ideal * 0.20 < c['area'] < ideal * 3.0
               and c['aspect'] < 2.6]
        n = len(mem)
        if n not in (3, 4):
            continue
        angs = [axes[c['id']][0] for c in mem]
        cohs = [axes[c['id']][1] for c in mem]
        modes, _ = circ_modes_pi(angs, cohs)
        pt, resid = intersect_residual(mem, axes, leaf_r)

        s_res = max(0.0, 1.0 - resid / 1.2)
        if n == 4:
            s_cnt = 1.0 if len(modes) == 2 else 0.35 if len(modes) == 3 else 0.0
            s_sp = spacing_score(modes, 90.0)
        else:
            s_cnt = 1.0 if len(modes) == 3 else 0.35 if len(modes) == 2 else 0.0
            s_sp = spacing_score(modes, 60.0)
        score = 0.42 * s_cnt + 0.28 * s_sp + 0.30 * s_res
        rec = dict(n=n, x=j['x'], y=j['y'], score=score, modes=len(modes),
                   resid=resid, members=mem)
        (quads if n == 4 else trios).append(rec)

    quads.sort(key=lambda d: -d['score'])
    trios.sort(key=lambda d: -d['score'])
    return quads, trios, res, axes


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'testdata', '*.jpg')))
    print(f'{"file":30s} {"leafR":>5s} {"cells":>5s} {"axes":>5s} '
          f'{"3":>4s} {"4":>4s} {"best4":>6s}  top-4 (modes/resid)')
    rows = []
    for f in files:
        name = os.path.basename(f).rsplit('.', 1)[0]
        r = SCALE.get(name, 26)
        quads, trios, res, axes = evaluate(f, r)
        best = quads[0]['score'] if quads else 0.0
        detail = ' '.join(f"{q['score']*100:.0f}%(m{q['modes']},r{q['resid']:.2f})"
                          for q in quads[:3])
        rows.append((name, name.startswith('p'), best))
        print(f'{name:30s} {r:5d} {len(res["cells"]):5d} {len(axes):5d} '
              f'{len(trios):4d} {len(quads):4d} {best*100:5.0f}%  {detail}')

    pos = [b for _, h, b in rows if h]
    neg = [b for _, h, b in rows if not h]
    print()
    print('positive:', [f'{v*100:.0f}' for v in sorted(pos, reverse=True)])
    print('negative:', [f'{v*100:.0f}' for v in sorted(neg, reverse=True)])
    print(f'min(pos)={min(pos)*100:.0f}  max(neg)={max(neg)*100:.0f}  '
          f'{"SEPARABLE" if min(pos) > max(neg) else "OVERLAP"}')
