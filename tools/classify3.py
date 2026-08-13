#!/usr/bin/env python3
"""Find clover heads by letting each leaflet vote for its own petiole.

Every leaflet knows which way its stalk end points (seg2 computes that from
the second moments). The leaflets of one plant all point at the same spot, so
accumulating a ray of votes per leaflet makes that spot a peak. Reading the
peak back gives the members, and len(members) is the leaflet count.

This replaces the junction search in classify2, which kept locking onto
incidental 3-way border meetings instead of the real petiole.
"""

import math
import numpy as np


def clamp01(v):
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def cv(vals):
    m = sum(vals) / len(vals)
    if m <= 0:
        return 1.0
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) / m


def vote(cells, shape, leaf_r, bin_px=4):
    """Hough-style accumulation of petiole votes."""
    h, w = shape
    bh, bw = h // bin_px + 1, w // bin_px + 1
    acc = np.zeros((bh, bw), np.float32)
    trails = []
    for c in cells:
        # sweep the plausible stalk distance instead of guessing one value
        lo, hi = 0.45 * c['r'], 2.2 * c['r']
        pts = []
        t = lo
        while t <= hi:
            x = c['x'] + c['px'] * t
            y = c['y'] + c['py'] * t
            if 0 <= x < w and 0 <= y < h:
                acc[int(y // bin_px), int(x // bin_px)] += 1.0
                pts.append((x, y))
            t += bin_px * 0.5
        trails.append(pts)
    return acc, trails


def peaks(acc, bin_px, leaf_r, min_votes=3):
    """Local maxima of the accumulator, at least a head apart."""
    bh, bw = acc.shape
    sep = max(1, int(leaf_r * 0.75 / bin_px))
    sm = acc.copy()
    # 3x3 smoothing keeps a peak from being split across neighbouring bins
    p = np.pad(acc, 1, mode='constant')
    sm = sum(p[dy:dy + bh, dx:dx + bw] for dy in range(3) for dx in range(3)) / 9.0

    idx = np.argsort(sm.ravel())[::-1]
    taken = np.zeros(acc.shape, bool)
    out = []
    for k in idx:
        v = sm.ravel()[k]
        if v < min_votes / 9.0:
            break
        y, x = divmod(int(k), bw)
        y0, y1 = max(0, y - sep), min(bh, y + sep + 1)
        x0, x1 = max(0, x - sep), min(bw, x + sep + 1)
        if taken[y0:y1, x0:x1].any():
            continue
        taken[y, x] = True
        out.append(((x + 0.5) * bin_px, (y + 0.5) * bin_px, float(v)))
    return out


def score_head(members, jx, jy, expect):
    ang = sorted(math.atan2(c['y'] - jy, c['x'] - jx) for c in members)
    gaps = []
    for i, a in enumerate(ang):
        nxt = ang[0] + math.tau if i == len(ang) - 1 else ang[i + 1]
        gaps.append((nxt - a) * 180 / math.pi)
    ideal = 360 / expect
    s_ang = clamp01(1 - (sum(abs(g - ideal) for g in gaps) / len(gaps)) / (ideal * 0.75))

    dists = [math.hypot(c['x'] - jx, c['y'] - jy) for c in members]
    s_rad = clamp01(1 - cv(dists) / 0.42)
    s_area = clamp01(1 - cv([c['area'] for c in members]) / 0.55)

    cosines = []
    for c in members:
        vx, vy = jx - c['x'], jy - c['y']
        n = math.hypot(vx, vy)
        cosines.append(0.0 if n < 1e-6 else (c['px'] * vx + c['py'] * vy) / n)
    s_orient = clamp01((sum(cosines) / len(cosines) - 0.55) / 0.42)
    s_worst = clamp01((min(cosines) - 0.30) / 0.55)

    s_shape = clamp01(1 - (sum(c['aspect'] for c in members) / len(members) - 1.0) / 1.2)

    score = (0.22 * s_ang + 0.12 * s_rad + 0.12 * s_area
             + 0.24 * s_orient + 0.18 * s_worst + 0.12 * s_shape)
    return score, dict(ang=s_ang, rad=s_rad, area=s_area, orient=s_orient,
                       worst=s_worst, shape=s_shape,
                       mean_cos=sum(cosines) / len(cosines), worst_cos=min(cosines))


def heads(lab_shape, cell_list, leaf_r, min_area_frac=0.20, max_area_frac=3.0,
          bin_px=4):
    ideal = math.pi * leaf_r * leaf_r
    lo, hi = ideal * min_area_frac, ideal * max_area_frac
    cells = [c for c in cell_list
             if lo < c['area'] < hi and c['aspect'] < 2.6 and c['elong'] < 3.2]
    if not cells:
        return []

    acc, trails = vote(cells, lab_shape, leaf_r, bin_px)
    out = []
    for (px, py, v) in peaks(acc, bin_px, leaf_r):
        members = []
        for c, pts in zip(cells, trails):
            if not pts:
                continue
            # does this leaflet's stalk ray pass close to the peak?
            if min(math.hypot(x - px, y - py) for (x, y) in pts) > leaf_r * 0.55:
                continue
            d = math.hypot(c['x'] - px, c['y'] - py)
            if d > leaf_r * 2.2 or d < leaf_r * 0.25:
                continue
            members.append(c)
        n = len(members)
        if n not in (3, 4):
            continue
        s, parts = score_head(members, px, py, n)
        out.append(dict(n=n, x=px, y=py, score=s, parts=parts, votes=v,
                        members=members,
                        r=max(math.hypot(c['x'] - px, c['y'] - py) + c['r']
                              for c in members)))
    out.sort(key=lambda d: -d['score'])
    return out
