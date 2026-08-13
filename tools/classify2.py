#!/usr/bin/env python3
"""Group segmented leaflets into clover heads and count them.

A clover's leaflets all attach at one petiole point, so in the watershed
labelling their cell borders converge at a single junction. Find junctions
where N distinct cells meet: N=3 is an ordinary clover, N=4 is the target.
Geometry around the junction then says how clover-like the arrangement is.
"""

import math
import numpy as np


def junctions(lab, leaf_r):
    """Points where 3+ cells meet, with the set of cells meeting there."""
    h, w = lab.shape
    r = max(1, int(leaf_r * 0.30))

    # a pixel is a junction if its (2r+1) neighbourhood holds 3+ distinct labels
    pad = np.pad(lab, r, mode='edge')
    stack = []
    for dy in range(0, 2 * r + 1):
        for dx in range(0, 2 * r + 1):
            stack.append(pad[dy:dy + h, dx:dx + w])
    nb = np.stack(stack, axis=-1)
    nb_sorted = np.sort(nb, axis=-1)
    distinct = (np.diff(nb_sorted, axis=-1) != 0).sum(axis=-1) + 1
    has_zero = (nb_sorted[..., 0] == 0)
    distinct = distinct - has_zero.astype(int)          # background is not a cell
    jmask = (distinct >= 3) & (lab > 0)

    # cluster junction pixels into points
    ys, xs = np.nonzero(jmask)
    if len(ys) == 0:
        return []
    order = np.argsort(-distinct[ys, xs])
    ys, xs = ys[order], xs[order]

    taken = np.zeros(lab.shape, bool)
    sep = max(2, int(leaf_r * 0.8))
    pts = []
    for y, x in zip(ys, xs):
        if taken[max(0, y - sep):y + sep + 1, max(0, x - sep):x + sep + 1].any():
            continue
        taken[y, x] = True
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        labs = np.unique(lab[y0:y1, x0:x1])
        labs = [int(v) for v in labs if v > 0]
        pts.append(dict(x=float(x), y=float(y), labels=labs))
    return pts


def clamp01(v):
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def cv(vals):
    m = sum(vals) / len(vals)
    if m <= 0:
        return 1.0
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) / m


def score_head(members, jx, jy, expect):
    """How much does this set of leaflets look like one clover head?"""
    ang = sorted(math.atan2(c['y'] - jy, c['x'] - jx) for c in members)
    gaps = []
    for i, a in enumerate(ang):
        nxt = ang[0] + math.tau if i == len(ang) - 1 else ang[i + 1]
        gaps.append((nxt - a) * 180 / math.pi)
    ideal = 360 / expect
    s_ang = clamp01(1 - (sum(abs(g - ideal) for g in gaps) / len(gaps)) / (ideal * 0.75))

    dists = [math.hypot(c['x'] - jx, c['y'] - jy) for c in members]
    s_rad = clamp01(1 - cv(dists) / 0.45)
    s_area = clamp01(1 - cv([c['area'] for c in members]) / 0.60)

    mean_d = sum(dists) / len(dists)
    mean_r = sum(c['r'] for c in members) / len(members)
    s_adj = clamp01(1 - abs(mean_d / max(0.001, mean_r) - 1.0) / 0.85)

    # a leaflet is a roundish blob, not a sliver
    s_shape = clamp01(1 - (sum(c['aspect'] for c in members) / len(members) - 1.0) / 1.3)

    # The discriminating cue: every leaflet of one plant points its narrow
    # (petiole) end at the same junction. Leaflets that merely happen to be
    # neighbours point at their own plants instead.
    cosines = []
    for c in members:
        vx, vy = jx - c['x'], jy - c['y']
        n = math.hypot(vx, vy)
        if n < 1e-6:
            cosines.append(0.0)
            continue
        cosines.append((c['px'] * vx + c['py'] * vy) / n)
    mean_cos = sum(cosines) / len(cosines)
    worst_cos = min(cosines)
    s_orient = clamp01((mean_cos - 0.25) / 0.70)
    s_worst = clamp01((worst_cos + 0.10) / 0.70)

    score = (0.20 * s_ang + 0.10 * s_rad + 0.10 * s_area + 0.13 * s_adj
             + 0.07 * s_shape + 0.25 * s_orient + 0.15 * s_worst)
    return score, dict(ang=s_ang, rad=s_rad, area=s_area, adj=s_adj,
                       shape=s_shape, orient=s_orient, worst=s_worst,
                       mean_cos=mean_cos, worst_cos=worst_cos)


def head_shape(lab, members, jx, jy, leaf_r):
    """A real clover head is a compact disc centred on the petiole: rays cast
    from the junction hit member leaflets in every direction. An accidental
    group of neighbouring leaflets leaves wedges of other plants or background."""
    import numpy as np
    ids = {m['id'] for m in members}
    h, w = lab.shape
    reach = max(m['r'] for m in members) + max(
        math.hypot(m['x'] - jx, m['y'] - jy) for m in members)
    reach = max(3.0, min(reach, leaf_r * 3.5))

    n_rays, hits, radii = 72, 0, []
    for k in range(n_rays):
        a = 2 * math.pi * k / n_rays
        dx, dy = math.cos(a), math.sin(a)
        last = 0.0
        t = 1.0
        while t <= reach:
            x, y = int(jx + dx * t), int(jy + dy * t)
            if x < 0 or y < 0 or x >= w or y >= h:
                break
            if int(lab[y, x]) in ids:
                last = t
            elif last > 0 and t > last + max(2.0, leaf_r * 0.25):
                break          # left the head for good
            t += 1.0
        if last > 0:
            hits += 1
            radii.append(last)

    coverage = hits / n_rays
    if not radii:
        return 0.0, 0.0
    m = sum(radii) / len(radii)
    var = sum((r - m) ** 2 for r in radii) / len(radii)
    roundness = clamp01(1 - (math.sqrt(var) / m) / 0.55) if m > 0 else 0.0
    return coverage, roundness


def heads(lab, cell_list, leaf_r, min_area_frac=0.25, max_area_frac=3.5):
    """Return three- and four-leaf candidates found at junctions."""
    by_id = {c['id']: c for c in cell_list}
    ideal_area = math.pi * leaf_r * leaf_r
    lo, hi = ideal_area * min_area_frac, ideal_area * max_area_frac

    out = []
    for j in junctions(lab, leaf_r):
        members = [by_id[i] for i in j['labels'] if i in by_id]
        members = [c for c in members if lo < c['area'] < hi and c['aspect'] < 2.6]
        n = len(members)
        if n not in (3, 4):
            continue
        s, parts = score_head(members, j['x'], j['y'], n)
        cov, rnd = head_shape(lab, members, j['x'], j['y'], leaf_r)
        parts['cover'] = cov
        parts['round'] = rnd
        s = 0.62 * s + 0.24 * clamp01((cov - 0.55) / 0.45) + 0.14 * rnd
        out.append(dict(n=n, x=j['x'], y=j['y'], score=s, parts=parts,
                        members=members,
                        r=max(math.hypot(c['x'] - j['x'], c['y'] - j['y']) + c['r']
                              for c in members)))
    out.sort(key=lambda d: -d['score'])
    return out
