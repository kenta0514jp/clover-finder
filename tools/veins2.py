#!/usr/bin/env python3
"""Phase 1b: resolve the 90-degree ambiguity in the vein axis.

Measured on known heads, the dominant vein direction inside a leaflet agrees
with the direction to the petiole within about 6 degrees — the signal the user
predicted is real and precise. But on some leaflets the pale chevron band,
which runs ACROSS the leaflet, wins the vote and the measured axis flips by 90.

Each leaflet therefore gets its top two candidate directions (they come out
roughly perpendicular). For a head of 3 or 4 leaflets that is at most 16
combinations, so we simply try them all and keep the one whose lines converge
best. The residual of that fit is itself the evidence that these leaflets
belong to one plant rather than being neighbours that happen to touch.
"""

import itertools
import math
import numpy as np

from veins import ridges


def cell_axis_modes(lum, veg, labels, cells, leaf_r, sigma, nbins=36, top=2):
    """Top-N orientation modes per cell, as (angle_rad, weight)."""
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
    th = (np.arctan2(dy, dx) % math.pi)
    bin_idx = (th / math.pi * nbins).astype(np.int32) % nbins

    K = int(labels.max()) + 1
    L = labels.ravel(); m = ok.ravel()
    keys = L[m].astype(np.int64) * nbins + bin_idx.ravel()[m]
    wts = s.ravel()[m]
    hist = np.bincount(keys, weights=wts, minlength=K * nbins).reshape(K, nbins)

    kern = np.array([1, 2, 3, 2, 1], np.float32); kern /= kern.sum()
    out = {}
    for c in cells:
        i = c['id']
        row = hist[i]
        if row.sum() <= 0:
            continue
        sm = np.convolve(np.r_[row, row, row], kern, 'same')[nbins:2 * nbins]
        peaks = [(sm[b], b) for b in range(nbins)
                 if sm[b] >= sm[(b - 1) % nbins] and sm[b] > sm[(b + 1) % nbins]]
        if not peaks:
            continue
        peaks.sort(reverse=True)
        tot = sm.sum()
        out[i] = [((b + 0.5) * math.pi / nbins, float(v / tot))
                  for v, b in peaks[:top]]
    return out


def fit_intersection(points, dirs):
    """Least-squares point closest to a set of infinite lines."""
    A = np.zeros((2, 2)); b = np.zeros(2)
    for p, u in zip(points, dirs):
        P = np.eye(2) - np.outer(u, u)
        A += P; b += P @ p
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, 1e9
    res = 0.0
    for p, u in zip(points, dirs):
        d = (x - p) - np.dot(x - p, u) * u
        res += float(d @ d)
    return x, math.sqrt(res / len(points))


def solve_head(members, modes, leaf_r):
    """Pick one direction per leaflet so the axes converge; return the fit."""
    choices = []
    for c in members:
        ms = modes.get(c['id'])
        if not ms:
            return None
        choices.append(ms)

    pts = [np.array([c['x'], c['y']]) for c in members]
    best = None
    for combo in itertools.product(*[range(len(m)) for m in choices]):
        dirs = []
        wsum = 0.0
        for ci, c in zip(combo, choices):
            a, w = c[ci]
            dirs.append(np.array([math.cos(a), math.sin(a)]))
            wsum += w
        x, res = fit_intersection(pts, dirs)
        if x is None:
            continue
        # the petiole must sit inside the head, not out at infinity
        d = [float(np.hypot(*(p - x))) for p in pts]
        if max(d) > leaf_r * 3.0 or min(d) < leaf_r * 0.15:
            continue
        norm = res / leaf_r
        # Letting every leaflet flip freely gives 16 degrees of freedom, and
        # then even unrelated neighbours converge. Flipping is the exception
        # (a chevron winning the vote), so charge for it.
        flips = sum(1 for ci in combo if ci > 0) / len(combo)
        cost = norm + 0.55 * flips - 0.15 * (wsum / len(choices))
        if best is None or cost < best['cost']:
            best = dict(cost=cost, x=x, resid=norm, dists=d, flips=flips,
                        dirs=dirs, weight=wsum / len(choices))
    return best


def head_score(members, fit, leaf_r, expect):
    """How much does this converged group look like one clover head?"""
    x = fit['x']
    ang = sorted(math.degrees(math.atan2(c['y'] - x[1], c['x'] - x[0])) % 360
                 for c in members)
    gaps = []
    for i, a in enumerate(ang):
        nxt = ang[0] + 360 if i == len(ang) - 1 else ang[i + 1]
        gaps.append(nxt - a)
    ideal = 360.0 / expect
    ang_err = sum(abs(g - ideal) for g in gaps) / len(gaps)
    s_ang = max(0.0, 1.0 - ang_err / (ideal * 0.55))

    # the axes converged — that is the evidence they share a stalk
    s_res = max(0.0, 1.0 - fit['resid'] / 0.55)

    d = fit['dists']
    s_rad = max(0.0, 1.0 - (np.std(d) / max(np.mean(d), 1e-6)) / 0.40)
    areas = [c['area'] for c in members]
    s_area = max(0.0, 1.0 - (np.std(areas) / max(np.mean(areas), 1e-6)) / 0.55)

    score = 0.30 * s_res + 0.30 * s_ang + 0.20 * s_rad + 0.20 * s_area
    return score, dict(res=s_res, ang=s_ang, rad=s_rad, area=s_area,
                       resid=fit['resid'], gaps=[round(g) for g in gaps])
