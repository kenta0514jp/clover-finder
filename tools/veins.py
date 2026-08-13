#!/usr/bin/env python3
"""Phase 1: tell three-leaf from four-leaf by the angles between midribs.

The user's observation: the midribs of a three-leaf clover leave the petiole
about 120 degrees apart, a four-leaf about 90. Midribs are crisp line
structures, so their direction is far better defined than the principal axis
of a nearly-round leaflet blob — which is exactly why every orientation-based
attempt in classify2/classify3 failed.

Ridges are found with the Hessian: a line has one large and one small
eigenvalue, and the small one's eigenvector runs along the line.
"""

import math
import numpy as np


# ---------------------------------------------------------------- filters

def gaussian(a, sigma):
    """Separable Gaussian blur, edge-clamped."""
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma * sigma))
    k /= k.sum()
    pad = np.pad(a, ((0, 0), (r, r)), mode='edge')
    out = np.zeros_like(a, dtype=np.float32)
    for i, kv in enumerate(k):
        out += kv * pad[:, i:i + a.shape[1]]
    pad = np.pad(out, ((r, r), (0, 0)), mode='edge')
    res = np.zeros_like(a, dtype=np.float32)
    for i, kv in enumerate(k):
        res += kv * pad[i:i + a.shape[0], :]
    return res


def ridges(lum, sigma, dark_lines=None):
    """Hessian ridge detection.

    Returns (strength, dirx, diry) where (dirx,diry) is the unit vector ALONG
    the line — the eigenvector of the smaller-magnitude eigenvalue.
    """
    g = gaussian(lum.astype(np.float32), sigma)
    # second derivatives by central differences
    gx = np.pad(g, ((0, 0), (1, 1)), mode='edge')
    gy = np.pad(g, ((1, 1), (0, 0)), mode='edge')
    Lxx = gx[:, 2:] - 2 * g + gx[:, :-2]
    Lyy = gy[2:, :] - 2 * g + gy[:-2, :]
    gxy = np.pad(g, 1, mode='edge')
    Lxy = (gxy[2:, 2:] - gxy[2:, :-2] - gxy[:-2, 2:] + gxy[:-2, :-2]) / 4.0

    tr = Lxx + Lyy
    det_part = np.sqrt(np.maximum((Lxx - Lyy) ** 2 + 4 * Lxy ** 2, 0))
    lam1 = (tr + det_part) / 2          # algebraically larger
    lam2 = (tr - det_part) / 2          # algebraically smaller

    # pick the eigenvalue with the larger magnitude as "across the line"
    big_is_1 = np.abs(lam1) >= np.abs(lam2)
    lam_across = np.where(big_is_1, lam1, lam2)
    lam_along = np.where(big_is_1, lam2, lam1)

    strength = np.abs(lam_across) - np.abs(lam_along)
    strength = np.maximum(strength, 0)

    if dark_lines is True:
        strength = np.where(lam_across > 0, strength, 0)   # valley (dark line)
    elif dark_lines is False:
        strength = np.where(lam_across < 0, strength, 0)   # crest (light line)

    # theta1 is the eigenvector direction of lam1 (the algebraically larger).
    # Which of the two runs along the line depends on which one is the big one
    # in magnitude: a bright ridge has a large NEGATIVE curvature across it,
    # so there the across-direction is lam2 and lam1 already runs along.
    theta1 = 0.5 * np.arctan2(2 * Lxy, Lxx - Lyy)
    theta_along = np.where(big_is_1, theta1 + math.pi / 2, theta1)
    return strength, np.cos(theta_along), np.sin(theta_along)


# ---------------------------------------------------------------- voting

def vote_petioles(strength, dirx, diry, mask, leaf_r, bin_px=4, min_frac=0.30):
    """Each ridge pixel votes along its own line, both ways. Midribs of one
    plant converge on the petiole, so that point accumulates."""
    h, w = strength.shape
    thr = strength.max() * min_frac
    ys, xs = np.nonzero((strength > thr) & mask)
    if len(ys) == 0:
        return np.zeros((h // bin_px + 1, w // bin_px + 1), np.float32), (ys, xs)

    bh, bw = h // bin_px + 1, w // bin_px + 1
    acc = np.zeros((bh, bw), np.float32)
    wgt = strength[ys, xs]
    dx, dy = dirx[ys, xs], diry[ys, xs]

    lo, hi = 0.30 * leaf_r, 2.0 * leaf_r
    step = bin_px * 0.5
    t = lo
    while t <= hi:
        for s in (+1.0, -1.0):
            px = xs + dx * t * s
            py = ys + dy * t * s
            ok = (px >= 0) & (py >= 0) & (px < w) & (py < h)
            np.add.at(acc, ((py[ok] // bin_px).astype(int),
                            (px[ok] // bin_px).astype(int)), wgt[ok])
        t += step
    return acc, (ys, xs)


def peaks(acc, bin_px, leaf_r, top_n=40):
    bh, bw = acc.shape
    p = np.pad(acc, 1, mode='constant')
    sm = sum(p[dy:dy + bh, dx:dx + bw] for dy in range(3) for dx in range(3)) / 9.0
    sep = max(1, int(leaf_r * 0.9 / bin_px))
    order = np.argsort(sm.ravel())[::-1]
    taken = np.zeros(acc.shape, bool)
    out = []
    for k in order[:20000]:
        v = float(sm.ravel()[k])
        if v <= 0:
            break
        y, x = divmod(int(k), bw)
        y0, y1 = max(0, y - sep), min(bh, y + sep + 1)
        x0, x1 = max(0, x - sep), min(bw, x + sep + 1)
        if taken[y0:y1, x0:x1].any():
            continue
        taken[y, x] = True
        out.append(((x + 0.5) * bin_px, (y + 0.5) * bin_px, v))
        if len(out) >= top_n:
            break
    return out


# ---------------------------------------------------------------- angles

def azimuth_histogram(strength, dirx, diry, mask, px, py, leaf_r,
                      radial_cos=0.80, nbins=72):
    """Histogram of the directions in which radially-aligned ridges leave the
    candidate petiole. Midribs survive; the pale chevron, which runs across
    the leaflet, does not."""
    h, w = strength.shape
    R = leaf_r * 2.2
    x0, x1 = max(0, int(px - R)), min(w, int(px + R) + 1)
    y0, y1 = max(0, int(py - R)), min(h, int(py + R) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(nbins, np.float32)

    sub_s = strength[y0:y1, x0:x1]
    sub_m = mask[y0:y1, x0:x1]
    thr = strength.max() * 0.30
    yy, xx = np.nonzero((sub_s > thr) & sub_m)
    if len(yy) == 0:
        return np.zeros(nbins, np.float32)

    gx = xx + x0 - px
    gy = yy + y0 - py
    dist = np.hypot(gx, gy)
    keep = (dist > leaf_r * 0.25) & (dist < R)
    if not keep.any():
        return np.zeros(nbins, np.float32)
    gx, gy, yy, xx, dist = gx[keep], gy[keep], yy[keep], xx[keep], dist[keep]

    rx, ry = gx / dist, gy / dist
    dx = dirx[yy + y0, xx + x0]
    dy = diry[yy + y0, xx + x0]
    align = np.abs(rx * dx + ry * dy)            # ridge runs radially?
    sel = align > radial_cos
    if not sel.any():
        return np.zeros(nbins, np.float32)

    ang = np.arctan2(gy[sel], gx[sel]) % (2 * math.pi)
    wgt = sub_s[yy[sel], xx[sel]] * align[sel]
    hist = np.zeros(nbins, np.float32)
    np.add.at(hist, (ang / (2 * math.pi) * nbins).astype(int) % nbins, wgt)

    # circular smoothing so one midrib makes one mode, not a picket fence
    k = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], np.float32)
    k /= k.sum()
    return np.convolve(np.r_[hist, hist, hist], k, 'same')[nbins:2 * nbins]


def count_modes(hist, min_frac=0.35):
    """Peaks of the circular histogram, and how evenly they are spaced."""
    n = len(hist)
    if hist.max() <= 0:
        return [], 0.0
    thr = hist.max() * min_frac
    idx = [i for i in range(n)
           if hist[i] >= thr
           and hist[i] >= hist[(i - 1) % n]
           and hist[i] > hist[(i + 1) % n]]
    if not idx:
        return [], 0.0
    angles = sorted(i * 360.0 / n for i in idx)
    return angles, hist.max()


def spacing_score(angles, expect):
    """1.0 when the modes sit exactly 360/expect apart."""
    if len(angles) != expect:
        return 0.0
    gaps = []
    for i, a in enumerate(angles):
        nxt = angles[0] + 360 if i == len(angles) - 1 else angles[i + 1]
        gaps.append(nxt - a)
    ideal = 360.0 / expect
    err = sum(abs(g - ideal) for g in gaps) / len(gaps)
    return max(0.0, 1.0 - err / (ideal * 0.55))
