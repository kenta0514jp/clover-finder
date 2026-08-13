#!/usr/bin/env python3
"""Segmentation v2: marker-controlled watershed on the gradient terrain.

v1 thresholded the canopy into "leaflet" blobs and lost most real leaves in the
process. Watershed instead assigns every vegetation pixel to some cell, so a
leaflet cannot vanish — the only question is whether the cell borders land on
real leaf borders.

Everything here is written the way it would be written in JS (bucket queue,
separable filters, union-find) so the prototype ports 1:1.
"""

import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ------------------------------------------------------------------ filters

def box_blur(a, r):
    """Separable mean over (2r+1)^2, edge-clamped. a: float32 2-D."""
    if r < 1:
        return a
    pad = np.pad(a, r, mode='edge')
    c = np.cumsum(pad, axis=1)
    c = np.concatenate([np.zeros((c.shape[0], 1), c.dtype), c], axis=1)
    a = (c[:, 2 * r + 1:] - c[:, :-(2 * r + 1)]) / (2 * r + 1)
    c = np.cumsum(a, axis=0)
    c = np.concatenate([np.zeros((1, c.shape[1]), c.dtype), c], axis=0)
    return (c[2 * r + 1:, :] - c[:-(2 * r + 1), :]) / (2 * r + 1)


def min_filter(a, r):
    if r < 1:
        return a
    pad = np.pad(a, r, mode='edge')
    h = sliding_window_view(pad, (1, 2 * r + 1)).min(axis=-1)[:, :, 0]
    pad2 = h[:, :]
    v = sliding_window_view(pad2, (2 * r + 1, 1)).min(axis=-2)[:, :, 0]
    return v


def sobel_mag(lum):
    p = np.pad(lum, 1, mode='edge')
    gx = np.abs(p[1:-1, 2:] - p[1:-1, :-2])
    gy = np.abs(p[2:, 1:-1] - p[:-2, 1:-1])
    return np.clip(gx + gy, 0, 255)


# ------------------------------------------------------------------ stage 1

def vegetation(rgb, green_thr):
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    lum = (r * 77 + g * 151 + b * 28) / 256.0
    exg = (2 * g - r - b) / (r + g + b + 1)
    veg = (exg > green_thr / 1000) & (lum > 26) & (lum < 250)
    return lum, veg


# ------------------------------------------------------------------ stage 2

def markers(grad, veg, leaf_r):
    """One seed per leaflet: minima of a heavily blurred gradient.

    Blurring at ~leaf_r/3 wipes out the pale chevron band and vein noise that
    would otherwise put several minima inside one leaflet."""
    gs = box_blur(grad.astype(np.float32), max(1, int(leaf_r / 3)))
    sup = max(1, int(leaf_r * 0.6))
    local_min = gs <= min_filter(gs, sup) + 1e-6
    return local_min & veg, gs


def label_seeds(seed_mask):
    """Connected components of the seed mask; one label per component."""
    h, w = seed_mask.shape
    lab = np.zeros((h, w), np.int32)
    nxt = 0
    flat = seed_mask.ravel()
    L = lab.ravel()
    for i in np.flatnonzero(flat):
        if L[i]:
            continue
        nxt += 1
        st = [int(i)]
        L[i] = nxt
        while st:
            p = st.pop()
            py, px = divmod(p, w)
            for dy in (-1, 0, 1):
                yy = py + dy
                if yy < 0 or yy >= h:
                    continue
                for dx in (-1, 0, 1):
                    xx = px + dx
                    if xx < 0 or xx >= w:
                        continue
                    q = yy * w + xx
                    if flat[q] and not L[q]:
                        L[q] = nxt
                        st.append(q)
    return lab, nxt


# ------------------------------------------------------------------ stage 3

def watershed(grad, seeds, veg):
    """Meyer flooding with a 256-bucket priority queue — O(n), no heap."""
    h, w = grad.shape
    g = grad.astype(np.int32).ravel()
    lab = seeds.astype(np.int32).ravel().copy()
    inside = veg.ravel()
    buckets = [[] for _ in range(256)]

    for i in np.flatnonzero(lab):
        i = int(i)
        py, px = divmod(i, w)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = py + dy, px + dx
            if 0 <= yy < h and 0 <= xx < w:
                q = yy * w + xx
                if inside[q] and not lab[q]:
                    buckets[g[q]].append(q)

    level = 0
    while level < 256:
        b = buckets[level]
        if not b:
            level += 1
            continue
        p = b.pop()
        if lab[p]:
            continue
        py, px = divmod(p, w)
        found = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = py + dy, px + dx
            if 0 <= yy < h and 0 <= xx < w:
                q = yy * w + xx
                if lab[q]:
                    found = lab[q]
                    break
        if not found:
            continue
        lab[p] = found
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = py + dy, px + dx
            if 0 <= yy < h and 0 <= xx < w:
                q = yy * w + xx
                if inside[q] and not lab[q]:
                    lv = g[q] if g[q] > level else level
                    buckets[lv].append(q)
                    if lv < level:
                        level = lv
    return lab.reshape(h, w)


# ------------------------------------------------------------------ stage 4

def merge_weak(lab, grad, n_labels, merge_t, min_area):
    """Collapse cells separated by a weak border — watershed always
    over-segments, and the border strength says which splits are real."""
    h, w = lab.shape
    L = lab.ravel()
    g = grad.astype(np.float32).ravel()

    lo_all, hi_all, str_all = [], [], []
    for (dy, dx) in ((0, 1), (1, 0)):
        a = lab[:h - dy, :w - dx].ravel()
        b = lab[dy:, dx:].ravel()
        ga = grad[:h - dy, :w - dx].ravel()
        gb = grad[dy:, dx:].ravel()
        sel = (a != b) & (a > 0) & (b > 0)
        lo_all.append(np.minimum(a[sel], b[sel]))
        hi_all.append(np.maximum(a[sel], b[sel]))
        str_all.append(np.maximum(ga[sel], gb[sel]))
    lo = np.concatenate(lo_all); hi = np.concatenate(hi_all)
    st = np.concatenate(str_all).astype(np.float64)

    key = lo.astype(np.int64) * (n_labels + 1) + hi
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=st)
    cnts = np.bincount(inv)
    edges = {(int(k // (n_labels + 1)), int(k % (n_labels + 1))): [s, c]
             for k, s, c in zip(uk, sums, cnts)}

    area = np.bincount(L, minlength=n_labels + 1)

    parent = list(range(n_labels + 1))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    ranked = sorted(((s / c, c, k) for k, (s, c) in edges.items()), key=lambda t: t[0])
    for strength, count, (x, y) in ranked:
        rx, ry = find(x), find(y)
        if rx == ry:
            continue
        small = min(area[rx], area[ry]) < min_area
        if strength < merge_t or small:
            parent[ry] = rx
            area[rx] += area[ry]

    remap = np.array([find(i) for i in range(n_labels + 1)], np.int32)
    return remap[lab]


# ------------------------------------------------------------------ stage 5

def cells(lab, leaf_r):
    """Per-region stats for every label, in one vectorised pass."""
    h, w = lab.shape
    L = lab.ravel()
    n = int(L.max()) + 1
    ys, xs = np.mgrid[0:h, 0:w]
    ys = ys.ravel(); xs = xs.ravel()

    area = np.bincount(L, minlength=n).astype(np.float64)
    sx = np.bincount(L, weights=xs, minlength=n)
    sy = np.bincount(L, weights=ys, minlength=n)
    x0 = np.full(n, w, np.int64); x1 = np.zeros(n, np.int64)
    y0 = np.full(n, h, np.int64); y1 = np.zeros(n, np.int64)
    np.minimum.at(x0, L, xs); np.maximum.at(x1, L, xs)
    np.minimum.at(y0, L, ys); np.maximum.at(y1, L, ys)

    safe = np.where(area > 0, area, 1)
    cx = sx / safe
    cy = sy / safe

    # second moments -> principal axis of each leaflet
    dx = xs - cx[L]
    dy = ys - cy[L]
    mu20 = np.bincount(L, weights=dx * dx, minlength=n) / safe
    mu02 = np.bincount(L, weights=dy * dy, minlength=n) / safe
    mu11 = np.bincount(L, weights=dx * dy, minlength=n) / safe
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

    # which end of that axis is the narrow one? that end points at the petiole.
    ux, uy = np.cos(theta), np.sin(theta)
    proj = dx * ux[L] + dy * uy[L]          # along the axis
    perp = np.abs(-dx * uy[L] + dy * ux[L])  # distance from the axis
    pos = proj > 0
    w_pos = np.bincount(L, weights=np.where(pos, perp, 0.0), minlength=n)
    n_pos = np.bincount(L, weights=pos.astype(np.float64), minlength=n)
    w_neg = np.bincount(L, weights=np.where(~pos, perp, 0.0), minlength=n)
    n_neg = np.bincount(L, weights=(~pos).astype(np.float64), minlength=n)
    mean_pos = w_pos / np.where(n_pos > 0, n_pos, 1)
    mean_neg = w_neg / np.where(n_neg > 0, n_neg, 1)
    # narrow end = smaller mean half-width; point the vector that way
    sign = np.where(mean_pos < mean_neg, 1.0, -1.0)

    tot = mu20 + mu02
    dif = np.sqrt(np.maximum((mu20 - mu02) ** 2 + 4 * mu11 ** 2, 0))
    lam1 = (tot + dif) / 2
    lam2 = np.maximum((tot - dif) / 2, 1e-6)
    elong = np.sqrt(lam1 / lam2)

    out = []
    for i in range(1, n):
        a = area[i]
        if a <= 0:
            continue
        bw = x1[i] - x0[i] + 1
        bh = y1[i] - y0[i] + 1
        out.append(dict(
            id=i, x=cx[i], y=cy[i], area=int(a),
            aspect=max(bw, bh) / max(1, min(bw, bh)),
            fill=a / (bw * bh),
            r=math.sqrt(a / math.pi),
            elong=float(elong[i]),
            # unit vector from the leaflet centroid toward its own petiole
            px=float(ux[i] * sign[i]), py=float(uy[i] * sign[i]),
        ))
    return out


def segment(rgb, leaf_r, green=55, merge_t=26.0):
    lum, veg = vegetation(rgb, green)
    grad = sobel_mag(lum)
    seed_mask, gs = markers(grad, veg, leaf_r)
    seeds, n = label_seeds(seed_mask)
    lab = watershed(grad, seeds, veg)
    lab = merge_weak(lab, grad, n, merge_t, min_area=math.pi * (leaf_r * 0.45) ** 2)
    return dict(lum=lum, veg=veg, grad=grad, seeds=seeds, n_seeds=n,
                labels=lab, cells=cells(lab, leaf_r))
