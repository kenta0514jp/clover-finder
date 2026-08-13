#!/usr/bin/env python3
"""Reference port of the detector in index.html.

Same pipeline, same constants, same order of operations — so that a result
here predicts what the web app does on the same picture. Used to tune the
defaults on real photos without a browser.

  python3 tools/detect_ref.py testdata/*.jpg -o out/
"""

import argparse, glob, math, os, sys
from PIL import Image, ImageDraw

WORK_W = 480          # matches the app's still-image path

DEFAULTS = dict(sens=55, size=26, green=55, tex=24)


# ---------------------------------------------------------------- primitives

def box_blur(src, w, h, r):
    """Separable running-sum mean over a (2r+1)^2 window, edge-clamped."""
    n = w * h
    tmp = [0] * n
    out = bytearray(n)
    win = 2 * r + 1
    for y in range(h):
        o = y * w
        s = 0
        for x in range(-r, r + 1):
            s += src[o + min(w - 1, max(0, x))]
        for x in range(w):
            tmp[o + x] = s
            s -= src[o + min(w - 1, max(0, x - r))]
            s += src[o + min(w - 1, max(0, x + r + 1))]
    win2 = win * win
    for x in range(w):
        s = 0
        for y in range(-r, r + 1):
            s += tmp[min(h - 1, max(0, y)) * w + x]
        for y in range(h):
            out[y * w + x] = int(s / win2)
            s -= tmp[min(h - 1, max(0, y - r)) * w + x]
            s += tmp[min(h - 1, max(0, y + r + 1)) * w + x]
    return out


def morph(src, w, h, r, is_max):
    n = w * h
    tmp = bytearray(n)
    out = bytearray(n)
    for y in range(h):
        o = y * w
        for x in range(w):
            v = 0 if is_max else 1
            for k in range(-r, r + 1):
                xx = x + k
                if xx < 0 or xx >= w:
                    if not is_max:
                        v = 0
                        break
                    continue
                s = src[o + xx]
                if is_max:
                    if s:
                        v = 1
                        break
                elif not s:
                    v = 0
                    break
            tmp[o + x] = v
    for x in range(w):
        for y in range(h):
            v = 0 if is_max else 1
            for k in range(-r, r + 1):
                yy = y + k
                if yy < 0 or yy >= h:
                    if not is_max:
                        v = 0
                        break
                    continue
                s = tmp[yy * w + x]
                if is_max:
                    if s:
                        v = 1
                        break
                elif not s:
                    v = 0
                    break
            out[y * w + x] = v
    return out


def components(mask, w, h, min_a, max_a):
    """8-connected flood fill; returns leaflet-candidate blob stats."""
    n = w * h
    seen = bytearray(n)
    blobs = []
    for i in range(n):
        if not mask[i] or seen[i]:
            continue
        st = [i]
        seen[i] = 1
        cnt = sx = sy = 0
        x0, x1, y0, y1 = w, 0, h, 0
        while st:
            p = st.pop()
            py, px = divmod(p, w)
            cnt += 1
            sx += px
            sy += py
            if px < x0: x0 = px
            if px > x1: x1 = px
            if py < y0: y0 = py
            if py > y1: y1 = py
            for dy in (-1, 0, 1):
                yy = py + dy
                if yy < 0 or yy >= h:
                    continue
                base = yy * w
                for dx in (-1, 0, 1):
                    xx = px + dx
                    if xx < 0 or xx >= w:
                        continue
                    q = base + xx
                    if mask[q] and not seen[q]:
                        seen[q] = 1
                        st.append(q)
        if cnt < min_a or cnt > max_a:
            continue
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        blobs.append(dict(
            x=sx / cnt, y=sy / cnt, area=cnt, w=bw, h=bh,
            aspect=max(bw, bh) / max(1, min(bw, bh)),
            fill=cnt / (bw * bh),
            r=math.sqrt(cnt / math.pi),
        ))
        if len(blobs) > 260:
            break
    return blobs


# ---------------------------------------------------------------- detector

def build_mask(rgb, w, h, P):
    n = w * h
    lum = bytearray(n)
    veg = bytearray(n)
    exg_min = P['green'] / 1000
    for i in range(n):
        r, g, b = rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]
        L = (r * 77 + g * 151 + b * 28) >> 8
        lum[i] = L
        exg = (2 * g - r - b) / (r + g + b + 1)
        veg[i] = 1 if (exg > exg_min and 26 < L < 250) else 0

    grad = bytearray(n)
    for y in range(1, h - 1):
        o = y * w
        for x in range(1, w - 1):
            i = o + x
            v = abs(lum[i + 1] - lum[i - 1]) + abs(lum[i + w] - lum[i - w])
            grad[i] = 255 if v > 255 else v
    smooth = box_blur(grad, w, h, 3)

    return bytearray(1 if (veg[i] and smooth[i] < P['tex']) else 0 for i in range(n))


def cv(vals):
    m = sum(vals) / len(vals)
    if m <= 0:
        return 1.0
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(var) / m


def clamp01(v):
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def score_cluster(g, expect):
    cx = sum(l['x'] for l in g) / len(g)
    cy = sum(l['y'] for l in g) / len(g)

    ang = sorted(math.atan2(l['y'] - cy, l['x'] - cx) for l in g)
    gaps = []
    for i, a in enumerate(ang):
        nx = ang[0] + math.tau if i == len(ang) - 1 else ang[i + 1]
        gaps.append((nx - a) * 180 / math.pi)
    ideal = 360 / expect
    ang_err = sum(abs(x - ideal) for x in gaps) / len(gaps)
    s_ang = clamp01(1 - ang_err / (ideal * 0.72))

    dists = [math.hypot(l['x'] - cx, l['y'] - cy) for l in g]
    s_rad = clamp01(1 - cv(dists) / 0.5)
    s_area = clamp01(1 - cv([l['area'] for l in g]) / 0.75)

    mean_d = sum(dists) / len(g)
    mean_r = sum(l['r'] for l in g) / len(g)
    ratio = mean_d / max(0.001, mean_r)
    s_adj = clamp01(1 - abs(ratio - 1.15) / 0.95)

    score = 0.34 * s_ang + 0.21 * s_rad + 0.17 * s_area + 0.28 * s_adj
    return dict(x=cx, y=cy, r=mean_d + mean_r, score=score, n=len(g), leaves=g,
                parts=dict(ang=s_ang, rad=s_rad, area=s_area, adj=s_adj))


def detect(rgb, w, h, P):
    mask0 = build_mask(rgb, w, h, P)
    opened = morph(morph(mask0, w, h, 1, False), w, h, 1, True)

    raw = components(opened, w, h, P['size'], P['size'] * 40)
    leaves = [b for b in raw if b['aspect'] < 2.5 and 0.40 < b['fill'] < 0.97]

    parent = list(range(len(leaves)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            a, b = leaves[i], leaves[j]
            if max(a['area'], b['area']) / min(a['area'], b['area']) > 3.2:
                continue
            if math.hypot(a['x'] - b['x'], a['y'] - b['y']) < 1.35 * (a['r'] + b['r']):
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra

    groups = {}
    for i, l in enumerate(leaves):
        groups.setdefault(find(i), []).append(l)

    thr = 0.35 + (P['sens'] / 100) * 0.40
    quads, trios = [], []
    for g in groups.values():
        if len(g) == 4:
            c = score_cluster(g, 4)
            if c['score'] >= thr:
                quads.append(c)
        elif len(g) == 3:
            c = score_cluster(g, 3)
            if c['score'] >= thr:
                trios.append(c)
    quads.sort(key=lambda c: -c['score'])

    sizes = sorted(len(g) for g in groups.values())
    return dict(quads=quads, trios=trios, leaves=leaves, mask=opened,
                threshold=thr, group_sizes=sizes)


# ---------------------------------------------------------------- rendering

def annotate(img, res, w, h, scale=2, show_mask=True):
    out = img.resize((w * scale, h * scale), Image.LANCZOS).convert('RGB')
    if show_mask:
        m = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        mp = m.load()
        for i, v in enumerate(res['mask']):
            if v:
                mp[i % w, i // w] = (127, 179, 213, 110)
        out = Image.alpha_composite(
            out.convert('RGBA'), m.resize((w * scale, h * scale), Image.NEAREST)
        ).convert('RGB')

    d = ImageDraw.Draw(out)
    for l in res['leaves']:
        d.ellipse([(l['x'] - l['r']) * scale, (l['y'] - l['r']) * scale,
                   (l['x'] + l['r']) * scale, (l['y'] + l['r']) * scale],
                  outline=(201, 211, 199), width=1)
    for t in res['trios']:
        r = t['r'] * scale
        d.ellipse([t['x'] * scale - r, t['y'] * scale - r,
                   t['x'] * scale + r, t['y'] * scale + r],
                  outline=(127, 179, 213), width=2)
    for q in res['quads']:
        r = max(20, q['r'] * scale * 1.25)
        d.ellipse([q['x'] * scale - r, q['y'] * scale - r,
                   q['x'] * scale + r, q['y'] * scale + r],
                  outline=(242, 180, 65), width=4)
        d.text((q['x'] * scale - r, q['y'] * scale - r - 12),
               f"4-leaf {q['score']*100:.0f}%", fill=(242, 180, 65))
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('-o', '--out', default='out')
    ap.add_argument('-w', '--width', type=int, default=WORK_W)
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k, type=int, default=v)
    ap.add_argument('--no-image', action='store_true')
    a = ap.parse_args()

    P = {k: getattr(a, k) for k in DEFAULTS}
    os.makedirs(a.out, exist_ok=True)
    print(f'params {P}  threshold={0.35 + P["sens"]/100*0.40:.2f}  width={a.width}')
    print(f'{"file":30s} {"leaflets":>8s} {"3-leaf":>7s} {"4-LEAF":>7s}  best  group sizes')

    files = [f for pat in a.files for f in sorted(glob.glob(pat))]
    for f in files:
        img = Image.open(f).convert('RGB')
        w = a.width
        h = round(w * img.height / img.width)
        small = img.resize((w, h), Image.LANCZOS)
        rgb = small.tobytes()
        res = detect(rgb, w, h, P)
        best = max((q['score'] for q in res['quads']), default=0)
        sizes = res['group_sizes']
        hist = {n: sizes.count(n) for n in sorted(set(sizes)) if n >= 2}
        print(f'{os.path.basename(f):30s} {len(res["leaves"]):8d} '
              f'{len(res["trios"]):7d} {len(res["quads"]):7d}  '
              f'{best*100:4.0f}%  {hist}')
        if not a.no_image:
            annotate(small, res, w, h).save(
                os.path.join(a.out, os.path.splitext(os.path.basename(f))[0] + '.png'))


if __name__ == '__main__':
    main()
