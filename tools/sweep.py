#!/usr/bin/env python3
"""Parameter sweep over the reference detector.

The expensive part (vegetation index + gradient) does not depend on the
threshold being swept, so it is computed once per image and reused.
"""

import glob, math, os, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_ref import box_blur, morph, components, score_cluster

W = 480


def precompute(path, w=W):
    img = Image.open(path).convert('RGB')
    h = round(w * img.height / img.width)
    small = img.resize((w, h), Image.LANCZOS)
    rgb = small.tobytes()
    n = w * h
    lum = bytearray(n)
    exg = [0.0] * n
    for i in range(n):
        r, g, b = rgb[i*3], rgb[i*3+1], rgb[i*3+2]
        lum[i] = (r*77 + g*151 + b*28) >> 8
        exg[i] = (2*g - r - b) / (r + g + b + 1)
    grad = bytearray(n)
    for y in range(1, h-1):
        o = y*w
        for x in range(1, w-1):
            i = o+x
            v = abs(lum[i+1]-lum[i-1]) + abs(lum[i+w]-lum[i-w])
            grad[i] = 255 if v > 255 else v
    return dict(path=path, w=w, h=h, img=small, lum=lum, exg=exg, grad=grad,
                blur={})


def masked(pc, green, tex, blur_r):
    if blur_r not in pc['blur']:
        pc['blur'][blur_r] = box_blur(pc['grad'], pc['w'], pc['h'], blur_r)
    sm = pc['blur'][blur_r]
    gmin = green / 1000
    lum, exg = pc['lum'], pc['exg']
    return bytearray(
        1 if (exg[i] > gmin and 26 < lum[i] < 250 and sm[i] < tex) else 0
        for i in range(pc['w'] * pc['h'])
    )


def analyse(pc, green, tex, blur_r, erode_r, min_a, max_a):
    m = masked(pc, green, tex, blur_r)
    op = morph(morph(m, pc['w'], pc['h'], erode_r, False), pc['w'], pc['h'], erode_r, True)
    raw = components(op, pc['w'], pc['h'], min_a, max_a)
    leaves = [b for b in raw if b['aspect'] < 2.5 and 0.40 < b['fill'] < 0.97]

    parent = list(range(len(leaves)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(len(leaves)):
        for j in range(i+1, len(leaves)):
            a, b = leaves[i], leaves[j]
            if max(a['area'], b['area']) / min(a['area'], b['area']) > 3.2:
                continue
            if math.hypot(a['x']-b['x'], a['y']-b['y']) < 1.35*(a['r']+b['r']):
                ra, rb = find(i), find(j)
                if ra != rb: parent[rb] = ra
    groups = {}
    for i, l in enumerate(leaves):
        groups.setdefault(find(i), []).append(l)
    sizes = [len(g) for g in groups.values()]
    quads = [score_cluster(g, 4) for g in groups.values() if len(g) == 4]
    trios = [score_cluster(g, 3) for g in groups.values() if len(g) == 3]
    areas = sorted(l['area'] for l in leaves)
    med = areas[len(areas)//2] if areas else 0
    return dict(leaves=leaves, groups=groups, sizes=sizes, quads=quads,
                trios=trios, median_area=med)


if __name__ == '__main__':
    files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'testdata/*.jpg'))
    pcs = [precompute(f) for f in files]
    print(f'{"file":28s} {"tex":>4s} {"leaf":>5s} {"medA":>6s} {"g3":>3s} {"g4":>3s} '
          f'{"best4":>6s}')
    for tex in (8, 12, 16, 20, 24, 30):
        print('-'*66)
        for pc in pcs:
            r = analyse(pc, green=55, tex=tex, blur_r=3, erode_r=1,
                        min_a=30, max_a=60000)
            b4 = max((q['score'] for q in r['quads']), default=0)
            print(f'{os.path.basename(pc["path"]):28s} {tex:4d} {len(r["leaves"]):5d} '
                  f'{r["median_area"]:6d} {r["sizes"].count(3):3d} {r["sizes"].count(4):3d} '
                  f'{b4*100:5.0f}%')
