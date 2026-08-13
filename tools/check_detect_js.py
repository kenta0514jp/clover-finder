#!/usr/bin/env python3
"""Prove detect.js decodes the model exactly as a reference implementation does.

The ONNX graph was already checked against PyTorch tensor-for-tensor. What is
still unverified at that point is the JavaScript around it: the per-class NMS
and the transform from letterboxed pixels back into source coordinates. That is
where a sign error turns into a box on the wrong plant, and comparing tensors
cannot catch it.

Note that ultralytics' own predict() is NOT the reference here — it uses
rectangular inference with grey padding, so its boxes come from a differently
shaped input. The browser draws a square canvas. The comparison below therefore
feeds one identical output tensor to both a numpy decode and to the real
detect.js decode under quickjs.

The second half reports what the browser pipeline actually finds on testdata/,
which is the number the app is entitled to quote.
"""

import os, glob, json, argparse
import numpy as np
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def js_decode_source():
    """detect.js's iou/nms, verbatim, plus the decode loop it drives."""
    src = open(os.path.join(BASE, 'detect.js')).read()
    keep = []
    for fn in ('function iou(', 'function nms('):
        i = src.index(fn)
        k = src.index('{', i)
        depth = 0
        while True:
            if src[k] == '{': depth += 1
            elif src[k] == '}':
                depth -= 1
                if depth == 0: break
            k += 1
        keep.append(src[i:k + 1])
    return '\n'.join(keep) + r'''
function decode(v, anchors, nc, conf, iouThr, map, srcW, srcH){
  const dets = [];
  for (let i = 0; i < anchors; i++){
    let best = -1, bestScore = conf;
    for (let c = 0; c < nc; c++){
      const s = v[(4 + c) * anchors + i];
      if (s > bestScore){ bestScore = s; best = c; }
    }
    if (best < 0) continue;
    const cx = v[i], cy = v[anchors + i];
    const w = v[2 * anchors + i], h = v[3 * anchors + i];
    dets.push({ cls: best, score: bestScore,
                x0: cx - w/2, y0: cy - h/2, x1: cx + w/2, y1: cy + h/2 });
  }
  return nms(dets, iouThr).map(d => ({
    cls: d.cls, score: d.score,
    x0: (d.x0 - map.dx) / map.s / srcW, y0: (d.y0 - map.dy) / map.s / srcH,
    x1: (d.x1 - map.dx) / map.s / srcW, y1: (d.y1 - map.dy) / map.s / srcH,
  }));
}
function run(p){
  p = JSON.parse(p);
  return JSON.stringify(decode(p.v, p.anchors, p.nc, p.conf, p.iou,
                               p.map, p.srcW, p.srcH));
}
'''


def np_decode(out, conf, iou_thr, m, W, H):
    """The same algorithm in numpy — greedy NMS in descending score order."""
    v = out[0]
    nc = v.shape[0] - 4
    scores = v[4:]
    cls = scores.argmax(0)
    best = scores.max(0)
    sel = best > conf
    cx, cy, w, h = v[0][sel], v[1][sel], v[2][sel], v[3][sel]
    boxes = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], 1)
    d = list(zip(best[sel], cls[sel], boxes))
    d.sort(key=lambda t: -t[0])

    keep = []
    for s, c, b in d:
        drop = False
        for ks, kc, kb in keep:
            if kc != c:
                continue
            x0, y0 = max(kb[0], b[0]), max(kb[1], b[1])
            x1, y1 = min(kb[2], b[2]), min(kb[3], b[3])
            iw, ih = x1 - x0, y1 - y0
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            a = (kb[2]-kb[0])*(kb[3]-kb[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
            if inter / a > iou_thr:
                drop = True
                break
        if not drop:
            keep.append((s, c, b))
    return [{'cls': int(c), 'score': float(s),
             'x0': float((b[0]-m['dx'])/m['s']/W), 'y0': float((b[1]-m['dy'])/m['s']/H),
             'x1': float((b[2]-m['dx'])/m['s']/W), 'y1': float((b[3]-m['dy'])/m['s']/H)}
            for s, c, b in keep]


def letterbox(img, size, fill):
    """Exactly what the browser canvas does: fill, then draw scaled + centred."""
    w, h = img.size
    s = min(size / w, size / h)
    nw, nh = round(w * s), round(h * s)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    pad = Image.new('RGB', (size, size), fill)
    pad.paste(img.resize((nw, nh), Image.BILINEAR), (dx, dy))
    return pad, {'s': s, 'dx': dx, 'dy': dy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imgsz', type=int, default=512)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--fill', default='0,0,0',
                    help='letterbox colour the canvas is cleared to')
    ap.add_argument('--out', help='contact sheet of the boxes, as the app draws them')
    a = ap.parse_args()
    fill = tuple(int(v) for v in a.fill.split(','))

    import quickjs, onnxruntime as ort
    ctx = quickjs.Context()
    ctx.eval(js_decode_source())
    run = ctx.get('run')

    sess = ort.InferenceSession(os.path.join(BASE, 'model.onnx'),
                                providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name

    files = sorted(glob.glob(os.path.join(BASE, 'testdata', '*.jpg')))
    print(f'imgsz {a.imgsz}  conf {a.conf}  iou {a.iou}  pad {fill}\n')
    print(f'{"file":28s} {"truth":>6s} {"4":>3s} {"3":>3s} {"best4":>7s}  '
          f'{"js-vs-numpy (px)":>16s}')

    worst = 0.0
    hit = fp = 0
    panels = []
    for f in files:
        img = Image.open(f).convert('RGB')
        W, H = img.size
        pad, m = letterbox(img, a.imgsz, fill)
        x = np.ascontiguousarray(
            np.asarray(pad, np.float32).transpose(2, 0, 1)[None] / 255.0)
        out = sess.run(None, {iname: x})[0]
        _, rows, anchors = out.shape

        js = json.loads(run(json.dumps({
            'v': out[0].reshape(-1).tolist(), 'anchors': anchors, 'nc': rows - 4,
            'conf': a.conf, 'iou': a.iou, 'map': m, 'srcW': W, 'srcH': H})))
        npd = np_decode(out, a.conf, a.iou, m, W, H)

        if len(js) != len(npd):
            err = float('inf')
        else:
            js.sort(key=lambda d: -d['score'])
            npd.sort(key=lambda d: -d['score'])
            err = max([0.0] + [abs(j[k]*(W if 'x' in k else H) - n[k]*(W if 'x' in k else H))
                               for j, n in zip(js, npd) for k in ('x0','y0','x1','y1')])
        worst = max(worst, err)

        name = os.path.basename(f).rsplit('.', 1)[0]
        truth = '4-leaf' if name.startswith('p') else 'none'
        f4 = [d for d in js if d['cls'] == 1]
        f3 = [d for d in js if d['cls'] == 0]
        best4 = max([d['score'] for d in f4], default=0.0)
        if truth == '4-leaf' and f4: hit += 1
        if truth == 'none': fp += len(f4)
        shown = 'COUNT DIFFERS' if err == float('inf') else f'{err:.2e}'
        print(f'{name:28s} {truth:>6s} {len(f4):3d} {len(f3):3d} '
              f'{best4*100:6.1f}%  {shown:>16s}')

        if a.out:
            from PIL import ImageDraw
            vis = img.copy()
            vis.thumbnail((560, 560), Image.LANCZOS)
            d = ImageDraw.Draw(vis)
            for det in sorted(js, key=lambda t: t['score']):
                four = det['cls'] == 1
                col = (242, 180, 65) if four else (127, 179, 213)
                d.rectangle([det['x0']*vis.width, det['y0']*vis.height,
                             det['x1']*vis.width, det['y1']*vis.height],
                            outline=col, width=4 if four else 2)
                if four:
                    d.text((det['x0']*vis.width + 4, det['y0']*vis.height + 4),
                           f"{det['score']*100:.0f}", fill=col)
            d.rectangle([0, 0, vis.width, 18], fill=(10, 14, 11))
            d.text((4, 4), f'{name}  truth={truth}  4-leaf={len(f4)} '
                           f'best={best4*100:.0f}%', fill=(242, 180, 65))
            panels.append(vis)

    poss = sum(1 for f in files if os.path.basename(f).startswith('p'))
    negs = len(files) - poss
    print(f'\nbrowser pipeline: {hit}/{poss} positives found, '
          f'{fp/negs:.1f} false four-leaf boxes per clean frame')
    print(f'js vs numpy worst corner: {worst:.2e} px  '
          f'{"OK" if worst < 1e-3 else "MISMATCH"}')

    if a.out and panels:
        cols = 3
        cw = max(p.width for p in panels)
        rh = max(p.height for p in panels)
        sheet = Image.new('RGB', (cols*cw, ((len(panels)+cols-1)//cols)*rh), (10, 14, 11))
        for i, p in enumerate(panels):
            sheet.paste(p, ((i % cols)*cw, (i // cols)*rh))
        sheet.save(a.out)
        print('saved', a.out)


if __name__ == '__main__':
    main()
