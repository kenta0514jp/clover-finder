/* YOLO detection in the browser, on top of a vendored onnxruntime-web.
 *
 * The runtime and the weights live in this repo rather than on a CDN so the
 * service worker can cache them — a clover lawn is usually a park with no
 * signal, and the app is meant to keep working there.
 *
 * Model output is [1, 4 + nClasses, nAnchors]: rows 0..3 are cx,cy,w,h in
 * letterboxed pixels, the remaining rows are per-class scores (already
 * sigmoid'd by the export). Decoding is threshold -> per-class NMS -> undo the
 * letterbox.
 */

const Detector = (() => {
  let session = null;
  let inputName = null;
  let imgsz = 640;
  let names = ['clover-3', 'clover-4'];

  /* one reusable canvas for the letterboxed frame */
  const pad = document.createElement('canvas');
  const pctx = pad.getContext('2d', { willReadFrequently: true });
  let chw = null;

  async function load(opts = {}){
    imgsz = opts.imgsz || 640;
    if (opts.names) names = opts.names;

    ort.env.wasm.wasmPaths = 'vendor/ort/';
    /* cross-origin isolation is not set on GitHub Pages, so threads are
       unavailable; asking for them anyway just produces a warning */
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.logLevel = 'error';

    session = await ort.InferenceSession.create(opts.model || 'model.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    inputName = session.inputNames[0];

    pad.width = imgsz; pad.height = imgsz;
    chw = new Float32Array(3 * imgsz * imgsz);
    return { imgsz, names, inputNames: session.inputNames,
             outputNames: session.outputNames };
  }

  function ready(){ return session !== null; }

  /* The graph is exported with dynamic height/width, so the analysis size is a
     runtime choice. Must stay a multiple of 32 — the stride of the deepest
     feature map. */
  function setSize(px){
    px = Math.max(160, Math.round(px / 32) * 32);
    if (px === imgsz) return imgsz;
    imgsz = px;
    pad.width = imgsz; pad.height = imgsz;
    chw = new Float32Array(3 * imgsz * imgsz);
    return imgsz;
  }

  /* Draw the source into a square canvas, letterboxed, and return the mapping
     needed to put boxes back where they belong. */
  function letterbox(drawFn, srcW, srcH){
    const s = Math.min(imgsz / srcW, imgsz / srcH);
    const w = Math.round(srcW * s), h = Math.round(srcH * s);
    const dx = Math.floor((imgsz - w) / 2), dy = Math.floor((imgsz - h) / 2);
    pctx.fillStyle = '#000';
    pctx.fillRect(0, 0, imgsz, imgsz);
    drawFn(pctx, dx, dy, w, h);
    return { s, dx, dy };
  }

  function toTensor(){
    const d = pctx.getImageData(0, 0, imgsz, imgsz).data;
    const n = imgsz * imgsz;
    for (let i = 0, p = 0; i < n; i++, p += 4){
      chw[i]         = d[p]     / 255;
      chw[n + i]     = d[p + 1] / 255;
      chw[2 * n + i] = d[p + 2] / 255;
    }
    return new ort.Tensor('float32', chw, [1, 3, imgsz, imgsz]);
  }

  function iou(a, b){
    const x0 = Math.max(a.x0, b.x0), y0 = Math.max(a.y0, b.y0);
    const x1 = Math.min(a.x1, b.x1), y1 = Math.min(a.y1, b.y1);
    const w = x1 - x0, h = y1 - y0;
    if (w <= 0 || h <= 0) return 0;
    const inter = w * h;
    const areaA = (a.x1 - a.x0) * (a.y1 - a.y0);
    const areaB = (b.x1 - b.x0) * (b.y1 - b.y0);
    return inter / (areaA + areaB - inter);
  }

  function nms(dets, thr){
    dets.sort((p, q) => q.score - p.score);
    const keep = [];
    for (const d of dets){
      if (keep.some(k => k.cls === d.cls && iou(k, d) > thr)) continue;
      keep.push(d);
    }
    return keep;
  }

  /* drawFn(ctx, dx, dy, w, h) must paint the frame into the given rectangle.
     Returns boxes in normalised [0,1] coordinates of the source. */
  async function detect(drawFn, srcW, srcH, opts = {}){
    if (!session) return [];
    const conf = opts.conf ?? 0.25;
    const iouThr = opts.iou ?? 0.45;
    const map = letterbox(drawFn, srcW, srcH);
    const out = await session.run({ [inputName]: toTensor() });
    const t = out[session.outputNames[0]];
    const [, rows, anchors] = t.dims;      // [1, 4+nc, anchors]
    const v = t.data;
    const nc = rows - 4;

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
      dets.push({
        cls: best, score: bestScore,
        x0: cx - w / 2, y0: cy - h / 2, x1: cx + w / 2, y1: cy + h / 2,
      });
    }

    return nms(dets, iouThr).map(d => ({
      cls: d.cls,
      name: names[d.cls] || String(d.cls),
      score: d.score,
      /* undo letterbox, then normalise against the source */
      x0: (d.x0 - map.dx) / map.s / srcW,
      y0: (d.y0 - map.dy) / map.s / srcH,
      x1: (d.x1 - map.dx) / map.s / srcW,
      y1: (d.y1 - map.dy) / map.s / srcH,
    })).map(d => ({
      ...d,
      x: (d.x0 + d.x1) / 2, y: (d.y0 + d.y1) / 2,
      w: d.x1 - d.x0, h: d.y1 - d.y0,
    }));
  }

  return { load, detect, ready, setSize, get imgsz(){ return imgsz; },
           get names(){ return names; } };
})();

if (typeof module !== 'undefined') module.exports = { Detector };
