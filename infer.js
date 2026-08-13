/* TinyNet forward pass in plain JS — whole-frame, one pass.
 *
 * The network is conv/pool/global-average-pool/linear with NO padding, so the
 * classifier head is a 1x1 convolution and the global pool is a fixed-size
 * average. Feeding a whole frame instead of a 96x96 crop therefore produces a
 * dense grid of window scores at stride 16, and — because the convolutions are
 * valid — each score is bit-for-bit what cropping that window would have given.
 *
 *   96x96 crop          ~26M MACs -> 1 score
 *   256x192 whole frame ~142M MACs -> ~190 scores
 *
 * BatchNorm was folded into the convolutions at export time, so nothing here
 * needs an inference runtime: conv / relu / maxpool / average / linear only.
 */

function decodeF32(b64){
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i=0; i<bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

function buildModel(spec){
  const layers = spec.layers.map(l => ({
    cin: l.cin, cout: l.cout,
    w: decodeF32(l.w),          // [cout][cin][3][3]
    b: decodeF32(l.b)
  }));
  const fc = { cin: spec.fc.cin, cout: spec.fc.cout,
               w: decodeF32(spec.fc.w), b: decodeF32(spec.fc.b) };
  const SIZE = spec.size;
  const FEAT = spec.feat;       // 4: the final map for one training crop
  const CELL = spec.cell;       // 16: input pixels per score-grid step

  /* valid 3x3 convolution: output is 2 smaller in each dimension */
  function conv3(src, cin, wIn, hIn, layer, dst){
    const { cout, w, b } = layer;
    const wOut = wIn - 2, hOut = hIn - 2;
    const nOut = wOut * hOut, nIn = wIn * hIn;
    for (let o=0; o<cout; o++){
      const off = o * nOut;
      dst.fill(b[o], off, off + nOut);
      for (let c=0; c<cin; c++){
        const wbase = (o * cin + c) * 9;
        const sbase = c * nIn;
        for (let ky=0; ky<3; ky++){
          for (let kx=0; kx<3; kx++){
            const k = w[wbase + ky*3 + kx];
            if (k === 0) continue;
            for (let y=0; y<hOut; y++){
              let di = off + y*wOut;
              let si = sbase + (y + ky)*wIn + kx;
              for (let x=0; x<wOut; x++) dst[di++] += k * src[si++];
            }
          }
        }
      }
    }
    return [wOut, hOut];
  }

  function reluPool2(src, ch, wIn, hIn, dst){
    const wOut = wIn >> 1, hOut = hIn >> 1;
    const nIn = wIn * hIn, nOut = wOut * hOut;
    for (let c=0; c<ch; c++){
      const si = c * nIn, di = c * nOut;
      for (let y=0; y<hOut; y++){
        const r0 = si + (2*y)*wIn, r1 = r0 + wIn;
        for (let x=0; x<wOut; x++){
          const a = src[r0 + 2*x], b2 = src[r0 + 2*x + 1];
          const c2 = src[r1 + 2*x], d2 = src[r1 + 2*x + 1];
          let m = a > b2 ? a : b2;
          if (c2 > m) m = c2;
          if (d2 > m) m = d2;
          dst[di + y*wOut + x] = m > 0 ? m : 0;
        }
      }
    }
    return [wOut, hOut];
  }

  /* scratch buffers are sized by input dimensions, so cache them per size */
  const cache = new Map();
  function scratch(w, h){
    const key = w + 'x' + h;
    let s = cache.get(key);
    if (s) return s;
    s = { input: new Float32Array(3*w*h), bufs: [] };
    let cw = w, ch = h;
    for (const l of layers){
      const convN = l.cout * (cw-2) * (ch-2);
      const poolN = l.cout * ((cw-2)>>1) * ((ch-2)>>1);
      s.bufs.push({ conv: new Float32Array(convN), pool: new Float32Array(poolN) });
      cw = (cw-2) >> 1; ch = (ch-2) >> 1;
    }
    s.fw = cw; s.fh = ch;
    cache.set(key, s);
    if (cache.size > 8) cache.delete(cache.keys().next().value);
    return s;
  }

  function features(rgba, w, h){
    const s = scratch(w, h);
    const n = w * h, inp = s.input;
    for (let i=0, p=0; i<n; i++, p+=4){
      inp[i]       = rgba[p]   / 255;
      inp[n + i]   = rgba[p+1] / 255;
      inp[2*n + i] = rgba[p+2] / 255;
    }
    let src = inp, cin = 3, cw = w, ch = h;
    for (let i=0; i<layers.length; i++){
      const l = layers[i], bf = s.bufs[i];
      const [aw, ah] = conv3(src, cin, cw, ch, l, bf.conv);
      const [pw, ph] = reluPool2(bf.conv, l.cout, aw, ah, bf.pool);
      src = bf.pool; cin = l.cout; cw = pw; ch = ph;
    }
    return { feat: src, ch: cin, w: cw, h: ch };
  }

  function head(feat, ch, fw, fh, ox, oy){
    /* average the FEAT x FEAT block whose top-left is (ox, oy) */
    const area = FEAT * FEAT;
    const logits = new Float32Array(fc.cout);
    for (let o=0; o<fc.cout; o++) logits[o] = fc.b[o];
    for (let c=0; c<ch; c++){
      const base = c * fw * fh;
      let s = 0;
      for (let y=0; y<FEAT; y++){
        const row = base + (oy + y)*fw + ox;
        for (let x=0; x<FEAT; x++) s += feat[row + x];
      }
      const v = s / area;
      for (let o=0; o<fc.cout; o++) logits[o] += fc.w[o*fc.cin + c] * v;
    }
    let mx = -Infinity;
    for (const v of logits) if (v > mx) mx = v;
    let sum = 0;
    for (let o=0; o<fc.cout; o++){ logits[o] = Math.exp(logits[o] - mx); sum += logits[o]; }
    for (let o=0; o<fc.cout; o++) logits[o] /= sum;
    return logits;
  }

  /* one 96x96 crop -> class probabilities */
  function predict(rgba){
    const f = features(rgba, SIZE, SIZE);
    return head(f.feat, f.ch, f.w, f.h, 0, 0);
  }

  /* whole frame -> {scores, cols, rows}: P(four) on a stride-CELL grid.
     Grid cell (i,j) is the window with top-left (i*CELL, j*CELL). */
  function predictDense(rgba, w, h, classIdx){
    const f = features(rgba, w, h);
    const cols = f.w - FEAT + 1, rows = f.h - FEAT + 1;
    if (cols <= 0 || rows <= 0) return { scores: new Float32Array(0), cols: 0, rows: 0 };
    const out = new Float32Array(cols * rows);
    for (let j=0; j<rows; j++)
      for (let i=0; i<cols; i++)
        out[j*cols + i] = head(f.feat, f.ch, f.w, f.h, i, j)[classIdx];
    return { scores: out, cols, rows };
  }

  return { predict, predictDense, size: SIZE, feat: FEAT, cell: CELL,
           classes: spec.classes, fourIdx: spec.four_idx };
}

if (typeof module !== 'undefined') module.exports = { buildModel, decodeF32 };
