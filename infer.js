/* TinyNet forward pass in plain JS.
 *
 * BatchNorm was folded into the convolutions at export time, so all that is
 * needed here is conv3x3 / relu / maxpool2 / global-average-pool / linear.
 * No inference runtime, nothing fetched — the artifact CSP blocks all of that.
 *
 * ~26M multiply-accumulates per call, which is a few tens of milliseconds on a
 * phone, so buffers are allocated once and reused rather than per frame.
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

  /* scratch space: sizes are fixed by the architecture, so make them once */
  const S = spec.size;
  const bufs = [];
  let side = S;
  for (const l of layers){
    bufs.push({ conv: new Float32Array(l.cout * side * side),
                pool: new Float32Array(l.cout * (side >> 1) * (side >> 1)),
                side });
    side >>= 1;
  }
  const input = new Float32Array(3 * S * S);

  function conv3(src, cin, sideIn, layer, dst){
    const { cout, w, b } = layer;
    const n = sideIn * sideIn;
    for (let o=0; o<cout; o++){
      const off = o * n;
      dst.fill(b[o], off, off + n);
      for (let c=0; c<cin; c++){
        const wbase = (o * cin + c) * 9;
        const sbase = c * n;
        for (let ky=0; ky<3; ky++){
          for (let kx=0; kx<3; kx++){
            const k = w[wbase + ky*3 + kx];
            if (k === 0) continue;
            const dy = ky - 1, dx = kx - 1;
            const y0 = Math.max(0, -dy), y1 = Math.min(sideIn, sideIn - dy);
            const x0 = Math.max(0, -dx), x1 = Math.min(sideIn, sideIn - dx);
            for (let y=y0; y<y1; y++){
              let di = off + y*sideIn + x0;
              let si = sbase + (y + dy)*sideIn + (x0 + dx);
              for (let x=x0; x<x1; x++) dst[di++] += k * src[si++];
            }
          }
        }
      }
    }
  }

  function reluPool2(src, ch, sideIn, dst){
    const half = sideIn >> 1;
    const nIn = sideIn * sideIn, nOut = half * half;
    for (let c=0; c<ch; c++){
      const si = c * nIn, di = c * nOut;
      for (let y=0; y<half; y++){
        const r0 = si + (2*y) * sideIn, r1 = r0 + sideIn;
        for (let x=0; x<half; x++){
          const a = src[r0 + 2*x], b2 = src[r0 + 2*x + 1];
          const c2 = src[r1 + 2*x], d2 = src[r1 + 2*x + 1];
          let m = a > b2 ? a : b2;
          if (c2 > m) m = c2;
          if (d2 > m) m = d2;
          dst[di + y*half + x] = m > 0 ? m : 0;
        }
      }
    }
  }

  /* rgba: Uint8ClampedArray of a size x size RGBA image */
  function predict(rgba){
    const n = S * S;
    for (let i=0, p=0; i<n; i++, p+=4){
      input[i]         = rgba[p]   / 255;
      input[n + i]     = rgba[p+1] / 255;
      input[2*n + i]   = rgba[p+2] / 255;
    }
    let src = input, cin = 3, side = S;
    for (let i=0; i<layers.length; i++){
      const l = layers[i], bf = bufs[i];
      conv3(src, cin, side, l, bf.conv);
      reluPool2(bf.conv, l.cout, side, bf.pool);
      src = bf.pool; cin = l.cout; side >>= 1;
    }
    const nOut = side * side;
    const feat = new Float32Array(cin);
    for (let c=0; c<cin; c++){
      let s = 0;
      for (let i=0; i<nOut; i++) s += src[c*nOut + i];
      feat[c] = s / nOut;
    }
    const logits = new Float32Array(fc.cout);
    for (let o=0; o<fc.cout; o++){
      let s = fc.b[o];
      for (let c=0; c<fc.cin; c++) s += fc.w[o*fc.cin + c] * feat[c];
      logits[o] = s;
    }
    let mx = -Infinity;
    for (const v of logits) if (v > mx) mx = v;
    let sum = 0;
    const pr = new Float32Array(fc.cout);
    for (let o=0; o<fc.cout; o++){ pr[o] = Math.exp(logits[o] - mx); sum += pr[o]; }
    for (let o=0; o<fc.cout; o++) pr[o] /= sum;
    return pr;
  }

  return { predict, size: S, classes: spec.classes, fourIdx: spec.four_idx };
}

if (typeof module !== 'undefined') module.exports = { buildModel, decodeF32 };
