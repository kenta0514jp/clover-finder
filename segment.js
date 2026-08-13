/* Leaflet segmentation — the JS port of tools/seg2.py.
 *
 * Marker-controlled watershed on the gradient terrain. Every vegetation pixel
 * ends up in some cell, so a leaflet cannot vanish the way it did when we
 * thresholded the canopy. Validated against testdata/ before porting.
 *
 * segment(imageData, w, h, {leafR, green, mergeT}) -> {labels, nLabels, cells, veg}
 */

/* ---------- separable primitives ---------- */

function boxBlurF(src, w, h, r){
  const n = w*h, tmp = new Float32Array(n), out = new Float32Array(n);
  const win = 2*r + 1;
  for (let y=0; y<h; y++){
    const o = y*w;
    let sum = 0;
    for (let k=-r; k<=r; k++) sum += src[o + Math.min(w-1, Math.max(0, k))];
    for (let x=0; x<w; x++){
      tmp[o+x] = sum / win;
      sum -= src[o + Math.min(w-1, Math.max(0, x-r))];
      sum += src[o + Math.min(w-1, Math.max(0, x+r+1))];
    }
  }
  for (let x=0; x<w; x++){
    let sum = 0;
    for (let k=-r; k<=r; k++) sum += tmp[Math.min(h-1, Math.max(0, k))*w + x];
    for (let y=0; y<h; y++){
      out[y*w+x] = sum / win;
      sum -= tmp[Math.min(h-1, Math.max(0, y-r))*w + x];
      sum += tmp[Math.min(h-1, Math.max(0, y+r+1))*w + x];
    }
  }
  return out;
}

/* sliding-window minimum via a monotonic deque — O(n) whatever the radius */
function minFilterF(src, w, h, r){
  const n = w*h, tmp = new Float32Array(n), out = new Float32Array(n);
  const win = 2*r + 1;
  const dq = new Int32Array(Math.max(w, h) + win);

  for (let y=0; y<h; y++){
    const o = y*w;
    let head = 0, tail = 0;
    for (let x=0; x<w + r; x++){
      const cur = Math.min(w-1, x);
      while (tail > head && src[o + dq[tail-1]] >= src[o + cur]) tail--;
      dq[tail++] = cur;
      const outX = x - r;
      if (outX >= 0){
        while (dq[head] < outX - r) head++;
        tmp[o + outX] = src[o + dq[head]];
      }
    }
  }
  for (let x=0; x<w; x++){
    let head = 0, tail = 0;
    for (let y=0; y<h + r; y++){
      const cur = Math.min(h-1, y);
      while (tail > head && tmp[dq[tail-1]*w + x] >= tmp[cur*w + x]) tail--;
      dq[tail++] = cur;
      const outY = y - r;
      if (outY >= 0){
        while (dq[head] < outY - r) head++;
        out[outY*w + x] = tmp[dq[head]*w + x];
      }
    }
  }
  return out;
}

/* ---------- stage 1: what is a plant ---------- */

function vegetation(data, w, h, greenThr){
  const n = w*h;
  const lum = new Float32Array(n);
  const veg = new Uint8Array(n);
  const thr = greenThr / 1000;
  for (let i=0, p=0; i<n; i++, p+=4){
    const r = data[p], g = data[p+1], b = data[p+2];
    const L = (r*77 + g*151 + b*28) / 256;
    lum[i] = L;
    const exg = (2*g - r - b) / (r + g + b + 1);
    veg[i] = (exg > thr && L > 26 && L < 250) ? 1 : 0;
  }
  return { lum, veg };
}

function gradient(lum, w, h){
  const n = w*h, g = new Float32Array(n);
  for (let y=0; y<h; y++){
    const o = y*w;
    const yUp = (y > 0 ? y-1 : 0)*w, yDn = (y < h-1 ? y+1 : h-1)*w;
    for (let x=0; x<w; x++){
      const xl = x > 0 ? x-1 : 0, xr = x < w-1 ? x+1 : w-1;
      const v = Math.abs(lum[o+xr] - lum[o+xl]) + Math.abs(lum[yDn+x] - lum[yUp+x]);
      g[o+x] = v > 255 ? 255 : v;
    }
  }
  return g;
}

/* ---------- stage 2: one seed per leaflet ---------- */

function seedLabels(grad, veg, w, h, leafR){
  const n = w*h;
  const gs = boxBlurF(grad, w, h, Math.max(1, Math.round(leafR/3)));
  const mn = minFilterF(gs, w, h, Math.max(1, Math.round(leafR*0.6)));

  const isSeed = new Uint8Array(n);
  for (let i=0; i<n; i++) isSeed[i] = (veg[i] && gs[i] <= mn[i] + 1e-6) ? 1 : 0;

  /* one label per connected clump of minima */
  const lab = new Int32Array(n);
  const stack = new Int32Array(n);
  let next = 0;
  for (let i=0; i<n; i++){
    if (!isSeed[i] || lab[i]) continue;
    next++;
    let sp = 0; stack[sp++] = i; lab[i] = next;
    while (sp > 0){
      const p = stack[--sp];
      const py = (p/w)|0, px = p - py*w;
      for (let dy=-1; dy<=1; dy++){
        const yy = py+dy; if (yy < 0 || yy >= h) continue;
        for (let dx=-1; dx<=1; dx++){
          const xx = px+dx; if (xx < 0 || xx >= w) continue;
          const q = yy*w + xx;
          if (isSeed[q] && !lab[q]){ lab[q] = next; stack[sp++] = q; }
        }
      }
    }
  }
  return { seeds: lab, nSeeds: next };
}

/* ---------- stage 3: flood the terrain ---------- */

function watershed(grad, seeds, veg, w, h){
  const n = w*h;
  const lab = Int32Array.from(seeds);
  const buckets = new Array(256);
  for (let i=0; i<256; i++) buckets[i] = [];

  const push = (p) => {
    const py = (p/w)|0, px = p - py*w;
    for (let k=0; k<4; k++){
      const yy = py + (k === 0 ? -1 : k === 1 ? 1 : 0);
      const xx = px + (k === 2 ? -1 : k === 3 ? 1 : 0);
      if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
      const q = yy*w + xx;
      if (veg[q] && !lab[q]) buckets[grad[q]|0].push(q);
    }
  };
  for (let i=0; i<n; i++) if (lab[i]) push(i);

  let level = 0;
  while (level < 256){
    const b = buckets[level];
    if (b.length === 0){ level++; continue; }
    const p = b.pop();
    if (lab[p]) continue;
    const py = (p/w)|0, px = p - py*w;

    let found = 0;
    for (let k=0; k<4 && !found; k++){
      const yy = py + (k === 0 ? -1 : k === 1 ? 1 : 0);
      const xx = px + (k === 2 ? -1 : k === 3 ? 1 : 0);
      if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
      const q = yy*w + xx;
      if (lab[q]) found = lab[q];
    }
    if (!found) continue;
    lab[p] = found;

    for (let k=0; k<4; k++){
      const yy = py + (k === 0 ? -1 : k === 1 ? 1 : 0);
      const xx = px + (k === 2 ? -1 : k === 3 ? 1 : 0);
      if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
      const q = yy*w + xx;
      if (veg[q] && !lab[q]){
        const g = grad[q]|0;
        const lv = g > level ? g : level;
        buckets[lv].push(q);
        if (lv < level) level = lv;
      }
    }
  }
  return lab;
}

/* ---------- stage 4: undo the over-segmentation ---------- */

function mergeWeak(lab, grad, nLab, w, h, mergeT, minArea){
  const n = w*h, K = nLab + 1;
  const sums = new Map(), counts = new Map();
  const add = (a, b, s) => {
    const lo = a < b ? a : b, hi = a < b ? b : a;
    const key = lo*K + hi;
    sums.set(key, (sums.get(key) || 0) + s);
    counts.set(key, (counts.get(key) || 0) + 1);
  };
  for (let y=0; y<h; y++){
    for (let x=0; x<w; x++){
      const i = y*w + x, a = lab[i];
      if (!a) continue;
      if (x+1 < w){ const b = lab[i+1]; if (b && b !== a) add(a, b, Math.max(grad[i], grad[i+1])); }
      if (y+1 < h){ const b = lab[i+w]; if (b && b !== a) add(a, b, Math.max(grad[i], grad[i+w])); }
    }
  }

  const area = new Int32Array(K);
  for (let i=0; i<n; i++) area[lab[i]]++;

  const parent = new Int32Array(K);
  for (let i=0; i<K; i++) parent[i] = i;
  const find = (a) => { while (parent[a] !== a){ parent[a] = parent[parent[a]]; a = parent[a]; } return a; };

  const list = [];
  for (const [key, s] of sums) list.push([s / counts.get(key), key]);
  list.sort((p, q) => p[0] - q[0]);

  for (const [strength, key] of list){
    const a = find(Math.floor(key / K)), b = find(key % K);
    if (a === b) continue;
    if (strength < mergeT || Math.min(area[a], area[b]) < minArea){
      parent[b] = a;
      area[a] += area[b];
    }
  }

  const remap = new Int32Array(K);
  let next = 0;
  for (let i=1; i<K; i++){
    const r = find(i);
    if (remap[r] === 0){ next++; remap[r] = next; }
    remap[i] = remap[r];
  }
  const out = new Int32Array(n);
  for (let i=0; i<n; i++) out[i] = lab[i] ? remap[lab[i]] : 0;
  return { labels: out, nLabels: next };
}

/* ---------- stage 5: describe each cell ---------- */

function cellStats(lab, nLab, w, h){
  const K = nLab + 1;
  const area = new Float64Array(K), sx = new Float64Array(K), sy = new Float64Array(K);
  const x0 = new Int32Array(K).fill(w), x1 = new Int32Array(K);
  const y0 = new Int32Array(K).fill(h), y1 = new Int32Array(K);

  for (let y=0; y<h; y++){
    for (let x=0; x<w; x++){
      const l = lab[y*w+x];
      if (!l) continue;
      area[l]++; sx[l] += x; sy[l] += y;
      if (x < x0[l]) x0[l] = x; if (x > x1[l]) x1[l] = x;
      if (y < y0[l]) y0[l] = y; if (y > y1[l]) y1[l] = y;
    }
  }
  const cx = new Float64Array(K), cy = new Float64Array(K);
  for (let l=1; l<K; l++) if (area[l] > 0){ cx[l] = sx[l]/area[l]; cy[l] = sy[l]/area[l]; }

  const m20 = new Float64Array(K), m02 = new Float64Array(K), m11 = new Float64Array(K);
  for (let y=0; y<h; y++){
    for (let x=0; x<w; x++){
      const l = lab[y*w+x];
      if (!l) continue;
      const dx = x - cx[l], dy = y - cy[l];
      m20[l] += dx*dx; m02[l] += dy*dy; m11[l] += dx*dy;
    }
  }

  const cells = [];
  const byId = new Array(K).fill(null);
  for (let l=1; l<K; l++){
    const a = area[l];
    if (a <= 0) continue;
    const u20 = m20[l]/a, u02 = m02[l]/a, u11 = m11[l]/a;
    const tot = u20 + u02;
    const dif = Math.sqrt(Math.max((u20-u02)*(u20-u02) + 4*u11*u11, 0));
    const lam1 = (tot + dif)/2, lam2 = Math.max((tot - dif)/2, 1e-6);
    const bw = x1[l]-x0[l]+1, bh = y1[l]-y0[l]+1;
    const c = {
      id: l, x: cx[l], y: cy[l], area: a,
      w: bw, h: bh,
      aspect: Math.max(bw, bh) / Math.max(1, Math.min(bw, bh)),
      fill: a / (bw*bh),
      r: Math.sqrt(a / Math.PI),
      elong: Math.sqrt(lam1 / lam2)
    };
    cells.push(c);
    byId[l] = c;
  }
  return { cells, byId };
}

/* ---------- the pipeline ---------- */

function segment(imageData, w, h, opts){
  const leafR = opts.leafR;
  const { lum, veg } = vegetation(imageData.data, w, h, opts.green);
  const grad = gradient(lum, w, h);
  const { seeds, nSeeds } = seedLabels(grad, veg, w, h, leafR);
  const flooded = watershed(grad, seeds, veg, w, h);
  const minArea = Math.PI * Math.pow(leafR * 0.45, 2);
  const { labels, nLabels } = mergeWeak(flooded, grad, nSeeds, w, h, opts.mergeT, minArea);
  const { cells, byId } = cellStats(labels, nLabels, w, h);
  return { labels, nLabels, cells, byId, veg, grad, nSeeds };
}

/* ---------- tap to count ----------
 * The human points at the stalk; we report which leaflets meet there. Finding
 * the head is the part people do instantly and the machine kept getting wrong.
 */
function countAt(seg, w, h, tx, ty, leafR){
  const { labels, byId } = seg;
  const rad = Math.max(3, leafR * 0.75);
  const seen = new Map();
  const r2 = rad*rad;
  const x0 = Math.max(0, Math.floor(tx - rad)), x1 = Math.min(w-1, Math.ceil(tx + rad));
  const y0 = Math.max(0, Math.floor(ty - rad)), y1 = Math.min(h-1, Math.ceil(ty + rad));
  for (let y=y0; y<=y1; y++){
    for (let x=x0; x<=x1; x++){
      const dx = x - tx, dy = y - ty;
      if (dx*dx + dy*dy > r2) continue;
      const l = labels[y*w + x];
      if (!l) continue;
      seen.set(l, (seen.get(l) || 0) + 1);
    }
  }

  /* a leaflet must actually occupy the neighbourhood, not just graze it */
  const touch = Math.max(4, Math.round(Math.PI * rad * rad * 0.02));
  let members = [];
  for (const [l, cnt] of seen){
    const c = byId[l];
    if (c && cnt >= touch) members.push(c);
  }

  /* leaflets of one plant are comparable in size; drop the odd one out */
  if (members.length > 1){
    const areas = members.map(c => c.area).sort((a, b) => a - b);
    const med = areas[Math.floor(areas.length/2)];
    members = members.filter(c => c.area > med*0.30 && c.area < med*3.2);
  }
  members.sort((a, b) => b.area - a.area);
  return members;
}

if (typeof module !== 'undefined') module.exports = { segment, countAt };
