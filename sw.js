/* Offline cache — a clover lawn is often a park with no signal.
 *
 * Two strategies, because the assets differ in kind:
 *
 *   shell   index.html, detect.js — small, and they change with every fix.
 *           Network first, cache as a fallback. A cache-first shell means a
 *           deployed fix never reaches the phone: the page kept running an old
 *           detect.js against a new index.html for an evening.
 *   heavy   the weights and the WASM runtime — 25 MB, and they change only
 *           when the model does. Cache first; the cache name carries the
 *           version, so bumping CACHE is what ships new weights.
 */
const CACHE = 'clover-scope-v4';   // v4: detector retrained with backgrounds

const SHELL = ['./', './index.html', './manifest.webmanifest', './icon.svg', './detect.js'];
const HEAVY = [
  './model.onnx',
  './vendor/ort/ort.wasm.min.js',
  './vendor/ort/ort-wasm-simd-threaded.mjs',
  './vendor/ort/ort-wasm-simd-threaded.wasm',
];
const isHeavy = url =>
  /\.(onnx|wasm)$/.test(url.pathname) || url.pathname.includes('/vendor/ort/');

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE)
    .then(c => c.addAll(SHELL)
      /* best effort: a flaky connection should still leave a usable, if
         online-only, app rather than failing the whole install */
      .then(() => Promise.all(HEAVY.map(u => c.add(u).catch(() => {})))))
    .then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;

  const keep = res => {
    if (res && res.ok){
      const copy = res.clone();
      caches.open(CACHE).then(c => c.put(e.request, copy)).catch(() => {});
    }
    return res;
  };

  if (isHeavy(url)){
    e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request).then(keep)));
    return;
  }
  e.respondWith(
    fetch(e.request).then(keep)
      .catch(() => caches.match(e.request).then(hit => hit || Response.error()))
  );
});
