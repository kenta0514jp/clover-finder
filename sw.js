/* Offline cache — a clover lawn is often a park with no signal.
   Bumping CACHE invalidates everything from the previous version. */
const CACHE = 'clover-scope-v2';
/* the shell must all arrive or the install fails */
const SHELL = ['./', './index.html', './manifest.webmanifest', './icon.svg', './detect.js'];
/* ~25 MB of runtime and weights — fetched best-effort so a flaky connection
   still leaves a working (if online-only) app */
const HEAVY = [
  './model.onnx',
  './vendor/ort/ort.wasm.min.js',
  './vendor/ort/ort-wasm-simd-threaded.mjs',
  './vendor/ort/ort-wasm-simd-threaded.wasm',
];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE)
    .then(c => c.addAll(SHELL)
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
  e.respondWith(
    caches.match(e.request).then(hit => hit || fetch(e.request).then(res => {
      if (res.ok && new URL(e.request.url).origin === location.origin){
        const copy = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, copy));
      }
      return res;
    }).catch(() => hit))
  );
});
