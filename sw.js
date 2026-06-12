const CACHE_NAME = 'predicho-v8';
const DATA_URL   = 'https://raw.githubusercontent.com/hola-johi/wc2026-/main/output/live_predictions.json';

const STATIC_ASSETS = [
  '/wc2026-/',
  '/wc2026-/index.html',
  '/wc2026-/manifest.json',
  '/wc2026-/assets/icono/icon-home-180.png',
  '/wc2026-/assets/icono/icon-home-192.png',
  '/wc2026-/assets/icono/icon-home-512.png',
  '/wc2026-/assets/anim/2026/copa_dorada.png',
  '/wc2026-/assets/anim/2026/mexico.png',
  '/wc2026-/assets/anim/2026/unidos.png',
  '/wc2026-/assets/anim/2026/canada.png',
  '/wc2026-/assets/anim/2026/copa.png',
  '/wc2026-/assets/anim/2026/pelota.png',
  '/wc2026-/assets/anim/2026/logo.png',
  '/wc2026-/assets/sound/final.mp3',
  'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache =>
      cache.addAll(STATIC_ASSETS).catch(err => console.log('Cache parcial:', err))
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  const url = event.request.url;
  const isPage = event.request.mode === 'navigate' || event.request.destination === 'document';

  if (isPage) {
    event.respondWith(
      fetch(event.request, { cache: 'no-store' })
        .then(res => {
          if (res.ok) {
            const clone = res.clone();
            caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
          }
          return res;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Datos live — network first, fallback a cache
  if (url.includes('raw.githubusercontent.com') || url.includes('live_predictions')) {
    event.respondWith(
      fetch(event.request, { cache: 'no-store' })
        .then(res => {
          if (res.ok) {
            const clone = res.clone();
            caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
          }
          return res;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Assets estáticos — cache first, fallback a network
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(res => {
        if (res.ok) {
          const clone = res.clone();
          caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
        }
        return res;
      });
    })
  );
});
