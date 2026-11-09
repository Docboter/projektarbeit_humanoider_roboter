// TL;DR: Steuerung der Präsentation: Schritte, Tasten (Pfeile, Leertaste, B, F, P, H), URL-Hash
// (#12, #12.1), Overlay, Lite-Modus (?lite=1), reduzierte Bewegung, Sprecherfenster, Export-Hooks.
(function () {
  'use strict';
  const params = new URLSearchParams(location.search);
  if (params.get('speaker') === '1') return; // Sprecheransicht: siehe speaker.js
  const LITE = params.get('lite') === '1';
  const CAPTURE = params.has('capture');
  const NOOVERLAY = params.has('nooverlay');
  const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches || params.has('still');
  const STILL = REDUCED || CAPTURE;

  const { stations } = window.PresStations;
  const flat = [];
  stations.forEach((s, si) => {
    for (let k = 0; k < s.steps; k++) flat.push({ si, k });
  });

  const $ = (id) => document.getElementById(id);
  const stage = $('stage'), slide = $('slide'), source = $('source'), pageno = $('pageno');
  const bar = $('progress').firstElementChild;
  document.body.classList.toggle('lite', LITE);
  document.body.classList.toggle('reduced', REDUCED);
  document.body.classList.toggle('nooverlay', NOOVERLAY);
  document.body.classList.toggle('capture', CAPTURE);

  let cur = -1, curStation = -1, enteredAt = 0, black = false, worldOk = false;
  let speaker = null;

  // ---------- Skalierung auf 1920×1080 ----------
  function layout() {
    const s = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    stage.style.transform = `translate(${(window.innerWidth - 1920 * s) / 2}px, ${(window.innerHeight - 1080 * s) / 2}px) scale(${s})`;
    if (worldOk) {
      const pr = Math.min(1.5, s * (window.devicePixelRatio || 1));
      PresWorld.resize(Math.max(0.5, pr));
    }
  }
  window.addEventListener('resize', layout);

  // ---------- Navigation ----------
  function parseHash() {
    const m = /^#(\d+)(?:\.(\d+))?/.exec(location.hash);
    if (!m) return 0;
    const si = Math.max(0, Math.min(stations.length - 1, +m[1] - 1));
    const k = Math.min(stations[si].steps - 1, +(m[2] || 0));
    return flat.findIndex((f) => f.si === si && f.k === k);
  }

  function hashOf(i) {
    const f = flat[i];
    return '#' + (f.si + 1) + (f.k ? '.' + f.k : '');
  }

  function go(i, opts) {
    opts = opts || {};
    i = Math.max(0, Math.min(flat.length - 1, i));
    if (i === cur && !opts.force) return;
    const prev = cur;
    cur = i;
    const { si, k } = flat[i];
    const st = stations[si];
    const stationChanged = si !== curStation;
    curStation = si;
    enteredAt = performance.now();
    history.replaceState(null, '', location.pathname + location.search + hashOf(i));

    // Overlay
    const render = () => {
      slide.className = 'slide layout-' + (st.layout || 'default') + ' st-' + st.id;
      slide.innerHTML = `<div class="kicker"><span class="num">${String(si + 1).padStart(2, '0')}</span>${st.kicker}</div>` + st.html(k);
      source.textContent = st.source ? 'Quelle: ' + st.source : '';
      bindInteractive();
      if (stationChanged) {
        slide.classList.add('enter');
        void slide.offsetWidth;
        slide.classList.remove('enter');
      }
    };
    if (stationChanged && prev >= 0 && !CAPTURE) {
      slide.classList.add('leave');
      setTimeout(render, REDUCED ? 120 : 220);
    } else render();
    pageno.textContent = `${si + 1} / ${stations.length}`;
    bar.style.width = ((i + 1) / flat.length) * 100 + '%';

    // 3D
    const state = st.state(k);
    if (LITE) showLite(si, k);
    else if (worldOk) {
      const jump = opts.instant || prev < 0;
      if (REDUCED && !jump && stationChanged) {
        // Überblenden statt Kamerafahrt
        const veil = $('veil');
        veil.classList.add('on');
        setTimeout(() => {
          PresWorld.apply(state, true);
          veil.classList.remove('on');
        }, 260);
      } else PresWorld.apply(state, jump || REDUCED);
    }
    notifySpeaker();
  }

  const next = () => go(cur + 1);
  const prev = () => go(cur - 1);

  // ---------- Lite: statische Renderbilder ----------
  let liteFront = 'liteA';
  function showLite(si, k) {
    const src = `assets/lite/s${String(si + 1).padStart(2, '0')}-${k}.jpg`;
    const back = liteFront === 'liteA' ? 'liteB' : 'liteA';
    const im = $(back);
    im.onload = () => {
      im.classList.add('on');
      $(liteFront).classList.remove('on');
      liteFront = back;
    };
    im.onerror = () => {
      // Kein Standbild vorhanden: beim vorherigen Bild bleiben (oder dunkler Hintergrund)
      if (!$(liteFront).getAttribute('src')) $(liteFront).classList.remove('on');
    };
    im.src = src;
  }

  // ---------- Interaktive Elemente (Vorher/Nachher-Regler) ----------
  function bindInteractive() {
    slide.querySelectorAll('[data-compare] .cmp').forEach((el) => {
      const set = (f) => el.style.setProperty('--p', (Math.max(0, Math.min(1, f)) * 100).toFixed(1) + '%');
      set(0.5);
      let drag = false;
      const at = (e) => {
        const r = el.getBoundingClientRect();
        set((e.clientX - r.left) / r.width);
      };
      el.addEventListener('dragstart', (e) => e.preventDefault());
      el.addEventListener('pointerdown', (e) => {
        e.preventDefault(); // sonst startet das native Bild-Ziehen und bricht die Zeigerereignisse ab
        drag = true;
        el.setPointerCapture(e.pointerId);
        at(e);
      });
      el.addEventListener('pointermove', (e) => drag && at(e));
      el.addEventListener('pointerup', () => (drag = false));
    });
  }

  // ---------- Tasten, Klick ----------
  let typed = '';
  window.addEventListener('keydown', (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const key = e.key;
    if (/^[0-9]$/.test(key)) {
      typed += key;
      return;
    }
    if (key === 'Enter' && typed) {
      const si = Math.max(1, Math.min(stations.length, +typed)) - 1;
      typed = '';
      go(flat.findIndex((f) => f.si === si));
      return;
    }
    typed = '';
    switch (key) {
      case 'ArrowRight': case 'ArrowDown': case 'PageDown': case ' ': case 'Enter':
        e.preventDefault();
        next();
        break;
      case 'ArrowLeft': case 'ArrowUp': case 'PageUp': case 'Backspace':
        e.preventDefault();
        prev();
        break;
      case 'Home':
        go(0);
        break;
      case 'End':
        go(flat.length - 1);
        break;
      case 'b': case 'B': case '.':
        toggleBlack();
        break;
      case 'f': case 'F':
        if (document.fullscreenElement) document.exitFullscreen();
        else document.documentElement.requestFullscreen().catch(() => {});
        break;
      case 'p': case 'P':
        openSpeaker();
        break;
      case 'h': case 'H': case '?':
        $('help').classList.toggle('on');
        break;
      case 'Escape':
        $('help').classList.remove('on');
        break;
    }
  });
  stage.addEventListener('click', (e) => {
    if (e.target.closest('.interactive')) return;
    next();
  });
  stage.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    prev();
  });
  window.addEventListener('hashchange', () => {
    const i = parseHash();
    if (i !== cur) go(i, { instant: true });
  });

  function toggleBlack() {
    black = !black;
    $('black').classList.toggle('on', black);
    notifySpeaker();
  }

  // ---------- Sprecherfenster ----------
  function openSpeaker() {
    const url = location.pathname + '?speaker=1' + hashOf(cur);
    speaker = window.open(url, 'groot-sprecher', 'width=1280,height=800');
    setTimeout(notifySpeaker, 800);
  }
  function notifySpeaker() {
    if (speaker && !speaker.closed) {
      try {
        speaker.postMessage({ type: 'state', i: cur, black }, '*');
      } catch (e) { /* Fenster geschlossen */ }
    }
  }
  window.addEventListener('message', (e) => {
    const d = e.data || {};
    if (d.type === 'hello') {
      speaker = e.source;
      notifySpeaker();
    } else if (d.type === 'nav') {
      if (d.dir === 1) next();
      else if (d.dir === -1) prev();
      else if (typeof d.i === 'number') go(d.i);
    } else if (d.type === 'black') toggleBlack();
  });

  // ---------- Export-Hooks (tools/export_pdf.mjs) ----------
  window.__pres = {
    count: flat.length,
    flat,
    go: (i) => go(i, { force: true }),
    settled: () => !worldOk || PresWorld.isSettled(),
    ready: false,
    // Feste Simulationsuhr für reproduzierbare Standbilder (nur ?capture=1)
    advance: (sec) => {
      if (!worldOk) return;
      const n = Math.round(sec * 30);
      for (let j = 0; j < n; j++) PresWorld.update(1 / 30, j < n - 1);
    },
    captureTime: () => stations[flat[cur].si].capture || 3.0,
  };

  // ---------- Start ----------
  async function start() {
    layout();
    if (!LITE) {
      try {
        if (document.fonts) {
          await Promise.all([document.fonts.load('600 96px Inter'), document.fonts.load('400 60px Inter')]);
        }
        PresWorld.init($('gl'), { still: STILL });
        worldOk = true;
        layout();
      } catch (err) {
        console.error(err);
        document.body.classList.add('lite', 'nogl');
      }
    }
    go(parseHash(), { instant: true });
    window.__pres.ready = true;

    if (worldOk && !CAPTURE) {
      let last = performance.now(), frames = 0, acc = 0;
      const fpsEl = params.has('fps') ? $('fps') : null;
      if (fpsEl) fpsEl.classList.add('on');
      const loop = (now) => {
        requestAnimationFrame(loop);
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;
        if (black || document.hidden) return;
        PresWorld.update(dt);
        if (fpsEl) {
          frames++;
          acc += dt;
          if (acc >= 1) {
            fpsEl.textContent = Math.round(frames / acc) + ' fps';
            frames = 0;
            acc = 0;
          }
        }
      };
      requestAnimationFrame(loop);
    }
  }
  start();
})();
