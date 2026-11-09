// TL;DR: Sprecheransicht (index.html?speaker=1, Taste P): Notizen, Uhr gegen Zeitplan, Vorschau des
// aktuellen und nächsten Schritts (Lite-Standbilder), steuert das Hauptfenster per postMessage.
(function () {
  'use strict';
  const params = new URLSearchParams(location.search);
  if (params.get('speaker') !== '1') return;
  document.body.className = 'speaker';
  const { stations } = window.PresStations;
  const flat = [];
  stations.forEach((s, si) => {
    for (let k = 0; k < s.steps; k++) flat.push({ si, k });
  });
  const total = stations.reduce((a, s) => a + s.time, 0);
  const planStart = [];
  stations.reduce((a, s, i) => ((planStart[i] = a), a + s.time), 0);

  document.body.innerHTML = `
    <div class="sp">
      <header>
        <div class="sp-clock"><b id="elapsed">00:00</b><span id="plan"></span></div>
        <div class="sp-ctrl">
          <button id="b-prev">◀ zurück</button><button id="b-next">weiter ▶</button>
          <button id="b-black">B schwarz</button><button id="b-reset">Uhr neu</button>
        </div>
        <div id="black-ind"></div>
      </header>
      <main>
        <section class="sp-now"><h3>Jetzt</h3><div class="sp-title" id="now-title"></div>
          <img id="now-img" alt=""><div class="sp-notes" id="notes"></div></section>
        <section class="sp-next"><h3>Als Nächstes</h3><div class="sp-title" id="next-title"></div>
          <img id="next-img" alt=""><div class="sp-hint">Pfeiltasten steuern das Hauptfenster · Uhr startet beim ersten Weiterschalten</div></section>
      </main>
    </div>`;

  const $ = (id) => document.getElementById(id);
  const titleOf = (si) => {
    const d = document.createElement('div');
    d.innerHTML = stations[si].html(0);
    const h = d.querySelector('h1, h2');
    return h ? h.textContent : stations[si].kicker;
  };
  const shot = (i) => {
    if (i >= flat.length) return '';
    const f = flat[i];
    return `assets/lite/s${String(f.si + 1).padStart(2, '0')}-${f.k}.jpg`;
  };
  const setImg = (el, src) => {
    el.style.visibility = 'hidden';
    if (!src) return;
    el.onload = () => (el.style.visibility = 'visible');
    el.onerror = () => (el.style.visibility = 'hidden');
    el.src = src;
  };
  const mmss = (s) => {
    const neg = s < 0;
    s = Math.abs(Math.round(s));
    return (neg ? '−' : '') + String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
  };

  let cur = 0, t0 = null;
  function show(i, black) {
    if (t0 === null && i > 0) t0 = Date.now();
    cur = i;
    const f = flat[i];
    const st = stations[f.si];
    $('now-title').textContent = `${f.si + 1}/${stations.length} · ${titleOf(f.si)}` + (st.steps > 1 ? ` · Schritt ${f.k + 1}/${st.steps}` : '');
    $('notes').innerHTML = st.notes
      .split('\n')
      .map((p) => `<p>${p.replace(/\[Klick\]/g, '<mark>Klick</mark>')}</p>`)
      .join('');
    const n = flat[i + 1];
    $('next-title').textContent = n ? `${n.si + 1} · ${titleOf(n.si)}` + (n.si === f.si ? ` (Schritt ${n.k + 1})` : '') : 'Ende';
    setImg($('now-img'), shot(i));
    setImg($('next-img'), shot(i + 1));
    $('black-ind').textContent = black ? 'BILDSCHIRM SCHWARZ' : '';
    tick();
  }
  function tick() {
    const el = t0 === null ? 0 : (Date.now() - t0) / 1000;
    $('elapsed').textContent = mmss(el);
    const si = flat[cur].si;
    const plan = planStart[si];
    const diff = el - plan;
    $('plan').textContent = `Plan bei Station ${si + 1}: ${mmss(plan)} von ${mmss(total)} · ${diff > 30 ? 'hinter Plan ' + mmss(diff) : diff < -30 ? 'vor Plan ' + mmss(-diff) : 'im Plan'} · diese Station ${mmss(stations[si].time)}`;
  }
  setInterval(tick, 1000);

  const send = (msg) => window.opener && window.opener.postMessage(msg, '*');
  $('b-prev').onclick = () => send({ type: 'nav', dir: -1 });
  $('b-next').onclick = () => send({ type: 'nav', dir: 1 });
  $('b-black').onclick = () => send({ type: 'black' });
  $('b-reset').onclick = () => {
    t0 = Date.now();
    tick();
  };
  window.addEventListener('keydown', (e) => {
    if (['ArrowRight', 'ArrowDown', 'PageDown', ' '].includes(e.key)) {
      e.preventDefault();
      send({ type: 'nav', dir: 1 });
    } else if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(e.key)) {
      e.preventDefault();
      send({ type: 'nav', dir: -1 });
    } else if (e.key === 'b' || e.key === 'B') send({ type: 'black' });
  });
  window.addEventListener('message', (e) => {
    const d = e.data || {};
    if (d.type === 'state') show(d.i, d.black);
  });
  show(0, false);
  send({ type: 'hello' });
  if (!window.opener) $('now-title').textContent = 'Sprecheransicht bitte aus der Präsentation mit Taste P öffnen.';
})();
