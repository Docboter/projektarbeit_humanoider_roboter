// TL;DR: SVG-Diagramme der Präsentation, direkt aus den eingebetteten CSVs (assets/data.js):
// Val-MSE Lauf 3/4, Versatz-Sweep Lauf 62, visuelles Gate (MSE Hände). Real = Bernstein, Sim = Cyan.
(function () {
  'use strict';
  const AMBER = '#f5a524', AMBER2 = '#ffd79a', CYAN = '#22d3ee', INK = '#e8edf2', DIM = '#8a97a6', GRID = '#243041';

  function csv(name) {
    const txt = (window.PRES_CSV || {})[name] || '';
    const rows = [];
    let header = null;
    for (const line of txt.trim().split('\n')) {
      const cells = [];
      let cur = '', q = false;
      for (const ch of line) {
        if (ch === '"') q = !q;
        else if (ch === ',' && !q) {
          cells.push(cur);
          cur = '';
        } else cur += ch;
      }
      cells.push(cur);
      if (!header) header = cells;
      else rows.push(Object.fromEntries(header.map((h, i) => [h, cells[i]])));
    }
    return rows;
  }

  const de = (x, d) => x.toFixed(d).replace('.', ',');
  const thou = (x) => String(x).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
  function text(x, y, s, o) {
    o = o || {};
    return `<text x="${x}" y="${y}" fill="${o.fill || INK}" font-size="${o.size || 20}" font-weight="${o.weight || 400}" text-anchor="${o.anchor || 'start'}"${o.rot ? ` transform="rotate(${o.rot} ${x} ${y})"` : ''}>${esc(s)}</text>`;
  }

  // Station 5: Validierungs-MSE je Checkpoint (20 Testepisoden × 750 Schritte)
  function valMse() {
    const rows = csv('val_mse_lauf3_lauf4.csv');
    const W = 880, H = 500, L = 110, R = 40, Tp = 30, B = 80;
    const x = (s) => L + (s / 45000) * (W - L - R);
    const y = (v) => Tp + (1 - (v - 0.004) / (0.0105 - 0.004)) * (H - Tp - B);
    let s = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="Validierungs-MSE Lauf 3 und Lauf 4">`;
    for (const v of [0.004, 0.006, 0.008, 0.01]) {
      s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="${GRID}"/>`;
      s += text(L - 12, y(v) + 7, de(v, 3), { anchor: 'end', fill: DIM, size: 19 });
    }
    for (const t of [0, 10000, 20000, 30000, 40000]) {
      s += `<line x1="${x(t)}" x2="${x(t)}" y1="${H - B}" y2="${H - B + 8}" stroke="${DIM}"/>`;
      s += text(x(t), H - B + 32, thou(t), { anchor: 'middle', fill: DIM, size: 19 });
    }
    s += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="${DIM}"/>`;
    s += text((L + W - R) / 2, H - 14, 'Trainingsschritt', { anchor: 'middle', size: 21 });
    s += text(26, (Tp + H - B) / 2, 'Validierungs-MSE', { anchor: 'middle', size: 21, rot: -90 });
    const series = [
      ['mse_lauf3_n16', AMBER, '', 'N1.6 (Lauf 3)'],
      ['mse_lauf4_n17', AMBER2, '8 7', 'N1.7 (Lauf 4)'],
    ];
    series.forEach(([k, col, dash, name], si) => {
      const pts = rows.filter((r) => r[k]).map((r) => [+r.schritt, +r[k]]);
      s += `<polyline fill="none" stroke="${col}" stroke-width="4" ${dash ? `stroke-dasharray="${dash}"` : ''} points="${pts.map((p) => x(p[0]) + ',' + y(p[1])).join(' ')}"/>`;
      for (const p of pts) s += `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="6" fill="${col}"/>`;
      s += `<line x1="${W - R - 250}" x2="${W - R - 200}" y1="${Tp + 20 + si * 34}" y2="${Tp + 20 + si * 34}" stroke="${col}" stroke-width="4" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
      s += text(W - R - 188, Tp + 27 + si * 34, name, { size: 20 });
    });
    // Bestwerte
    s += `<circle cx="${x(44000)}" cy="${y(0.00482)}" r="11" fill="none" stroke="${AMBER}" stroke-width="2.5"/>`;
    s += text(x(44000) - 4, y(0.00482) + 40, '0,00482', { anchor: 'end', fill: AMBER, size: 21, weight: 600 });
    s += `<circle cx="${x(35000)}" cy="${y(0.00512)}" r="11" fill="none" stroke="${AMBER2}" stroke-width="2.5"/>`;
    s += text(x(35000), y(0.00512) - 20, '0,00512', { anchor: 'middle', fill: AMBER2, size: 21, weight: 600 });
    return s + '</svg>';
  }

  // Station 8: Orakel-Versatz-Sweep (Lauf 62), lifted_clear k von 8
  function heatmap() {
    const rows = csv('orakel_versatz_lauf62.csv');
    const dxs = [-1.5, 0, 1.5, 3], dys = [0, 1.5, 3, 4.5, 6];
    const W = 760, H = 410, L = 150, Tp = 20, cw = 112, ch = 66;
    const get = (dx, dy) => rows.find((r) => +r.dx_cm === dx && +r.dy_cm === dy);
    let s = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="Versatz-Sweep Lauf 62">`;
    dxs.forEach((dx, i) => {
      const yy = Tp + i * ch;
      s += text(L - 14, yy + ch / 2 + 7, (dx > 0 ? '+' : '') + de(dx, 1), { anchor: 'end', fill: DIM, size: 19 });
      dys.forEach((dy, j) => {
        const xx = L + j * cw;
        const r = get(dx, dy);
        if (!r) {
          s += `<rect x="${xx + 3}" y="${yy + 3}" width="${cw - 6}" height="${ch - 6}" rx="6" fill="none" stroke="${GRID}" stroke-dasharray="4 4"/>`;
          s += text(xx + cw / 2, yy + ch / 2 + 7, '–', { anchor: 'middle', fill: '#4a5869', size: 20 });
          return;
        }
        const k = +r.lifted_clear_k, n = +r.episoden;
        const a = 0.08 + 0.85 * (k / n);
        s += `<rect x="${xx + 3}" y="${yy + 3}" width="${cw - 6}" height="${ch - 6}" rx="6" fill="rgba(34,211,238,${a.toFixed(2)})"/>`;
        s += text(xx + cw / 2, yy + ch / 2 + 8, `${k}/${n}`, { anchor: 'middle', fill: k / n > 0.45 ? '#06222a' : INK, size: 22, weight: 600 });
        if (dx === 0 && (dy === 0 || dy === 3)) {
          s += `<rect x="${xx + 1}" y="${yy + 1}" width="${cw - 2}" height="${ch - 2}" rx="8" fill="none" stroke="${INK}" stroke-width="2.5"/>`;
        }
      });
    });
    dys.forEach((dy, j) => {
      s += text(L + j * cw + cw / 2, Tp + dxs.length * ch + 30, de(dy, 1), { anchor: 'middle', fill: DIM, size: 19 });
    });
    s += text(L + (dys.length * cw) / 2, Tp + dxs.length * ch + 66, 'dy: seitlich nach außen [cm]', { anchor: 'middle', size: 20 });
    s += text(30, Tp + (dxs.length * ch) / 2, 'dx: weg vom Roboter [cm]', { anchor: 'middle', size: 20, rot: -90 });
    return s + '</svg>';
  }

  // Station 10: visuelles Gate, MSE Hände vor/nach Co-Training (Mittel über 3 Seeds)
  function gate() {
    const rows = csv('gate_mse_haende.csv');
    const pick = (p) => rows.find((r) => r.checkpoint.indexOf(p) === 0);
    const groups = [
      ['N1.6', pick('N1.6 vision-v2 @44000'), pick('C N1.6 cube4 @44000')],
      ['N1.7', pick('N1.7 vision-v1 @44000'), pick('D N1.7 cube4 @44000')],
    ];
    const conds = [['echt_mittel', 'Echtbild', AMBER], ['sim5_mittel', 'Sim 5 cm', CYAN], ['sim4_mittel', 'Sim 4 cm', CYAN]];
    const W = 900, H = 520, L = 100, R = 20, Tp = 60, B = 92, max = 0.08;
    const y = (v) => Tp + (1 - v / max) * (H - Tp - B);
    let s = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="MSE Hände vor und nach Co-Training">`;
    for (const v of [0, 0.02, 0.04, 0.06, 0.08]) {
      s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="${v ? GRID : DIM}"/>`;
      s += text(L - 12, y(v) + 7, de(v, 2), { anchor: 'end', fill: DIM, size: 19 });
    }
    s += text(24, (Tp + H - B) / 2, 'MSE Hände', { anchor: 'middle', size: 21, rot: -90 });
    const gw = (W - L - R) / 2, bw = 52;
    groups.forEach(([name, before, after], gi) => {
      const gx = L + gi * gw + 24;
      conds.forEach(([k, lab, col], ci) => {
        const cx = gx + ci * 128;
        const vb = +before[k], va = +after[k];
        s += `<rect x="${cx}" y="${y(vb)}" width="${bw}" height="${y(0) - y(vb)}" fill="${col}" fill-opacity="0.22" stroke="${col}" stroke-width="2"/>`;
        s += `<rect x="${cx + bw + 6}" y="${y(va)}" width="${bw}" height="${y(0) - y(va)}" fill="${col}"/>`;
        s += text(cx + bw / 2, y(vb) - 8, de(vb, 4), { anchor: 'middle', fill: DIM, size: 16 });
        s += text(cx + bw * 1.5 + 6, y(va) - 8, de(va, 4), { anchor: 'middle', fill: col, size: 16, weight: 600 });
        s += text(cx + bw + 3, H - B + 28, lab, { anchor: 'middle', fill: DIM, size: 18 });
      });
      s += text(gx + 180, H - B + 64, name + (gi ? ' (D, cube4)' : ' (C, cube4)'), { anchor: 'middle', size: 22, weight: 600 });
    });
    // Legende
    s += `<rect x="${L + 10}" y="14" width="26" height="22" fill="#9aa7b4" fill-opacity="0.22" stroke="#9aa7b4" stroke-width="2"/>`;
    s += text(L + 46, 32, 'ohne Co-Training', { size: 19 });
    s += `<rect x="${L + 250}" y="14" width="26" height="22" fill="#9aa7b4"/>`;
    s += text(L + 286, 32, 'mit Co-Training (Mischung 0,15)', { size: 19 });
    return s + '</svg>';
  }

  window.PresCharts = { csv, valMse, heatmap, gate };
})();
