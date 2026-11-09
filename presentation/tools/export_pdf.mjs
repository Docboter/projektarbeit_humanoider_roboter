#!/usr/bin/env node
// TL;DR: Exportiert die 3D-Präsentation mit Playwright: ein Screenshot je Schritt, daraus ein PDF
// (eine Seite pro Schritt) und optional die Lite-Standbilder (assets/lite/) ohne Text-Overlay.
//
// Aufruf (einmalig: cd presentation/tools && npm install):
//   node presentation/tools/export_pdf.mjs                 # PDF nach presentation/export/
//   node presentation/tools/export_pdf.mjs --lite          # zusätzlich assets/lite/*.jpg erzeugen
//   node presentation/tools/export_pdf.mjs --shots DIR     # nur Screenshots (PNG) nach DIR
//   Optionen: --width 1280 --height 720 · --only 7 (nur Station 7) · --no-pdf
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const PRES = resolve(HERE, '..');
const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args[i + 1] : def;
};
const flag = (name) => args.includes('--' + name);

const W = +opt('width', 1920);
const H = +opt('height', 1080);
const only = opt('only', null);
const lite = flag('lite');
const noPdf = flag('no-pdf') || !!opt('shots', null);
const shotDir = resolve(opt('shots', join(PRES, 'export', 'shots')));
const pdfPath = resolve(opt('pdf', join(PRES, 'export', 'praesentation.pdf')));
const liteDir = join(PRES, 'assets', 'lite');

mkdirSync(shotDir, { recursive: true });
if (lite) mkdirSync(liteDir, { recursive: true });

const browser = await chromium.launch({
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
const problems = [];
page.on('pageerror', (e) => problems.push('Seitenfehler: ' + e.message));
page.on('console', (m) => {
  if (m.type() === 'error') problems.push('Konsole: ' + m.text());
});
page.on('requestfailed', (r) => {
  if (!r.url().includes('/assets/lite/')) problems.push('Fehlende Datei: ' + r.url());
});

const url = pathToFileURL(join(PRES, 'index.html')).href + '?capture=1';
await page.goto(url);
await page.waitForFunction(() => window.__pres && window.__pres.ready, null, { timeout: 60000 });
const flat = await page.evaluate(() => window.__pres.flat);

const shots = [];
for (let i = 0; i < flat.length; i++) {
  const { si, k } = flat[i];
  if (only && +only !== si + 1) continue;
  await page.evaluate((j) => window.__pres.go(j), i);
  const t = await page.evaluate(() => window.__pres.captureTime());
  await page.evaluate((sec) => window.__pres.advance(sec), t);
  await page.waitForTimeout(150); // Bilder im Overlay dekodieren
  const name = `s${String(si + 1).padStart(2, '0')}-${k}`;
  if (lite) {
    await page.evaluate(() => document.body.classList.add('nooverlay'));
    await page.screenshot({ path: join(liteDir, name + '.jpg'), type: 'jpeg', quality: 82 });
    await page.evaluate(() => document.body.classList.remove('nooverlay'));
  }
  const file = join(shotDir, name + (noPdf ? '.png' : '.jpg'));
  await page.screenshot(noPdf ? { path: file } : { path: file, type: 'jpeg', quality: 90 });
  shots.push(file);
  process.stdout.write(`${name} `);
}
process.stdout.write('\n');

if (!noPdf) {
  const html = `<!doctype html><html><head><style>
    @page { size: ${W}px ${H}px; margin: 0 }
    html, body { margin: 0; background: #000 }
    img { display: block; width: ${W}px; height: ${H}px; page-break-after: always; break-after: page }
  </style></head><body>${shots.map((s) => `<img src="${pathToFileURL(s).href}">`).join('')}</body></html>`;
  const tmp = join(shotDir, '_pdf.html');
  writeFileSync(tmp, html);
  const p2 = await browser.newPage();
  await p2.goto(pathToFileURL(tmp).href, { waitUntil: 'load' });
  mkdirSync(dirname(pdfPath), { recursive: true });
  await p2.pdf({ path: pdfPath, width: `${W}px`, height: `${H}px`, printBackground: true });
  rmSync(tmp);
  console.log('PDF:', pdfPath, `(${shots.length} Seiten)`);
}
await browser.close();
if (problems.length) {
  console.log('Hinweise:\n  ' + [...new Set(problems)].join('\n  '));
  process.exitCode = 1;
}
