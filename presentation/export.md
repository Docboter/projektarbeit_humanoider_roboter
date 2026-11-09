# PDF-Export der Präsentation

> **TL;DR:** Wie aus der 3D-Präsentation ein PDF entsteht (eine Seite je Schritt, 23 Seiten) und wie
> die Lite-Standbilder erneuert werden. Bedienung und Aufbau der Präsentation stehen in
> [README.md](README.md).

## Einmalig einrichten

```bash
cd presentation/tools
npm install                       # Playwright 1.63 (passt zu Chromium 1243)
npx playwright install chromium   # nur falls Chromium noch fehlt
```

## PDF erzeugen

```bash
node presentation/tools/export_pdf.mjs
```

Ergebnis: `presentation/export/praesentation.pdf`, Seitengröße 1920×1080 px, eine Seite je Schritt
(Stationen mit Unterschritten erscheinen mehrfach). Die Einzelbilder liegen daneben unter
`export/shots/`. Der Ordner `export/` ist nicht eingecheckt.

Das Skript öffnet `index.html?capture=1` in einem unsichtbaren Chromium (Software-WebGL), geht
alle Schritte durch, lässt die Szene mit fester Simulationsuhr bis zum Standbild-Zeitpunkt laufen
(`capture` je Station in `js/stations.js`, sonst 3 s) und macht einen Screenshot. Gleiche Eingabe
ergibt gleiche Bilder. Fehlende Dateien und Skriptfehler meldet es am Ende und endet dann mit
Code 1.

## Optionen

| Option | Wirkung |
|---|---|
| `--lite` | zusätzlich `assets/lite/sNN-K.jpg` ohne Text erzeugen (Lite-Modus, Sprechervorschau) |
| `--width 1280 --height 720` | in Beamer-Auflösung exportieren |
| `--only 9` | nur Station 9 |
| `--shots DIR` | nur PNG-Screenshots nach `DIR`, kein PDF |
| `--pdf PFAD` | anderer Ausgabepfad |
| `--no-pdf` | Screenshots ohne PDF |

Nach Änderungen an Folientext oder Kameras einmal `--lite` laufen lassen und `assets/lite/`
mit einchecken.
