# 3D-Präsentation „Die Reise eines Griffs“

> **TL;DR:** Kurzanleitung zur interaktiven Abschlusspräsentation (Three.js, offline): starten,
> Tasten, Sprechermodus, Lite-Modus, Assets neu bauen. Wie man ein PDF erzeugt, steht in
> [export.md](export.md); woher jede Zahl auf den Folien stammt, in [zahlen.md](zahlen.md).

16 Stationen (23 Schritte, Zeitplan 18 min) in einer durchgehenden 3D-Szene: der Unitree G1 mit
DEX3-Händen am Tisch, der sich ab Station 6 in einen Real- (Bernstein) und einen Sim-Zwilling
(Cyan) teilt. Folientext steht als HTML-Ebene über der Szene, Details in den Sprechernotizen.

## Starten

- **Doppelklick** auf `index.html` (Chrome, Edge oder Chromium). Alles liegt lokal, kein Netz nötig.
- Oder als lokaler Server, falls der Browser `file://` einschränkt:

  ```bash
  cd presentation && python3 -m http.server 8000   # dann http://localhost:8000
  ```

Getestet mit Chromium 1.63 (Playwright) bei 1920×1080 und 1280×720. Auf einer Intel-UHD-iGPU
(Raptor Lake) läuft die Szene in 1920×1080 mit rund 120 fps (Bildschirm-Obergrenze), höchstens
~150 Draw-Calls; Messung mit `?fps=1`.

## Tasten

| Taste | Wirkung |
|---|---|
| → · Leertaste · Bild ↓ · Klick | nächster Schritt |
| ← · Bild ↑ · Rechtsklick | vorheriger Schritt |
| Zahl + Enter | zu Station springen (z. B. `1` `2` Enter) |
| Pos1 / Ende | erste / letzte Station |
| B (oder Punkt) | schwarzer Bildschirm |
| F | Vollbild |
| P | Sprecheransicht im zweiten Fenster |
| H | Tastenhilfe |

Der Vorher/Nachher-Regler in Station 6 lässt sich mit der Maus ziehen, ohne weiterzuschalten.

## URL-Schalter

| Schalter | Wirkung |
|---|---|
| `#12`, `#12.1` | Station 12, Schritt 2 (die URL folgt beim Blättern) |
| `?lite=1` | statt 3D die vorgerenderten Standbilder aus `assets/lite/` (für schwache Rechner) |
| `?fps=1` | Bildrate oben rechts |
| `?still=1` | ohne Kamerafahrten (wie `prefers-reduced-motion`) |
| `?speaker=1` | Sprecheransicht direkt (normal: Taste P) |

Ist im Betriebssystem „Bewegung reduzieren“ aktiv, ersetzt die Präsentation Kamerafahrten durch
kurze Überblendungen.

## Sprecheransicht

Taste **P** öffnet ein zweites Fenster mit den Notizen der aktuellen Station (mit Zeitangabe und
markierten Klicks), der Uhr gegen den Zeitplan sowie Standbildern des aktuellen und des nächsten
Schritts. Pfeiltasten und Knöpfe dort steuern das Hauptfenster. Die Uhr startet beim ersten
Weiterschalten, „Uhr neu“ setzt sie zurück. Pop-up-Blocker für die Datei ggf. erlauben.

## Aufbau

| Pfad | Inhalt |
|---|---|
| `index.html`, `css/style.css` | Bühne 1920×1080 (skaliert), Overlay, Sprecheransicht |
| `js/stations.js` | **Inhalt:** Folientext, Notizen, Zeiten, Kameras, 3D-Zustand je Schritt; Titeldaten in `META` (Datum dort eintragen) |
| `js/world.js` | 3D-Szene: Zwillinge, Pipeline-Pfad, Monitore, Balken, Säulen, Animationen |
| `js/robot.js` | G1-Gelenkbaum, Handschluss, CCD-IK für die Arme |
| `js/charts.js` | SVG-Diagramme aus den eingebetteten CSVs |
| `js/app.js`, `js/speaker.js` | Navigation, Tasten, Lite, Export-Hooks; Sprecherfenster |
| `assets/robot.js`, `images.js`, `data.js` | eingebettetes Robotermodell, Bilder, Messdaten (generiert) |
| `assets/lite/` | Standbilder je Schritt (generiert, für `?lite=1` und die Sprechervorschau) |
| `vendor/` | three.js r159 (MIT), Schrift Inter (SIL OFL 1.1) mit Lizenzen |
| `tools/` | Generatoren und Export (siehe unten) |

Die Assets sind als `<script>`-Dateien eingebettet, weil Chrome über `file://` weder ES-Module
noch Bilder für WebGL-Texturen lädt. Die Konsole meldet, dass `three.min.js` veraltet ist; das ist
erwartet (r159 ist die letzte Version mit klassischem Skript-Build).

## Assets neu bauen

Nur nötig, wenn sich Quellen ändern; die Ergebnisse sind eingecheckt.

```bash
python3 presentation/tools/build_assets.py      # Bilder + CSVs -> assets/images.js, data.js
uv run --no-project --with trimesh --with fast-simplification --with numpy \
  python presentation/tools/build_robot.py      # URDF (data/unitree_ros) -> assets/robot.js
cd presentation/tools && npm install && cd -    # einmalig für Export und Standbilder
node presentation/tools/export_pdf.mjs --lite   # PDF + assets/lite/ neu erzeugen
```

Nach Änderungen an Text oder Kameras `export_pdf.mjs --lite` erneut laufen lassen, damit Lite-Modus
und Sprechervorschau stimmen. Das Robotermodell stammt aus dem Submodul `data/unitree_ros`
(`g1_29dof_with_hand.urdf`, BSD-3-Clause), vereinfacht auf ~55 000 Dreiecke.
