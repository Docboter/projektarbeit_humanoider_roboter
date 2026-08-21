# Würfellage aus den Realbildern

**Stand:** 2026-08-21 · aktuelles Verfahren: `pick_anchored_homography` Version 4

Der Realdatensatz enthält Roboterzustände, Aktionen und Videos, aber keine Objektposen. Für einen
brauchbaren Replay muss der Sim-Würfel dort beginnen, wo er im Realbild lag. Andernfalls greift der
Roboter neben den Würfel und ein daraus erzeugter Co-Training-Datensatz koppelt Bild und Aktion
falsch.

Das aktuelle Verfahren lernt die Abbildung von den beiden Kopfkamerabildern nach Tisch-XY aus
realen Pick-Ereignissen. Es korrigiert weder FOV noch Brennweite aus der scheinbaren Würfelgröße
und
verschiebt keine Zielpose nachträglich zu einem Greifpunkt.

## 1. Aktueller Datenfluss

```text
40 Kalibrierungsepisoden
  ├─ Farbtracking beider Kopfkameras → Bewegungsbeginn je Würfel
  ├─ stabile Top-Face-Pixel strikt vor dem Bewegungsbeginn
  ├─ wenige observation.state-Zeilen direkt in Isaac setzen → Fingerkuppen-FK
  ├─ eindeutige Hand + Würfelfarbe → Pick-Anker Pixel ↔ Tisch-XY
  └─ 80/20-Split ganzer Episoden, Seed 17
       ├─ RANSAC-Homographie je Kamera auf Fit-Episoden
       └─ Qualitäts-Gates auf unberührten Holdout-Episoden

separat gewählte Ziel-Episoden
  ├─ stabile Würfelpixel vor ihrer ersten Bewegung
  ├─ Homographie je verfügbarer Kopfkamera
  ├─ Stereo-Median oder strenger Einzelkamera-Fallback
  └─ cube_poses.json v4
       └─ einmalig setzen → ausschließlich PhysX → replay_manifest.json v2
```

Die Auswahl ist bewusst getrennt:

- `REPLAY_CALIBRATION_NUM_EPISODES=40` wählt ab Episode 0 ausschließlich die
  Kalibrierungsmenge.
- `REPLAY_NUM_EPISODES`, `REPLAY_START_EPISODE` und `REPLAY_EPISODE_IDS` wählen ausschließlich
  die später zu rekonstruierenden und zu rendernden Ziel-Episoden.

Eine Ziel-Episode darf auch in den 40 Kalibrierungsepisoden enthalten sein. Der Holdout prüft die
Generalisierung der Homographie innerhalb der Kalibrierungsmenge; er ist kein Train/Test-Split des
späteren Replay-Datensatzes.

## 2. Anker aus Bild und sparsamer FK

### 2.1 Bewegungsbeginn und stationäre Bildmessung

`collect_replay_anchors.py` verfolgt rot, grün und gelb in beiden Kopfkameras bei halber
Auflösung. Die frühe Basisposition ist der Median der ersten zehn gültigen Messungen. Ein
Bewegungsbeginn wird erst akzeptiert, wenn der Farbmittelpunkt mindestens 8 px verschoben bleibt
und das für fünf aufeinanderfolgende Frames. Die Onsets der beiden Kameras dürfen höchstens
zwölf
Frames auseinanderliegen.

Die Oberseitenmessung verwendet ausschließlich Frames vor dem jeweiligen Kamera-Onset. Sie
benötigt mindestens fünf räumlich konsistente Messungen; Bildrandtreffer und Messungen mit mehr
als
12 px Abstand zum Median werden verworfen. Fehlt ein Onset, können frühe Frames im Bericht
erscheinen, aber nicht als Kalibrierungsanker verwendet werden.

### 2.2 Sparse Direct-State-FK

Rund um den visuellen Onset werden Schließintervalle im aufgezeichneten Gelenkzustand gesucht. Nur
deren Anfangs- und Endzustände sowie die Zustände bei Onset −2, Onset und Onset +2 werden direkt
in
Isaac gesetzt. Es findet kein Action-Replay statt.

Ein Anker wird nur akzeptiert, wenn:

- eine Hand mindestens 6 mm schließt;
- ihre Schließung mindestens 2 mm stärker als die andere Hand ist;
- die drei echten Fingerkuppen verfügbar sind;
- ihr Schwerpunkt im Arbeitsraum x = 0,20–0,50 m, y = −0,30–0,30 m und
  z = 0,82–1,05 m liegt;
- beide Kameras einen ausreichend ähnlichen Bewegungsbeginn und stabile Vorher-Pixel liefern.

Der Anker besteht aus Episode, Farbe, Onset, Hand, den beiden Pixelmessungen und dem Median der
Fingerkuppenpositionen. Originalaktionen werden nur gehasht und nach der Verarbeitung erneut auf
Unverändertheit geprüft.

## 3. Homographie und Abnahme

Je Kopfkamera wird mit RANSAC eine projektive Abbildung von Pixel `(u, v)` nach env-lokalem
Tischpunkt `(x, y)` gelernt. Der RANSAC-Inlier-Schwellwert beträgt 2 cm. Der Split erfolgt auf
vollständigen Episoden: 80 % Fit, 20 % Holdout, deterministisch mit Seed 17.

`geometry_calibration.json` wird nur geschrieben, wenn alle Bedingungen gelten:

| Gate | Grenzwert |
|---|---:|
| Fit-Anker / Fit-Episoden | mindestens 24 / 8 |
| Holdout-Anker / Holdout-Episoden | mindestens 6 / 3 |
| Arbeitsraumabdeckung je Kamera | mindestens 12 cm in x und y |
| Holdout-Fehler je Kamera | Median ≤ 1,5 cm, p90 ≤ 3 cm |
| Differenz der Kameraschätzungen im Holdout | Median ≤ 2 cm, p90 ≤ 3 cm |

Das Ergebnis hat `version: 4`, `method: "pick_anchored_homography"` und `valid: true`. Ein
fehlgeschlagener Lauf schreibt weiterhin den vollständigen `calibration_report.json`, aber keine
verwendbare Kalibrierungsdatei. Eine eventuell vorhandene ungültig gewordene Ausgabedatei wird
entfernt.

Die aus einer achsenparallelen Bounding-Box geschätzte 5-cm-Würfelkante ist nur noch Diagnose. Der
fehlgeschlagene Lauf mit 3,68 cm zeigte, dass Perspektive, Gier und unvollständige Oberseiten diese
Skalenschätzung verzerren. Sie verändert deshalb ausdrücklich keine Intrinsics und erzeugt keine
FOV-Korrektur.

## 4. Anfangsposen der Ziel-Episoden

`replay-poses` verfolgt jede Farbe in jeder Ziel-Episode erneut. Es verwendet mindestens fünf
stabile Messungen vor dem jeweiligen Bewegungsbeginn und wendet die geprüfte Homographie der
jeweiligen Kamera an.

- Liefern beide Kameras eine Position, wird ihr Median verwendet. Mehr als 3 cm Differenz verwirft
  den Würfel.
- Liefert nur eine Kamera eine Position, sind mindestens fünf echte Top-Face-Detektionen nötig.
  Verwendet wird dann ausschließlich der transformierte Median dieser Top-Face-Pixel, nicht ein
  Full-Blob-Median.
- Das Ergebnis muss innerhalb x = 0,25–0,45 m und y = −0,25–0,25 m liegen.
- Fehlt einer der drei Würfel, erhält die Episode `status: "skipped"` und wird nicht gerendert.

`cube_poses.json` hat `version: 4` und `method: "stationary_top_face_homography"`. Die Höhe bleibt
fest beim Würfelmittelpunkt z = 0,915 m, die Orientierung vorerst bei Identität. Getrennte
Kameraschätzungen, verwendete Frames und Detektionsquelle stehen am Block. Die
`pick_anchor_diagnostics` stehen dagegen einmal im jeweiligen Episode-Eintrag. Auf Dokumentebene
stehen Quelldatensatz, Kalibrierungspfad und `calibration_sha256`.

Die Position stammt vollständig aus der Homographie. Es gibt keine 75/25-Mischung mit einem
Fingerkuppenschwerpunkt, keinen Greifpunkt-Fallback und keine Zufallsposition.

## 5. Physikbasierte Validierung

Der Renderer setzt Roboter und Würfel vor Frame 0. Danach gibt es keine weiteren Pose-Schreibungen,
kein Tracking und kein kinematisches Attach. Jede Bewegung eines Würfels stammt ausschließlich aus
PhysX unter den unveränderten Originalaktionen.

Vor dem Action-Replay werden die gesetzten Würfel in beiden Kopfkameras gegen ihre erwartete
Projektion geprüft. Die Abnahme verlangt sechs Messungen sowie höchstens 5 px Median und 10 px
p90.
Im Videomodus bleibt eine Abweichung als sichtbare Diagnose erlaubt; im Dataset-Modus wird die
Episode verworfen.

Während des Replays werden Würfel- und Fingerkuppenpositionen nur gelesen. Das Manifest Version 2
bewertet den erwarteten ersten Pick aus den Ankerdiagnosen. Erfolg bedeutet: Der erwartete Würfel
wird nach diesem Pick mindestens 2 cm angehoben und bleibt mindestens fünf aufeinanderfolgende
Frames über dieser Schwelle. Diese Metrik verändert die Simulation nicht.

Vor dem Rendern werden beide v4-Eingabeartefakte, ihre Methoden, ihr Quelldatensatz und der
Kalibrierungs-Hash in `cube_poses.json` geprüft. Overwrite umgeht diese Prüfung nicht. Im Manifest
Version 2 stehen `calibration_sha256` und `poses_sha256` auf oberster Ebene, nicht in den
Episode-Einträgen. Alte Renderausgaben werden nur fortgesetzt, wenn diese Felder und die
Griffvalidierung kompatibel sind. Andernfalls sind ein neues Ausgabeverzeichnis oder
`REPLAY_OVERWRITE=1` erforderlich.

## 6. Historische Ansätze und ihr Befund

Die folgenden Verfahren sind dokumentierte Fehlversuche oder Ablationen, nicht der aktuelle Pfad.

### 6.1 Globales Minimum der Fingeröffnung aus `scan.json`

Das Minimum liegt bei einem Pick-and-Place oft irgendwo im breiten geschlossenen Transportintervall
statt am Pick. Im Lauf vom 2026-08-17 lagen 48 von 116 erkannten Griffen hinter 60 % der Episode,
21 sogar hinter 80 %. Zusätzlich wurde z verworfen und nur ein Greifpunkt je Hand erzeugt. Dieser
Ansatz darf nicht mehr als Würfelposition verwendet werden.

### 6.2 Nur bis zum Griff rendern

`--stop-at-grasp` begrenzt falsche Frames nach dem Griff, rekonstruiert aber keine Anfangsposition.
Liegt der Würfel falsch, ist bereits die gesamte Anfahrt falsch. Der Schnitt kann für den alten
Co-Training-Renderer weiterhin eine Sicherheitsmaßnahme sein, gehört aber nicht zum neuen
physikbasierten Replay.

### 6.3 Blob-Schwerpunkt plus analytisches Pinhole-Modell

Der Schwerpunkt der sichtbaren Silhouette enthält Ober- und Seitenflächen und liegt nicht auf der
Projektion des Würfelmittelpunkts. Eine additive Bias-Korrektur ist nur lokal gültig. Außerdem
kann
ein in sich exakter `project`/`backproject`-Roundtrip einen gemeinsamen Fehler der realen und
simulierten Kamerapose nicht erkennen.

`camera_geometry.py` bleibt für Projektion, Debugging und Renderer-Abnahme verfügbar. Die
produktive Pixel-zu-Tisch-Abbildung der Version 4 kommt jedoch aus Holdout-geprüften Pick-Ankern.

### 6.4 Oberseiten-AABB als FOV-Kalibrierung und 75/25-Fusion

Die zwischenzeitliche Version schätzte die Kameraskala aus der größten achsenparallelen
Oberseiten-Ausdehnung und kombinierte erkannte Würfel pauschal zu 75 % mit einem Greifpunkt. Beide
Schritte sind entfernt: Die AABB ist bei Perspektive und Würfelgier kein zuverlässiges
Längennormal, und ein Greifpunkt ist ohne eindeutige zeitliche Farbzuordnung keine Startpose.

## 7. Artefakte und Befehle

| Artefakt | Inhalt |
|---|---|
| `replay_anchors.json` | v4-Anker, Ablehnungsgründe, Action-Hashes und Bilddiagnosen |
| `geometry_calibration.json` | gültige v4-Homographien, Fit/Holdout und Qualitätswerte |
| `calibration_report.json` | vollständiger Bericht auch bei fehlgeschlagenem Gate |
| `cube_poses.json` | v4-Anfangsposen der separat ausgewählten Ziel-Episoden |
| `replay_manifest.json` | v2-Render-, Projektions- und Griffdiagnose |

```bash
REPLAY_CALIBRATION_NUM_EPISODES=40 \
REPLAY_CALIBRATION_HOLDOUT_RATIO=0.2 \
REPLAY_CALIBRATION_SEED=17 \
./Simulation/server_rl_run.sh replay-calibrate

REPLAY_EPISODE_IDS="0 12 41" REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-poses

DR_ENABLED=0 REPLAY_EPISODE_IDS="0 12 41" REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-render
```

Die vollständige Bedienung und alle Umgebungsvariablen stehen in
[replay-videos-aus-realdaten.md](replay-videos-aus-realdaten.md).
