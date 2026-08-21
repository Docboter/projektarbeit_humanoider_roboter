# Physikbasierte Replay-Videos aus dem Realdatensatz

Der Workflow rekonstruiert die einmalige Anfangsposition der drei Würfel aus realen
Kopfkamerabildern. Die aktuelle Kalibrierung ist `pick_anchored_homography` Version 4:
Bewegungsbeginne im Realvideo werden mit wenigen, direkt gesetzten Roboterzuständen in Isaac
verknüpft. Daraus wird je Kopfkamera eine Abbildung von Pixelkoordinaten nach Tisch-XY gelernt.

Die Originalaktionen werden während der Kalibrierung nicht abgespielt. Beim finalen Replay wird
jeder Würfel genau einmal vor Frame 0 gesetzt; anschließend verändert ausschließlich PhysX seine
Pose. Es gibt weder Tracking noch kinematisches Attach.

## Schnellstart

```bash
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh replay-prepare

REPLAY_CALIBRATION_NUM_EPISODES=40 \
./Simulation/server_rl_run.sh replay-calibrate

REPLAY_EPISODE_IDS="0" REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-poses

DR_ENABLED=0 REPLAY_EPISODE_IDS="0" REPLAY_MAX_FRAMES=60 REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-render
```

Nach der Sichtprüfung des 60-Frame-Videos folgt dieselbe Episode vollständig und danach ein Lauf
über zehn Ziel-Episoden:

```bash
DR_ENABLED=0 REPLAY_EPISODE_IDS="0" REPLAY_MAX_FRAMES=0 REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-render

REPLAY_NUM_EPISODES=10 REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-poses

DR_ENABLED=0 REPLAY_NUM_EPISODES=10 REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-render
```

Die Videos liegen standardmäßig unter `$RL_HOST_DATA_DIR/cube_replay/videos/`, Diagnoseartefakte
unter `$RL_HOST_DATA_DIR/cube_replay/work/`.

## Kalibrierung: `pick_anchored_homography` v4

`replay-calibrate` verarbeitet standardmäßig die ersten 40 Trainingsepisoden. Diese
Kalibrierungsauswahl ist absichtlich unabhängig von `REPLAY_NUM_EPISODES`,
`REPLAY_START_EPISODE` und `REPLAY_EPISODE_IDS`, die nur Ziel-Episoden für `replay-prepare`,
`replay-poses` und `replay-render` auswählen.

Der Schritt besteht aus zwei Phasen:

1. `collect_replay_anchors.py` verfolgt die Farbzentren beider Kopfkameras in halber Auflösung.
   Ein Bewegungsbeginn gilt nach mindestens fünf aufeinanderfolgenden Frames mit mindestens 8 px
   Abstand zur frühen Basisposition. Die beiden Kamera-Onsets dürfen höchstens zwölf Frames
   auseinanderliegen.
2. Für einen möglichen Pick werden nur wenige aufgezeichnete `observation.state`-Zeilen direkt in
   Isaac gesetzt. Es werden keine Actions ausgeführt. Die echte Fingerkuppen-FK muss eine
   eindeutige
   Hand mit mindestens 6 mm Schließung und 2 mm Vorsprung gegenüber der anderen Hand liefern. Der
   Anker verbindet dann die stabilen Pixel vor dem Bewegungsbeginn mit dem Median der
   Fingerkuppenpositionen bei Onset −2, Onset und Onset +2.

Aus den akzeptierten Ankern wird je Kopfkamera eine RANSAC-Homographie
`pixel_to_table_xy` gelernt. Vollständige Episoden werden deterministisch mit Seed 17 in 80 % Fit
und 20 % Holdout aufgeteilt; kein Anker einer Holdout-Episode fließt in den Fit ein.

Die Kalibrierung wird nur als gültige `geometry_calibration.json` geschrieben, wenn alle Gates
erfüllt sind:

- mindestens 24 Fit-Anker aus mindestens acht Fit-Episoden;
- mindestens sechs Holdout-Anker aus mindestens drei Holdout-Episoden;
- je Kamera mindestens vier Fit-Anker und mindestens 12 cm Abdeckung in x und y;
- je Kamera Holdout-Fehler höchstens 1,5 cm Median und 3 cm p90;
- Differenz der beiden Kameraschätzungen höchstens 2 cm Median und 3 cm p90.

Ein fehlgeschlagenes Gate entfernt eine eventuell vorhandene gültige Kalibrierungsdatei und
schreibt die vollständige Diagnose nach `calibration_report/calibration_report.json`. Das
Anchor-Dokument
muss Version 4, denselben Datensatz und exakt dieselben 40 Kalibrierungsepisoden abdecken; veraltete
oder teilweise Artefakte werden nicht wiederverwendet.

Die bildbasierte AABB-Würfelkante bleibt als Diagnose im Bericht. Sie verändert weder Brennweite
noch FOV. Es gibt keine FOV-Korrektur aus der scheinbaren Würfelgröße.

## Zielposen: `cube_poses.json` v4

`replay-poses` arbeitet ausschließlich auf der separat gewählten Zielmenge. Für jede Farbe und
Kamera wird zuerst der Bewegungsbeginn verfolgt. Aus mindestens fünf stabilen Frames strikt davor
wird eine robuste Oberseitenposition bestimmt und durch die kalibrierte Homographie nach Tisch-XY
abgebildet.

Wenn beide Kameras gültig sind, wird ihr Median verwendet; mehr als 3 cm Widerspruch verwirft den
Würfel. Eine einzelne Kamera ist nur erlaubt, wenn mindestens fünf echte Top-Face-Detektionen
vorliegen. In diesem Fall wird ausdrücklich deren separater `top_face_uv`-Median transformiert,
nicht der Median einer möglichen Full-Blob-Mischung. Positionen außerhalb x = 0,25–0,45 m oder
y = −0,25–0,25 m werden verworfen. Nur Episoden mit drei gültigen Würfeln erhalten
`status: "ok"`.

Das Dokument hat `version: 4` und `method: "stationary_top_face_homography"`. Auf Dokumentebene
stehen `source_dataset`, der Pfad `calibration` und deren `calibration_sha256`. Jeder Block enthält
unter anderem `position_m`, `position_source`, `confidence` und die getrennten
`camera_estimates`. `pick_anchor_diagnostics` gehört zum jeweiligen Episode-Eintrag, nicht zu einem
Block. Die Pick-Anker dienen nur der Diagnose und Kalibrierung; es gibt keine 75/25-Fusion, keinen
Greifpunkt-Fallback und keine Zufallsplatzierung.

Ohne `REPLAY_OVERWRITE=1` werden vorhandene Episoden nur übernommen, wenn das Pose-Dokument
exakt Version 4 und `stationary_top_face_homography` verwendet, zum aufgelösten Quelldatensatz
gehört und sein `calibration_sha256` dem aktuellen Kalibrierungsinhalt entspricht. Bei einem
inkompatiblen Pose-Dokument wird nichts daraus fortgesetzt; die gewählte Zielmenge wird neu
rekonstruiert.
`REPLAY_OVERWRITE=1` rekonstruiert die gewählten Posen immer neu.

## Replay, Renderer-Gates und Griffmetrik

`replay-render` setzt den Roboter auf `observation.state[0]`, setzt die drei Würfel einmalig und
führt danach bei 30 Hz unveränderte `action[t]` aus. Die Actions werden elementweise und per
SHA-256 geprüft. Pro Episode entstehen fünf MP4s: beide Kopfkameras, beide Wrist-Kameras und die
Szenenkamera.

Vor dem Replay prüft der Renderer alle sechs Kopfkamera-Projektionen. Erlaubt sind höchstens 5 px
Median und 10 px p90. Im Videomodus wird bei Abweichung zur manuellen Diagnose weitergerendert; der
Dataset-Modus verwirft die Episode. Die Wrist-Abnahme verlangt beide Wrist-Kameras, mindestens zwei
Messungen, höchstens 10 px Median und 20 px p90. Der Dataset-Modus bleibt zusätzlich gesperrt,
solange die Kalibrierung nicht ausdrücklich `wrist_calibration.dataset_ready` enthält.

Das `replay_manifest.json` hat Version 2. `source_dataset`, `calibration`,
`calibration_sha256` und `poses_sha256` sind Felder auf der obersten Manifestebene. Pro Episode
enthält es:

- Hashes von Kalibrierung, Posen und Originalaktionen;
- Kopf- und Wrist-Projektionsfehler;
- die erwartete erste Farbe und Hand aus den Pick-Ankern;
- minimale Fingerkuppenabstände und maximale Anhebung je Würfel;
- `grasp_success=true`, wenn der erwartete Würfel nach dem erwarteten Pick mindestens 2 cm für
  mindestens fünf aufeinanderfolgende Frames angehoben ist.

Die Griffmetrik liest nur PhysX-Positionen. Sie setzt oder korrigiert keine Würfelpose. Ein
fehlender eindeutiger erster Pick ergibt `status: "unavailable"`, nicht einen erfundenen Erfolg.

Vor jedem Rendern werden die Eingabeartefakte strikt geprüft: Posen und Kalibrierung müssen
Version 4, die erwarteten Methoden und denselben aufgelösten Quelldatensatz tragen; außerdem muss
der im Pose-Dokument gespeicherte `calibration_sha256` dem gelesenen Kalibrierungsinhalt
entsprechen.
`REPLAY_OVERWRITE=1` umgeht diese Eingabeprüfung nicht.

Ohne Overwrite wird eine Renderausgabe nur fortgesetzt, wenn Manifest-Version, Quelldatensatz,
Kalibrierungs-Hash, Posen-Hash und die Version-2-Griffvalidierung jeder vorhandenen Episode
kompatibel sind. Der vollständige Episode-Eintrag bleibt erhalten und wird nur mit
`render_status: "existing"` markiert. Existierende MP4-/Parquet-Dateien ohne passenden
Manifest-Eintrag werden nicht stillschweigend übernommen. Bei inkompatibler Ausgabe ist
`REPLAY_OVERWRITE=1` oder ein neues Ausgabeverzeichnis erforderlich; Overwrite erzeugt die
gewählten Renderausgaben neu.

## Variablen

| Variable | Default | Bedeutung |
|---|---:|---|
| `REPLAY_CALIBRATION_NUM_EPISODES` | `40` | Separate Kalibrierungsepisoden ab Episode 0 |
| `REPLAY_CALIBRATION_HOLDOUT_RATIO` | `0.2` | Anteil vollständiger Holdout-Episoden |
| `REPLAY_CALIBRATION_SEED` | `17` | Deterministischer Split und RANSAC |
| `REPLAY_NUM_EPISODES` | `10` | Maximale Anzahl gewählter Ziel-Episoden |
| `REPLAY_START_EPISODE` | `0` | Erste Ziel-Episode bei fortlaufender Auswahl |
| `REPLAY_EPISODE_IDS` | leer | Explizite Ziel-Episoden; überschreibt Start/Anzahl |
| `REPLAY_MAX_FRAMES` | `0` | `0` vollständig, sonst Videotechniktest |
| `REPLAY_OVERWRITE` | `0` | Posen beziehungsweise Renderausgabe neu erzeugen |
| `REPLAY_OUTPUT_MODE` | `videos` | `videos` oder `dataset` |
| `REPLAY_WORK` | `/data/cube_replay/work` | Kalibrierung, Posen und Berichte |
| `REPLAY_OUT` | `/data/cube_replay/videos` | MP4-Ausgabe |
| `REPLAY_DATASET_OUT` | `/data/cube_replay/dataset` | LeRobot-v2.1-Ausgabe |
| `REPLAY_DATASET` | `/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset` | Quelle |

Übersprungene Ziel-Episoden werden nicht durch weitere ersetzt. Der Workflow benötigt keine neue
Maschineninitialisierung gegenüber der bestehenden Server-Einrichtung.
