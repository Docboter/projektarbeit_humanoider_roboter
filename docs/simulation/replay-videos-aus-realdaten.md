# Physikbasierte Replay-Videos aus dem Realdatensatz

Der Workflow bestimmt die anfänglichen Würfelpositionen aus den ersten 30 Frames der
beiden Kopfkameras. Eine effiziente Greifstütze ergänzt die CV-Schätzung: Sie erkennt
Fingerbewegungen im aufgezeichneten Gelenkzustand und wertet nur wenige Zustände per
Isaac-Vorwärtskinematik aus. Vollständige Trajektorien werden erst beim finalen Rendering
abgespielt.

Die Würfel werden vor Frame 0 genau einmal gesetzt. Danach verändert ausschließlich PhysX
ihre Pose; es gibt weder Tracking noch kinematisches Attach.

## Schnellstart

```bash
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh replay-prepare

REPLAY_NUM_EPISODES=10 \
./Simulation/server_rl_run.sh replay-calibrate

REPLAY_EPISODE_IDS="0" \
REPLAY_GRASP_SUPPORT=1 \
REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-poses

DR_ENABLED=0 \
REPLAY_EPISODE_IDS="0" \
REPLAY_MAX_FRAMES=60 \
REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-render
```

Die Videos liegen standardmäßig unter:

```text
$RL_HOST_DATA_DIR/cube_replay/videos/
```

Auf dem IKR-Server gehört der benutzerspezifische Datenpfad in `.env.local`, zum Beispiel:

```bash
: "${RL_HOST_DATA_DIR:=$HOME/project/data/RL}"
```

Dadurch funktioniert derselbe Workflow unter verschiedenen Benutzern. Der Container mountet
dieses Verzeichnis nach `/data` und prüft einen bereits vorhandenen Mount auf Widersprüche.

## Was die Schritte tun

### `replay-calibrate`

- läuft ohne Isaac und ohne GPU;
- liest höchstens 30 Frames je Kopfkamera und Episode;
- segmentiert die hellen Oberseiten der roten, grünen und gelben Würfel;
- prüft die Kameraskala gegen die bekannte Würfelkante von 5 cm;
- verwendet fest `z=0.915 m`, statt eine unzuverlässige Tischhöhe zu triangulieren;
- schreibt `geometry_calibration.json` und Kontrollbilder unter `cube_replay/work/`.

### `replay-poses`

Zuerst werden alle drei XY-Positionen aus beiden Kopfkameras bestimmt. Stimmen die beiden
Schätzungen um mehr als 3 cm nicht überein oder fehlt eine Farbe, wird die Episode
übersprungen.

Mit `REPLAY_GRASP_SUPPORT=1` startet anschließend einmalig ein Isaac-Prozess ohne
Kamerarendering. Aus den Fingerzuständen werden wenige mögliche Schließintervalle gewählt.
Nur deren Anfangs- und Endzustände werden direkt gesetzt; die Actions werden nicht
ausgeführt oder verändert. Eine Greifstütze gilt nur, wenn:

- mindestens zwei Fingergelenke koordiniert bewegt werden;
- die gemessene Fingeröffnung um mindestens 6 mm abnimmt;
- die Fingerkuppen im Arbeitsbereich liegen;
- der nächste CV-Würfel höchstens 8 cm entfernt und mindestens 3 cm eindeutiger als der
  zweitnächste ist.

Dann wird nur XY kombiniert:

```text
Endposition = 0,75 × Fingerkuppenschwerpunkt + 0,25 × CV-Position
```

Nicht gegriffene oder mehrdeutige Würfel bleiben rein CV-basiert. Jeder Block enthält in
`cube_poses.json` sowohl `cv_position_m` als auch die tatsächlich verwendete `position_m`
und die vollständige Entscheidungsdiagnose unter `grasp_support`.

Ein reiner CV-Vergleich ist möglich mit:

```bash
REPLAY_EPISODE_IDS="0" \
REPLAY_GRASP_SUPPORT=0 \
REPLAY_POSES=/data/cube_replay/work/cube_poses_cv_only.json \
REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-poses
```

### `replay-render`

- setzt den Roboter auf `observation.state[0]`;
- setzt jeden Würfel einmal vor Frame 0;
- nimmt Frame `t` auf und führt danach die unveränderte Originalaktion `action[t]` aus;
- prüft die Actions elementweise und per SHA-256;
- schreibt fünf MP4s pro Episode mit 30 fps.

Eine Abweichung der Kopfkamera-Projektion wird im schnellen Videomodus gemeldet, blockiert
aber nicht die manuelle Sichtprüfung. Der strengere Dataset-Modus bleibt bis zu einer
separaten Wrist-Kamera-Abnahme gesperrt.

## Vollständiger Testlauf

Nach erfolgreichem 60-Frame-Test:

```bash
DR_ENABLED=0 \
REPLAY_EPISODE_IDS="0" \
REPLAY_MAX_FRAMES=0 \
REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-render
```

Danach auf zehn Episoden erhöhen:

```bash
REPLAY_NUM_EPISODES=10 REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-poses

DR_ENABLED=0 REPLAY_NUM_EPISODES=10 REPLAY_OVERWRITE=1 \
./Simulation/server_rl_run.sh replay-render
```

## Variablen

| Variable | Default | Bedeutung |
|---|---:|---|
| `REPLAY_NUM_EPISODES` | `10` | Maximal verarbeitete Quellepisoden |
| `REPLAY_START_EPISODE` | `0` | Erste Episode der fortlaufenden Auswahl |
| `REPLAY_EPISODE_IDS` | leer | Explizite, leerzeichengetrennte Episoden |
| `REPLAY_GRASP_SUPPORT` | `1` | Sparse Greifpunktstütze aktivieren |
| `REPLAY_MAX_FRAMES` | `0` | `0` vollständig, sonst Videotechniktest |
| `REPLAY_OVERWRITE` | `0` | Vorhandene Ergebnisse neu erzeugen |
| `REPLAY_OUTPUT_MODE` | `videos` | `videos` oder später `dataset` |
| `REPLAY_WORK` | `/data/cube_replay/work` | Kalibrierung, Posen und Kontrollbilder |
| `REPLAY_OUT` | `/data/cube_replay/videos` | MP4-Ausgabe |
| `REPLAY_DATASET` | `/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset` | Quelle |

`REPLAY_NUM_EPISODES=10` bedeutet maximal zehn ausgewählte Quellepisoden. Übersprungene
Episoden werden nicht durch weitere ersetzt. Bereits fertige Videos werden ohne
`REPLAY_OVERWRITE=1` beibehalten.
