# Physikbasierte Replay-Videos aus dem Realdatensatz

Dieser Workflow verankert reale Würfelpixel an den simulierten Finger-Schließpunkten und
spielt danach die originalen 28-DoF-Actions in Isaac Lab ab. Die Kalibrierung verfolgt
Farbwürfel offline im Realvideo; im finalen Replay werden sie genau einmal gesetzt. Danach
verändert nur PhysX ihre Pose, ohne Tracking oder Attach.

## Schnellstart mit höchstens zehn Episoden

Auf dem Simulationsserver im Repository:

```bash
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh replay-prepare
REPLAY_NUM_EPISODES=10 ./Simulation/server_rl_run.sh replay-calibrate
REPLAY_NUM_EPISODES=10 ./Simulation/server_rl_run.sh replay-poses
DR_ENABLED=0 REPLAY_OUTPUT_MODE=videos REPLAY_NUM_EPISODES=10 \
    ./Simulation/server_rl_run.sh replay-render
```

Die Videos liegen auf dem Host standardmäßig unter:

```text
$RL_HOST_DATA_DIR/cube_replay/videos/
```

Auf dem IKR-Server empfiehlt sich in `.env.local` der benutzerunabhängige Eintrag
`: "${RL_HOST_DATA_DIR:=$HOME/project/data/RL}"`. Das Skript vergleicht diesen Pfad mit
dem tatsächlichen `/data`-Mount eines vorhandenen Containers und bricht bei einem
Widerspruch mit beiden Pfaden ab. Dadurch können Logs und Ergebnisse nicht unbemerkt in
verschiedenen Benutzerverzeichnissen landen.

Je Episode entstehen fünf MP4s: beide Kopfkameras, beide Wrist-Kameras und `scene`.
Kalibrierungsdaten und Kontrollbilder liegen getrennt unter `cube_replay/work/`.

`replay-calibrate` ist jetzt GPU-basiert: Es spielt die Originalaktionen mit ausgelagerten
Würfeln ab, misst die Fingerkuppen und verbindet eindeutige Schließereignisse mit dem
Bewegungsbeginn eines Farbblocks im Realvideo. Actions werden nur gelesen und per SHA-256
vor und nach dem Lauf auf Unverändertheit geprüft.

Aus zeitgleichen Annäherungsframes werden außerdem die link-relativen Wrist-Kameraposen
optimiert. Abschließend rendert Isaac bekannte Markerpositionen. Die Kalibrierung gilt nur
bei höchstens 5 px Median und 10 px p90 als erfolgreich.

## Kleine Tests und gezielte Episoden

Nur 60 Frames einer Episode rendern:

```bash
REPLAY_NUM_EPISODES=1 REPLAY_MAX_FRAMES=60 \
    ./Simulation/server_rl_run.sh replay-render
```

Nach erfolgreicher Sichtprüfung einen trainierbaren Datensatz erzeugen:

```bash
REPLAY_OUTPUT_MODE=dataset REPLAY_NUM_EPISODES=10 \
    ./Simulation/server_rl_run.sh replay-render
```

Der Dataset-Modus schreibt LeRobot v2.1 mit vier Policy-Kameras, tatsächlichem Sim-State
und unveränderten Original-Actions nach `/data/cube_replay/dataset`. Metadaten, numerische
Statistiken, ursprüngliche Task-IDs und das Replay-Manifest werden mitgeführt. Teil-Episoden
via `REPLAY_MAX_FRAMES` sind in diesem Modus absichtlich verboten. Eine unzureichende
Wrist-Kalibrierung warnt im Videomodus, sperrt aber bewusst den Dataset-Modus.

Bestimmte Episoden verwenden:

```bash
REPLAY_EPISODE_IDS="0 12 41" ./Simulation/server_rl_run.sh replay-calibrate
REPLAY_EPISODE_IDS="0 12 41" ./Simulation/server_rl_run.sh replay-poses
REPLAY_EPISODE_IDS="0 12 41" ./Simulation/server_rl_run.sh replay-render
```

`REPLAY_EPISODE_IDS` überschreibt `REPLAY_NUM_EPISODES` und `REPLAY_START_EPISODE`.
Ohne explizite IDs werden ab `REPLAY_START_EPISODE=0` höchstens zehn Episoden verarbeitet.
Übersprungene Episoden werden nicht durch weitere ersetzt. Vorhandene Videos und Posen
werden beibehalten; `REPLAY_OVERWRITE=1` erzeugt sie neu.

## Verhalten und Grenzen

- Der vollständige Datensatz wird nach LeRobot v2.1 konvertiert. Der eingecheckte Ordner
  `data/G1_Dex3_BlockStacking` enthält nur Metadaten.
- Eine Episode wird nicht gerendert, wenn nicht alle drei Farben in beiden Kopfkameras
  zuverlässig erkannt werden.
- Die Würfelkante ist global 5 cm. XY stammt aus trajektorienverankerten Homographien;
  gegriffene Würfel verwenden ihren eindeutigen Bewegungsanker, sofern Bild und Anker
  höchstens 3 cm auseinanderliegen.
- Die gemeinsame Würfelhöhe stammt aus dem robusten Median der Fingerkuppenanker; mehr als
  4 cm p90-p10-Streuung oder eine unplausible Höhe bricht die Kalibrierung ab.
- Mindestens acht räumlich verteilte Anker sind erforderlich. Reichen zehn Episoden nicht,
  wird abgebrochen und die Kalibrierung mit zwanzig Episoden wiederholt.
- `replay-calibrate` prüft bekannte Positionen gegen den tatsächlichen Isaac-Renderer.
  Beim Rendern werden Kopfkameras in Frame 0 und Wrist-Kameras in den jeweils zeitgleichen
  Trajektorienframes erneut geprüft.
- Der Replay läuft mit `dt=1/210 s` und sieben Physics-Schritten je Frame, also exakt 30 Hz.
- Ein Aufgabenerfolg löst im Replay keinen Auto-Reset aus. Die Originalepisode läuft bis
  zu ihrem Ende.
- Ob ein Griff gelingt, wird in dieser Version bewusst nur durch Sichtprüfung beurteilt.

## Variablen

| Variable | Default | Bedeutung |
|---|---:|---|
| `REPLAY_NUM_EPISODES` | `10` | Maximal verarbeitete Quell­episoden |
| `REPLAY_START_EPISODE` | `0` | Erste Episode der fortlaufenden Auswahl |
| `REPLAY_EPISODE_IDS` | leer | Explizite, leerzeichengetrennte Episoden |
| `REPLAY_MAX_FRAMES` | `0` | `0` vollständig, sonst Techniktest |
| `REPLAY_OVERWRITE` | `0` | Vorhandene Ergebnisse neu erzeugen |
| `REPLAY_OUTPUT_MODE` | `videos` | `videos` oder trainierbarer `dataset` |
| `REPLAY_WORK` | `/data/cube_replay/work` | Kalibrierung, Posen und Kontrollbilder |
| `REPLAY_OUT` | `/data/cube_replay/videos` | MP4-Ausgabe |
| `REPLAY_DATASET_OUT` | `/data/cube_replay/dataset` | LeRobot-v2.1-Ausgabe |
| `REPLAY_DATASET` | `/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset` | Quelldatensatz |
