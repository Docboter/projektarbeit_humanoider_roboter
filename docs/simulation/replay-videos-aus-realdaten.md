# Physikbasierte Replay-Videos aus dem Realdatensatz

Dieser Workflow bestimmt die drei Würfelposen aus den realen Kopfkameras und spielt danach
die originalen 28-DoF-Actions in Isaac Lab ab. Die Würfel werden pro Episode genau einmal
gesetzt. Danach verändert nur PhysX ihre Pose; es gibt kein Tracking und keinen Attach.

## Schnellstart mit höchstens zehn Episoden

Auf dem Simulationsserver im Repository:

```bash
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh replay-prepare
REPLAY_NUM_EPISODES=10 ./Simulation/server_rl_run.sh replay-calibrate
REPLAY_NUM_EPISODES=10 ./Simulation/server_rl_run.sh replay-poses
DR_ENABLED=0 REPLAY_NUM_EPISODES=10 ./Simulation/server_rl_run.sh replay-render
```

Die Videos liegen auf dem Host standardmäßig unter:

```text
$RL_HOST_DATA_DIR/cube_replay/videos/
```

Je Episode entstehen fünf MP4s: beide Kopfkameras, beide Wrist-Kameras und `scene`.
Kalibrierungsdaten und Kontrollbilder liegen getrennt unter `cube_replay/work/`.

## Kleine Tests und gezielte Episoden

Nur 60 Frames einer Episode rendern:

```bash
REPLAY_NUM_EPISODES=1 REPLAY_MAX_FRAMES=60 \
    ./Simulation/server_rl_run.sh replay-render
```

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
- Die Würfelkante ist global 5 cm. Ihre gemessene Bildgröße kalibriert die Brennweite;
  die gemeinsame Tischhöhe folgt aus der Stereo-Triangulation beider Kopfkameras. Größe
  und Tischhöhe werden nicht je Episode verändert.
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
| `REPLAY_WORK` | `/data/cube_replay/work` | Kalibrierung, Posen und Kontrollbilder |
| `REPLAY_OUT` | `/data/cube_replay/videos` | MP4-Ausgabe |
| `REPLAY_DATASET` | `/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset` | Quelldatensatz |
