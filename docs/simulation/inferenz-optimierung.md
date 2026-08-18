# GR00T-N1.6: ONNX/TensorRT und schnelleres Sim-Rendering

Die Optimierung hat zwei getrennte Teile:

1. Der iterative **DiT-Aktionskopf** wird nach ONNX exportiert und als GPU-spezifische
   TensorRT-Engine ausgeführt. Vision-/Sprach-Backbone, Processor und Action-Decoding bleiben
   PyTorch. Eine ONNX-Datei allein ist kein ausführbares Programm.
2. Mit `CAMERA_RENDER_EVERY_N=EXECUTION_HORIZON` rendert Isaac Sim die vier Policy-Kameras
   nur direkt vor einer neuen Policy-Anfrage. Die Bilder an diesen Zeitpunkten bleiben
   640×480; ungenutzte Zwischenframes entfallen.

Der historische Default bleibt `GROOT_INFERENCE_BACKEND=eager` und
`CAMERA_RENDER_EVERY_N=1`.

## Backend auswählen

Das Backend wird beim Aufruf von `server_rl_run.sh eval` über
`GROOT_INFERENCE_BACKEND` gewählt. ONNX ist dabei nur das Zwischenformat für den
TensorRT-Build und kein eigenes Laufzeit-Backend.

| Wert | Ausführung | Voraussetzung |
|---|---|---|
| `eager` | bisheriger PyTorch-DiT; Referenz und Default | keine Optimierungsartefakte |
| `compile` | derselbe DiT mit `torch.compile(mode="max-autotune")` | aktuelles Sim-Image; erster Start kompiliert und dauert länger |
| `tensorrt` | DiT über die BF16-TensorRT-Engine | vorher `server_rl_run.sh optimize all` |

```bash
# Referenzpfad
GROOT_INFERENCE_BACKEND=eager ./Simulation/server_rl_run.sh eval

# PyTorch-Compiler
GROOT_INFERENCE_BACKEND=compile ./Simulation/server_rl_run.sh eval

# Schnellmodus: TensorRT plus nur ein Kamerarendering je ausgeführtem Action-Chunk
GROOT_INFERENCE_BACKEND=tensorrt CAMERA_RENDER_EVERY_N=8 SCENE_CAM=0 \
    ./Simulation/server_rl_run.sh eval
```

Ohne Angabe gilt immer `eager`. Die Wahl kann alternativ für mehrere Aufrufe exportiert
werden: `export GROOT_INFERENCE_BACKEND=compile`. `CAMERA_RENDER_EVERY_N` ist unabhängig
vom Inferenz-Backend; Werte größer als 1 müssen exakt `EXECUTION_HORIZON` entsprechen.
Bei TensorRT wird die Engine automatisch anhand des Checkpoint-Fingerprints gefunden.
`GROOT_TRT_ENGINE_PATH=/data/optimized/.../dit_model_bf16.trt` überschreibt den Pfad
explizit. Eine fehlende oder inkompatible Engine führt bewusst zum Abbruch, nicht zu einem
stillen Eager-Fallback.

## Einmaliges Setup auf dem IKR-Server

```bash
ssh mspaeth@ikr-ki-server-01
cd ~/projektarbeit_humanoider_roboter
git pull

export HF_TOKEN
export RL_HOST_DATA_DIR="$HOME/project/data/RL"
export RL_NETWORK_MODE=host
export LIVESTREAM_HOST_ADDR="$(hostname -I | awk '{print $1}')"
export RL_SKIP_PULL=1

mkdir -p "$RL_HOST_DATA_DIR"
./Simulation/update_sim_image.sh --vastai --skip-push
./Simulation/server_rl_run.sh preflight
./Simulation/server_rl_run.sh clean
./Simulation/server_rl_run.sh setup
./Simulation/server_rl_run.sh livecheck
```

`clean` entfernt die Workbench-Container, nicht `$RL_HOST_DATA_DIR`.

## Engine bauen und automatisch testen

```bash
./Simulation/server_rl_run.sh optimize all
```

`all` führt diese Schritte aus:

- ONNX-Export mit den echten vier-Kamera-Sim-Shapes,
- TensorRT-BF16-Build auf der aktuellen GPU,
- Checkpoint-/GPU-/Runtime-Fingerprint,
- Aktionsparität und Policy-Latenz für Eager, `torch.compile` und TensorRT,
- kurzer Closed-Loop-A/B-Lauf: bisheriger Renderpfad gegen TensorRT + synchrones Rendering.

Artefakte liegen unter:

```text
$RL_HOST_DATA_DIR/optimized/<checkpoint-fingerprint>/
├── dit_model.onnx
├── dit_model.onnx.data          # falls ONNX externe Gewichte verwendet
├── dit_model_bf16.trt
├── export_metadata.json
├── benchmark_report.json
├── sim_baseline_eager.json
└── sim_optimized_tensorrt.json
```

Ein verfehltes 2×-Durchsatzziel wird im Report markiert, verwirft aber keine funktionsfähige
Engine. Eine fehlgeschlagene numerische Parität bricht dagegen ab.

Einzelphasen sind ebenfalls möglich:

```bash
./Simulation/server_rl_run.sh optimize export
./Simulation/server_rl_run.sh optimize build
./Simulation/server_rl_run.sh optimize validate
./Simulation/server_rl_run.sh optimize benchmark
```

## Optimierte Eval mit Webstream

Browser-Viewer in einem zweiten Terminal:

```bash
cd ~/projektarbeit_humanoider_roboter
export RL_HOST_DATA_DIR="$HOME/project/data/RL"
export LIVESTREAM_HOST_ADDR="$(hostname -I | awk '{print $1}')"
./Simulation/server_rl_run.sh webview
```

Dann `http://<IKR-SERVER-IP>:8210/` in Chrome/Chromium/Edge öffnen und im ersten Terminal:

```bash
GROOT_INFERENCE_BACKEND=tensorrt \
CAMERA_RENDER_EVERY_N=8 \
SCENE_CAM=0 \
LIVESTREAM=2 \
NUM_EPISODES=2 \
EPISODE_LENGTH_S=120 \
./Simulation/server_rl_run.sh eval
```

`CAMERA_RENDER_EVERY_N` muss im vergleichbaren schnellen Modus exakt
`EXECUTION_HORIZON` entsprechen. Die TensorRT-Engine wird automatisch über den
Checkpoint-Fingerprint gefunden. Ein expliziter Pfad ist mit `GROOT_TRT_ENGINE_PATH` möglich.

## Wichtige Grenzen

- TensorRT-Pläne sind nicht portabel. Bei anderem Checkpoint, GPU-Modell, Compute Capability
  oder anderer TensorRT-Version verweigert der Server den Start.
- Der WebRTC-Viewport erhält im synchronen Modus nur an Replanning-Grenzen ein neues Bild.
  Die simulierte Bewegung kann dadurch schneller, der Stream aber weniger flüssig werden.
- Die tatsächliche Beschleunigung auf der RTX PRO 6000 Blackwell ist erst durch den auf dem
  Server erzeugten `benchmark_report.json` belegt.
