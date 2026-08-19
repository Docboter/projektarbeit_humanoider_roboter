# GR00T-N1.6: ONNX/TensorRT und schnelleres Sim-Rendering

> **TL;DR:** Anleitung zur Inferenz-Optimierung: DiT-Aktionskopf per ONNX/TensorRT beschleunigen
> (`GROOT_INFERENCE_BACKEND`) und Kamerarendering per `CAMERA_RENDER_EVERY_N` auf den Action-Chunk
> synchronisieren. Für alle, die Latenz und Durchsatz der Closed-Loop-Sim-Eval verbessern wollen;
> enthält die vollständigen IKR-Server-Kommandos.
> **N1.6-only.** Der `compile`/`tensorrt`-Pfad exportiert den N1.6-DiT-Aktionskopf
> (`Gr00tN1d6ActionHead`) über ONNX; `groot_inference_backend.py`,
> `optimize_groot_inference.py` und `run_groot_optimized_server.py` prüfen explizit auf diese
> Klasse. Für `GROOT_VERSION=1.7` (16-Layer-DiT statt 32) gibt es noch keine Portierung —
> `entrypoint_sim.sh` fällt dort automatisch auf `eager` zurück. Details:
> [groot-n17-migration.md §5](../weiterfuehrend/groot-n17-migration.md#stand-der-umsetzung-2026-08-19).

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

## Latenz-Messung und Denoising-Schritte

*Verschoben aus [live-ansicht.md](live-ansicht.md) (2026-08-18) — reine Modell-Performance-Analyse
ohne Sim-Bezug, passt hier besser neben die anderen Inferenz-Optimierungen.*

### Für echte Hardware zählt eine andere Zahl

Auf einem realen Roboter entfällt das Rendering ersatzlos — die Kameras liefern ihre Bilder
selbst. Übrig bleiben die 5 %: die Zeit vom Observation-Dict bis zum Action-Chunk. Die misst
[`policy_latency.py`](../../Simulation/scripts/policy_latency.py) isoliert, in-process, ohne
Sim und ohne ZMQ:

```bash
HF_TOKEN=hf_... ./Simulation/server_rl_run.sh latency
```

Maßstab ist das Chunk-Budget: ein Aufruf deckt `EXECUTION_HORIZON` Schritte ab, bei 8 Schritten
und 30 Hz also 267 ms. Berichtet werden Mittel, Median, p95 und **Maximum** — im Echtzeitbetrieb
ist der schlechteste Aufruf die relevante Zahl, weil ein einzelner Ausreißer über dem Budget
eine Lücke in der Aktionsfolge bedeutet.

**Messung 2026-08-13** (RTX PRO 6000 Blackwell, 50 Aufrufe, 4 Kameras à 640×480):

| Bedingung | Mittel | Spanne | Budget-Auslastung (schlechtester Aufruf) |
|---|---|---|---|
| **freie GPU** | **74,4 ms** | 72,1–79,5 ms | 30 % |
| GPU geteilt (zweiter GR00T-Server) | 107,8 ms | 91,4–188,2 ms | 71 % |
| über ZMQ, in der Sim gemessen | ~88 ms | — | — |

Drei Ablesungen daraus:

1. **Die Policy ist schnell genug.** 74 ms gegen 267 ms Budget heißt 3,6-fache Reserve; die
   Streuung von 7 ms macht sie zudem vorhersagbar. Ein *synchroner* 30-Hz-Regelkreis wäre mit
   13,4 Hz zwar unmöglich — genau dafür gibt es das Action-Chunking, das 16 Schritte auf
   einmal liefert.
2. **Der Transport kostet ~14 ms** (88 − 74). Der ZMQ-Weg mit 3,7 MB unkomprimierten Bildern
   je Aufruf ist lokal also kein Engpass. Über ein Netz zum Roboter wäre er einer.
3. **Fremdlast ist das eigentliche Risiko.** Eine geteilte Karte kostet nicht nur 45 % im
   Mittel, sie macht die Latenz unvorhersagbar (Spanne 97 statt 7 ms). Auf einem Roboter
   gehört die Policy auf eine dedizierte GPU — für Echtzeit zählt die Vorhersagbarkeit, nicht
   der Durchschnitt.

> **Noch offen:** dieselbe Messung auf der Zielhardware (z. B. Jetson Thor). Eine RTX PRO 6000
> mit 300 W ist keine Referenz für das, was auf dem Roboter steckt — dort ist mit einem
> Vielfachen zu rechnen, und erst dann entscheidet sich, ob `EXECUTION_HORIZON` oder die
> Modellgröße angefasst werden muss.

### Woraus die 74 ms bestehen

`--denoising-sweep` variiert `num_inference_timesteps` und trennt über die Steigung den
iterativen Aktionskopf vom festen Rest (der Backbone läuft **einmal**, vor der
Denoising-Schleife — [`gr00t_n1d6.py`](../../app/Groot-1.6/gr00t/model/gr00t_n1d6/gr00t_n1d6.py)):

| Denoising-Schritte | Latenz |
|---|---|
| 1 | 45,2 ms |
| 2 | 54,7 ms |
| 4 (Default) | 74,0 ms |

```
t(n) = 35,6 ms + n × 9,6 ms        (Vorhersage für n=2: 54,8 ms — gemessen 54,7 ms)
```

Damit ist der **Aktionskopf die größere Hälfte**: 4 × 9,6 = 38,4 ms gegen 35,6 ms für Vision,
LLM und Transformationen zusammen. Das war nicht die Erwartung — bei vier Kamerabildern durch
einen 3B-VLM hätte man den Backbone vorn vermutet — und es dreht die Rangfolge der Hebel um:

- **Für eine langsamere Zielplattform** ist `num_inference_timesteps` der erste Hebel:
  4 → 2 spart 26 % der Latenz, 4 → 1 spart 39 %.
- **35,6 ms sind die Untergrenze.** Darunter kommt man nur über den Backbone: weniger Kameras,
  kleinere Eingabe, TensorRT/FP8.
- **Heute wird nichts davon gebraucht** — 74 ms gegen 267 ms Budget. Die Zerlegung ist ein
  Planungswerkzeug für die Portierung, keine offene Baustelle.

> ⚠️ **Falle:** [`open_loop_eval.py`](../../app/Groot-1.6/gr00t/eval/open_loop_eval.py) im
> GR00T-Fork deklariert eine Option `denoising_steps`, **wendet sie aber nirgends an** (der
> Name kommt in der Datei genau einmal vor, in seiner eigenen Definition). Wer die
> Qualitätskosten damit misst, vergleicht zwei identische Läufe. Für eine echte Messung muss
> der Wert nach dem Laden am `action_head` gesetzt werden — so, wie es
> [`policy_latency.py`](../../Simulation/scripts/policy_latency.py) tut.

Die Differenz zwischen dieser Zahl und dem `Inferenz … ms/Aufruf` aus der Eval ist der
**Transport-Overhead** des ZMQ-Wegs (4 unkomprimierte Bilder ≈ 3,7 MB je Aufruf).

> Die Balance des Roboters hängt **nicht** an dieser Schleife: der Whole-Body-Controller läuft
> entkoppelt mit eigener, deutlich höherer Rate und folgt nur Geschwindigkeitsbefehlen der VLA
> ([lokomotion-recherche.md §3.1](../weiterfuehrend/lokomotion-recherche.md)). Eine zu langsame
> Policy erzeugt ruckelige Bewegung, keinen Sturz.

## Wichtige Grenzen

- TensorRT-Pläne sind nicht portabel. Bei anderem Checkpoint, GPU-Modell, Compute Capability
  oder anderer TensorRT-Version verweigert der Server den Start.
- Der WebRTC-Viewport erhält im synchronen Modus nur an Replanning-Grenzen ein neues Bild.
  Die simulierte Bewegung kann dadurch schneller, der Stream aber weniger flüssig werden.
- Die tatsächliche Beschleunigung auf der RTX PRO 6000 Blackwell ist erst durch den auf dem
  Server erzeugten `benchmark_report.json` belegt.
