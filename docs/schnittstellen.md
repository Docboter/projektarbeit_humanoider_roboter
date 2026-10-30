# Schnittstellen

> **TL;DR:** Alle Schnittstellen des Projekts auf einer Seite, aus dem Code abgeleitet:
> Datensatz, Checkpoint, Policy-Server (ZMQ), Container und Konfiguration, Sim-Eval mit
> Metriken, Hilfswerkzeuge und was für den echten Roboter fehlt. Jede Angabe nennt die Datei,
> aus der sie stammt. Bedienung steht in der [Quickstart-Seite](quickstart.md), Messergebnisse
> stehen in der [Diagnose-Chronik](ergebnisse/diagnose-chronik.md).

Stand: 2026-10-29, Branch `training-luca-IKR-IS6.0`. Zeilennummern beziehen sich auf diesen
Stand; bei Abweichung gilt der Symbolname.

```
HF-Datensatz ──► Training (Docker / KISSKI) ──► Checkpoint (HF luca-mue/groot-g1dex3-*)
 (LeRobot v3.0)   entrypoint.sh                        │
                                                       ▼
                     Sim-Eval (Isaac Lab) ◄── ZMQ ──► Policy-Server (run_gr00t_server.py)
                     oder Roboter (fehlt)   :5555
```

## 1. Datensatz

| Was | Wert | Quelle |
|---|---|---|
| HF-Repo | `unitreerobotics/G1_Dex3_BlockStacking_Dataset`, LeRobot v3.0, fest im Code | `Training/scripts/download_data.sh:119-124` |
| Ablage im Container | `/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset` | `download_data.sh:119-124` |
| Konvertierung v3.0 → v2.1 | `scripts/lerobot_conversion/convert_v3_to_v2_standalone.py` im GR00T-Baum, schreibt an denselben Pfad, Backup unter `<pfad>_v30`; übersprungen bei `SKIP_CONVERT=1` oder wenn `meta/modality.json` existiert | `Training/scripts/entrypoint.sh:209-218`, Docstring des Skripts |
| `meta/modality.json` | immer Kopie von `examples/G1_DEX3/modality_4cam.json` (die `modality_2cam.json` wird von keinem Trainingsskript gewählt) | `entrypoint.sh:219-220`, `run_finetuning.sh:118-120` |
| Embodiment | Tag `NEW_EMBODIMENT`, Modalitäten aus `examples/G1_DEX3/g1_dex3_config.py` (für N1.6 und N1.7 identisch) | `run_finetuning.sh:31-33,183-184`, `g1_dex3_config.py:29-87` |
| Sprache | Schlüssel `annotation.human.task_description` (in `modality.json` auf `task_index` abgebildet) | `g1_dex3_config.py:83-86`, `modality_4cam.json:52-56` |

**Kameras** (`modality_4cam.json:38-51`): `cam_left_high`, `cam_right_high`, `cam_left_wrist`,
`cam_right_wrist`, jeweils aus `observation.images.<name>`. Die Bilder sind 640 × 480 RGB
(Datensatz-Referenzbilder `Simulation/camera_reference/dataset_cam_*.png`; Sim-Kameras gleich:
`Simulation/g1_dex3_sim/camera_geometry.py:84-85`).

**Zustand und Aktion**: je 28 Werte, gleiche Aufteilung (`modality_4cam.json:2-37`,
Gelenknamen im Kommentar `g1_dex3_config.py:11-23`):

| Schlüssel | Index | Gelenke | Aktion |
|---|---|---|---|
| `left_arm` | 0:7 | ShoulderPitch/Roll/Yaw, Elbow, WristRoll/Pitch/Yaw | relativ (Delta zum Zustand) |
| `right_arm` | 7:14 | wie links | relativ |
| `left_dex3` | 14:21 | Thumb0/1/2, **Middle0/1, Index0/1** | absolut |
| `right_dex3` | 21:28 | Thumb0/1/2, **Index0/1, Middle0/1** | absolut |

Die Finger-Reihenfolge ist links und rechts verschieden (Middle/Index vertauscht). Die
Gelenkreihenfolge der Sim-USD ist eine andere; die Abbildung steht in
`Simulation/g1_dex3_sim/isaac_joint_order.txt`. Aktionsfenster: 16 Schritte
(`delta_indices=range(16)`), Zustand und Bild: nur der aktuelle Schritt
(`g1_dex3_config.py:30-49`). Relativ/absolut: `g1_dex3_config.py:56-80`.

**Train/Test-Split** (`Training/scripts/lib_split.sh:32-75`): Bei `TRAIN_TEST_SPLIT=1` gilt
`n_train = int(total_episodes × TRAIN_SPLIT_RATIO)`, eingetragen als
`info.json["splits"] = {"train": "0:n", "test": "n:total"}`, protokolliert in
`<OUTPUT_DIR>/split.json`. Mit 301 Episoden und Ratio 0,8 ergibt das **0:240 / 240:301**.
Default ist `TRAIN_TEST_SPLIT=0` (alle Episoden im Training), außer bei `USE_COTRAIN=1`
(`entrypoint.sh:116-118`).

## 2. Checkpoint

| Was | Wert | Quelle |
|---|---|---|
| Pfad im Trainings-Container | `/data/g1_dex3_finetune/<lauf>/<experiment>/checkpoint-<N>`; `<lauf>` = `blockstacking`, `blockstacking_vision` oder `blockstacking_cotrain`, bei N1.7 mit Suffix `_n17`; `<experiment>` z. B. `g1_dex3_blockstacking_v1` | `run_finetuning.sh:30-31`, `run_finetuning_vision.sh:40-41`, `run_finetuning_cotrain.sh:46-47`; `app/Groot-1.6/gr00t/experiment/experiment.py:127-131` |
| Speichertakt | lokal alle 2000 Steps, höchstens 15 behalten; KISSKI 5000 / 40 | `run_finetuning.sh:54-55`, `Training/kisski_submit.sh:184-185` |
| Inhalt | `config.json`, Gewichte (safetensors), `experiment_cfg/` (`config.yaml`, `conf.yaml`), `processor/`, `wandb_config.json` (`project`, `run_id`), `trainer_state.json`, Optimizer-Zustand | `experiment.py:137-150`; Rest: Standard des HF-Trainers |
| Version erkennen | `model_type` in `config.json`: `Gr00tN1d6` → 1.6, `Gr00tN1d7` → 1.7. `GROOT_VERSION=auto` liest das aus (Fallback 1.6 mit Warnung); eine explizit falsche Version erzeugt nur eine Warnung | `Training/scripts/lib_groot_version.sh:110-146` (wortgleich `Simulation/scripts/lib_groot_version.sh`) |
| Nicht kreuzweise ladbar | N1.6: Backbone `nvidia/Eagle-Block2A-2B-v2`, Zustand/Aktion bis 29 Dim., Horizont 16. N1.7: Backbone `nvidia/Cosmos-Reason2-2B` (Qwen3-VL, gated, wird bei jedem Laden vom Hub geholt), 132 Dim., Horizont 40 | `app/Groot-1.6/gr00t/configs/model/gr00t_n1d6.py:21,57-59`, `app/Groot-1.7/gr00t/configs/model/gr00t_n1d7.py:31,40,74-76`, `lib_groot_version.sh:33-37` |
| Hochladen | `python Simulation/scripts/upload_checkpoint.py --checkpoint <dir> --repo <hf-repo>`; Repo immer privat; ohne `--include-optimizer` fallen `optimizer.pt*`, `scheduler.pt`, `rng_state*.pth`, `training_args.bin`, DeepSpeed-Zustände weg | `upload_checkpoint.py:41-50,89-105` |
| HF-Namen | Schema `luca-mue/groot-g1dex3-<lauf>-ckpt<N>` (nur durch Beispiele belegt, nicht erzwungen); Default der Sim `luca-mue/groot-g1dex3-checkpoint` → `/data/checkpoints/groot-g1dex3-checkpoint` | `Simulation/server_rl_run.sh:223-224,2373-2374` |

Trotz Horizont 40 im N1.7-Modell liefert der Server für G1_DEX3 bei beiden Versionen 16
Schritte: Die Embodiment-Konfiguration legt das Fenster fest, und der Sim-Client bricht bei
einer anderen Form ab (`Simulation/g1_dex3_sim/client.py:186-190`). Die N1.7-Läufe 54/55 und
81 liefen mit diesem Client.

## 3. Policy-Server

Der Server lädt einen Checkpoint und beantwortet Anfragen synchron über ZMQ. Mit den Angaben
hier lässt sich ein eigener Client schreiben (Referenz: `Simulation/g1_dex3_sim/client.py`).

| Was | Wert | Quelle |
|---|---|---|
| Start | `python gr00t/eval/run_gr00t_server.py --model-path <ckpt> --embodiment-tag NEW_EMBODIMENT --use-sim-policy-wrapper --port 5555` | `Simulation/scripts/entrypoint_sim.sh:362-370` |
| Weitere Optionen | `--host` (0.0.0.0), `--device` (cuda), `--strict` (True); kein Argument für Denoising-Schritte | `app/Groot-1.6/gr00t/eval/run_gr00t_server.py:15-50` |
| Optimierte Variante (nur N1.6) | `Simulation/scripts/run_groot_optimized_server.py --backend eager\|compile\|tensorrt`, gleiches Protokoll | `entrypoint_sim.sh:371-382`, `run_groot_optimized_server.py:21-29` |
| Transport | ZMQ REP auf `tcp://<host>:5555`, Client REQ; streng eine Anfrage, dann eine Antwort | `app/Groot-1.6/gr00t/policy/server_client.py:70-71`, `client.py:103` |
| Kodierung | msgpack. numpy-Arrays als `{"__ndarray_class__": True, "as_npy": <np.save-Bytes>}`. Der N1.7-Server antwortet im msgpack-numpy-Format `{nd, type, kind, shape, data}`, nimmt aber beide Formate an | `server_client.py:15-45`; `app/Groot-1.7/gr00t/policy/server_client.py:34-106`; `client.py:30-71` |
| Anfrage | `{"endpoint": <name>, "data": {...}}`; Endpunkte `ping`, `reset`, `get_action`, `get_modality_config`, `kill` | `server_client.py:76-84,204-208` |
| Fehler | Antwort `{"error": "<text>"}` | `server_client.py:149` |
| Authentifizierung | optional `api_token`; im Projekt nicht genutzt | `server_client.py:109-115` |

**Beobachtung** (`data = {"observation": obs, "options": None}`, `client.py:208-248`):

| Schlüssel | Form, Typ |
|---|---|
| `video.cam_left_high`, `video.cam_right_high`, `video.cam_left_wrist`, `video.cam_right_wrist` | `(1, 1, 480, 640, 3)` uint8, RGB (Batch, Zeit, H, W, C) |
| `state.left_arm`, `state.right_arm`, `state.left_dex3`, `state.right_dex3` | `(1, 1, 7)` float32, Gelenkwinkel aus Abschnitt 1 |
| `annotation.human.task_description` | Tupel mit einem String, kein Array |

Prompt: `TASK_DESCRIPTION` → sonst `meta/tasks.*` des Datensatzes → sonst
`"stack three block"` (`Simulation/g1_dex3_sim/task_text.py:23-81`). Der Server prüft
Schlüssel, Typen und Dimensionen streng (`app/Groot-1.6/gr00t/policy/gr00t_policy.py:452-530`).

**Antwort** auf `get_action`: Liste `[action_dict, info_dict]`. `action_dict` enthält
`action.left_arm`, `action.right_arm`, `action.left_dex3`, `action.right_dex3`, je
`(1, 16, 7)` float32. Der Server rechnet die relativen Arm-Aktionen mit dem mitgeschickten
Zustand in **absolute Gelenkziele** um (`gr00t_policy.py:350-352`); der Client erhält also 16
absolute Ziele × 28 Gelenke (`client.py:174-190`).

**Ausführung in der Sim** (`Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py`):

- Takt 30 Hz: Physik `dt = 1/210 s`, 7 Teilschritte je Policy-Schritt
  (`g1_dex3_blockstack_env.py:399,407-413`).
- Von den 16 Schritten werden `EXECUTION_HORIZON` (Default 8) ausgeführt, dann wird neu
  angefragt (`run_g1_dex3_sim_eval.py:53,480-483`).
- Latenzbudget = `1000 × EXECUTION_HORIZON / 30` ms, bei 8 also 267 ms; gemeldet je Episode als
  `inference_ms.budget_ms` (`:545,597-602`). Die Sim wartet auf die Antwort.
- `reset` einmal je Episode (`:325`). Timeout 20 s je Anfrage (`:745`), `ping` mit
  Wiederholungen (`--ping-retries`, Entrypoint: 20). `get_action` wird nicht wiederholt: Ein
  Fehler markiert die Episode als ungültig (`:467-473`). Nach einem Timeout baut der Client den
  REQ-Socket neu auf, sonst ist er blockiert (`client.py:110-135`).

Minimaler Client (nach `client.py`, N1.6-Antwortformat; am 2026-10-29 gegen den `PolicyServer`
aus `app/Groot-1.6/gr00t/policy/server_client.py` mit einer Dummy-Policy geprüft, Ergebnis
`(16, 28)` float32):

```python
import io, msgpack, numpy as np, zmq

def enc(o):
    if isinstance(o, np.ndarray):
        b = io.BytesIO(); np.save(b, o, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": b.getvalue()}
    raise TypeError(type(o))

def dec(o):
    if isinstance(o, dict) and "__ndarray_class__" in o:
        return np.load(io.BytesIO(bytes(o["as_npy"])), allow_pickle=False)
    return o

s = zmq.Context().socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 20000)
s.connect("tcp://localhost:5555")
def call(req):
    s.send(msgpack.packb(req, default=enc))
    r = msgpack.unpackb(s.recv(), object_hook=dec)
    if isinstance(r, dict) and "error" in r: raise RuntimeError(r["error"])
    return r

call({"endpoint": "ping"}); call({"endpoint": "reset", "data": {}})
cams = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist")
obs = {f"video.{c}": np.zeros((1, 1, 480, 640, 3), np.uint8) for c in cams}
obs |= {f"state.{g}": np.zeros((1, 1, 7), np.float32)
        for g in ("left_arm", "right_arm", "left_dex3", "right_dex3")}
obs["annotation.human.task_description"] = ("stack three block",)
act, _ = call({"endpoint": "get_action", "data": {"observation": obs, "options": None}})
chunk = np.concatenate([act[f"action.{k}"][0] for k in
                        ("left_arm", "right_arm", "left_dex3", "right_dex3")], -1)  # (16, 28)
```

## 4. Container und Konfiguration

| Image | Tags | Basis | Quelle |
|---|---|---|---|
| Training `<ns>/projekt-humanoider-roboter` | `:latest` (N1.6), `:latest-n17` (N1.7), `:latest-n16-n17` (beide); außerdem `:<branch>-<gen>`, `:<zeitstempel>-<gen>`. `latest*` nur mit `--push-latest` | `nvidia/cuda:12.8.0-devel-ubuntu22.04` | `Training/update_image.sh:150-152,270-292`, `Training/Dockerfile:38` |
| Sim `<ns>/projekt-humanoider-roboter-sim-standalone` | `:<branch>`, `:<zeitstempel>`, `:latest` (mit `--push-latest`) | `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1` (Isaac Sim 6.0) | `Simulation/update_sim_image.sh:53,144-158`, `Simulation/Dockerfile.standalone:48` |

`<ns>` ist der Docker-Hub-Namensraum: Die Build-Skripte fragen ihn ab (`update_image.sh:161-168`),
die Launcher fallen auf `lucam03` zurück (`server_rl_run.sh:130`).

| Einstieg | Zweck | Quelle |
|---|---|---|
| `./run.sh [domäne] [aktion]` | Menü; wählt nur aus und übergibt an einen der Launcher unten, `MENU=0` schaltet ab | `run.sh:1-16`, `tools/menu/_domains.spec` |
| `Training/setup_and_train_dockerhub_pull.sh` | Image ziehen, Container `groot-train` starten (ohne `--rm`); `--resume`, `--skip-pull` | Skriptkopf `:1-22` |
| `/scripts/entrypoint.sh` (im Image) | Download (`.done`-Marken in `/data/.download_state/`) → Konvertierung → Training: `run_finetuning.sh`, bei `TUNE_VISUAL=1` `run_finetuning_vision.sh`, bei `USE_COTRAIN=1` `run_finetuning_cotrain.sh` | `Training/scripts/entrypoint.sh:192-289` |
| `sbatch --export=ALL Training/kisski_submit.sh` | KISSKI-Job: Apptainer, 4 × A100, 48 h, Log `logs/slurm-<id>.out`; bindet `<projekt>/data` als `/data` und `Training/scripts` als `/scripts` | `kisski_submit.sh:54-63,77-121,338-388` |
| `Training/kisski_chain.sh` | Läufe aus Vorlagen hintereinander einreihen (Training, danach Open-Loop-Sweep) | `kisski_chain.sh:5-18` |
| `Simulation/server_rl_run.sh <aktion>` | Sim-Werkbank: ein langlebiger Container `groot-rl`, jede Aktion per `docker exec`; Aktionen siehe Abschnitt 5 | `server_rl_run.sh:130-137,2176,2512-2541` |

**Datenablage**

| Ort | Inhalt | Quelle |
|---|---|---|
| Training `/data` (im Container, KISSKI: `/mnt/vast-kisski/projects/kisski-humrob/data`) | `models/<basismodell>`, `unitreerobotics/<datensatz>`, `g1_dex3_finetune/` (Checkpoints, W&B), `logs/`, `hf_cache/`, `.download_state/` | `entrypoint.sh:175-187`, `kisski_submit.sh:108` |
| Sim-Host `~/groot-rl-data` (`RL_HOST_DATA_DIR`) = `/data` im Sim-Container | `checkpoints/`, `eval/` (Layouts), `sim_results/`, `sim_videos/`, `cotrain/`, `logs/`, `hf_cache/`, `optimized/`, `g1_dex3_rl/` | `server_rl_run.sh:137,511,545` |

**Wichtigste Variablen** (vollständig: [training/env-vars.md](training/env-vars.md); Sim:
`tools/menu/sim-*.spec`):

| Variable | Default | Quelle |
|---|---|---|
| `HF_TOKEN` | Pflicht | `entrypoint.sh:151-153` |
| `GROOT_VERSION` | `1.6` (`auto` liest den Checkpoint) | `Training/Dockerfile:81`, `lib_groot_version.sh:128-139` |
| `MAX_STEPS` | 20000 (Launcher lokal 30000, KISSKI 44000) | `entrypoint.sh:99` |
| `GLOBAL_BATCH_SIZE` / `NUM_GPUS` | 8 / 1 | `entrypoint.sh:100-101` |
| `LEARNING_RATE` | 1e-4 | `run_finetuning.sh:56` |
| `TUNE_VISUAL` / `USE_COTRAIN` | 0 / 0 | `entrypoint.sh:107,109` |
| `TRAIN_TEST_SPLIT` / `TRAIN_SPLIT_RATIO` | 0 (1 bei Co-Training) / 0,8 | `entrypoint.sh:116-118`, `run_finetuning.sh:66` |

## 5. Sim-Eval

### Eingaben

**Layout-JSON** (erzeugt von `Simulation/g1_dex3_sim/extract_block_layout.py:1036-1074`,
gelesen von `run_g1_dex3_sim_eval.py:207-241`):

```json
{
  "episodes": {"240": {"cubes": [[x, y], [x, y], [x, y]], "cubes_yaw_deg": [..] , "frame": 0, ...}},
  "split": "test", "n_train": 240, "total_episodes": 301, "train_split_ratio": 0.8,
  "cameras": {...}, "bias_cm": ..., "z_plane": ..., "source_dataset": "..."
}
```

- `cubes`: drei Würfel rot/grün/gelb, x/y in m im Env-Koordinatensystem; z setzt die Env auf
  Tischhöhe (`extract_block_layout.py:61`, `g1_dex3_blockstack_env.py:1090`).
- `cubes_yaw_deg`: Grad, modulo 90°; `null` = nicht sicher gemessen, die Env nimmt dann 0°
  (`extract_block_layout.py:1044-1048`, `g1_dex3_blockstack_env.py:1092`).
- Episoden werden **numerisch sortiert** durchlaufen und bei Bedarf wiederholt
  (`run_g1_dex3_sim_eval.py:208`, `g1_dex3_blockstack_env.py:1080`); Episoden mit weniger als
  drei Würfeln werden übersprungen (`:209-213`).
- Episoden ab `n_train` sind Testlagen. Fehlt `n_train`, wird es aus `meta/info.json`
  nachgerechnet (mit Warnung, `split_math.py:79-104`).
- Dateien: `/data/eval/layout_heldout.json` (Testlagen, `LAYOUT_HELD_OUT=1
  server_rl_run.sh layout`, `server_rl_run.sh:1503`); `/data/eval/layout_heldout_dy3.json`
  (Testlagen je Seite 3 cm nach außen, Korrektur aus Lauf 62; Default von
  `tools/sim_cl_watchdog.sh:43`), erzeugt mit `tools/layout_shift.py --dy 0.03`.

**Variablen der Eval** (`run_g1_dex3_sim_eval.py`, `Simulation/scripts/entrypoint_sim.sh:155-161`,
`tools/menu/sim-eval.spec:10-83`):

| Variable | Default | Bedeutung |
|---|---|---|
| `CHECKPOINT_PATH` | `/data/checkpoints/groot-g1dex3-checkpoint` | Modell des Servers |
| `NUM_EPISODES` | 20 | Episoden |
| `EPISODE_LENGTH_S` | 0 = 300 s | immer setzen (Protokoll: 120) |
| `EVAL_LAYOUT` | leer = Zufallslagen (Warnung) | Pfad zur Layout-JSON |
| `EXECUTION_HORIZON` | 8 | ausgeführte Schritte je Chunk |
| `DR_ENABLED` / `DR_SEED` | 0 (Menü) / 0 | Domain-Randomisierung, Seed |
| `SCENE_CAM`, `CAM_RES_SCALE`, `CAMERA_RENDER_EVERY_N` | 1, 1, 1 | Render-Optionen; `CAMERA_RENDER_EVERY_N > 1` nur gleich `EXECUTION_HORIZON` |

**Szenen-Variablen** (nur für Diagnosen und das Protokoll ab Lauf 70):

| Variable | Default | Protokoll (Watchdog) | Quelle |
|---|---|---|---|
| `SIM_TABLE_SIZE` | `0.8 0.6` m | `0.8 2.0` | `scene_geometry.py:20-27` |
| `SIM_CUBE_EDGE` | 0,05 m | (4-cm-Läufe: 0,04) | `scene_geometry.py:31` |
| `SIM_WRIST_EYE` / `SIM_WRIST_TARGET` | im Code | `0.02 0.0 0.08` / `0.30 0.0 -0.08` | `camera_geometry.py:228-239` |
| `SIM_WRIST_HFOV` | 60,4° | 85 | `camera_geometry.py:100` |

### Ausgaben

`/data/sim_results/results.json` (letzter Lauf) und eine Archivkopie
`results_<JJJJMMTT-HHMMSS>_<ckpt-ordner>.json` daneben, auf dem Host unter
`~/groot-rl-data/sim_results/` (`run_g1_dex3_sim_eval.py:982-1000`). Abbruch: Exit-Code 2.

- Kopf (`:943-978`): `num_episodes`, `num_valid`, `num_aborted`, `num_success`,
  `success_rate`, `milestone_ci95` (je Stufe `k`, `n`, `rate`, `ci95`), `policy`
  (`groot` / `oracle_replay`), `groot_version`, `checkpoint` (`path`, `run_id`, `global_step`),
  `cube_layout`, `domain_randomization`, `execution_horizon`, `episode_length_s`, `episodes[]`.
- Je Episode (`:582-635`, `:814`): `episode`, `error`, `error_reason`, `success`, `num_steps`, `inference_ms`,
  `min_reach_m`, `block_shift_max_m`, `finger_span_max_rad`, `max_cube_lift_cm`,
  `max_clear_lift_cm`, `layout_episode`, `milestones`.
- Episoden mit `error = true` zählen in keiner Rate (`:580-581`). 95-%-Intervalle nach
  Clopper-Pearson (`eval_metrics.py:137`).

### Metriken (`milestones`)

| Stufe | Bedingung | Quelle |
|---|---|---|
| `reached` | kleinster Abstand Fingerkuppe ↔ Würfelmitte ≤ 4 cm | `run_g1_dex3_sim_eval.py:274` |
| `grasp_attempted` | Fingerspanne des Kommandos ≥ 30 % von 2,094 rad (Demo) | `:290-291` |
| `touched` | Würfel um > 1 cm verschoben (nach 30 Einschwingschritten) | `:277` |
| `lifted` | Würfelschwerpunkt > 2 cm angehoben | `:280` |
| `lifted_clear` | Würfel-**Unterseite** > 2 cm über dem Tisch **und** im selben Schritt eine Fingerkuppe ≤ 4 cm von der Würfelmitte | `eval_metrics.py:32-97` |
| `stacked` | drei Würfel: Mitten horizontal < 3 cm auseinander, Turm > 8 cm hoch, alle < 0,05 m/s | `g1_dex3_blockstack_env.py:444-446,1385-1413` |

Die Stufen sind keine Kette; jede wird für sich ausgewertet (`run_g1_dex3_sim_eval.py:909-914`).
Grenzen:

- `lifted` zählt auch einen gekippten Würfel (Kante +1,04 cm, Ecke +1,83 cm Schwerpunkt,
  `eval_metrics.py:8-13`).
- `lifted_clear` zählt auch einen Würfel, der auf einem anderen liegt, während eine Hand nah ist
  (Lauf 94). Ein Treffer gilt erst nach Sichtung des Videos.
- Würfelwürfe (Verschiebung > 0,5 m oder Hub > 10 cm) markiert `tools/sim_cl_summary.py:9-12`
  als Artefakt.

### Visuelles Gate (`server_rl_run.sh visgate`)

Open-Loop-Vorhersage ohne Isaac Sim im GR00T-venv (`server_rl_run.sh:1341-1392`, Skript
`Simulation/scripts/finger_span_openloop.py`). Gemessen wird der Median der MSE über die
Hand-Dimensionen (`mse_hand`):

- auf Echtbildern (Trainingsepisoden `VISGATE_REAL_TRAJ_IDS = 0 4 8 12`),
- auf den gerenderten Prüfsets `/data/cotrain/vis_gate_v3_cube5` und `…cube4`
  (Positionen 0–3),
- je 600 Schritte, Seed `VISGATE_SEED` (Default 0, je Episode `seed + traj_id`,
  `finger_span_openloop.py:243-247`).

Ausgabe: `sim_results/visgate_<tag>_real.json` und `visgate_<tag>_<set>.json`,
`<tag> = <ckpt-ordner>_s<seed>` (`server_rl_run.sh:1352-1353`). Felder je Episode: `mse`,
`mse_arm`, `mse_hand`, `finger_span_pred_rad`, `finger_span_gt_rad`, `ratio`
(`finger_span_openloop.py:259-272`). Die Schwelle steht nicht im Code. Die Regel steht in
[co-training.md §5](training/co-training.md#5-vorregistrierte-regel--was-den-lauf-zum-erfolg-oder-zum-ende-macht).

### Aktionen von `server_rl_run.sh` (Hilfe: `server_rl_run.sh help`)

| Gruppe | Aktionen |
|---|---|
| Einrichten | `preflight` (Image, GPU, RT-Cores), `setup` (Checkpoint und USD von HF), `shell`, `clean` (Container weg, Daten bleiben) |
| Messen | `eval` (Closed Loop), `oracle` (Demo-Aktionen statt Policy, obere Schranke), `visgate`, `span` (Fingerspanne auf Echtbildern), `grasp` (Greifphysik), `latency` |
| Szene prüfen | `view`, `webview`, `cams`, `gap` (SigLIP-Abstand, nicht der Encoder der Policy), `check`, `livecheck` |
| Würfellage | `layout`, `layoutcheck`, `layoutreport`, `yawcheck`, `topface`, `tipcheck` |
| Daten erzeugen | `render` (Co-Training-Datensatz), `replay-prepare`, `replay-calibrate`, `replay-poses`, `replay-render` |
| Sonstiges | `rl` (nur N1.6), `optimize [export\|build\|validate\|benchmark\|all]` |

## 6. Werkzeuge (`tools/`, nur Host)

| Skript | Aufruf | Ergebnis |
|---|---|---|
| `sim_cl_watchdog.sh` | `TAG CONTAINER GPU GROOT_VERSION HF_REPO CHECKPOINT_PATH` | Closed Loop mit Protokoll-Szene (20 × 120 s, `layout_heldout_dy3.json`); startet nach 900 s ohne Log-Fortschritt neu, höchstens 2-mal. Log `~/groot-rl-data/logs/<TAG>_<datum>.log` (`:31-90`) |
| `sim_visgate_seeds.sh` | `CONTAINER GROOT_VERSION GPU REPO...` | `visgate` mit Seeds 0/1/2 für jeden HF-Checkpoint; lädt fehlende ohne Optimizer nach `/data/checkpoints/<repo ohne groot-g1dex3->` |
| `sim_cl_summary.py` | `results_*.json …` | Tabelle der Stufen als k/n mit 95-%-Intervall, Fingerspanne, Hub, Artefakt-Verdacht |
| `sim_visgate_seeds_summary.py` | `RESULTS_DIR TAG...` | Mittel der Seed-Mediane `mse_hand` je Set (Echtbild, cube5, cube4) mit Spanne |
| `layout_shift.py` | `SRC DST [--dy 0.03] [--dx 0]` | Layout-JSON mit Würfeln gespiegelt nach außen (`y += dy·sign(y)`), z. B. `layout_heldout_dy3.json` (Lauf 62) |
| `check_links.py` | — | prüft Markdown-Links und Anker |

## 7. Echter Roboter

Im Repo gibt es **keine Anbindung an den G1**: keinen DDS-/`unitree_sdk2`-Client im eigenen
Code (`Training/`, `Simulation/`, `tools/`). `app/Groot-1.6/gr00t/eval/real_robot/` enthält
nur das SO100-Beispiel von NVIDIA. `Simulation/scripts/policy_latency.py` misst ausdrücklich
nur die Policy, nicht Kamera, Netz und Regelung.

Eine Roboter-Seite müsste:

1. die vier Kamerabilder als 640 × 480 RGB aufnehmen,
2. die 28 Gelenkwinkel in der Reihenfolge aus Abschnitt 1 lesen (DDS),
3. beides als Beobachtung nach Abschnitt 3 an den Server schicken,
4. die 16 × 28 absoluten Gelenkziele mit 30 Hz an Arme und Hände geben (8 davon, dann neu
   anfragen),
5. Gleichgewicht und Beine selbst regeln: Die Policy steuert nur Arme und Hände.
