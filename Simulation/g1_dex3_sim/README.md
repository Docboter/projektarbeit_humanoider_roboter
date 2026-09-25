# g1_dex3_sim — Orientierung

> **TL;DR:** Datei-für-Datei-Übersicht über diesen Ordner (29 Python-Module + 1 Daten-Asset,
> Stand 2026-08-26): Pipeline-Kern, Diagnose-Tools, einmalige Setup-Tools und Tests, je mit
> einer Zeile Zweck (aus den `# TL;DR:`-Kopfzeilen der Dateien). Kein Ersatz für die
> ausführlichen Guides — Workflows und Hintergrund stehen in [`docs/simulation/`](../../docs/simulation/README.md).

Dieser Ordner wird 1:1 in die Sim-Images kopiert (`Dockerfile`/`Dockerfile.standalone`, Ziel
`/workspace/g1_dex3_sim/`). Bedient wird er über [`Simulation/server_rl_run.sh`](../server_rl_run.sh)
(own-server) bzw. die `Simulation/scripts/entrypoint_*.sh` (vast.ai/KISSKI) — siehe
[rl-anleitung.md](../../docs/weiterfuehrend/rl-anleitung.md) für den Ablauf.

## a) Pipeline-Kern

Haupt-Einstiegspunkte (je über eine `server_rl_run.sh`-Aktion oder eine andere Skript-Datei
gestartet) plus die Module, die sie importieren:

| Datei | Zweck |
|---|---|
| `run_g1_dex3_sim_eval.py` | Closed-Loop-Eval: verbindet Sim mit GR00T-Policy-Server per ZMQ; `server_rl_run.sh eval` |
| `render_cotrain_dataset.py` | Rendert den Co-Training-Datensatz (Sim-Bild + echte Aktion); `server_rl_run.sh render` |
| `rl_finetune.py` | FPO-RL-Trainer für den GR00T-Action-Head auf Block-Stacking; `server_rl_run.sh rl` |
| `view_sim.py` | Zeigt die Szene ohne Modell/Checkpoint (Home-Pose); `server_rl_run.sh view` |
| `extract_block_layout.py` | Liest Würfellagen aus Realbildern (Blob-Detektion + Rückprojektion) für den Renderer |
| `run_g1_dex3_replay.py` | Open-Loop-Diagnose: spielt echte Dataset-Aktionen in die Env (kein Modell/Server) |
| `g1_dex3_blockstack_env.py` | Isaac-Lab-Env (Roboter, Tisch, Würfel, 4 Policy-Kameras + Szenenkamera) für Eval und RL |
| `g1_dex3_cfg.py` | Articulation-Config (Joints, Aktuatoren) für G1+Dex3; Kamera-Teil aus `camera_geometry.py` |
| `camera_geometry.py` | Kameraposen, Intrinsics, Pinhole-Modell (numpy, Isaac-frei); von `g1_dex3_cfg.py` importiert |
| `client.py` | ZMQ-PolicyClient für den Sim-Container (kein torch/gr00t); von Eval-Skripten importiert |
| `live_view.py` | MJPEG-Live-Ansicht der Sim im Browser (Spur B); von `rl_finetune.py` per `LIVE_VIEW=1` genutzt |

**Pfad-B-Kette** (Würfellage-Rekonstruktion aus Realbildern, s. [wuerfellage-rekonstruktion.md](../../docs/simulation/wuerfellage-rekonstruktion.md)):

| Datei | Zweck |
|---|---|
| `collect_replay_anchors.py` | Sammelt sparse Real-Bild/FK-Anker für Würfelkalibrierung ohne volle Action-Wiedergabe |
| `reconstruct_cube_poses.py` | Rekonstruiert initiale Würfelposen aus echten RGB-Bildern mittels Anker und Homographie |
| `run_dataset_replay_videos.py` | Rendert echte G1-DEX3-Dataset-Bewegungen mit einmalig gesetzten Würfeln via PhysX |

**Helfer** (reine Funktionsmodule, von der Pfad-B-Kette importiert, Isaac-frei testbar):

| Datei | Zweck |
|---|---|
| `replay_calibration.py` | Reine NumPy-Hilfsfunktionen für anker-basierte Würfel-Wiedergabe-Kalibrierung |
| `grasp_pose_support.py` | Hilfsfunktionen für sparse, zustandsbasierte Greif-Unterstützung bei Würfelposenschätzung |
| `replay_grasp_metrics.py` | Hilfsfunktionen zur physikalischen Greif-Validierung während der Datensatz-Wiedergabe |

**Daten-Asset:** [`replay_episode0.npz`](replay_episode0.npz) — gebündelte Ground-Truth-Aktionen
(Dataset-Episode 0) für `run_g1_dex3_replay.py`, kein Code.

## b) Diagnose

| Datei | Zweck |
|---|---|
| `dump_camera_poses.py` | Diagnose: konfigurierte vs. tatsächlich gerenderte Kamera-Pose; `server_rl_run.sh cams` |
| `project_fingertips_check.py` | Projiziert Fingerkuppen der FK ins Realbild zur Validierung von Kamerapose und Hand-FK |
| `dump_policy_cams.py` | Speichert Standbilder der 4 Policy-Kameras für den Domain-Gap-Vergleich mit Realbildern |

## c) Einmalige Setup-Tools

| Datei | Zweck |
|---|---|
| `convert_urdf_to_usd.py` | Einmalig: konvertiert das G1+Dex3-URDF in eine Isaac-Sim-USD-Datei |
| `recolor_hands_black.py` | Einmaliges USD-Tool: färbt die DEX3-Hände schwarz (Domain-Gap-Fix), kein Isaac Sim nötig |

## d) Tests

`test_*.py` — von pytest per Auto-Discovery gefunden, keine Isaac-Sim-Abhängigkeit (reine
Funktionstests der Helfer- und Geometrie-Module):

| Datei | Zweck |
|---|---|
| `test_cube_yaw.py` | Prüft Würfel-Gierwinkel-Schätzer gegen synthetische Würfel mit bekanntem Kameramodell |
| `test_grasp_pose_support.py` | Prüft Hilfsfunktionen für Greif-Positionsunterstützung |
| `test_layout_file_merge.py` | Prüft, dass partielle Layout-Extraktion eine bestehende `layout.json` nicht leert |
| `test_motion_onset.py` | Prüft Bewegungsbeginn-Erkennung in Farbspuren |
| `test_pick_anchored_calibration.py` | Prüft Funktionen der anker-basierten Kalibrierung |
| `test_replay_calibration.py` | Prüft Aktions-Hash und Top-Face-Erkennung aus `replay_calibration` sowie die Pinhole-Projektion |
| `test_replay_grasp_metrics.py` | Prüft Greif-Metrik-Validierungsfunktionen |

Ausführen (läuft ohne Isaac Sim, nur für die Isaac-freien Module oben):

```bash
cd Simulation/g1_dex3_sim && uv run --project ../../app/Groot-1.6 python -m pytest -q
```
