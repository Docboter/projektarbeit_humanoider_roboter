# Dokumentation — Übersicht

Navigations-Hub für die gesamte Projekt-Dokumentation. Das [Projekt-README](../README.md) im
Repo-Root gibt den Schnellstart; hier liegen die ausführlichen Anleitungen, thematisch sortiert.

## Training (Fine-tuning von GR00T N1.6)

| Dokument | Inhalt |
|---|---|
| [training/](training/README.md) | **Einstieg Training** (Index) |
| [training/anleitung.md](training/anleitung.md) | **Schritt-für-Schritt-Anleitung** — die vier Wege zum Trainieren (vast.ai, lokal mit Skript, lokal mit `docker run`, KISSKI), Daten retten, FAQ |
| [training/kisski-hpc.md](training/kisski-hpc.md) | **HPC-Training auf KISSKI** — SIF-Konvertierung, VAST-Storage, SLURM-Job, Monitoring, Checkpoint-Export, KISSKI-Troubleshooting |
| [training/env-vars.md](training/env-vars.md) | **Konfigurationsreferenz** — alle Env-Vars + VRAM-Richtwerte (Single Source of Truth) |
| [training/train-test-split.md](training/train-test-split.md) | 80/20-Datensatz-Split — Implementierung und Nutzung |
| [training/wandb-offline-sync.md](training/wandb-offline-sync.md) | W&B-Offline-Sync auf KISSKI |
| [training/erster-trainingsdurchlauf-auswertung.md](training/erster-trainingsdurchlauf-auswertung.md) | **Abschluss-Auswertung 1. Lauf** (175k Steps) — Metriken + Verhaltens-Evaluation, Diagnose (visueller Domain-Gap), Empfehlungen |
| [training/fixes-aus-erstem-lauf.md](training/fixes-aus-erstem-lauf.md) | **Fixes aus dem 1. Lauf** — Domain-Gap-Maßnahmen (Würfelfarbe, Stapel-Band, schwarze Hände, `BLACK_HANDS`-Auto-Recolor) + Werkzeuge & offene Punkte |

## Simulation (Closed-Loop-Eval in Isaac Lab)

| Dokument | Inhalt |
|---|---|
| [simulation/](simulation/) | **Einstieg Simulation** (Index) |
| [simulation/vastai-anleitung.md](simulation/vastai-anleitung.md) | **Primärer Workflow** — Closed-Loop-Sim-Eval auf vast.ai, Schritt für Schritt (inkl. Open-Loop-Replay-Diagnose) |
| [simulation/implementation-notes.md](simulation/implementation-notes.md) | **READ FIRST** — Lessons Learned, bekannte Fixes, aktueller Stand |
| [simulation/livestream-plan.md](simulation/livestream-plan.md) | **Plan** — Live-Stream der Sim via WebRTC (Echtzeit-Viewport vom Remote-GPU), noch nicht umgesetzt |
| [simulation/archiv/](simulation/archiv/) | Historische Planungs-/Analyse-Docs (überholt, als Kontext erhalten) |

## Submodul-Dokumentation (`app/Groot-1.6/examples/G1_DEX3/`)

| Dokument | Inhalt |
|---|---|
| [`SETUP_DOCUMENTATION.md`](../app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md) | Setup & Architektur des G1/DEX3-Embodiments |
| [`FINETUNING_GUIDE.md`](../app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md) | Fine-tuning Schritt für Schritt + volle Parameter-Referenz |
| [`README.md`](../app/Groot-1.6/examples/G1_DEX3/README.md) | G1/DEX3 Joint-Layout & Datensätze |

---

## Projektstruktur

```
/
├── README.md                  # Projekt-Überblick & Schnellstart (Landing)
├── CLAUDE.md                  # Anweisungen für Claude Code
├── docs/                      # ▶ Diese Dokumentation
│   ├── README.md              # dieser Navigations-Hub
│   ├── training/              # Trainings-Doku
│   └── simulation/            # Sim-Eval-Doku (+ archiv/)
├── Training/                  # Alles rund ums Training (Build, Run, Skripte)
│   ├── Dockerfile             # Container-Definition mit ENTRYPOINT
│   ├── kisski_submit.sh       # SLURM-Job-Script für KISSKI
│   ├── update_image.ps1       # Host-Build/Push-Tool
│   ├── setup_and_train_*.{sh,ps1}   # Host-Launcher (Pull bzw. lokaler Build)
│   └── scripts/               # In das Image kopiert (→ /scripts)
│       ├── entrypoint.sh      # orchestriert Download → Convert → Train
│       ├── download_data.sh   # HuggingFace-Download
│       └── run_finetuning.sh  # Trainings-Launcher im Container
├── Simulation/                # Sim-Client-Code, Dockerfiles, Build-Tools
│   ├── Dockerfile             # KISSKI: schlanker Isaac-Lab-Sim-Client
│   ├── Dockerfile.vastai      # vast.ai: kombiniert Isaac Sim + GR00T
│   ├── update_sim_image.ps1   # Build/Push-Tool (-VastAI-Flag)
│   ├── g1_dex3_sim/           # Sim-Code (Env, Cams, Client, Eval, Replay)
│   ├── camera_reference/      # Dataset-Referenzframes für Kamera-Kalibrierung
│   └── scripts/               # In das vast.ai-Image kopiert (→ /scripts)
├── data/                      # Lokale Assets + Submodule (überwiegend gitignored)
│   └── unitree_ros/           # Git-Submodul — Unitree-ROS (URDF-Quelle)
└── app/                       # Git-Submodul, im Image geklont
    └── Groot-1.6/             # GR00T N1.6 + eigene G1/DEX3-Configs
```

Zur Laufzeit (lokal/vast.ai im Container-Filesystem, auf KISSKI unter dem VAST-Projekt-Storage):
```
/data/
├── models/GR00T-N1.6-3B/     # Modellgewichte (~6 GB)
├── unitreerobotics/           # Rohdatensatz (~18 GB)
├── g1_dex3_finetune/         # Checkpoints
└── logs/                      # Trainings-Logs
```

## Externe Ressourcen

- [GWDG HPC Dokumentation](https://docs.hpc.gwdg.de) — KISSKI/Grete-Cluster
- [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/groot)
- [LeRobot](https://github.com/huggingface/lerobot)
