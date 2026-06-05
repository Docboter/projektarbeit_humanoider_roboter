# Dokumentation — Übersicht

Navigations-Hub für die gesamte Projekt-Dokumentation. Das [Projekt-README](../README.md) im
Repo-Root gibt den Schnellstart; hier liegen die ausführlichen Anleitungen, thematisch sortiert.
Die Gliederung folgt dem Projektverlauf: **Anleitung → Ergebnis → Ausblick**.

## Training (Fine-tuning von GR00T N1.6)

Operative Anleitungen zum Trainieren. Die Auswertung der Läufe steht unter [Ergebnisse](#ergebnisse--evaluation).

| Dokument | Inhalt |
|---|---|
| [training/](training/README.md) | **Einstieg Training** (Index) |
| [training/trainingsverfahren.md](training/trainingsverfahren.md) | **Was wird trainiert?** — Modellarchitektur (GR00T N1.6), G1/DEX3-Embodiment (28 DOF), Flow-Matching-Action-Head, Datensatz |
| [training/anleitung.md](training/anleitung.md) | **Schritt-für-Schritt-Anleitung** — die vier Wege zum Trainieren (vast.ai, lokal mit Skript, lokal mit `docker run`, KISSKI), Daten retten, FAQ |
| [training/kisski-hpc.md](training/kisski-hpc.md) | **HPC-Training auf KISSKI** — SIF-Konvertierung, VAST-Storage, SLURM-Job, Monitoring, Checkpoint-Export, KISSKI-Troubleshooting |
| [training/multi-gpu.md](training/multi-gpu.md) | **Multi-GPU-Training** — DeepSpeed/DDP, Batch-Size-Skalierung, KISSKI-Defaults (4× A100) |
| [training/env-vars.md](training/env-vars.md) | **Konfigurationsreferenz** — alle Env-Vars + VRAM-Richtwerte (Single Source of Truth) |
| [training/train-test-split.md](training/train-test-split.md) | 80/20-Datensatz-Split — Implementierung und Nutzung |
| [training/wandb-offline-sync.md](training/wandb-offline-sync.md) | W&B-Offline-Sync auf KISSKI |
| [training/fixes-aus-erstem-lauf.md](training/fixes-aus-erstem-lauf.md) | **Fixes aus dem 1. Lauf** — Domain-Gap-Maßnahmen (Würfelfarbe, Stapel-Band, schwarze Hände, `BLACK_HANDS`-Auto-Recolor) + Werkzeuge & offene Punkte |

## Simulation (Closed-Loop-Eval in Isaac Lab)

Operativer Sim-Eval-Workflow. Mess- und Methodik-Ergebnisse stehen unter [Ergebnisse](#ergebnisse--evaluation).

| Dokument | Inhalt |
|---|---|
| [simulation/](simulation/README.md) | **Einstieg Simulation** (Index) |
| [simulation/vastai-anleitung.md](simulation/vastai-anleitung.md) | **Primärer Workflow** — Closed-Loop-Sim-Eval auf vast.ai, Schritt für Schritt (inkl. Open-Loop-Replay-Diagnose) |
| [simulation/umsetzungsnotizen.md](simulation/umsetzungsnotizen.md) | **READ FIRST** — Lessons Learned, bekannte Fixes, aktueller Stand |
| [simulation/archiv/](simulation/archiv/) | Historische Planungs-/Analyse-Docs (überholt, als Kontext erhalten) |

## Ergebnisse & Evaluation

Alle Auswertungen, Messungen und Methodik-Reviews gebündelt — die „Was kam raus?"-Anlaufstelle.

| Dokument | Inhalt |
|---|---|
| [ergebnisse/](ergebnisse/README.md) | **Einstieg Ergebnisse** (Index) |
| [ergebnisse/lauf1-auswertung.md](ergebnisse/lauf1-auswertung.md) | **Abschluss-Auswertung 1. Lauf** (175k Steps) — Metriken + Verhaltens-Evaluation, Diagnose (visueller Domain-Gap), Empfehlungen |
| [ergebnisse/wandb-run-auswertung.md](ergebnisse/wandb-run-auswertung.md) | W&B-Run-Auswertung — Metriken-Momentaufnahme des 1. Laufs (Detail-Charts in [`wandb-run-charts.html`](ergebnisse/wandb-run-charts.html)) |
| [ergebnisse/domain-gap-analyse.md](ergebnisse/domain-gap-analyse.md) | **Domain-Gap-Messung** — Cosine-Distanz Real→Sim pro Kamera via frozen SigLIP-ViT (Mittel 0.26, `cam_left_wrist` kritisch bei 0.43) + drei Handlungsoptionen |
| [ergebnisse/sim-bewertung.md](ergebnisse/sim-bewertung.md) | **Methodik-Review** — Ist Closed-Loop-Sim sinnvoll/korrekt? Belegt: 0-%-Ergebnis ist der erwartete Real→Sim-Gap; Open-Loop-MSE ist die valide Metrik. Mit Code-Befunden + Quellen |
| [ergebnisse/baseline-unitree-g1.md](ergebnisse/baseline-unitree-g1.md) | **Baseline-Vergleich** — un-finetuntes `GR00T-N1.6-3B` + stock G1-Greifer (`UNITREE_G1`) auf Block-Stacking; parallele Pipeline (`SIM_MODE=baseline`) + TODO-Checkliste vor dem ersten Run |

## Weiterführende Arbeiten

Konzept- und Planungs-Dokumente für Folgeschritte — **recherchiert, aber noch nicht umgesetzt**.
Diese Sammlung speist das gleichnamige Kapitel der Projektarbeit.

| Dokument | Inhalt |
|---|---|
| [weiterfuehrend/](weiterfuehrend/README.md) | **Einstieg Weiterführende Arbeiten** (Index) |
| [weiterfuehrend/reinforcement-learning-plan.md](weiterfuehrend/reinforcement-learning-plan.md) | **RL-Plan (Konzept)** — möglicher nächster Schritt nach dem Domain-Gap-Befund: Algorithmen-Vergleich, Infrastruktur, offene Punkte |
| [weiterfuehrend/lokomotion-recherche.md](weiterfuehrend/lokomotion-recherche.md) | **Lokomotions-Recherche** — Warum der Roboter fixiert ist, GR00T-N1.6-Whole-Body-Control (entkoppelt: RL-Beine + IK/VLA-Arme), Unitree-G1-Lokomotions-Stacks, Integrationspfade + Quellen |
| [weiterfuehrend/livestream-plan.md](weiterfuehrend/livestream-plan.md) | **Livestream-Plan** — Live-Stream der Sim via WebRTC (Echtzeit-Viewport vom Remote-GPU), noch nicht umgesetzt |

## Querschnitt (Training + Simulation)

| Dokument | Inhalt |
|---|---|
| [umgebungsanalyse.md](umgebungsanalyse.md) | **Umgebungsanalyse / Audit** — konzeptuelle Stärken & Schwächen von Training- und Sim-Setup, konkrete Bug-/Risiko-Liste, priorisierte Empfehlungen |
| [fehlerbehebung.md](fehlerbehebung.md) | **Fehlerbehebung** — gebündelte Fehlerlösungen: Domain-Gap, OOM/VRAM, KISSKI-Queue & W&B-Pflicht, Sim-GPU-Anforderung |

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
│   ├── training/              # operative Trainings-Doku
│   ├── simulation/            # operative Sim-Eval-Doku (+ archiv/)
│   ├── ergebnisse/            # Auswertungen, Messungen, Methodik-Reviews
│   ├── weiterfuehrend/        # Konzept-/Plan-Docs (RL, Lokomotion, Livestream)
│   ├── umgebungsanalyse.md    # Querschnitt-Audit
│   └── fehlerbehebung.md      # Querschnitt-Fehlerlösungen
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
