# Dokumentation — Übersicht

> **TL;DR:** Globaler Navigations-Hub für die gesamte Projekt-Dokumentation, thematisch nach
> Training, Simulation, Ergebnisse und Weiterführende Arbeiten sortiert. Hier einsteigen, um
> die passende Detail-Anleitung zu finden.

Navigations-Hub für die gesamte Projekt-Dokumentation. Das [Projekt-README](../README.md) im
Repo-Root gibt den Schnellstart; hier liegen die ausführlichen Anleitungen, thematisch sortiert.
Die Gliederung folgt dem Projektverlauf: **Anleitung → Ergebnis → Ausblick**.
Veraltete Inhalte werden nicht in den Dokumenten mitgeschleppt, sondern nach
[historie.md](historie.md) ausgelagert (Changelog-Rohmaterial). *(Struktur-Stand: 2026-08-18)*

> **Konvention — TL;DR zuerst:** Jede Doku-Seite beginnt unter ihrer Überschrift mit einem
> `> **TL;DR:**`-Block (was ist das, wann lesen, was steht *nicht* hier), jedes Skript unter
> `Training/`, `Simulation/`, `tools/` mit einer Zeile `# TL;DR: …` nach dem Shebang.
> `tools/check_tldr.sh` prüft das, `tools/check_tldr.sh --list` druckt alle TL;DRs als Übersicht.

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
| [training/train-test-split.md](training/train-test-split.md) | 80/20-Datensatz-Split **und Checkpoint-Auswahl** — Implementierung, Nutzung, und warum die Validierung erst nach dem Lauf läuft (`checkpoint_sweep.py`) |
| [training/co-training.md](training/co-training.md) | **Schritt 4 — Co-Training echt + gerendert** (`USE_COTRAIN=1`). Renderer für Sim-Bilder zu echten Aktionen, Zwei-Datensatz-Training mit `mix_ratio`, begründete Episodenzahl/Mischung, vorregistrierte Erfolgsregel. Werkzeuge gebaut, Lauf steht aus |
| [training/wandb-offline-sync.md](training/wandb-offline-sync.md) | W&B-Offline-Sync auf KISSKI |

## Augmentation (Video-Augmentierung mit Cosmos-Transfer2.5)

Erzeugt zusätzliche Trainingsdaten, indem echte Trainingsvideos per NVIDIA
Cosmos-Transfer2.5 stilvariiert werden (Domain-Randomization). Läuft auf einem eigenen
Docker-Server (A100/H100-Klasse). Das Ergebnis speist den Co-Training-Pfad unter Training.

| Dokument | Inhalt |
|---|---|
| [augmentation/](augmentation/README.md) | **Einstieg Augmentation** (Index) |
| [augmentation/anleitung.md](augmentation/anleitung.md) | **Bedienungsanleitung** — Warum Cosmos-Transfer2.5, Build/Run, Env-Var-Referenz, Einbindung ins Training (`USE_COTRAIN`), offene Punkte |

## Simulation (Closed-Loop-Eval in Isaac Lab)

Operativer Sim-Eval-Workflow. Mess- und Methodik-Ergebnisse stehen unter [Ergebnisse](#ergebnisse--evaluation).

| Dokument | Inhalt |
|---|---|
| [simulation/](simulation/README.md) | **Einstieg Simulation** (Index) |
| [simulation/vastai-anleitung.md](simulation/vastai-anleitung.md) | **Primärer Workflow** — Closed-Loop-Sim-Eval auf vast.ai, Schritt für Schritt (inkl. Open-Loop-Replay-Diagnose) |
| [simulation/live-ansicht.md](simulation/live-ansicht.md) | **Live zuschauen** — Isaac-Sim-Viewport per WebRTC auf dem eigenen Rechner öffnen (`LIVESTREAM=2`, nativer Streaming-Client) statt hinterher MP4s zu holen. Gilt für Sim-Eval, Baseline, Greif-Test und RL. Gebaut 2026-08-13, Hardware-Test offen |
| [simulation/umsetzungsnotizen.md](simulation/umsetzungsnotizen.md) | **READ FIRST** — Lessons Learned, bekannte Fixes (Stand bis Juni 2026; die Sim-Erkenntnisse seit August — Isaac-Sim-6.0-Port, Kamera-Neukalibrierung, Greif-Diagnostik — stehen in [ergebnisse/diagnose-chronik.md](ergebnisse/diagnose-chronik.md)) |
| [simulation/wuerfellage-rekonstruktion.md](simulation/wuerfellage-rekonstruktion.md) | **Würfellage aus den Realbildern** — Verfahren und Koordinatentransformation (Bild → Kamerastrahl → Sim-Koordinate) für den Co-Training-Renderer. Übergabedokument mit Annahmenliste, den zwei gescheiterten Vorgängerverfahren und dem offenen Abnahme-Test. |
| [simulation/basismodell-referenzaufgabe.md](simulation/basismodell-referenzaufgabe.md) | **Referenzaufgabe zur Sim-Validierung** — was das Basismodell laut NVIDIA können muss, als Prüfstein für die eigene Pipeline (Stand 2026-06-16) |
| [simulation/robocasa-referenz-eval.md](simulation/robocasa-referenz-eval.md) | **RoboCasa-GR-1-Referenz-Eval (Bedienung)** — `server_robocasa_ref_run.sh`; validiert die Pipeline gegen NVIDIAs publizierte Zahlen. Ergebnis in [ergebnisse/basismodell-referenz-eval.md](ergebnisse/basismodell-referenz-eval.md) |
| [simulation/baseline-eval.md](simulation/baseline-eval.md) | **Baseline-Eval (Vorbereitung)** — un-finetuntes `GR00T-N1.6-3B` + stock G1-Greifer (`SIM_MODE=baseline`) auf Block-Stacking; parallele Pipeline + TODO-Checkliste, **erster Lauf steht aus** |
| [simulation/archiv/](simulation/archiv/) | Historische Planungs-/Analyse-Docs (überholt, als Kontext erhalten — jede Datei trägt einen Vorbehalts-Banner) |

## Ergebnisse & Evaluation

Alle Auswertungen, Messungen und Methodik-Reviews gebündelt — die „Was kam raus?"-Anlaufstelle.

| Dokument | Inhalt |
|---|---|
| [ergebnisse/](ergebnisse/README.md) | **Einstieg Ergebnisse** (Index) |
| [ergebnisse/lauf1-auswertung.md](ergebnisse/lauf1-auswertung.md) | **Abschluss-Auswertung 1. Lauf** (175k Steps) — Metriken + Verhaltens-Evaluation, Diagnose (visueller Domain-Gap), Empfehlungen |
| [ergebnisse/lauf2-vision-auswertung.md](ergebnisse/lauf2-vision-auswertung.md) | **Auswertung 2. Lauf** — Training mit Vision-Encoder (`TUNE_VISUAL=1`, Namespace `blockstacking_vision`) |
| [ergebnisse/lauf3-vision-split-auswertung.md](ergebnisse/lauf3-vision-split-auswertung.md) | **Auswertung 3. Lauf** (`tp1nc699`, Vision + Color-Jitter + 80/20-Split) — **erste echte Validierungszahl im Projekt.** Checkpoint-Sweep zeigt U-Kurve: bester Checkpoint **30.000**, der letzte (44.000) ist 25 % schlechter → Overfitting erstmals belegt. **Closed-Loop nachgemessen (Lauf 34):** dieser Checkpoint hebt die Sim-Fingerspanne auf 27,6 % — erster Lauf mit belegter Verhaltenswirkung |
| [ergebnisse/lauf1-zwischenstand.md](ergebnisse/lauf1-zwischenstand.md) | **Zwischenstand 1. Lauf** — W&B-Health-Check des laufenden Trainings (Detail-Charts in [`wandb-run-charts.html`](ergebnisse/wandb-run-charts.html)) |
| [ergebnisse/diagnose-chronik.md](ergebnisse/diagnose-chronik.md) | **Diagnose-Chronik (Läufe 08–34)** — chronologisches Protokoll der Sim-/RL-Diagnoseläufe: Kamera-Fixes, Domain-Gap-Messläufe, Greif-Physik (Lauf 29), `span`-Gate (Lauf 32), `TUNE_VISUAL` im Closed Loop (Lauf 34). Die projektweit referenzierten Lauf-Nummern leben hier |
| [ergebnisse/basismodell-referenz-eval.md](ergebnisse/basismodell-referenz-eval.md) | **Referenz-Eval-Ergebnis** — RoboCasa GR-1, Aggregat 47,7 % über 12 Tasks; validiert die Eval-Pipeline gegen NVIDIAs Zahlen (Re-Run der restlichen 12 Tasks offen) |
| [ergebnisse/domain-gap-analyse.md](ergebnisse/domain-gap-analyse.md) | **Domain-Gap-Messung** — Cosine-Distanz Real→Sim pro Kamera via frozen SigLIP-ViT. Neumessung 2026-08-08 nach Kamerakalibrierung + Albedo-Fixes: Mittel 0.22, `cam_left_wrist` 0.36 (Juni-Erstmessung 0.26/0.43 überholt) |
| [ergebnisse/sim-bewertung.md](ergebnisse/sim-bewertung.md) | **Methodik-Review** — Ist Closed-Loop-Sim sinnvoll/korrekt? Belegt: 0-%-Ergebnis ist der erwartete Real→Sim-Gap; Open-Loop-MSE ist die valide Metrik. Mit Code-Befunden + Quellen |

## Weiterführende Arbeiten

Konzept- und Planungs-Dokumente für Folgeschritte — recherchiert; **RL-Pipeline und
Livestream-Spur B sind inzwischen umgesetzt**, die Lokomotion bleibt Konzept.
Diese Sammlung speist das gleichnamige Kapitel der Projektarbeit.

| Dokument | Inhalt |
|---|---|
| [weiterfuehrend/](weiterfuehrend/README.md) | **Einstieg Weiterführende Arbeiten** (Index) |
| [weiterfuehrend/reinforcement-learning-plan.md](weiterfuehrend/reinforcement-learning-plan.md) | **RL-Plan + Implementierung** — Algorithmen-Vergleich, Infrastruktur, Status der gebauten Bausteine; Pipeline läuft seit 2026-08-08 end-to-end, der Greif-Physik-Blocker der Läufe 25–28 ist mit **Lauf 29** gefallen (Würfel hebt 7,9 cm). **Lauf 30** schließt die Diagnose: die Politik kommandiert nur 19 % der demonstrierten Greifbewegung. **Lauf 32** (`span`-Gate) entscheidet die Ursache: auf echten Bildern erreicht dieselbe Policy 100 % → **Domain-Gap bestätigt**. **Lauf 34** misst den daraufhin trainierten Vision-Checkpoint im Closed Loop: 27,6 % statt 20,5 % (p = 3,3 · 10⁻⁴) — die Maßnahme wirkt, `lifted` bleibt aber 0/10 → nächster Schritt ist **Co-Training auf gerenderten Bildern**, **RL kommt danach** |
| [weiterfuehrend/rl-anleitung.md](weiterfuehrend/rl-anleitung.md) | **RL-Bedienungsanleitung (operativ, schlank)** — Image bauen → BC-Checkpoint → RT-Core-GPU (eigener Docker-Server via `server_rl_run.sh` oder vast.ai) → RL starten → überwachen → Checkpoints sichern; plus Diagnose-Werkzeuge `gap`/`eval`/`grasp`/`span`. Die Lauf-Historie dazu: [ergebnisse/diagnose-chronik.md](ergebnisse/diagnose-chronik.md) |
| [weiterfuehrend/lokomotion-recherche.md](weiterfuehrend/lokomotion-recherche.md) | **Lokomotions-Recherche** — Warum der Roboter fixiert ist, GR00T-N1.6-Whole-Body-Control (entkoppelt: RL-Beine + IK/VLA-Arme), Unitree-G1-Lokomotions-Stacks, Integrationspfade + Quellen |
| [weiterfuehrend/livestream-plan.md](weiterfuehrend/livestream-plan.md) | **Livestream-Plan** — **Spur B** (MJPEG-Frame-Stream im Browser, `LIVE_VIEW=1`) ist für den RL-Lauf gebaut; **Spur A** (WebRTC-Echtzeit-Viewport) ist gebaut, aber auf Hardware ungetestet — inzwischen mit zwei Clients: nativer App und **Browser** (`webview`, Port 8210). Bedienung: [simulation/live-ansicht.md](simulation/live-ansicht.md) |
| [weiterfuehrend/cli-menuefuehrung.md](weiterfuehrend/cli-menuefuehrung.md) | **Geführte CLI-Menüs (umgesetzt 2026-08-20, Router 2026-08-21)** — [`./run.sh`](../run.sh) ist der eine Einstiegspunkt: Domäne wählen, dann führt der jeweilige Launcher weiter. Skriptstart ohne Parameter führt durch die nötigen Werte und erklärt sie. Lange Aktionslisten bekommen eine Gruppenebene mit Vorschau der enthaltenen Aktionen (`[*]` = doch alles auf einen Schirm). `MENU=0` schaltet ab, `MENU_ARROWS=0` nur die Pfeiltasten, `MENU_NEST=0`/`1` die Schachtelung. `[←]`/`[z]` führt aus jeder Ebene zurück bis ins Hauptmenü. Parameter-Specs unter [`tools/menu/`](../tools/menu/) sind die einzige Quelle, `tools/gen_docs.sh` prüft sie gegen Skripte und Doku-Tabellen. §12 hält die Abweichungen vom Plan fest und zwei dabei gefundene Defekte im Trainings-Launcher, §13 den Router, die Pfeiltasten, die Gruppenebene, den Rückweg, die Einordnung von KISSKI unter „Training“ und (§13.7) den Checkpoint als Grundfrage jedes Messlaufs — `CHECKPOINT_PATH` ist der Hebel, `HF_CHECKPOINT_REPO` greift nur, solange der Pfad im Container fehlt |
| [weiterfuehrend/wiki-migration-plan.md](weiterfuehrend/wiki-migration-plan.md) | **GitHub-Wiki-Migration (Plan)** — ob/wie sich `docs/` ins GitHub-Wiki übertragen lässt (eigenes Git-Repo, Link-Rewriting nötig, kein Auto-Sync); Alternative GitHub Pages/MkDocs. Reine Recherche, nichts umgesetzt |

## Querschnitt (Training + Simulation)

| Dokument | Inhalt |
|---|---|
| [fehlerbehebung.md](fehlerbehebung.md) | **Fehlerbehebung** — gebündelte Fehlerlösungen: Domain-Gap, OOM/VRAM, KISSKI-Queue & W&B-Pflicht, Sim-GPU-Anforderung |
| [historie.md](historie.md) | **Historie / Changelog-Rohmaterial** — chronologisch gesammelte, aus der Haupt-Doku ausgelagerte veraltete Inhalte (u. a. das frühere Audit `umgebungsanalyse.md`, die Fixes aus dem 1. Lauf, die Domain-Gap-Erstmessung) |
| [portabilitaet.md](portabilitaet.md) | **Portabilität / Fremdnutzung** — das Repo auf einem anderen Rechner betreiben. `.env.local` für den Docker-Server, `KISSKI_PROJECT_DIR`/`KISSKI_SIF_DIR` für den Cluster; enthält die **Migrationsschritte**, mit denen die eigenen Maschinen nach dem Umbau vom 2026-08-18 wieder exakt wie vorher laufen |

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
├── run.sh                     # ★ Der EINE Einstiegspunkt: fragt Simulation/Training/KISSKI
│                              #   ab und übergibt an den passenden Host-Launcher. Wählt
│                              #   nur — baut selbst nichts, kennt keine Aktionsliste.
│                              #   [←] führt aus jeder Ebene zurück bis hierher.
│                              #   Zuordnung Domäne → Launcher: tools/menu/_domains.spec
├── README.md                  # Projekt-Überblick & Schnellstart (Landing)
├── CLAUDE.md                  # Anweisungen für Claude Code
├── docs/                      # ▶ Diese Dokumentation
│   ├── README.md              # dieser Navigations-Hub
│   ├── training/              # operative Trainings-Doku
│   ├── simulation/            # operative Sim-Eval-Doku (+ archiv/)
│   ├── ergebnisse/            # Auswertungen, Messungen, Methodik-Reviews
│   ├── weiterfuehrend/        # Konzept-/Plan-Docs (RL, Lokomotion, Livestream, CLI-Menüführung)
│   ├── fehlerbehebung.md      # Querschnitt-Fehlerlösungen
│   ├── portabilitaet.md       # Repo auf anderen Maschinen betreiben
│   └── historie.md            # ausgelagerte veraltete Inhalte (Changelog-Rohmaterial)
├── Training/                  # Alles rund ums Training (Build, Run, Skripte)
│   ├── Dockerfile             # Container-Definition mit ENTRYPOINT
│   ├── kisski_submit.sh       # SLURM-Job-Script für KISSKI
│   ├── update_image.sh        # Host-Build/Push-Tool
│   ├── setup_and_train_*.sh   # Host-Launcher (Pull bzw. lokaler Build)
│   └── scripts/               # In das Image kopiert (→ /scripts)
│       ├── entrypoint.sh      # orchestriert Download → Convert → Train
│       ├── download_data.sh   # HuggingFace-Download
│       └── run_finetuning.sh  # Trainings-Launcher im Container
├── Augmentation/               # Cosmos-Transfer2.5-Video-Augmentierung (eigener Docker-Server)
│   ├── Dockerfile              # Cosmos-Transfer2.5 + vendorter v3→v2.1-Konverter, gepinnter Commit
│   ├── update_image.sh         # Host-Build/Push-Tool
│   ├── setup_and_augment_DockerHub-pull.sh  # Host-Launcher (Dauerbetrieb, wie Training/)
│   └── scripts/                # In das Image kopiert (→ /scripts)
│       ├── entrypoint.sh       # Download → Control-Videos → Specs → Cosmos-Inferenz → Datensatz
│       ├── build_controlnet_specs.py  # Episode/Kamera/Variante → Cosmos-Spec-JSON (Edge-Control on-the-fly)
│       └── assemble_dataset.py # Cosmos-Rohvideos → COTRAIN_DATASET_PATH-kompatibler Datensatz
├── Simulation/                # Sim-Client-Code, Dockerfiles, Build-Tools
│   ├── Dockerfile             # KISSKI: schlanker Isaac-Lab-Sim-Client
│   ├── Dockerfile.vastai      # vast.ai: kombiniert Isaac Sim + GR00T
│   ├── Dockerfile.webviewer   # Browser-Client für den WebRTC-Viewport (Port 8210).
│   │                          #   Enthält KEINEN Simulator — serviert nur die Seite
│   ├── server_rl_run.sh       # ★ Eigener-Server-Workflow (Docker): preflight/setup/view/check/
│   │                          #   cams/gap/eval/grasp/span/render/rl/livecheck/webview/shell/clean
│   │                          #   (render = Co-Training-Datensatz, s. training/co-training.md;
│   │                          #    view = Szene ohne Gewichte; webview = Viewport im Browser —
│   │                          #    beides in simulation/live-ansicht.md)
│   ├── server_robocasa_ref_run.sh  # RoboCasa-GR-1-Referenz-Eval (Pipeline-Validierung)
│   ├── update_sim_image.sh    # Build/Push-Tool (--vastai-Flag)
│   ├── g1_dex3_sim/           # Sim-Code (Env, Cams, Client, Eval, Replay)
│   ├── camera_reference/      # Dataset-Referenzframes für Kamera-Kalibrierung
│   └── scripts/               # In das vast.ai-Image kopiert (→ /scripts)
├── tools/                     # Host-Helfer, nie im Image: Menü-Engine + Specs, gen_docs.sh,
│                              #   test_menu.sh, check_tldr.sh (TL;DR-Konvention)
│   └── menu/_domains.spec     #   Domänenliste für run.sh (Launcher, --needs, Erklärtext)
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
