# Repo-Struktur und `/data`-Layout

> **TL;DR:** Verzeichnisbaum des Repos (eine Zeile je Datei oder Ordner mit Zweck) und das
> Laufzeit-Layout unter `/data` in den Containern. Nachschlagen, wenn man eine Datei sucht; den
> Einstieg und die Leserichtung gibt [docs/README.md](README.md), die Kurzfassung die
> Repo-Karte im [Projekt-README](../README.md#repo-karte).

Dies ist die **eine** Strukturübersicht des Repos (bis 2026-10-13 stand eine zweite Fassung in
`CLAUDE.md`, bis 2026-11-09 stand diese hier im Doku-Hub). *(Stand: 2026-11-09)*

```
/
├── run.sh                     # ★ Der EINE Einstiegspunkt: fragt Simulation/Training/KISSKI
│                              #   ab und übergibt an den passenden Host-Launcher. Wählt
│                              #   nur — baut selbst nichts, kennt keine Aktionsliste.
│                              #   [←] führt aus jeder Ebene zurück bis hierher.
│                              #   Zuordnung Domäne → Launcher: tools/menu/_domains.spec
├── README.md                  # Projekt-Überblick & Schnellstart (Landing)
├── next-steps.md              # ★ Projektstatus (§0) und priorisierte nächste Schritte
├── CLAUDE.md                  # Anweisungen für Claude Code (schlank; verweist hierher)
├── .env.local.example         # Vorlage für host-spezifische Konfiguration (→ .env.local kopieren,
│                              #   gitignored); gelesen von den Host-Launchern über tools/lib_env_local.sh
├── docs/                      # ▶ Diese Dokumentation
│   ├── README.md              # dieser Navigations-Hub
│   ├── quickstart.md          # Neu hier? Systemvoraussetzungen, Accounts/Tokens, Schnellstart
│   ├── schnittstellen.md      # Datensatz, Checkpoint, Policy-Server, Sim-Eval, Werkzeuge (aus dem Code)
│   ├── training/              # operative Trainings-Doku (anleitung, kisski-hpc, env-vars, multi-gpu,
│   │                          #   train-test-split, wandb-offline-sync, trainingsverfahren, co-training,
│   │                          #   synth-datensatz)
│   ├── simulation/            # operative Sim-Eval-Doku (umsetzungsnotizen.md ZUERST lesen; + archiv/)
│   ├── augmentation/          # Cosmos-Transfer2.5-Video-Augmentierung
│   ├── ergebnisse/            # Auswertungen, Messungen, Methodik-Reviews, diagnose-chronik.md
│   │                          #   („Lauf N“-Verweise lösen dort auf)
│   ├── weiterfuehrend/        # Konzept-/Plan-Docs (RL, Lokomotion, Livestream, CLI-Menüführung, N1.7-Migration)
│   ├── fehlerbehebung.md      # Querschnitt-Fehlerlösungen
│   ├── portabilitaet.md       # Repo auf anderen Maschinen betreiben (.env.local, KISSKI_PROJECT_DIR)
│   └── historie.md            # ausgelagerte veraltete Inhalte (Changelog-Rohmaterial)
├── tools/                     # ★ Host-Helfer — NIE im Image, Änderungen brauchen keinen Rebuild
│   ├── lib_menu.sh            #   Menü-Engine (reines Bash, UI auf stderr; Pfeiltasten, Domänenebene)
│   ├── lib_env_local.sh       #   gemeinsamer .env.local-Lader
│   ├── lib_docker_ns.sh       #   DOCKER_NAMESPACE: fragt vor einem Push nach dem Docker-Hub-Konto
│   ├── gen_docs.sh            #   Drift-Check: Spec-Default vs. ${VAR:-…} vs. Doku-Tabellen
│   ├── test_menu.sh           #   Abnahmeplan der Menüführung, ausführbar
│   ├── check_tldr.sh          #   TL;DR-Konvention prüfen; --list = Überblick auf einen Schirm
│   ├── sim_cl_watchdog.sh     #   Closed Loop (Protokoll Lauf 82) mit Hänger-Wächter (Chronik Läufe 89–95)
│   ├── sim_visgate_seeds.sh   #   visuelles Gate, Seeds 0/1/2, für eine Liste von HF-Checkpoints (Lauf 92)
│   ├── sim_cl_summary.py      #   Closed-Loop-JSONs → Tabelle mit exaktem KI, Artefakt-Verdacht, Pooling
│   ├── sim_visgate_seeds_summary.py # Gate-Kennzahl: Mittel der Seed-Mediane MSE Hände je Checkpoint
│   ├── layout_shift.py        #   Layout-JSON: Würfel gespiegelt nach außen (dy3-Korrektur, Lauf 62)
│   ├── check_links.py         #   relative Markdown-Links und Anker prüfen
│   ├── ci_check_pins.sh       #   GR00T-Pins in den Dockerfiles == Submodul-Gitlinks (CI)
│   └── menu/                  #   _domains.spec (Domänenliste für run.sh) + eine *.spec je Aktion
├── Training/                  # Alles rund ums Training (Build, Run, Skripte)
│   ├── Dockerfile             # Image, ENTRYPOINT = /scripts/entrypoint.sh; Build-Kontext = Training/
│   ├── .dockerignore          # hält den Kontext auf scripts/ — die Host-Launcher gehören nie ins Image
│   ├── docker-compose.yml     # optional (Dev-Komfort, keine Host-Mounts)
│   ├── kisski_submit.sh       # SLURM-Job für KISSKI (bleibt bewusst allein scp-bar, keine tools/-Abhängigkeit)
│   ├── kisski_menu.sh         # Login-Knoten-Helfer: fragt Partition/Walltime, baut die `sbatch --export=ALL …`-Zeile
│   ├── kisski_chain.sh        # Login-Knoten: mehrere Trainingsläufe hintereinander (afterany),
│   │                          #   je mit Checkpoint-Sweep (afterok); Presets z. B. synth_jointspace, synth_v22
│   ├── kisski_open_loop_eval.sh   # SLURM: Open-Loop-Checkpoint-Sweep (ohne Server)
│   ├── kisski_rl_submit.sh    # SLURM: RL (FPO) mit Sim-SIF, RT-Core-Guard (TEMPLATE)
│   ├── update_image.sh        # Host-Build/Push-Tool (--groot=1.6|1.7|both; muss neben dem Dockerfile liegen)
│   ├── setup_and_train_dockerhub_pull.sh   # Host-Launcher: docker pull + docker run
│   ├── setup_and_train_container_build.sh  # baut das Image lokal, übergibt dann an den Pull-Launcher
│   │                          #   (Bootstrap-Branch: REPO_BRANCH, sonst der des Checkouts, sonst main)
│   └── scripts/               # In das Image kopiert (→ /scripts)
│       ├── entrypoint.sh      # orchestriert Download → Convert → Train (TUNE_VISUAL/USE_COTRAIN routen hier)
│       ├── download_data.sh   # HuggingFace-Download (Modell + Datensatz)
│       ├── run_finetuning.sh  # Trainings-Launcher (torchrun bei NUM_GPUS>1)
│       ├── run_finetuning_vision.sh   # TUNE_VISUAL=1, Namespace blockstacking_vision
│       ├── run_finetuning_cotrain.sh  # USE_COTRAIN=1 (echte + gerenderte Bilder), Namespace blockstacking_cotrain
│       ├── run_finetuning.ps1 # VERALTET (Windows; ohne Split/Resume-Schutz — WSL2 + bash nutzen)
│       ├── launch_cotrain.py / launch_cotrain_n17.py   # Zwei-Datensatz-Einstieg (mix_ratio) für N1.6 bzw.
│       │                      #   N1.7: Kopie der Fork-launch_finetune.py mit EINER Änderung (Datensatzliste),
│       │                      #   daher kein Submodul-Eingriff. Bei GR00T-Updates nachprüfen
│       ├── lib_resume_guard.sh   # verhindert stilles Fortsetzen eines alten Laufs
│       ├── lib_split.sh       # gemeinsame Train/Test-Split-Logik; schreibt split.json neben die Checkpoints
│       ├── lib_groot_version.sh  # ★ GROOT_VERSION (1.6|1.7|auto) → Codebaum, venv, Modell-Repo,
│       │                      #   Namespace-Suffix. Identische Kopie in Simulation/scripts/
│       ├── checkpoint_sweep.py   # ★ Checkpoint-AUSWAHL: Open-Loop-MSE/MAE je Checkpoint auf den
│       │                      #   zurückgehaltenen Episoden (der Fork hat keine Eval im Training);
│       │                      #   läuft unter beiden venvs
│       ├── harmonize_synth_dataset.py  # fremden Sim-Datensatz (Cube_Stacking_synth) aufs echte Schema;
│       │                      #   MISST die Gelenkzuordnung statt sie zu raten (synth-datensatz.md)
│       └── dex3_fingerorder_probe.py   # belegt, dass die DEX3-Fingerreihenfolge nicht aus der
│                              #   Kinematik bestimmbar ist (Isaac-freie FK aus der URDF)
├── Augmentation/              # Cosmos-Transfer2.5-Video-Augmentierung (eigener Docker-Server, Dauerbetrieb)
│   ├── Dockerfile             # klont cosmos-transfer2.5 (gepinnt) + sparse Checkout des GR00T-Fork-Konverters
│   ├── .dockerignore          # wie Training/.dockerignore
│   ├── update_image.sh        # Host-Build/Push-Tool (Konto: DOCKER_NAMESPACE)
│   ├── setup_and_augment_dockerhub_pull.sh  # Host-Launcher (langlebiger Container, kein --rm/-v)
│   └── scripts/               # In das Image kopiert (→ /scripts)
│       ├── entrypoint.sh      # Download → Specs → Cosmos-Inferenz (Kanten on-the-fly) → Datensatz
│       ├── download_source_dataset.sh  # echter Datensatz + vendorter v3→v2.1-Konverter
│       ├── build_controlnet_specs.py   # Episode/Kamera/Variante → Cosmos-Spec-JSON
│       ├── run_inference.sh   # dünner Wrapper um examples/inference.py von cosmos-transfer2.5
│       └── assemble_dataset.py   # Cosmos-Rohvideos → COTRAIN_DATASET_PATH-kompatibler LeRobot-v2.1-Satz
│   └── tests/                 # unittest der Skripte (test_augment_scripts.py, braucht ffmpeg; läuft in der CI)
├── Simulation/                # Closed-Loop-Sim-Eval
│   ├── Dockerfile             # KISSKI: schlanker Isaac-Lab-Sim-Client (ohne GR00T)
│   ├── Dockerfile.standalone  # Standalone: Isaac Sim + GR00T in einem Container — IKR-Server & vast.ai
│   ├── Dockerfile.webviewer   # Browser-Client für den WebRTC-Viewport (Port 8210); enthält KEINEN
│   │                          #   Simulator — prüft zusätzlich, dass NVIDIAs sed-Patch gegriffen hat
│   ├── server_rl_run.sh       # ★ Eigener-Server-Workflow (Docker), 29 Aktionen in 7 Gruppen:
│   │                          #   preflight/setup/view/check/cams/gap/eval/oracle/grasp/span/visgate/render/rl/
│   │                          #   livecheck/webview/layout/layoutcheck/layoutreport/yawcheck/tipcheck/topface/
│   │                          #   latency/optimize/replay-*/shell/clean (Liste: `--help`).
│   │                          #   view = Szene ohne Gewichte/HF_TOKEN; RL_NETWORK_MODE=host =
│   │                          #   Host-Netz (erster Verdacht bei schwarzem WebRTC-Viewport)
│   ├── kisski_sim_submit.sh / kisski_replay_submit.sh / kisski_robocasa_ref_submit.sh
│   │                          #   SLURM-Jobs (Sim läuft auf KISSKI faktisch nicht: keine RT-Cores)
│   ├── compare_domain_gap.sh  # Dataset- vs. Sim-Kameraframes nebeneinander
│   ├── server_robocasa_ref_run.sh  # RoboCasa-GR-1-Referenz-Eval (Pipeline-Validierung)
│   ├── update_sim_image.sh    # Build/Push-Tool (--standalone)
│   ├── runs/                  # Logs der Sim-Läufe je Datum/Nummer (gitignored; „Lauf N“ → diagnose-chronik.md)
│   ├── g1_dex3_sim/           # In das Image kopiert (→ /workspace/g1_dex3_sim/)
│   │   ├── README.md          # Datei-für-Datei-Übersicht dieses Ordners
│   │   ├── run_g1_dex3_sim_eval.py   # Haupt-Eval-Schleife (ZMQ-Client zum GR00T-Server)
│   │   ├── eval_metrics.py    # Isaac-frei: kipp-robustes `lifted_clear` + Clopper-Pearson-KI
│   │   ├── oracle_replay.py   # Isaac-frei: Orakel-Client spielt Datensatz-Aktionen statt Policy ab (`oracle`)
│   │   ├── split_math.py      # Isaac-frei: Split-Grenze, Spiegel von lib_split.sh
│   │   ├── g1_dex3_blockstack_env.py # Isaac-Lab-Env (Roboter, Tisch, Würfel, 4 Policy- + 1 Szenen-Kamera;
│   │   │                      #   reward_mode binary|shaped, get_obs_batched)
│   │   ├── client.py          # ZMQ-Policy-Client + build_obs (Zustand → Modality-Keys)
│   │   ├── task_text.py       # Sprachbefehl aus dem Datensatz (Fallback „stack three block“)
│   │   ├── scene_geometry.py  # EINE Quelle für Tisch, Würfelhöhe, Roboterbasis
│   │   ├── camera_geometry.py # Kameraposen, Intrinsik, Lochkameramodell (Isaac-frei, außerhalb prüfbar)
│   │   ├── g1_dex3_cfg.py     # Articulation-Config; Kamerateil aus camera_geometry re-exportiert
│   │   ├── rl_finetune.py     # FPO-RL-Trainer (nur Action-Head)
│   │   ├── live_view.py       # MJPEG-Live-Ansicht (LIVE_VIEW=1)
│   │   ├── view_sim.py        # Szene OHNE Policy (kein Checkpoint, kein GR00T, kein HF_TOKEN)
│   │   ├── render_cotrain_dataset.py # ★ echte Aktionen abspielen, 4 Policy-Kameras aufzeichnen →
│   │   │                      #   LeRobot-v2.1-Satz (Stufen scan/render; rendert nie Held-out-Episoden)
│   │   ├── extract_block_layout.py   # ★ Würfellage aus dem ECHTEN Bild (Farbblob → Strahl → Würfelebene)
│   │   │                      #   → layout.json; die richtige Quelle für render (vorher `layoutcheck`)
│   │   ├── reconstruct_cube_poses.py / replay_calibration.py / collect_replay_anchors.py /
│   │   │   run_dataset_replay_videos.py / grasp_pose_support.py  # physikbasierte Replay-Videos (älterer Weg)
│   │   ├── replay_grasp_metrics.py   # Greif-Metriken des Replays
│   │   ├── run_g1_dex3_replay.py     # Open-Loop-Replay-DIAGNOSE (ohne Server/Modell); replay_episode0.npz
│   │   ├── project_fingertips_check.py  # tipcheck: FK-Fingerkuppen ins Realbild (fand den Vorzeichenfehler)
│   │   ├── dump_camera_poses.py      # konfigurierte vs. gerenderte Kamerapose + PNG je Kamera
│   │   ├── dump_policy_cams.py       # Standbilder der 4 Policy-Kameras (Domain-Gap-Vergleich)
│   │   ├── recolor_hands_black.py    # einmaliges USD-Tool: DEX3-Hände schwarz (→ g1_dex3_blackhands.usd)
│   │   ├── convert_urdf_to_usd.py    # einmalige URDF→USD-Konvertierung
│   │   ├── isaac_joint_order.txt     # 43 Gelenknamen in ISAAC-Reihenfolge (≠ Datensatz-Reihenfolge);
│   │   │                      #   Eingabe für harmonize_synth_dataset.py --joint-names
│   │   └── test_*.py          # Isaac-freie Tests (u. a. test_scene_geometry.py, test_eval_metrics.py,
│   │                          #   test_oracle_replay.py)
│   ├── camera_reference/      # Dataset-Referenzframes für die Kamera-Kalibrierung
│   ├── g1_gripper_sim/        # Stock-G1-Greifer-Baseline (SIM_MODE=baseline)
│   ├── robocasa_reference/    # RoboCasa-GR-1-Referenz-Eval-Skripte
│   └── scripts/               # In das Standalone-Image kopiert (→ /scripts)
│       ├── entrypoint_sim.sh / entrypoint_replay.sh / entrypoint_baseline.sh / entrypoint_rl.sh
│       ├── lib_livestream.sh  # gemeinsame LIVE-Varianten-Logik (alle vier Entrypoints + server_rl_run.sh)
│       ├── lib_groot_version.sh  # identische Kopie von Training/scripts/lib_groot_version.sh
│       ├── measure_domain_gap.py / overlay_camera_check.py   # Domain-Gap (SigLIP) / Kamera-Overlay
│       ├── finger_span_openloop.py   # Fingerspanne auf ECHTEN Bildern (`span`)
│       ├── visgate_summary.py        # Tabelle der Aktion `visgate` (Sim-Prüfsets gegen Lauf 72)
│       ├── policy_latency.py  # reine Policy-Latenz in-process (`latency`)
│       ├── check_action_norm.py / dump_unitree_g1_dims.py
│       ├── groot_inference_backend.py / run_groot_optimized_server.py /
│       │   optimize_groot_inference.py / summarize_optimization.py   # torch.compile/TensorRT (N1.6-only)
│       └── upload_checkpoint.py  # HF-Upload (ohne optimizer.pt)
├── data/                      # Lokale Assets + Submodule (überwiegend gitignored)
│   ├── unitree_ros/           # Git-Submodul — Unitree-ROS (URDF-Quelle)
│   ├── g1_dex3.usd            # erzeugtes Roboter-USD (convert_urdf_to_usd.py); .asset_hash + config.yaml
│   │                          #   = Konvertierungs-Metadaten
│   ├── g1_dex3_blackhands.usd # Variante mit schwarzen Händen (recolor_hands_black.py)
│   ├── g1_gripper.usd         # Stock-G1-Greifer (Baseline)
│   ├── configuration/         # zugehörige USD-Dateien
│   ├── G1_Dex3_BlockStacking/ # nur meta/ versioniert (info/stats/tasks); Videos/Parquets kommen per Download
│   ├── g1_dex3_finetune/      # nur experiment_cfg/ + processor/ früher Läufe versioniert (keine Gewichte)
│   └── cotrain_synth*/, cube_stacking_synth*/   # lokale Synth-Datensätze (gitignored, synth-datensatz.md)
├── latex/                     # Projektarbeit/Thesis-Dokument (LuaLaTeX; eigenes Root-Verzeichnis, kein Submodul)
└── app/                       # zwei Git-Submodule, in den Dockerfiles beim Build geklont
    ├── Groot-1.6/             # DEFAULT — Fork lucam06/Isaac-GR00T, Branch luca/g1-dex3, gepinnter Commit
    │   ├── gr00t/experiment/launch_finetune.py   # Trainings-Einstieg
    │   ├── gr00t/policy/      # Gr00tPolicy (Inferenz)
    │   ├── examples/G1_DEX3/  # Embodiment-Config, Modality-JSONs, Guides
    │   └── scripts/lerobot_conversion/   # Datensatz-Konverter v3.0 → v2.1
    └── Groot-1.7/             # PARALLELER PFAD (GROOT_VERSION=1.7) — gleicher Fork, Branch
                               #   luca/g1-dex3-n17 (Upstream-GA + G1_DEX3 + Split-Patch + Offline-Fix);
                               #   gleiche Struktur. Pins: groot-n17-migration.md, Phase 1
```

Jeder Baum hat seine eigene venv: `app/Groot-1.6/.venv` (Python 3.10) und `app/Groot-1.7/.venv`
(Python 3.12); `uv run` im jeweiligen Baum nutzt die passende.

Zur Laufzeit (lokal/vast.ai im Container-Filesystem, auf KISSKI unter dem VAST-Projekt-Storage):
```
/data/
├── models/GR00T-N1.6-3B/     # Modellgewichte (~6 GB)
├── unitreerobotics/           # Rohdatensatz (9,2 GB)
├── g1_dex3_finetune/         # Checkpoints
└── logs/                      # Trainings-Logs
```

Der `Augmentation/`-Container ist ein **eigenes** Image mit eigenem `/data` (nicht mit dem
Trainings-Container geteilt — jeder lädt seine eigene Kopie des Datensatzes):
```
/data/
├── unitreerobotics/G1_Dex3_BlockStacking_Dataset/  # dasselbe HF-Repo, eigene Kopie
├── hf_cache/                          # Cache des Cosmos-Transfer2.5-2B-Checkpoints
├── augmentation/
│   ├── specs/                         # Cosmos-Spec-JSONs je Episode/Kamera/Variante
│   ├── raw_output/                    # rohe Cosmos-Videos vor dem Zusammenbau
│   └── g1_dex3_cosmos_augmented/      # fertiger LeRobot-v2.1-Satz (→ COTRAIN_DATASET_PATH)
└── logs/
```
