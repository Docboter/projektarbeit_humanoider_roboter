# Dokumentation — Übersicht

> **TL;DR:** Globaler Navigations-Hub für die gesamte Projekt-Dokumentation, thematisch nach
> Training, Simulation, Ergebnisse und Weiterführende Arbeiten sortiert. Hier einsteigen, um
> die passende Detail-Anleitung zu finden.

Navigations-Hub für die gesamte Projekt-Dokumentation. Das [Projekt-README](../README.md) im
Repo-Root gibt den Schnellstart; hier liegen die ausführlichen Anleitungen, thematisch sortiert.
Die Gliederung folgt dem Projektverlauf: **Anleitung → Ergebnis → Ausblick**.
Veraltete Inhalte werden nicht in den Dokumenten mitgeschleppt, sondern nach
[historie.md](historie.md) ausgelagert (Changelog-Rohmaterial). *(Struktur-Stand: 2026-10-11)*

> **Konvention — TL;DR zuerst:** Jede Doku-Seite beginnt unter ihrer Überschrift mit einem
> `> **TL;DR:**`-Block (was ist das, wann lesen, was steht *nicht* hier), jedes Skript unter
> `Training/`, `Simulation/`, `tools/` mit einer Zeile `# TL;DR: …` nach dem Shebang.
> `tools/check_tldr.sh` prüft das, `tools/check_tldr.sh --list` druckt alle TL;DRs als Übersicht.

**Einstieg in drei Seiten:** [Projekt-README](../README.md) (Ergebnis, Repo-Karte) →
[quickstart.md](quickstart.md) (Training starten, Checkpoint messen) →
[schnittstellen.md](schnittstellen.md) (Datensatz, Checkpoint, Policy-Server, Sim-Eval, Werkzeuge).
Alles Weitere auf dieser Seite ist Referenz.

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
| [training/co-training.md](training/co-training.md) | **Schritt 4 — Co-Training echt + gerendert** (`USE_COTRAIN=1`). Renderer für Sim-Bilder zu echten Aktionen, Zwei-Datensatz-Training mit `mix_ratio`, begründete Episodenzahl/Mischung, vorregistrierte Erfolgsregel. Ergebnis v3 (Läufe 73–97) in §3.5 |
| [training/synth-datensatz.md](training/synth-datensatz.md) | **Fremder Sim-Datensatz im Co-Training** (`Fichtl00/Cube_Stacking_synth`) — warum er schematisch nicht passt (57-Dim-State, 50 fps), wie `harmonize_synth_dataset.py` die Gelenk-Zuordnung *misst* statt sie zu raten, begründetes Mischungsverhältnis, vorregistrierte Bewertungsregel |
| [training/wandb-offline-sync.md](training/wandb-offline-sync.md) | W&B-Offline-Sync auf KISSKI |

## Augmentation (Video-Augmentierung mit Cosmos-Transfer2.5)

Erzeugt zusätzliche Trainingsdaten, indem echte Trainingsvideos per NVIDIA
Cosmos-Transfer2.5 stilvariiert werden (Domain-Randomization). Läuft auf einem eigenen
Docker-Server (A100/H100-Klasse). Das Ergebnis speist den Co-Training-Pfad unter Training.
**Stand 2026-10-13: gebaut, nicht durchgängig erprobt** (erster Lauf scheiterte am gated
`nvidia/Cosmos-Predict2.5-2B`; passender ~65-GB-Docker-Server fehlt derzeit).

| Dokument | Inhalt |
|---|---|
| [augmentation/](augmentation/README.md) | **Einstieg Augmentation** (Index) |
| [augmentation/anleitung.md](augmentation/anleitung.md) | **Bedienungsanleitung** — Warum Cosmos-Transfer2.5, Build/Run, Env-Var-Referenz, Einbindung ins Training (`USE_COTRAIN`), offene Punkte |

## Simulation (Closed-Loop-Eval in Isaac Lab)

Operativer Sim-Eval-Workflow. Mess- und Methodik-Ergebnisse stehen unter [Ergebnisse](#ergebnisse--evaluation).

| Dokument | Inhalt |
|---|---|
| [simulation/](simulation/README.md) | **Einstieg Simulation** (Index) |
| [simulation/sim-eval-anleitung.md](simulation/sim-eval-anleitung.md) | **Primärer Workflow** — Closed-Loop-Sim-Eval Schritt für Schritt: gemeinsame Einmal-Schritte, dann Weg A IKR-Server (Standard) oder Weg B vast.ai (Cloud-Alternative); inkl. Open-Loop-Replay-Diagnose |
| [simulation/live-ansicht.md](simulation/live-ansicht.md) | **Live zuschauen** — Isaac-Sim-Viewport per WebRTC auf dem eigenen Rechner öffnen (`LIVESTREAM=2`, nativer Streaming-Client) statt hinterher MP4s zu holen. Gilt für Sim-Eval, Baseline, Greif-Test und RL. Gebaut 2026-08-13, Hardware-Test offen |
| [simulation/umsetzungsnotizen.md](simulation/umsetzungsnotizen.md) | **READ FIRST** — Lessons Learned, bekannte Fixes (Stand bis Juni 2026; die Sim-Erkenntnisse seit August — Isaac-Sim-6.0-Port, Kamera-Neukalibrierung, Greif-Diagnostik — stehen in [ergebnisse/diagnose-chronik.md](ergebnisse/diagnose-chronik.md)) |
| [simulation/wuerfellage-rekonstruktion.md](simulation/wuerfellage-rekonstruktion.md) | **Würfellage aus den Realbildern** — Verfahren und Koordinatentransformation (Bild → Kamerastrahl → Sim-Koordinate) für den Co-Training-Renderer. Übergabedokument mit Annahmenliste, den zwei gescheiterten Vorgängerverfahren und dem offenen Abnahme-Test. |
| [simulation/wuerfellage-rekonstruktion-bewertung.md](simulation/wuerfellage-rekonstruktion-bewertung.md) | **Bewertung der Implementierung** (Stand 2026-08-21) — Doku-vs.-Code-Abgleich von `pick_anchored_homography` v4, priorisierte Befundliste und Empfehlungen |
| [simulation/replay-videos-aus-realdaten.md](simulation/replay-videos-aus-realdaten.md) | **Replay-Videos aus Realdaten** — physikbasierter Dataset-Replay (`run_dataset_replay_videos.py`) mit Würfelkalibrierung aus den Realbildern; Werkzeug und Befunde |
| [simulation/inferenz-optimierung.md](simulation/inferenz-optimierung.md) | **Inferenz-Optimierung** — Latenz-Zerlegung der GR00T-Inferenz, `torch.compile`/TensorRT-Backend für das DiT (`server_rl_run.sh optimize`/`latency`) |
| [simulation/basismodell-referenzaufgabe.md](simulation/basismodell-referenzaufgabe.md) | **Referenzaufgabe zur Sim-Validierung** — was das Basismodell laut NVIDIA können muss, als Prüfstein für die eigene Pipeline (Stand 2026-06-16) |
| [simulation/robocasa-referenz-eval.md](simulation/robocasa-referenz-eval.md) | **RoboCasa-GR-1-Referenz-Eval (Bedienung)** — `server_robocasa_ref_run.sh`; validiert die Pipeline gegen NVIDIAs publizierte Zahlen. Ergebnis in [ergebnisse/basismodell-referenz-eval.md](ergebnisse/basismodell-referenz-eval.md) |
| [simulation/archiv/](simulation/archiv/) | Historische Planungs-/Analyse-Docs (überholt, als Kontext erhalten — jede Datei trägt einen Vorbehalts-Banner) |

## Ergebnisse & Evaluation

Alle Auswertungen, Messungen und Methodik-Reviews gebündelt — die „Was kam raus?"-Anlaufstelle.

| Dokument | Inhalt |
|---|---|
| [ergebnisse/](ergebnisse/README.md) | **Einstieg Ergebnisse** (Index) |
| [ergebnisse/lauf1-auswertung.md](ergebnisse/lauf1-auswertung.md) | **Abschluss-Auswertung 1. Lauf** (175k Steps) — Metriken + Verhaltens-Evaluation, Diagnose (visueller Domain-Gap), Empfehlungen |
| [ergebnisse/lauf2-vision-auswertung.md](ergebnisse/lauf2-vision-auswertung.md) | **Auswertung 2. Lauf** — Training mit Vision-Encoder (`TUNE_VISUAL=1`, Namespace `blockstacking_vision`) |
| [ergebnisse/lauf3-vision-split-auswertung.md](ergebnisse/lauf3-vision-split-auswertung.md) | **Auswertung 3. Lauf** (`tp1nc699`, Vision + Color-Jitter + 80/20-Split) — **erste echte Validierungszahl im Projekt.** Der erste Checkpoint-Sweep (6 × 300) zeigte eine U-Kurve mit Optimum 30.000 — **überholt**, siehe Nachtrag. **Closed-Loop nachgemessen (Lauf 34):** dieser Checkpoint hebt die Sim-Fingerspanne auf 27,6 % — erster Lauf mit belegter Verhaltenswirkung. **Nachtrag 2026-10-12:** die Nachmessung mit 20×750 widerlegt die U-Kurve — bester Checkpoint ist 44.000, kein Overfitting |
| [ergebnisse/lauf4-n17-vergleich.md](ergebnisse/lauf4-n17-vergleich.md) | **Auswertung 4. Lauf:** GR00T N1.7 gegen N1.6 (Lauf 3), gleiche Konfiguration, 44k Steps. Open-Loop am Ende gleichauf (n.s.), N1.7 lernt früher und trainiert 1,6× schneller. Enthält die 20×750-Nachmessung von Lauf 3 |
| [ergebnisse/diagnose-chronik.md](ergebnisse/diagnose-chronik.md) | **Diagnose-Chronik (fortlaufend ab Lauf 08)** — chronologisches Protokoll der Sim-/RL-Diagnoseläufe: Kamera-Fixes, Domain-Gap-Messläufe, Greif-Physik (Lauf 29), `span`-Gate (Lauf 32), `TUNE_VISUAL` im Closed Loop (Lauf 34), Würfellage aus dem Bild (35–37), Geometriefehler Fingergelenke/Beckenhöhe/Basis (38–51), Gierwinkel (52/53), N1.7 in der Sim (54/55), Orakel und Würfelversatz (57–69), Szenen-Angleichung (70/71), Co-Training v3 mit Gate und Closed Loop (72–97). Die projektweit referenzierten Lauf-Nummern leben hier |
| [ergebnisse/basismodell-referenz-eval.md](ergebnisse/basismodell-referenz-eval.md) | **Referenz-Eval-Ergebnis** — RoboCasa GR-1, Aggregat 47,7 % über 12 Tasks; validiert die Eval-Pipeline gegen NVIDIAs Zahlen (Re-Run der restlichen 12 Tasks offen) |
| [ergebnisse/domain-gap-analyse.md](ergebnisse/domain-gap-analyse.md) | **Domain-Gap-Messung** — Cosine-Distanz Real→Sim pro Kamera via vortrainiertes SigLIP (Näherung, nicht der Encoder der Policy). Neumessung 2026-08-08 nach Kamerakalibrierung + Albedo-Fixes: Mittel 0.22, `cam_left_wrist` 0.36 (Juni-Erstmessung 0.26/0.43 überholt) |
| [ergebnisse/sim-bewertung.md](ergebnisse/sim-bewertung.md) | **Methodik-Review** — Ist Closed-Loop-Sim sinnvoll/korrekt? Belegt: 0-%-Ergebnis ist der erwartete Real→Sim-Gap; Open-Loop-MSE ist die valide Metrik. Mit Code-Befunden + Quellen |

## Weiterführende Arbeiten

Konzept- und Planungs-Dokumente für Folgeschritte — recherchiert; **RL-Pipeline und
Livestream-Spur B sind inzwischen umgesetzt**, die Lokomotion bleibt Konzept.
Diese Sammlung speist das gleichnamige Kapitel der Projektarbeit.

| Dokument | Inhalt |
|---|---|
| [weiterfuehrend/](weiterfuehrend/README.md) | **Einstieg Weiterführende Arbeiten** (Index) |
| [weiterfuehrend/reinforcement-learning-plan.md](weiterfuehrend/reinforcement-learning-plan.md) | **RL-Plan + Implementierung** — Algorithmen-Vergleich, Infrastruktur, Status der gebauten Bausteine; Pipeline läuft seit 2026-08-08 end-to-end, der Greif-Physik-Blocker der Läufe 25–28 ist mit **Lauf 29** gefallen (Würfel hebt 7,9 cm). **Lauf 30** schließt die Diagnose: die Policy kommandiert nur 19 % der demonstrierten Greifbewegung. **Lauf 32** (`span`-Gate) entscheidet die Ursache: auf echten Bildern erreicht dieselbe Policy 100 % → **starker Hinweis auf den Domain-Gap**. **Lauf 34** misst den daraufhin trainierten Vision-Checkpoint im Closed Loop: 27,6 % statt 20,5 % (p = 3,3 · 10⁻⁴) — die Maßnahme wirkt, `lifted` bleibt aber 0/10 → nächster Schritt ist **Co-Training auf gerenderten Bildern**, **RL kommt danach**. **Vorbehalt (Stand 2026-10-11):** die Geometrie-Korrekturen vom 2026-08-24 (Fingervorzeichen, Beckenhöhe, Basis-x) stellen alle älteren Sim-Greifzahlen unter Vorbehalt; N1.6-Neumessung und neuer Renderdatensatz sind inzwischen gelaufen (Läufe 72–97, Stand: [next-steps.md §0](../next-steps.md)). Erster Co-Training-Lauf `cotrain_synth_v1` ist am 2026-09-07 nach 138 s gescheitert |
| [weiterfuehrend/rl-anleitung.md](weiterfuehrend/rl-anleitung.md) | **RL-Bedienungsanleitung (operativ, schlank)** — Image bauen → BC-Checkpoint → RT-Core-GPU (eigener Docker-Server via `server_rl_run.sh` oder vast.ai) → RL starten → überwachen → Checkpoints sichern; plus Diagnose-Werkzeuge `gap`/`eval`/`grasp`/`span`. Die Lauf-Historie dazu: [ergebnisse/diagnose-chronik.md](ergebnisse/diagnose-chronik.md) |
| [weiterfuehrend/lokomotion-recherche.md](weiterfuehrend/lokomotion-recherche.md) | **Lokomotions-Recherche** — Warum der Roboter fixiert ist, GR00T-N1.6-Whole-Body-Control (entkoppelt: RL-Beine + IK/VLA-Arme), Unitree-G1-Lokomotions-Stacks, Integrationspfade + Quellen |
| [weiterfuehrend/groot-n17-migration.md](weiterfuehrend/groot-n17-migration.md) | **Umstieg GR00T N1.6 → N1.7** — Recherche + Migrationsplan; **als paralleler Pfad umgesetzt** (zweiter Submodul-Pfad `app/Groot-1.7`, `GROOT_VERSION`-Laufzeit-Auswahl, ein Trainings-Image je Generation); voller KISSKI-Lauf (44k) und Closed Loop nach Abnahmeprotokoll (Läufe 81, 89, 93) gelaufen — der N1.7-Status steht nur im Abschnitt [„Stand der Umsetzung“](weiterfuehrend/groot-n17-migration.md#stand-der-umsetzung-2026-08-19). N1.7 (Cosmos-Reason2-2B/Qwen3-VL, GA 2026-07-07) im Überblick, Unterschiede zu unserem Stand (gated Backbone, Py 3.12/torch 2.9, N1.6-Checkpoints nicht ladbar; `NEW_EMBODIMENT`/CLI/Datensatz/ZMQ bleiben kompatibel), 8-Phasen-Plan mit Abnahmekriterien |
| [weiterfuehrend/livestream-plan.md](weiterfuehrend/livestream-plan.md) | **Livestream-Plan** — **Spur B** (MJPEG-Frame-Stream im Browser, `LIVE_VIEW=1`) ist für den RL-Lauf gebaut; **Spur A** (WebRTC-Echtzeit-Viewport) ist gebaut, aber auf Hardware ungetestet — inzwischen mit zwei Clients: nativer App und **Browser** (`webview`, Port 8210). Bedienung: [simulation/live-ansicht.md](simulation/live-ansicht.md) |
| [weiterfuehrend/cli-menuefuehrung.md](weiterfuehrend/cli-menuefuehrung.md) | **Geführte CLI-Menüs (umgesetzt 2026-08-20, Router 2026-08-21)** — [`./run.sh`](../run.sh) ist der eine Einstiegspunkt: Domäne wählen, dann führt der jeweilige Launcher weiter. Skriptstart ohne Parameter führt durch die nötigen Werte und erklärt sie. Lange Aktionslisten bekommen eine Gruppenebene mit Vorschau der enthaltenen Aktionen (`[*]` = doch alles auf einen Schirm). `MENU=0` schaltet ab, `MENU_ARROWS=0` nur die Pfeiltasten, `MENU_NEST=0`/`1` die Schachtelung. `[←]`/`[z]` führt aus jeder Ebene zurück bis ins Hauptmenü. Parameter-Specs unter [`tools/menu/`](../tools/menu/) sind die einzige Quelle, `tools/gen_docs.sh` prüft sie gegen Skripte und Doku-Tabellen. §12 hält die Abweichungen vom Plan fest und zwei dabei gefundene Defekte im Trainings-Launcher, §13 den Router, die Pfeiltasten, die Gruppenebene, den Rückweg, die Einordnung von KISSKI unter „Training“ und (§13.7) den Checkpoint als Grundfrage jedes Messlaufs — `CHECKPOINT_PATH` ist der Hebel, `HF_CHECKPOINT_REPO` greift nur, solange der Pfad im Container fehlt |

## Archiv

Abgeschlossene Pläne, Reviews und Zwischenstände: [archiv/](archiv/README.md). Ältere
Sim-Planungsdokumente liegen weiter unter [simulation/archiv/](simulation/archiv/), weil der Code
auf sie verweist.

## Querschnitt (Training + Simulation)

| Dokument | Inhalt |
|---|---|
| [abgabe-plan.md](abgabe-plan.md) | **Plan bis zur Abgabe (2026-10-29)** — Umbau der Markdown-Doku (Einstieg, neue Schnittstellen-Seite, Inventur) und der LaTeX-Ausarbeitung (kürzer, Schnittstellen und Anleitung im Mittelpunkt), Prüfschritte, offene Entscheidungen |
| [../next-steps.md](../next-steps.md) | **Nächste Schritte** — laufender Prioritäten-Tracker: Stand der Messkette, priorisierte nächste Schritte, offene Widersprüche zwischen den Ergebnisdokumenten |
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

Dies ist die **eine** Strukturübersicht des Repos (bis 2026-10-13 stand eine zweite, ausführlichere
Fassung in `CLAUDE.md`; deren Zusatzangaben sind hier eingearbeitet).

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
├── Simulation/                # Closed-Loop-Sim-Eval
│   ├── Dockerfile             # KISSKI: schlanker Isaac-Lab-Sim-Client (ohne GR00T)
│   ├── Dockerfile.standalone  # Standalone: Isaac Sim + GR00T in einem Container — IKR-Server & vast.ai
│   ├── Dockerfile.webviewer   # Browser-Client für den WebRTC-Viewport (Port 8210); enthält KEINEN
│   │                          #   Simulator — prüft zusätzlich, dass NVIDIAs sed-Patch gegriffen hat
│   ├── server_rl_run.sh       # ★ Eigener-Server-Workflow (Docker), 29 Aktionen in 7 Gruppen:
│   │                          #   preflight/setup/view/check/cams/gap/eval/oracle/grasp/span/visgate/render/rl/
│   │                          #   livecheck/webview/layout/layoutcheck/yawcheck/tipcheck/topface/
│   │                          #   latency/optimize/replay-*/shell/clean (Liste: `--help`).
│   │                          #   view = Szene ohne Gewichte/HF_TOKEN; RL_NETWORK_MODE=host =
│   │                          #   Host-Netz (erster Verdacht bei schwarzem WebRTC-Viewport)
│   ├── kisski_sim_submit.sh / kisski_replay_submit.sh / kisski_robocasa_ref_submit.sh
│   │                          #   SLURM-Jobs (Sim läuft auf KISSKI faktisch nicht: keine RT-Cores)
│   ├── compare_domain_gap.sh  # Dataset- vs. Sim-Kameraframes nebeneinander
│   ├── server_robocasa_ref_run.sh  # RoboCasa-GR-1-Referenz-Eval (Pipeline-Validierung)
│   ├── update_sim_image.sh    # Build/Push-Tool (--standalone)
│   ├── g1_dex3_sim/           # In das Image kopiert (→ /workspace/g1_dex3_sim/)
│   │   ├── run_g1_dex3_sim_eval.py   # Haupt-Eval-Schleife (ZMQ-Client zum GR00T-Server)
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
│   │   ├── run_g1_dex3_replay.py     # Open-Loop-Replay-DIAGNOSE (ohne Server/Modell); replay_episode0.npz
│   │   ├── project_fingertips_check.py  # tipcheck: FK-Fingerkuppen ins Realbild (fand den Vorzeichenfehler)
│   │   ├── dump_camera_poses.py      # konfigurierte vs. gerenderte Kamerapose + PNG je Kamera
│   │   ├── convert_urdf_to_usd.py    # einmalige URDF→USD-Konvertierung
│   │   ├── isaac_joint_order.txt     # 43 Gelenknamen in ISAAC-Reihenfolge (≠ Datensatz-Reihenfolge);
│   │   │                      #   Eingabe für harmonize_synth_dataset.py --joint-names
│   │   └── test_*.py          # Isaac-freie Tests
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
│   ├── g1_dex3.usd            # erzeugtes Roboter-USD (convert_urdf_to_usd.py)
│   └── configuration/         # zugehörige USD-Dateien
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
├── unitreerobotics/           # Rohdatensatz (~18 GB)
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

## Entwicklung: Checks und CI

Seit 2026-10-13 gibt es ein minimales CI ohne GPU:
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml), bei jedem Push und Pull Request.

| Job | Prüft |
|---|---|
| Specs, TL;DR, Links, Menü, Pins | `tools/gen_docs.sh` (Spec ↔ Skript ↔ Doku), `tools/check_tldr.sh`, `tools/check_links.py` (relative Markdown-Links und Anker), `tools/test_menu.sh` (mit ungesetztem `CI`/`GITHUB_ACTIONS`, sonst schaltet sich das Menü ab), `diff` der beiden `lib_groot_version.sh`, [`tools/ci_check_pins.sh`](../tools/ci_check_pins.sh) (GR00T-Pins in den Dockerfiles == Submodul-Gitlinks; Submodule werden nicht geklont) |
| ruff + shellcheck | `ruff check` auf `Training/`, `Simulation/`, `Augmentation/`, `tools/`; `ruff format --check` vorerst **nicht blockierend** (bis zu einem einmaligen Format-Commit); `shellcheck --severity=error` |
| Isaac-freie Tests | `pytest` über `Simulation/g1_dex3_sim/test_*.py` mit `numpy`, `pytest`, `pyzmq`, `msgpack` (gepinnt; `test_rl_fpo.py` braucht torch und wird übersprungen) und `unittest` über `Augmentation/tests/` (mit `ffmpeg`) |

Die Ruff-Konfiguration für den eigenen Code steht in der Root-[`pyproject.toml`](../pyproject.toml)
(gespiegelt aus `app/Groot-1.6/pyproject.toml`; die Forks unter `app/` behalten ihre eigene).
Lokal dieselben Checks: die vier `tools/`-Skripte oben, `ruff check .` und `ruff format --check .`.

**`graphify-out/` gehört nicht ins Repo.** `.gitignore` schließt es (auch unter `latex/`) aus; der
Graph ist rein lokal und wird mit `graphify update .` neu erzeugt. Seit 2026-10-13 (`76f2775`)
liegt er nicht mehr im Index; die alte Historie enthält ihn noch.

## Externe Ressourcen

- [GWDG HPC Dokumentation](https://docs.hpc.gwdg.de) — KISSKI/Grete-Cluster
- [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/groot)
- [LeRobot](https://github.com/huggingface/lerobot)
