# Dokumentation — Übersicht

> **TL;DR:** Globaler Navigations-Hub für die gesamte Projekt-Dokumentation, thematisch nach
> Training, Simulation, Ergebnisse und Weiterführende Arbeiten sortiert. Hier einsteigen, um
> die passende Detail-Anleitung zu finden.

Die Gliederung folgt dem Projektverlauf: **Anleitung → Ergebnis → Ausblick**.
Veraltete Inhalte werden nicht in den Dokumenten mitgeschleppt, sondern nach
[historie.md](historie.md) ausgelagert (Changelog-Rohmaterial). *(Struktur-Stand: 2026-11-09)*

> **Konvention — TL;DR zuerst:** Jede Doku-Seite beginnt unter ihrer Überschrift mit einem
> `> **TL;DR:**`-Block (was ist das, wann lesen, was steht *nicht* hier), jedes Skript unter
> `Training/`, `Simulation/`, `tools/` mit einer Zeile `# TL;DR: …` nach dem Shebang.
> `tools/check_tldr.sh` prüft das, `tools/check_tldr.sh --list` druckt alle TL;DRs als Übersicht.

**Einstieg in drei Seiten:** [Projekt-README](../README.md) (Ergebnis, Repo-Karte) →
[quickstart.md](quickstart.md) (Training starten, Checkpoint messen) →
[schnittstellen.md](schnittstellen.md) (Datensatz, Checkpoint, Policy-Server, Sim-Eval, Werkzeuge).
Alles Weitere auf dieser Seite ist Referenz.

| Ich will … | Zuerst lesen | Details |
|---|---|---|
| ein Training starten (lokal oder KISSKI) | [Quickstart §1](quickstart.md#1-training-starten) | [training/anleitung.md](training/anleitung.md), [training/kisski-hpc.md](training/kisski-hpc.md) |
| einen Checkpoint in der Sim messen (Gate, Closed Loop) | [Quickstart §2](quickstart.md#2-checkpoint-messen-sim-server-z-b-ikr) | [simulation/sim-eval-anleitung.md](simulation/sim-eval-anleitung.md) |
| einen eigenen Client oder den Roboter an den Policy-Server anbinden | [schnittstellen.md](schnittstellen.md) | [simulation/inferenz-optimierung.md](simulation/inferenz-optimierung.md) (Latenz) |
| wissen, was herauskam | [Ergebnisse auf einen Blick](ergebnisse/README.md#ergebnisse-auf-einen-blick-stand-2026-10-29) | [Diagnose-Chronik](ergebnisse/diagnose-chronik.md) (jede Zahl mit Lauf-Nr.) |
| wissen, was als Nächstes zu tun ist | [next-steps.md §0 und §2](../next-steps.md) | — |
| eine Umgebungsvariable nachschlagen | [training/env-vars.md](training/env-vars.md) | `tools/menu/*.spec` (Sim-Aktionen) |
| einen Fehler beheben | [fehlerbehebung.md](fehlerbehebung.md) | [Quickstart §3](quickstart.md#3-wenn-etwas-hakt) |
| das Repo mit eigenen Konten oder auf einem anderen Rechner betreiben | [portabilitaet.md](portabilitaet.md) | — |
| eine Datei oder den `/data`-Aufbau finden | [repo-struktur.md](repo-struktur.md) | — |

## Training (Fine-tuning von GR00T N1.6 / N1.7)

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
**Stand 2026-11-09: gebaut, einmal end-to-end gelaufen** (2026-09-23, IKR-Server; Testlauf
2 Quell-Episoden × 3 Stilvarianten = 6 Episoden, 6003 Frames in `luca-mue/g1-dex3-cosmos-augmented`;
kein Trainingslauf damit).

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
| [simulation/umsetzungsnotizen.md](simulation/umsetzungsnotizen.md) | **Hintergrund: bekannte Fixes und Lessons Learned bis Juni 2026** (Isaac-Lab-Aufbau, frühe Sim-Fehler). Nicht der Einstieg; der aktuelle Workflow steht in [sim-eval-anleitung.md](simulation/sim-eval-anleitung.md), alles seit August in der [Diagnose-Chronik](ergebnisse/diagnose-chronik.md) |
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
| [ergebnisse/domain-gap-analyse.md](ergebnisse/domain-gap-analyse.md) | **Domain-Gap per SigLIP-Bildabstand (überholt als Maß)** — Cosine-Distanz Real→Sim je Kamera (2026-08-08: Mittel 0,22). Misst nur Bildähnlichkeit; die maßgebliche Messung ist der Hand-Fehler der Policy auf Sim-Bildern (×13 / ×36, Lauf 66, siehe [Ergebnisse auf einen Blick](ergebnisse/README.md#ergebnisse-auf-einen-blick-stand-2026-10-29)) |
| [ergebnisse/sim-bewertung.md](ergebnisse/sim-bewertung.md) | **Methodik-Review** — Ist Closed-Loop-Sim sinnvoll/korrekt? Belegt: 0-%-Ergebnis ist der erwartete Real→Sim-Gap; Open-Loop-MSE ist die valide Metrik. Mit Code-Befunden + Quellen |

## Weiterführende Arbeiten

Erweiterungen über die Kernaufgabe (Training, Messung) hinaus. Stand jeweils im Kopf der Seite,
gesamter Projektstand in [next-steps.md §0](../next-steps.md).

**Umgesetzt** (gehört heute zum Projekt, die Seiten sind Plan plus Umsetzungsprotokoll):

| Dokument | Inhalt |
|---|---|
| [weiterfuehrend/groot-n17-migration.md](weiterfuehrend/groot-n17-migration.md) | **GR00T N1.7 als paralleler Pfad** (`GROOT_VERSION=1.6\|1.7\|auto`, `app/Groot-1.7`, eigenes Image `:latest-n17`); Unterschiede zu N1.6, Abnahmeprotokoll, Läufe 4, 81, 89, 93. Aktueller Stand: [„Stand der Umsetzung“](weiterfuehrend/groot-n17-migration.md#stand-der-umsetzung-2026-08-19) |
| [weiterfuehrend/cli-menuefuehrung.md](weiterfuehrend/cli-menuefuehrung.md) | **Geführte Menüs** über [`./run.sh`](../run.sh) (`MENU=0` schaltet ab); Parameter-Specs unter [`tools/menu/`](../tools/menu/) sind die einzige Quelle, `tools/gen_docs.sh` prüft sie. Bedienung: §13 |
| [weiterfuehrend/livestream-plan.md](weiterfuehrend/livestream-plan.md) | **Live-Ansicht der Sim**: MJPEG im Browser für RL-Läufe (`LIVE_VIEW=1`) und WebRTC-Viewport (`view`/`webview`, auf Hardware ungetestet). Bedienung: [simulation/live-ansicht.md](simulation/live-ansicht.md) |

**Ausblick** (gebaut oder recherchiert, ohne Ergebnis):

| Dokument | Inhalt |
|---|---|
| [weiterfuehrend/rl-anleitung.md](weiterfuehrend/rl-anleitung.md) | **RL-Bedienung** (FPO, nur N1.6): Image, Startcheckpoint, `server_rl_run.sh rl`, Überwachung. Gebaut, kein RL-Trainingslauf; Startpunkt wäre Modell C |
| [weiterfuehrend/reinforcement-learning-plan.md](weiterfuehrend/reinforcement-learning-plan.md) | **RL-Plan**: Algorithmen-Vergleich, Infrastruktur, Status der Bausteine; die Sim-Diagnose der Läufe 25–34 ist in die [Diagnose-Chronik](ergebnisse/diagnose-chronik.md) übergegangen |
| [weiterfuehrend/lokomotion-recherche.md](weiterfuehrend/lokomotion-recherche.md) | **Fortbewegung** (nicht Teil der Aufgabe): warum der Rumpf fest ist, Whole-Body-Control, Unitree-Lokomotions-Stacks, Integrationspfade |
| [weiterfuehrend/](weiterfuehrend/README.md) | Index des Ordners |

## Archiv

Abgeschlossene Pläne, Reviews und Zwischenstände: [archiv/](archiv/README.md). Ältere
Sim-Planungsdokumente liegen weiter unter [simulation/archiv/](simulation/archiv/), weil der Code
auf sie verweist.

## Querschnitt (Training + Simulation)

| Dokument | Inhalt |
|---|---|
| [abgabe-plan.md](abgabe-plan.md) | **Abgabeplan (Stand 2026-11-09)** — Umbau der Markdown-Doku (Einstieg, neue Schnittstellen-Seite, Inventur) und der LaTeX-Ausarbeitung (kürzer, Schnittstellen und Anleitung im Mittelpunkt), Prüfschritte, offene Entscheidungen |
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

Verzeichnisbaum des Repos und `/data`-Layout der Container: [repo-struktur.md](repo-struktur.md).

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
