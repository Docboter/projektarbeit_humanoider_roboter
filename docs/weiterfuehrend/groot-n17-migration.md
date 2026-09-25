# Umstieg GR00T N1.6 → N1.7: Überblick und Migrationsplan

> **TL;DR:** Recherche und Migrationsplan für den Umstieg GR00T N1.6 → N1.7, plus den erreichten
> Umsetzungsstand. N1.7 ist als **paralleler Pfad** gebaut — zweiter Code-Baum `app/Groot-1.7`,
> Laufzeit-Auswahl per `GROOT_VERSION` —; **Trainings-Smoke-Tests am 2026-09-25 auf dem
> IKR-Blackwell bestanden** (1 GPU, 2 GPUs, `TUNE_VISUAL` + Split, Checkpoint-Sweep) ([Ergebnis + Übergabe an KISSKI](#smoke-test-auf-dem-ikr-blackwell-2026-09-25-und-übergabe-an-kisski)),
> noch kein voller Lauf, kein KISSKI-, kein Sim-Lauf. Was sich ändert (gated Cosmos-Reason2-2B-Backbone, Python 3.12/torch 2.9,
> N1.6-Checkpoints nicht ladbar), was N1.6-only bleibt, und die 8 Phasen mit Abnahmekriterien.

Recherche- und Planungsdokument. **Umsetzung als paralleler Pfad begonnen** (Stand 2026-08-19 —
siehe [„Stand der Umsetzung"](#stand-der-umsetzung-2026-08-19) unten): Was ist GR00T N1.7, was
ändert sich gegenüber unserem N1.6-Stand, wie aufwendig ist der Umstieg, und in welchen Phasen
läuft er ab. Operative Referenzen: [Training](../training/README.md) ·
[Simulation](../simulation/README.md) · [Portabilität](../portabilitaet.md) ·
[Diagnose-Chronik](../ergebnisse/diagnose-chronik.md) (Läufe, gegen die ein N1.7-Checkpoint
verglichen wird).


> Stand 2026-08-19. Quellen: Upstream-Repo `NVIDIA/Isaac-GR00T` (lokaler Tag-Diff `n1.6-release..n1.7-release` im Submodul, Rohdateien von `main`), HF-Model-Cards, NVIDIA-Forum. Unser Stand: Fork-Commit `26ad1f7` = `n1.6-release` + 3 Commits; Dockerfile-Pin `9508b49` (2 Commits älter — Drift, siehe §3).

## 0. Kurzfazit

**Ist der Umstieg sehr komplex? — Nein, „mittel".** Kein Neubau, aber auch kein reiner Versions-Bump:

- **Was uns rettet:** Unsere gesamte Pipeline hängt am Tag `NEW_EMBODIMENT` — der ist in N1.7 **unverändert** (Wert `"new_embodiment"`). `FinetuneConfig`/CLI-Flags (`--tune_visual`, `--color_jitter_params`, `--state_dropout_prob`, `--modality_config_path` …) sind **strukturell gleich**. Dataset-Format (LeRobot v2.1 + `meta/modality.json`) **unverändert**. `Gr00tPolicy`, `PolicyServer`, `Gr00tSimPolicyWrapper`, `run_gr00t_server.py --use-sim-policy-wrapper` **existieren weiter**; der ZMQ-Wire-Format ist gleich geblieben (nur Timeout-Härtung). Unser `g1_dex3_config.py` (28-dim State/Action, 16er-Action-Chunk, 4 Kameras) passt in die neuen Maxima (132 dim / 40 steps). Die 3 Fork-Commits (G1_DEX3-Beispiel, Train/Test-Split, v3→v2-Konverter) **mergen laut `git merge-tree` konfliktfrei** auf `n1.7-release` (nur `uv.lock`).
- **Was echten Aufwand macht:** (1) **Backbone-Wechsel Eagle → Cosmos-Reason2-2B (Qwen3-VL)**: das Backbone ist ein **gated HF-Repo**, das *jeder* N1.7-Checkpoint beim Laden nachzieht → HF-Zugang beantragen, Offline-Cache für KISSKI, `download_data.sh`/Dockerfiles anpassen. (2) **N1.6-Checkpoints sind nicht ladbar** (anderes Modellpaket `gr00t_n1d6`→`gr00t_n1d7`, anderes Backbone, 32→16 DiT-Layer) → **alle Läufe neu trainieren**; die bisherigen Ergebnisse (Lauf 1–3, TUNE_VISUAL, Sweep) bleiben N1.6-Ergebnisse. (3) **Vier eigene Skripte koppeln an N1.6-Interna** und müssen nachgezogen werden: `rl_finetune.py` (Replikat von `Gr00tN1d6ActionHead.forward`/Processor-Maske), `policy_latency.py`, `groot_inference_backend.py`/`optimize_groot_inference.py` (Monkeypatch + TensorRT-Export des DiT), `launch_cotrain.py` (hardcoded `eagle_collator`/`model_name`). (4) **Entscheidung EA-Tag vs. GA-`main`**: GA (seit 2026-07-07) verlangt **Python 3.12 + torch 2.9 + flash-attn 2.8.3** → beide Docker-Images müssen neu gebaut werden (Basis-Image bleibt CUDA 12.8).
- **Grobe Schätzung:** ~3–5 Arbeitstage Engineering (Fork-Branch, zwei Images, Skripte, Diagnose-Tools, Doku) **plus** ein Vergleichs-Trainingslauf (KISSKI, ~1 Tag Rechenzeit) **plus** Sim-Eval. Risiko: moderat; größte Unbekannte sind VRAM-Bedarf auf der 32-GB-5090 bei `--tune_visual` und das Verhalten der Qwen3-VL-Bildverarbeitung (native Seitenverhältnisse, Crop 230/Resize 256) im Sim-Real-Gap.
- **Empfehlung — so umgesetzt (Details: [„Stand der Umsetzung"](#stand-der-umsetzung-2026-08-19)):** paralleler Pfad, aber anders als hier ursprünglich vorgeschlagen: statt separater Image-Tags `:n17` wählt **ein** Image beide Codebäume/venvs zur Laufzeit über `GROOT_VERSION` (Build-Arg `GROOT_VERSIONS` steuert, welche Bäume überhaupt gebaut werden). N1.6-Pipeline bleibt unangetastet (weiterhin Default). Der erste N1.7-Lauf soll ein **Baseline-Vergleich** gegen den besten N1.6-Checkpoint (Lauf 3, ckpt 30000) mit identischen Hyperparametern sein — **dieser Vergleichslauf steht noch aus**, ebenso wie jeder Image-Build und jeder N1.7-Trainings-/Sim-Lauf.

## Stand der Umsetzung (2026-08-19)

Umgesetzt wurde N1.7 als **paralleler Pfad neben N1.6** — nicht als Ersatz. Codebäume, Images
und Scripts sind angepasst; **kein Image wurde neu gebaut, und kein N1.7-Trainings- oder
Sim-Lauf hat bisher stattgefunden.** Der Rest dieses Dokuments (§1–§8) bleibt als
Recherche-/Planungsgrundlage stehen; dieser Abschnitt hält den tatsächlichen Umsetzungsstand
je Phase fest.

**Phase 0 — Entscheidungen: getroffen.**
- Pin-Strategie: GA-`main`-Commit als Basis (`376ba89`), nicht der EA-Tag.
- Paralleler Pfad statt Ersetzung: Laufzeit-Auswahl über `GROOT_VERSION` (`1.6`|`1.7`|`auto`),
  N1.6 bleibt Default.
- Submodul-Pfad: **neues** `app/Groot-1.7` **neben** `app/Groot-1.6` (nicht dessen Umbenennung).
- Abweichung von der ursprünglichen Empfehlung oben (separate Image-Tags `:n16`/`:n17`):
  stattdessen **ein** Image mit **beiden** venvs, gesteuert per Build-Arg `GROOT_VERSIONS`
  (Default `"1.6 1.7"`; `GROOT_VERSIONS=1.6` baut nur den alten Baum, halbe Image-Größe).

**Phase 1 — Fork-Branch: fertig.** `app/Groot-1.7` zeigt auf `lucam06/Isaac-GR00T`, Branch
`luca/g1-dex3-n17`, gepinnter Commit `efa0169` (= Upstream `main` `376ba89` + unser
G1_DEX3-Beispiel/Konverter + Train/Test-Split-Patch + eine README-Notiz, erneut angewendet).
`app/Groot-1.6` bleibt unverändert (Branch `luca/g1-dex3`). `.gitmodules` führt beide Submodule.

**Phase 2 — Training-Image: Dockerfile + Scripts fertig, Image nicht gebaut.**
[`Training/Dockerfile`](../../Training/Dockerfile) baut standardmäßig beide Bäume: N1.6-venv
Python 3.10 unter `/app/Groot-1.6/.venv`, N1.7-venv Python 3.12 (uv-verwaltetes CPython unter
`/opt/uv/python`, torch 2.9.0+cu128, flash-attn 2.8.3 cp312-Wheel, transformers 4.57.3,
torchcodec 0.8.0) unter `/app/Groot-1.7/.venv`; separate `ARG GROOT16_COMMIT`/`GROOT17_COMMIT`;
`ENV GROOT_VERSION=1.6 HF_HOME=/data/hf_cache`.
[`Training/update_image.sh`](../../Training/update_image.sh) prüft/aktualisiert beide Pins
(`--update-commit`), kennt `GROOT17_BRANCH` und reicht `GROOT_VERSIONS` als Build-Arg durch.
Alle Trainings-Launcher (`entrypoint.sh`, `download_data.sh`, `run_finetuning*.sh`, `.ps1`,
`setup_and_train_*.sh`, `docker-compose.yml`) lesen `GROOT_VERSION`; die Auflösung
(Codebaum/venv/Modell-Repo/Namespace-Suffix) übernimmt der neue
[`Training/scripts/lib_groot_version.sh`](../../Training/scripts/lib_groot_version.sh).
**Nicht gemacht:** kein tatsächlicher `docker build`. Lokal geprüft wurden nur
`docker build --check` (Lint) und `uv sync --dry-run` des N1.7-`uv.lock` unter Python 3.12
(löst 173 Pakete auf, darunter torch 2.9.0+cu128 und das flash-attn-2.8.3-cp312-Wheel — ohne
etwas zu installieren).

**Phase 3 — lokaler Trainings-Smoke-Test: erledigt am 2026-09-25** (Training mit und ohne
`TUNE_VISUAL`, 2 GPUs, Split, VRAM, Sweep; siehe
[Smoke-Test](#smoke-test-auf-dem-ikr-blackwell-2026-09-25-und-übergabe-an-kisski)). Der Rest
dieses Absatzes ist der überholte Stand vom 2026-08-19. Kein Trainingslauf (mit oder ohne
`TUNE_VISUAL`), keine VRAM-Messung, kein `checkpoint_sweep.py`-Lauf gegen einen echten
N1.7-Checkpoint. `checkpoint_sweep.py` wurde zwar so angepasst, dass es unter beiden venvs
läuft (introspiziert `execution_horizon` vs. `action_horizon`, `decoder_kwargs` vs.
`video_backend`, `EmbodimentTag.resolve`), aber das ist ungetestet.

**Phase 4 — KISSKI: Scripts fertig, kein Lauf.**
[`kisski_submit.sh`](../../Training/kisski_submit.sh) reicht `GROOT_VERSION` durch, bindet für
`1.7` einen zweiten Fork-Clone (`GROOT17_FORK_DIR`, Default
`$KISSKI_PROJECT_DIR/repo-groot-n17`), setzt `HF_HOME=/data/hf_cache` und
`HF_HUB_OFFLINE=1`, und prüft **vor** dem Start auf dem Login-Knoten, ob Modell und gated
Backbone im HF-Cache liegen — sonst Abbruch mit den nötigen
`apptainer exec … huggingface-cli download`-Kommandos.
[`kisski_open_loop_eval.sh`](../../Training/kisski_open_loop_eval.sh) unterstützt
`GROOT_VERSION` (Namespace-Suffix `_n17`). **Nicht gemacht:** kein Vergleichslauf, keine
Sweep-Tabelle, HF-Zugang zum gated Backbone noch nicht bestätigt (muss je Token beantragt
werden).

**Phase 5 — Sim-Image & Inferenz: Scripts fertig, Image nicht gebaut, kein Lauf.**
[`Simulation/Dockerfile.standalone`](../../Simulation/Dockerfile.standalone) baut ebenfalls beide
venvs (N1.7 mit System-Python 3.12 von Ubuntu 24.04 + `python3.12-dev`, `usd-core` installiert,
deepspeed entfernt, Build-Time-Smoke-Test); `ENV GROOT_VERSION=auto`; das fest gesetzte
`GROOT_ROOT` wurde entfernt (die Lib leitet es her).
[`entrypoint_sim.sh`](../../Simulation/scripts/entrypoint_sim.sh) löst `auto` aus dem
Checkpoint auf (`model_type`), lädt bei Bedarf das Backbone einmalig vor und fällt für N1.7
bei `GROOT_INFERENCE_BACKEND=compile|tensorrt` bewusst auf `eager` zurück (das optimierte
Backend hängt am N1.6-DiT: `groot_inference_backend.py`/`optimize_groot_inference.py`/
`run_groot_optimized_server.py` prüfen nur gegen `Gr00tN1d6`).
[`entrypoint_baseline.sh`](../../Simulation/scripts/entrypoint_baseline.sh) bricht bei N1.7
kontrolliert ab (`UNITREE_G1` bedeutet in N1.7 etwas anderes — Sim-Ganzkörper-Tag ohne
Zero-Shot-Kopf im Basismodell; `REAL_G1` als Alternative ist ungeprüft).
[`entrypoint_rl.sh`](../../Simulation/scripts/entrypoint_rl.sh) bricht bei N1.7 ebenfalls ab.
`server_rl_run.sh` sourct die Lib, hält für `docker exec`-Aktionen (`span`, `latency`, `gap`, …)
host-seitig standardmäßig `1.6`, reicht ein **explizit** gesetztes `GROOT_VERSION` an den
Container durch (sonst bleibt `auto` aktiv), verweigert `rl`/`check`/`optimize` für `1.7`, und
`preflight` prüft auch die N1.7-venv.
[`kisski_sim_submit.sh`](../../Simulation/kisski_sim_submit.sh) bindet bei ungesetztem
`GROOT_VERSION` **beide** Fork-Bäume und erkennt die Version aus dem Checkpoint; ein explizit
gesetzter Wert bindet nur den passenden Baum. `--no-flash-attn` wird nur für `1.6` übergeben —
der N1.7-Fork kennt das Flag noch nicht, und flash-attn ist auf der Turing-RTX-5000 der
`jupyter`-Partition ohnehin nicht unterstützt → **N1.7-Sim auf KISSKI ist ungetestet und
vermutlich blockiert**, bis das Flag auf den N1.7-Fork-Branch portiert ist.
`kisski_rl_submit.sh` und `kisski_robocasa_ref_submit.sh` brechen bei `GROOT_VERSION=1.7`
kontrolliert ab (RL nicht portiert; das Embodiment `GR1` ist in N1.7 entfernt).
**Nicht gemacht:** kein `docker build` des Sim-Images, kein Sim-Eval-Lauf mit einem
N1.7-Checkpoint.

**Phase 6 — RL & Co-Training: teilweise.** Co-Training: neuer
[`Training/scripts/launch_cotrain_n17.py`](../../Training/scripts/launch_cotrain_n17.py)
(Spiegel der N1.7-`launch_finetune.py` mit derselben Mix-Ratio-Ergänzung wie beim
N1.6-Pendant); `run_finetuning_cotrain.sh` ruft ihn bei `GROOT_VERSION=1.7` auf. **Kein
Smoke-Test gefahren.** RL: **nicht portiert.** Das Isaac-Sim-Python im Sim-Image enthält
weiterhin nur die N1.6-`gr00t`-Installation — `rl_finetune.py`, `entrypoint_rl.sh`,
`server_rl_run.sh rl/check/optimize` und `kisski_rl_submit.sh` brechen bei N1.7 kontrolliert
mit einer deutschen Fehlermeldung ab, statt etwas Falsches zu versuchen.

**Phase 7 — Umschalten & Aufräumen: nicht begonnen.** `GROOT_VERSION`-Default bleibt `1.6`,
kein Retag, keine Pfad-Umbenennung.

**Phase 8 — Doku-Pass: dieser Durchgang.** Die in §8 unten genannten Dateien (plus einige mehr,
die beim eigentlichen Bauen entstanden sind, z. B. `co-training.md`, `fehlerbehebung.md`) wurden
mit diesem Stand aktualisiert; Changelog-Eintrag in [historie.md](../historie.md).

**Nachtrag 2026-09-25 — Upstream-Merge, Datensatz-Kompatibilität, ein Image je Generation.**
Weiterhin gilt: **kein Image gebaut, kein N1.7-Lauf.** Alles Folgende ist statisch geprüft
(gegen die gepinnten Fork-Commits N1.6 `9508b49` aus `Training/Dockerfile` und N1.7 `efa0169`).

- **Upstream eingemergt:** `training-luca-IKR-IS6.0` (Stand 2026-09-23, 69 Commits) ist per
  **Merge** (kein Rebase) in diesen Branch geholt — der N1.7-Branch bleibt eigenständig, der
  spätere Rück-Merge wird kleiner. Dortige Umbenennungen (`Dockerfile.vastai` →
  `Dockerfile.standalone`, `--vastai` → `--standalone`, `setup_and_train_dockerhub_pull.sh`,
  `sim-eval-anleitung.md`) sind nachgezogen; `GROOT_VERSION` läuft auch durch die neuen
  Upstream-Pfade (`run.sh`, `lib_resume_guard.sh`, `--gradient_accumulation_steps`).
- **Datensatz `unitreerobotics/G1_Dex3_BlockStacking_Dataset` passt ohne Umbau** (LeRobot v3.0,
  301 Episoden, 4 × AV1-Video 640×480, 28-dim State/Action): Konverter v3→v2.1,
  `g1_dex3_config.py` und `modality_4cam.json` sind in beiden Fork-Bäumen byte-identisch und
  nutzen die N1.7-API korrekt; jedes Flag der Launcher existiert unverändert in der
  N1.7-`FinetuneConfig`; N1.7 liest `info.json["splits"]` im Format von `lib_split.sh`.
  Die mitgelieferte `stats.json` hat keine `q01`/`q99` — N1.7 rechnet `stats.json` und
  `relative_stats.json` beim ersten Start auf Rank 0 selbst nach (`DatasetFactory.build`), ein
  zusätzlicher Datensatz-Durchlauf. N1.6 ignoriert die Zusatzschlüssel; derselbe Datensatz unter
  `/data` bleibt für beide Generationen nutzbar.
- **Entscheidung: N1.7 trainiert mit den NVIDIA-Standardwerten.** Ziel ist das bestmögliche
  N1.7-Ergebnis, nicht ein streng gleicher Vergleich. Betroffen:
  - **State-Dropout** (`state_dropout_prob`: im Training wird der State — die 28 Gelenkwinkel —
    zufällig ausgeblendet, das Modell muss dann allein aus den Bildern handeln): N1.6-Default
    `0.0`, N1.7-Default `0.2`. `STATE_DROPOUT_PROB` ist jetzt **leer = Modell-Default**; ein
    gesetzter Wert wird immer übergeben (vorher wurde `0` verschluckt, unter N1.7 ließ sich der
    Dropout also nicht abschalten).
  - **Normalisierung** (`use_percentiles`, nur N1.7, Default `True`): Perzentile `q01`/`q99`
    statt Min/Max wie bei N1.6. Nicht als Env-Var verdrahtet.
  - Folge: N1.6 vs. N1.7 vergleicht Modell **und** Trainingsrezept. Für die Ausarbeitung
    vermerkt in `latex/chapters/03_methoden.tex` (Kommentar `TODO(N1.7, …)`).
- **Ein Image je Generation:** `Training/update_image.sh --groot=1.6|1.7|both` (ohne Flag am
  Terminal: Menü, Spec `tools/menu/image-build.spec`; sonst Default `1.6`). `both` ist nie
  Default. Tags: `:<branch>-n16|-n17|-n16-n17`, `:<zeitstempel>-<gen>`; mit `--push-latest`
  N1.6 → `:latest` (unverändert), N1.7 → `:latest-n17`. `Training/Dockerfile` baut ohne
  Build-Arg nur N1.6; `GROOT_VERSION_DEFAULT` setzt `GROOT_VERSION`/venv/`PATH` im Image.
  Verbraucher folgen `GROOT_VERSION`: `setup_and_train_dockerhub_pull.sh` zieht bei `1.7`
  `:latest-n17` und prüft nach dem Pull das Label `de.humrob.groot-versions`;
  `docker-compose.yml` nutzt `${GROOT_IMAGE_TAG:-latest}`; KISSKI (`kisski_submit.sh`,
  `kisski_open_loop_eval.sh`) wählt bei `1.7` die SIF `projekt-humanoider-roboter-n17.sif`;
  der Entrypoint bricht klar ab, wenn die venv der gewählten Generation im Image fehlt.
  Das Sim-Image (`update_sim_image.sh`) baut weiterhin standardmäßig beide.
- **Früher Zugriffs-Check für das gated Backbone:** `groot_check_backbone_access` in
  `lib_groot_version.sh` (HEAD-Request auf `config.json`: 401 = Token fehlt/ungültig, 403 = nicht
  freigeschaltet → Abbruch; offline/Netzfehler → nur Warnung). Aufgerufen vom Docker-Launcher
  vor dem Image-Pull, vom Entrypoint vor allen Downloads (nur wenn das Backbone noch nicht im
  Cache liegt) und von `kisski_menu.sh` auf dem Login-Node vor `sbatch` — dort zusammen mit der
  Cache-Prüfung, die `kisski_submit.sh` sonst erst im Job (nach der Queue-Wartezeit) macht.
- **Offen / ungetestet:** AV1-Dekodierung über torchcodec 0.8 mit dem FFmpeg 4.4 aus Ubuntu
  22.04 (der Build-Smoke-Test importiert torchcodec nur); VRAM auf der 32-GB-5090; ob
  `nvidia/GR00T-N1.7-3B` gegen `efa0169` sauber lädt. Der Submodul-Pointer `app/Groot-1.6`
  (`26ad1f7`) ist im Fork nicht mehr abrufbar (`git submodule update` scheitert; der Build nutzt
  `GROOT16_COMMIT=9508b49` und ist nicht betroffen). `setup_and_train_container_build.sh`
  „baut“ per `docker compose build`, aber `docker-compose.yml` hat seit `cb388a1` keinen
  `build:`-Abschnitt mehr — der Schritt ist wirkungslos (bestand schon vorher).

## Smoke-Test auf dem IKR-Blackwell (2026-09-25) und Übergabe an KISSKI

> **Kurz:** Alle vier N1.7-Tests auf dem IKR-Server sind **bestanden**: Einzel-GPU-Training
> (200 Steps), Multi-GPU mit `torchrun` + DeepSpeed (2 GPUs), `TUNE_VISUAL=1` mit 80/20-Split
> sowie Laden + Open-Loop-Auswertung der feinjustierten Checkpoints (`checkpoint_sweep.py`).
> Getestet ist damit die ganze Kette: Image → Download (inkl. gated Backbone) → Konvertierung →
> Training → Checkpoint → Inferenz. **Offen ist nur, was es ausschließlich auf KISSKI gibt**
> (Apptainer, Offline-Cache, A100/H100) und die Sim. Durchgeführt von Matthias; Übergabe an Luca
> für den vollen KISSKI-Lauf.

### Gemeinsamer Aufbau

| | |
|---|---|
| Rechner | `ikr-ki-server-01`, 2 × RTX PRO 6000 Blackwell Max-Q (96 GB) |
| Image | `lucam03/projekt-humanoider-roboter:training-luca-IKR-IS6.0-GN1.7-n17` — auf dem Server mit `update_image.sh --groot=1.7` gebaut, gestartet mit `--skip-pull` |
| Skripte | per Bind-Mount aus dem Repo (Branch `training-luca-IKR-IS6.0-GN1.7`, inkl. `22c3183`) |
| Rezept | Augmentierung an (Color-Jitter), LR 1e-4, Warmup 5 %, `GLOBAL_BATCH_SIZE=8`, `STATE_DROPOUT_PROB` leer = N1.7-Default 0.2, ohne W&B |
| Container | Test 0 in `groot-n17-smoke`; Tests 1–3 in `groot-n17-tests` (`SKIP_TRAIN=1`, dann von Hand in der Container-Shell) |

### Die Tests im Überblick

| # | Test | Aufruf (im Container, außer Test 0) | Ergebnis |
|---|---|---|---|
| 0 | Einzel-GPU, Standard | `GROOT_VERSION=1.7 … MAX_STEPS=200 SAVE_STEPS=100 NUM_GPUS=1 ./Training/setup_and_train_dockerhub_pull.sh --skip-pull` | ✅ 200/200, Loss 1,283 → 1,031, `checkpoint-100/200` |
| 1 | Multi-GPU (`torchrun` + DeepSpeed ZeRO-2) | `NUM_GPUS=2 MAX_STEPS=50 SAVE_STEPS=50 EXPERIMENT_NAME=smoke_2gpu bash /scripts/run_finetuning.sh` | ✅ 50/50, Ranks 0 und 1 aktiv, Loss 1,271 → 1,091, `checkpoint-50` |
| 2 | `TUNE_VISUAL=1` + `TRAIN_TEST_SPLIT=1` + VRAM | `TRAIN_TEST_SPLIT=1 MAX_STEPS=100 SAVE_STEPS=50 bash /scripts/run_finetuning_vision.sh` | ✅ 100/100, Split 240/61 Episoden, Loss 1,275 → 1,181, **VRAM-Spitze 59 322 MiB** |
| 3 | Checkpoint laden + Open-Loop-Eval | `uv run --no-sync python /scripts/checkpoint_sweep.py --run-dir …/blockstacking_vision_n17 --split test --num-trajectories 2 --steps 150` | ✅ beide Checkpoints geladen, MSE 0,2527 → 0,2494 |

### Messwerte

| Messgröße | Wert |
|---|---|
| Zugangs-Check `nvidia/Cosmos-Reason2-2B` | bestanden (Launcher **und** Entrypoint) |
| Download | GR00T-N1.7-3B (~7 GB) 2:35 min · Cosmos-Reason2-2B (4,9 GB) 2:00 min · Datensatz 4:29 min |
| v3→v2.1-Konvertierung | ~1 min, fehlerfrei; `modality_4cam.json` → `modality.json` |
| Modell | 3 144 016 000 Parameter, davon **1 620 515 968 trainierbar (51,5 %)** ohne `TUNE_VISUAL`; DiT 1,09 Mrd., SelfAttention 0,20 Mrd. |
| Datensatz | 301 Episoden, 276 681 Frames, 271 Shards; `stats.json`/`relative_stats.json` beim ersten Start nachgerechnet (~27 s), danach aus dem Cache |
| Anlauf | ~100 s Shard-Caching bis zum ersten Step (1 GPU, 8 Worker); ~155–200 s bei 2 GPUs (2 × 8 Worker) |
| **Durchsatz (eingeschwungen, bs 8)** | Standard 1 GPU: **3,1–3,4 Steps/s** · Standard 2 GPUs: **2,75 Steps/s** · `TUNE_VISUAL` 1 GPU: **1,96 Steps/s** |
| grad_norm | 0,4–1,05, einmal 3,43 im ersten Log-Schritt von Test 1 (Warmup); keine NaN/Inf |
| **VRAM** (`TUNE_VISUAL=1`, bs 8, 1 GPU, `nvidia-smi memory.used`) | **59 322 MiB ≈ 58 GiB** — inkl. PyTorch-Cache, also eine Obergrenze |
| Split | `lib_split.sh` hält die Episoden 240–300 (61) zurück; `checkpoint_sweep.py` findet und nutzt sie |
| Open-Loop (Test 3, 2 Test-Episoden, 150 Steps) | ckpt-50: MSE 0,2527 / MAE 0,3742 · ckpt-100: MSE 0,2494 / MAE 0,3710 — nach 100 Steps **nicht aussagekräftig**, belegt nur, dass Laden und Inferenz funktionieren |
| Präzision | bf16 bestätigt (`Casting fp32 inputs back to torch.bfloat16 for flash-attn compatibility`) |

Zur Einordnung: 2 GPUs bei globaler bs 8 (4 je Karte) sind **nicht** schneller als eine. Bei so
kleiner Batch je Karte überwiegt der Synchronisationsaufwand. Test 1 prüft nur, dass der
Multi-GPU-Pfad funktioniert, nicht wie schnell er ist. Auf KISSKI mit größerer Batch sieht das
anders aus. Grob hochgerechnet (1 GPU, bs 8): Standard ~2,5–3 h, `TUNE_VISUAL` ~4,5 h für
30 000 Steps.

**VRAM-Hinweis für die Batch-Größe:** N1.7 mit `TUNE_VISUAL=1` braucht bei 8 Samples auf einer
Karte schon ~58 GiB. Auf der 32-GB-RTX-5090 passt das **nicht** (N1.6 lag dort bei ~31 GB).
Auf einer A100 80 GB sind 8 Samples je GPU sicher. Mehr je GPU ist ungetestet, DeepSpeed ZeRO-2
entlastet aber etwas, weil es die Optimizer-Zustände auf die Karten verteilt.

**Beobachtung (Test 1, harmlos):** Nach `Training completed!` meldete der Abbau der Dataloader
`RuntimeError: DataLoader worker (pid …) is killed by signal: Killed` („Exception ignored“).
Das Modell war da schon gespeichert, und der Launcher meldete Erfolg. Falls das auf KISSKI
**während** des Trainings auftritt, ist es vermutlich der Host-RAM (2 Ranks × 8 Worker cachen
Shards). Dann `DATALOADER_WORKERS` senken.

Harmlose Warnungen im Log, keine Aktion nötig: Flash-Attention-2-Hinweis „current dtype is
float32“ (Laden in fp32, Training läuft in bf16, s. o.), `albumentations`-Update,
`image_crop_size … deprecated`, TF32-API, `huggingface-cli download is deprecated`,
`Could not estimate the number of tokens`.

### Gefundener und behobener Fehler

Der erste Versuch von Test 0 brach im ersten Step mit `StopIteration` in
`gr00t_n1d7.py:618` (`next(iter(self.parameters())).device`) innerhalb von
`torch/nn/parallel/data_parallel.py` ab. Ursache: `--gpus all` zeigt dem Container beide Karten,
`NUM_GPUS=1` startet aber kein `torchrun`. Dann wickelt der HF-Trainer das Modell selbst in
`nn.DataParallel`, und dessen Replika hat keine Parameter. **Fix `22c3183`:** Die drei
Trainings-Launcher setzen bei `NUM_GPUS=1` und mehreren sichtbaren Karten
`CUDA_VISIBLE_DEVICES=0` (ein vorgegebener Wert, z. B. von SLURM, bleibt). Das betrifft nur
Rechner mit mehreren GPUs ohne SLURM. Auf KISSKI setzt SLURM die Variable selbst.

### Was noch offen ist

| Punkt | Warum relevant | Wo prüfbar |
|---|---|---|
| Apptainer-SIF, `HF_HUB_OFFLINE=1`, Cache auf VAST | Compute-Knoten haben kein Internet; N1.7 lädt das Backbone bei jedem Modell-Laden | nur KISSKI |
| A100/H100, Batch > 8 je GPU | VRAM-Grenze mit `TUNE_VISUAL` | nur KISSKI |
| Aussagekräftige Validierung (voller Lauf + Sweep) | Vergleich N1.7 vs. N1.6 (Lauf 3, ckpt 30000) | KISSKI |
| Sim-Eval mit N1.7-Checkpoint | Phase 5; Sim-Image noch nicht gebaut | IKR |

### Übergabe an KISSKI (Luca)

Die operativen Schritte stehen in [kisski-hpc.md](../training/kisski-hpc.md), jeweils in den
Abschnitten „Nur für `GROOT_VERSION=1.7`“. Reihenfolge:

1. **SIF:** `kisski_submit.sh` erwartet für N1.7 `projekt-humanoider-roboter-n17.sif`. Die
   Doku zieht `:latest-n17`, der hier getestete Build trägt aber den Branch-Tag
   `:training-luca-IKR-IS6.0-GN1.7-n17`. Vor dem Pull auf Docker Hub prüfen, welcher Tag
   existiert, und genau das getestete Image nehmen:
   `apptainer pull $HOME/images/projekt-humanoider-roboter-n17.sif docker://lucam03/projekt-humanoider-roboter:<tag>`.
2. **Fork-Clone** `luca/g1-dex3-n17` nach `$KISSKI_PROJECT_DIR/repo-groot-n17` anlegen
   (kisski-hpc.md, Schritt 2b).
3. **Modell + gated Backbone auf dem Login-Knoten** in den VAST-Cache laden. Die Compute-Knoten
   haben kein Internet, der Job läuft mit `HF_HUB_OFFLINE=1`. Der Token braucht die
   Cosmos-Freigabe **auf dem eigenen Konto**. Test: `curl -I -H "Authorization: Bearer $HF_TOKEN"
   https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json` muss `200` liefern.
4. **Zuerst ein kurzer Job** (`GROOT_VERSION=1.7 TUNE_VISUAL=1 MAX_STEPS=200 SAVE_STEPS=100`,
   gleiche GPU-Zahl und Batch wie später). Er prüft Apptainer, Offline-Cache und VRAM auf der
   Zielkarte in einem Durchlauf, bevor 48 h Walltime verbraucht werden. Namespace:
   `blockstacking_vision_n17` bzw. `blockstacking_n17`. Der Resume-Schutz bricht ab, wenn dort
   schon Checkpoints liegen, also vor dem echten Lauf `EXPERIMENT_NAME` ändern oder das
   Verzeichnis leeren.
5. **Vergleichslauf** wie in Phase 4 geplant (`TUNE_VISUAL=1`, `TRAIN_TEST_SPLIT=1`), danach
   `kisski_open_loop_eval.sh` bzw. den Sweep auf den Test-Episoden fahren.

Für einen Coding-Agent: Einstieg ist dieser Abschnitt. Die relevanten Code-Pfade sind
`Training/kisski_submit.sh`, `Training/kisski_menu.sh` (Login-Knoten; prüft Zugang und Cache vor
`sbatch`), `Training/scripts/lib_groot_version.sh`, `Training/scripts/run_finetuning*.sh`,
`Training/scripts/checkpoint_sweep.py`.

## 1. Was ist GR00T N1.7?

| | |
|---|---|
| **Zeitlinie** | `n1.6-release` 2026-04-14/15 · `n1.7-release` (Early Access) **2026-04-17/18** · `n1.6.1-release` 2026-04-22 (Hotfix, Seitenzweig) · **GA-Commit `1a1837f` „GR00T N1.7 General Release" 2026-07-07 auf `main`** — es gibt **keinen GA-Tag**, GA = `main`. Jüngster `main`-Commit: `376ba89` (2026-08-10). N1.6-Code lebt jetzt auf Branch `n1d6`. |
| **HF-Checkpoints** | `nvidia/GR00T-N1.7-3B` (Base, ~3 B Param.), `-LIBERO`, `-DROID`, `-SimplerEnv-Bridge`, `-SimplerEnv-Fractal`, `-ApplePnP-V1`. (`GR00T-H-N1.7` = **medizinische** Robotik, nicht Humanoid.) |
| **Backbone** | **Cosmos-Reason2-2B (Qwen3-VL-Architektur)** statt Eagle-Block2A-2B-v2. „Flexible resolution, native aspect ratio without padding." Backbone wird **nicht mehr im Repo mitgeliefert**, sondern vom HF-Hub geladen (`Qwen3VLForConditionalGeneration.from_pretrained("nvidia/Cosmos-Reason2-2B")`, `gr00t/model/modules/qwen3_backbone.py`) — **gated Repo**, Zugang auf der Model-Page beantragen; ohne Zugang `GatedRepoError/401` bei *jedem* Laden eines N1.7-Checkpoints. |
| **Action-Head** | Weiter Flow-Matching-DiT, aber **16 statt 32 Layer**; `max_state_dim`/`max_action_dim` 29→**132**; `action_horizon` 16→**40**; neu: `state_history_length`, `vl_self_attention`-Block, **RTC** (Real-Time Chunking: `get_action(..., options={rtc_overlap_steps, …})`). Entfernt: `state_additive_noise_scale`. Defaults: `select_layer` 16→12, `tune_top_llm_layers` 4→**0**, `load_bf16` True→False, `state_dropout_prob` 0.0→0.2, `override_pretraining_statistics` False→True. |
| **Pretraining** | Robot-Daten + **20 000 h EgoScale-Egovideo (Mensch)**, relativer EEF-Action-Space über Mensch/Roboter hinweg. NVIDIA: „comparable performance to N1.6, with improved generalization and language-following". |
| **Embodiment-Tags** | Enum **neu geschnitten**: `GR1`, `ROBOCASA_PANDA_OMRON`, `BEHAVIOR_R1_PRO` weg; `UNITREE_G1` heißt jetzt `"unitree_g1_full_body_with_waist_height_nav_cmd"` (Sim, Ganzkörper); neu `REAL_G1` (`real_g1_relative_eef_relative_joints`, Pretrain-Tag, Zero-Shot), `XDOF*`, `REAL_R1_PRO_SHARPA*`, auf `main` zusätzlich `UNITREE_G1_SONIC` (Ganzkörper via GEAR-SONIC, Repo `NVlabs/GR00T-WholeBodyControl`). **`NEW_EMBODIMENT = "new_embodiment"` unverändert.** Neu: `EmbodimentTag.resolve()` (case-insensitive). |
| **Code** | Gleiches Repo/Package `gr00t/`; Modellpaket `gr00t_n1d6`→`gr00t_n1d7`, `processing_gr00t_n1d7.py` („simplified data processing pipeline"); Video-Key-Auto-Mapping im Loader; Multi-Dataset via `--dataset-path a:b` + `ds_weights_alpha` (GA); `save_only_model`, `resume_from_checkpoint`, `use_percentiles`, `crop_fraction`, `shortest_image_edge` neu in `FinetuneConfig`; vollständiger **ONNX/TensorRT-Full-Pipeline-Export** (`scripts/deployment/`, 35.9 Hz auf H100). Rollout-Flag `--action-horizon` → `--execution-horizon` (wir nutzen bereits letzteres). |
| **Abhängigkeiten** | EA-Tag `n1.7-release`: Python **3.10**, torch 2.7.1, flash-attn 2.7.4.post1, **transformers 4.57.3** (Qwen3-VL), CUDA 12.8 → passt fast 1:1 auf unser Training-Image. **GA `main`:** Python **3.12**, **torch 2.9.0**, torchvision 0.24, **flash-attn 2.8.3**, torchcodec 0.8 (einziger Video-Backend, FFmpeg < 8), triton 3.5, transformers 4.57.3, CUDA 12.8. |
| **Hardware** | Inferenz 16 GB+; Fine-Tuning 40 GB+ empfohlen, Default (Projector + DiT) „unter ~35 GB", `--tune-visual`/`--tune-llm` 80 GB+ empfohlen. Eager ~7.8 Hz (L40) … 12.8 Hz (RTX Pro 6000); TRT bis 35.9 Hz. |
| **Lizenz** | Code Apache 2.0, Gewichte NVIDIA Open Model License (kommerziell nutzbar); `GR00T-N1.7-3B` selbst nicht gated, **Backbone `Cosmos-Reason2-2B` gated**. |
| **In-Training-Eval** | Weiterhin tot (`factory.py` asserts `eval_strategy == "no"` auch auf `main`) → `checkpoint_sweep.py` bleibt nötig. |

## 2. Unterschiede N1.6 → N1.7, die uns betreffen

| Bereich | N1.6 (wir) | N1.7 | Auswirkung |
|---|---|---|---|
| Basis-Modell-ID | `nvidia/GR00T-N1.6-3B` → `/data/models/GR00T-N1.6-3B` | `nvidia/GR00T-N1.7-3B` | `download_data.sh:28`, `entrypoint.sh:131`, `run_finetuning*.sh` (`MODEL_PATH`), `.ps1`, Robocasa-Skripte |
| Backbone-Bezug | Eagle im Repo vendored (5 MB Config/Tokenizer), Gewichte im Checkpoint | `nvidia/Cosmos-Reason2-2B` vom Hub (gated) bei jedem Laden | HF-Zugang je Token; `HF_HOME` auf `/data`; KISSKI: Vorab-Download auf Login-Knoten + `HF_HUB_OFFLINE=1`; Sim-Image (vast.ai) ebenso |
| Embodiment-Tag | `NEW_EMBODIMENT` | `NEW_EMBODIMENT` | **keine Änderung** |
| Modality-Config | `examples/G1_DEX3/g1_dex3_config.py` (`register_modality_config`) | gleiche API | keine Änderung; prüfen, ob `override_pretraining_statistics=True`/`use_percentiles=True` die Normalisierung ändern (`check_action_norm.py`) |
| Dataset | LeRobot v2.1 + `modality.json`, v3→v2-Konverter im Fork | gleich; Loader toleranter | keine Änderung; Split-Patch (`lib_split.sh` + Fork-Patch in `lerobot_episode_loader.py`) nach Rebase **manuell gegenlesen** (gleiche Methode umgebaut) |
| Trainings-CLI | `launch_finetune.py` + Flags | gleiche Flags; `embodiment_tag` jetzt `str` | keine Änderung in `run_finetuning*.sh`; `launch_cotrain.py` neu von der N1.7-Vorlage ableiten (Zeilen 135–136 hardcoded Eagle!) |
| Checkpoints | `gr00t_n1d6`, 32-Layer-DiT | `gr00t_n1d7`, 16-Layer-DiT | **alle bestehenden Checkpoints bleiben N1.6-only**; Neu-Training nötig |
| Inferenz-Server | `run_gr00t_server.py --model-path … --embodiment-tag NEW_EMBODIMENT --use-sim-policy-wrapper --port` | gleiche Flags (`main` verifiziert) | keine Änderung in `entrypoint_sim.sh:242-248` |
| ZMQ-Client (vendored `client.py`) | msgpack, `action.<key>` (1,16,7) | gleich; `CHUNK_SIZE=16` kommt aus unseren `delta_indices` | keine Änderung, aber Smoke-Test (`assert shape == (16,28)` in `run_g1_dex3_sim_eval.py:561`) |
| Optimierte Inferenz (`ad0127d`) | Monkeypatch `policy.model.action_head.model.forward`, TRT-Export des DiT | DiT anders (16 Layer, RTC-Optionen), Upstream bringt eigenen Full-Pipeline-TRT | Re-Export + Paritätstest; Option: Upstream-TRT-Pipeline statt eigener |
| RL (`rl_finetune.py`) | Replikat von `Gr00tN1d6ActionHead.forward` / Processor-Maske (`processing_gr00t_n1d6.py:240-248, 339-341`) | `Gr00tN1d7ActionHead`, `state_history_length`, `options`-Dict | **Neu abgleichen** — höchster Einzelaufwand |
| Deps/Images | CUDA 12.8, Py 3.10, torch 2.7.1, flash-attn 2.7.4 | EA: nur transformers-Bump · GA: Py 3.12, torch 2.9, flash-attn 2.8.3 | beide Dockerfiles; vast.ai-Basis (`isaac-lab:3.0.0-beta2-post1`, Ubuntu 24.04) hat Python 3.12 nativ |
| Docs | 23 Dateien nennen N1.6 | — | Doku-Pass (Tabelle in §5 Phase 8) |

## 3. Bestandsaufnahme unserer N1.6-Kopplung (klassifiziert)

**A — unverändert nutzbar (0 Aufwand):** `examples/G1_DEX3/g1_dex3_config.py`, `modality_*.json`, `lib_split.sh` (patcht nur `meta/info.json`), `run_finetuning.sh` / `_vision.sh` / `_cotrain.sh` (nur `MODEL_PATH`-Default), `entrypoint.sh` (nur `MODEL_DIR`), `kisski_submit.sh` (Bind-Mounts bleiben; Kommentar Zeile 345 „Container hat nur gr00t_n1d7-Code, Modell ist aber N1.6" ist mit N1.7 endlich stimmig — **vorher verifizieren, was das SIF heute enthält**), `client.py` / `client_g1.py` (vendored, protokollgleich), `check_action_norm.py` (stdlib, 28-dim-Layout), `entrypoint_sim.sh`-Serverstart, `server_rl_run.sh`, `camera_geometry.py`, Sim-Env.

**B — kleine Anpassung (Zeilen):** `download_data.sh` (+ Cosmos-Backbone in HF-Cache), `Training/Dockerfile:55-56` + `Simulation/Dockerfile.standalone` (Clone-Branch/Pin; `update_image.sh` kann den Pin per `--update-commit` setzen), `.gitmodules` (Branch), `checkpoint_sweep.py` (Import-Check, `EmbodimentTag`-Resolve), `finger_span_openloop.py`, `dump_unitree_g1_dims.py`, Robocasa-Skripte (Modell-ID; **Achtung:** `run_robocasa_ref_eval.sh:51` nutzt `RC_EMBODIMENT_TAG=GR1` — in N1.7 **entfernt**, ebenso das Robocasa-Env im EA-Tag → Referenz-Eval bleibt N1.6 oder entfällt), **`entrypoint_baseline.sh:70,163` + `g1_gripper_sim/` (Stock-G1-Baseline, `UNITREE_G1`, 16-dim Actions): in N1.7 ist `UNITREE_G1` ein *Posttrain*-Tag mit anderer Semantik (Sim-Ganzkörper, `ego_view`, Waist/Nav-Cmds, 50er-Chunk) und **ohne Zero-Shot-Kopf im Base-Modell** → Baseline-Eval in heutiger Form **bricht**; Ersatzkandidat ist der Pretrain-Tag `REAL_G1` (Keys prüfen, §6), `entrypoint_sim.sh:236-241` (Flash-Attn-Pflichthinweis ist N1.7-spezifisch zu prüfen: Qwen3-Backbone fällt auf `sdpa` zurück, wenn flash-attn fehlt).

**C — Neubau/Abgleich (Tage):** `launch_cotrain.py` (Kopie der **N1.7**-`launch_finetune.py` mit nur dem `datasets`-Block; Upstream-Multi-Path reicht nicht, da wir pro Datensatz `mix_ratio` brauchen), `rl_finetune.py` (Action-Head-Forward/Maske/Decode gegen `gr00t_n1d7` + `processing_gr00t_n1d7.py` neu verifizieren), `policy_latency.py` (greift `Gr00tN1d6ActionHead` an), `groot_inference_backend.py` + `optimize_groot_inference.py` + `run_groot_optimized_server.py` (TRT-DiT-Export für neue Architektur; Alternative Upstream-Pipeline `scripts/deployment/export_onnx_n1d7.py` + `build_trt_pipeline.py`).

## 4. Aufwand & Risiken

| Posten | Aufwand | Risiko |
|---|---|---|
| Fork-Branch auf N1.7 rebasen (3 Commits) | ½ Tag | niedrig (merge-tree: nur `uv.lock`), Split-Patch gegenlesen |
| Training-Image (GA-Stack Py 3.12/torch 2.9) bauen + pushen | ½–1 Tag (Build ~30–60 min, ggf. 2 Iterationen) | mittel (flash-attn-Wheel cp312/torch2.9, torchcodec/FFmpeg) |
| HF-Gating + Offline-Cache (lokal, KISSKI, vast.ai) | ½ Tag | niedrig, aber **blockierend** ohne Zugang |
| Host-Skripte/Env-Vars/Downloads | ½ Tag | niedrig |
| Smoke-Test lokal (5090, bs klein) + VRAM-Messung | ½ Tag | mittel (VRAM bei `--tune_visual` unbekannt; Default-Finetune ist mit `tune_top_llm_layers=0` eher **kleiner** als N1.6) |
| KISSKI-Vergleichslauf (gleiche HP wie Lauf 3) + Sweep | ~1 Tag Rechenzeit, ½ Tag Auswertung | niedrig |
| Sim-Image (vast.ai/Server) + Sim-Eval des N1.7-Checkpoints | 1 Tag | mittel (Bildvorverarbeitung anders → Domain-Gap neu messen: `measure_domain_gap.py` ist SigLIP-basiert und modell­unabhängig, `finger_span_openloop.py` modellabhängig) |
| Diagnose-/RL-Tools nachziehen (C) | 1–2 Tage | mittel–hoch (Interna) |
| Doku (23 Dateien, CLAUDE.md, README) | ½ Tag | niedrig |

## 5. Migrationsplan (Phasen, jeweils mit Abnahmekriterium)

### Phase 0 — Entscheidungen & Vorbereitung (kein Code)
1. **Pin-Strategie festlegen** (Empfehlung: **GA = fester `main`-Commit**, z. B. `376ba89` 2026-08-10; Fallback: EA-Tag `n1.7-release`, wenn der Py-3.12/torch-2.9-Build Probleme macht — EA ist explizit „best-effort, keine Stabilitätsgarantie" und hat weder GA-Fixes noch Multi-Dataset/`resume_from_checkpoint`).
2. **HF-Zugang** für `nvidia/Cosmos-Reason2-2B` beantragen — für **jeden** verwendeten Token (lokal, KISSKI, vast.ai). Prüfen: `huggingface-cli download nvidia/Cosmos-Reason2-2B --dry-run` o. ä.
3. **Submodul-Pfad:** `/app/Groot-1.6` ist in 54 Dateien/156 Stellen verdrahtet (`GROOT_ROOT`-Default in 10 Skripten, Dockerfiles, CLAUDE.md). Empfehlung: **Pfad vorerst beibehalten** (Blast-Radius), Umbenennung nach `app/Groot` als separater Folge-Commit.
4. N1.6-Stand einfrieren: Image-Tags `lucam03/projekt-humanoider-roboter:n16` und `...-sim:n16` (Retag von `:latest`), damit `:latest` später N1.7 werden kann.
- **Abnahme:** Entscheidungen dokumentiert (in dieser Datei, Abschnitt „Entscheidungslog"), HF-Zugang bestätigt.

### Phase 1 — Fork-Branch `luca/g1-dex3-n17` in `lucam06/Isaac-GR00T`
1. `git fetch upstream` (Upstream-Remote existiert im Submodul), neuen Branch vom gewählten Pin abzweigen.
2. Die 3 Fork-Commits cherry-picken (`c047ce2` G1_DEX3-Beispiel + Konverter, `9508b49` Split, `26ad1f7` Parameter); `uv.lock` nicht übernehmen, sondern `uv lock` neu (Py 3.12).
3. Split-Patch in `gr00t/data/dataset/lerobot_episode_loader.py` manuell gegen die N1.7-Fassung prüfen (Video-Key-Mapping/Modality-Filter sitzen in derselben Methode); `factory.py`/`sharded_single_step_dataset.py` sind trivial.
4. Upstream-Tests für SO100/NEW_EMBODIMENT lokal (CPU) laufen lassen: `pytest tests/ -m 'not gpu'`.
5. `.gitmodules` → `branch = luca/g1-dex3-n17`, Submodul-Pointer im Hauptrepo umhängen.
- **Abnahme:** `python -c "import gr00t.model.gr00t_n1d7; from examples.G1_DEX3.g1_dex3_config import *"` im neuen venv; Dataset-Loader lädt unseren Datensatz mit `split=train/test`.

### Phase 2 — Training-Image (`Training/Dockerfile`)
1. Python 3.12 statt 3.10 (Basis `nvidia/cuda:12.8.0-devel-ubuntu22.04` → 3.12 via deadsnakes oder `uv python install 3.12`); Clone-Branch/Pin auf Phase-1-Commit (`update_image.sh --update-commit`); `uv sync --frozen` gegen neues `uv.lock`; `jsonlines` ist jetzt Upstream-Dep (Zeile 64 entfällt).
2. **HF-Cache persistent:** `ENV HF_HOME=/data/hf_cache` (liegt im Container-FS bzw. auf KISSKI-VAST) — sonst lädt jeder Neustart 2 B Backbone-Gewichte neu.
3. `download_data.sh`: Modell-ID `GR00T-N1.7-3B`, **zusätzlich** `huggingface-cli download nvidia/Cosmos-Reason2-2B` (in den HF-Cache, *ohne* `--local-dir`, damit `from_pretrained("nvidia/Cosmos-Reason2-2B")` offline trifft).
4. `entrypoint.sh`/`run_finetuning*.sh`: `MODEL_DIR`/`MODEL_PATH`-Default; `WANDB_PROJECT`-Default ggf. `gr00t-g1-dex3-n17`; Namespace-Defaults `blockstacking_n17`/`…_vision_n17` (damit Checkpoints nicht mit N1.6-Läufen kollidieren).
5. `launch_cotrain.py` von der **N1.7**-`launch_finetune.py` neu ableiten (Docstring verlangt das ausdrücklich); Eagle-Zeilen raus.
6. Build + Push als `:n17` (nicht `:latest`), `SKIP_TRAIN=1`-Smoke: Download → Konvertierung → Shell.
- **Abnahme:** `docker run … -e SKIP_TRAIN=1` läuft bis zur Shell; `python -c "from gr00t.policy import Gr00tPolicy; Gr00tPolicy('/data/models/GR00T-N1.7-3B','new_embodiment')"` scheitert nur an fehlender Modality-Config (erwartet), nicht an Gating/Deps.

### Phase 3 — Lokaler Trainings-Smoke-Test (Blackwell-Server, RTX 5090 32 GB)
1. `MAX_STEPS=200 GLOBAL_BATCH_SIZE=4 SAVE_STEPS=100` ohne und mit `TUNE_VISUAL=1`; VRAM-Peak notieren (Default ohne LLM-Layer-Tuning sollte ≤ N1.6 sein; `--tune_visual` auf Qwen3-ViT ist die Unbekannte).
2. `checkpoint_sweep.py` auf die 2 Mini-Checkpoints (Held-out-Episoden) → MSE/MAE-Zahlen kommen raus, `split.json` liegt neben den Checkpoints.
3. `check_action_norm.py` gegen den Mini-Checkpoint (Stats-Layout mit `use_percentiles=True` prüfen).
- **Abnahme:** Training läuft, Sweep liefert Zahlen, VRAM-Tabelle in CLAUDE.md aktualisiert.

### Phase 4 — KISSKI
1. SIF aus `:n17` ziehen (`apptainer pull … :n17` → `KISSKI_SIF_DIR`), Fork-Clone `repo-groot` auf Branch `luca/g1-dex3-n17` (Bind-Mounts in `kisski_submit.sh:322-349` bleiben).
2. **Login-Knoten:** `GR00T-N1.7-3B` **und** `Cosmos-Reason2-2B` in `/mnt/vast-kisski/projects/kisski-humrob/data` (`models/` bzw. `hf_cache/`) laden; `kisski_submit.sh`: `--env HF_HOME=/data/hf_cache --env HF_HUB_OFFLINE=1`.
3. **Vergleichslauf:** identische HP wie Lauf 3 (`TUNE_VISUAL=1`, `TRAIN_TEST_SPLIT=1`, Split-Ratio 0.8, bs 32, 30k Steps, gleiche Augmentation), Namespace `blockstacking_vision_n17`.
4. Sweep auf Held-out wie bei Lauf 3 → U-Kurve N1.7 vs. N1.6 (Lauf 3, ckpt 30000).
- **Abnahme:** Lauf beendet, Sweep-Tabelle in `docs/ergebnisse/` (neue Datei `lauf4-n17-vergleich.md`), W&B-Offline-Sync.

### Phase 5 — Sim-Image & Inferenz (`Simulation/Dockerfile.standalone`, Server-Workflow)
1. GR00T-venv auf Python 3.12 (Ubuntu 24.04-Basis hat 3.12 nativ → `uv venv --python /usr/bin/python3.12`), Pin wie Training-Image, `HF_HOME` + Backbone-Download in `entrypoint_sim.sh`/`entrypoint_rl.sh`/`entrypoint_baseline.sh` (oder per `HF_CHECKPOINT_REPO`-Pfad mitladen).
2. `entrypoint_sim.sh:236-241`: Flash-Attn-Hinweis auf Qwen3 anpassen (Fallback `sdpa` existiert jetzt — `NO_FLASH_ATTN` könnte wieder wirken).
3. Eager-Pfad zuerst: `server_rl_run.sh eval` mit N1.7-Checkpoint → Phase-D-Assert `(16,28)` in `run_g1_dex3_sim_eval.py:561` muss halten.
4. `policy_latency.py` anpassen (Zugriff auf Action-Head-Klasse generisch über `policy.model.action_head`), Latenz N1.7 vs. N1.6 messen (16- statt 32-Layer-DiT → erwartbar schneller).
5. Optimierter Backend: `optimize_groot_inference.py` re-exportieren + Paritätstest; **Option** Upstream-TRT-Full-Pipeline (`scripts/deployment/`) evaluieren, ggf. eigene DiT-only-Lösung ablösen (`docs/simulation/inferenz-optimierung.md` ergänzen).
6. Domain-Gap: `measure_domain_gap.py` (SigLIP, modellunabhängig) unverändert; `finger_span_openloop.py` mit N1.7-Checkpoint wiederholen (Diskriminator-Test Lauf 32/34).
- **Abnahme:** 20-Episoden-Sim-Eval mit N1.7-Checkpoint läuft durch; Latenz- und Finger-Span-Zahlen in `diagnose-chronik.md` (neuer Lauf-Eintrag).

### Phase 6 — RL & Co-Training nachziehen
1. `rl_finetune.py`: Replikate gegen `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` (Action-Head-Forward, `state_history_length`, `options`/RTC) und `processing_gr00t_n1d7.py` (Masken-Logik `[:, :action_horizon, :action_dim]`) neu herleiten; Zeilenreferenzen in den Kommentaren aktualisieren; „eine volle Iteration auf Hardware" erneut verifizieren.
2. `run_finetuning_cotrain.sh` + `render_cotrain_dataset.py`: keine API-Änderung erwartet; Smoke mit `COTRAIN_MIX_RATIO=0.25`.
- **Abnahme:** je ein Smoke-Durchlauf (RL 1 Iteration, Co-Training 100 Steps).

### Phase 7 — Umschalten & Aufräumen
1. `:n17` → `:latest` retaggen, `.gitmodules`/Dockerfile-Pins final, `update_image.sh`-Default-Branch.
2. Optional: Submodul-Pfad `app/Groot-1.6` → `app/Groot` (156 Stellen, `GROOT_ROOT`-Defaults), N1.6-Pfad als `:n16`-Images + Branch `luca/g1-dex3` archivieren.

### Phase 8 — Doku-Pass
`CLAUDE.md` (Overview, Env-Var-Tabelle, VRAM-Tabelle, Architektur-Baum), `README.md`, `docs/README.md`, `docs/training/{anleitung,env-vars,kisski-hpc,trainingsverfahren,co-training}.md`, `docs/simulation/{sim-eval-anleitung,umsetzungsnotizen,inferenz-optimierung,baseline-eval}.md`, `docs/weiterfuehrend/{rl-anleitung,reinforcement-learning-plan}.md`, `docs/portabilitaet.md` (Hinweis „HF-Repo nicht gated" → Backbone **ist** gated), `app/Groot-1.6/examples/G1_DEX3/*.md` (Fork), `docs/historie.md` (Eintrag „Umstieg N1.6→N1.7"). Ergebnis-Dateien `lauf1–3` bleiben unverändert (historisch, N1.6).

## 6. Offene Entscheidungen / Optionen (nicht Teil der Basis-Migration)

- **Action-Horizon erweitern** (16 → z. B. 32/40 in `g1_dex3_config.py`): N1.7 erlaubt es; ändert `CHUNK_SIZE` in `client.py`/Assert und die Sim-Eval-Execution-Horizon-Logik — eigener Versuch nach dem Baseline-Vergleich.
- **`REAL_G1`-Pretrain-Tag** (Real-Unitree-G1, relative EEF + Joints) prüfen: passen State-/Action-Keys zu unserem DEX3-Datensatz? (In `embodiment_configs.py` auf `main` nicht gefunden — Config steckt vermutlich im Checkpoint; via `processor.get_modality_configs()['real_g1_…']` auslesen.) Falls kompatibel: Fine-Tune vom G1-Prior statt von `NEW_EMBODIMENT` als Experiment.
- **Upstream-TRT-Full-Pipeline** statt eigener DiT-only-Optimierung.
- **RTC (Real-Time Chunking)** im Sim-Loop nutzen (`options`-Dict in `get_action`) — Latenz-Glättung.
- **Robocasa/GR1-Referenz-Eval** (`server_robocasa_ref_run.sh`): Tags in N1.7 entfernt → entweder auf N1.6-Image belassen oder durch LIBERO/SimplerEnv-Referenz ersetzen.
- **Stock-G1-Baseline** (`entrypoint_baseline.sh`, `g1_gripper_sim/`, `docs/simulation/baseline-eval.md`): mit N1.7 nur noch über `REAL_G1` (Keys/Dims aus dem Checkpoint lesen: `dump_unitree_g1_dims.py --embodiment real_g1_relative_eef_relative_joints`) oder auf dem `:n16`-Image belassen.

## 7. Rückfallebene

N1.6 bleibt vollständig lauffähig: Branch `luca/g1-dex3`, Images `:n16`, Upstream-Branch `n1d6`, alle bisherigen Checkpoints. Kein Schritt des Plans verändert den N1.6-Pfad vor Phase 7.

## 8. Quellen

- Upstream-Repo: `https://github.com/NVIDIA/Isaac-GR00T` — Tags `n1.6-release` (ead5283), `n1.7-release` (23ace64, 2026-04-17), `n1.6.1-release` (5dc80c4); GA-Commit `1a1837f` (2026-07-07); `main` `376ba89` (2026-08-10); README „What's New in GR00T N1.7", `getting_started/{policy,finetune_new_embodiment,hardware_recommendation}.md`, `gr00t/configs/finetune_config.py`, `gr00t/configs/model/gr00t_n1d7.py`, `gr00t/data/embodiment_tags.py`, `gr00t/model/modules/qwen3_backbone.py`, `gr00t/eval/run_gr00t_server.py`, `pyproject.toml` (EA vs. main).
- Lokaler Diff im Submodul: `git -C app/Groot-1.6 diff n1.6-release..n1.7-release` (295 Dateien; für uns relevant ~650 Zeilen in `gr00t/{configs,data,experiment,policy}`), `git merge-tree --write-tree HEAD n1.7-release` (nur `uv.lock`-Konflikt).
- HF: `nvidia/GR00T-N1.7-3B` (Model-Card, NVIDIA Open Model License), `nvidia/Cosmos-Reason2-2B` (gated); HF-Blog `nvidia/gr00t-n1-7` (EgoScale-Skalierung — Zahlen nicht primär verifiziert).
- NVIDIA Developer Forum: „Early Access: Isaac GR00T N1.7" (2026-04-17/18).
- Repo-Inventar (eigene Skripte): `Training/scripts/{download_data,entrypoint,run_finetuning*,launch_cotrain,checkpoint_sweep}`, `Training/{Dockerfile,kisski_submit.sh,update_image.sh}`, `Simulation/{Dockerfile.standalone,server_rl_run.sh}`, `Simulation/scripts/{entrypoint_sim,entrypoint_baseline,policy_latency,finger_span_openloop,groot_inference_backend,optimize_groot_inference,run_groot_optimized_server,check_action_norm}`, `Simulation/g1_dex3_sim/{client,rl_finetune,run_g1_dex3_sim_eval}`.

## 9. Entscheidungslog

| Datum | Entscheidung | Begründung |
|---|---|---|
| 2026-08-19 | Plan erstellt | Recherche-/Planungsstand |
| 2026-08-19 | Umsetzung als paralleler Pfad **begonnen und größtenteils durchgeführt** (Dockerfiles, Scripts, zweiter Fork-Branch, KISSKI-Anpassungen, Doku) | siehe [„Stand der Umsetzung"](#stand-der-umsetzung-2026-08-19) oben |
| 2026-08-19 | Pin-Strategie: GA-`main`-Commit (`376ba89`), nicht EA-Tag | Phase 0; N1.7-Fork-Commit `efa0169` basiert darauf |
| 2026-08-19 | Submodul-Pfad `app/Groot-1.6` beibehalten, **neues** `app/Groot-1.7` daneben | kein Umbenennen/Ersetzen — kleinerer Blast-Radius |
| 2026-08-19 | **Ein** Image mit beiden venvs (Build-Arg `GROOT_VERSIONS`) statt separater Image-Tags `:n16`/`:n17` | weicht von der ursprünglichen Phase-0-Empfehlung oben ab; einfacherer Rollout, ein Pull genügt |
| 2026-08-19 | Kein Image gebaut, kein N1.7-Trainings-/Sim-Lauf gefahren | Umsetzung endet vorerst bei Code + Doku; Vergleichslauf und HF-Backbone-Zugang stehen aus |
