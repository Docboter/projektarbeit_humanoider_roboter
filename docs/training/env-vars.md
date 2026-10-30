# Konfiguration über Env-Vars

> **TL;DR:** Referenz aller Trainings-Env-Vars (Pflicht- und optionale Parameter,
> VRAM-Richtwerte) — die Single Source of Truth, auf die andere Trainingsdocs verweisen; dazu die
> Kurzreferenz des Sim-Eval-Containers, die Host-Helfer (KISSKI-Menü, `server_rl_run.sh`) und
> der Verweis auf die Menü-Specs für die aktionsspezifischen Sim-Variablen. Zum gezielten
> Nachschlagen einzelner Variablen, nicht zum Auswendiglernen.

Alle Trainings-Parameter werden über Umgebungsvariablen gesteuert — auf vast.ai, KISSKI und
lokal **identisch**. Der Entrypoint (`/scripts/entrypoint.sh`) liest sie ein.

> **Wo die Defaults stehen:** Ein Teil ist als `ENV` im [Dockerfile](../../Training/Dockerfile)
> gesetzt (`MAX_STEPS`, `GLOBAL_BATCH_SIZE`, `NUM_GPUS`, `WANDB_PROJECT`, `DATA_DIR`,
> `SKIP_*`, `SHELL_ON_ERROR`, `GROOT_VERSION`, `HF_HOME`). Die übrigen — u. a. `TUNE_VISUAL`, `USE_RL`,
> `TRAIN_TEST_SPLIT`, `USE_AUGMENTATION`, `LEARNING_RATE`, `SAVE_STEPS`, `WEIGHT_DECAY`,
> `WARMUP_RATIO` — sind Shell-Defaults in
> [`entrypoint.sh`](../../Training/scripts/entrypoint.sh) bzw.
> [`run_finetuning.sh`](../../Training/scripts/run_finetuning.sh). Für die Bedienung macht das
> keinen Unterschied; beim Suchen im Code schon.
>
> **Build-Zeit vs. Laufzeit:** `GROOT_VERSION` wählt zur **Laufzeit** zwischen den Codebäumen,
> die im Image liegen. Welche Bäume überhaupt **gebaut** werden, entscheidet das
> Docker-Build-Arg `GROOT_VERSIONS` (Default seit 2026-09-25: `1.6`, ein Image je Generation) —
> `--build-arg GROOT_VERSIONS=1.7` baut nur den N1.7-Baum, `"1.6 1.7"` beide venvs in einem
> (Dual-Image, doppelte Größe, nie Default). Bequemer über
> [`update_image.sh`](../../Training/update_image.sh) `--groot=1.6|1.7|both` (fragt ohne
> Flag am Terminal, sonst Default `1.6`) — das Skript setzt auch `GROOT_VERSION_DEFAULT`
> (welche Generation der Container ohne `-e GROOT_VERSION=…` startet) und taggt passend
> (`:latest` bleibt N1.6, N1.7 → `:latest-n17`). Siehe
> [`Training/Dockerfile`](../../Training/Dockerfile).

> **Nicht auswendig lernen:** Die Host-Launcher fragen diese Werte ab, wenn man sie ohne
> Parameter startet, und erklären sie dabei — siehe
> [cli-menuefuehrung.md](../weiterfuehrend/cli-menuefuehrung.md). Die Menü-Beschreibungen
> unter [`tools/menu/`](../../tools/menu/) und diese Tabelle werden von
> `tools/gen_docs.sh` gegeneinander abgeglichen; weicht ein Default ab, schlägt der
> Abgleich fehl. Diese Tabelle bleibt die ausführliche Fassung — sie trägt Begründungen,
> die eine Menü-Zeile nicht fassen kann.

> ⚠️ **`MAX_STEPS` — zwei verschiedene Defaults, je nach Startweg.**
> [`entrypoint.sh`](../../Training/scripts/entrypoint.sh) und das Dockerfile sagen
> `20000` (der Wert in der Tabelle unten); der Host-Launcher
> [`setup_and_train_dockerhub_pull.sh`](../../Training/setup_and_train_dockerhub_pull.sh)
> überstimmt ihn mit `30000`. Wer über den Launcher startet, bekommt also 30000, wer das
> Image direkt fährt (vast.ai), 20000. Beim Umsetzen der Menüführung aufgefallen.

| Variable | Default | Beschreibung |
|---|---|---|
| `HF_TOKEN` | — | **Pflicht.** HuggingFace-Token (Lese-Berechtigung reicht) |
| `WANDB_API_KEY` | — | Optional. W&B-Key. Ohne diesen läuft Training ohne W&B. |
| `GROOT_VERSION` | `1.6` | `1.6` \| `1.7` — wählt die GR00T-Generation (Codebaum `/app/Groot-1.6` bzw. `/app/Groot-1.7`, venv, Modell-Repo, Output-Namespace-Suffix). Auflösung: [`lib_groot_version.sh`](../../Training/scripts/lib_groot_version.sh). `1.7` lädt zusätzlich das **gated** Backbone `nvidia/Cosmos-Reason2-2B` (Zugang separat beantragen) und hängt allen Output-Namespaces `_n17` an (`blockstacking_n17`, `blockstacking_vision_n17`, `blockstacking_cotrain_n17`). Voller Lauf auf KISSKI (44 000 Steps, 2026-10-13) gelaufen; Status nur in [groot-n17-migration.md](../weiterfuehrend/groot-n17-migration.md#stand-der-umsetzung-2026-08-19). |
| `DOCKER_NAMESPACE` | — | **Host-seitig**, kein Container-Env. Docker-Hub-Konto der Images: `update_image.sh` / `update_sim_image.sh` pushen dorthin (ohne Angabe fragen sie am Terminal, ohne Terminal Abbruch vor dem Build), die Launcher ziehen von dort (ohne Angabe Team-Konto `lucam03`). Am besten in `.env.local` — siehe [portabilitaet.md § 3.4](../portabilitaet.md#34-eigene-images-bauen-eigenes-docker-hub-konto). |
| `HF_HOME` | `$DATA_DIR/hf_cache` | HF-Cache-Verzeichnis. Nur für `GROOT_VERSION=1.7` relevant: das Cosmos-Reason2-2B-Backbone wird bei **jedem** Checkpoint-Laden erneut vom Hub gezogen, sofern es nicht schon im Cache liegt — der Cache muss deshalb persistent sein (liegt standardmäßig im Container-FS bzw. auf dem KISSKI-VAST-Mount). |
| `MAX_STEPS` | `20000` | Anzahl Trainings-Steps. Auf ~241–301 Block-Stacking-Episoden sättigt BC früh; 20k konvergieren sauber (Lauf 1 war bei 30k bereits konvergiert). KISSKI-Multi-GPU-Default: `44000`. |
| `GLOBAL_BATCH_SIZE` | `8` | Globale Batch-Size (8 für < 40 GB VRAM, 32 für 4x A100 80 GB) |
| `NUM_GPUS` | `1` | Anzahl genutzter GPUs. `>1` startet `torchrun`. Bei `1` und mehreren sichtbaren Karten setzen die Trainings-Launcher `CUDA_VISIBLE_DEVICES=0`, sonst wickelt der HF-Trainer das Modell in `nn.DataParallel` (N1.7 bricht dann mit `StopIteration` ab). Ein vorgegebenes `CUDA_VISIBLE_DEVICES` bleibt unangetastet — andere Karte also z. B. per `-e CUDA_VISIBLE_DEVICES=1` bzw. `DOCKER_GPUS='"device=1"'` |
| `WANDB_PROJECT` | `gr00t-g1-dex3` | W&B-Projektname |
| `DATA_DIR` | `/data` | Datenverzeichnis im Container |
| `SKIP_DOWNLOAD` | `0` | `1` = HF-Download überspringen (Daten schon vorhanden) |
| `SKIP_CONVERT` | `0` | `1` = v3→v2-Konvertierung überspringen (`modality.json` existiert) |
| `SKIP_TRAIN` | `0` | `1` = nur Setup, dann Shell |
| `SHELL_ON_ERROR` | `0` | `1` = bei Fehler in Shell fallen statt zu beenden |
| `TUNE_VISUAL` | `0` | `1` = Vision-Encoder mittrainieren (`--tune_visual`). Entrypoint startet dann `run_finetuning_vision.sh` mit eigenem Output-/Experiment-Namespace (`blockstacking_vision`). LLM bleibt eingefroren. Höherer VRAM-Bedarf. |
| `LEARNING_RATE` | `1e-4` | Lernrate (`--learning_rate`); KISSKI-Multi-GPU-Default im Standardlauf: `2e-4` (sqrt-skaliert); bei `TUNE_VISUAL=1`/`USE_COTRAIN=1` bleibt es auf KISSKI bei `1e-4` (so liefen Lauf 2/3) |
| `WEIGHT_DECAY` | `1e-5` | Gewichtszerfall (`--weight_decay`) |
| `WARMUP_RATIO` | `0.05` | Anteil Warmup-Steps (`--warmup_ratio`); bei `TUNE_VISUAL=1` auf KISSKI: `0.1` |
| `DATALOADER_WORKERS` | `8` | Dataloader-Worker (`--dataloader_num_workers`); KISSKI-Default: `4` |
| `SAVE_STEPS` | `2000` | Checkpoint-Intervall in Steps (`--save_steps`), gleich in allen drei `run_finetuning*.sh` (Standard, Vision, Co-Training — die frühere Vision-/Co-Training-Ausnahme `1000` gibt es seit 2026-10-13 nicht mehr); KISSKI-Default: `5000`. Basis für die Open-Loop-Checkpoint-Auswahl. |
| `SAVE_TOTAL_LIMIT` | `15` | Max. Anzahl behaltener Checkpoints (`--save_total_limit`); KISSKI-Default: `40`. `15` × 2000 Steps hält **jeden** Checkpoint bis 30 000 Steps (Default des Pull-Launchers), bei 44k die letzten 15 ab 16 000 (bis 2026-10-13: `10`, vision/cotrain `5` — bei 44k Steps zu wenig). **Plattenbedarf:** je Checkpoint ~22 GB inkl. `optimizer.pt` (~13 GB), 15 Checkpoints also bis ~330 GB. |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | Gradienten-Akkumulation (`--gradient_accumulation_steps`). Entkoppelt die effektive Batchgröße vom Speicherbedarf: `experiment.py:188` rechnet `per_device = GLOBAL_BATCH_SIZE / NUM_GPUS`, die Akkumulation kommt oben drauf. Auf einer Karte, deren VRAM teilweise anderweitig belegt ist, hält man damit die Batchgröße des Referenzlaufs, statt sie zu senken und das Optimierungsregime mitzuändern. `1` = aus. |
| `USE_WANDB` | *auto* | W&B an/aus. Wird vom Entrypoint automatisch gesetzt: `1` wenn `WANDB_API_KEY` vorhanden, sonst `0`. Manuell `USE_WANDB=0` erzwingt Training ohne W&B. |
| `WANDB_MODE` | *auto* | W&B-Modus, vom Entrypoint gesetzt, falls leer: unter Apptainer/SLURM (KISSKI) `offline` — dort puffert W&B lokal und muss danach gesynct werden, siehe [wandb-offline-sync.md](wandb-offline-sync.md) —, lokal und auf vast.ai `online`. Bis 2026-10-13 war es überall `offline`. Ein gesetzter Wert hat Vorrang. |

> **Lernrate und Warmup hängen vom Startweg und vom Modus ab** (geprüft 2026-10-13 in
> `run_finetuning*.sh` und [`kisski_submit.sh`](../../Training/kisski_submit.sh)):
>
> | Startweg | Standardlauf | `TUNE_VISUAL=1` / `USE_COTRAIN=1` |
> |---|---|---|
> | Container direkt / Docker-Launcher (`run_finetuning*.sh`) | LR `1e-4`, Warmup `0.05` | LR `1e-4`, Warmup `0.05` |
> | KISSKI (`kisski_submit.sh`) | LR **`2e-4`** (√4-skaliert für 4 GPUs), Warmup `0.05` | LR `1e-4`, Warmup **`0.1`** — der mittrainierte ViT verträgt `2e-4` nicht |
>
> Ein gesetztes `LEARNING_RATE`/`WARMUP_RATIO` gewinnt immer. Lauf 2/3 liefen mit `1e-4` / `0.1`.

## Optionale Trainings-Features (getrennt schaltbar)

Diese Schalter aktivieren einzelne Verfahren beim Trainingsstart — alle unabhängig voneinander.

| Variable | Default | Beschreibung |
|---|---|---|
| `TRAIN_TEST_SPLIT` | `0` | Default `0`, aber **`1` bei `USE_COTRAIN=1`** (gesetzt in `entrypoint.sh` und `kisski_submit.sh`; das Dockerfile setzt seit 2026-10-13 kein `ENV` mehr, das den Co-Training-Default überstimmte). `1` = 80/20-Split scharf schalten: [`lib_split.sh`](../../Training/scripts/lib_split.sh) patcht `meta/info.json`, sodass nur die ersten `TRAIN_SPLIT_RATIO` der Episoden als `train` geladen werden; die restlichen stehen als `test` für die Open-Loop-Eval auf **ungesehenen** Episoden bereit. Zusätzlich landet ein Protokoll `split.json` im `OUTPUT_DIR`. `0` = kompletter Datensatz; ein zuvor gesetzter Split wird dann automatisch zurückgesetzt. **Gilt seit 2026-08-13 auch für `TUNE_VISUAL=1`** (vorher nur im Standard-Skript). |
| `TRAIN_SPLIT_RATIO` | `0.8` | Trainingsanteil bei `TRAIN_TEST_SPLIT=1` (Rest = Test). |
| `USE_AUGMENTATION` | `1` | Bild-Augmentierung / Domain-Randomization (Color-Jitter, optional Rotation/State-Dropout) gegen den Sim-Real-Domain-Gap. `0` = explizit aus (Color-Jitter auf 0, kein Modell-Default-Jitter). **Gilt seit 2026-08-13 auch für `TUNE_VISUAL=1`** — vorher war der Jitter im Vision-Skript fest verdrahtet und der Schalter dort wirkungslos. |
| `CJ_BRIGHTNESS` / `CJ_CONTRAST` / `CJ_SATURATION` / `CJ_HUE` | `0.3` / `0.4` / `0.5` / `0.08` | Color-Jitter-Stärken (nur bei `USE_AUGMENTATION=1`). |
| `RANDOM_ROTATION_ANGLE` | *(leer)* | Max. Rotationswinkel (Grad) für Bild-Rotations-Augmentierung; leer = aus. |
| `STATE_DROPOUT_PROB` | *(leer)* | Dropout-Wahrscheinlichkeit auf den State-Inputs (Regularisierung). Leer = Modell-Default der gewählten Generation (N1.6 `0.0`, N1.7 `0.2` — bewusst übernommen, siehe [groot-n17-migration.md § Stand der Umsetzung, Nachtrag 2026-09-25](../weiterfuehrend/groot-n17-migration.md#stand-der-umsetzung-2026-08-19)); ein gesetzter Wert (auch `0`) wird immer übergeben. Bei `USE_AUGMENTATION=0` gilt leer = `0` (keine Regularisierung durch Modell-Defaults, auch nicht unter N1.7). |
| `USE_RL` | `0` | `1` = RL-Fine-tuning (FPO) gewünscht. Läuft **nicht** im BC-Trainingsimage (kein Isaac Sim): der BC-Entrypoint bricht mit einem Hinweis auf den RL-Pfad ab. RL braucht den kombinierten Isaac-Sim + GR00T-Container ([`Simulation/scripts/entrypoint_rl.sh`](../../Simulation/scripts/entrypoint_rl.sh) bzw. [`Training/kisski_rl_submit.sh`](../../Training/kisski_rl_submit.sh)) auf einer **RT-Core-GPU**. Details: [reinforcement-learning-plan.md](../weiterfuehrend/reinforcement-learning-plan.md). |
| `USE_COTRAIN` | `0` | `1` = **Co-Training auf echten UND gerenderten Bildern** (Schritt 4). Entrypoint startet [`run_finetuning_cotrain.sh`](../../Training/scripts/run_finetuning_cotrain.sh) mit eigenem Namespace `blockstacking_cotrain`; setzt `--tune_visual` selbst und hat Vorrang vor `TUNE_VISUAL`. Braucht einen gerenderten Datensatz (siehe unten). Anleitung: [co-training.md](co-training.md). |

### Co-Training: gerenderten Datensatz dazumischen (`USE_COTRAIN=1`)

Der gerenderte Datensatz entsteht auf dem **Sim-Server** (RT-Core-GPU) mit
`./Simulation/server_rl_run.sh render` und wird zum Trainings-Rechner transportiert
(~1–2 GB für 60 Episoden). Vollständige Herleitung der Werte:
[co-training.md §4](co-training.md#4-die-beiden-entscheidungen--und-wie-sie-begründet-sind).

`COTRAIN_DATASET_PATH`/`COTRAIN_HF_REPO` ist die Quelle egal — statt aus Isaac Lab gerendert
kann der zweite Datensatz auch per NVIDIA Cosmos-Transfer2.5 aus echten Videos stilvariiert
entstehen: [`Augmentation/`](../augmentation/anleitung.md). Beide liefern dasselbe
LeRobot-v2.1-Schema.

| Variable | Default | Beschreibung |
|---|---|---|
| `COTRAIN_DATASET_PATH` | `/data/cotrain/g1_dex3_rendered` | Gerenderter Datensatz im Container (LeRobot v2.1, wie der echte). |
| `COTRAIN_HF_REPO` | *(leer)* | HF-**Dataset**-Repo; wird geladen, wenn `COTRAIN_DATASET_PATH` leer ist. |
| `COTRAIN_MIX_RATIO` | `0.25` | Anteil gerenderter Stichproben (Sampling-Wahrscheinlichkeit, **kein** Längenverhältnis). Bis 2026-10-13 stand der Code-Default auf `0.5` und widersprach der Empfehlung; seitdem ist `0.25` überall Default (Skript, `launch_cotrain.py`, `kisski_submit.sh`, Menü). Begründung: bei 60 gerenderten gegen 240 echte Episoden sähe das Modell bei 0.5 jedes Sim-Bild 4× so oft wie jedes echte, und die Hälfte aller Updates entfiele auf ein Viertel des Bewegungsrepertoires. `kisski_chain.sh` setzt für den Synth-Datensatz bewusst `0.10`. |
| `TRAIN_TEST_SPLIT` | `1` **hier** | Bei `USE_COTRAIN=1` standardmäßig an — seit 2026-10-13 auch über Entrypoint und KISSKI (vorher setzten Dockerfile/Entrypoint `0` vor dem `:-1` des Skripts, der Split war dort still aus). Ohne zurückgehaltene Episoden fehlt die Leitplanke, an der ein Rückschritt auf der Realdomäne sichtbar würde. |

Die Renderer-seitigen Variablen (`RENDER_EPISODES`, `RENDER_STAGE`, `RENDER_MAX_FRAMES`, …)
stehen in [co-training.md §7](co-training.md#7-env-vars) und in `./Simulation/server_rl_run.sh help`.

### Namespace des Laufs — ⚠️ der Fork setzt ungefragt fort

> **Der GR00T-Fork ruft `trainer.train(resume_from_checkpoint=True)` fest verdrahtet auf**
> ([`experiment.py:288`](../../app/Groot-1.6/gr00t/experiment/experiment.py)). Es gibt keinen
> Schalter dagegen. Der HuggingFace-Trainer sucht den letzten Checkpoint in
> `OUTPUT_DIR/EXPERIMENT_NAME` und stellt dessen `global_step` wieder her.
>
> **Ein neuer Lauf in ein belegtes Verzeichnis trainiert also nicht neu, sondern setzt fort.**
> Liegt der vorhandene Checkpoint schon bei `MAX_STEPS`, ist die Schleife sofort zu Ende: der
> Job meldet „Training completed", legt einen Checkpoint mit den **alten** Gewichten ab und
> sieht erfolgreich aus. Genau so ist Job 15271760 am 2026-08-13 gelaufen — 44001 „Schritte" in
> 255 s, kein einziger Loss-Wert protokolliert, `checkpoint-44001` mit den Gewichten von Lauf 2.
>
> [`lib_resume_guard.sh`](../../Training/scripts/lib_resume_guard.sh) bricht seitdem vorher ab.

| Variable | Default | Beschreibung |
|---|---|---|
| `OUTPUT_DIR` | `/data/g1_dex3_finetune/blockstacking` (Vision-Lauf: `…_vision`, Co-Training: `…_cotrain`) | Wurzel des Lauf-Namespace. |
| `EXPERIMENT_NAME` | `g1_dex3_blockstacking_v1` (Vision: `…_vision_v1`, Co-Training: `…_cotrain_v1`) | Unterverzeichnis **und** W&B-Run-Name. Für jeden neuen Lauf hochzählen. |
| `RESUME` | `0` | `1` = Fortsetzen ist gewollt (z. B. nach Walltime-Abbruch). `0` = Abbruch, wenn schon Checkpoints da sind. |

Auf KISSKI werden alle drei seit 2026-08-13 von `kisski_submit.sh` durchgereicht.

### Checkpoint-Auswahl nach dem Lauf ([`checkpoint_sweep.py`](../../Training/scripts/checkpoint_sweep.py))

> **Warum ein eigenes Werkzeug:** Der GR00T-Fork hat **keine** Eval während des Trainings.
> `enable_open_loop_eval`, `eval_set_split_ratio` und `open_loop_eval_*` stehen zwar in
> `TrainingConfig`, werden aber **nirgends gelesen**; und `DatasetFactory.build()` bricht mit
> `assert eval_strategy == "no"` ab, gibt fest `split="train"` vor und liefert `eval_dataset=None`.
> Ein `eval_strategy="steps"` würde also nicht evaluieren, sondern abstürzen. Die Validierung
> muss deshalb **nach** dem Lauf über die gespeicherten Checkpoints laufen.

Der Sweep lädt jeden `checkpoint-*` einzeln und misst MSE/MAE gegen die **zurückgehaltenen**
Episoden. Voraussetzung ist ein Trainingslauf mit `TRAIN_TEST_SPLIT=1`.

| Variable | Default | Beschreibung |
|---|---|---|
| `RUN_DIR` | `/data/g1_dex3_finetune/blockstacking` | Lauf-Verzeichnis; wird **rekursiv** nach `checkpoint-<step>` durchsucht. Für den Vision-Lauf: `…/blockstacking_vision`. |
| `EVAL_SPLIT` | `test` | Datensatz-Split. `test` = die zurückgehaltenen Episoden — das ist der Sinn der Übung. |
| `EVAL_NUM_TRAJ` | `6` | Anzahl gleichmäßig über den Split verteilter Episoden. |
| `EVAL_TRAJ_POSITIONS` | *(leer)* | Statt `EVAL_NUM_TRAJ`: explizite Positionen **im Split** (kommagetrennt). |
| `EVAL_STEPS` | `300` | Max. Steps je Episode (auf die Episodenlänge gedeckelt). |
| `EVAL_CHECKPOINTS` | *(leer)* | Nur bestimmte Step-Nummern auswerten; leer = alle gefundenen. |
| `ALLOW_TRAIN_SPLIT` | `0` | `1` = trotzdem messen, wenn kein `test`-Split existiert, ein anderer Split gewählt ist oder `info.json` nicht zum `split.json` des Laufs passt. Sonst bricht der Sweep seit 2026-10-13 mit Exit ≠ 0 ab, statt Trainings-MSE als Validierung auszugeben. |
| `EVAL_OUT` | `<run-dir>/checkpoint_sweep.json` | JSON mit Tabelle, bestem Step und den ausgewerteten Episoden. |
| `EVAL_PLOT_DIR` | `<run-dir>/open_loop_plots` | Plots (Ist- vs. Vorhersage-Aktionen), je Checkpoint und Episode. |
| `EVAL_DEVICE` | *(leer)* | Torch-Device, z. B. `cuda:1`, wenn die andere GPU belegt ist. |

> ⚠️ **Positionen ≠ Episoden-Indizes.** `loader[idx]` indiziert in die **gefilterte** Liste. Nach
> einem 80/20-Split ist Position 0 die Episode 240, nicht die Episode 0. Der Sweep protokolliert
> deshalb immer beides. Aus demselben Grund taugt `gr00t/eval/open_loop_eval.py` nicht direkt: es
> reicht kein `split` durch und misst nach einem Split-Lauf gegen **Trainings**-Episoden.

Auf KISSKI als Job: [`Training/kisski_open_loop_eval.sh`](../../Training/kisski_open_loop_eval.sh).

### RL-Env-Vars (nur im Sim-Image, `entrypoint_rl.sh`)

| Variable | Default | Beschreibung |
|---|---|---|
| `CHECKPOINT_PATH` | `/data/checkpoints/groot-g1dex3-checkpoint` | BC-Checkpoint als RL-Startpunkt |
| `HF_CHECKPOINT_REPO` | — | Optional: BC-Checkpoint von HF nach `CHECKPOINT_PATH` laden |
| `RL_NUM_ENVS` | `16` | Parallele Sim-Envs (Render-Durchsatz beachten — RT-Cores nötig) |
| `RL_ITERATIONS` | `500` | RL-Iterationen (Rollout + Update) |
| `RL_ROLLOUT_STEPS` | `32` | Env-Steps pro Rollout pro Env |
| `RL_LR` | `1e-5` | Lernrate (nur Action-Head wird getunt) |
| `RL_KL_COEF` | `0.1` | KL-Regularisierung gegen den BC-Checkpoint |
| `RL_CLIP` | `0.2` | PPO/FPO-Clip-Epsilon |
| `RL_SAVE_EVERY` | `100` | Checkpoint alle N Iterationen (ganzes Modell, ~6 GB je Checkpoint) |
| `RL_MINIBATCH_SIZE` | `64` | `(t, env)`-Paare je Update-Schritt |
| `RL_FPO_MC_SAMPLES` | `4` | K Ziehungen für den FPO-Proxy. Skaliert Speicher **und** Rechenzeit linear — erster Hebel bei OOM |
| `RL_EPOCHS_PER_ITER` | `2` | PPO-Epochen je Rollout |
| `RL_EXEC_HORIZON` | `1` | Ausgeführte Chunk-Schritte je Policy-Abfrage (`--exec-horizon`, erlaubt 1 bis Chunk-Länge 16). `1` = bisheriges Verhalten (jeden Schritt neu planen); die **Eval führt 8** je Chunk aus — mit `8` optimiert RL dasselbe Verhalten, das gemessen wird (Review 10-13 S4) |
| `RL_REF_DEVICE` | `auto` | Gerät des eingefrorenen Referenzmodells (KL). `auto` = zweite sichtbare GPU, `same` = wie die Policy, sonst z. B. `cuda:1`. Entlastet die Trainingskarte um ~6–7 GB; mit nur einer GPU wirkungslos |
| `RL_WANDB_VIDEO_EVERY` | `0` | Alle N Iterationen einen Rollout als Video ins W&B-Dashboard (`0` = aus). Braucht `WANDB_API_KEY`; ohne `moviepy` fällt der Trainer automatisch auf einen Filmstreifen aus Einzelbildern zurück |

> **Hinweis:** Der RL-Trainer ([`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py)) läuft seit 2026-08-08 end-to-end auf RT-Core-Hardware (RTX PRO 6000 Blackwell, Isaac Sim 6.0) — Rollout, FPO-Update und Checkpoint-Schreiben sind nachgewiesen. Offen ist die **Lernwirkung** (steigt `success_rate` über viele Iterationen?); die Hyperparameter oben sind ungetunt. Der Greif-Physik-Blocker der Läufe 25–28 („auch ohne Modell wird kein Würfel angehoben") ist mit **Lauf 29** (2026-08-12) gefallen: mit korrigiertem Fingerkuppen-Messpunkt hebt die Hand den Würfel 7,9 cm — ein Lernsignal ist damit erreichbar. Details: [diagnose-chronik.md](../ergebnisse/diagnose-chronik.md#lauf-29-der-würfel-hebt-ab-mit-einem-vorbehalt).

### Live-Ansicht des Laufs (nur im Sim-Image, opt-in)

MJPEG-Stream des laufenden Rollouts im Browser („Spur B" aus dem
[Livestream-Plan](../weiterfuehrend/livestream-plan.md)). Kostet **keinen zusätzlichen
Render-Pass** — `cam_scene` wird ohnehin jeden Env-Step gerendert. Bei `LIVE_VIEW=0` ist der
Codepfad ein reiner Early-Return, das Verhalten also identisch zu vorher.

| Variable | Default | Beschreibung |
|---|---|---|
| `LIVE_VIEW` | `0` | `1` = Live-Ansicht aktiv (`http://<server-ip>:8900/`) |
| `LIVE_VIEW_PORT` | `8900` | HTTP-Port. Container-Port mappen (`-p 8900:8900`) — `server_rl_run.sh` tut das beim Anlegen automatisch |
| `LIVE_VIEW_EVERY_N` | `1` | Nur jedes n-te Frame senden (Drosselung bei hohem Durchsatz) |
| `LIVE_VIEW_CAMS` | `cam_left_high,cam_left_wrist` | Kameras, kommagetrennt. Default sind die **kalibrierten Policy-Kameras** — sie zeigen genau die Modell-Eingabe. Weitere: `cam_right_high`, `cam_right_wrist`. `cam_scene` (Übersicht) ist **unvalidiert** und zeigte am 2026-08-08 nur Hintergrund — Diagnose mit `server_rl_run.sh cams` |

> Ohne offenen Port geht auch ein Tunnel: `ssh -L 8900:localhost:8900 <server>`, dann
> `http://localhost:8900/`. Der Stream hat **keine Authentifizierung** — im VPN/Institutsnetz
> vertretbar, auf einer öffentlichen vast.ai-IP nur per SSH-Tunnel nutzen.

### LIVE-Variante: Isaac-Sim-Viewport statt Videos (nur im Sim-Image, opt-in)

„Spur A" aus dem [Livestream-Plan](../weiterfuehrend/livestream-plan.md): Isaac Sim streamt
seinen **3D-Viewport** per WebRTC, geöffnet wird er von der Desktop-App *Isaac Sim WebRTC
Streaming Client* — freie Kamera, Szene drehen, Isaac-Sim-UI. Wirkt auf **alle vier Läufe**
(Sim-Eval, Baseline, Replay/Greif-Test, RL). Bedienung inkl. Client-Installation und
Fehlersuche: **[live-ansicht.md](../simulation/live-ansicht.md)**.

Bei `LIVESTREAM=0` (Default) ist das Verhalten identisch zu vorher; bei `≠0` wird
`--video-dir` leer übergeben — **live statt Video**, es entstehen also keine MP4s.

| Variable | Default | Beschreibung |
|---|---|---|
| `LIVESTREAM` | `0` | `0`=aus (headless), `1`=WebRTC öffentlich (vast.ai; ungeschützt!), `2`=WebRTC privat/lokal — auf dem eigenen Server der richtige Wert |
| `LIVESTREAM_PORT` | `49100` | Signaling, **TCP**. Intern == extern mappen (WebRTC bettet den Port in die SDP-Aushandlung ein); `server_rl_run.sh` mappt beim Anlegen |
| `LIVESTREAM_MEDIA_PORT` | `47998` | Medien, **UDP**. Muss durch die Firewall — der wahrscheinlichste Stolperstein |
| `LIVE_KEEP_VIDEO` | `0` | `1` = zusätzlich MP4s schreiben (Live **und** Video) |
| `LIVESTREAM_UPDATE_EVERY_N` | `1` | Nur RL: alle n Rollout-Steps `simulation_app.update()`, damit der Viewport nachzieht. `0` = nie |
| `LIVESTREAM_SETTINGS_STYLE` | `auto` | `auto\|new\|old\|both` — welche Kit-Settings-Pfade gesetzt werden. `auto` liest die Isaac-Sim-`VERSION` (6.0 benannte sie um) |
| `LIVESTREAM_KIT_ARGS` | — | Manueller Override der kompletten Kit-Settings-Zeile |
| `PUBLIC_IP` | — | Nur bei `LIVESTREAM=1`; sonst wird sie gar nicht erst ermittelt |
| `LIVESTREAM_HOST_ADDR` | auto | Server-Adresse für den Verbindungshinweis **und** für den `webview`-Build. Auto = erstes Feld von `hostname -I`; das kann die docker0-Bridge sein |
| `RL_NETWORK_MODE` | `bridge` | `host` legt den Sim-Container mit `--network=host` an. NVIDIA nennt das für WebRTC erforderlich; wir fahren Bridge mit 1:1-Mapping. Erster Verdacht, wenn der Viewport trotz offenem UDP schwarz bleibt. Verlangt `clean` |

**Im Browser statt in der App** (seit 2026-08-17): `./Simulation/server_rl_run.sh webview`
startet NVIDIAs Web-Viewer als eigenen Container und zeigt denselben Viewport samt Maus- und
Tastatursteuerung unter `http://<server-ip>:8210/` (Chromium/Chrome/Edge). Er braucht
parallel einen Lauf mit `LIVESTREAM=2` und spart die App-Installation, **nicht** den
UDP-Port. Schalter: `WEBVIEW_PORT` (8210), `WEBVIEW_IMAGE`/`WEBVIEW_CONTAINER`.

> **Reifegrad:** Der Isaac-Sim-seitige Code ist vollständig und trocken geprüft, lief aber
> **noch nie auf Hardware**. Vor dem ersten Versuch `./Simulation/server_rl_run.sh livecheck`
> (prüft NVENC, Livestream-Extension, Isaac-Sim-Version, Port-Veröffentlichung).
> Der Web-Viewer aus `webview` ist dagegen lokal **end-to-end verifiziert** (Build, Seite,
> eingebackene Adresse im JS-Bundle) — offen ist dort nur der Handshake mit Isaac Sim.


## Sim-Eval-Container (`entrypoint_sim.sh`) — Kurzreferenz

Die Variablen des Standalone-Sim-Images (IKR-Server über `server_rl_run.sh`, vast.ai) für den
Closed-Loop-Eval. Schritt-für-Schritt-Anleitung und die vast.ai-Eintragung:
[sim-eval-anleitung.md](../simulation/sim-eval-anleitung.md). `HF_TOKEN`, `CHECKPOINT_PATH`,
`HF_CHECKPOINT_REPO`, `SHELL_ON_ERROR` und die Live-Variablen stehen in den Tabellen oben; im
Sim-Container gilt für `HF_CHECKPOINT_REPO`: gesetzt ⇒ der Entrypoint lädt Checkpoint **und**
USD von dort nach `/data/checkpoints/<repo-name>/` (`SHELL_ON_ERROR=1` ist dort empfohlen).

| Variable | Default | Beschreibung |
|---|---|---|
| `ASSET_PATH` | `/workspace/assets/g1_dex3_blackhands.usd` | USD-Roboter-Asset. Beim HF-Download z. B. `/data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd` (USD und `configuration/` liegen im selben Repo) |
| `NUM_EPISODES` | `20` | Eval-Episoden. 20 ist das Abnahmeprotokoll; 10 das bisherige Vergleichsprotokoll (Lauf 34, 54/55) |
| `EPISODE_LENGTH_S` | `0` | Zeitbudget je Episode in Sekunden. `0` = bis zu 9000 Steps je Episode — 20 Episoden dauern dann Stunden. **Immer setzen** und beim Vergleich konstant halten (Vergleichsprotokoll: `40`) |
| `EXECUTION_HORIZON` | `8` | Aktionen je Inferenz-Chunk (Chunk-Länge 16) |
| `EVAL_LAYOUT` | *(leer)* | Echte Würfel-Startlagen, z. B. `/data/eval/layout_heldout.json` (zurückgehaltene Episoden 240–300, aus `LAYOUT_HELD_OUT=1 server_rl_run.sh layout`). Leer = Zufallsbänder, die nicht zu den Realdaten passen |
| `DR_SEED` | `0` | Fester Seed für Domain-Randomisierung und Würfel-Zufall; negativ = je Prozess neu. Gibt es erst seit 2026-10-13 — frühere Läufe sind ungeseedet |
| `TASK_DESCRIPTION` | *(leer)* | Leer = aus `meta/tasks.*` des Datensatzes, Fallback `stack three block` (der einzige Trainings-Task). Vor 2026-10-13 schickte jeder Closed-Loop-Lauf `stack the blocks` |
| `BLACK_HANDS` | `1` | Hände schwarz färben (Domain-Gap-Maßnahme) |
| `TRAIN_SPLIT_RATIO` (Sim) | `0.8` | Auch sim-seitig gelesen: Rückfall für ältere `layout.json` ohne `n_train`, um Trainings- und Held-out-Episoden zu trennen (Formel in `split_math.py`, Spiegel von `lib_split.sh`; mit Warnung im Log) |
| `GROOT_VERSION` (Sim) | `auto` | Im Sim-Image Default `auto`: aus `model_type` der Checkpoint-`config.json` aufgelöst. N1.6-only: optimiertes Inferenz-Backend (`compile`/`tensorrt` fällt auf N1.7 auf `eager` zurück), Stock-G1-Baseline (`SIM_MODE=baseline`) und RL — die Entrypoints brechen bei N1.7 ab. Das Sim-Image auf Docker Hub (`:latest`) enthält seit 2026-10-14 N1.6 und N1.7; `RL_SKIP_PULL=1` nur nach einem eigenen lokalen Build |

GPU: Ampere oder neuer **mit RT-Cores** (L40, RTX 4090, A6000, RTX PRO 6000). A100/H100 haben keine
RT-Cores, die RTX 5000 der KISSKI-`jupyter`-Partition ist Turing und zu alt für Isaac Sim 5.1/6.0.

## Host-Helfer und aktionsspezifische Sim-Variablen

Variablen, die nicht im Trainings-Container, sondern in einem Host-Werkzeug wirken.

| Variable | Default | Beschreibung |
|---|---|---|
| `KISSKI_PARTITION` | *(leer)* | Nur [`kisski_menu.sh`](../../Training/kisski_menu.sh) (Login-Knoten): gesetzt ⇒ `sbatch --partition=…`; leer = die `#SBATCH`-Zeile des Skripts (`kisski`). Seit 2026-10-13 passt das Menü die GPU-Anforderung mit an: Typ nach Partition (`kisski` → A100, `kisski-h100` → H100, `jupyter` → RTX5000), Anzahl beim Training nach `NUM_GPUS` (leer = 4 aus dem Skript) — vorher blieb `-G A100:4` fest stehen ([Review 10-13 T2](../archiv/projekt-review-2026-10-13.md#3-code-training)). Direkt per `sbatch` gilt weiter die `#SBATCH`-Zeile |
| `KISSKI_TIME` | *(leer)* | Nur `kisski_menu.sh`: gesetzt ⇒ `sbatch --time=…`; leer = `#SBATCH --time` des Skripts (Training 48 h). Das Menü schlägt für den Open-Loop-Sweep `04:00:00` vor |
| `SKIP_GIT_PULL` | `1` | Nur `kisski_submit.sh`: `1` = den Repo-Klon auf KISSKI **nicht** aktualisieren (Default, damit ein Job den Klon nicht auf einen älteren Stand setzt); `0` = vor dem Job `git pull` |
| `RL_SHM_SIZE` | `16g` | Nur [`server_rl_run.sh`](../../Simulation/server_rl_run.sh): `--shm-size` beim **Anlegen** des Sim-Containers (wirkt erst nach `clean`) |
| `RL_EXEC_GPU` | *(leer)* | Nur `server_rl_run.sh`: Container-GPU-Index für einen einzelnen `docker exec`-Lauf. Leer = automatisch die Karte mit dem meisten freien Speicher (`rl` ausgenommen, der Trainer braucht beide). `RL_EXEC_GPU_AUTO=0` schaltet die Automatik ab |
| `RL_OUTPUT_DIR` | `/data/g1_dex3_rl` | Nur [`entrypoint_rl.sh`](../../Simulation/scripts/entrypoint_rl.sh): Ausgabeverzeichnis der RL-Checkpoints im Container |
| `MASTER_PORT` | `12341` | Nur Augmentation ([`run_inference.sh`](../../Augmentation/scripts/run_inference.sh)): `torchrun --master_port` für Cosmos. Das Trainings-Image setzt keinen eigenen Wert (torchrun-Default) |

**Aktionsspezifische Sim-Variablen** (`server_rl_run.sh <aktion>`): Für die Präfixe `OPTIMIZE_*`,
`LATENCY_*`, `LAYOUT_*`, `RENDER_*`, `TIPCHECK_*` und `REPLAY_*` sind die **Menü-Specs die
Referenz** — je Aktion eine Datei `tools/menu/sim-<aktion>.spec` mit Default, Ebene und Erklärung
jedes Parameters; `tools/gen_docs.sh` prüft sie gegen die `${VAR:-…}` im Skript. Die Spec zeigt
auch das Menü selbst (`./Simulation/server_rl_run.sh <aktion>` ohne Parameter). Hier stehen sie
bewusst nicht ein zweites Mal.

| Präfix | Aktion(en) | Spec | Fachliche Doku |
|---|---|---|---|
| `OPTIMIZE_*` | `optimize` | [`sim-optimize.spec`](../../tools/menu/sim-optimize.spec) | [inferenz-optimierung.md](../simulation/inferenz-optimierung.md) |
| `LATENCY_*` | `latency` | [`sim-latency.spec`](../../tools/menu/sim-latency.spec) | [inferenz-optimierung.md](../simulation/inferenz-optimierung.md) |
| `LAYOUT_*` | `layout`, `layoutcheck`, `layoutreport`, `topface`, `yawcheck` | [`sim-layout.spec`](../../tools/menu/sim-layout.spec) u. a. | [wuerfellage-rekonstruktion.md](../simulation/wuerfellage-rekonstruktion.md) |
| `RENDER_*` | `render` | [`sim-render.spec`](../../tools/menu/sim-render.spec) | [co-training.md §7](co-training.md#7-env-vars) |
| `TIPCHECK_*` | `tipcheck` | [`sim-tipcheck.spec`](../../tools/menu/sim-tipcheck.spec) | [diagnose-chronik.md](../ergebnisse/diagnose-chronik.md) |
| `REPLAY_*` | `replay-prepare`, `replay-calibrate`, `replay-poses`, `replay-render` | `sim-replay-*.spec` | [replay-videos-aus-realdaten.md](../simulation/replay-videos-aus-realdaten.md) |

Ohne Spec, nur im Skript (Stand 2026-10-13): `LATENCY_SWEEP` (`1,2,4`, Denoising-Schritte der
Latenzmessung), `OPTIMIZE_MEAN_TOLERANCE` / `OPTIMIZE_MAX_TOLERANCE` (`0.005` / `0.05`,
Abweichungsgrenzen der TensorRT-Prüfung) und die Pfad-Overrides `REPLAY_CALIBRATION`,
`REPLAY_CALIBRATION_ANCHORS`, `REPLAY_CALIBRATION_DEBUG_DIR`, `REPLAY_POSES`,
`REPLAY_POSE_DEBUG_DIR`, `REPLAY_RENDER_REPORT_DIR` (alle relativ zu `REPLAY_WORK`).

## VRAM-Richtwerte

| VRAM | `GLOBAL_BATCH_SIZE` | `MAX_STEPS` | Plattform |
|---|---|---|---|
| 24 GB | 1–2 | 30 000 | lokal (RTX 4090, Minimum) — sehr langsam |
| 32 GB | 4–8 | 30 000 | lokal (RTX 5090, ~31 GB bei bs 8) |
| 80 GB | 64–128 | 50 000+ | KISSKI A100 |
| 94 GB | 128+ | 50 000+ | KISSKI H100 |

Die Tabelle gilt für **N1.6**. **N1.7** mit `TUNE_VISUAL=1` erreichte **~58 GiB bei bs 8 auf einer
GPU** (IKR RTX PRO 6000, 2026-09-25) — passt also **nicht** auf eine 32-GB-RTX-5090; auf KISSKI
(4 × A100, bs 8 je GPU) waren es ~33 GB je GPU.

> **Full Fine-tuning benötigt laut NVIDIA ≥ 40 GB VRAM.** Karten mit < 24 GB VRAM führen zu
> OOM-Fehlern.

Bei `CUDA out of memory` zuerst `GLOBAL_BATCH_SIZE` halbieren. Volle Parameter-Referenz:
[`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](../../app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md).
