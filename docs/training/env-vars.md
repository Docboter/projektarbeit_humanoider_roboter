# Konfiguration über Env-Vars

> **TL;DR:** Referenz aller Trainings-Env-Vars (Pflicht- und optionale Parameter,
> VRAM-Richtwerte) — die Single Source of Truth, auf die andere Trainingsdocs verweisen.
> Zum gezielten Nachschlagen einzelner Variablen, nicht zum Auswendiglernen.

Alle Trainings-Parameter werden über Umgebungsvariablen gesteuert — auf vast.ai, KISSKI und
lokal **identisch**. Der Entrypoint (`/scripts/entrypoint.sh`) liest sie ein.

> **Wo die Defaults stehen:** Ein Teil ist als `ENV` im [Dockerfile](../../Training/Dockerfile)
> gesetzt (`MAX_STEPS`, `GLOBAL_BATCH_SIZE`, `NUM_GPUS`, `WANDB_PROJECT`, `DATA_DIR`,
> `SKIP_*`, `SHELL_ON_ERROR`). Die übrigen — u. a. `TUNE_VISUAL`, `USE_RL`,
> `TRAIN_TEST_SPLIT`, `USE_AUGMENTATION`, `LEARNING_RATE`, `SAVE_STEPS`, `WEIGHT_DECAY`,
> `WARMUP_RATIO` — sind Shell-Defaults in
> [`entrypoint.sh`](../../Training/scripts/entrypoint.sh) bzw.
> [`run_finetuning.sh`](../../Training/scripts/run_finetuning.sh). Für die Bedienung macht das
> keinen Unterschied; beim Suchen im Code schon.

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
> [`setup_and_train_DockerHub-pull.sh`](../../Training/setup_and_train_DockerHub-pull.sh)
> überstimmt ihn mit `30000`. Wer über den Launcher startet, bekommt also 30000, wer das
> Image direkt fährt (vast.ai), 20000. Beim Umsetzen der Menüführung aufgefallen.

| Variable | Default | Beschreibung |
|---|---|---|
| `HF_TOKEN` | — | **Pflicht.** HuggingFace-Token (Lese-Berechtigung reicht) |
| `WANDB_API_KEY` | — | Optional. W&B-Key. Ohne diesen läuft Training ohne W&B. |
| `MAX_STEPS` | `20000` | Anzahl Trainings-Steps. Auf ~241–301 Block-Stacking-Episoden sättigt BC früh; 20k konvergieren sauber (Lauf 1 war bei 30k bereits konvergiert). KISSKI-Multi-GPU-Default: `44000`. |
| `GLOBAL_BATCH_SIZE` | `8` | Globale Batch-Size (8 für < 40 GB VRAM, 32 für 4x A100 80 GB) |
| `NUM_GPUS` | `1` | Anzahl genutzter GPUs |
| `WANDB_PROJECT` | `gr00t-g1-dex3` | W&B-Projektname |
| `DATA_DIR` | `/data` | Datenverzeichnis im Container |
| `SKIP_DOWNLOAD` | `0` | `1` = HF-Download überspringen (Daten schon vorhanden) |
| `SKIP_CONVERT` | `0` | `1` = v3→v2-Konvertierung überspringen (`modality.json` existiert) |
| `SKIP_TRAIN` | `0` | `1` = nur Setup, dann Shell |
| `SHELL_ON_ERROR` | `0` | `1` = bei Fehler in Shell fallen statt zu beenden |
| `TUNE_VISUAL` | `0` | `1` = Vision-Encoder mittrainieren (`--tune_visual`). Entrypoint startet dann `run_finetuning_vision.sh` mit eigenem Output-/Experiment-Namespace (`blockstacking_vision`). LLM bleibt eingefroren. Höherer VRAM-Bedarf. |
| `LEARNING_RATE` | `1e-4` | Lernrate (`--learning_rate`); KISSKI-Multi-GPU-Default: `2e-4` |
| `WEIGHT_DECAY` | `1e-5` | Gewichtszerfall (`--weight_decay`) |
| `WARMUP_RATIO` | `0.05` | Anteil Warmup-Steps (`--warmup_ratio`); bei `TUNE_VISUAL=1` auf KISSKI: `0.1` |
| `DATALOADER_WORKERS` | `8` | Dataloader-Worker (`--dataloader_num_workers`); KISSKI-Default: `4` |
| `SAVE_STEPS` | `2000` | Checkpoint-Intervall in Steps (`--save_steps`); KISSKI-Default: `5000`. `2000` + Limit `10` → 10 gleichmäßig verteilte Checkpoints über den 20k-Lauf (Basis für die Open-Loop-Checkpoint-Auswahl). |
| `SAVE_TOTAL_LIMIT` | `10` | Max. Anzahl behaltener Checkpoints (`--save_total_limit`); KISSKI-Default: `40` |
| `USE_WANDB` | *auto* | W&B an/aus. Wird vom Entrypoint automatisch gesetzt: `1` wenn `WANDB_API_KEY` vorhanden, sonst `0`. Manuell `USE_WANDB=0` erzwingt Training ohne W&B. |
| `WANDB_MODE` | `offline` | W&B-Modus (vom Entrypoint gesetzt). `offline` puffert lokal — danach manuell syncen, siehe [wandb-offline-sync.md](wandb-offline-sync.md). |

## Optionale Trainings-Features (getrennt schaltbar)

Diese Schalter aktivieren einzelne Verfahren beim Trainingsstart — alle unabhängig voneinander.

| Variable | Default | Beschreibung |
|---|---|---|
| `TRAIN_TEST_SPLIT` | `0` | `1` = 80/20-Split scharf schalten: [`lib_split.sh`](../../Training/scripts/lib_split.sh) patcht `meta/info.json`, sodass nur die ersten `TRAIN_SPLIT_RATIO` der Episoden als `train` geladen werden; die restlichen stehen als `test` für die Open-Loop-Eval auf **ungesehenen** Episoden bereit. Zusätzlich landet ein Protokoll `split.json` im `OUTPUT_DIR`. `0` = kompletter Datensatz; ein zuvor gesetzter Split wird dann automatisch zurückgesetzt. **Gilt seit 2026-08-13 auch für `TUNE_VISUAL=1`** (vorher nur im Standard-Skript). |
| `TRAIN_SPLIT_RATIO` | `0.8` | Trainingsanteil bei `TRAIN_TEST_SPLIT=1` (Rest = Test). |
| `USE_AUGMENTATION` | `1` | Bild-Augmentierung / Domain-Randomization (Color-Jitter, optional Rotation/State-Dropout) gegen den Sim-Real-Domain-Gap. `0` = explizit aus (Color-Jitter auf 0, kein Modell-Default-Jitter). **Gilt seit 2026-08-13 auch für `TUNE_VISUAL=1`** — vorher war der Jitter im Vision-Skript fest verdrahtet und der Schalter dort wirkungslos. |
| `CJ_BRIGHTNESS` / `CJ_CONTRAST` / `CJ_SATURATION` / `CJ_HUE` | `0.3` / `0.4` / `0.5` / `0.08` | Color-Jitter-Stärken (nur bei `USE_AUGMENTATION=1`). |
| `RANDOM_ROTATION_ANGLE` | *(leer)* | Max. Rotationswinkel (Grad) für Bild-Rotations-Augmentierung; leer = aus. |
| `STATE_DROPOUT_PROB` | `0.0` | Dropout-Wahrscheinlichkeit auf den State-Inputs (Regularisierung); `0.0` = aus. |
| `USE_RL` | `0` | `1` = RL-Fine-tuning (FPO) gewünscht. Läuft **nicht** im BC-Trainingsimage (kein Isaac Sim): der BC-Entrypoint bricht mit einem Hinweis auf den RL-Pfad ab. RL braucht den kombinierten Isaac-Sim + GR00T-Container ([`Simulation/scripts/entrypoint_rl.sh`](../../Simulation/scripts/entrypoint_rl.sh) bzw. [`Training/kisski_rl_submit.sh`](../../Training/kisski_rl_submit.sh)) auf einer **RT-Core-GPU**. Details: [reinforcement-learning-plan.md](../weiterfuehrend/reinforcement-learning-plan.md). |
| `USE_COTRAIN` | `0` | `1` = **Co-Training auf echten UND gerenderten Bildern** (Schritt 4). Entrypoint startet [`run_finetuning_cotrain.sh`](../../Training/scripts/run_finetuning_cotrain.sh) mit eigenem Namespace `blockstacking_cotrain`; setzt `--tune_visual` selbst und hat Vorrang vor `TUNE_VISUAL`. Braucht einen gerenderten Datensatz (siehe unten). Anleitung: [co-training.md](co-training.md). |

### Co-Training: gerenderten Datensatz dazumischen (`USE_COTRAIN=1`)

Der gerenderte Datensatz entsteht auf dem **Sim-Server** (RT-Core-GPU) mit
`./Simulation/server_rl_run.sh render` und wird zum Trainings-Rechner transportiert
(~1–2 GB für 60 Episoden). Vollständige Herleitung der Werte:
[co-training.md §4](co-training.md#4-die-beiden-entscheidungen--und-wie-sie-begründet-sind).

| Variable | Default | Beschreibung |
|---|---|---|
| `COTRAIN_DATASET_PATH` | `/data/cotrain/g1_dex3_rendered` | Gerenderter Datensatz im Container (LeRobot v2.1, wie der echte). |
| `COTRAIN_HF_REPO` | *(leer)* | HF-**Dataset**-Repo; wird geladen, wenn `COTRAIN_DATASET_PATH` leer ist. |
| `COTRAIN_MIX_RATIO` | `0.5` ⚠️ | ⚠️ **Der Code-Default ist `0.5`, empfohlen wird `0.25`** — beides steht so im Repo, seit dem Menü-Abgleich vom 2026-08-20 dokumentiert. Entweder den Default in [`run_finetuning_cotrain.sh`](../../Training/scripts/run_finetuning_cotrain.sh) auf `0.25` ziehen oder die Empfehlung streichen. Anteil gerenderter Stichproben (Sampling-Wahrscheinlichkeit, **kein** Längenverhältnis). **Für den ersten Lauf 0.25 setzen:** bei 60 gerenderten gegen 240 echte Episoden sähe das Modell sonst jedes Sim-Bild 4× so oft wie jedes echte, und die Hälfte aller Updates entfiele auf ein Viertel des Bewegungsrepertoires. |
| `TRAIN_TEST_SPLIT` | `1` **hier** | Im Co-Training-Skript standardmäßig an — ohne zurückgehaltene Episoden fehlt die Leitplanke, an der ein Rückschritt auf der Realdomäne sichtbar würde. |

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
| `OUTPUT_DIR` | `/data/g1_dex3_finetune/blockstacking` (Vision-Lauf: `…_vision`) | Wurzel des Lauf-Namespace. |
| `EXPERIMENT_NAME` | `g1_dex3_blockstacking_v1` (Vision: `…_vision_v1`) | Unterverzeichnis **und** W&B-Run-Name. Für jeden neuen Lauf hochzählen. |
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



> **Full Fine-tuning benötigt laut NVIDIA ≥ 40 GB VRAM.** Karten mit < 24 GB VRAM führen zu
> OOM-Fehlern.

Bei `CUDA out of memory` zuerst `GLOBAL_BATCH_SIZE` halbieren. Volle Parameter-Referenz:
[`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](../../app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md).
