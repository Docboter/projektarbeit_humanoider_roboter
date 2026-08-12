# Konfiguration über Env-Vars

Alle Trainings-Parameter werden über Umgebungsvariablen gesteuert — auf vast.ai, KISSKI und
lokal **identisch**. Der Entrypoint (`/scripts/entrypoint.sh`) liest sie ein; Defaults sind als
`ENV` im [Dockerfile](../../Training/Dockerfile) gesetzt.

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
| `LEARNING_RATE` | `1e-4` | Lernrate (`--learning_rate`) |
| `DATALOADER_WORKERS` | `8` | Dataloader-Worker (`--dataloader_num_workers`); KISSKI-Default: `4` |
| `SAVE_STEPS` | `2000` | Checkpoint-Intervall in Steps (`--save_steps`); KISSKI-Default: `5000`. `2000` + Limit `10` → 10 gleichmäßig verteilte Checkpoints über den 20k-Lauf (Basis für die Open-Loop-Checkpoint-Auswahl). |
| `SAVE_TOTAL_LIMIT` | `10` | Max. Anzahl behaltener Checkpoints (`--save_total_limit`); KISSKI-Default: `40` |
| `USE_WANDB` | *auto* | W&B an/aus. Wird vom Entrypoint automatisch gesetzt: `1` wenn `WANDB_API_KEY` vorhanden, sonst `0`. Manuell `USE_WANDB=0` erzwingt Training ohne W&B. |
| `WANDB_MODE` | `offline` | W&B-Modus (vom Entrypoint gesetzt). `offline` puffert lokal — danach manuell syncen, siehe [wandb-offline-sync.md](wandb-offline-sync.md). |

## Optionale Trainings-Features (getrennt schaltbar)

Diese Schalter aktivieren einzelne Verfahren beim Trainingsstart — alle unabhängig voneinander.

| Variable | Default | Beschreibung |
|---|---|---|
| `TRAIN_TEST_SPLIT` | `0` | `1` = 80/20-Split scharf schalten: `run_finetuning.sh` patcht `meta/info.json`, sodass nur die ersten `TRAIN_SPLIT_RATIO` der Episoden als `train` geladen werden; die restlichen stehen als `test` für die Open-Loop-Eval auf **ungesehenen** Episoden bereit. `0` = kompletter Datensatz (bisheriges Verhalten); ein zuvor gesetzter Split wird dann automatisch zurückgesetzt. |
| `TRAIN_SPLIT_RATIO` | `0.8` | Trainingsanteil bei `TRAIN_TEST_SPLIT=1` (Rest = Test). |
| `USE_AUGMENTATION` | `1` | Bild-Augmentierung / Domain-Randomization (Color-Jitter, optional Rotation/State-Dropout) gegen den Sim-Real-Domain-Gap. `0` = explizit aus (Color-Jitter auf 0, kein Modell-Default-Jitter). |
| `CJ_BRIGHTNESS` / `CJ_CONTRAST` / `CJ_SATURATION` / `CJ_HUE` | `0.3` / `0.4` / `0.5` / `0.08` | Color-Jitter-Stärken (nur bei `USE_AUGMENTATION=1`). |
| `RANDOM_ROTATION_ANGLE` | *(leer)* | Max. Rotationswinkel (Grad) für Bild-Rotations-Augmentierung; leer = aus. |
| `STATE_DROPOUT_PROB` | `0.0` | Dropout-Wahrscheinlichkeit auf den State-Inputs (Regularisierung); `0.0` = aus. |
| `USE_RL` | `0` | `1` = RL-Fine-tuning (FPO) gewünscht. Läuft **nicht** im BC-Trainingsimage (kein Isaac Sim): der BC-Entrypoint bricht mit einem Hinweis auf den RL-Pfad ab. RL braucht den kombinierten Isaac-Sim + GR00T-Container ([`Simulation/scripts/entrypoint_rl.sh`](../../Simulation/scripts/entrypoint_rl.sh) bzw. [`Training/kisski_rl_submit.sh`](../../Training/kisski_rl_submit.sh)) auf einer **RT-Core-GPU**. Details: [reinforcement-learning-plan.md](../weiterfuehrend/reinforcement-learning-plan.md). |

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

> **Hinweis:** Der RL-Trainer ([`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py)) läuft seit 2026-08-08 end-to-end auf RT-Core-Hardware (RTX PRO 6000 Blackwell, Isaac Sim 6.0) — Rollout, FPO-Update und Checkpoint-Schreiben sind nachgewiesen. Offen ist die **Lernwirkung** (steigt `success_rate` über viele Iterationen?); die Hyperparameter oben sind ungetunt. Der Greif-Physik-Blocker der Läufe 25–28 („auch ohne Modell wird kein Würfel angehoben") ist mit **Lauf 29** (2026-08-12) gefallen: mit korrigiertem Fingerkuppen-Messpunkt hebt die Hand den Würfel 7,9 cm — ein Lernsignal ist damit erreichbar. Details: [rl-anleitung.md](../weiterfuehrend/rl-anleitung.md#lauf-29-der-würfel-hebt-ab-mit-einem-vorbehalt).

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



> **Full Fine-tuning benötigt laut NVIDIA ≥ 40 GB VRAM.** Karten mit < 24 GB VRAM führen zu
> OOM-Fehlern.

Bei `CUDA out of memory` zuerst `GLOBAL_BATCH_SIZE` halbieren. Volle Parameter-Referenz:
[`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](../../app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md).
