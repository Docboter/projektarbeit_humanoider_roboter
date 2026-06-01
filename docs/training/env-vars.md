# Konfiguration über Env-Vars

Alle Trainings-Parameter werden über Umgebungsvariablen gesteuert — auf vast.ai, KISSKI und
lokal **identisch**. Der Entrypoint (`/scripts/entrypoint.sh`) liest sie ein; Defaults sind als
`ENV` im [Dockerfile](../../Training/Dockerfile) gesetzt.

| Variable | Default | Beschreibung |
|---|---|---|
| `HF_TOKEN` | — | **Pflicht.** HuggingFace-Token (Lese-Berechtigung reicht) |
| `WANDB_API_KEY` | — | Optional. W&B-Key. Ohne diesen läuft Training ohne W&B. |
| `MAX_STEPS` | `30000` | Anzahl Trainings-Steps |
| `GLOBAL_BATCH_SIZE` | `8` | Globale Batch-Size (8 für 8 GB VRAM, 32+ für A100 80 GB) |
| `NUM_GPUS` | `1` | Anzahl genutzter GPUs |
| `WANDB_PROJECT` | `gr00t-g1-dex3` | W&B-Projektname |
| `DATA_DIR` | `/data` | Datenverzeichnis im Container |
| `SKIP_DOWNLOAD` | `0` | `1` = HF-Download überspringen (Daten schon vorhanden) |
| `SKIP_CONVERT` | `0` | `1` = v3→v2-Konvertierung überspringen (`modality.json` existiert) |
| `SKIP_TRAIN` | `0` | `1` = nur Setup, dann Shell |
| `SHELL_ON_ERROR` | `0` | `1` = bei Fehler in Shell fallen statt zu beenden |

## VRAM-Richtwerte

| VRAM | `GLOBAL_BATCH_SIZE` | `MAX_STEPS` | Umgebung |
|---|---|---|---|
| 24 GB | 1–2 | 30 000 | Lokal (RTX 4090, min.) — sehr langsam |
| 32 GB | 4–8 | 30 000 | Lokal (RTX 5090, ~31 GB bei bs=8) |
| 40 GB | 16–32 | 50 000 | vast.ai A100 40 GB |
| 80 GB | 64–128 | 50 000+ | KISSKI A100 80 GB |
| 94 GB | 128+ | 50 000+ | KISSKI H100 94 GB |

> **Full Fine-tuning benötigt laut NVIDIA ≥ 40 GB VRAM.** Karten mit < 24 GB VRAM führen zu
> OOM-Fehlern.

Bei `CUDA out of memory` zuerst `GLOBAL_BATCH_SIZE` halbieren. Volle Parameter-Referenz:
[`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](../../app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md).
