# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fine-tuning of **NVIDIA GR00T N1.6** (Vision-Language-Action model) on the **Unitree G1 + DEX3-Hand** for block-stacking tasks. The full environment runs in a **self-contained Docker container** (CUDA 12.8, Python 3.10, package manager: `uv`).

The container is autonomous: launching the image triggers `scripts/entrypoint.sh`, which orchestrates download → conversion → training. The same image runs locally and on cloud-GPU platforms like vast.ai — configuration is via env vars only.

**Storage model:** No host-side persistent storage. All data, checkpoints, and logs live in the container filesystem (`/data` is a regular directory inside the image, **not** a volume mount). The container is meant to be long-lived: `stop`/`start` preserves state; only `docker rm` destroys it. This matches the vast.ai semantics where one instance == one container.

Detailed guides:
- **User-facing instructions (German):** [`Anleitung.md`](Anleitung.md)
- **Project README (German):** [`README.md`](README.md)
- **Setup & Architecture:** [`app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md`](app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md)
- **Fine-tuning step-by-step:** [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md)
- **G1/DEX3 joint layout & datasets:** [`app/Groot-1.6/examples/G1_DEX3/README.md`](app/Groot-1.6/examples/G1_DEX3/README.md)

## Key commands

### Run the full pipeline (download → convert → train)

```bash
# Local: thin launcher (handles existing-container detection)
HF_TOKEN=hf_... WANDB_API_KEY=... ./setup_and_train_DockerHub-pull.sh

# Or manually — NO --rm, NO -v mount:
docker run --name groot-train --gpus all --ipc=host --shm-size=16g \
  -e HF_TOKEN=hf_... -e WANDB_API_KEY=... \
  -e MAX_STEPS=30000 -e GLOBAL_BATCH_SIZE=8 \
  -it lucam03/projekt-humanoider-roboter:latest
```

**Critical:** never pass `--rm`. The container holds all data and checkpoints — `--rm` wipes them on stop.

### Resume / inspect / destroy

```bash
docker start -ai groot-train             # continue after stop (data preserved)
docker exec -it groot-train bash         # shell into running container
docker cp groot-train:/data/g1_dex3_finetune ./checkpoints   # extract artifacts
docker rm -f groot-train                 # destroy everything
```

### Build the image

```bash
docker build -t projektarbeit-humanoider-roboter .
# ~30 min first time (PyTorch, flash-attn). The Dockerfile clones the
# Groot-1.6 submodule itself — no `git clone --recurse-submodules` needed
# before building.
```

### Interactive shell (entrypoint bypassed when a command is passed)

```bash
docker run -it --name groot-debug --gpus all --ipc=host --shm-size=16g \
  lucam03/projekt-humanoider-roboter:latest bash
```

Inside the container:

```bash
# Re-run just the training (entrypoint normally calls this last)
bash /scripts/run_finetuning.sh

# Tests (CPU)
cd /app/Groot-1.6 && python -m pytest tests/ -m 'not gpu' -v --timeout=300

# Lint + format
cd /app/Groot-1.6 && pre-commit run --all-files
```

## Env-var configuration

The entrypoint reads everything from env vars. Defaults are set as `ENV` in the Dockerfile.

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | **Required.** HuggingFace token |
| `WANDB_API_KEY` | — | Optional. Without it, training runs without W&B |
| `MAX_STEPS` | `30000` | Training steps |
| `GLOBAL_BATCH_SIZE` | `8` | 8 for 8 GB VRAM, 16–32 for 16 GB |
| `NUM_GPUS` | `1` | GPUs to use |
| `WANDB_PROJECT` | `gr00t-g1-dex3` | W&B project name |
| `DATA_DIR` | `/data` | Container-side data dir |
| `SKIP_DOWNLOAD` | `0` | Skip HF download (data already present in container) |
| `SKIP_CONVERT` | `0` | Skip v3→v2 conversion (modality.json exists) |
| `SKIP_TRAIN` | `0` | Run setup only, then drop to shell |
| `SHELL_ON_ERROR` | `0` | Drop to shell on error instead of exiting |

## Code style

**Formatter:** `ruff format` — **Linter:** `ruff check` (rules E, F, I) — **Line length:** 100
Config: [`app/Groot-1.6/pyproject.toml`](app/Groot-1.6/pyproject.toml) under `[tool.ruff]`

## Architecture

```
/ (project root = container root after build)
├── Dockerfile                          # Defines image; ENTRYPOINT = /scripts/entrypoint.sh
├── docker-compose.yml                  # Optional (dev convenience; no host volume mounts)
├── setup_and_train_DockerHub-pull.sh   # Thin host launcher: docker pull + docker run
├── setup_and_train_DockerHub-pull.ps1  # Windows variant
├── scripts/                            # COPIED into image at /scripts/
│   ├── entrypoint.sh                   # Autonomous orchestrator (download→convert→train)
│   ├── download_data.sh                # HuggingFace download (model + dataset)
│   └── run_finetuning.sh               # Training launcher (called by entrypoint)
└── app/                                # Git submodule, cloned in Dockerfile at build time
    └── Groot-1.6/                      # PRIMARY — custom fork (lucam06, pinned commit)
        ├── gr00t/experiment/launch_finetune.py  # Training entry point
        ├── gr00t/policy/               # Gr00tPolicy inference class
        ├── examples/G1_DEX3/           # Embodiment config, modality JSONs, guides
        └── scripts/lerobot_conversion/ # Dataset format converter (v3.0 → v2.1)
```

At runtime, the container holds (no host mount):
```
/data/
├── models/GR00T-N1.6-3B/    # ~6 GB (downloaded by entrypoint)
├── unitreerobotics/          # ~18 GB
├── g1_dex3_finetune/         # checkpoints
└── logs/                     # training logs
```

**All G1/DEX3 work goes in `app/Groot-1.6/`.** The Python venv lives at `app/Groot-1.6/.venv`; use `uv run` to invoke it.

### Important: scripts are COPIED into the image, no host-side data persistence

- Changes to `scripts/*.sh` require a rebuild (no bind-mount).
- All data (`/data`) lives only in the container filesystem.
- `docker run --rm` would destroy training results — never use it with this image.
- For dev iteration: bind-mount manually:
  ```bash
  docker run -it --name groot-dev -v $(pwd)/scripts:/scripts \
    --gpus all --ipc=host --shm-size=16g \
    lucam03/projekt-humanoider-roboter:latest bash
  ```

## VRAM constraints

| VRAM | `GLOBAL_BATCH_SIZE` | `MAX_STEPS` |
|---|---|---|
| 8 GB  | 8  | 30 000 |
| 16 GB | 16–32 | 50 000 |

Reduce batch size first on OOM. See [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md) for full parameter reference.
