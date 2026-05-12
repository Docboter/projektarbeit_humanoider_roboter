# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fine-tuning of **NVIDIA GR00T N1.6** (Vision-Language-Action model) on the **Unitree G1 + DEX3-Hand** for block-stacking tasks. The full environment runs in Docker (CUDA 12.8, Python 3.10, package manager: `uv`).

Detailed guides live in the submodule:
- **Setup & Architecture:** [`app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md`](app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md)
- **Fine-tuning step-by-step:** [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md)
- **G1/DEX3 joint layout & datasets:** [`app/Groot-1.6/examples/G1_DEX3/README.md`](app/Groot-1.6/examples/G1_DEX3/README.md)
- **Full setup (German):** [`README.md`](README.md)

## Key commands (inside container)

```bash
# Build image (~30 min, first time only)
docker compose build

# Interactive shell
docker compose run --rm groot-training /bin/bash

# Download model + dataset (~25 GB, one-time)
docker compose run --rm groot-training bash /scripts/download_data.sh

# Convert dataset LeRobot v3.0 → v2.1 (one-time)
docker compose run --rm groot-training bash -c "
  cd /app/Groot-1.6
  python scripts/lerobot_conversion/convert_v3_to_v2_standalone.py \
    --repo-id unitreerobotics/G1_Dex3_BlockStacking_Dataset --root /data
  cp examples/G1_DEX3/modality_4cam.json \
     /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/modality.json
"

# Run fine-tuning (8 GB VRAM)
docker compose run --rm groot-training bash /scripts/run_finetuning.sh

# Run tests (CPU)
docker compose run --rm groot-training bash -c "
  cd /app/Groot-1.6 && python -m pytest tests/ -m 'not gpu' -v --timeout=300
"

# Lint + format
docker compose run --rm groot-training bash -c "
  cd /app/Groot-1.6 && pre-commit run --all-files
"
```

## Code style

**Formatter:** `ruff format` — **Linter:** `ruff check` (rules E, F, I) — **Line length:** 100  
Config: [`app/Groot-1.6/pyproject.toml`](app/Groot-1.6/pyproject.toml) under `[tool.ruff]`

## Architecture

```
/ (project root = container root)
├── Dockerfile / docker-compose.yml      # Container definition; mounts ./data → /data
├── scripts/
│   ├── download_data.sh                 # HuggingFace download (model + dataset)
│   └── run_finetuning.sh / .ps1         # Fine-tuning launcher
└── app/                                 # Git submodule (pinned commit, cloned in Dockerfile)
    └── Groot-1.6/                       # PRIMARY — custom fork (lucam06, branch luca/g1-dex3)
        ├── gr00t/experiment/launch_finetune.py   # Training entry point
        ├── gr00t/policy/                # Gr00tPolicy inference class
        ├── examples/G1_DEX3/           # Embodiment config, modality JSONs, guides
        └── scripts/lerobot_conversion/ # Dataset format converter
```

**All G1/DEX3 work goes in `app/Groot-1.6/`.** The Python venv lives at `app/Groot-1.6/.venv`; use `uv run` to invoke it. Data is never committed — it lives in `./data/` on the host, mounted at `/data` in the container.

## VRAM constraints

| VRAM | `--global_batch_size` | `--max_steps` |
|---|---|---|
| 8 GB  | 8  | 30 000 |
| 16 GB | 16–32 | 50 000 |

Reduce batch size first on OOM. See [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md) for full parameter reference.
