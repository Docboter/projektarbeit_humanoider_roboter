# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fine-tuning of **NVIDIA GR00T N1.6** (Vision-Language-Action model) on the **Unitree G1 + DEX3-Hand** for block-stacking tasks. The full environment runs in a **self-contained Docker container** (CUDA 12.8, Python 3.10, package manager: `uv`).

The container is autonomous: launching the image triggers `/scripts/entrypoint.sh` (source: `Training/scripts/entrypoint.sh`), which orchestrates download → conversion → training. The same image runs locally, on cloud-GPU platforms like vast.ai, and on the **KISSKI HPC cluster** (GWDG Göttingen) via Apptainer — configuration is via env vars only.

**Storage model:** No host-side persistent storage by default. All data, checkpoints, and logs live in the container filesystem (`/data` is a regular directory inside the image, **not** a volume mount). The container is meant to be long-lived: `stop`/`start` preserves state; only `docker rm` destroys it. This matches the vast.ai semantics where one instance == one container. **Exception — KISSKI:** the cluster uses Apptainer (not Docker) and bind-mounts `/mnt/vast-kisski/projects/kisski-humrob/data` to `/data` inside the container; data therefore persists on the cluster's VAST project storage across job runs. (The old SCRATCH-SCC storage `/scratch/` was decommissioned on 2026-03-31.)

Detailed guides (all prose docs live under [`docs/`](docs/README.md)):
- **Doc navigation hub:** [`docs/README.md`](docs/README.md)
- **User-facing training instructions (German):** [`docs/training/anleitung.md`](docs/training/anleitung.md)
- **Project README (German):** [`README.md`](README.md)
- **Env-var reference:** [`docs/training/env-vars.md`](docs/training/env-vars.md)
- **KISSKI HPC training:** [`docs/training/kisski-hpc.md`](docs/training/kisski-hpc.md)
- **Setup & Architecture:** [`app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md`](app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md)
- **Fine-tuning step-by-step:** [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md)
- **G1/DEX3 joint layout & datasets:** [`app/Groot-1.6/examples/G1_DEX3/README.md`](app/Groot-1.6/examples/G1_DEX3/README.md)
- **Sim eval on vast.ai (German):** [`docs/simulation/vastai-anleitung.md`](docs/simulation/vastai-anleitung.md)
- **Sim implementation notes & lessons learned:** [`docs/simulation/umsetzungsnotizen.md`](docs/simulation/umsetzungsnotizen.md)
- **Results & evaluation (German):** [`docs/ergebnisse/`](docs/ergebnisse/README.md) — run analyses, domain-gap, sim methodology review, baseline
- **Further work / concepts (German):** [`docs/weiterfuehrend/`](docs/weiterfuehrend/README.md) — RL plan + operative RL guide ([`rl-anleitung.md`](docs/weiterfuehrend/rl-anleitung.md); RL runs end-to-end on the Blackwell server; the runs 25–28 grasp-physics blocker fell in run 29 (2026-08-12) — with the corrected fingertip reference point the hand lifts a cube 7.9 cm, so a reward signal is reachable; learning effect still unverified), locomotion research (not implemented), livestream plan (Spur A/WebRTC open; Spur B/MJPEG `LIVE_VIEW` is built)

## Key commands

### Run the full pipeline (download → convert → train)

```bash
# Local: thin launcher (handles existing-container detection)
HF_TOKEN=hf_... WANDB_API_KEY=... ./Training/setup_and_train_DockerHub-pull.sh

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

### KISSKI HPC cluster (Apptainer + SLURM)

```bash
# One-time: convert Docker image to Apptainer SIF (on glogin-gpu.hpc.gwdg.de)
module load apptainer
apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
    docker://lucam03/projekt-humanoider-roboter:latest

# Submit training job
export HF_TOKEN=hf_... WANDB_API_KEY=... GLOBAL_BATCH_SIZE=32
sbatch Training/kisski_submit.sh

# Monitor
squeue -u $USER
tail -f logs/slurm-<jobid>.out

# Retrieve checkpoints
rsync -avz <username>@transfer.hpc.gwdg.de:/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/ ./checkpoints/
```

KISSKI partitions: `kisski` (A100 80 GB) and `kisski-h100` (H100 94 GB), max walltime 48 h.

### Sim eval on vast.ai (build → push → run)

Full guide: [`docs/simulation/vastai-anleitung.md`](docs/simulation/vastai-anleitung.md)
Known fixes & GPU requirements: [`docs/simulation/umsetzungsnotizen.md`](docs/simulation/umsetzungsnotizen.md)

```powershell
# 1. Build + push sim image (includes entrypoint_sim.sh with unset VIRTUAL_ENV fix)
.\Simulation\update_sim_image.ps1 -VastAI

# 2. Upload checkpoint to HuggingFace (skips optimizer.pt by default)
python Simulation\scripts\upload_checkpoint.py `
  --checkpoint "C:\path\to\checkpoint-3000" --repo luca-mue/groot-g1dex3-checkpoint

# 3. Generate USD asset (one-time, local Docker)
#    → see docs/simulation/vastai-anleitung.md Schritt 3, or data/g1_dex3.usd already exists
```

On vast.ai: GPU must be **Ampere+ with RT-Cores** (L40, RTX 4090, A6000) — A100/H100 lack RT-Cores; RTX 5000 is Turing (too old for Isaac Sim 4.x). Env vars for the sim container:

| Variable | Example | Purpose |
|---|---|---|
| `HF_TOKEN` | `hf_...` | Required for HF checkpoint download |
| `HF_CHECKPOINT_REPO` | `luca-mue/groot-g1dex3-checkpoint` | Auto-download checkpoint + USD from HF |
| `CHECKPOINT_PATH` | `/data/checkpoints/groot-g1dex3-checkpoint` | Path after download |
| `ASSET_PATH` | `/data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd` | USD robot asset |
| `NUM_EPISODES` | `20` | Eval episodes |
| `EPISODE_LENGTH_S` | `40` | Cap episode length in seconds. Unset = up to 9000 steps per episode — 20 episodes can take hours |
| `SHELL_ON_ERROR` | `1` | Drop to shell on failure (recommended) |
| `LIVESTREAM` | `0` | `0`=headless (default), `1`=WebRTC public, `2`=WebRTC private/local — **the LIVE variant**: Isaac Sim streams its 3D viewport, opened by the native *Isaac Sim WebRTC Streaming Client* on your own machine. Applies to all four runs (sim-eval, baseline, replay/grasp, RL). Replaces MP4 recording by default. Shared logic in [`lib_livestream.sh`](Simulation/scripts/lib_livestream.sh); guide: [live-ansicht.md](docs/simulation/live-ansicht.md). Built 2026-08-13, **untested on hardware** — run `server_rl_run.sh livecheck` first |
| `LIVESTREAM_PORT` | `49100` | WebRTC signaling port (TCP). **On vast.ai: set to the externally-mapped port** (internal==external, else SDP port mismatch). `server_rl_run.sh` maps `49100/tcp` + `47998/udp` when creating the container |
| `LIVESTREAM_MEDIA_PORT` | `47998` | WebRTC media port (UDP) — must pass the firewall, else the client connects but the picture stays black |
| `LIVE_KEEP_VIDEO` | `0` | `1` = also write MP4s while the LIVE variant runs (default: live **instead of** video — `--video-dir` is passed empty) |
| `LIVE_VIEW` | `0` | `1` = MJPEG frame stream in the browser ("Spur B", [`live_view.py`](Simulation/g1_dex3_sim/live_view.py)) — plain HTTP, unlimited viewers, stateless, `ssh -L`-tunnelable. Wired into the RL trainer; costs no extra render pass (`cam_scene` is rendered every step anyway) |
| `LIVE_VIEW_PORT` | `8900` | HTTP port of the frame stream (map `-p 8900:8900`; `server_rl_run.sh` does it on container creation) |
| `LIVE_VIEW_EVERY_N` | `1` | Publish only every n-th frame |
| `LIVE_VIEW_CAMS` | `cam_left_high,cam_left_wrist` | Comma-separated cameras shown side by side. Defaults to the **calibrated policy cameras** (= the model's actual input). `cam_scene` is an unvalidated overview cam — diagnose with `server_rl_run.sh cams` ([`dump_camera_poses.py`](Simulation/g1_dex3_sim/dump_camera_poses.py)) |
| `RL_WANDB_VIDEO_EVERY` | `0` | RL only: log a rollout video to W&B every N iterations (`0` = off) |

### Build the image

```bash
docker build -t projektarbeit-humanoider-roboter Training/   # build context = Training/
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
| `MAX_STEPS` | `20000` | Training steps (KISSKI multi-GPU default: 44000) |
| `GLOBAL_BATCH_SIZE` | `8` | 8 for 8 GB VRAM, 16–32 for 16 GB |
| `NUM_GPUS` | `1` | GPUs to use |
| `WANDB_PROJECT` | `gr00t-g1-dex3` | W&B project name |
| `DATA_DIR` | `/data` | Container-side data dir |
| `SKIP_DOWNLOAD` | `0` | Skip HF download (data already present in container) |
| `SKIP_CONVERT` | `0` | Skip v3→v2 conversion (modality.json exists) |
| `SKIP_TRAIN` | `0` | Run setup only, then drop to shell |
| `SHELL_ON_ERROR` | `0` | Drop to shell on error instead of exiting |
| `TUNE_VISUAL` | `0` | `1` = also train the vision encoder (`--tune_visual`); entrypoint routes to `run_finetuning_vision.sh` (own `blockstacking_vision` namespace, LLM stays frozen, higher VRAM) |
| `TRAIN_TEST_SPLIT` | `0` | `1` = activate 80/20 split (`run_finetuning.sh` patches `meta/info.json`; test episodes held out). `TRAIN_SPLIT_RATIO` (0.8) sets the ratio |
| `USE_AUGMENTATION` | `1` | Image augmentation / domain randomization (color jitter; `CJ_*`, `RANDOM_ROTATION_ANGLE`, `STATE_DROPOUT_PROB`). `0` = explicitly off |
| `USE_RL` | `0` | `1` = RL fine-tuning (FPO). Not in the BC image (no Isaac Sim) — BC entrypoint errors with a pointer to the sim-image RL path (`entrypoint_rl.sh` / `kisski_rl_submit.sh`, RT-core GPU). See [docs/training/env-vars.md](docs/training/env-vars.md) + [RL plan](docs/weiterfuehrend/reinforcement-learning-plan.md) |
| `USE_COTRAIN` | `0` | `1` = co-training on real **and** rendered images (step 4); routes to `run_finetuning_cotrain.sh` (namespace `blockstacking_cotrain`, sets `--tune_visual` itself, takes precedence over `TUNE_VISUAL`). Needs a rendered dataset from `server_rl_run.sh render` via `COTRAIN_DATASET_PATH` / `COTRAIN_HF_REPO`; `COTRAIN_MIX_RATIO` is the share of rendered samples (0.25 recommended, see [co-training.md](docs/training/co-training.md)) |

## Code style

**Formatter:** `ruff format` — **Linter:** `ruff check` (rules E, F, I) — **Line length:** 100
Config: [`app/Groot-1.6/pyproject.toml`](app/Groot-1.6/pyproject.toml) under `[tool.ruff]`

**Commit messages: English.** Subject and body. This holds regardless of the language of the
code comments or docs the commit touches — those stay German where they already are.

## Architecture

```
repo root
├── README.md                           # Slim landing page (overview + quickstart + doc links)
├── docs/                               # ALL prose docs live here
│   ├── README.md                       # Doc navigation hub + project structure
│   ├── training/                       # operative guides: anleitung.md, kisski-hpc.md, env-vars.md,
│   │                                   #   multi-gpu.md, train-test-split.md, wandb-offline-sync.md,
│   │                                   #   fixes-aus-erstem-lauf.md, co-training.md (step 4: real +
│   │                                   #   rendered images; tools built 2026-08-14, run still pending)
│   ├── simulation/                     # operative guides: vastai-anleitung.md, umsetzungsnotizen.md (READ FIRST)
│   │   └── archiv/                     # superseded planning docs (isaac-lab-plan, sim-docker-build,
│   │                                   #   kisski-desktop, gpu-kompatibilitaet)
│   ├── ergebnisse/                     # evaluations: lauf1-auswertung.md, lauf2-vision-auswertung.md,
│   │                                   #   lauf3-vision-split-auswertung.md (first real validation:
│   │                                   #   checkpoint sweep U-curve, best ckpt 30000, last one 25% worse),
│   │                                   #   wandb-run-auswertung.md, domain-gap-analyse.md,
│   │                                   #   sim-bewertung.md, baseline-unitree-g1.md
│   ├── weiterfuehrend/                 # reinforcement-learning-plan.md (+ rl-anleitung.md operative guide;
│   │                                   #   RL runs end-to-end; runs 25–28 grasp blocker fell in run 29 —
│   │                                   #   cube lifts 7.9 cm), lokomotion-recherche.md
│   │                                   #   (not implemented), livestream-plan.md (Spur A open, Spur B built)
│   ├── umgebungsanalyse.md             # cross-cutting audit
│   └── fehlerbehebung.md               # cross-cutting troubleshooting
├── Training/                           # Everything training-related (build, run scripts)
│   ├── Dockerfile                      # Defines image; ENTRYPOINT = /scripts/entrypoint.sh
│   │                                   #   build context = Training/ (so COPY scripts/ works)
│   ├── docker-compose.yml              # Optional (dev convenience; no host volume mounts)
│   ├── kisski_submit.sh                # SLURM batch script for KISSKI HPC cluster
│   ├── kisski_open_loop_eval.sh        # SLURM job: open-loop checkpoint eval (open_loop_eval.py, no server)
│   ├── kisski_rl_submit.sh             # SLURM job: RL fine-tuning (FPO) — sim SIF, RT-core GPU guard (TEMPLATE)
│   ├── update_image.ps1                # Host build/push tool (must sit next to Dockerfile)
│   ├── update_image.sh                 # Linux/bash port of update_image.ps1
│   ├── setup_and_train_DockerHub-pull.sh   # Thin host launcher: docker pull + docker run
│   ├── setup_and_train_DockerHub-pull.ps1  # Windows variant
│   ├── setup_and_train_Container-build.* # Host launcher that builds the image locally
│   └── scripts/                        # COPIED into image at /scripts/
│       ├── entrypoint.sh               # Autonomous orchestrator (download→convert→train; TUNE_VISUAL routes here)
│       ├── download_data.sh            # HuggingFace download (model + dataset)
│       ├── run_finetuning.sh           # Training launcher (called by entrypoint; torchrun for NUM_GPUS>1)
│       ├── run_finetuning.ps1          # Windows variant of the training launcher
│       ├── run_finetuning_vision.sh    # Vision-encoder variant (TUNE_VISUAL=1, blockstacking_vision namespace)
│       ├── run_finetuning_cotrain.sh   # ★ Step 4: co-training on real + RENDERED images (USE_COTRAIN=1,
│       │                               #   blockstacking_cotrain namespace, split on by default)
│       ├── launch_cotrain.py           # Two-dataset training entry point (mix_ratio). Copy of the fork's
│       │                               #   launch_finetune.py with ONE change — the datasets list — so no
│       │                               #   submodule change / image rebuild is needed. Re-check on GR00T bumps
│       ├── lib_split.sh                # Shared train/test-split logic, sourced by all training launchers;
│       │                               #   writes a split.json record next to the checkpoints
│       └── checkpoint_sweep.py         # ★ Checkpoint SELECTION: open-loop MSE/MAE of every checkpoint on the
│                                       #   HELD-OUT episodes. The fork has no in-training eval
│                                       #   (enable_open_loop_eval is dead config; factory.py asserts
│                                       #   eval_strategy=="no"), so validation happens after the run
├── Simulation/                         # Closed-loop sim eval
│   ├── Dockerfile                      # KISSKI-only: slim Isaac Lab sim-client (no GR00T)
│   ├── Dockerfile.vastai               # vast.ai: combined Isaac Sim + GR00T in one container
│   ├── kisski_sim_submit.sh            # SLURM job for sim eval (jupyter partition, RTX 5000)
│   ├── server_rl_run.sh                # ★ Own-server workflow (Docker): preflight/setup/check/cams/
│   │                                   #   gap/eval/grasp/span/render/livecheck/rl/shell/clean subcommands
│   │                                   #   (see rl-anleitung.md; livecheck = phase 0 of the LIVE variant;
│   │                                   #   render = builds the co-training dataset, see co-training.md)
│   ├── server_robocasa_ref_run.sh      # Own-server RoboCasa GR-1 reference eval (pipeline validation)
│   ├── update_sim_image.ps1            # Build/push tool (-VastAI flag for Dockerfile.vastai)
│   ├── update_sim_image.sh             # Linux/bash port of update_sim_image.ps1
│   │                                   #   (sim docs moved to docs/simulation/)
│   ├── g1_dex3_sim/                    # COPIED into image at /workspace/g1_dex3_sim/
│   │   ├── run_g1_dex3_sim_eval.py     # Main eval loop (model-based, ZMQ client to GR00T server)
│   │   ├── run_g1_dex3_replay.py       # Open-loop dataset-replay DIAGNOSTIC (no server/model)
│   │   ├── render_cotrain_dataset.py   # ★ Step 4: replays REAL dataset actions and records the 4 policy
│   │   │                               #   cameras → LeRobot v2.1 dataset of (sim image, real action) pairs.
│   │   │                               #   Two stages: `scan` finds each episode's grasp point (cheap,
│   │   │                               #   cameras at 1/10), `render` puts the cubes there and writes at the
│   │   │                               #   calibrated 640×480. Resumable; never renders held-out episodes
│   │   ├── replay_episode0.npz         # Bundled ground-truth actions (dataset ep. 0) for replay
│   │   ├── g1_dex3_blockstack_env.py   # Isaac Lab env (robot, table, cubes, 4 policy + 1 scene cam; reward_mode binary|shaped, get_obs_batched)
│   │   ├── rl_finetune.py              # FPO RL trainer (action-head only; shaped reward; one full iteration verified on hardware)
│   │   ├── live_view.py                # MJPEG live view (stdlib + Pillow; LIVE_VIEW=1) — hooked into the RL rollout
│   │   ├── dump_camera_poses.py        # Diagnostic: configured vs. actually rendered camera pose + one PNG per cam
│   │   ├── g1_dex3_cfg.py              # Articulation + camera config (look_at_world_quat helper)
│   │   ├── client.py                   # ZMQ policy client + build_obs (state split into modality keys)
│   │   ├── convert_urdf_to_usd.py      # One-time URDF→USD conversion
│   │   └── ...
│   ├── camera_reference/               # Dataset reference frames (camera-calibration targets)
│   ├── g1_gripper_sim/                 # Stock-G1 gripper baseline sim (SIM_MODE=baseline)
│   ├── robocasa_reference/             # RoboCasa GR-1 reference-eval scripts (run_robocasa_ref_eval.sh)
│   └── scripts/                        # COPIED into vastai image at /scripts/
│       ├── entrypoint_sim.sh           # Autonomous entrypoint (model eval) for Dockerfile.vastai
│       ├── entrypoint_replay.sh        # Entrypoint for the open-loop replay diagnostic
│       ├── entrypoint_baseline.sh      # Entrypoint for baseline eval (SIM_MODE=baseline: un-finetuned model + stock G1)
│       ├── entrypoint_rl.sh            # Entrypoint for RL fine-tuning (FPO; loads BC checkpoint, runs rl_finetune.py)
│       ├── lib_livestream.sh           # Shared LIVE-variant logic (WebRTC viewport): sourced by all four
│       │                               #   entrypoints AND by server_rl_run.sh on the host. Version-aware Kit
│       │                               #   settings, live-instead-of-video rule, connection banner
│       ├── measure_domain_gap.py       # Real→sim cosine-distance per camera via frozen SigLIP-ViT
│       ├── overlay_camera_check.py     # Overlays dataset vs sim camera frames (calibration check)
│       ├── dump_unitree_g1_dims.py     # Dumps stock UNITREE_G1 link/joint dimensions
│       ├── check_action_norm.py        # Static check: checkpoint action-norm stats vs. dataset (no GPU/torch)
│       ├── finger_span_openloop.py     # Commanded finger span on REAL images — domain-gap vs. model discriminator
│       ├── policy_latency.py           # Pure policy latency (ms per action chunk), in-process, no sim/ZMQ.
│       │                               #   The one number here that also holds on real hardware — rendering
│       │                               #   (94% of sim wall-clock) does not exist there. `server_rl_run.sh latency`
│       └── upload_checkpoint.py        # HuggingFace upload helper (skips optimizer.pt by default)
├── data/                               # Local assets and submodules (mostly gitignored)
│   ├── unitree_ros/                    # Git submodule — Unitree ROS packages (URDF source)
│   ├── g1_dex3.usd                     # Generated robot USD asset (run convert_urdf_to_usd.py)
│   └── configuration/                  # Companion USD files referenced by g1_dex3.usd
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

- Changes to `Training/scripts/*.sh` require a rebuild (no bind-mount).
- All data (`/data`) lives only in the container filesystem (or on the VAST project storage `/mnt/vast-kisski/projects/kisski-humrob/data` on KISSKI).
- `docker run --rm` would destroy training results — never use it with this image.
- `Training/scripts/run_finetuning.sh` detects both Docker (`.dockerenv`) and Apptainer (`$APPTAINER_NAME` / `$SINGULARITY_NAME`) environments — no changes needed when running on KISSKI.
- `kisski_submit.sh` bind-mounts `${REPO_DIR}/Training/scripts:/scripts` so repo changes take effect without an image rebuild.
- For dev iteration: bind-mount manually:
  ```bash
  docker run -it --name groot-dev -v $(pwd)/Training/scripts:/scripts \
    --gpus all --ipc=host --shm-size=16g \
    lucam03/projekt-humanoider-roboter:latest bash
  ```

## VRAM constraints

| VRAM | `GLOBAL_BATCH_SIZE` | `MAX_STEPS` | Platform |
|---|---|---|---|
| 24 GB | 1–2 | 30 000 | Local (RTX 4090, min) — very slow |
| 32 GB | 4–8 | 30 000 | Local (RTX 5090, ~31 GB at bs=8) |
| 80 GB | 64–128 | 50 000+ | KISSKI A100 |
| 94 GB | 128+ | 50 000+ | KISSKI H100 |

Full fine-tuning requires ≥ 40 GB VRAM per NVIDIA's official recommendation.

Reduce batch size first on OOM. See [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md) for full parameter reference.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
