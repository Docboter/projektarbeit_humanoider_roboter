# Setup and Testing

Last reviewed against local implementation: 2026-08-12.

This document describes the normal two-service Docker Compose setup. For the
Vast.AI single-container variant, see [Vast Single Container](VAST_SINGLE_CONTAINER.md).
For architecture, constraints and the current action contract, see
[Project Context](PROJECT_CONTEXT.md).

## Services

Docker services:

- `groot-server`: GR00T PolicyServer, usually GPU 0.
- `unitree-sim`: Isaac Sim / Unitree Sim / `GrootActionProvider`, usually GPU 1.

Both services use `network_mode: host`. The sim reaches the GR00T server at
`127.0.0.1:5555` by default.

The Compose implementation is in:

```text
docker/docker-compose.groot-unitree.yml
```

## Prerequisites

On the workstation:

```bash
nvidia-smi
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
git --version
git lfs version
```

If the CUDA container does not see a GPU, fix NVIDIA Container Toolkit before
debugging this project.

## Clone and Checkpoint

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
```

If cloned without submodules:

```bash
git submodule update --init --recursive
```

Expected checkpoint path:

```text
repos/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

The checkpoint is loaded by the GR00T PolicyServer, not directly by Isaac Sim.

## Configure

Create the local env file:

```bash
cp docker/.env.groot-unitree.example docker/.env.groot-unitree
```

Important defaults:

```bash
PROJECTS_DIR=..
GROOT_GPU=0
SIM_GPU=1
GROOT_PORT=5555
GROOT_HOST=127.0.0.1
UNITREE_EXTRA_ARGS=--groot_debug --groot_dry_run
```

`PROJECTS_DIR=..` is correct when running Compose from the project root with
`-f docker/docker-compose.groot-unitree.yml`, because relative paths in Compose
are resolved from the `docker/` directory. If Compose is launched from a
different context, set `PROJECTS_DIR` to the absolute project root.

## Local Checks

Layout check:

```bash
./scripts/verify_layout.sh .
```

Lightweight module checks:

```bash
./scripts/run_module_checks.sh .
```

Expected final line:

```text
Module checks passed.
```

The module check verifies project layout, shell syntax, selected Python syntax
inside the submodules and Docker Compose rendering when Docker is available. It
does not start GR00T or Isaac Sim.

## Build Images

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml build groot-server
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml build unitree-sim
```

The first build is expected to be slow because GR00T and Isaac Sim have large
dependencies.

## Test Sequence

Start GR00T:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml up -d groot-server
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml logs -f groot-server
```

Expected log:

```text
Server is ready and listening
```

Query GR00T from the Unitree side:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml run --rm unitree-sim \
  conda run -n unitree_sim_env python3 tools/test_groot_server_action.py \
    --host 127.0.0.1 \
    --port 5555 \
    --timeout-ms 120000 \
    --prompt "pick up the cylinder"
```

Expected output:

```text
GR00T server ping ok.
Action keys: [...]
```

Start Isaac/Unitree in dry-run mode:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml up unitree-sim
```

Good dry-run logs:

```text
[GrootActionProvider] connected to GR00T server 127.0.0.1:5555
[GrootActionProvider][debug] GR00T action keys: [...]
[GrootActionProvider][debug] dry-run enabled
```

Send prompts:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml exec unitree-sim \
  conda run -n unitree_sim_env python3 groot_prompt_terminal.py --fifo groot_prompt.pipe
```

Fallback:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml exec unitree-sim \
  bash -lc 'echo "pick up the cylinder" > /home/code/unitree_sim_isaaclab/groot_prompt.txt'
```

Only after dry-run logs look sane, remove `--groot_dry_run`:

```bash
UNITREE_EXTRA_ARGS=--groot_debug
```

Then restart `unitree-sim`.

## Useful Commands

Logs:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml logs -f groot-server
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml logs -f unitree-sim
```

Stop:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml down
```

Container shells:

```bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml exec unitree-sim bash
docker compose --env-file docker/.env.groot-unitree -f docker/docker-compose.groot-unitree.yml exec groot-server bash
```

VRAM:

```bash
watch -n 0.5 nvidia-smi
```

## Troubleshooting

- `flash_attn ... CPU backend`: GR00T started on CPU. The current checkpoint
  expects CUDA/FlashAttention.
- `CUDA out of memory`: use separate GPUs or a larger GPU.
- `Cannot connect to GR00T PolicyServer`: start `groot-server` first and wait
  for `Server is ready and listening`.
- `No reader on FIFO`: usually not critical; prompt fallback writes to
  `groot_prompt.txt`.
- Robot jumps or tips over: return to `--groot_dry_run` and inspect action keys
  and applied target deltas.
- Docker sees no GPU: fix NVIDIA Container Toolkit first.
- `set: pipefail\r: invalid option name`: normalize shell script line endings
  to LF before running the Bash checks on Windows.
