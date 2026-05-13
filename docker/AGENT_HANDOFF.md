# Agent Handoff: Single Docker Setup for Vast.AI

## User goal

The user wants to test the GR00T + Unitree Isaac Sim integration on Vast.AI.
Vast.AI gives the user one container as the rented instance, so the current
goal is to run both logical services inside one Docker image:

- GR00T PolicyServer
- Isaac Sim / Unitree Sim with `GrootActionProvider`

The user wants to view Isaac Sim remotely from their local Windows machine via
Isaac Sim WebRTC streaming.

## Current implementation

A new single-container setup was added here:

```text
docker/1-docker-setup/
```

Important files:

```text
docker/1-docker-setup/Dockerfile
docker/1-docker-setup/README.md
docker/1-docker-setup/.env.single-container.example
docker/1-docker-setup/Dockerfile.dockerignore
docker/1-docker-setup/scripts/start_groot_server.sh
docker/1-docker-setup/scripts/start_unitree_sim.sh
```

The root `README.md` now links to `docker/1-docker-setup/README.md`.

Image name chosen by the user:

```text
groot-unitreesim-single-docker-setup
```

## Design

The image intentionally keeps two separate Python environments:

- GR00T:
  - source: `repos/Isaac-GR00T/docker/Dockerfile`
  - path in image: `/workspace/Isaac-GR00T`
  - venv: `/workspace/Isaac-GR00T/.venv`
  - Python 3.10, `uv`

- Isaac/Unitree:
  - source: `repos/unitree_sim_isaaclab/Dockerfile`
  - path in image: `/home/code/unitree_sim_isaaclab`
  - env: `/opt/conda/envs/unitree_sim_env`
  - Python 3.11, Conda, Isaac Sim 5.1.0, IsaacLab

This avoids mixing GR00T dependencies directly into the Isaac Sim Conda env.

Runtime model:

```text
one Vast container
|-- start_groot_server.sh
`-- start_unitree_sim.sh
```

Do not try Docker Compose inside the Vast container unless the user explicitly
wants Docker-in-Docker experiments.

## Build and push target

The user is on Windows. Docker works in normal PowerShell but did not initially
work inside the VS Code PowerShell terminal. The likely fix is restarting VS
Code after Docker Desktop installation or adding Docker Desktop's bin directory
to the VS Code terminal PATH:

```text
C:\Program Files\Docker\Docker\resources\bin
```

Build from project root:

```bash
docker build -f docker/1-docker-setup/Dockerfile -t groot-unitreesim-single-docker-setup:latest .
```

Push to Docker Hub:

```bash
docker login
docker tag groot-unitreesim-single-docker-setup:latest <dockerhub-user>/groot-unitreesim-single-docker-setup:latest
docker push <dockerhub-user>/groot-unitreesim-single-docker-setup:latest
```

Vast image:

```text
<dockerhub-user>/groot-unitreesim-single-docker-setup:latest
```

## Checkpoints and assets

The Dockerfile-specific ignore file excludes GR00T checkpoints by default.
The checkpoint must be mounted or copied on Vast to:

```text
/workspace/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

If Unitree/Isaac assets are missing after launch, run inside the container:

```bash
cd /home/code/unitree_sim_isaaclab
. fetch_assets.sh
```

## Vast.AI deployment notes

Suggested exposed ports:

```text
-p 5555:5555 -p 49100:49100 -p 47998:47998/udp -p 8210:8210
```

GR00T port:

```text
5555/tcp
```

Isaac Sim WebRTC ports commonly needed:

```text
49100/tcp
47998/udp
8210/tcp optional web viewer
```

Vast may map ports to random external ports. If WebRTC does not connect, first
suspect Vast NAT/port mapping before changing Isaac/GR00T code.

Avoid A100 if the user wants livestreaming, because NVIDIA notes that A100 does
not support NVENC for Isaac Sim livestreaming. Prefer L40S 48 GB, RTX 6000 Ada
48 GB, or two consumer/pro GPUs with enough VRAM.

## Test sequence on Vast

Use `tmux` or two SSH terminals.

1. Check GPU:

```bash
nvidia-smi
```

2. Start GR00T:

```bash
CUDA_VISIBLE_DEVICES=0 start_groot_server.sh
```

Wait until the server reports readiness.

3. Test GR00T from the Unitree side:

```bash
cd /home/code/unitree_sim_isaaclab

conda run --no-capture-output -n unitree_sim_env \
  python3 tools/test_groot_server_action.py \
  --host 127.0.0.1 \
  --port 5555 \
  --timeout-ms 120000 \
  --prompt "pick up the cylinder"
```

4. Start Isaac/Unitree in dry-run mode:

```bash
export PUBLIC_IP=<vast-public-ip-or-hostname>
export UNITREE_EXTRA_ARGS="--groot_debug --groot_dry_run"

CUDA_VISIBLE_DEVICES=0 start_unitree_sim.sh
```

For a two-GPU instance, use GPU 0 for GR00T and GPU 1 for Isaac:

```bash
CUDA_VISIBLE_DEVICES=0 start_groot_server.sh
CUDA_VISIBLE_DEVICES=1 start_unitree_sim.sh
```

5. Send a prompt:

```bash
cd /home/code/unitree_sim_isaaclab
echo "pick up the cylinder" > groot_prompt.txt
```

Expected Isaac logs:

```text
[GrootActionProvider] connected to GR00T server 127.0.0.1:5555
[GrootActionProvider][debug] GR00T action keys: [...]
[GrootActionProvider][debug] dry-run enabled
```

Only remove `--groot_dry_run` after the dry-run logs look sane.

## Windows viewer note

The user asked whether the Isaac Sim WebClient exists for Windows and whether a
browser is enough.

Current recommendation:

- Use the official Isaac Sim WebRTC Streaming Client for Windows first.
- Browser viewing may be possible with NVIDIA's web viewer flow, but this image
  does not yet include a dedicated web viewer service.
- If the Windows client fails to connect on Vast, investigate port mapping/NAT
  before changing the sim code.

## Local verification limits so far

This workspace is Windows and the Codex sandbox did not have Docker, WSL, or
Bash available. Therefore the image has not yet been built locally and the shell
scripts have not been run with `bash -n`.

Files were checked manually for paths, line endings, and consistency. Shell
scripts have LF line endings.

## Known follow-up tasks

- Run the Docker build for the first time.
- Fix any build errors caused by dependency version conflicts between CUDA
  12.8, Isaac Sim 5.1.0, PyTorch 2.7.0, and GR00T.
- Confirm whether Dockerfile-specific ignore file
  `docker/1-docker-setup/Dockerfile.dockerignore` is honored by the user's
  Docker version.
- Verify whether Isaac WebRTC works through Vast port mapping.
- If WebRTC is blocked by Vast NAT, consider adding a browser web viewer,
  Tailscale/VPN approach, or explicit livestream port configuration.
