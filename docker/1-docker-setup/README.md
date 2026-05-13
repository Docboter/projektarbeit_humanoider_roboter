# 1 Docker Setup: GR00T + Isaac/Unitree in one image

This setup is for Vast.AI-style machines where the rented instance is already a
single Docker container. It does not replace the two-service Docker Compose
setup. It is a pragmatic test image that runs two processes inside one
container:

- `start_groot_server.sh`: GR00T ZeroMQ PolicyServer
- `start_unitree_sim.sh`: Isaac Sim / Unitree Sim / `GrootActionProvider`

## Why one image here?

The existing project architecture uses two services:

- `groot-server` from `repos/Isaac-GR00T/docker/Dockerfile`
- `unitree-sim` from `repos/unitree_sim_isaaclab/Dockerfile`

That is clean on a normal Docker host with Docker Compose. On Vast.AI, however,
the instance itself is already a container. Running Docker Compose inside that
container adds Docker-in-Docker, GPU passthrough and port-forwarding complexity.

This image keeps the logical service split, but puts both runtimes into one
container.

## What comes from GR00T?

Source file:

```text
repos/Isaac-GR00T/docker/Dockerfile
```

Used for:

- CUDA 12.8 devel base image
- Python 3.10 system packages
- `uv`
- GR00T `.venv` at `/workspace/Isaac-GR00T/.venv`
- GR00T and `groot_infra` editable installs
- EGL/PyOpenGL/MuJoCo defaults
- optional aarch64 `torchcodec` source build

Runtime entrypoint:

```bash
start_groot_server.sh
```

Default server config:

```text
GROOT_MODEL_PATH=checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
GROOT_EMBODIMENT_TAG=UNITREE_G1
GROOT_HOST_BIND=0.0.0.0
GROOT_PORT=5555
GROOT_CUDA_DEVICE=0
```

## What comes from Isaac/Unitree?

Source file:

```text
repos/unitree_sim_isaaclab/Dockerfile
```

Used for:

- Miniconda at `/opt/conda`
- `unitree_sim_env` with Python 3.11
- PyTorch 2.7.0 CUDA 12.6 wheels
- Isaac Sim 5.1.0 pip package
- IsaacLab install at `/home/code/IsaacLab`
- CycloneDDS at `/cyclonedds/install`
- `unitree_sdk2_python`
- local `repos/unitree_sim_isaaclab` copied to `/home/code/unitree_sim_isaaclab`

Runtime entrypoint:

```bash
start_unitree_sim.sh
```

Default simulation config:

```text
UNITREE_TASK=Isaac-Move-Cylinder-G129-Dex3-Wholebody
ROBOT_TYPE=g129
DEX_HAND_ARG=--enable_dex3_dds
GROOT_HOST=127.0.0.1
GROOT_PORT=5555
CAMERA_WIDTH=320
CAMERA_HEIGHT=240
UNITREE_EXTRA_ARGS=--groot_debug --groot_dry_run
```

## Build

Run this from the project root:

```bash
docker build -f docker/1-docker-setup/Dockerfile -t groot-unitreesim-single-docker-setup:latest .
```

The Dockerfile-specific ignore file excludes checkpoints by default:

```text
docker/1-docker-setup/Dockerfile.dockerignore
```

Mount checkpoints at runtime instead of baking them into the image:

```text
/workspace/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

## Run locally on a Docker host

Example for one GPU:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display,video \
  -e PUBLIC_IP=127.0.0.1 \
  -e UNITREE_EXTRA_ARGS="--groot_debug --groot_dry_run" \
  -v "$(pwd)/repos/Isaac-GR00T/checkpoints:/workspace/Isaac-GR00T/checkpoints:ro" \
  groot-unitreesim-single-docker-setup:latest
```

Inside the container, use two terminals or two `tmux` panes:

```bash
start_groot_server.sh
```

```bash
start_unitree_sim.sh
```

## Run on two GPUs

In pane 1:

```bash
CUDA_VISIBLE_DEVICES=0 start_groot_server.sh
```

In pane 2:

```bash
CUDA_VISIBLE_DEVICES=1 start_unitree_sim.sh
```

Inside each process the selected GPU appears as `cuda:0`. Keep
`GROOT_CUDA_DEVICE=0` unless you deliberately expose multiple GPUs to the GR00T
process.

## Run on one large GPU

Use the same visible GPU for both processes:

```bash
CUDA_VISIBLE_DEVICES=0 start_groot_server.sh
CUDA_VISIBLE_DEVICES=0 start_unitree_sim.sh
```

This is most realistic with 48 GB VRAM or more. With 24 GB VRAM, expect possible
CUDA out-of-memory errors during model load, Isaac Sim startup, or first camera
frames.

## Vast.AI notes

Expose at least the GR00T and Isaac WebRTC ports if you need remote access:

```text
-p 5555:5555 -p 49100:49100 -p 47998:47998/udp -p 8210:8210
```

Set `PUBLIC_IP` to the public IP or hostname that the Isaac WebRTC client should
connect to:

```bash
export PUBLIC_IP=<vast-public-ip>
```

Start GR00T first. Wait until the log says the server is ready, then start
Isaac/Unitree.

## Prompt test

After `unitree-sim` is running:

```bash
cd /home/code/unitree_sim_isaaclab
echo "pick up the cylinder" > groot_prompt.txt
```

Keep `UNITREE_EXTRA_ARGS="--groot_debug --groot_dry_run"` until GR00T action
keys are visible in the Isaac logs.
