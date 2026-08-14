# Vast Single Container

Last reviewed against local implementation: 2026-08-14.

This document covers the experimental Vast.AI setup under
`docker/1-docker-setup/`.
For shared architecture, action mapping and validated constraints, see
[Project Context](PROJECT_CONTEXT.md).

## Goal

Vast.AI usually gives the user one rented container. The normal two-service
Docker Compose setup would require Docker-in-Docker, extra GPU passthrough and
more port-forwarding complexity. This image keeps the logical service split but
runs both processes inside one container:

- `start_groot_server.sh`: GR00T ZeroMQ PolicyServer.
- `start_unitree_sim.sh`: Isaac Sim / Unitree Sim with `GrootActionProvider`.

## Important Files

```text
docker/1-docker-setup/Dockerfile
docker/1-docker-setup/.env.single-container.example
docker/1-docker-setup/Dockerfile.dockerignore
docker/1-docker-setup/scripts/start_groot_server.sh
docker/1-docker-setup/scripts/start_unitree_sim.sh
```

Image name used so far:

```text
groot-unitreesim-single-docker-setup
```

## Runtime Design

The image intentionally keeps two separate Python environments.

GR00T:

- Source pattern: `repos/Isaac-GR00T/docker/Dockerfile`
- Path in image: `/workspace/Isaac-GR00T`
- Virtual env: `/workspace/Isaac-GR00T/.venv`
- Python: 3.10
- Package manager: `uv`

Isaac/Unitree:

- Source pattern: `repos/unitree_sim_isaaclab/Dockerfile`
- Path in image: `/home/code/unitree_sim_isaaclab`
- Conda env: `/opt/conda/envs/unitree_sim_env`
- Python: 3.11
- Isaac Sim: 5.1.0

This avoids mixing GR00T dependencies directly into the Isaac Sim Conda
environment.

## Build

Run from the project root:

```bash
docker build -f docker/1-docker-setup/Dockerfile -t groot-unitreesim-single-docker-setup:latest .
```

The Dockerfile-specific ignore file excludes checkpoints by default. Mount or
copy the checkpoint at runtime instead:

```text
/workspace/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

Open follow-up: verify whether `docker/1-docker-setup/Dockerfile.dockerignore`
is honored by the installed Docker version.

## Push

```bash
docker login
docker tag groot-unitreesim-single-docker-setup:latest <dockerhub-user>/groot-unitreesim-single-docker-setup:latest
docker push <dockerhub-user>/groot-unitreesim-single-docker-setup:latest
```

Vast image:

```text
<dockerhub-user>/groot-unitreesim-single-docker-setup:latest
```

## Run Locally

Example for one large GPU:

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

## Runtime Configuration

GR00T defaults from `start_groot_server.sh`:

```text
GROOT_MODEL_PATH=checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
GROOT_EMBODIMENT_TAG=UNITREE_G1
GROOT_HOST_BIND=0.0.0.0
GROOT_PORT=5555
GROOT_CUDA_DEVICE=0
```

Isaac/Unitree defaults from `start_unitree_sim.sh`:

```text
UNITREE_TASK=Isaac-Move-Cylinder-G129-Dex3-Wholebody
ROBOT_TYPE=g129
DEX_HAND_ARG=--enable_dex3_dds
GROOT_HOST=127.0.0.1
GROOT_PORT=5555
GROOT_TIMEOUT_MS=120000
GROOT_ACTION_STEP=0
CAMERA_WIDTH=320
CAMERA_HEIGHT=240
LIVESTREAM_TYPE=1
PUBLIC_IP=127.0.0.1
UNITREE_EXTRA_ARGS=--groot_debug --groot_dry_run
```

The sim launcher currently passes:

- `--headless`
- `--no_render`
- `--livestream_type`
- `--public_ip`
- `--device cuda`
- `--enable_cameras`
- `--action_source groot`
- `--enable_wholebody_dds`
- `--minimal_scene`
- prompt FIFO and prompt file paths

## GPU Selection

For two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 start_groot_server.sh
CUDA_VISIBLE_DEVICES=1 start_unitree_sim.sh
```

Inside each process the selected GPU appears as `cuda:0`. Keep
`GROOT_CUDA_DEVICE=0` unless multiple GPUs are intentionally visible to the
GR00T process.

For one large GPU:

```bash
CUDA_VISIBLE_DEVICES=0 start_groot_server.sh
CUDA_VISIBLE_DEVICES=0 start_unitree_sim.sh
```

This is most realistic with 48 GB VRAM or more. With 24 GB VRAM, expect possible
CUDA out-of-memory errors during model load, Isaac Sim startup or first camera
frames.

## Vast.AI Ports and WebRTC

Expose at least:

```text
-p 5555:5555 -p 49100:49100 -p 47998:47998/udp -p 8210:8210
```

GR00T:

```text
5555/tcp
```

Isaac WebRTC commonly needs:

```text
49100/tcp
47998/udp
8210/tcp optional web viewer
```

Set:

```bash
export PUBLIC_IP=<vast-public-ip-or-hostname>
```

Vast may map ports to random external ports. If WebRTC does not connect, first
suspect Vast NAT or port mapping before changing Isaac/GR00T code.

For Windows viewing, try the official Isaac Sim WebRTC Streaming Client first.
Browser viewing may be possible with NVIDIA's web viewer flow, but this image
does not currently include a dedicated browser web viewer service.

Avoid A100 when livestreaming is required because A100 does not provide NVENC
for Isaac Sim livestreaming. Prefer L40S 48 GB, RTX 6000 Ada 48 GB or a
two-GPU setup with enough VRAM.

## Vast Test Sequence

Check GPU:

```bash
nvidia-smi
```

Start GR00T:

```bash
CUDA_VISIBLE_DEVICES=0 start_groot_server.sh
```

Wait for readiness.

Test GR00T from the Unitree side:

```bash
cd /home/code/unitree_sim_isaaclab
conda run --no-capture-output -n unitree_sim_env \
  python3 tools/test_groot_server_action.py \
    --host 127.0.0.1 \
    --port 5555 \
    --timeout-ms 120000 \
    --prompt "pick up the cylinder"
```

Start Isaac/Unitree in dry-run mode:

```bash
export PUBLIC_IP=<vast-public-ip-or-hostname>
export UNITREE_EXTRA_ARGS="--groot_debug --groot_dry_run"
CUDA_VISIBLE_DEVICES=0 start_unitree_sim.sh
```

Send a prompt:

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

## Recording Fallback

Use camera recording when Vast port mapping or NAT prevents WebRTC viewing. The
Unitree simulation supports recording `front_camera` frames and producing an
MP4 during cleanup. Keep this path in GR00T dry-run mode while validating
inference and images.

The repository includes a helper for an existing Vast instance:

```bash
VAST_HOST=<vast-host> VAST_PORT=<ssh-port> scripts/vast_record_groot_test.sh
```

Optional variables include `PROMPT`, `RECORD_NAME`, `RECORD_FRAMES`,
`RECORD_EVERY`, `RECORD_FPS` and `LOCAL_OUT`.

The helper copies the current Unitree test files to the remote container,
starts GR00T and the dry-run simulation in a `tmux` session named
`groot_test`, waits for `front_camera.mp4`, then downloads the MP4 and log to
`LOCAL_OUT`. It replaces an existing `groot_test` tmux session, so only use it
for a dedicated test session.

For manual recording, pass the `--record_camera`, `--record_dir`,
`--record_every`, `--record_max_frames` and `--record_fps` options through
`UNITREE_EXTRA_ARGS`.

## Local Verification Limits

As of 2026-08-12, the single-container image has not yet been built successfully
in this workspace. Known next work is to run the first real build, then fix the
first concrete build or dependency error.
