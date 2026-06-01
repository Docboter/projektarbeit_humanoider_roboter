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
- GR00T editable install
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

## End-to-end workflow

The intended flow is:

1. Build the image locally.
2. Push it to Docker Hub.
3. Start that Docker Hub image as the Vast.AI instance image.
4. Put the GR00T checkpoint on the Vast machine.
5. Start GR00T first, then Isaac/Unitree.

Build from the project root:

```bash
docker build -f docker/1-docker-setup/Dockerfile -t groot-unitreesim-single-docker-setup:latest .
```

Push to Docker Hub:

```bash
docker login
docker tag groot-unitreesim-single-docker-setup:latest <dockerhub-user>/groot-unitreesim-single-docker-setup:latest
docker push <dockerhub-user>/groot-unitreesim-single-docker-setup:latest
```

Use this image name on Vast.AI:

```text
<dockerhub-user>/groot-unitreesim-single-docker-setup:latest
```

## Build notes

The Dockerfile-specific ignore file excludes checkpoints by default:

```text
docker/1-docker-setup/Dockerfile.dockerignore
```

Mount or copy checkpoints at runtime instead of baking them into the image.

## Checkpoint

The default checkpoint path inside the container is:

```text
/workspace/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

That default comes from `start_groot_server.sh`:

```text
GROOT_MODEL_PATH=checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

The checkpoint is not present in this repository. Use NVIDIA's Hugging Face
checkpoint `nvidia/GR00T-N1.6-G1-PnPAppleToPlate`:

```bash
mkdir -p /workspace/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
/workspace/Isaac-GR00T/.venv/bin/huggingface-cli download \
  nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
  --local-dir /workspace/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

You can also point GR00T directly at Hugging Face:

```bash
GROOT_MODEL_PATH=nvidia/GR00T-N1.6-G1-PnPAppleToPlate start_groot_server.sh
```

For quick Vast.AI tests, downloading the checkpoint to the local container path
is usually more robust than relying on model download during server startup.

Sources:

- GR00T checkpoint: <https://huggingface.co/nvidia/GR00T-N1.6-G1-PnPAppleToPlate>
- Isaac Sim WebRTC ports/client: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/manual_livestream_clients.html>

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

## WebRTC / watching the simulation

The primary viewing path is the Isaac Sim WebRTC Streaming Client on your local
machine. For Vast.AI, make sure these ports are exposed/mapped:

```text
49100/tcp  WebRTC signaling
47998/udp  WebRTC media stream
8210/tcp   optional web viewer port, not a dedicated service in this image
```

Set `PUBLIC_IP` before starting the sim:

```bash
export PUBLIC_IP=<vast-public-ip-or-hostname>
```

Avoid A100 if you need live video: Isaac Sim livestreaming needs NVENC, and A100
does not provide it. Prefer GPUs such as L40S 48 GB, RTX 6000 Ada 48 GB, or
consumer/pro RTX GPUs with enough VRAM.

If WebRTC does not connect, first debug Vast port mapping, NAT, and firewall
rules. Useful fallbacks are SSH tunneling, Tailscale/VPN, adding a dedicated web
viewer later, or debugging through logs until the stream path is fixed.

## Data flow

At runtime there are two long-running processes inside the same container:

- `start_groot_server.sh` starts `gr00t/eval/run_gr00t_server.py` as a ZeroMQ
  PolicyServer on port `5555`.
- `start_unitree_sim.sh` starts `sim_main.py` with `--action_source groot`,
  camera resolution, prompt file/FIFO, and the GR00T host/port.

The sim creates `GrootActionProvider` through `create_action_provider()`. The
provider reads prompts from `groot_prompt.pipe` or `groot_prompt.txt`; if neither
has a prompt, it uses `pick up the cylinder`.

For each provider step, `GrootActionProvider` reads `front_camera` as RGB and
the robot joint states from Isaac. It sends this observation to GR00T:

```text
video.ego_view
state.left_leg
state.right_leg
state.waist
state.left_arm
state.right_arm
state.left_hand
state.right_hand
language.annotation.human.task_description
```

Observations and actions move over ZeroMQ with msgpack plus NumPy array
serialization. Returned GR00T actions are currently applied to arms, hands, and
waist. `base_height_command` and `navigate_command` may be received, but are
only logged and not mapped to Isaac joint targets yet.

Keep this for the first real test:

```bash
export UNITREE_EXTRA_ARGS="--groot_debug --groot_dry_run"
```

This asks GR00T for actions and prints diagnostics, but does not apply the
actions to the simulated robot.

## Test sequence on Vast.AI

Use `tmux` or two SSH terminals.

1. Check the GPU:

```bash
nvidia-smi
```

2. Download or mount the checkpoint at:

```text
/workspace/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

3. Start GR00T in terminal 1:

```bash
CUDA_VISIBLE_DEVICES=0 start_groot_server.sh
```

Wait for:

```text
Server is ready and listening
```

4. Dry-test the GR00T server from terminal 2:

```bash
cd /home/code/unitree_sim_isaaclab
conda run --no-capture-output -n unitree_sim_env \
  python3 tools/test_groot_server_action.py \
  --host 127.0.0.1 \
  --port 5555 \
  --timeout-ms 120000 \
  --prompt "pick up the cylinder"
```

5. Start Isaac/Unitree in dry-run mode from terminal 2:

```bash
export PUBLIC_IP=<vast-public-ip-or-hostname>
export UNITREE_EXTRA_ARGS="--groot_debug --groot_dry_run"
CUDA_VISIBLE_DEVICES=0 start_unitree_sim.sh
```

For two GPUs, use GPU 0 for GR00T and GPU 1 for Isaac:

```bash
CUDA_VISIBLE_DEVICES=0 start_groot_server.sh
CUDA_VISIBLE_DEVICES=1 start_unitree_sim.sh
```

6. Send a prompt after the sim is running:

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

Only remove `--groot_dry_run` after the action keys, camera shape, and state
debug logs look plausible.

## Recorded visual test workflow

Use this path when WebRTC is blocked by Vast.AI port mapping. The sim records
`front_camera` frames in the container, converts them to MP4 with `ffmpeg`, and
you download the MP4 to your local machine.

This requires an image built after the recording flags were added to
`sim_main.py`.

Fast local automation for an existing Vast container:

```bash
VAST_HOST=45.81.32.13 VAST_PORT=22924 scripts/vast_record_groot_test.sh
```

Useful overrides:

```bash
PROMPT="pick up the cylinder" \
RECORD_FRAMES=1200 \
RECORD_EVERY=5 \
RECORD_FPS=20 \
LOCAL_OUT=vast-recordings/groot-dryrun \
VAST_HOST=45.81.32.13 \
VAST_PORT=22924 \
scripts/vast_record_groot_test.sh
```

The script copies the current `sim_main.py` and
`tools/test_groot_server_action.py` to Vast, starts GR00T in `tmux`, starts the
recording sim, sends the prompt, waits for `front_camera.mp4`, and downloads the
MP4 plus log to `LOCAL_OUT`.

1. Build and push the updated image from the project root:

```bash
docker build -f docker/1-docker-setup/Dockerfile -t groot-unitreesim-single-docker-setup:latest .
docker tag groot-unitreesim-single-docker-setup:latest <dockerhub-user>/groot-unitreesim-single-docker-setup:latest
docker push <dockerhub-user>/groot-unitreesim-single-docker-setup:latest
```

2. Start a new Vast.AI instance from the pushed image and connect over SSH.

3. Confirm the checkpoint exists:

```bash
ls -lah /workspace/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate
```

4. In `tmux` window 1, start GR00T:

```bash
CUDA_VISIBLE_DEVICES=0 start_groot_server.sh
```

Wait for:

```text
Server is ready and listening
```

5. In `tmux` window 2, check that GR00T returns actions:

```bash
cd /home/code/unitree_sim_isaaclab
conda run --no-capture-output -n unitree_sim_env \
  python3 tools/test_groot_server_action.py \
  --host 127.0.0.1 \
  --port 5555 \
  --timeout-ms 120000 \
  --prompt "pick up the cylinder"
```

6. In `tmux` window 2, start the sim with GR00T dry-run and recording:

```bash
mkdir -p /workspace/recordings
rm -rf /workspace/recordings/groot-dryrun
export UNITREE_EXTRA_ARGS="--groot_debug --groot_dry_run --record_camera front_camera --record_dir /workspace/recordings/groot-dryrun --record_every 5 --record_max_frames 600 --record_fps 20"
CUDA_VISIBLE_DEVICES=0 start_unitree_sim.sh 2>&1 | tee /workspace/recordings/groot-dryrun.log
```

The sim stops automatically after `600` recorded frames. Expected recorder logs:

```text
[recorder] wrote 1 frames
[recorder] wrote 50 frames
[recorder] reached max frames, stopping simulation
[recorder] mp4: /workspace/recordings/groot-dryrun/front_camera.mp4
```

7. Send a prompt while the sim is running:

```bash
cd /home/code/unitree_sim_isaaclab
echo "pick up the cylinder" > groot_prompt.txt
```

8. Confirm outputs on Vast:

```bash
ls -lah /workspace/recordings/groot-dryrun
ls -lah /workspace/recordings/groot-dryrun/frames | head
grep -Ei 'GrootActionProvider|GR00T action keys|rgb_shape|dry-run|recorder|failed|error' /workspace/recordings/groot-dryrun.log | tail -120
```

9. Download the visual output from your local machine:

```bash
scp -P <vast-ssh-port> root@<vast-host>:/workspace/recordings/groot-dryrun/front_camera.mp4 .
scp -P <vast-ssh-port> root@<vast-host>:/workspace/recordings/groot-dryrun.log .
```

Example with a direct Vast SSH endpoint:

```bash
scp -P 31999 root@85.218.235.6:/workspace/recordings/groot-dryrun/front_camera.mp4 .
scp -P 31999 root@85.218.235.6:/workspace/recordings/groot-dryrun.log .
```

10. Open `front_camera.mp4` locally. If no MP4 was created, download the frames:

```bash
scp -P <vast-ssh-port> -r root@<vast-host>:/workspace/recordings/groot-dryrun/frames .
```
