# Remote Unitree Simulation

Last reviewed against the local implementation: 2026-08-12.

This is the first project milestone: run Isaac Sim reliably on a remote IKR
server and view a stationary Unitree G1/G129 with DEX3 hands in front of a
table with three colored cubes. GR00T is intentionally not part of this start
path.

## Resulting Runtime

The existing Unitree task is reused:

```text
Isaac-Stack-RgyBlock-G129-Dex3-Joint
```

It contains:

- a base-fixed Unitree G1/G129 with DEX3 hands;
- a table;
- red, green and yellow cubes;
- scene and robot cameras.

The `idle` action source holds the task's default joint pose. It does not start
GR00T, the Unitree DDS command channels or the additional TeleImager server.
This avoids accidental interaction with Unitree DDS participants on the IKR
network while the simulation baseline is being validated. The Compose default
extends the original task's 20-second episode to one hour, avoiding periodic
time-limit resets during the stability observation.

## Server Prerequisites

The server must run Linux and provide an NVIDIA RTX-capable GPU, a compatible
NVIDIA driver, Docker Engine, Docker Compose and NVIDIA Container Toolkit.

Check without modifying Docker state:

```bash
nvidia-smi
docker version
docker compose version
docker info --format '{{json .Runtimes}}'
docker ps
```

Do not use `docker system prune`, `docker image prune`, `docker volume prune`
or broad cleanup commands on a shared server.

## One-Time Setup

Clone the repository with submodules and download the Unitree assets:

```bash
git clone --recurse-submodules <repository-url>
cd projektarbeit_humanoider_roboter
./scripts/fetch_unitree_assets.sh .
cp docker/.env.simulation.example docker/.env.simulation
```

The asset helper downloads into a temporary directory and moves the completed
asset tree into place. It does not delete an existing asset directory.

Set `PUBLIC_IP` in `docker/.env.simulation` to the server address reachable
from the laptop. On an IKR LAN or institute VPN, keep `LIVESTREAM_TYPE=2`.

## Build and Start

Build only this project's image:

```bash
docker compose \
  --env-file docker/.env.simulation \
  -f docker/docker-compose.simulation.yml \
  build unitree-simulation
```

Start it in the background:

```bash
docker compose \
  --env-file docker/.env.simulation \
  -f docker/docker-compose.simulation.yml \
  up -d unitree-simulation
```

Follow startup and shader-cache progress:

```bash
docker compose \
  --env-file docker/.env.simulation \
  -f docker/docker-compose.simulation.yml \
  logs -f unitree-simulation
```

Expected project-specific log lines include:

```text
Starting stable Unitree scene: Isaac-Stack-RgyBlock-G129-Dex3-Joint
create environment success
idle mode: external image server and DDS disabled
```

The first launch can take several minutes because Isaac Sim builds shader
caches. Named volumes preserve caches across container recreation.

Stop only this Compose project:

```bash
docker compose \
  --env-file docker/.env.simulation \
  -f docker/docker-compose.simulation.yml \
  down
```

This command does not include `--volumes`, so the project caches remain.

## Live View Recommendation

Use the native NVIDIA Isaac Sim WebRTC Streaming Client on the desk laptop.
It has the fewest moving parts for Isaac Sim 5.1 and avoids X11, VNC and a
virtual desktop on the server.

NVIDIA references:

- [Isaac Sim container installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
- [Isaac Sim livestream clients](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)

The server and firewall must allow:

| Port | Protocol | Purpose |
|---|---|---|
| `49100` | TCP | WebRTC signaling |
| `47998` | UDP | WebRTC media |

The container uses `network_mode: host`, which NVIDIA requires for WebRTC.
Connect the client to the configured `PUBLIC_IP` after the simulator has fully
loaded. Only one streaming client should connect at a time.

Recommended network order:

1. IKR LAN or institute VPN plus native WebRTC client.
2. A private WireGuard/Tailscale-style overlay, if permitted by IKR policy.
3. NVIDIA's browser WebRTC viewer as a later convenience layer.

Do not expose the stream ports directly to the public internet without access
control and encryption. An SSH TCP tunnel alone is not sufficient for the UDP
media path. No port publishing is configured because Docker bridge networking
does not provide the WebRTC host address inside the container.

## Verification Stages

Perform these in order:

1. `docker compose ... config` renders successfully.
2. The image builds without changing other Docker projects.
3. The Unitree asset check passes.
4. Isaac Sim reports successful environment creation.
5. The laptop receives a WebRTC image.
6. The G1 remains in its default pose in front of the table and cubes for at
   least ten minutes without simulation errors.

This milestone does not prove GR00T inference, action mapping, DDS control or
robot motion. Those remain separate follow-up stages.

## Troubleshooting

- No signaling connection: check `PUBLIC_IP`, TCP `49100`, host firewall and
  whether the server is reachable over LAN/VPN.
- Signaling works but no video: check UDP `47998`, host networking, GPU video
  capability and the NVIDIA driver.
- Black image during first launch: wait for shader compilation and inspect the
  logs before restarting.
- Missing Unitree asset: rerun `./scripts/fetch_unitree_assets.sh .`; it will
  leave an existing asset directory untouched.
- GPU initialization failure: validate NVIDIA Container Toolkit with a small
  CUDA `nvidia-smi` container before rebuilding this image.
