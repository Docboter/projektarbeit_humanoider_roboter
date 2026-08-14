# Project Context

Last reviewed against the local implementation: 2026-08-12.

This is the durable technical context for the project. Keep current facts,
architecture, validated constraints, decisions and the technical backlog here.
Setup commands belong in the task-specific guides, not in this document.

## Goal and System Flow

The project integrates NVIDIA GR00T with the Unitree Isaac Sim / IsaacLab
environment so a Unitree G1/G129 with DEX3 hands can be controlled from
language prompts in simulation.

Current example prompt:

```text
pick up the cylinder
```

End-to-end flow:

1. The GR00T PolicyServer loads `GR00T-N1.6-G1-PnPAppleToPlate`.
2. Isaac/Unitree starts `sim_main.py` with `--action_source groot`.
3. `GrootActionProvider` reads `groot_prompt.pipe` or `groot_prompt.txt`.
4. It builds camera, robot-state and language observations.
5. Observations are sent to GR00T over ZeroMQ/MsgPack.
6. Returned actions are inspected in dry-run mode before application.

## Repository Boundaries

The integration repository owns:

- `AGENTS.md`, `README.md` and `docs/`
- `docker/docker-compose.groot-unitree.yml`
- `docker/.env.groot-unitree.example`
- `docker/1-docker-setup/`
- `scripts/`

Implementation-critical submodules:

- `repos/Isaac-GR00T`: GR00T fork with a lightweight PolicyServer patch;
  configured branch `integration/groot-server-fix`.
- `repos/unitree_sim_isaaclab`: Unitree IsaacLab fork with
  `GrootActionProvider`; configured branch
  `integration/groot-action-provider`.

Observed remote:
`https://github.com/Docboter/projektarbeit_humanoider_roboter.git`.

## Runtime Architecture

### Current Simulation Baseline

- The first milestone is a GR00T-free remote simulation on an IKR server.
- Task: `Isaac-Stack-RgyBlock-G129-Dex3-Joint` (G1/G129 DEX3, table and
  red/green/yellow cubes).
- `idle` action mode holds the default pose without starting Unitree DDS or the
  additional TeleImager server.
- Deployment: `docker/docker-compose.simulation.yml` using the official Isaac
  Sim 5.1 base image and host networking for WebRTC.
- Preferred viewer: native Isaac Sim WebRTC Streaming Client over IKR LAN/VPN.

### Normal Workstation

- Docker Compose runs `groot-server` and `unitree-sim` separately.
- Both use `network_mode: host`; Unitree normally reaches GR00T at
  `127.0.0.1:5555`.
- Separate GPUs are preferred: GR00T on GPU 0 and Isaac Sim on GPU 1.

### Vast.AI

- A single image runs two processes because the rented instance is already a
  container.
- `start_groot_server.sh` and `start_unitree_sim.sh` keep separate Python
  environments.
- Remote Isaac WebRTC requires NVENC. A100 is unsuitable when livestreaming is
  required; an L40S, RTX 6000 Ada or suitable two-GPU setup is preferred.

## Action Contract

The current `UNITREE_G1` checkpoint expects:

- video: `ego_view`
- state: `left_leg`, `right_leg`, `waist`, `left_arm`, `right_arm`,
  `left_hand`, `right_hand`
- language: `annotation.human.task_description`
- actions: `left_arm`, `right_arm`, `left_hand`, `right_hand`, `waist`,
  `base_height_command`, `navigate_command`

Currently applied by `GrootActionProvider`:

- `left_arm`, `right_arm`, `waist`, `left_hand`, `right_hand`

Currently logged but not mapped to IsaacLab locomotion:

- `base_height_command`, `navigate_command`

## Validated Facts and Constraints

- Expected checkpoint path:
  `repos/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate`
- The checkpoint uses HuggingFace/PyTorch `.safetensors`; a missing `.pt` file
  alone is not an error.
- CUDA and FlashAttention are expected for the current GR00T checkpoint.
- GR00T and Isaac Sim are not expected to fit together on 12 GB VRAM.
- `PROJECTS_DIR=..` is correct in `docker/.env.groot-unitree.example` because
  Compose resolves relative paths from the `docker/` directory.
- On the reviewed Windows checkout, root shell scripts had CRLF worktree
  endings despite LF Git attributes. Bash failed with `pipefail\r`; normalize
  them to LF before relying on the Bash checks.
- The Vast.AI single-container image has not yet been built successfully in
  this workspace.

## Implementation Map

- Compose: `docker/docker-compose.groot-unitree.yml`
- Workstation env template: `docker/.env.groot-unitree.example`
- Single-container Dockerfile: `docker/1-docker-setup/Dockerfile`
- GR00T launcher: `docker/1-docker-setup/scripts/start_groot_server.sh`
- Unitree launcher: `docker/1-docker-setup/scripts/start_unitree_sim.sh`
- Layout check: `scripts/verify_layout.sh`
- Module check: `scripts/run_module_checks.sh`
- Remote simulation Compose: `docker/docker-compose.simulation.yml`
- Remote simulation image: `docker/simulation/Dockerfile`
- Remote simulation guide: `docs/REMOTE_SIMULATION.md`

## Current Technical Backlog

Keep this list ordered and update it after substantial progress.

- [ ] Build and run the remote simulation image on the IKR server.
- [ ] Verify a ten-minute stable G1/table/cubes scene over WebRTC.
- [ ] Run the first complete single-container Docker build.
- [ ] Resolve concrete CUDA, Isaac Sim, PyTorch or GR00T dependency errors from
      that build.
- [ ] Verify whether `docker/1-docker-setup/Dockerfile.dockerignore` is honored
      by the installed Docker version.
- [ ] Test Isaac WebRTC through Vast.AI port mapping.
- [ ] Evaluate an alternative viewer, VPN or explicit port configuration if
      WebRTC fails through Vast NAT.
- [ ] Validate non-dry-run action application in Isaac Sim.
- [ ] Decide how `base_height_command` and `navigate_command` map into
      IsaacLab locomotion control.
- [ ] Normalize root shell-script line endings for Windows checkouts.

## Durable Decisions

- `2026-08-12`: Use two Compose services on normal workstations and one image
  with two processes on Vast.AI.
- `2026-08-12`: Validate GR00T actions in dry-run before applying them.
- `2026-08-12`: Keep always-loaded agent instructions concise and store
  technical context in this task-routed document.
- `2026-08-12`: Establish a GR00T-free `idle` simulation baseline before
  testing inference or robot actions; disable DDS in this mode to avoid
  interacting with other Unitree participants on the shared IKR network.
- `2026-08-12`: Use native WebRTC over host networking as the first remote
  viewer path; keep browser viewing as an optional later layer.

Add future entries as `YYYY-MM-DD: decision - reason`. Remove or supersede
entries that no longer apply.
