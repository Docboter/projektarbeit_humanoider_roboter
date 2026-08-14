# Agent Instructions

Codex loads this file automatically for work in this repository. The user's
current request always takes precedence over this file.

## Start Here

Before changing files:

1. Inspect `git status` and preserve changes you did not create.
2. Use Documentation Routing to load only the context needed for the task.
3. Verify relevant documentation against the implementation before relying on
   it.

Do not ask the user to repeat information recorded here. Make reasonable,
reversible in-scope decisions autonomously. Ask before destructive, expensive,
externally visible or architecture-changing actions.

## Project Direction

Integrate NVIDIA GR00T with Unitree Isaac Sim / IsaacLab so a Unitree G1/G129
with DEX3 hands can execute language instructions in simulation.

Target flow:

```text
language prompt -> GR00T inference -> validated action mapping -> Unitree G1 simulation
```

### User Project Profile

Fill only the fields that matter right now. Unfilled fields are unknown, not
requirements. This is the complete user-facing configuration area: ten fields
are enough for a new agent to make good decisions.

- Final project outcome: `A reliable thesis demo integrating GR00T with Unitree G1 simulation.`
- Current milestone: `Stable remote Docker simulation on an IKR server, before GR00T integration.`
- Success criterion: `G1/G129 with DEX3 remains stable in front of a table and colored cubes and is visible from the desk laptop over WebRTC.`
- Demo or evaluation scenario: `Isaac-Stack-RgyBlock-G129-Dex3-Joint, observed for at least ten minutes without simulation errors.`
- Deadline, time or budget limit: `[optional]`
- Scope limits and non-goals: `[what agents should not pursue]`
- Runtime target and hardware: `Remote Linux server at IKR with NVIDIA GPU; exact GPU/VRAM/driver still to be recorded.`
- Development mode: `demo`
- Autonomy and change scope: `[propose first | implement in scope; minimal/moderate refactors]`
- Collaboration preferences: `German; be especially cautious on shared Docker hosts; never use Docker prune or broad cleanup commands.`

### Current Handoff

- The current milestone is a remote, GR00T-free baseline using
  `--action_source idle` with `Isaac-Stack-RgyBlock-G129-Dex3-Joint`.
- Idle holds the G1/G129 at its default pose and deliberately disables GR00T,
  Unitree DDS command channels and the additional TeleImager server.
- Next session: follow `docs/REMOTE_SIMULATION.md` to build and run the image
  on the IKR server, then verify a stable ten-minute WebRTC view before adding
  GR00T or motion control.

## Non-Negotiable Rules

- Default to `--groot_debug --groot_dry_run` when testing GR00T actions.
- Apply actions only after server readiness, action keys, dimensions and values
  have been inspected and look sane.
- Return to dry-run immediately if the robot jumps, tips over or receives
  implausible targets.
- Do not claim GPU, runtime, WebRTC or robot behavior was tested when only
  static checks ran.
- Treat `repos/Isaac-GR00T` and `repos/unitree_sim_isaaclab` as separate Git
  repositories. Inspect their status before editing and never discard dirty
  submodule work.
- Keep GR00T and Unitree Python environments separate unless a verified design
  change requires otherwise.

## Working Rules

- Prefer existing patterns and the smallest coherent change.
- Keep edits scoped to the requested outcome and preserve observability around
  model inputs, returned actions and applied targets.
- Add checks in proportion to risk; report checks that could not run.
- Update docs when behavior, setup, architecture or durable decisions change.

Baseline checks when relevant:

```bash
./scripts/verify_layout.sh .
./scripts/run_module_checks.sh .
```

For runtime changes, progress through static checks, image or Compose checks,
server readiness, a direct GR00T query, Isaac dry-run, and only then applied
simulation actions.

## Documentation Routing

- `README.md`: project entry point and quickstart.
- `docs/PROJECT_CONTEXT.md`: architecture, implementation state, constraints,
  decisions and backlog. Read when work touches runtime design, submodules,
  action mapping, validated constraints or technical priorities.
- `docs/SETUP_AND_TESTING.md`: normal two-service workstation setup, commands,
  testing and troubleshooting. Read for Compose, setup or runtime work.
- `docs/VAST_SINGLE_CONTAINER.md`: Vast.AI image, ports and single-container
  operation. Read only for Vast.AI or single-container work.

## Definition of Done

A task is done when the requested outcome exists, relevant checks pass or their
blockers are stated, documentation matches behavior, and unrelated user changes
remain intact.

## Context Maintenance

Agents should preserve important long-term information learned in chats. Add
verified project decisions, user preferences, constraints, milestones and
lessons from completed work to the appropriate context file without waiting for
an explicit reminder. Keep only information likely to help future tasks; do not
store transient discussion, command output, credentials or other sensitive
data.

Update `docs/PROJECT_CONTEXT.md` when architecture, validated constraints,
action mapping, technical priorities or durable decisions change. Update this
file only when the project direction, universal rules or user profile changes.
Keep transient logs and session notes out of both files.
