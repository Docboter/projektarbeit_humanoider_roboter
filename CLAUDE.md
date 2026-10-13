# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠ Open maintenance issues — ask the user every session

Known, deliberately postponed defects. **At the start of every session, before other work, list
these to the user and ask whether to fix one now.** If they decline, carry on and do not fix them
unasked. When one is fixed, delete its entry here (and note the fix in `docs/historie.md`).

1. **`_widen_finger_joint_limits` does not cover `thumb_0`/`thumb_2`** (`Simulation/g1_dex3_sim/`),
   although the dataset init reaches down to −0,71 rad there, so the sim may clamp the thumb.
   Needs hardware: run `tipcheck` once on the IKR server (`./Simulation/server_rl_run.sh tipcheck`)
   and widen the limits if the thumb is clamped (Review 10-13 S7).

## Project overview

Fine-tuning of **NVIDIA GR00T** (Vision-Language-Action model) on the **Unitree G1 + DEX3 hand**
for block stacking, plus a closed-loop **Isaac Lab sim eval**. Two GR00T generations live side by
side, selected at runtime via `GROOT_VERSION=1.6|1.7|auto`: **N1.6** (default, `app/Groot-1.6`,
Python 3.10) and **N1.7** (parallel path, `app/Groot-1.7`, Python 3.12/torch 2.9, gated
Cosmos-Reason2-2B backbone; checkpoints are not cross-loadable). Everything runs in a
**self-contained Docker container** (CUDA 12.8, `uv`): launching the image runs
`/scripts/entrypoint.sh` (source `Training/scripts/entrypoint.sh`), which does download →
v3→v2 conversion → training, configured by env vars only. The same image runs locally, on vast.ai
and on the **KISSKI HPC cluster** (GWDG) via Apptainer. One training image per generation
(`Training/update_image.sh --groot=1.6|1.7|both`, tags `:latest` / `:latest-n17`).

**Status (2026-10-13), short:** best validated N1.6 checkpoint is vision-v2 (run 3) **step 44000**
(20×750 re-measurement refutes the old 30000 U-curve). N1.7: full KISSKI run (44k) done, open loop
on par with N1.6; sim runs 54/55 → `lifted` 0/10. Closed loop overall: 0 successes; cause not
identified (domain gap is one candidate among several). Co-training, Cosmos augmentation and RL:
tools built, no result. **The status is maintained in exactly one place:
[`next-steps.md` §0](next-steps.md)** (N1.7 detail:
[`groot-n17-migration.md` → Stand der Umsetzung](docs/weiterfuehrend/groot-n17-migration.md#stand-der-umsetzung-2026-08-19)).
Do not repeat status narrative here — link instead.

## Critical rules

- **Never `docker run --rm`** with the training image. The container holds all data and
  checkpoints; `--rm` wipes them on stop. `stop`/`start` preserves state, only `docker rm` destroys it.
- **Storage model:** no host-side persistent storage, no `-v` mount by default (opt-in only via
  `TRAIN_HOST_DATA_DIR` in the pull launcher, see `docs/portabilitaet.md`). `/data` is a regular directory
  inside the container (matches vast.ai: one instance == one container). **Exception — KISSKI:**
  Apptainer bind-mounts `/mnt/vast-kisski/projects/kisski-humrob/data` to `/data`, so data persists
  on VAST project storage (old `/scratch/` decommissioned 2026-03-31).
- **Scripts are COPIED into the image.** Changes to `Training/scripts/*` (or `Simulation/scripts/*`,
  `Simulation/g1_dex3_sim/*`) need a rebuild. Exception: `kisski_submit.sh` bind-mounts
  `${REPO_DIR}/Training/scripts:/scripts`, so repo changes take effect on KISSKI without a rebuild.
  `tools/` is host-only and never needs a rebuild.
- `run_finetuning.sh` detects Docker (`.dockerenv`) and Apptainer (`$APPTAINER_NAME` /
  `$SINGULARITY_NAME`) itself — no changes needed for KISSKI.
- **All G1/DEX3 work goes in `app/Groot-1.6/`** (and `app/Groot-1.7/` for the N1.7 path). Both are
  submodules (fork `lucam06/Isaac-GR00T`); each has its own venv — use `uv run` in the matching tree.
  A change there needs a fork commit + pin update in the Dockerfiles.
- **Sim GPU:** Ampere+ **with RT cores** (L40, RTX 4090, A6000, RTX PRO 6000). A100/H100 have no RT
  cores; the KISSKI jupyter RTX 5000 is Turing (too old for Isaac Sim 5.1/6.0). So the sim runs on
  the IKR server (or a rented RT-core box), never on KISSKI.

## Key commands

```bash
./run.sh                                      # THE entry point: domain -> action -> parameters
./run.sh sim eval                             # straight to eval's parameters
MENU=0 ./run.sh sim eval                      # menu off — behaves exactly as before
```

`run.sh` only picks (mapping in `tools/menu/_domains.spec`) and hands over to the host launchers
(`Simulation/server_rl_run.sh`, `Training/setup_and_train_dockerhub_pull.sh`,
`Training/kisski_menu.sh`), which ask their own questions and print the equivalent one-liner.
No menu in containers, under SLURM, in CI or without a TTY. Details, switches (`MENU_ARROWS`,
`MENU_NEST`, back path): [`cli-menuefuehrung.md`](docs/weiterfuehrend/cli-menuefuehrung.md) §13.

```bash
# Training, local (launcher handles existing-container detection)
HF_TOKEN=hf_... WANDB_API_KEY=... ./Training/setup_and_train_dockerhub_pull.sh
# ...or manually — NO --rm, NO -v:
export HF_TOKEN=hf_... WANDB_API_KEY=...       # secrets: pass by NAME (-e HF_TOKEN), never -e NAME=value
docker run --name groot-train --gpus all --ipc=host --shm-size=16g \
  -e HF_TOKEN -e WANDB_API_KEY -e MAX_STEPS=30000 -e GLOBAL_BATCH_SIZE=8 \
  -it lucam03/projekt-humanoider-roboter:latest            # N1.7: :latest-n17 + -e GROOT_VERSION=1.7
docker start -ai groot-train                               # resume after stop
docker exec -it groot-train bash                           # shell into running container
docker cp groot-train:/data/g1_dex3_finetune ./checkpoints # extract artifacts

# KISSKI (login node glogin-gpu.hpc.gwdg.de); SIF once: apptainer pull, see kisski-hpc.md
export HF_TOKEN=hf_... WANDB_API_KEY=... GLOBAL_BATCH_SIZE=32
sbatch --export=ALL Training/kisski_submit.sh   # --export=ALL needed (login node sets SBATCH_EXPORT=none)
squeue -u $USER; tail -f logs/slurm-<jobid>.out

# Build images (tags :<repo-branch> + :<timestamp>; :latest only with --push-latest)
./Training/update_image.sh --groot=1.6          # or 1.7 | both (both = dual venv, never default)
./Simulation/update_sim_image.sh --standalone   # sim image; Docker Hub sim image is still N1.6-only
python Simulation/scripts/upload_checkpoint.py --checkpoint /path/to/checkpoint-44000 \
  --repo luca-mue/groot-g1dex3-checkpoint       # skips optimizer.pt
```

Dev loop inside the container (entrypoint is bypassed when a command is passed):

```bash
docker run -it --name groot-dev -v $(pwd)/Training/scripts:/scripts \
  --gpus all --ipc=host --shm-size=16g lucam03/projekt-humanoider-roboter:latest bash
bash /scripts/run_finetuning.sh                                           # re-run training only
cd /app/Groot-1.6 && python -m pytest tests/ -m 'not gpu' -v --timeout=300   # CPU tests
cd /app/Groot-1.6 && pre-commit run --all-files                           # lint + format
```

Host checks (run after doc/spec/script changes): `tools/gen_docs.sh` (spec defaults vs.
`${VAR:-…}` vs. doc tables), `tools/check_tldr.sh` (TL;DR convention), `tools/test_menu.sh`
(menu acceptance suite).

## Configuration (env vars)

The entrypoint reads everything from env vars; defaults are `ENV` in the Dockerfile or shell
defaults in the scripts. **Reference:** [`docs/training/env-vars.md`](docs/training/env-vars.md) —
training, mode-dependent learning rate (KISSKI: `2e-4` standard, `1e-4` with `TUNE_VISUAL`/
`USE_COTRAIN`; locally always `1e-4`), the sim-eval container, host helpers, VRAM table, and the
pointer to `tools/menu/*.spec` for action-specific sim variables. Sim step-by-step:
[`sim-eval-anleitung.md`](docs/simulation/sim-eval-anleitung.md); augmentation:
[`docs/augmentation/anleitung.md`](docs/augmentation/anleitung.md).

Most important switches: `HF_TOKEN` (required), `WANDB_API_KEY`, `GROOT_VERSION`, `MAX_STEPS`
(entrypoint 20000, launcher 30000, KISSKI 44000), `GLOBAL_BATCH_SIZE`, `NUM_GPUS`, `TUNE_VISUAL`,
`TRAIN_TEST_SPLIT`, `USE_AUGMENTATION`, `USE_COTRAIN`, `DOCKER_NAMESPACE` (host-side; push account,
asked for — never silently `lucam03`). Sim: `CHECKPOINT_PATH`, `NUM_EPISODES`, `EPISODE_LENGTH_S`
(always set it), `EVAL_LAYOUT`, `DR_SEED`, `RL_SKIP_PULL=1` for a locally built sim image.

**VRAM (N1.6):** 24 GB → bs 1–2 (very slow), 32 GB → bs 4–8, 80 GB A100 → 64–128, 94 GB H100 →
128+. N1.7 with `TUNE_VISUAL=1` needs ~58 GiB at bs 8 on one GPU (does not fit a 32 GB card).
On OOM, halve `GLOBAL_BATCH_SIZE` first. KISSKI partitions: `kisski` (A100 80 GB), `kisski-h100`
(H100 94 GB), max walltime 48 h.

## Code style and conventions

- **Formatter** `ruff format`, **linter** `ruff check` (rules E, F, I), **line length 100**;
  config for our own code in the root `pyproject.toml` (the forks under `app/` keep their own).
- **CI** (`.github/workflows/ci.yml`, no GPU): `gen_docs`, `check_tldr`, `test_menu`,
  `lib_groot_version.sh` diff, `tools/ci_check_pins.sh`, `ruff check`, `ruff format --check`
  (non-blocking for now), `shellcheck --severity=error`, Isaac-free sim tests. Details:
  [`docs/README.md` → Entwicklung](docs/README.md#entwicklung-checks-und-ci).
- `graphify-out/` is local only (gitignored); regenerate with `graphify update .`.
- **Commit messages: English**, subject and body — regardless of the language of the code
  comments or docs the commit touches (those stay German where they already are).
- **TL;DR convention (every doc page and every script).** A Markdown page under `docs/` (plus
  `README.md`, `next-steps.md`) starts with a `> **TL;DR:** …` blockquote directly under its H1:
  1–3 German sentences — what the page is, when to read it, what is *not* here (link). A script
  under `Training/`, `Simulation/`, `tools/` (`.sh`/`.py`/`.ps1`/`.spec`, `Dockerfile*`,
  `docker-compose.yml`) carries `# TL;DR: …` as the first comment line after the shebang (line 2
  of a Dockerfile, below `# syntax=`). New files follow suit; `tools/check_tldr.sh` fails when one
  is missing and `tools/check_tldr.sh --list` prints the whole overview.
- Docs are German. Superseded findings go to `docs/historie.md` or get an "überholt durch …" note
  with link — history is not rewritten.

## Where things are

- **Doc hub + full repo structure tree + runtime `/data` layout:** [`docs/README.md`](docs/README.md)
- **Status and priorities:** [`next-steps.md`](next-steps.md)
- **Quickstart / accounts / tokens (SSOT):** [`docs/quickstart.md`](docs/quickstart.md)
- **Training:** [`anleitung.md`](docs/training/anleitung.md) · [`env-vars.md`](docs/training/env-vars.md) ·
  [`kisski-hpc.md`](docs/training/kisski-hpc.md) · [`co-training.md`](docs/training/co-training.md) ·
  [`train-test-split.md`](docs/training/train-test-split.md)
- **Simulation:** [`sim-eval-anleitung.md`](docs/simulation/sim-eval-anleitung.md) ·
  [`umsetzungsnotizen.md`](docs/simulation/umsetzungsnotizen.md) (known fixes, read first) ·
  [`live-ansicht.md`](docs/simulation/live-ansicht.md)
- **Augmentation (Cosmos-Transfer2.5, built, not verified end-to-end):**
  [`docs/augmentation/anleitung.md`](docs/augmentation/anleitung.md)
- **Results:** [`docs/ergebnisse/`](docs/ergebnisse/README.md); "Lauf N" references resolve in
  [`diagnose-chronik.md`](docs/ergebnisse/diagnose-chronik.md)
- **Further work:** RL ([plan](docs/weiterfuehrend/reinforcement-learning-plan.md),
  [guide](docs/weiterfuehrend/rl-anleitung.md), N1.6-only), [N1.7 migration](docs/weiterfuehrend/groot-n17-migration.md),
  [CLI menus](docs/weiterfuehrend/cli-menuefuehrung.md), livestream, locomotion
- **Portability (`.env.local`, `KISSKI_PROJECT_DIR`/`KISSKI_SIF_DIR`):** [`docs/portabilitaet.md`](docs/portabilitaet.md)
- **Troubleshooting:** [`docs/fehlerbehebung.md`](docs/fehlerbehebung.md) · **Reviews:**
  [`projekt-review-2026-10-13.md`](docs/projekt-review-2026-10-13.md)
- **Fork guides:** `app/Groot-1.6/examples/G1_DEX3/` (`SETUP_DOCUMENTATION.md`, `FINETUNING_GUIDE.md`, `README.md`)

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
