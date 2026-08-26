# Repository Guidelines

## Project Structure & Module Organization

This repository fine-tunes NVIDIA GR00T N1.6 for the Unitree G1 with DEX3 hands and evaluates it in Isaac Lab.

- `Training/`: Docker and Apptainer launchers, training entrypoints, and checkpoint tools.
- `Simulation/`: Isaac Lab environments, ZMQ policy clients, evaluation scripts, Dockerfiles, and server workflows.
- `app/Groot-1.6/`: GR00T fork tracked as a Git submodule.
- `data/`: robot assets and the `unitree_ros` submodule; avoid committing generated datasets or checkpoints.
- `docs/`: German operational guides, troubleshooting, and experiment reports.
- `latex/`: thesis sources, figures, bibliography, and Makefile.

## Build, Test, and Development Commands

```bash
docker build -t projektarbeit-humanoider-roboter Training/
./Training/setup_and_train_dockerhub_pull.sh
./Simulation/update_sim_image.sh --standalone --skip-push
./Simulation/server_rl_run.sh preflight
./Simulation/server_rl_run.sh eval
cd latex && make
```

The training launcher downloads data and runs fine-tuning in a persistent container. Never add `--rm`: checkpoints and logs normally live inside that container. Simulation commands require a compatible NVIDIA GPU and should be validated on the target server.

## Coding Style & Naming Conventions

Use four spaces in Python and two spaces in shell continuation blocks. Python follows Ruff with a 100-character line limit (`ruff format`, `ruff check --select E,F,I`). Use `snake_case` for Python functions/files, uppercase names for environment variables, and descriptive shell functions such as `do_preflight`. Preserve the surrounding language: code identifiers and commits are English; existing user-facing documentation and comments are commonly German.

## Testing Guidelines

Run the narrowest relevant checks first:

```bash
bash -n Simulation/server_rl_run.sh Training/scripts/entrypoint.sh
cd app/Groot-1.6 && python -m pytest tests/ -m 'not gpu' -v --timeout=300
cd app/Groot-1.6 && pre-commit run --all-files
```

Name Python tests `test_*.py`. Clearly distinguish static/local validation from GPU, Isaac Sim, KISSKI, or Blackwell hardware validation in documentation and PRs.

## Commit & Pull Request Guidelines

Use concise English subjects following the existing convention, for example `feat: Add TensorRT policy backend` or `fix: Reject stale camera frames`. PRs should explain the motivation, affected training/simulation path, validation performed, and any hardware tests still pending. Link relevant issues and include screenshots or benchmark artifacts for visual or performance changes.

## Security & Configuration

Pass `HF_TOKEN`, `WANDB_API_KEY`, and server addresses through environment variables. Never commit credentials, downloaded model weights, generated TensorRT engines, datasets, logs, or local cache directories.
