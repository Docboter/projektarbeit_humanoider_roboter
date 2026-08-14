#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$(pwd)}"
cd "${REPO_DIR}"

echo "== Project layout =="
required_paths=(
  "repos/Isaac-GR00T"
  "repos/unitree_sim_isaaclab"
  "docker/docker-compose.groot-unitree.yml"
  "docker/docker-compose.simulation.yml"
  "docker/simulation/Dockerfile"
  "docker/simulation/start_simulation.sh"
  "docker/.env.groot-unitree.example"
  "scripts/verify_layout.sh"
)

for path in "${required_paths[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "ok: ${path}"
  else
    echo "missing: ${path}" >&2
    exit 1
  fi
done

echo
echo "== Submodule status =="
git submodule status repos/Isaac-GR00T repos/unitree_sim_isaaclab

echo
echo "== Shell syntax =="
bash -n scripts/verify_layout.sh
bash -n scripts/run_module_checks.sh
bash -n scripts/fetch_unitree_assets.sh
bash -n docker/simulation/start_simulation.sh
echo "ok: shell scripts parse"

echo
echo "== Python syntax: Unitree integration =="
python3 -m py_compile \
  repos/unitree_sim_isaaclab/sim_main.py \
  repos/unitree_sim_isaaclab/action_provider/action_provider_groot.py \
  repos/unitree_sim_isaaclab/action_provider/action_provider_idle.py \
  repos/unitree_sim_isaaclab/action_provider/groot_onnx.py \
  repos/unitree_sim_isaaclab/action_provider/create_action_provider.py \
  repos/unitree_sim_isaaclab/groot_prompt_terminal.py \
  repos/unitree_sim_isaaclab/tools/test_groot_server_action.py
echo "ok: Unitree integration files compile"

echo
echo "== Python syntax: GR00T server patch =="
python3 -m py_compile repos/Isaac-GR00T/gr00t/eval/run_gr00t_server.py
echo "ok: GR00T server file compiles"

echo
echo "== Docker Compose config =="
if command -v docker >/dev/null 2>&1; then
  PROJECTS_DIR="${REPO_DIR}" docker compose \
    --env-file docker/.env.groot-unitree.example \
    -f docker/docker-compose.groot-unitree.yml \
    config >/dev/null
  echo "ok: docker compose config renders"
  docker compose \
    --env-file docker/.env.simulation.example \
    -f docker/docker-compose.simulation.yml \
    config >/dev/null
  echo "ok: simulation compose config renders"
else
  echo "skip: docker command not found"
fi

echo
echo "Module checks passed."
