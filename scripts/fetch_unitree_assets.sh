#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$(pwd)}"
TARGET_DIR="${REPO_DIR}/repos/unitree_sim_isaaclab/assets"
ASSET_URL="https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds"

if [[ -d "${TARGET_DIR}" ]]; then
  echo "Assets already exist: ${TARGET_DIR}"
  echo "Nothing changed. Remove that directory explicitly if a fresh download is required."
  exit 0
fi

for command in git git-lfs unzip; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing command: ${command}" >&2
    exit 1
  fi
done

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "${tmp_dir}"
}
trap cleanup EXIT

echo "Downloading Unitree assets (more than 1 GB)..."
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${ASSET_URL}" "${tmp_dir}/download"
git -C "${tmp_dir}/download" lfs pull

archive="${tmp_dir}/download/assets.zip"
if [[ ! -f "${archive}" ]] || [[ "$(stat -c%s "${archive}")" -le 1073741824 ]]; then
  echo "Asset archive is missing or incomplete; check Git LFS and network access." >&2
  exit 1
fi

unzip -q "${archive}" -d "${tmp_dir}/unpacked"
if [[ ! -d "${tmp_dir}/unpacked/assets" ]]; then
  echo "Downloaded archive does not contain the expected assets directory." >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET_DIR}")"
mv "${tmp_dir}/unpacked/assets" "${TARGET_DIR}"
echo "Assets installed at ${TARGET_DIR}"
