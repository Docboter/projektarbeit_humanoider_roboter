# syntax=docker/dockerfile:1
# TL;DR: Trainings-Image (CUDA 12.8, uv) — ENTRYPOINT ist entrypoint.sh; Build-Kontext ist Training/.
# Projektarbeit Humanoider Roboter – Unitree G1 mit GR00T N1.6
#
# Build (Build-Context ist der Training/-Ordner, der dieses Dockerfile + scripts/ enthält):
#   docker build -t projektarbeit-humanoider-roboter Training/      # vom Repo-Root aus
#   cd Training && docker build -t projektarbeit-humanoider-roboter .
#
# Braucht BuildKit (Docker >= 23, sonst DOCKER_BUILDKIT=1): dieses Dockerfile nutzt
# `RUN --mount=type=cache`, `COPY --chmod` und Heredocs. Der alte Classic Builder
# scheitert daran. Eine .dockerignore hält den Build-Kontext auf scripts/ beschränkt.
#
# Run (autonom — KEIN --rm, KEIN -v; alle Daten leben im Container-Filesystem):
#   docker run --name groot-train --gpus all --ipc=host --shm-size=16g \
#     -e HF_TOKEN=hf_... \
#     -e WANDB_API_KEY=... \
#     -e MAX_STEPS=30000 \
#     -e GLOBAL_BATCH_SIZE=8 \
#     -it projektarbeit-humanoider-roboter
#
# Interaktive Shell (für Debugging):
#   docker run -it --name groot-debug --gpus all --ipc=host --shm-size=16g \
#     projektarbeit-humanoider-roboter bash

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Damit der apt-Cache-Mount unten überhaupt etwas zu sehen bekommt, sind ZWEI Dinge nötig
# (verifiziert — nur eines davon lässt den Cache leer):
#   1. docker-clean entfernen: diese apt-Konfiguration des Base-Image löscht nach JEDEM
#      apt-Aufruf /var/cache/apt/archives/*.deb.
#   2. Keep-Downloaded-Packages=true: apt selbst räumt die .deb nach der Installation
#      sonst ebenfalls weg (Default seit apt 2.x).
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# System-Pakete
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        git-lfs \
        curl \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python-is-python3 \
        ffmpeg \
        libegl1 \
    && git lfs install

# uv Paketmanager (exakte Version wie im Original)
RUN curl -LsSf https://astral.sh/uv/0.8.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Repositories klonen (gepinnte Commits für Reproduzierbarkeit)
# Branch: luca/g1-dex3 — enthält N1.6-kompatibles gr00t-Modul + korrektes uv.lock
# Commit aktualisieren: ./update_image.sh --update-commit
# (update_image.sh sucht den Pin per `grep 'checkout [0-9a-f]{40}'` — die Zeile
#  darf den Commit also nicht in eine ARG-Variable auslagern.)
RUN git clone --branch luca/g1-dex3 https://github.com/lucam06/Isaac-GR00T.git /app/Groot-1.6 \
    && git -C /app/Groot-1.6 checkout 9508b498cb60c0048a8a3ce562b518b53f3d0bcf \
    && git -C /app/Groot-1.6 lfs pull

# Python-Abhängigkeiten installieren (aus gefrorenem uv.lock)
WORKDIR /app/Groot-1.6
RUN UV_PREVIEW=1 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 \
    uv sync --frozen --no-install-project --extra dev --no-cache
RUN uv pip install --python /app/Groot-1.6/.venv/bin/python -e . --no-deps
# jsonlines fehlt im pyproject.toml/uv.lock des Forks, wird aber vom v3→v2-Konverter
# gebraucht (scripts/lerobot_conversion/convert_v3_to_v2.py). Version pinnen, sonst ist
# das die einzige ungepinnte Abhängigkeit in einem sonst vollständig gepinnten Image.
RUN uv pip install --python /app/Groot-1.6/.venv/bin/python jsonlines==4.0.0

# EGL/Vulkan für headless Rendering (MuJoCo, PyOpenGL).
# Heredoc-COPY statt `RUN cat > … << 'EOF'`: inhaltlich identisch und ein Layer weniger,
# weil COPY fehlende Zielverzeichnisse selbst anlegt (kein `mkdir -p` nötig). Ausserdem
# konnte der Dockerfile-Parser der VS-Code-Container-Extension (dockerfile-utils) Heredocs
# INNERHALB eines RUN nicht lesen und meldete dort 10 Phantomfehler; diese Form versteht er.
COPY <<'EOF' /usr/share/glvnd/egl_vendor.d/10_nvidia.json
{
    "file_format_version": "1.0.0",
    "ICD": { "library_path": "libEGL_nvidia.so.0" }
}
EOF

COPY <<'EOF' /usr/share/vulkan/icd.d/nvidia_icd.json
{
    "file_format_version": "1.0.0",
    "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.2.140" }
}
EOF

ENV PATH="/app/Groot-1.6/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/app/Groot-1.6/.venv" \
    GROOT_ROOT="/app/Groot-1.6" \
    MUJOCO_GL="egl" \
    PYOPENGL_PLATFORM="egl" \
    __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json" \
    PYTHONUNBUFFERED=1 \
    UV_PREVIEW=1

# Datenverzeichnisse anlegen.
# ACHTUNG — KEIN Host-Volume: /data liegt im Container-Filesystem und ist die einzige
# Wahrheit. `docker run --rm` bzw. `docker rm` löscht Daten und Checkpoints unwiderruflich.
# (Ausnahme KISSKI: dort bind-mountet Apptainer die VAST-Projektablage nach /data.)
RUN mkdir -p /data/models /data/unitreerobotics /data/G1_Dex3_BlockStacking /data/g1_dex3_finetune /data/logs

# Scripts ins Image kopieren — kein Volume-Mount mehr nötig.
# --chmod setzt die Rechte beim Kopieren (ein Layer statt zwei) und hängt nicht an
# einem Glob wie `chmod +x /scripts/*.py`, der bricht, sobald der Typ mal fehlt.
COPY --chmod=755 scripts/ /scripts/

# Defaults für autonomen Container-Lauf (vast.ai etc.).
# HF_TOKEN und WANDB_API_KEY MÜSSEN beim docker run via -e gesetzt werden.
# Diese Werte müssen mit den ${VAR:-…}-Defaults in scripts/entrypoint.sh bzw.
# scripts/run_finetuning*.sh übereinstimmen — tools/gen_docs.sh prüft genau das.
ENV MAX_STEPS=20000 \
    GLOBAL_BATCH_SIZE=8 \
    NUM_GPUS=1 \
    WANDB_PROJECT=gr00t-g1-dex3 \
    DATA_DIR=/data \
    SKIP_DOWNLOAD=0 \
    SKIP_CONVERT=0 \
    SKIP_TRAIN=0 \
    SHELL_ON_ERROR=0 \
    TUNE_VISUAL=0 \
    USE_COTRAIN=0 \
    USE_RL=0 \
    TRAIN_TEST_SPLIT=0 \
    TRAIN_SPLIT_RATIO=0.8 \
    USE_AUGMENTATION=1

# Entrypoint orchestriert Download → Konvertierung → Training autonom.
# Beim Aufruf mit Argumenten (z. B. `docker run … bash`) wird das Argument
# stattdessen direkt ausgeführt — Entrypoint-Skript reicht durch.
# WORKDIR bleibt /app/Groot-1.6 (weiter oben gesetzt, danach unverändert).
ENTRYPOINT ["/scripts/entrypoint.sh"]
