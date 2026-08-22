# syntax=docker/dockerfile:1
# TL;DR: Augmentierungs-Image (Cosmos-Transfer2.5) — nimmt echte Trainingsvideos, erzeugt
# stilvariierte Kopien (Domain-Randomization) als COTRAIN_DATASET_PATH-kompatiblen Datensatz.
# Projektarbeit Humanoider Roboter – Video-Augmentierung mit NVIDIA Cosmos-Transfer2.5
#
# Build (Build-Kontext ist der Augmentation/-Ordner, der dieses Dockerfile + scripts/ enthält):
#   docker build -t projekt-humanoider-roboter-augmentation Augmentation/   # vom Repo-Root aus
#   cd Augmentation && docker build -t projekt-humanoider-roboter-augmentation .
#
# Braucht BuildKit (Docker >= 23, sonst DOCKER_BUILDKIT=1): dieses Dockerfile nutzt Multi-Stage-
# Builds, `RUN --mount=type=cache` und `COPY --from=<image>`. Eine .dockerignore hält den
# Build-Kontext auf scripts/ beschränkt.
#
# Run (autonom — KEIN --rm, KEIN -v; alle Daten leben im Container-Filesystem):
#   docker run --name groot-augment --gpus all --ipc=host --shm-size=16g \
#     -e HF_TOKEN=hf_... \
#     -e AUGMENT_EPISODE_LIMIT=2 \
#     -it projekt-humanoider-roboter-augmentation
#
# Interaktive Shell (für Debugging):
#   docker run -it --name groot-augment-debug --gpus all --ipc=host --shm-size=16g \
#     projekt-humanoider-roboter-augmentation bash
#
# Voraussetzung, die dieses Dockerfile NICHT automatisieren kann: die "NVIDIA Open Model
# License Agreement" muss einmalig manuell auf huggingface.co/nvidia/Cosmos-Transfer2.5-2B
# akzeptiert werden — der Entrypoint prüft das beim Start und bricht mit einer klaren
# Fehlermeldung ab, statt mitten im Lauf mit einem kryptischen 403 zu scheitern.

# ── Stage 1: cosmos-transfer2.5-Quellcode, gepinnter Commit ─────────────────────────────
# Reiner Clone-Helper, landet NICHT im finalen Image. Commit aktualisieren:
# ./update_image.sh --update-commit (sucht den Pin per `grep '^ARG COSMOS_COMMIT=[0-9a-f]{40}'`
# — die ARG-Zeile muss den Commit deshalb als sichtbaren Default tragen, nicht nur die
# RUN-Zeile referenzieren).
#
# GEPINNT AUF DEN TAG v1.5.0, NICHT auf main-HEAD: main (und Tag v1.5.4) haben
# .python-version von 3.10 auf 3.13 angehoben, aber der custom flash-attn-Wheel-Index
# (cu128_torch27, v1.2.0 in pyproject.toml) liefert bislang NUR cp310-Wheels — `just
# install cu128` bricht dort mit "doesn't have a source distribution or wheel for the
# current platform" ab. v1.5.0 ist der letzte Tag vor dem Python-Bump und noch stimmig
# (verifiziert: .python-version=3.10, examples/inference.py + assets/robot_example/
# unverändert vorhanden). update_image.sh verfolgt deshalb bewusst die neueste TAG-Version,
# nicht main's beweglichen HEAD (sonst holt --update-commit dieselbe Inkonsistenz zurück).
FROM alpine/git:latest AS cosmos-src
ARG COSMOS_COMMIT=41fb49f9e9fdf3025731d4bf60142e7c6093006e
RUN git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git /cosmos-transfer2.5 \
    && git -C /cosmos-transfer2.5 checkout "$COSMOS_COMMIT"

# ── Stage 2: GR00T-Fork — NUR der v3→v2.1-Konverter + kanonische modality.json ───────────
# Sparse Checkout: kein uv sync, kein git-lfs, kein Torch/flash-attn aus diesem Fork — wir
# brauchen nur den Konverter, nicht das komplette Trainings-Setup.
# KEIN --depth 1: der gepinnte Commit ist ein aelterer, bewusst fixierter Stand (nicht
# zwingend der aktuelle Branch-Tip) — ein flacher Klon haette dessen Baum gar nicht erst
# ("fatal: unable to read tree"). Training/Dockerfile klont denselben Fork aus demselben
# Grund ohne Tiefenlimit.
# ACHTUNG: GROOT_FORK_COMMIT muss mit dem Pin in Training/Dockerfile Schritt halten (dort:
# git -C /app/Groot-1.6 checkout <sha>), sonst driftet die hier kopierte modality_4cam.json
# von der, die Training tatsächlich verwendet — und der COTRAIN-Gleichheitscheck
# (run_finetuning_cotrain.sh, cmp -s meta/modality.json) schlägt später fehl.
FROM alpine/git:latest AS groot-convert-src
ARG GROOT_FORK_COMMIT=9508b498cb60c0048a8a3ce562b518b53f3d0bcf
RUN git clone --no-checkout --branch luca/g1-dex3 \
        https://github.com/lucam06/Isaac-GR00T.git /groot-src \
    && git -C /groot-src sparse-checkout init --cone \
    && git -C /groot-src sparse-checkout set scripts/lerobot_conversion examples/G1_DEX3 \
    && git -C /groot-src checkout "$GROOT_FORK_COMMIT"

# ── Stage 3: Hauptimage ───────────────────────────────────────────────────────────────────
# Basis-Image + System-Pakete von Hand aus cosmos-transfer2.5/Dockerfile (Commit s.o.)
# übernommen — NICHT von dort FROM-t. Projekt-Konvention: das Image baut sich selbst, statt
# auf einem fremden Laufzeit-Image aufzusetzen (siehe Training/Dockerfile). Bei künftigen
# Commit-Updates ggf. mit `git show <sha>:Dockerfile` im cosmos-transfer2.5-Repo abgleichen —
# NOCH NICHT gegen einen echten Build verifiziert (offener Punkt, siehe anleitung.md).
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        git \
        git-lfs \
        tree \
        wget \
    && git lfs install

# uv Paketmanager + just Task-Runner — exakt wie im Original-Dockerfile von cosmos-transfer2.5,
# damit `just install` (unten) dieselben Rezepte findet, die dort getestet wurden.
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy \
    UV_TOOL_BIN_DIR=/usr/local/bin
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
    | bash -s -- --to /usr/local/bin --tag 1.42.4

# cosmos-transfer2.5: Quellcode rein, dann dessen EIGENES Install-Rezept fahren (statt den
# Bind-Mount/STANDALONE-Umweg des Original-Dockerfiles nachzubauen — wir haben die volle
# Quelle bereits per COPY --from, brauchen also keinen Build-Context-Trick).
# `just install` ruft laut justfile zuerst das `setup`-Rezept auf (Inhalt bei Redaktionsschluss
# nicht eingesehen — evtl. lädt es zusätzliche GPU-spezifische Wheels wie cosmos-transfer1 es
# für apex/flash_attn/natten/transformer_engine tat; kann den Build spürbar verlängern) und
# danach `uv sync --extra=cu128`. Braucht Internet für das git+https-Dependency
# video-depth-anything in pyproject.toml.
WORKDIR /app/cosmos-transfer2.5
COPY --from=cosmos-src /cosmos-transfer2.5 /app/cosmos-transfer2.5
ARG CUDA_NAME=cu128
RUN --mount=type=cache,target=/root/.cache/uv \
    just install ${CUDA_NAME}
ENV PATH="/app/cosmos-transfer2.5/.venv/bin:${PATH}"

# Zweite, schlanke venv fürs Datensatz-Tooling — bewusst getrennt von Cosmos' gepinntem
# Torch/Diffusers-Stack, damit unsere Deps (pandas/pyarrow für den vendorten
# v3→v2.1-Konverter, huggingface_hub für Down-/Upload) dessen Wheels nicht
# durcheinanderbringen. `uv venv --python` lädt die Python-Version bei Bedarf selbst nach —
# kein zusätzliches System-Python-Paket nötig. KEIN opencv-python-headless (frühere
# Version): dessen mitgeliefertes ffmpeg-Backend konnte AV1-Quellvideos nicht decodieren.
# Mittlerweile auch kein eigener Control-Video-Schritt mehr noetig — cosmos_transfer2/
# config.py::EdgeConfig erzeugt die Kantenerkennung selbst on-the-fly aus dem Quellvideo
# (siehe build_controlnet_specs.py).
RUN uv venv --python 3.11 /opt/venv-tools \
    && uv pip install --python /opt/venv-tools/bin/python \
        "huggingface_hub[cli]==0.34.*" \
        pandas \
        pyarrow \
        jsonlines==4.0.0 \
        numpy \
        tqdm
ENV TOOLS_PYTHON=/opt/venv-tools/bin/python

# GR00T-Fork: nur Konverter + kanonische modality_4cam.json (siehe Stage 2 oben)
COPY --from=groot-convert-src /groot-src/scripts/lerobot_conversion /app/groot-convert/lerobot_conversion
COPY --from=groot-convert-src /groot-src/examples/G1_DEX3/modality_4cam.json /app/groot-convert/modality_4cam.json

# Datenverzeichnisse anlegen.
# ACHTUNG — KEIN Host-Volume: /data liegt im Container-Filesystem und ist die einzige
# Wahrheit. `docker run --rm` bzw. `docker rm` löscht Daten unwiderruflich (siehe
# Training/Dockerfile für die ausführliche Begründung — hier gilt dieselbe Regel).
RUN mkdir -p /data/unitreerobotics /data/hf_cache \
             /data/augmentation/specs /data/augmentation/raw_output /data/logs

COPY --chmod=755 scripts/ /scripts/

# Defaults für autonomen Container-Lauf. Werte MÜSSEN mit den ${VAR:-…}-Defaults in
# scripts/entrypoint.sh übereinstimmen.
ENV DATA_DIR=/data \
    HF_HOME=/data/hf_cache \
    NUM_GPU=1 \
    SKIP_DOWNLOAD=0 \
    SKIP_CONVERT=0 \
    SOURCE_DATASET_REPO=unitreerobotics/G1_Dex3_BlockStacking_Dataset \
    AUGMENT_TRAIN_RATIO=0.8 \
    AUGMENT_EPISODE_LIMIT=2 \
    AUGMENT_EPISODE_IDS="" \
    AUGMENT_CAMERAS=cam_left_high,cam_right_high,cam_left_wrist,cam_right_wrist \
    AUGMENT_VARIANTS=1 \
    AUGMENT_EDGE_THRESHOLD=medium \
    AUGMENT_MODEL_VARIANT=edge \
    AUGMENT_NUM_STEPS=35 \
    AUGMENT_STRICT_FRAME_CHECK=1 \
    AUGMENT_OUT_DIR=/data/augmentation/g1_dex3_cosmos_augmented \
    AUGMENT_HF_REPO="" \
    SKIP_INFERENCE=0 \
    SKIP_ASSEMBLE=0 \
    AUGMENT_OVERWRITE=0 \
    SHELL_ON_ERROR=0

# Entrypoint orchestriert Download → Control-Videos → Specs → Cosmos-Inferenz → Datensatz
# autonom. Beim Aufruf mit Argumenten (z. B. `docker run … bash`) wird das Argument
# stattdessen direkt ausgeführt — Entrypoint-Skript reicht durch.
WORKDIR /app/cosmos-transfer2.5
ENTRYPOINT ["/scripts/entrypoint.sh"]
