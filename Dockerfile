# syntax=docker/dockerfile:1
# Projektarbeit Humanoider Roboter – Unitree G1 mit GR00T N1.6
#
# Build:
#   docker build -t projektarbeit-humanoider-roboter .
#
# Run (ohne docker-compose):
#   docker run -it --rm --gpus all --ipc=host \
#     -v $(pwd)/data:/data \
#     projektarbeit-humanoider-roboter

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# System-Pakete
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        git-lfs \
        curl \
        wget \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        python-is-python3 \
        software-properties-common \
        ffmpeg \
        libegl1 \
    && git lfs install

# uv Paketmanager (exakte Version wie im Original)
RUN curl -LsSf https://astral.sh/uv/0.8.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Repositories klonen (gepinnte Commits für Reproduzierbarkeit)
RUN git clone https://github.com/lucam06/Isaac-GR00T.git /app/Groot-1.6 \
    && git -C /app/Groot-1.6 checkout c047ce2386eb11964548faf0c2cb53f895b11201 \
    && git -C /app/Groot-1.6 lfs pull

# Python-Abhängigkeiten installieren (aus gefrorenem uv.lock)
WORKDIR /app/Groot-1.6
RUN UV_PREVIEW=1 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 \
    uv sync --frozen --no-install-project --extra dev --no-cache
RUN uv pip install --python /app/Groot-1.6/.venv/bin/python -e . --no-deps
RUN uv pip install --python /app/Groot-1.6/.venv/bin/python jsonlines

# EGL/Vulkan für headless Rendering (MuJoCo, PyOpenGL)
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json << 'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": { "library_path": "libEGL_nvidia.so.0" }
}
EOF
RUN mkdir -p /usr/share/vulkan/icd.d && \
    cat > /usr/share/vulkan/icd.d/nvidia_icd.json << 'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.2.140" }
}
EOF

ENV PATH="/app/Groot-1.6/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/app/Groot-1.6/.venv" \
    MUJOCO_GL="egl" \
    PYOPENGL_PLATFORM="egl" \
    __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

# Datenverzeichnisse anlegen (werden per Volume gemountet)
RUN mkdir -p /data/models /data/unitreerobotics /data/G1_Dex3_BlockStacking /data/g1_dex3_finetune

WORKDIR /app/Groot-1.6
CMD ["/bin/bash"]
