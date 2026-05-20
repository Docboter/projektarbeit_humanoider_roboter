# syntax=docker/dockerfile:1
#
# Single-container test image for Vast.AI.
#
# Build context:
#   project root: projektarbeit_humanoider_roboter
#
# Build:
#   docker build -f docker/1-docker-setup/Dockerfile -t groot-unitreesim-single-docker-setup:latest .
#
# Runtime model:
#   one Docker container, two processes:
#   - /usr/local/bin/start_groot_server.sh
#   - /usr/local/bin/start_unitree_sim.sh
#
# The two Python environments intentionally stay separate:
#   - GR00T: /workspace/Isaac-GR00T/.venv, Python 3.10, uv-managed
#   - Isaac/Unitree: /opt/conda/envs/unitree_sim_env, Python 3.11, conda-managed

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display,video \
    CUDA_HOME=/usr/local/cuda \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    OMNI_KIT_ALLOW_ROOT=1 \
    OMNI_KIT_DISABLE_STARTUP=1 \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_ENV_PRIVACY_CONSENT=Y \
    CYCLONEDDS_HOME=/cyclonedds/install

# -----------------------------------------------------------------------------
# Shared base packages
# Source:
#   - repos/Isaac-GR00T/docker/Dockerfile system dependencies
#   - repos/unitree_sim_isaaclab/Dockerfile builder/runtime dependencies
# Why:
#   - GR00T needs build tooling, Python 3.10, uv, ffmpeg.
#   - Isaac/Unitree needs Conda, GCC/CMake, Vulkan/GL, git-lfs, unzip.
# -----------------------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked <<'EOF'
set -e
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    gcc-12 \
    g++-12 \
    cmake \
    curl \
    wget \
    git \
    git-lfs \
    unzip \
    ffmpeg \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    python-is-python3 \
    software-properties-common \
    libglu1-mesa \
    libglu1-mesa-dev \
    libegl1 \
    libvulkan1 \
    vulkan-tools \
    zenity \
    pkg-config

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

python -m pip install --upgrade pip setuptools wheel
curl -LsSf https://astral.sh/uv/0.8.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
git lfs install --system
EOF

# -----------------------------------------------------------------------------
# GR00T environment
# Source:
#   repos/Isaac-GR00T/docker/Dockerfile
# What this section installs:
#   - GR00T source from repos/Isaac-GR00T
#   - uv dependencies into /workspace/Isaac-GR00T/.venv
#   - editable install for Isaac-GR00T
# Notes:
#   - Checkpoints are excluded by Dockerfile.dockerignore and should be mounted
#     at runtime unless you intentionally remove that ignore rule.
# -----------------------------------------------------------------------------
WORKDIR /workspace/Isaac-GR00T
COPY repos/Isaac-GR00T/pyproject.toml repos/Isaac-GR00T/uv.lock ./

RUN <<'EOF'
set -e
# This checkout does not contain the optional groot_infra workspace package.
# Remove that workspace stanza before resolving the GR00T environment.
sed -i '/^\[tool\.uv\.workspace\]/d; /^members = \["groot_infra"\]/d' pyproject.toml
UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv sync \
    --no-install-workspace \
    --extra dev \
    --no-cache
EOF

COPY repos/Isaac-GR00T/ /workspace/Isaac-GR00T/

RUN <<'EOF'
set -e
uv pip install --python /workspace/Isaac-GR00T/.venv/bin/python -e . --no-deps
EOF

ENV GROOT_VENV=/workspace/Isaac-GR00T/.venv

# -----------------------------------------------------------------------------
# Headless EGL/Vulkan setup for GR00T examples and Isaac rendering
# Source:
#   repos/Isaac-GR00T/docker/Dockerfile plus Unitree Isaac Vulkan variables.
# What this section provides:
#   - NVIDIA EGL vendor file
#   - NVIDIA Vulkan ICD file
#   - MuJoCo/PyOpenGL EGL defaults used by GR00T-side tooling
# -----------------------------------------------------------------------------
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /usr/share/vulkan/icd.d /etc/vulkan/icd.d && \
    cat >/usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

RUN cat >/usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.2.140"
    }
}
EOF

RUN ln -sf /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# -----------------------------------------------------------------------------
# GR00T torchcodec fallback for aarch64
# Source:
#   repos/Isaac-GR00T/docker/Dockerfile
# What this section does:
#   - On aarch64 only, builds torchcodec from source.
#   - On x86_64 Vast machines, this is skipped.
# -----------------------------------------------------------------------------
ARG TORCHCODEC_GIT_URL=https://github.com/pytorch/torchcodec.git
ARG TORCHCODEC_GIT_REF=v0.10.0
RUN ARCH="$(uname -m)" && \
    if [ "${ARCH}" = "aarch64" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
            libavutil-dev libswresample-dev libswscale-dev pybind11-dev \
        && rm -rf /var/lib/apt/lists/* \
        && uv pip install --python /workspace/Isaac-GR00T/.venv/bin/python setuptools \
        && git clone --depth 1 --branch "${TORCHCODEC_GIT_REF}" "${TORCHCODEC_GIT_URL}" /tmp/torchcodec \
        && cd /tmp/torchcodec \
        && I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 uv pip install --python /workspace/Isaac-GR00T/.venv/bin/python . --no-build-isolation \
        && rm -rf /tmp/torchcodec; \
    else \
        echo "Skipping torchcodec source build for ARCH='${ARCH}'."; \
    fi

# -----------------------------------------------------------------------------
# Isaac/Unitree Conda environment
# Source:
#   repos/unitree_sim_isaaclab/Dockerfile
# What this section installs:
#   - Miniconda
#   - unitree_sim_env with Python 3.11
#   - PyTorch 2.7 CUDA 12.6 wheels
#   - Isaac Sim 5.1.0 pip package
#   - IsaacLab
#   - CycloneDDS
#   - unitree_sdk2_python
#   - local repos/unitree_sim_isaaclab checkout
# -----------------------------------------------------------------------------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n unitree_sim_env python=3.11 -y && \
    conda clean -afy

RUN conda run -n unitree_sim_env conda install -y -c conda-forge "libgcc-ng>=12" "libstdcxx-ng>=12" && \
    conda clean -afy

RUN conda run -n unitree_sim_env pip install --upgrade pip && \
    conda run -n unitree_sim_env pip install \
        torch==2.7.0 \
        torchvision==0.22.0 \
        torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu126 && \
    conda run -n unitree_sim_env pip install \
        "isaacsim[all,extscache]==5.1.0" \
        --extra-index-url https://pypi.nvidia.com

WORKDIR /home/code

RUN git clone https://github.com/isaac-sim/IsaacLab.git /home/code/IsaacLab && \
    cd /home/code/IsaacLab && \
    conda run -n unitree_sim_env ./isaaclab.sh --install

RUN conda run -n unitree_sim_env pip install \
        torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu126

RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x /cyclonedds && \
    cd /cyclonedds && \
    mkdir build install && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    cmake --build . --target install

RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python /home/code/unitree_sdk2_python && \
    cd /home/code/unitree_sdk2_python && \
    conda run -n unitree_sim_env pip install -e .

COPY repos/unitree_sim_isaaclab/ /home/code/unitree_sim_isaaclab/

RUN cd /home/code/unitree_sim_isaaclab && \
    git submodule update --init --depth 1 || true && \
    cd teleimager && \
    conda run -n unitree_sim_env pip install -e . && \
    cd .. && \
    conda run -n unitree_sim_env pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Runtime helpers
# Source:
#   New wrappers for the single-container Vast.AI workflow.
# What this section adds:
#   - start_groot_server.sh: starts the GR00T ZeroMQ PolicyServer
#   - start_unitree_sim.sh: starts Isaac/Unitree with GR00T action source
# -----------------------------------------------------------------------------
COPY docker/1-docker-setup/scripts/start_groot_server.sh /usr/local/bin/start_groot_server.sh
COPY docker/1-docker-setup/scripts/start_unitree_sim.sh /usr/local/bin/start_unitree_sim.sh
RUN chmod +x /usr/local/bin/start_groot_server.sh /usr/local/bin/start_unitree_sim.sh

RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> /root/.bashrc && \
    echo 'conda activate unitree_sim_env' >> /root/.bashrc && \
    echo 'export OMNI_KIT_ALLOW_ROOT=1' >> /root/.bashrc && \
    echo 'export OMNI_KIT_DISABLE_STARTUP=1' >> /root/.bashrc && \
    echo 'export CYCLONEDDS_HOME=/cyclonedds/install' >> /root/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
