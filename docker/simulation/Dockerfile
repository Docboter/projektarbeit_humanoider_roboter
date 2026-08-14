# syntax=docker/dockerfile:1

ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
FROM ${ISAAC_SIM_IMAGE}

USER root
SHELL ["/bin/bash", "-c"]

ARG ISAAC_LAB_REF=80094be3245aa5c8376a7464d29cb4412ea518f5

ENV DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    OMNI_KIT_ALLOW_ROOT=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
    CYCLONEDDS_HOME=/opt/cyclonedds/install \
    PYTHONUNBUFFERED=1

# Only dependencies that Unitree adds on top of the official Isaac Sim image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        git-lfs \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab \
    && git -C /opt/IsaacLab checkout "${ISAAC_LAB_REF}" \
    && ln -s /isaac-sim /opt/IsaacLab/_isaac_sim \
    && cd /opt/IsaacLab \
    && ./isaaclab.sh --install

RUN git clone --branch releases/0.10.x --depth 1 \
        https://github.com/eclipse-cyclonedds/cyclonedds.git /opt/cyclonedds \
    && cmake -S /opt/cyclonedds -B /opt/cyclonedds/build \
        -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds/install \
    && cmake --build /opt/cyclonedds/build --target install --parallel

RUN git clone --depth 1 \
        https://github.com/unitreerobotics/unitree_sdk2_python.git /opt/unitree_sdk2_python \
    && /isaac-sim/python.sh -m pip install --no-cache-dir -e /opt/unitree_sdk2_python

WORKDIR /workspace/unitree_sim_isaaclab
COPY repos/unitree_sim_isaaclab/requirements.txt ./requirements.txt
COPY repos/unitree_sim_isaaclab/teleimager ./teleimager
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r requirements.txt \
    && /isaac-sim/python.sh -m pip install --no-cache-dir -e teleimager

COPY repos/unitree_sim_isaaclab/ ./
COPY docker/simulation/start_simulation.sh /usr/local/bin/start-unitree-simulation
RUN chmod +x /usr/local/bin/start-unitree-simulation

ENTRYPOINT ["/usr/local/bin/start-unitree-simulation"]
