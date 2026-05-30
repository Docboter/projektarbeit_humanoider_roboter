# syntax=docker/dockerfile:1
# Sim-Client-Container: Isaac Lab + schlanker GR00T-Client
#
# Build (vom Repo-Root):
#   docker build --platform linux/amd64 -t lucam03/projekt-humanoider-roboter-sim:latest Simulation/
#
# Test (interaktiv, headless):
#   docker run -it --rm --gpus all --network=host \
#     -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
#     -v $(pwd)/data:/data \
#     lucam03/projekt-humanoider-roboter-sim:latest bash
#
# Voraussetzungen:
#   docker login nvcr.io   (Username: $oauthtoken  Password: <NGC-API-Key>)
#   docker login           (Docker Hub, für den Push)
#
# Hinweise:
#   - Isaac Sim / Isaac Lab hat inkompatible Dependency-Stacks mit GR00T.
#     Dieser Container enthält NUR den schlanken ZMQ-Client (pyzmq + msgpack).
#     Die schwere GR00T-Policy läuft im separaten Training-Container.
#   - Das Basis-Image ist groß (~20-30 GB). Genug Plattenplatz einplanen (50-80 GB).
#   - GPU-Anforderung: RT-Cores! A100/H100 nicht unterstützt für Kamera-Rendering.
#     Auf GWDG: jupyter-Partition (Quadro RTX 5000).
#     Siehe Simulation/SIM_GPU_COMPATIBILITY.md für Details.

FROM nvcr.io/nvidia/isaac-lab:2.3.2

SHELL ["/bin/bash", "-c"]

# EULA/Privacy non-interaktiv akzeptieren (sonst hängt der erste Isaac-Sim-Start)
# EULA/Privacy non-interaktiv akzeptieren. Isaac Sim schreibt Cache-Daten;
# Standard-Pfade auf /tmp umlenken, damit sie im Read-Only-SIF nicht fehlen.
# Auf KISSKI: per --bind auf /mnt/vast-kisski/projects/kisski-humrob/isaac-cache
# mounten (siehe kisski_sim_submit.sh).
ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    NVIDIA_DRIVER_CAPABILITIES=all \
    OMNI_KIT_ACCEPT_EULA=yes \
    OMNI_CACHE=/tmp/omni_cache \
    NVCACHE_PATH=/tmp/nvcache

# Schlanke Client-Dependencies in Isaacs gebündeltes Python installieren.
# KEIN torch / gr00t hier — nur das ZMQ/msgpack-Protokoll.
# ${ISAACLAB_PATH}/isaaclab.sh -p ist der offizielle Weg, Isaacs Python anzusprechen.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
        pyzmq \
        msgpack \
        opencv-python-headless \
        imageio[ffmpeg]

# Eigener Sim-Code und vendorter Client
# Struktur (vgl. ISAAC_LAB_SIM_PLAN.md Abschnitt 8):
#   g1_dex3_sim/
#     client.py                  — vendorter MsgSerializer + PolicyClient (kein gr00t-Import)
#     g1_dex3_cfg.py             — G1_DEX3_CFG Articulation-Config (Actuators, Gains, Limits)
#     g1_dex3_blockstack_env.py  — DirectRLEnv: Szene, Kameras, Success, reset/step
#     run_g1_dex3_sim_eval.py    — Client: Rollout-Loop, Metriken, Video-Aufzeichnung
COPY g1_dex3_sim/ /workspace/g1_dex3_sim/

# Assets-Verzeichnis anlegen (USD-Dateien kommen per Volume-Mount oder manuell rein)
# g1_dex3.usd muss aus dem Unitree-URDF konvertiert werden (siehe ISAAC_LAB_SIM_PLAN.md)
RUN mkdir -p /workspace/assets

WORKDIR /workspace

# Kein ENTRYPOINT: der SLURM-Job ruft
#   ${ISAACLAB_PATH}/isaaclab.sh -p g1_dex3_sim/run_g1_dex3_sim_eval.py --headless ...
# explizit auf (siehe Simulation/kisski_sim_submit.sh).
