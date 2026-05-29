# Zweiten Docker-Container für die Sim bauen — Isaac Lab + GR00T-Client

Begleitdokument zu [ISAAC_LAB_SIM_PLAN.md](ISAAC_LAB_SIM_PLAN.md).
Beschreibt, wie der **zweite Container** (Isaac-Lab-Sim-Client) lokal auf dem Laptop
gebaut, nach Docker Hub gepusht und auf KISSKI als SIF gezogen wird — analog zum
bestehenden Training-Image `lucam03/projekt-humanoider-roboter`.

---

## 0. Warum überhaupt ein zweiter Container?

Aus dem Plan (Abschnitt 1): Isaac Lab und GR00T haben **inkompatible Dependency-Stacks**
(Isaac Sim bringt ein eigenes, gebündeltes Python + PyTorch + USD mit). Deshalb:

```
┌──────────────────────────────┐    ZMQ :5555     ┌──────────────────────────────┐
│  Container 1 (existiert)      │ ◀── obs ──────── │  Container 2 (NEU, dieses Doc)│
│  GR00T-Policy-Server          │ ─── action ────▶ │  Isaac-Lab-Sim-Client         │
│  lucam03/…-roboter (~14 GB)   │                  │  lucam03/…-roboter-sim        │
│  euer Checkpoint, GPU         │                  │  Isaac Sim 4.x, headless EGL  │
└──────────────────────────────┘                  └──────────────────────────────┘
```

Beide laufen später im **selben SLURM-Job auf demselben Node**, Server im Hintergrund,
Client im Vordergrund, Kommunikation über `localhost:5555`.

> **Wichtig:** Den GR00T-Server bauen wir **nicht** neu — das ist das vorhandene
> Training-Image. Dieses Dokument betrifft ausschließlich den **Sim-Client-Container**.

---

## 1. Strategie-Entscheidung: nicht „from scratch“, sondern auf NVIDIAs Image aufsetzen

Isaac Sim/Isaac Lab baut man **nicht selbst** zusammen. NVIDIA liefert fertige Images
auf NGC (`nvcr.io`). Wir setzen mit `FROM` darauf auf und legen nur unseren eigenen
Code + den schlanken GR00T-**Client** obendrauf.

| Variante | Aufwand | Empfehlung |
|---|---|---|
| `FROM nvcr.io/nvidia/isaac-lab:2.3.2` + eigene Layer | gering | ✅ **so machen wir es** |
| Isaac Lab aus Quellen / pip in eigenes CUDA-Image | sehr hoch (Vulkan/Treiber-Hölle) | ❌ nur Notnagel |

Stand Mai 2026 ist `2.3.2` das aktuellste Major-Release mit vorgebautem Image
(es gibt Images nur für Majors: 2.0.0, 2.1.0, 2.3.2 — **nicht** für Minor-Versionen).
Das Isaac-Lab-Image enthält bereits Isaac Sim, CUDA, USD, PyTorch und alle Render-Libs.

**Größenwarnung:** Das Image ist groß (Isaac-Sim-Basis ~20–30 GB). Plant entsprechend
Download- (von NGC) und Upload-Bandbreite (nach Docker Hub) ein. Das fertige SIF auf
KISSKI wird deutlich größer als die ~14 GB des Trainings-SIF.

---

## 2. Voraussetzungen auf dem Laptop (einmalig)

1. **Docker Desktop** läuft (wie beim Training-Image).
2. **NGC-Zugang**, um die Basis von `nvcr.io` zu ziehen. Kostenloser NGC-Account →
   API-Key erzeugen (https://ngc.nvidia.com → Setup → Generate API Key), dann:
   ```bash
   docker login nvcr.io
   # Username:  $oauthtoken      (genau so, mit Dollarzeichen)
   # Password:  <euer NGC-API-Key>
   ```
3. **Docker-Hub-Login** (für den Push, wie gehabt):
   ```bash
   docker login
   ```
4. Genug Plattenplatz: rechnet mit **50–80 GB** frei (Basis + Layer + Cache).

> **Lizenz-Hinweis:** NVIDIAs Isaac-Sim-Image unterliegt der NVIDIA-Software-Lizenz
> (EULA). Das Weiterverteilen über euer öffentliches Docker-Hub-Repo ist für interne
> Projektnutzung in der Regel ok, aber haltet das Repo im Zweifel **privat**
> (`docker login` auf KISSKI ist dann nötig — siehe Abschnitt 6).

---

## 3. Was kommt in den Container?

Auf die Isaac-Lab-Basis legen wir nur drei Dinge:

1. **Den schlanken GR00T-Client.** Der Client (`PolicyClient` aus
   [gr00t/policy/server_client.py](../app/Groot-1.6/gr00t/policy/server_client.py))
   braucht zur Laufzeit nur `msgpack`, `numpy`, `pyzmq` — die schwere ML-Seite läuft
   im Server-Container. **Nicht** das ganze `gr00t`-Paket in Isaacs Python installieren
   (PyTorch-Konflikt!). Zwei saubere Optionen:
   - **A (empfohlen): Client vendorn.** Nur `MsgSerializer` + eine minimale
     `get_action()`-Funktion kopieren (das ZMQ/msgpack-Protokoll ist klein und stabil).
     Keine `gr00t`-Importe → kein Dependency-Clash.
   - **B:** `pip install -e . --no-deps` des gr00t-Pakets in Isaacs Python und nur
     `msgpack pyzmq` nachinstallieren. Riskanter, weil `PolicyClient` `gr00t.data.types`
     / `gr00t.data.utils` importiert, die transitiv mehr ziehen können.
2. **Den Sim-Code** aus dem Plan (Abschnitt 8): `g1_dex3_blockstack_env.py`,
   `g1_dex3_cfg.py`, `run_g1_dex3_sim_eval.py`.
3. **Die Assets**: `g1_dex3.usd` etc. (oder zur Laufzeit per Bind-Mount aus `/data`).

> **Reihenfolge der Arbeit:** Zuerst funktioniert der Container „leer“ (Phase A des
> Plans: leere Szene rendert ein RGB-Bild headless). Sim-Code und Assets kommen iterativ
> dazu. Für den ersten Build reicht Isaac-Lab-Basis + Client-Deps.

---

## 4. Dockerfile

Anlegen unter `Simulation/Dockerfile` (Build-Kontext = `Simulation/`).

```dockerfile
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

FROM nvcr.io/nvidia/isaac-lab:2.3.2

SHELL ["/bin/bash", "-c"]

# EULA/Privacy non-interaktiv akzeptieren (sonst hängt der erste Start)
ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    NVIDIA_DRIVER_CAPABILITIES=all

# --- Schlanke Client-Dependencies in Isaacs gebündeltes Python ---
# Isaac Sim bringt sein eigenes Python mit; pip darüber installieren.
# KEIN torch/gr00t hier — nur das ZMQ/msgpack-Protokoll.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
        pyzmq msgpack

# --- Eigener Sim-Code + Client ---
# Erwartete Struktur (vgl. Plan Abschnitt 8):
#   Simulation/g1_dex3_sim/
#     ├── client.py                 # gevendorter MsgSerializer + get_action()
#     ├── g1_dex3_cfg.py
#     ├── g1_dex3_blockstack_env.py
#     ├── run_g1_dex3_sim_eval.py
#     └── assets/  (optional; sonst per Volume aus /data)
COPY g1_dex3_sim/ /workspace/g1_dex3_sim/

# Headless-Rendering: EGL ist im Isaac-Lab-Image bereits korrekt verdrahtet.
# Auf dem Cluster liefert der NVIDIA-Treiber via `apptainer --nv` die ICDs.

WORKDIR /workspace
# Kein ENTRYPOINT: der SLURM-Job ruft `isaaclab.sh -p run_g1_dex3_sim_eval.py …`
# bzw. den Client explizit auf (siehe Abschnitt 7).
```

**Hinweise:**
- `${ISAACLAB_PATH}/isaaclab.sh -p` ist der von Isaac Lab vorgesehene Weg, das
  gebündelte Python anzusprechen. (`ISAACLAB_PATH` ist im Basis-Image gesetzt.)
- Wir setzen **keinen** ENTRYPOINT, weil wir im SLURM-Job sehr gezielt Befehle absetzen.
- Vendor-Client (`client.py`): kopiert `MsgSerializer` und implementiert ein minimales
  `PolicyClient.get_action(obs) -> np.ndarray (16, 28)` per `zmq.REQ`-Socket auf
  `tcp://localhost:5555`. Vorlage: die Klassen `MsgSerializer`/`PolicyClient` in
  [gr00t/policy/server_client.py](../app/Groot-1.6/gr00t/policy/server_client.py).

---

## 5. Bauen und pushen (lokal, Laptop)

Analog zu `Training/update_image.ps1`. Erst der direkte Weg:

```bash
# vom Repo-Root
docker build --platform linux/amd64 \
  -t lucam03/projekt-humanoider-roboter-sim:latest \
  -t lucam03/projekt-humanoider-roboter-sim:$(date +%Y%m%d-%H%M%S) \
  Simulation/

docker push lucam03/projekt-humanoider-roboter-sim:latest
docker push lucam03/projekt-humanoider-roboter-sim:<datum-tag>
```

> Tipp: Den `--platform linux/amd64`-Flag immer setzen (Apple-Silicon-Laptops bauen
> sonst arm64 — auf KISSKI unbrauchbar). Das macht auch das vorhandene `update_image.ps1`.

Optional könnt ihr `Training/update_image.ps1` als Vorlage für ein
`Simulation/update_sim_image.ps1` kopieren (Image-Name + Build-Kontext anpassen,
den GR00T-Commit-Check braucht ihr hier nicht).

---

## 6. SIF auf KISSKI ziehen (einmalig, Login-Node)

Genau wie beim Training-Image — Compute-Nodes haben kein Internet, also auf dem
Login-Node ziehen:

```bash
module load apptainer

# Falls euer Docker-Hub-Repo PRIVAT ist, vorher anmelden:
#   apptainer remote login --username <dockerhub-user> docker://docker.io

apptainer pull \
  ~/.project/dir.project/images/projekt-humanoider-roboter-sim.sif \
  docker://lucam03/projekt-humanoider-roboter-sim:latest
```

**Cache/TMP auf scratch legen** (das Image ist groß, `$HOME`-Quota reicht evtl. nicht):

```bash
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer_cache
export APPTAINER_TMPDIR=/scratch/$USER/.apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
```

---

## 7. Beide Container in einem SLURM-Job starten

Skizze (ein eigenes `kisski_sim_submit.sh`, abgeleitet von `kisski_submit.sh`).
Wichtig: **GR00T-Server im Hintergrund**, kurz warten, dann **Sim-Client** im
selben Job/Node.

```bash
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 04:00:00

module load apptainer
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer_cache

SERVER_SIF=~/.project/dir.project/images/projekt-humanoider-roboter.sif
SIM_SIF=~/.project/dir.project/images/projekt-humanoider-roboter-sim.sif

# Schreibbare Cache-Dirs für Isaac Sim (Container-FS ist read-only!)
export ISAAC_CACHE=/scratch/$USER/isaac-cache
mkdir -p $ISAAC_CACHE/{kit,ov,pip,nv}

# --- 1) GR00T-Policy-Server im Hintergrund ---
apptainer exec --nv \
  --bind /scratch/$USER/data:/data \
  "$SERVER_SIF" \
  python /app/Groot-1.6/gr00t/eval/run_gr00t_server.py \
    --model-path /data/g1_dex3_finetune \
    --embodiment-tag new_embodiment \
    --embodiment-config-module examples.G1_DEX3.g1_dex3_config \
    --port 5555 &
SERVER_PID=$!

# Auf Server-Bereitschaft warten (Port offen)
sleep 60   # besser: aktiv auf tcp:5555 pollen

# --- 2) Isaac-Lab-Sim-Client (headless EGL) ---
apptainer exec --nv \
  --bind /scratch/$USER/data:/data \
  --bind $ISAAC_CACHE/kit:/isaac-sim/kit/cache \
  --bind $ISAAC_CACHE/ov:/root/.cache/ov \
  --bind $ISAAC_CACHE/nv:/root/.cache/nvidia \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y \
  "$SIM_SIF" \
  bash -lc '${ISAACLAB_PATH}/isaaclab.sh -p \
            /workspace/g1_dex3_sim/run_g1_dex3_sim_eval.py \
            --headless --server tcp://localhost:5555'

kill $SERVER_PID 2>/dev/null || true
```

**Stolperfallen Headless/Apptainer:**
- `--nv` ist Pflicht (reicht NVIDIA-Treiber + EGL/Vulkan-ICDs in den Container).
- Isaac Sim schreibt **viel** in Cache-Verzeichnisse. Da der SIF read-only ist,
  müssen `kit`/`ov`/`nv`-Caches per `--bind` auf beschreibbares `scratch` zeigen,
  sonst crasht der erste Start. Die exakten Pfade ggf. im interaktiven Shell-Run
  verifizieren (`apptainer shell …` und schauen, wohin Kit schreiben will).
- `--headless` an das Sim-Skript geben (kein Display auf dem Node).
- Den Plan-Hinweis beachten: `check_sim_eval_ready.py` zeigt das nötige EGL-Setup —
  bei NVIDIAs Image ist das aber schon vorkonfiguriert; meist genügt `--nv`.

---

## 8. Reihenfolge / Abnahme (passend zu Plan-Phasen A–G)

1. **Image bauen & SIF ziehen** → `apptainer exec … bash -lc '${ISAACLAB_PATH}/isaaclab.sh -p -c "print(1)"'` läuft. *(Container ok)*
2. **Phase A:** leeres Headless-Skript rendert ein RGB-Bild auf dem A100. *(EGL/Bind-Mounts ok)*
3. **Client-Roundtrip (Phase D):** Server + Client im selben Job; `get_action(dummy_obs)` liefert `(16, 28)`. *(ZMQ-Verbindung ok)*
4. Danach iterativ Asset/Kameras/Control-Loop wie im Plan.

> **Vorab (Plan Abschnitt 10):** Erst die Open-Loop-Eval auf dem Checkpoint laufen
> lassen. Lohnt sich der ganze Sim-Aufwand nicht, wenn die per-Joint-MSE schon schlecht ist.

---

## Quellen

- [Isaac Lab — Docker Guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html)
- [Isaac Lab — Cluster Guide (Apptainer/Singularity)](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html)
- [Isaac Sim — Container Installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
- [j3soon/singularity-isaac-sim — inoffizielle HPC-Anleitung](https://github.com/j3soon/singularity-isaac-sim)
