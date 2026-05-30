# Sim-Container auf KISSKI starten — Schritt-für-Schritt-Anleitung

Ziel: Den Isaac-Lab-Sim-Client-Container interaktiv in der **JupyterHPC-Desktop-Umgebung**
auf KISSKI testen (Phase A des Plans: leere Szene rendert RGB-Bild headless).

GPU: **Quadro RTX 5000** (`jupyter`-Partition) — einzige GWDG-GPU mit RT-Cores.
Hintergrund: [SIM_GPU_COMPATIBILITY.md](SIM_GPU_COMPATIBILITY.md)

---

## Voraussetzung: Image ist auf Docker Hub gepusht

Prüfen auf dem Laptop:
```powershell
docker manifest inspect lucam03/projekt-humanoider-roboter-sim:latest
```
Wenn das Ergebnis JSON-Manifest zeigt → fertig, weiter mit Schritt 1.

---

## Schritt 1 — SIF auf KISSKI ziehen (Login-Node, einmalig)

SSH auf den GPU-Login-Node:
```bash
ssh <username>@glogin-gpu.hpc.gwdg.de
```

Cache-Verzeichnisse anlegen (verhindert Quota-Probleme beim Ziehen):
```bash
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer_cache
export APPTAINER_TMPDIR=/scratch/$USER/.apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
```

Diese Exports dauerhaft in `~/.bashrc` eintragen (damit sie nicht jedes Mal eingegeben werden müssen):
```bash
echo 'export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer_cache' >> ~/.bashrc
echo 'export APPTAINER_TMPDIR=/scratch/$USER/.apptainer_tmp'     >> ~/.bashrc
```

Apptainer laden und SIF ziehen (~20–30 GB, dauert je nach Netz 15–45 Minuten):
```bash
module load apptainer

apptainer pull \
  /user/luca.muecke/u28320/.project/dir.project/images/projekt-humanoider-roboter-sim.sif \
  docker://lucam03/projekt-humanoider-roboter-sim:latest
```

Fertig-Check:
```bash
ls -lh /user/luca.muecke/u28320/.project/dir.project/images/projekt-humanoider-roboter-sim.sif
# → Datei sollte >20 GB groß sein
```

---

## Schritt 2 — JupyterHPC-Desktop starten

1. Im Browser öffnen: **https://jupyter.hpc.gwdg.de**
2. Einloggen mit GWDG-Account.
3. Oben im Launcher: **"Desktop"** auswählen (nicht "Notebook"!).
4. In den Ressourcen-Einstellungen:
   - **GPU:** `1` (bekommt automatisch Quadro RTX 5000)
   - **CPUs:** `8–16`
   - **RAM:** `16–32 GB`
   - **Laufzeit:** `2–4 Stunden` (für erste Tests ausreichend)
5. Session starten → Desktop-Umgebung öffnet im Browser.

> **Hinweis:** Die `jupyter`-Partition ist eine geteilte Interaktiv-Partition.
> Wartezeiten von einigen Minuten sind normal.

---

## Schritt 3 — Terminal im Desktop öffnen

Im Desktop-Fenster: Rechtsklick auf den Desktop → **"Open Terminal"**
(oder über das Anwendungsmenü oben links).

GPU prüfen:
```bash
nvidia-smi
# → Sollte "Quadro RTX 5000" zeigen
```

---

## Schritt 4 — Isaac-Sim-Cache-Verzeichnisse anlegen

Isaac Sim schreibt beim Start viel in Cache-Verzeichnisse.
Das SIF ist read-only → Caches müssen auf beschreibbaren Projektspeicher zeigen
(analog zu `DATA_DIR` im Training):

```bash
ISAAC_CACHE=/mnt/vast-kisski/projects/kisski-humrob/isaac-cache
mkdir -p $ISAAC_CACHE/{kit,kit_data,ov,pip,nv,logs,cameras_output}
```

---

## Schritt 5 — Phase-A-Test: Container starten

### 5a) Einfacher Smoke-Test (Python läuft, kein Rendering)

```bash
module load apptainer

SIM_SIF=/user/luca.muecke/u28320/.project/dir.project/images/projekt-humanoider-roboter-sim.sif
ISAAC_CACHE=/mnt/vast-kisski/projects/kisski-humrob/isaac-cache

apptainer exec --nv \
  --bind $ISAAC_CACHE/kit:/isaac-sim/kit/cache \
  --bind $ISAAC_CACHE/kit_data:/isaac-sim/kit/data \
  --bind $ISAAC_CACHE/ov:/root/.cache/ov \
  --bind $ISAAC_CACHE/nv:/root/.cache/nvidia \
  --bind $ISAAC_CACHE/pip:/root/.cache/pip \
  --bind $ISAAC_CACHE/logs:/root/.local/share/ov/data/Kit/logs \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  "$SIM_SIF" \
  bash -c '${ISAACLAB_PATH}/isaaclab.sh -p -c "print(1)"'
```

**Erwartet:** Isaac Sim initialisiert sich (~30–60 s), dann `1` in der Ausgabe.
Warnungen über DLSS/Denoising sind ok — RT-Cores sind vorhanden, aber die Turing-GPU
liegt unter NVIDIAs Empfehlung für aktuelle Isaac-Sim-Versionen.

### 5b) Kamera-Rendering-Test (Phase A, eigentliches Ziel)

Erst wenn 5a funktioniert:

```bash
apptainer exec --nv \
  --bind $ISAAC_CACHE/kit:/isaac-sim/kit/cache \
  --bind $ISAAC_CACHE/kit_data:/isaac-sim/kit/data \
  --bind $ISAAC_CACHE/ov:/root/.cache/ov \
  --bind $ISAAC_CACHE/nv:/root/.cache/nvidia \
  --bind $ISAAC_CACHE/pip:/root/.cache/pip \
  --bind $ISAAC_CACHE/logs:/root/.local/share/ov/data/Kit/logs \
  --bind $ISAAC_CACHE/cameras_output:/workspace/isaaclab/scripts/demos/sensors/output \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  "$SIM_SIF" \
  bash -c '${ISAACLAB_PATH}/isaaclab.sh -p \
    ${ISAACLAB_PATH}/scripts/demos/sensors/cameras.py \
    --headless --enable_cameras --num_envs 1'
```

Bilder prüfen nach dem Lauf:
```bash
ls $ISAAC_CACHE/cameras_output/
```

> **Hinweis:** `--num_envs 1` ist wichtig — der Default (4 Envs) erzeugt ein Terrain mit
> ~3,9M Faces. Der RayCasterCamera-BVH-Build dafür dauert 30+ Minuten (single-threaded CPU).
> Mit einem Env sinkt die Mesh-Größe auf ~¼.

### 5c) Kamera-Rendering-Test mit GUI (optional, nur im JupyterHPC-Desktop)

Nur im Desktop-Modus sinnvoll (`echo $DISPLAY` muss einen Wert zeigen):

```bash
apptainer exec --nv \
  --bind /tmp/.X11-unix:/tmp/.X11-unix \
  --bind $ISAAC_CACHE/kit:/isaac-sim/kit/cache \
  --bind $ISAAC_CACHE/kit_data:/isaac-sim/kit/data \
  --bind $ISAAC_CACHE/ov:/root/.cache/ov \
  --bind $ISAAC_CACHE/nv:/root/.cache/nvidia \
  --bind $ISAAC_CACHE/pip:/root/.cache/pip \
  --bind $ISAAC_CACHE/logs:/root/.local/share/ov/data/Kit/logs \
  --bind $ISAAC_CACHE/cameras_output:/workspace/isaaclab/scripts/demos/sensors/output \
  --env DISPLAY=$DISPLAY \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  "$SIM_SIF" \
  bash -c '${ISAACLAB_PATH}/isaaclab.sh -p \
    ${ISAACLAB_PATH}/scripts/demos/sensors/cameras.py \
    --enable_cameras --num_envs 1'
```

> **Hinweis:** Der Desktop nutzt VirtualGL (`libdlfaker.so`/`libvglfaker.so`), das im Container
> nicht verfügbar ist — Isaac Sim rendert trotzdem über Vulkan direkt. Falls kein Fenster
> aufgeht, zurück zu 5b (headless).

**Prozess beenden** (in einem zweiten Terminal):
```bash
pkill -9 -f "kit.sh"
# falls das nicht reicht:
kill -9 $(pgrep -f "apptainer|kit\.sh|omni\.kit|python3.*isaac") 2>/dev/null
```

**Erwartet:** Kein `createDLSSContext`-Fehler, Kamera-Frames werden erzeugt.
Falls dieser Fehler auftritt → RTX 5000 zu alt für Isaac Lab 2.3.2, dann eine
ältere Version testen (siehe Abschnitt „Fallback" unten).

---

## Schritt 6 — Interaktive Shell im Container (optional, für Debugging)

```bash
apptainer shell --nv \
  --bind $ISAAC_CACHE/kit:/isaac-sim/kit/cache \
  --bind $ISAAC_CACHE/kit_data:/isaac-sim/kit/data \
  --bind $ISAAC_CACHE/ov:/root/.cache/ov \
  --bind $ISAAC_CACHE/nv:/root/.cache/nvidia \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  "$SIM_SIF"

# Im Container-Shell:
echo $ISAACLAB_PATH          # Pfad zum Isaac-Lab-Root
${ISAACLAB_PATH}/isaaclab.sh -p -c "import isaaclab; print('ok')"
ls /workspace/g1_dex3_sim/   # Eigener Sim-Code
```

---

## Bekannte Probleme & Lösungen

| Problem | Lösung |
|---|---|
| `createDLSSContext error` | RTX 5000 zu alt für isaac-lab:2.3.2 → ältere Version probieren |
| `No space left` beim Pull | `APPTAINER_CACHEDIR` auf `/scratch` zeigt nicht — Schritt 1 wiederholen |
| Session läuft ab | Laufzeit beim Start erhöhen; SIF bleibt im `.project`-Storage erhalten |
| `Kit cache` Fehler | Bind-Mounts prüfen; `ISAAC_CACHE`-Verzeichnisse müssen existieren |
| Lange Startzeit (~60 s) | Normal — Isaac Sim kompiliert Shader beim ersten Start; danach gecacht |

---

## Fallback: Ältere Isaac-Lab-Version

Falls isaac-lab:2.3.2 auf der RTX 5000 Rendering-Fehler hat:

```bash
# Auf dem Login-Node eine ältere Version ziehen:
apptainer pull \
  /user/luca.muecke/u28320/.project/dir.project/images/projekt-humanoider-roboter-sim-2.1.sif \
  docker://nvcr.io/nvidia/isaac-lab:2.1.0

# Dockerfile anpassen: FROM nvcr.io/nvidia/isaac-lab:2.1.0
# Image neu bauen und pushen (Simulation/update_sim_image.ps1)
```

---

## Nächste Phase nach erfolgreichem Phase-A-Test

Wenn `print(1)` und Kamera-Test laufen:

1. **Open-Loop-Eval** auf dem GR00T-Checkpoint (A100, `kisski`-Partition):
   ```bash
   sbatch Training/kisski_submit.sh   # falls noch nicht gemacht
   ```
2. **Vollständigen Eval-Job** starten (beide Container):
   ```bash
   sbatch Simulation/kisski_sim_submit.sh
   ```
3. Ergebnisse abholen:
   ```bash
   rsync -avz <username>@transfer.hpc.gwdg.de:/mnt/vast-kisski/projects/kisski-humrob/data/sim_videos/ ./sim_videos/
   rsync -avz <username>@transfer.hpc.gwdg.de:/mnt/vast-kisski/projects/kisski-humrob/data/sim_results.json ./
   ```
