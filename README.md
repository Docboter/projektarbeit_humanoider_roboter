# Projektarbeit Humanoider Roboter
## GR00T N1.6 Fine-tuning — Unitree G1 mit DEX3-Hand

Dieses Repository dokumentiert das Fine-tuning von NVIDIAs **GR00T N1.6** Vision-Language-Action-Modell auf den Unitree G1 Roboter mit DEX3-Hand für die Aufgabe Block-Stacking.

Die Trainingsumgebung läuft **vollständig autonom in einem Container**: Container starten → Training läuft. Derselbe Container läuft lokal, auf Cloud-GPU-Plattformen wie [vast.ai](https://vast.ai) und auf dem **KISSKI HPC-Cluster** (GWDG Göttingen) — alle Schritte (Daten-Download, Konvertierung, Training) passieren im Container.

> **Speicher-Modell:** Es gibt **keinen persistenten Storage auf dem Host**. Daten, Checkpoints und Logs leben ausschließlich im Container-Filesystem. Auf vast.ai entspricht ein Container genau einer Instanz — wird die Instanz/Container zerstört, ist alles weg. Auf KISSKI wird `/mnt/vast-kisski/projects/kisski-humrob/data` (GWDG VAST-Projekt-Storage) als `/data` in den Container gemountet. Das Container-Filesystem ist die einzige Wahrheit.

---

## Inhaltsverzeichnis

1. [Schnellstart](#1-schnellstart)
2. [Voraussetzungen](#2-voraussetzungen)
3. [Cloud-Training auf vast.ai](#3-cloud-training-auf-vastai)
4. [HPC-Training auf KISSKI](#4-hpc-training-auf-kisski)
5. [Lokales Training](#5-lokales-training)
6. [Konfiguration über Env-Vars](#6-konfiguration-über-env-vars)
7. [Daten aus dem Container holen](#7-daten-aus-dem-container-holen)
8. [Training beobachten](#8-training-beobachten)
9. [Projektstruktur](#9-projektstruktur)
10. [Häufige Probleme](#10-häufige-probleme)
11. [Train-Test-Split](#11-train-test-split)

---

## 1. Schnellstart

**Auf KISSKI (HPC-Cluster, empfohlen für langes Training):**
```bash
# Einmalig auf dem Cluster-Login-Knoten:
module load apptainer
apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
    docker://lucam03/projekt-humanoider-roboter:latest

# Job einreichen:
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
sbatch Training/kisski_submit.sh
```
→ Vollständige Anleitung: [Abschnitt 4](#4-hpc-training-auf-kisski)

**Auf vast.ai:**
1. Instanz mit Image `lucam03/projekt-humanoider-roboter:latest` starten
2. In "Docker options" setzen: `-e HF_TOKEN=hf_… -e WANDB_API_KEY=…`
3. Instanz startet → Container startet automatisch → Training läuft autonom

**Lokal (Linux / WSL2):**
```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
./Training/setup_and_train_DockerHub-pull.sh
```

Detaillierte Schritt-für-Schritt-Anleitung: [Anleitung.md](Anleitung.md)

---

## 2. Voraussetzungen

### Hardware

| Anforderung | Minimum (Training) | Empfohlen |
|---|---|---|
| GPU | NVIDIA GPU, ≥ 24 GB VRAM | A100/H100 ≥ 40 GB |
| VRAM | 24 GB (batch_size ≤ 2) | 40–80 GB |
| RAM | 32 GB | 64 GB |
| Speicherplatz | 80 GB im Container-Filesystem | SSD empfohlen |

> **Hinweis:** GR00T N1.6 Full Fine-tuning verbraucht laut NVIDIA ~25–31 GB VRAM (bei batch_size=8). NVIDIA empfiehlt offiziell **≥ 40 GB VRAM** (H100, L40). Karten mit 24 GB (RTX 4090, A5000) können mit `GLOBAL_BATCH_SIZE=1–2` und `--no-tune_diffusion_model` laufen, jedoch sehr langsam. Für ernsthaftes Training: KISSKI A100 (80 GB).

### Software (nur für lokales Training)

| Software | Version | Installationslink |
|---|---|---|
| Docker | ≥ 4.x | https://www.docker.com/products/docker-desktop |
| NVIDIA Container Toolkit | aktuell | https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html |
| NVIDIA Treiber | ≥ 570 (CUDA 12.8) | https://www.nvidia.com/drivers |

Auf vast.ai und KISSKI sind diese vorinstalliert — du brauchst lokal nichts davon.

### Accounts

- **HuggingFace** mit Zugriff auf:
  - [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) (Lizenz akzeptieren!)
  - [`unitreerobotics/G1_Dex3_BlockStacking_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset)
- **WandB** (optional, kostenlos) für Trainings-Logging: https://wandb.ai
- **KISSKI-Account** (nur für HPC-Training): Beantragung über das [GWDG-Portal](https://docs.hpc.gwdg.de)

---

## 3. Cloud-Training auf vast.ai

### Image

```
lucam03/projekt-humanoider-roboter:latest
```

Image ist auf Docker Hub veröffentlicht. Beim Start läuft `/scripts/entrypoint.sh` automatisch und macht:

1. GPU-Check (`nvidia-smi`)
2. HuggingFace-Token aus Env-Var einlesen
3. Modell + Datensatz nach `/data` herunterladen (~25 GB, idempotent — übersprungen bei Restart)
4. Datensatz LeRobot v3.0 → v2.1 konvertieren (idempotent)
5. Fine-tuning starten (optional mit W&B-Logging)

### Vast.ai-Setup

1. **Instanz auswählen:** mindestens 8 GB VRAM, ≥ 80 GB Disk (am Container hängend, kein Extra-Volume)
2. **Image-URL:** `lucam03/projekt-humanoider-roboter:latest`
3. **Docker options** (im Web-UI):
   ```
   -e HF_TOKEN=hf_xxx
   -e WANDB_API_KEY=xxx
   -e MAX_STEPS=30000
   -e GLOBAL_BATCH_SIZE=8
   --shm-size=16g
   ```
4. **Launch** — der Container startet, `entrypoint.sh` läuft, Training beginnt.

> **Kein `-v`-Mount, kein Extra-Volume nötig.** Auf vast.ai gibt es genau einen Container pro Instanz; das Container-Filesystem ist persistent über Stop/Restart hinweg und wird nur beim Destroy gelöscht.

Den Trainingsfortschritt verfolgst du live über das WandB-Dashboard. Checkpoints landen unter `/data/g1_dex3_finetune/` **im Container**.

### Daten aus dem Container holen

Bevor du die vast.ai-Instanz zerstörst (sonst sind die Checkpoints weg), per SSH:

```bash
# Auf der vast.ai-Instanz oder per SSH-Forwarding
docker cp <container>:/data/g1_dex3_finetune ./checkpoints
```

Oder direkt aus dem Container heraus per `huggingface-cli upload`, `rclone`, `scp` etc. — siehe [Abschnitt 7](#7-daten-aus-dem-container-holen).

---

## 4. HPC-Training auf KISSKI

KISSKI ist ein dedizierter GPU-Cluster der GWDG Göttingen. Er läuft **kein Docker**, sondern **Apptainer** (früher Singularity) als Container-Runtime und **SLURM** als Job-Scheduler. Das bedeutet: kein `docker run`, sondern `sbatch Training/kisski_submit.sh`.

### Verfügbare GPU-Partitionen

| Partition | GPU | VRAM | Max. Walltime |
|-----------|-----|------|---------------|
| `kisski` | NVIDIA A100 | 80 GB | 48 h |
| `kisski-h100` | NVIDIA H100 | 94 GB | 48 h |

`kisski_submit.sh` verwendet standardmäßig `-p kisski` (A100). Für H100 in der Datei `#SBATCH -p kisski` auf `#SBATCH -p kisski-h100` und `-G A100:1` auf `-G H100:1` ändern.

---

### Schritt 1 — Image einmalig zu SIF konvertieren (auf dem Login-Knoten)

Das Docker Hub-Image muss einmalig in das Apptainer-Format (`.sif`) umgewandelt werden. Dies geschieht **auf dem Login-Knoten** `glogin-gpu.hpc.gwdg.de` und dauert einige Minuten.

```bash
ssh <username>@glogin-gpu.hpc.gwdg.de

module load apptainer
mkdir -p $HOME/images

apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
    docker://lucam03/projekt-humanoider-roboter:latest
```

Das erzeugt `$HOME/images/projekt-humanoider-roboter.sif` (~10–15 GB). Diese Datei nur neu erstellen, wenn ein neues Image auf Docker Hub gepusht wurde.

---

### Schritt 2 — Daten auf VAST-Projekt-Storage übertragen (einmalig oder bei Update)

Der Cluster nutzt für dieses Projekt den **GWDG VAST-Projekt-Storage** unter
`/mnt/vast-kisski/projects/kisski-humrob/data/` (persistente SSD-Storage — kein Scratch!).
`kisski_submit.sh` setzt diesen Pfad als Standard-`DATA_DIR`.

> **Hinweis:** Der alte SCRATCH-SCC-Speicher (`/scratch/`) wurde am 31.03.2026 abgeschaltet.
> Alle Daten müssen auf dem VAST-Projekt-Storage liegen.

**Option A: Von deinem lokalen Rechner mit rsync**

```bash
# Struktur auf dem Cluster anlegen
ssh <username>@transfer.hpc.gwdg.de "mkdir -p /mnt/vast-kisski/projects/kisski-humrob/data"

# Datensatz + Modell übertragen (falls schon lokal vorhanden)
rsync -avz --progress \
    ./data/ \
    <username>@transfer.hpc.gwdg.de:/mnt/vast-kisski/projects/kisski-humrob/data/
```

**Option B: Direkt vom Cluster aus herunterladen**

`kisski_submit.sh` setzt `SKIP_DOWNLOAD=1` als Standard (Daten auf `/mnt/vast-kisski/...` werden als vorhanden angenommen). Für den allerersten Lauf — wenn noch keine Daten auf dem Cluster-Storage liegen — muss `SKIP_DOWNLOAD=0` explizit gesetzt werden:

```bash
export SKIP_DOWNLOAD=0
sbatch Training/kisski_submit.sh
```

Der Entrypoint lädt Modell und Datensatz dann selbst von HuggingFace (~25 GB, ca. 10–30 min). Bei allen weiteren Läufen ist `SKIP_DOWNLOAD=1` korrekt (Download wird automatisch übersprungen).

---

### Schritt 3 — Tokens setzen und Job einreichen

```bash
# Auf dem Login-Knoten:
export HF_TOKEN=hf_...
export WANDB_API_KEY=...       # optional, aber empfohlen

# Optional: Trainings-Parameter überschreiben
export MAX_STEPS=30000
export GLOBAL_BATCH_SIZE=32    # A100 mit 80 GB VRAM verträgt deutlich mehr als 8

# Job einreichen
sbatch Training/kisski_submit.sh
```

SLURM gibt die Job-ID aus, z. B. `Submitted batch job 12345678`.

> **Tipp:** Tokens nicht dauerhaft in `.bashrc` speichern. Stattdessen vor jedem `sbatch` kurz exportieren oder in eine nicht-eingecheckte `.env`-Datei auf dem Cluster schreiben und dort sourcen.

---

### Job-Status verfolgen

```bash
# Eigene Jobs anzeigen
squeue -u $USER

# Logs live verfolgen (solange Job läuft oder danach)
tail -f logs/slurm-<jobid>.out

# Job abbrechen
scancel <jobid>

# Cluster-Auslastung anzeigen
sinfo -p kisski
```

SLURM-Logs landen in `logs/slurm-<jobid>.out` und `logs/slurm-<jobid>.err` im Verzeichnis, aus dem `sbatch` aufgerufen wurde. Trainings-Logs schreibt der Container zusätzlich nach `/mnt/vast-kisski/projects/kisski-humrob/data/logs/`.

---

### Checkpoints nach dem Training sichern

Checkpoints liegen nach dem Job in `/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/blockstacking/` auf dem Cluster. Von dort auf deinen lokalen Rechner oder nach HuggingFace exportieren:

```bash
# Auf deinem lokalen Rechner
rsync -avz --progress \
    <username>@transfer.hpc.gwdg.de:/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/ \
    ./checkpoints/

# Oder direkt vom Cluster aus nach HuggingFace pushen
# (interaktiven Job starten, dann im Container):
srun -p kisski:interactive -G 1g.10gb:1 --pty bash
apptainer shell --nv \
    --bind /mnt/vast-kisski/projects/kisski-humrob/data:/data \
    ~/.project/dir.project/images/projekt-humanoider-roboter.sif
# Innerhalb des Containers:
huggingface-cli upload <dein-namespace>/g1-dex3-blockstacking \
    /data/g1_dex3_finetune/blockstacking --repo-type=model
```

> **Hinweis:** Daten auf dem VAST-Projekt-Storage werden **nicht automatisch gelöscht** (anders als früher auf `/scratch`). Trotzdem empfiehlt sich ein regelmäßiger Export nach HuggingFace oder lokal.

---

### `kisski_submit.sh` anpassen

Die wichtigsten Stellschrauben in [Training/kisski_submit.sh](Training/kisski_submit.sh):

```bash
#SBATCH -p kisski          # Partition: kisski (A100 80GB) oder kisski-h100 (H100 94GB)
#SBATCH -G A100:1          # Anzahl GPUs — bei kisski-h100 auf H100:1 ändern
#SBATCH -c 16              # CPU-Kerne
#SBATCH --mem=64G          # RAM
#SBATCH -t 48:00:00        # Walltime (max. 48h)
```

Und die Trainings-Parameter entweder vor `sbatch` als `export` setzen oder direkt im Script:

```bash
export GLOBAL_BATCH_SIZE=32   # A100 (80 GB) verträgt viel mehr als die Standard-8
export MAX_STEPS=50000
```

---

## 5. Lokales Training

### Variante A: Mit dem Launcher-Skript (empfohlen)

```bash
git clone https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
git checkout training-luca-KISSKI

export HF_TOKEN=hf_...
export WANDB_API_KEY=...        # optional

./Training/setup_and_train_DockerHub-pull.sh
```

Das Skript zieht das Image, startet einen Container namens `groot-train` (**ohne `--rm`** und **ohne `-v`-Mount**) und lässt den Entrypoint laufen.

Optionen:
- `--skip-pull` — Image ist bereits lokal vorhanden
- `--interactive` — Shell statt automatisches Training
- `--resume` — bestehenden Container weiterlaufen lassen (Daten + Checkpoints bleiben erhalten)
- `--destroy` — Container löschen, alles verwerfen und neu starten
- `--dry-run` — Befehle nur anzeigen

### Variante B: Manuell mit `docker run`

```bash
docker pull lucam03/projekt-humanoider-roboter:latest

docker run --name groot-train --gpus all --ipc=host --shm-size=16g \
  -e HF_TOKEN=hf_... \
  -e WANDB_API_KEY=... \
  -e MAX_STEPS=30000 \
  -e GLOBAL_BATCH_SIZE=8 \
  -it lucam03/projekt-humanoider-roboter:latest
```

Wichtig: **kein `--rm`**, sonst sind Daten und Checkpoints nach Stop weg.

Nach `Ctrl+C` oder `docker stop`:
```bash
docker start -ai groot-train      # Training fortsetzen (Daten bleiben)
docker rm -f groot-train          # Komplett löschen (alles weg)
```

### Variante C: Image selbst bauen

```bash
git clone https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
git checkout training-luca-KISSKI
docker build -t projektarbeit-humanoider-roboter Training/   # Build-Context = Training/
```

Build dauert ~30 Minuten (PyTorch, flash-attn). Anschließend ein `docker run` wie in Variante B mit dem lokalen Image-Namen.

### Interaktive Shell (für Debugging)

```bash
docker run -it --name groot-debug --gpus all --ipc=host --shm-size=16g \
  lucam03/projekt-humanoider-roboter:latest bash
```

Wird beim Aufruf ein Befehl wie `bash` übergeben, wird der Entrypoint übersprungen und der Befehl direkt ausgeführt.

---

## 6. Konfiguration über Env-Vars

Alle Parameter werden über Umgebungsvariablen gesteuert — auf vast.ai, KISSKI und lokal identisch:

| Variable | Default | Beschreibung |
|---|---|---|
| `HF_TOKEN` | — | **Pflicht.** HuggingFace-Token (Lese-Berechtigung reicht) |
| `WANDB_API_KEY` | — | Optional. W&B-Key. Ohne diesen läuft Training ohne W&B. |
| `MAX_STEPS` | `30000` | Anzahl Trainings-Steps |
| `GLOBAL_BATCH_SIZE` | `8` | Globale Batch-Size (8 für 8 GB VRAM, 32+ für A100 80 GB) |
| `NUM_GPUS` | `1` | Anzahl genutzter GPUs |
| `WANDB_PROJECT` | `gr00t-g1-dex3` | W&B-Projektname |
| `DATA_DIR` | `/data` | Datenverzeichnis im Container |
| `SKIP_DOWNLOAD` | `0` | Auf `1` setzen, wenn Daten schon vorhanden sind |
| `SKIP_CONVERT` | `0` | Auf `1` setzen, wenn `modality.json` schon existiert |
| `SKIP_TRAIN` | `0` | Auf `1` setzen für reines Setup (fällt in Shell) |
| `SHELL_ON_ERROR` | `0` | Auf `1` setzen für Debug-Shell bei Fehler |

### VRAM-Richtwerte

| VRAM | `GLOBAL_BATCH_SIZE` | `MAX_STEPS` | Umgebung |
|---|---|---|---|
| 24 GB | 1–2 | 30 000 | Lokal (RTX 4090, min.) — sehr langsam |
| 32 GB | 4–8  | 30 000 | Lokal (RTX 5090, ~31 GB bei bs=8) |
| 40 GB | 16–32 | 50 000 | vast.ai A100 40 GB |
| 80 GB | 64–128 | 50 000+ | KISSKI A100 80 GB |

> **Full Fine-tuning benötigt laut NVIDIA ≥ 40 GB VRAM.** Karten mit < 24 GB VRAM führen zu OOM-Fehlern.

Reduziere bei `CUDA out of memory` zuerst `GLOBAL_BATCH_SIZE`. Volle Parameter-Referenz: [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md)

---

## 7. Daten aus dem Container holen

### Lokal / vast.ai: `docker cp`

```bash
docker cp groot-train:/data/g1_dex3_finetune ./checkpoints
docker cp groot-train:/data/logs ./logs
```

Funktioniert auch, während der Container läuft.

### KISSKI: `rsync` vom Cluster

```bash
rsync -avz --progress \
    <username>@transfer.hpc.gwdg.de:/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/ \
    ./checkpoints/
```

### Direkt nach HuggingFace pushen

Mit interaktiver Shell im Container:

```bash
cd /data/g1_dex3_finetune/blockstacking
huggingface-cli upload <dein-namespace>/g1-dex3-blockstacking . --repo-type=model
```

---

## 8. Training beobachten

### WandB (empfohlen)

Mit gesetztem `WANDB_API_KEY` läuft das Logging automatisch. Live verfolgen unter https://wandb.ai → Projekt `gr00t-g1-dex3` (bzw. dein `WANDB_PROJECT`). Funktioniert auf vast.ai und KISSKI gleich — der Container baut die Verbindung nach außen auf.

### Live-Logs

**Lokal / vast.ai:**
```bash
docker logs -f groot-train
```

**KISSKI (SLURM-Logs):**
```bash
tail -f logs/slurm-<jobid>.out
```

**KISSKI (Trainings-Logs auf VAST):**
```bash
ssh <username>@glogin-gpu.hpc.gwdg.de \
    "tail -f /mnt/vast-kisski/projects/kisski-humrob/data/logs/finetune-*.log"
```

### Checkpoints

**Lokal / vast.ai:**
```bash
docker exec groot-train ls -lht /data/g1_dex3_finetune/blockstacking/
```

**KISSKI:**
```bash
ls -lht /mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/blockstacking/
```

---

## 9. Projektstruktur

```
/
├── Training/                           # Alles rund ums Training (Build, Run, Doku)
│   ├── Dockerfile                      # Container-Definition mit ENTRYPOINT
│   │                                   #   Build-Context = Training/ (damit COPY scripts/ greift)
│   ├── docker-compose.yml              # Optional (lokale Dev-Variante)
│   ├── kisski_submit.sh                # SLURM-Job-Script für KISSKI HPC-Cluster
│   ├── update_image.ps1                # Host-Build/Push-Tool (muss neben dem Dockerfile liegen)
│   ├── setup_and_train_DockerHub-pull.sh   # Thin host-launcher (Linux/macOS/WSL2)
│   ├── setup_and_train_DockerHub-pull.ps1  # Thin host-launcher (Windows PowerShell)
│   ├── setup_and_train_Container-build.*   # Host-launcher mit lokalem Image-Build
│   ├── Train-Test-split.md             # Datensatz-Split (80/20)
│   ├── WANDB_OFFLINE_SYNC.md           # W&B-Offline-Sync auf KISSKI
│   └── scripts/                        # In das Image kopiert (→ /scripts)
│       ├── entrypoint.sh               # ENTRYPOINT — orchestriert Download → Convert → Train
│       ├── download_data.sh            # HuggingFace-Download
│       ├── run_finetuning.sh           # Trainings-Launcher (im Container)
│       └── run_finetuning.ps1          # Trainings-Launcher (Windows-Variante)
├── Simulation/                         # Closed-Loop-Sim-Eval (in Entwicklung)
│   ├── Dockerfile                      # Sim-Client-Container (Isaac Lab + GR00T-Client)
│   ├── kisski_sim_submit.sh            # SLURM-Job für Sim-Eval (jupyter-Partition, RTX 5000)
│   ├── update_sim_image.ps1            # Build/Push-Tool für das Sim-Image
│   ├── ISAAC_LAB_SIM_PLAN.md           # Implementierungsplan für die Isaac-Lab-Sim
│   ├── SIM_GPU_COMPATIBILITY.md        # GPU-Kompatibilität (RT-Cores, jupyter-Partition)
│   ├── SIM_DOCKER_BUILD.md             # Build- und Deployment-Anleitung Sim-Container
│   └── KISSKI_SIM_DESKTOP_ANLEITUNG.md # Schritt-für-Schritt für JupyterHPC-Desktop-Test
├── data/                               # Lokale Daten-/Checkpoint-Platzhalter (geteilt)
└── app/                                # Git-Submodule (im Image bereits geklont)
    └── Groot-1.6/                      # GR00T N1.6 + eigene G1/DEX3-Configs
        └── examples/G1_DEX3/
            ├── g1_dex3_config.py
            ├── modality_4cam.json
            ├── modality_2cam.json
            ├── FINETUNING_GUIDE.md
            └── SETUP_DOCUMENTATION.md
```

Zur Laufzeit (lokal/vast.ai im Container-Filesystem, auf KISSKI unter `/scratch/<username>/data/`):
```
/data/
├── models/GR00T-N1.6-3B/    # Modellgewichte (~6 GB)
├── unitreerobotics/          # Rohdatensatz (~18 GB)
├── g1_dex3_finetune/        # Checkpoints
└── logs/                     # Trainings-Logs
```

---

## 10. Häufige Probleme

### `docker: Error response from daemon: could not select device driver "nvidia"`

NVIDIA Container Toolkit fehlt. Auf vast.ai und KISSKI ist es vorinstalliert; lokal:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### `CUDA out of memory`

`GLOBAL_BATCH_SIZE` halbieren:
```bash
-e GLOBAL_BATCH_SIZE=4
```

### `HF_TOKEN ist nicht gesetzt`

Entrypoint bricht ab, wenn kein Token gesetzt ist. Token erstellen auf https://huggingface.co/settings/tokens (Read-Berechtigung reicht) und per `-e HF_TOKEN=hf_...` (Docker) oder `export HF_TOKEN=hf_...` (KISSKI) mitgeben.

### Container existiert bereits (`Conflict. The container name "/groot-train" is already in use`)

Du hast schon einen Container aus einem früheren Lauf:
```bash
./Training/setup_and_train_DockerHub-pull.sh --resume    # Daten + Checkpoints behalten
./Training/setup_and_train_DockerHub-pull.sh --destroy   # alles verwerfen und neu starten
```

### Ich habe `--rm` benutzt und meine Checkpoints sind weg

Das Image hat keine Volume-Mounts mehr — `--rm` zerstört das Container-Filesystem komplett. **Niemals `--rm` mit diesem Image verwenden**, außer du willst explizit alles verlieren.

### Download bricht ab

Container neu starten (`docker start -ai groot-train`) — `huggingface-cli` setzt teilweise heruntergeladene Dateien automatisch fort. Sind die Daten vorhanden, überspringt der Entrypoint den Download.

### `KeyError: annotation.human.task_description not found`

`modality.json` fehlt im Datensatz. Manuell kopieren:
```bash
docker exec groot-train cp \
  /app/Groot-1.6/examples/G1_DEX3/modality_4cam.json \
  /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/modality.json
```

### Debug: Shell bei Fehler

`-e SHELL_ON_ERROR=1` mitgeben — bei Fehler im Entrypoint landest du in einer interaktiven Shell statt dass der Container beendet wird.

### KISSKI: `SIF-Image nicht gefunden`

Das Apptainer-Image wurde noch nicht erstellt. Einmalig auf dem Login-Knoten:
```bash
module load apptainer
mkdir -p $HOME/images
apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
    docker://lucam03/projekt-humanoider-roboter:latest
```

### KISSKI: Job bleibt in Status `PD` (Pending)

Die Partition ist ausgelastet. Mit `squeue -p kisski` prüfen wie viele Jobs warten. Alternative: `kisski-h100`-Partition versuchen oder Walltime verkürzen (kürzere Jobs haben höhere Priorität).

### KISSKI: `No space left on device` im Container

Der VAST-Projekt-Storage ist voll. Mit `du -sh /mnt/vast-kisski/projects/kisski-humrob/*` prüfen. Alte Checkpoints unter `/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/` aufräumen — `save_total_limit=5` im Training-Script sorgt dafür, dass maximal 5 Checkpoints gleichzeitig vorgehalten werden.

---

## 11. Train-Test-Split

Der Datensatz ist in einen Trainings- und einen Test-Split aufgeteilt (80/20), damit das Modell nach dem Fine-tuning auf ungesehenen Episoden bewertet werden kann.

| Split | Episoden | Anteil |
|-------|----------|--------|
| `train` | 241 | 80 % |
| `test` | 60 | 20 % |

Vollständige Beschreibung: [Train-Test-split.md](Training/Train-Test-split.md)

---

## Weiterführende Dokumentation

- [Anleitung.md](Anleitung.md) — Schritt-für-Schritt-Anleitung zur Nutzung der Umgebung
- [Train-Test-split.md](Training/Train-Test-split.md) — Implementierung und Nutzung des 80/20-Splits
- [`app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md`](app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md)
- [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md)
- [GWDG HPC Dokumentation](https://docs.hpc.gwdg.de) — Offizielle Doku für KISSKI/Grete-Cluster
- [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/groot)
- [LeRobot](https://github.com/huggingface/lerobot)
