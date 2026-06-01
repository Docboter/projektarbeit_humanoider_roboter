# HPC-Training auf KISSKI

KISSKI ist ein dedizierter GPU-Cluster der GWDG Göttingen. Er läuft **kein Docker**, sondern
**Apptainer** (früher Singularity) als Container-Runtime und **SLURM** als Job-Scheduler. Das
bedeutet: kein `docker run`, sondern `sbatch Training/kisski_submit.sh`. Das Docker-Hub-Image
wird einmalig in eine Apptainer-`.sif`-Datei umgewandelt — danach läuft alles unverändert.

Zugang beantragen: [GWDG-Portal](https://docs.hpc.gwdg.de). SSH-Login-Knoten:
`glogin-gpu.hpc.gwdg.de`.

## Verfügbare GPU-Partitionen

| Partition | GPU | VRAM | Max. Walltime |
|-----------|-----|------|---------------|
| `kisski` | NVIDIA A100 | 80 GB | 48 h |
| `kisski-h100` | NVIDIA H100 | 94 GB | 48 h |

`kisski_submit.sh` verwendet standardmäßig `-p kisski` (A100). Für H100 in der Datei
`#SBATCH -p kisski` auf `#SBATCH -p kisski-h100` und `-G A100:1` auf `-G H100:1` ändern.

---

## Schritt 1 — Image einmalig zu SIF konvertieren (auf dem Login-Knoten)

Das Docker-Hub-Image muss einmalig in das Apptainer-Format (`.sif`) umgewandelt werden. Dies
geschieht **auf dem Login-Knoten** `glogin-gpu.hpc.gwdg.de` und dauert einige Minuten.

```bash
ssh <username>@glogin-gpu.hpc.gwdg.de

module load apptainer
mkdir -p $HOME/images

apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
    docker://lucam03/projekt-humanoider-roboter:latest
```

Das erzeugt `$HOME/images/projekt-humanoider-roboter.sif` (~10–15 GB). Diese Datei nur neu
erstellen, wenn ein neues Image auf Docker Hub gepusht wurde.

---

## Schritt 2 — Daten auf VAST-Projekt-Storage übertragen (einmalig oder bei Update)

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

`kisski_submit.sh` setzt `SKIP_DOWNLOAD=1` als Standard (Daten auf `/mnt/vast-kisski/...` werden
als vorhanden angenommen). Für den allerersten Lauf — wenn noch keine Daten auf dem
Cluster-Storage liegen — muss `SKIP_DOWNLOAD=0` explizit gesetzt werden:

```bash
export SKIP_DOWNLOAD=0
sbatch Training/kisski_submit.sh
```

Der Entrypoint lädt Modell und Datensatz dann selbst von HuggingFace (~25 GB, ca. 10–30 min).
Bei allen weiteren Läufen ist `SKIP_DOWNLOAD=1` korrekt (Download wird automatisch übersprungen).

---

## Schritt 3 — Tokens setzen und Job einreichen

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

> **Tipp:** Tokens nicht dauerhaft in `.bashrc` speichern. Stattdessen vor jedem `sbatch` kurz
> exportieren oder in eine nicht-eingecheckte `.env`-Datei auf dem Cluster schreiben und dort
> sourcen.

Alle Env-Vars sind in der [Konfigurationsreferenz](env-vars.md) beschrieben.

---

## Job-Status verfolgen

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

SLURM-Logs landen in `logs/slurm-<jobid>.out` und `logs/slurm-<jobid>.err` im Verzeichnis, aus
dem `sbatch` aufgerufen wurde. Trainings-Logs schreibt der Container zusätzlich nach
`/mnt/vast-kisski/projects/kisski-humrob/data/logs/`.

W&B funktioniert auf KISSKI im Offline-Modus — siehe [W&B-Offline-Sync](wandb-offline-sync.md).

---

## Checkpoints nach dem Training sichern

Checkpoints liegen nach dem Job in
`/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/blockstacking/` auf dem Cluster.
Von dort auf deinen lokalen Rechner oder nach HuggingFace exportieren:

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

> **Hinweis:** Daten auf dem VAST-Projekt-Storage werden **nicht automatisch gelöscht** (anders
> als früher auf `/scratch`). Trotzdem empfiehlt sich ein regelmäßiger Export nach HuggingFace
> oder lokal.

---

## `kisski_submit.sh` anpassen

Die wichtigsten Stellschrauben in [Training/kisski_submit.sh](../../Training/kisski_submit.sh):

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

## Häufige KISSKI-Probleme

### `SIF-Image nicht gefunden`

Das Apptainer-Image wurde noch nicht erstellt. Einmalig auf dem Login-Knoten:
```bash
module load apptainer
mkdir -p $HOME/images
apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
    docker://lucam03/projekt-humanoider-roboter:latest
```

### Job bleibt in Status `PD` (Pending)

Die Partition ist ausgelastet. Mit `squeue -p kisski` prüfen, wie viele Jobs warten.
Alternative: `kisski-h100`-Partition versuchen oder Walltime verkürzen (kürzere Jobs haben
höhere Priorität).

### `No space left on device` im Container

Der VAST-Projekt-Storage ist voll. Mit `du -sh /mnt/vast-kisski/projects/kisski-humrob/*`
prüfen. Alte Checkpoints unter `.../data/g1_dex3_finetune/` aufräumen — `save_total_limit=5`
im Training-Script sorgt dafür, dass maximal 5 Checkpoints gleichzeitig vorgehalten werden.
