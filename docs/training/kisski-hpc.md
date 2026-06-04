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
SKIP_DOWNLOAD=0 HF_TOKEN=hf_... WANDB_API_KEY=... sbatch Training/kisski_submit.sh
```

Der Entrypoint lädt Modell und Datensatz dann selbst von HuggingFace (~25 GB, ca. 10–30 min).
Bei allen weiteren Läufen ist `SKIP_DOWNLOAD=1` korrekt (Download wird automatisch übersprungen).

---

## Schritt 3 — Tokens setzen und Job einreichen

> **WICHTIG — Variablen als Inline-Prefix angeben.** Alle Tokens und Overrides müssen **direkt vor
> `sbatch` in einer Zeile** stehen (per `\` umgebrochen), **nicht** als getrennte `export`-Zeilen
> davor. Grund: Das Script setzt `#SBATCH --export=ALL`, das aber nur die Umgebung des
> `sbatch`-Aufrufs **selbst** an den Job weiterreicht. Getrennte `export`-Zeilen landen je nach
> Shell/Session (z. B. in einer `screen`-Session) nicht zuverlässig in genau dieser Umgebung —
> als Inline-Prefix sind die Variablen hingegen garantiert Teil davon.

```bash
# Auf dem Login-Knoten — ALLES in EINEM Befehl, Variablen direkt vor sbatch:
HF_TOKEN=hf_... \
WANDB_API_KEY=... \
sbatch Training/kisski_submit.sh
```

Mit optionalen Trainings-Overrides (ebenfalls als Inline-Prefix, einfach weitere Zeilen davor):

```bash
HF_TOKEN=hf_... \
WANDB_API_KEY=... \
MAX_STEPS=175000 \
GLOBAL_BATCH_SIZE=32 \
sbatch Training/kisski_submit.sh
```

SLURM gibt die Job-ID aus, z. B. `Submitted batch job 12345678`.

> **`WANDB_API_KEY` ist Pflicht** (nicht optional): `run_finetuning.sh` läuft mit `USE_WANDB=1` als
> Default und **bricht ohne Key sofort ab** — auch im Offline-Modus, da die Key-Prüfung vor dem
> Offline-Check greift. Wer ohne W&B trainieren will, setzt zusätzlich `USE_WANDB=0` davor.
>
> **Tipp:** Tokens nicht dauerhaft in `.bashrc` speichern. Stattdessen in eine nicht-eingecheckte
> `.env`-Datei auf dem Cluster schreiben und die Werte beim Submit referenzieren, z. B.
> `HF_TOKEN=$(grep -oP 'HF_TOKEN=\K.*' ~/.env) WANDB_API_KEY=... sbatch …`.

Alle Env-Vars sind in der [Konfigurationsreferenz](env-vars.md) beschrieben.

---

## Variante — Vision-Encoder mittrainieren (`TUNE_VISUAL=1`)

Standardmäßig ist der **Vision-Encoder eingefroren**. Der Standardlauf trainiert nur den
Multimodal-Projector und den Diffusion-Action-Head; das LLM-Backbone bleibt ebenfalls
eingefroren. Wer zusätzlich die *visuelle Repräsentation* an die eigene Domäne anpassen
will (z. B. wegen des Sim-zu-Real-Domain-Gaps), startet die Vision-Variante.

```bash
# Vision-Encoder mittrainieren — LR, Warmup und Output-Namespace werden automatisch gesetzt:
HF_TOKEN=hf_... \
WANDB_API_KEY=... \
TUNE_VISUAL=1 \
sbatch --export=ALL Training/kisski_submit.sh
```

**Was passiert dabei:**

| Komponente | Standardlauf | Vision-Lauf (`TUNE_VISUAL=1`) |
|---|---|---|
| LLM-Backbone (`tune_llm`)          | ❄️ eingefroren | ❄️ eingefroren |
| Vision-Encoder (`tune_visual`)     | ❄️ eingefroren | 🔥 **trainiert** |
| Multimodal-Projector (`tune_projector`) | 🔥 trainiert | 🔥 trainiert |
| Diffusion-Action-Head (`tune_diffusion_model`) | 🔥 trainiert | 🔥 trainiert |

**Automatik bei `TUNE_VISUAL=1`** (alles in `kisski_submit.sh`, jederzeit per Env-Var
überschreibbar):

- Der Container-Entrypoint startet **`run_finetuning_vision.sh`** statt `run_finetuning.sh`
  (das Standard-Skript bleibt unverändert).
- Eigener Output-Namespace: `…/data/g1_dex3_finetune/**blockstacking_vision**/` — überschreibt
  also keine Standard-Läufe. Experiment-Name: `g1_dex3_blockstacking_vision_v1`.
- **Lernrate `1e-4`** statt der Standard-`2e-4`. Begründung: Die Config hat *eine globale LR*
  für alle trainierbaren Parameter — der große vortrainierte Eagle-ViT teilt sie sich mit dem
  leichten Action-Head. `2e-4` würde die vortrainierten Visual-Features destabilisieren.
- **`WARMUP_RATIO=0.1`** (statt `0.05`) für einen sanfteren Start.
- **Batch-Size `32`** (= Standard-Default, per_device 8 auf 4×A100) bleibt unverändert.

**Empfehlungen / Fallstricke:**

- **VRAM:** Der entfrorene Vision-Encoder erhöht Activation- und Optimizer-Speicher. OOM tritt
  sofort beim Start auf (nicht erst nach Stunden) — bei OOM `GLOBAL_BATCH_SIZE=16` setzen
  (muss durch `NUM_GPUS` teilbar bleiben) und LR mit-runterskalieren (~`7e-5`).
- **Instabiler Loss:** als Fallback `LEARNING_RATE=5e-5` davorsetzen.
- **Checkpoint-Größe:** Mehr trainierbare Parameter ⇒ größere `optimizer.pt` pro Checkpoint.
  Bei `SAVE_TOTAL_LIMIT=40` ggf. das Storage-Budget (~770 GB) im Blick behalten und das Limit
  senken.

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

Trainings-Parameter werden entweder dauerhaft im Script geändert oder pro Lauf als Inline-Prefix
direkt vor `sbatch` übergeben (siehe [Schritt 3](#schritt-3--tokens-setzen-und-job-einreichen) —
**nicht** als getrennte `export`-Zeilen, sonst landen sie nicht im Job):

```bash
GLOBAL_BATCH_SIZE=32 \
MAX_STEPS=50000 \
HF_TOKEN=hf_... WANDB_API_KEY=... \
sbatch Training/kisski_submit.sh
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
