# HPC-Training auf KISSKI

KISSKI ist ein dedizierter GPU-Cluster der GWDG Göttingen. Er läuft **kein Docker**, sondern
**Apptainer** (früher Singularity) als Container-Runtime und **SLURM** als Job-Scheduler. Das
bedeutet: kein `docker run`, sondern `sbatch Training/kisski_submit.sh`. Das Docker-Hub-Image
wird einmalig in eine Apptainer-`.sif`-Datei umgewandelt — danach läuft alles unverändert.

Zugang beantragen: [GWDG-Portal](https://docs.hpc.gwdg.de). SSH-Login-Knoten:
`glogin-gpu.hpc.gwdg.de`.

## Pfad-Konfiguration — zwei Variablen, sonst nichts

Die sechs SLURM-Skripte enthalten seit 2026-08-18 **keine** kontogebundenen Pfade mehr. Sie
leiten alles aus zwei Variablen ab:

| Variable | Default | Wirkung |
|---|---|---|
| `KISSKI_PROJECT_DIR` | `/mnt/vast-kisski/projects/kisski-humrob` | daraus folgen `DATA_DIR`, `REPO_DIR`, `GROOT_FORK_DIR`, `ASSETS_DIR`, `ISAAC_CACHE`, `APPTAINER_CACHEDIR`/`_TMPDIR`, `SIM_CODE` |
| `KISSKI_SIF_DIR` | `$HOME/images` | wo die `.sif`-Dateien gesucht werden; existiert dort nichts, wird `$KISSKI_PROJECT_DIR/images` geprüft |

Für dieses Projekt sind beide Defaults richtig — vorausgesetzt, die SIFs liegen in
`$HOME/images` (so lädt sie Schritt 1 unten). Liegen sie woanders, genügt **ein** Symlink:

```bash
ln -s /pfad/zu/deinen/images "$HOME/images"     # oder: export KISSKI_SIF_DIR=/pfad/zu/deinen/images
```

Ein anderes KISSKI-Projekt braucht genau eine Zeile — das wirkt, weil alle Skripte
`#SBATCH --export=ALL` tragen:

```bash
KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> sbatch Training/kisski_submit.sh
```

**SLURM-Logs** schreiben alle Skripte nach `logs/` **relativ zum Aufrufverzeichnis** von
`sbatch` (in `#SBATCH`-Zeilen kann SLURM keine Variablen auflösen). Einmalig im Repo-Checkout
`mkdir -p logs` — sonst startet der Job nicht, weil SLURM die Datei, aber nicht den Ordner
anlegt. Anderer Ort: `sbatch --output=/pfad/%j.out --error=/pfad/%j.err <skript>`.

Vollständige Referenz und Migrationsschritte: [portabilitaet.md](../portabilitaet.md).

## Verfügbare GPU-Partitionen

| Partition | GPU | VRAM | Max. Walltime |
|-----------|-----|------|---------------|
| `kisski` | NVIDIA A100 | 80 GB | 48 h |
| `kisski-h100` | NVIDIA H100 | 94 GB | 48 h |

`kisski_submit.sh` verwendet standardmäßig `-p kisski` (A100). Für H100 in der Datei
`#SBATCH -p kisski` auf `#SBATCH -p kisski-h100` und `-G A100:1` auf `-G H100:1` ändern.

> ### ⚠️ RL-Fine-tuning (`USE_RL`) läuft auf KISSKI **nicht**
>
> Das **BC-Training** (Behavior Cloning) läuft auf KISSKI einwandfrei — A100/H100 sind dafür ideal.
> Das **RL-Fine-tuning** ([RL-Plan](../weiterfuehrend/reinforcement-learning-plan.md)) jedoch **nicht**, und zwar aus einem Hardware-Grund:
>
> - RL trainiert die Policy **in der Sim** und braucht dafür pro Schritt **kamerabasiertes
>   Rendering** der 4 Beobachtungskameras. Isaac Sim rendert diese Kamerabilder per **Raytracing**,
>   das **RT-Cores** auf der GPU voraussetzt.
> - Die KISSKI-Trainings-GPUs **A100 und H100 haben keine RT-Cores** — sie sind reine Compute-GPUs.
>   Das bildbasierte RL-Rollout kann darauf nicht (effizient/stabil) rendern.
> - Die einzige RT-Core-fähige KISSKI-Karte (RTX 5000 auf der `jupyter`-Partition) ist eine
>   **Turing-GPU und damit zu alt für Isaac Sim 4.x**.
>
> → Es gibt auf KISSKI **keine** Partition, die bildbasiertes RL ausführen kann. Der realistische
> Weg ist **vast.ai** mit einer **L40 / RTX 4090 / A6000** (Ampere+ **mit** RT-Cores) — siehe
> [reinforcement-learning-plan.md §3.5/§6](../weiterfuehrend/reinforcement-learning-plan.md). Das
> Vorlage-Skript [`Training/kisski_rl_submit.sh`](../../Training/kisski_rl_submit.sh) enthält
> deshalb einen **RT-Core-Guard**, der auf A100/H100 bewusst abbricht.
>
> `USE_RL=1` an `kisski_submit.sh` zu übergeben bewirkt nichts Sinnvolles: Der BC-Entrypoint
> bricht mit genau diesem Hinweis ab (das BC-Image enthält ohnehin kein Isaac Sim).

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

> Wo [`kisski_submit.sh`](../../Training/kisski_submit.sh) das SIF sucht, steht oben unter
> [Pfad-Konfiguration](#pfad-konfiguration--zwei-variablen-sonst-nichts): zuerst `$KISSKI_SIF_DIR`
> (Default `$HOME/images`), dann `$KISSKI_PROJECT_DIR/images`. Wer das Image wie oben nach
> `$HOME/images/` legt, ist damit bereits fertig — kein weiterer Handgriff nötig.

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

## Schritt 2b — Repos einmalig auf dem Login-Knoten klonen

`kisski_submit.sh` bind-mountet zwei Klone in den Container und **bricht ab**, wenn sie fehlen
(`FEHLER: …/Training/scripts nicht gefunden` bzw. `…/examples/G1_DEX3 nicht gefunden`). Die
Compute-Nodes haben keinen Internet-Zugang, deshalb muss das auf dem Login-Knoten passieren:

```bash
# 1) Dieses Repo (liefert Training/scripts/ — Änderungen wirken ohne Image-Rebuild)
git clone --branch training-luca-KISSKI --depth 1 \
    https://github.com/Docboter/projektarbeit_humanoider_roboter.git \
    /mnt/vast-kisski/projects/kisski-humrob/repo

# 2) Der GR00T-Fork (liefert examples/G1_DEX3 + LeRobot-Konverter — im Image fehlen sie)
git clone --branch luca/g1-dex3 --depth 1 \
    https://github.com/lucam06/Isaac-GR00T.git \
    /mnt/vast-kisski/projects/kisski-humrob/repo-groot
```

Pfade und Branch sind über `REPO_DIR`, `GITHUB_BRANCH` und `GROOT_FORK_DIR` überschreibbar.

---

## Schritt 3 — Tokens setzen und Job einreichen

> **Geführter Weg auf dem Login-Knoten:** `./Training/kisski_menu.sh` fragt Partition,
> Walltime und die Trainingsparameter ab und baut daraus die `sbatch`-Zeile
> (`--dry-run` zeigt sie nur an). Es setzt dabei immer `--export=ALL` — ohne das kommen
> die Variablen wegen `SBATCH_EXPORT=none` **nicht** im Job an, siehe unten. Das Menü ist
> ein Zusatz: `kisski_submit.sh` bleibt weiterhin allein auf den Cluster kopierbar.
> Details: [cli-menuefuehrung.md](../weiterfuehrend/cli-menuefuehrung.md)


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

## Variante — Weitere optionale Schalter

Zusätzlich zu `TUNE_VISUAL` lassen sich diese Verfahren **getrennt** kombinieren (Details in
[env-vars.md](env-vars.md)). Alle werden über `kisski_submit.sh` durchgereicht:

```bash
# 80/20-Train-Test-Split (Test-Episoden werden NICHT mittrainiert):
TRAIN_TEST_SPLIT=1 sbatch --export=ALL Training/kisski_submit.sh

# Bild-Augmentierung / Domain-Randomization abschalten (Default ist AN):
USE_AUGMENTATION=0 sbatch --export=ALL Training/kisski_submit.sh
```

- **`TRAIN_TEST_SPLIT=1`** — [`lib_split.sh`](../../Training/scripts/lib_split.sh) patcht vor dem
  Start `meta/info.json` auf `train: 0:N` / `test: N:total` (Ratio via `TRAIN_SPLIT_RATIO`,
  Default 0.8) und legt ein `split.json`-Protokoll im `OUTPUT_DIR` ab. Die Test-Episoden stehen
  danach für die Checkpoint-Auswahl auf **ungesehenen** Episoden bereit. Siehe
  [Train-Test-Split](train-test-split.md).
- **`USE_AUGMENTATION`** (Default `1`) — Color-Jitter/Domain-Randomization gegen den
  Sim-zu-Real-Domain-Gap; Stärken über `CJ_BRIGHTNESS/CONTRAST/SATURATION/HUE`. `0` = explizit aus.

> Ältere Images vor 2026-08-13 reichten `TRAIN_TEST_SPLIT`/`USE_AUGMENTATION` nicht durch —
> siehe [docs/historie.md](../historie.md).

> ⚠️ **Jeder neue Lauf braucht einen neuen `EXPERIMENT_NAME`** — sonst setzt der Fork
> unbemerkt am alten Checkpoint fort statt neu zu trainieren und meldet sich bei erreichtem
> `MAX_STEPS` sofort mit den alten Gewichten als „fertig". Details, Hintergrund und die
> Job-15271760-Geschichte: [env-vars.md § Namespace des Laufs](env-vars.md#namespace-des-laufs---der-fork-setzt-ungefragt-fort).
> Seit [`lib_resume_guard.sh`](../../Training/scripts/lib_resume_guard.sh) bricht das vorher ab;
> bewusstes Fortsetzen nach Walltime-Abbruch geht mit `RESUME=1`.

Nach dem Lauf steht die Checkpoint-Auswahl an — nicht blind den letzten Step nehmen:

```bash
RUN_DIR=/data/g1_dex3_finetune/blockstacking_vision sbatch Training/kisski_open_loop_eval.sh
```

> **`USE_RL` gehört NICHT hierher** — RL läuft auf KISSKI grundsätzlich nicht (kein RT-Core-Rendering,
> siehe [den Hinweis-Kasten oben](#verfügbare-gpu-partitionen)). Für RL den vast.ai-Pfad nutzen.

---

## Job-Status verfolgen

```bash
# Eigene Jobs anzeigen
squeue -u $USER

# Eigene Jobs mit prognostizierter Startzeit anzeigen (start spätestens zu...)
squeue -u $USER --start

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
    "${KISSKI_SIF_DIR:-$HOME/images}/projekt-humanoider-roboter.sif"
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
#SBATCH -G A100:4          # Anzahl GPUs — bei kisski-h100 auf H100:4 ändern
#SBATCH -c 96              # CPU-Kerne
#SBATCH --mem=384G         # RAM
#SBATCH -t 48:00:00        # Walltime (max. 48h)
```

> Das sind die **aktuellen Multi-GPU-Werte** (4× A100). Für einen 1-GPU-Lauf lassen sich `-G`,
> `-c` und `--mem` entsprechend reduzieren; Hintergrund und Messung in
> [multi-gpu.md](multi-gpu.md) — mit 4 GPUs reichten 256 GB RAM nicht (OOM-Kill der Dataloader).

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

Gesucht wird in dieser Reihenfolge: `$KISSKI_SIF_DIR` (Default `$HOME/images`), dann
`$KISSKI_PROJECT_DIR/images`. Die Fehlermeldung nennt den ersten Kandidaten. Entweder liegt das
Image an einem dritten Ort — dann `export KISSKI_SIF_DIR=…` bzw. `SIF_IMAGE=…` setzen —, oder es
wurde noch nicht erstellt. Dann einmalig auf dem Login-Knoten:
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
prüfen. Alte Checkpoints unter `.../data/g1_dex3_finetune/` aufräumen — `SAVE_TOTAL_LIMIT`
(Default `10`, auf KISSKI `40`, siehe [env-vars.md](env-vars.md)) begrenzt, wie viele
Checkpoints gleichzeitig vorgehalten werden; für weniger Speicherbedarf niedriger setzen.
