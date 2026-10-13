# HPC-Training auf KISSKI

> **TL;DR:** Anleitung für HPC-Training auf dem KISSKI-Cluster (Apptainer + SLURM statt
> Docker): SIF-Konvertierung, Pfad-Konfiguration, GPU-Partitionen, Monitoring und
> Checkpoint-Export. Für lange Trainingsläufe auf A100/H100.

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
KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> sbatch --export=ALL Training/kisski_submit.sh
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

`kisski_submit.sh` verwendet standardmäßig `-p kisski` mit `-G A100:4`. **Über
[`kisski_menu.sh`](../../Training/kisski_menu.sh)** (seit 2026-10-13) wählt man die Partition, und
das Menü gibt die passende GPU-Anforderung mit: `--gpus=A100:$NUM_GPUS` bzw. `--gpus=H100:$NUM_GPUS`
(bei Skripten mit `--gres` entsprechend `--gres=gpu:…`; leeres `NUM_GPUS` = die 4 aus dem Skript).
Die `#SBATCH`-Zeile von Hand ändern muss nur, wer `sbatch` direkt aufruft. `-c 96` und
`--mem=384G` bleiben dabei fest — das Menü passt sie nicht an.

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
>   **Turing-GPU und damit zu alt für Isaac Sim 5.1/6.0**.
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

**Nur für `GROOT_VERSION=1.7` — eigenes Image, eigene SIF.** Seit 2026-09-25 ist N1.7 ein
eigenes Docker-Image (`Training/update_image.sh --groot=1.7`, Tag `:latest-n17`); `kisski_submit.sh`
und `kisski_open_loop_eval.sh` wählen die passende SIF automatisch anhand von `GROOT_VERSION`:

```bash
apptainer pull $HOME/images/projekt-humanoider-roboter-n17.sif \
    docker://lucam03/projekt-humanoider-roboter:latest-n17
```

> **`/tmp` auf dem Login-Knoten ist ein gemeinsames tmpfs (63 GB, RAM).** Apptainer legt die
> Layer beim Pull über `TMPDIR` dort ab, nicht über `APPTAINER_TMPDIR`. Der N1.7-Pull brach
> deshalb am 2026-10-11 mit `no space left on device` ab. Vor dem Pull `TMPDIR` auf den
> Projektspeicher setzen:
> `export TMPDIR=/mnt/vast-kisski/projects/kisski-humrob/apptainer-tmp APPTAINER_TMPDIR=$TMPDIR`.
> Die Warnung „`nodev` mount option set on /mnt/vast-kisski“ ist dabei harmlos.

> Wo [`kisski_submit.sh`](../../Training/kisski_submit.sh) das SIF sucht, steht oben unter
> [Pfad-Konfiguration](#pfad-konfiguration--zwei-variablen-sonst-nichts): zuerst `$KISSKI_SIF_DIR`
> (Default `$HOME/images`), dann `$KISSKI_PROJECT_DIR/images`. Wer das Image wie oben nach
> `$HOME/images/` legt, ist damit bereits fertig — kein weiterer Handgriff nötig.

---

## Schritt 2 — Daten auf VAST-Projekt-Storage übertragen (einmalig oder bei Update)

Der Cluster nutzt für dieses Projekt den **GWDG VAST-Projekt-Storage** unter
`/mnt/vast-kisski/projects/kisski-humrob/data/` (persistente SSD-Storage — kein Scratch!).
`kisski_submit.sh` setzt diesen Pfad als Standard-`DATA_DIR`.

> **Begriffsklärung:** VAST ist das Storage-System der GWDG — nicht zu verwechseln mit
> **vast.ai**, der Cloud-GPU-Plattform, die an anderer Stelle in diesem Dokument für den
> RT-Core-Pfad (RL-Fine-tuning) erwähnt wird.

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
SKIP_DOWNLOAD=0 HF_TOKEN=hf_... WANDB_API_KEY=... sbatch --export=ALL Training/kisski_submit.sh
```

Der Entrypoint lädt Modell und Datensatz dann selbst von HuggingFace (~25 GB, ca. 10–30 min).
Bei allen weiteren Läufen ist `SKIP_DOWNLOAD=1` korrekt (Download wird automatisch übersprungen).

> Zur Einordnung gegen den Hinweis in Schritt 2b unten („Compute-Nodes haben keinen
> Internet-Zugang"): Für `GROOT_VERSION=1.7` müssen Modell **und** das gated Backbone in jedem
> Fall vorab auf dem **Login-Knoten** in den HF-Cache geladen werden (siehe unten) — im Job
> selbst läuft N1.7 mit `HF_HUB_OFFLINE=1`.

---

## Schritt 2b — Repos einmalig auf dem Login-Knoten klonen

`kisski_submit.sh` bind-mountet zwei Klone in den Container und **bricht ab**, wenn sie fehlen
(`FEHLER: …/Training/scripts nicht gefunden` bzw. `…/examples/G1_DEX3 nicht gefunden`). Die
Compute-Nodes haben keinen Internet-Zugang, deshalb muss das auf dem Login-Knoten passieren:

```bash
# 1) Dieses Repo (liefert Training/scripts/ — Änderungen wirken ohne Image-Rebuild)
git clone --branch training-luca-IKR-IS6.0 --depth 1 \
    https://github.com/Docboter/projektarbeit_humanoider_roboter.git \
    /mnt/vast-kisski/projects/kisski-humrob/repo

# 2) Der GR00T-Fork (liefert examples/G1_DEX3 + LeRobot-Konverter — im Image fehlen sie)
git clone --branch luca/g1-dex3 --depth 1 \
    https://github.com/lucam06/Isaac-GR00T.git \
    /mnt/vast-kisski/projects/kisski-humrob/repo-groot

# 3) Fork auf den Commit stellen, den das Image pinnt (Training/Dockerfile, "ARG GROOT16_COMMIT=<sha>")
PIN=$(sed -nE 's/^ARG GROOT16_COMMIT=([0-9a-f]{40}).*/\1/p' \
      /mnt/vast-kisski/projects/kisski-humrob/repo/Training/Dockerfile)
git -C /mnt/vast-kisski/projects/kisski-humrob/repo-groot fetch --depth 1 origin "$PIN"
git -C /mnt/vast-kisski/projects/kisski-humrob/repo-groot checkout "$PIN"
```

Pfade und Branch sind über `REPO_DIR`, `GITHUB_BRANCH` und `GROOT_FORK_DIR` überschreibbar.
`GITHUB_BRANCH` ist seit 2026-10-13 nicht mehr fest eingetragen: Default ist der Branch, auf dem
der Klon gerade steht, sonst `main` — der `--branch` oben ist also nur der Startpunkt. Ein
`GITHUB_TOKEN` (private Repos) wird nur noch über einen Credential-Helper aus der Umgebung
gelesen und landet **nicht** mehr in `.git/config`.

> **Warum Schritt 3:** `kisski_submit.sh` und `kisski_open_loop_eval.sh` binden
> `repo-groot/gr00t` über den Code im Image. Steht der Fork auf dem Branch-HEAD, läuft auf
> KISSKI anderer Code als im Image (Stand 2026-10-11: HEAD `dd87740` „(in testing!) …“ statt
> Pin `9508b49`). Seit 2026-10-13 brechen beide Jobs in diesem Fall mit einer Anleitung ab;
> `GROOT_FORK_ALLOW_DRIFT=1` erlaubt einen absichtlich anderen Stand.

**Nur für `GROOT_VERSION=1.7` — zweiten Fork-Clone anlegen.** `kisski_submit.sh` bindet für
N1.7 zusätzlich den Fork-Branch `luca/g1-dex3-n17` ein (`GROOT17_FORK_DIR`, Default
`$KISSKI_PROJECT_DIR/repo-groot-n17`):

```bash
git clone --branch luca/g1-dex3-n17 --depth 1 \
    https://github.com/lucam06/Isaac-GR00T.git \
    /mnt/vast-kisski/projects/kisski-humrob/repo-groot-n17
```

Danach wie in Schritt 3 auf den Pin auschecken — hier `ARG GROOT17_COMMIT` im
`Training/Dockerfile`; die Pin-Prüfung in `kisski_submit.sh` nennt den Befehl:

```bash
PIN17=$(sed -nE 's/^ARG GROOT17_COMMIT=([0-9a-f]{40}).*/\1/p' \
      /mnt/vast-kisski/projects/kisski-humrob/repo/Training/Dockerfile)
git -C /mnt/vast-kisski/projects/kisski-humrob/repo-groot-n17 fetch --depth 1 origin "$PIN17"
git -C /mnt/vast-kisski/projects/kisski-humrob/repo-groot-n17 checkout "$PIN17"
```

**Nur für `GROOT_VERSION=1.7` — Modell + gated Backbone vorab auf dem Login-Knoten laden.**
Compute-Nodes haben kein Internet (siehe Hinweis weiter unten); N1.7 braucht zusätzlich zum
Basismodell das **gated** Backbone `nvidia/Cosmos-Reason2-2B`. `kisski_submit.sh` prüft vor
jedem Start, ob beide bereits im HF-Cache (`$DATA_DIR/hf_cache`) bzw. unter
`$DATA_DIR/models/GR00T-N1.7-3B` liegen, und bricht sonst mit genau diesen Befehlen ab:

```bash
module load apptainer

# Gated Backbone — HF_TOKEN muss vorher auf https://huggingface.co/nvidia/Cosmos-Reason2-2B
# freigeschaltet sein (gleiches Konto wie der Token):
export APPTAINERENV_HF_TOKEN="$HF_TOKEN"   # statt --env HF_TOKEN=…: Token nicht in ps/cmdline
apptainer exec --bind /mnt/vast-kisski/projects/kisski-humrob/data:/data \
    --env HF_HOME=/data/hf_cache \
    $HOME/images/projekt-humanoider-roboter-n17.sif \
    huggingface-cli download nvidia/Cosmos-Reason2-2B

# Basismodell
apptainer exec --bind /mnt/vast-kisski/projects/kisski-humrob/data:/data \
    $HOME/images/projekt-humanoider-roboter-n17.sif \
    huggingface-cli download nvidia/GR00T-N1.7-3B --local-dir /data/models/GR00T-N1.7-3B
```

Die heruntergeladene `config.json` bleibt unverändert (`model_name: "nvidia/Cosmos-Reason2-2B"`).
Den Zugang des Tokens prüft
`curl -I -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json`
(muss `200` liefern, nicht `401`).

Der Job selbst setzt danach `--env HF_HOME=/data/hf_cache --env HF_HUB_OFFLINE=1`, damit das
Training den vorab gefüllten Cache trifft, statt (erfolglos) online nachzuladen. **Auf KISSKI
getestet am 2026-10-12** (Job 16630159, 4 × A100, `TUNE_VISUAL=1`, Split, bs 32, 200 Steps:
bestanden). Voraussetzung ist der Offline-Fix im Fork (`build_processor` in
`gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`): transformers 4.57.3 fragt beim Laden des
Tokenizers sonst trotz `HF_HUB_OFFLINE=1` die Hub-API ab (`_patch_mistral_regex` →
`model_info()`) und bricht mit `OfflineModeIsEnabled` ab. Details und Messwerte:
[groot-n17-migration.md](../weiterfuehrend/groot-n17-migration.md#kisski-smoke-test-2026-10-12).

---

## Schritt 3 — Tokens setzen und Job einreichen

> **Geführter Weg auf dem Login-Knoten:** `./Training/kisski_menu.sh` fragt Partition,
> Walltime und die Trainingsparameter ab und baut daraus die `sbatch`-Zeile
> (`--dry-run` zeigt sie nur an). Es setzt dabei immer `--export=ALL` — ohne das kommen
> die Variablen wegen `SBATCH_EXPORT=none` **nicht** im Job an, siehe unten. Das Menü ist
> ein Zusatz: `kisski_submit.sh` bleibt weiterhin allein auf den Cluster kopierbar.
> Details: [cli-menuefuehrung.md](../weiterfuehrend/cli-menuefuehrung.md)


> **WICHTIG — `SBATCH_EXPORT=none` auf dem Login-Knoten.** KISSKI setzt diese Variable und
> überstimmt damit das `#SBATCH --export=ALL` im Skript: Variablen, die nur inline vor `sbatch`
> stehen, kommen **nicht** im Job an. Zwei Wege, die funktionieren:
>
> 1. **Token-Dateien (Standard, siehe Kopfkommentar von `kisski_submit.sh`):** einmalig
>    `~/.hf_token` und `~/.wandb_key` (mode 600) anlegen — das Skript liest sie selbst ein, danach
>    genügt `sbatch Training/kisski_submit.sh`.
> 2. **`--export=ALL` auf der Kommandozeile** (schlägt die Env-Variable); `kisski_menu.sh` setzt
>    es immer.

```bash
# Variante mit Token-Dateien (einmalig: printf 'hf_...\n' > ~/.hf_token && chmod 600 ~/.hf_token usw.):
sbatch Training/kisski_submit.sh

# Variante über die Umgebung — mit --export=ALL:
HF_TOKEN=hf_... \
WANDB_API_KEY=... \
sbatch --export=ALL Training/kisski_submit.sh
```

Mit optionalen Trainings-Overrides (ebenfalls vor `sbatch`, mit `--export=ALL`):

```bash
MAX_STEPS=175000 \
GLOBAL_BATCH_SIZE=32 \
sbatch --export=ALL Training/kisski_submit.sh
```

Für **GR00T N1.7** statt des Defaults `1.6` genügt eine zusätzliche Zeile im selben Prefix
(Voraussetzung: N1.7-SIF aus `:latest-n17` gepullt — siehe [Schritt 1](#schritt-1--image-einmalig-zu-sif-konvertieren-auf-dem-login-knoten)
— sowie zweiter Fork-Clone + Backbone/Modell vorab im Cache, siehe
[Schritt 2b](#schritt-2b--repos-einmalig-auf-dem-login-knoten-klonen) oben — **ungetestet**,
siehe [groot-n17-migration.md](../weiterfuehrend/groot-n17-migration.md#stand-der-umsetzung-2026-08-19)):

```bash
HF_TOKEN=hf_... \
WANDB_API_KEY=... \
GROOT_VERSION=1.7 \
sbatch Training/kisski_submit.sh
```

SLURM gibt die Job-ID aus, z. B. `Submitted batch job 12345678`.

> **`WANDB_API_KEY` ist optional:** der Entrypoint setzt `USE_WANDB=1` nur, wenn ein Key da ist,
> und trainiert sonst mit Warnung ohne W&B. Der Key wird jedoch gebraucht, um die Läufe später
> zu syncen (siehe [W&B-Offline-Sync](wandb-offline-sync.md)). Nur wer `run_finetuning.sh` direkt
> mit `USE_WANDB=1` ohne Key aufruft, bekommt einen Abbruch.
>
> **Tipp:** Tokens nicht dauerhaft in `.bashrc` speichern; die mode-600-Dateien (`~/.hf_token`,
> `~/.wandb_key`) sind der vorgesehene Weg.

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
- **Lernrate `1e-4`** statt der 4-GPU-Standard-`2e-4` (sqrt-skaliert). Begründung: Die Config hat *eine globale LR*
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

Zusätzlich zu `TUNE_VISUAL` lassen sich `TRAIN_TEST_SPLIT` und `USE_AUGMENTATION` **getrennt**
kombinieren, alle über `kisski_submit.sh` durchgereicht. Bedeutung und Defaults stehen in der
SSOT [env-vars.md](env-vars.md):

```bash
# 80/20-Train-Test-Split (Test-Episoden werden NICHT mittrainiert):
TRAIN_TEST_SPLIT=1 sbatch --export=ALL Training/kisski_submit.sh

# Bild-Augmentierung / Domain-Randomization abschalten (Default ist AN):
USE_AUGMENTATION=0 sbatch --export=ALL Training/kisski_submit.sh
```

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
RUN_DIR=/data/g1_dex3_finetune/blockstacking_vision sbatch --export=ALL Training/kisski_open_loop_eval.sh
```

**Mehrere Läufe hintereinander** (Warteschlange statt Einzel-Einreichen):
[`kisski_chain.sh`](../../Training/kisski_chain.sh) reicht je Lauf Training + Sweep ein, die
Trainings per `--dependency=afterany` nacheinander, jeder Sweep per `afterok` hinter seinem
Training. Die Läufe sind im Skript als Voreinstellungen hinterlegt (`--list`; aktuell `synth_jointspace` und
`synth_v22`, beides Co-Training mit dem Fremd-Datensatz, siehe [synth-datensatz.md](synth-datensatz.md)); Datensätze und
freie Namespaces werden vor dem Einreichen geprüft.

```bash
./Training/kisski_chain.sh --dry-run synth_jointspace synth_v22   # anzeigen
./Training/kisski_chain.sh synth_jointspace synth_v22             # einreichen
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
#SBATCH -G A100:4          # Anzahl GPUs — kisski_menu.sh überschreibt Typ/Anzahl passend zur Partition
#SBATCH -c 96              # CPU-Kerne
#SBATCH --mem=384G         # RAM
#SBATCH -t 48:00:00        # Walltime (max. 48h)
```

> Das sind die **aktuellen Multi-GPU-Werte** (4× A100). `kisski_menu.sh` setzt `-G` passend zu
> Partition und `NUM_GPUS`; `-c` und `--mem` bleiben fest und lassen sich für einen 1-GPU-Lauf
> nur im Skript (oder per eigener `sbatch`-Option) reduzieren; Hintergrund und Messung in
> [multi-gpu.md](multi-gpu.md) — mit 4 GPUs reichten 256 GB RAM nicht (OOM-Kill der Dataloader).

Trainings-Parameter werden entweder dauerhaft im Script geändert oder pro Lauf vor `sbatch`
übergeben (siehe [Schritt 3](#schritt-3--tokens-setzen-und-job-einreichen) — mit `--export=ALL`,
sonst landen sie wegen `SBATCH_EXPORT=none` nicht im Job):

```bash
GLOBAL_BATCH_SIZE=32 \
MAX_STEPS=50000 \
HF_TOKEN=hf_... WANDB_API_KEY=... \
sbatch --export=ALL Training/kisski_submit.sh
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
Bei `GROOT_VERSION=1.7` wird stattdessen `projekt-humanoider-roboter-n17.sif` aus
`:latest-n17` gesucht — siehe [Schritt 1](#schritt-1--image-einmalig-zu-sif-konvertieren-auf-dem-login-knoten).

### Job bleibt in Status `PD` (Pending)

Die Partition ist ausgelastet. Mit `squeue -p kisski` prüfen, wie viele Jobs warten.
Alternative: `kisski-h100`-Partition versuchen oder Walltime verkürzen (kürzere Jobs haben
höhere Priorität).

### `No space left on device` im Container

Der VAST-Projekt-Storage ist voll. Mit `du -sh /mnt/vast-kisski/projects/kisski-humrob/*`
prüfen. Alte Checkpoints unter `.../data/g1_dex3_finetune/` aufräumen — `SAVE_TOTAL_LIMIT`
(Default `10`, auf KISSKI `40`, siehe [env-vars.md](env-vars.md)) begrenzt, wie viele
Checkpoints gleichzeitig vorgehalten werden; für weniger Speicherbedarf niedriger setzen.
