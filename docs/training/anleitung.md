# Anleitung — GR00T N1.6 Fine-tuning

Diese Anleitung beschreibt Schritt für Schritt, wie du das Fine-tuning von **GR00T N1.6** für den Unitree G1 mit DEX3-Hand startest. Die Umgebung ist als autonomer Docker-Container aufgebaut: **du gibst Env-Vars an, der Container macht den Rest.**

## Speicher-Modell — wichtig vorab

**Es gibt keinen persistenten Storage auf dem Host.** Daten, Checkpoints und Logs leben **ausschließlich im Container-Filesystem** unter `/data`. Das hat Konsequenzen:

- `docker stop` (oder `Ctrl+C`) → Container ist gestoppt, **Daten bleiben**. Mit `docker start -ai groot-train` wieder fortsetzen.
- `docker rm` (oder `--destroy` im Skript) → **alles weg**.
- **Niemals `docker run --rm`** mit diesem Image — das löscht den Container nach Stop und damit deine Trainingsergebnisse.
- Auf vast.ai entspricht ein Container genau einer Instanz: Stop/Start lässt den Container weiterleben, Destroy löscht alles.
- Vor dem Destroy → Checkpoints exportieren (siehe Abschnitt [Daten retten](#daten-retten)).

## Vier Wege zum Trainieren

- **Weg A — vast.ai** (oder andere Cloud-GPU-Anbieter): du brauchst nur das Image und einige Env-Vars
- **Weg B — Lokales Training mit dem Launcher-Skript**: bequem unter Linux/macOS/WSL2
- **Weg C — Lokales Training mit `docker run`**: maximale Kontrolle, kein Skript
- **Weg D — KISSKI HPC-Cluster**: langes Training auf A100/H100 per SLURM-Job

---

## Vorab: Accounts & Tokens

Diese brauchst du in jedem Fall:

1. **HuggingFace-Account** mit Zugriff auf:
   - [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) — Lizenz *vorher* auf der Seite akzeptieren!
   - [`unitreerobotics/G1_Dex3_BlockStacking_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset)
   - Token erstellen unter https://huggingface.co/settings/tokens (Lese-Berechtigung reicht)
   - → ergibt deinen `HF_TOKEN`, beginnt mit `hf_…`

2. **WandB-Account** (optional, aber empfohlen) für Trainings-Logging:
   - https://wandb.ai → Account anlegen → API-Key kopieren
   - → ergibt deinen `WANDB_API_KEY`

> Ohne `WANDB_API_KEY` läuft das Training trotzdem durch — nur ohne Online-Dashboard.

---

## Weg A — Cloud-Training auf vast.ai

Empfohlen, wenn du keinen eigenen NVIDIA-Rechner mit ≥ 8 GB VRAM hast oder schneller trainieren willst.

### A1. vast.ai-Konto vorbereiten

1. Account auf https://vast.ai erstellen, Guthaben aufladen (Training mit ~1 GPU kostet ~0,30–1 USD/h je nach GPU).
2. Optional: SSH-Key in deinem vast.ai-Profil hinterlegen, falls du dich später einloggen willst.

### A2. Instanz auswählen

Im vast.ai-Web-UI: **Search** → Filter setzen:

| Filter | Wert |
|---|---|
| GPU RAM | ≥ 8 GB (besser 16+) |
| Disk Space | **≥ 80 GB** (für Modell + Daten + Checkpoints — der Container braucht das alles im eigenen Filesystem) |
| CUDA | ≥ 12.8 |

Eine RTX 4090 (24 GB) ist ideal. Eine RTX 3090 / 4080 (≥ 16 GB) reicht ebenfalls.

### A3. Image und Env-Vars eintragen

Auf der Konfigurationsseite der Instanz:

- **Docker Image:** `lucam03/projekt-humanoider-roboter:latest`
- **Docker options / Environment Variables**:
  ```
  -e HF_TOKEN=hf_DEIN_TOKEN
  -e WANDB_API_KEY=dein_wandb_key
  -e MAX_STEPS=30000
  -e GLOBAL_BATCH_SIZE=8
  --shm-size=16g
  ```
  Bei 16 GB VRAM: `-e GLOBAL_BATCH_SIZE=16`. Bei 24 GB: `-e GLOBAL_BATCH_SIZE=32` möglich.

- **Kein zusätzliches Volume nötig.** Der Container hat seinen eigenen Filesystem-Space (dafür ist die Disk-Allocation gedacht). Wenn du die Instanz **stoppst** und später wieder **startest**, bleibt der Container-Zustand erhalten — erst beim **Destroy** ist alles weg.

### A4. Launch

Instanz starten. Über **Logs / Console** im UI siehst du:

```
==> GPU-Check
    GPU 0: NVIDIA GeForce RTX 4090 ...
 v   GPU verfügbar
==> HuggingFace-Token prüfen
 v   HF_TOKEN gesetzt
==> Schritt 1/3 — Download (Modell + Datensatz)
...
==> Schritt 3/3 — Fine-tuning starten
```

### A5. Training beobachten

- **WandB-Dashboard:** https://wandb.ai → Projekt `gr00t-g1-dex3` → Live-Charts (Loss, Lernrate, GPU-Auslastung)
- **vast.ai-Konsole:** zeigt stdout direkt
- **Checkpoints:** liegen unter `/data/g1_dex3_finetune/blockstacking/` **im Container**

### A6. Checkpoints sichern — VOR dem Destroy!

Per vast.ai SSH auf die Instanz und dann z. B. nach HuggingFace pushen:

```bash
# In den Container hineingehen
docker exec -it <container-name> bash

# Innerhalb des Containers:
cd /data/g1_dex3_finetune/blockstacking
huggingface-cli upload <dein-namespace>/g1-dex3-blockstacking . --repo-type=model
```

Oder per `docker cp` auf den vast.ai-Host und von dort per `scp` lokal runterziehen.

### A7. Aufräumen

Nach Abschluss des Trainings:
- **Stop** → die Instanz ist pausiert, du zahlst nur noch Storage
- **Destroy** → alles weg, keine Kosten mehr (Checkpoints VORHER sichern!)

---

## Weg B — Lokales Training mit dem Launcher-Skript

Empfohlen, wenn du eine eigene NVIDIA-GPU mit ≥ 8 GB VRAM hast.

### B1. Voraussetzungen installieren

| Software | Link |
|---|---|
| Docker | https://www.docker.com/products/docker-desktop |
| NVIDIA Treiber ≥ 570 | https://www.nvidia.com/drivers |
| NVIDIA Container Toolkit | https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html |
| Git | https://git-scm.com |

**Windows (WSL2):** Docker Desktop → Settings → *Resources → WSL Integration* aktivieren. NVIDIA Container Toolkit *innerhalb* der WSL2-Distro installieren.

### B2. Repository klonen

Submodule sind **nicht** mehr nötig — das Image bringt den Groot-1.6-Code selbst mit:

```bash
git clone https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
git checkout training-luca-IKR-IS6.0
```

### B3. Token setzen

```bash
export HF_TOKEN=hf_DEIN_TOKEN
export WANDB_API_KEY=dein_wandb_key      # optional
```

*(Unter Windows die Tokens ebenfalls in der WSL2-Bash exportieren — in PowerShell gesetzte
`$env:`-Variablen erreichen das Bash-Skript in B4 nicht.)*

### B4. Skript starten

Linux / macOS / WSL2:
```bash
./Training/setup_and_train_DockerHub-pull.sh
```

*(Die frühere Windows-PowerShell-Variante `setup_and_train_DockerHub-pull.ps1` wurde entfernt —
unter Windows das Bash-Skript in WSL2 ausführen.)*

Das Skript:
1. Prüft Docker und NVIDIA Container Toolkit
2. Prüft, ob ein Container namens `groot-train` schon existiert (Resume-Option!)
3. Lädt das Image `lucam03/projekt-humanoider-roboter:latest` von Docker Hub
4. Startet den Container **ohne `--rm`** und **ohne Host-Volume**

### B5. Optionen

```bash
./Training/setup_and_train_DockerHub-pull.sh --skip-pull       # Image schon lokal
./Training/setup_and_train_DockerHub-pull.sh --interactive     # Shell statt Training
./Training/setup_and_train_DockerHub-pull.sh --resume          # Bestehenden Container weiterlaufen lassen
./Training/setup_and_train_DockerHub-pull.sh --destroy         # Alten Container loeschen + neu starten
./Training/setup_and_train_DockerHub-pull.sh --dry-run         # Nur Befehle anzeigen
```

### B6. Konfiguration anpassen

Trainings-Parameter über Env-Vars vor dem Skriptaufruf:
```bash
export MAX_STEPS=50000
export GLOBAL_BATCH_SIZE=16
./Training/setup_and_train_DockerHub-pull.sh
```

### B7. Training pausieren und fortsetzen

```bash
# Mit Ctrl+C unterbrechen (oder `docker stop groot-train`)
# Später:
./Training/setup_and_train_DockerHub-pull.sh --resume
# Alternativ direkt:
docker start -ai groot-train
```

Der Container behält Daten und Checkpoints. Beim Resume sieht der Entrypoint, dass alle Daten vorhanden sind, überspringt Download + Konvertierung und springt direkt ins Training.

### B8. Ergebnisse holen

```bash
docker cp groot-train:/data/g1_dex3_finetune ./checkpoints
docker cp groot-train:/data/logs ./logs
```

Funktioniert sowohl bei laufendem als auch bei gestopptem Container.

### B9. Container loeschen

```bash
docker rm -f groot-train
# oder per Skript:
./Training/setup_and_train_DockerHub-pull.sh --destroy
```

> **Vorsicht:** Damit sind ALLE Daten weg — vorher `docker cp` ausführen, falls du etwas behalten willst.

---

## Weg D — HPC-Training auf KISSKI

Empfohlen für langes Training (> 30 000 Steps) oder wenn lokal keine ausreichende GPU vorhanden ist. KISSKI stellt A100 (80 GB) und H100 (94 GB) zur Verfügung. Der Cluster läuft **kein Docker**, sondern **Apptainer** als Container-Runtime und **SLURM** als Job-Scheduler.

Der Ablauf hat drei Stufen: **Docker-Image bauen & nach Docker Hub pushen** → **Image auf dem Login-Knoten zu Apptainer-`.sif` konvertieren** → **SLURM-Job einreichen**.

### D0. Docker-Image bauen und nach Docker Hub pushen

Der Cluster zieht das Image per `apptainer pull docker://lucam03/projekt-humanoider-roboter:latest` **direkt von Docker Hub**. Apptainer kann ein Image nur konvertieren, das dort bereits liegt — es baut nichts selbst. Deshalb muss das Docker-Image **vor** der SIF-Konvertierung existieren und aktuell sein.

**Wann ist dieser Schritt nötig?**

- **Überspringen,** wenn das Image auf Docker Hub bereits aktuell ist (Standardfall — du willst nur trainieren). Weiter mit [D1](#d1-image-einmalig-zu-sif-konvertieren).
- **Ausführen,** wenn du etwas am Image geändert hast: `Training/Dockerfile`, eines der `Training/scripts/*.sh`, oder den gepinnten GR00T-Commit. Diese Änderungen wirken **erst nach Rebuild + Push** — der Cluster bekommt sie sonst nicht.

> **Hinweis:** Der Build braucht **Docker auf deinem lokalen Rechner** (nicht auf dem Login-Knoten — dort läuft kein Docker). Du baust lokal, pushst nach Docker Hub und konvertierst dann auf dem Cluster.

**Variante 1 — mit dem Build-Skript (empfohlen):**

[`Training/update_image.sh`](../../Training/update_image.sh) prüft Docker-Login, baut und pusht in einem Rutsch:

```bash
docker login                            # einmalig
cd Training
./update_image.sh                       # Build + Push (aktueller Dockerfile-Stand)
./update_image.sh --update-commit       # zusätzlich neuesten GR00T-Commit ins Dockerfile eintragen
./update_image.sh --no-cache            # Build ohne Cache (z. B. nach flash-attn-Problemen)
./update_image.sh --skip-push           # nur lokal bauen, nicht pushen
./update_image.sh --dry-run             # nur Befehle anzeigen
```

Das Skript taggt das Image doppelt (`:latest` und `:<timestamp>`) und pusht beide.
*(Das frühere Windows-Pendant `update_image.ps1` wurde entfernt; unter Windows WSL2 verwenden.)*

**Variante 2 — manuell mit `docker` (Linux/macOS/WSL2):**

```bash
docker login                          # einmalig
# Build-Context ist Training/ (damit COPY scripts/ funktioniert). --platform für KISSKI-Kompatibilität:
docker build --platform linux/amd64 -t lucam03/projekt-humanoider-roboter:latest Training/
docker push lucam03/projekt-humanoider-roboter:latest
```

Der erste Build dauert ~30–60 min (PyTorch + flash-attn); danach greift der Docker-Cache. Das Dockerfile klont das GR00T-Submodul selbst und checkt einen **gepinnten Commit** aus — `git clone --recurse-submodules` vorab ist nicht nötig.

### D1. Image einmalig zu SIF konvertieren

Auf dem Login-Knoten wird das (frisch gepushte) Docker-Hub-Image in das Apptainer-Format umgewandelt:

```bash
# Auf dem Login-Knoten glogin-gpu.hpc.gwdg.de:
module load apptainer
apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
    docker://lucam03/projekt-humanoider-roboter:latest

# Job einreichen:
export HF_TOKEN=hf_...  WANDB_API_KEY=...  GLOBAL_BATCH_SIZE=32
sbatch Training/kisski_submit.sh
```

> Nach jedem neuen Push (Schritt D0) muss die `.sif`-Datei **neu erzeugt** werden, damit die Änderungen auf dem Cluster ankommen — `apptainer pull` überschreibt eine bestehende `.sif` nicht automatisch, ggf. vorher löschen oder `--force` verwenden.

Standardmäßig ist der Vision-Encoder eingefroren (es werden nur Projector + Diffusion-Action-Head
trainiert). Soll der **Vision-Encoder mittrainiert** werden, mit `TUNE_VISUAL=1` einreichen — LR
(`1e-4`), Warmup (`0.1`) und ein eigener Output-Namespace werden dann automatisch gesetzt:

```bash
TUNE_VISUAL=1 sbatch --export=ALL Training/kisski_submit.sh
```

**Weitere optionale Schalter** (getrennt kombinierbar, Details in [env-vars.md](env-vars.md)):

```bash
TRAIN_TEST_SPLIT=1 sbatch --export=ALL Training/kisski_submit.sh   # 80/20-Split, Test-Episoden held-out
USE_AUGMENTATION=0 sbatch --export=ALL Training/kisski_submit.sh   # Bild-Augmentierung aus (Default an)
```

- `TRAIN_TEST_SPLIT=1` schaltet den [80/20-Split](train-test-split.md) scharf (Test-Episoden werden
  nicht mittrainiert). **Ohne diesen Schalter gibt es hinterher nichts zu validieren** — die
  Checkpoint-Auswahl bleibt dann blind beim letzten Step, so wie in Lauf 1 und Lauf 2. Nach dem
  Lauf: `RUN_DIR=… sbatch Training/kisski_open_loop_eval.sh` wählt den Checkpoint mit der besten
  MSE auf den zurückgehaltenen Episoden.
- `USE_AUGMENTATION` steuert Color-Jitter/Domain-Randomization gegen den Sim-Real-Gap (Default an;
  Stärken über `CJ_BRIGHTNESS/CONTRAST/SATURATION/HUE`).
- **RL** (`USE_RL`) läuft **nicht** im BC-Image — es braucht den Isaac-Sim+GR00T-Container auf einer
  RT-Core-GPU. Siehe [RL-Plan](../weiterfuehrend/reinforcement-learning-plan.md) und
  [`Training/kisski_rl_submit.sh`](../../Training/kisski_rl_submit.sh).

→ **Vollständige Schritt-für-Schritt-Anleitung** (SIF-Konvertierung, VAST-Storage, Monitoring,
Checkpoint-Export, KISSKI-Troubleshooting): [HPC-Training auf KISSKI](kisski-hpc.md).
Details zur Vision-Variante: [Vision-Encoder mittrainieren](kisski-hpc.md#variante--vision-encoder-mittrainieren-tune_visual1).

---

## Weg C — Lokales Training mit `docker run`

Wenn du das Launcher-Skript überspringen willst.

```bash
docker pull lucam03/projekt-humanoider-roboter:latest

docker run --name groot-train --gpus all --ipc=host --shm-size=16g \
  -e HF_TOKEN=hf_DEIN_TOKEN \
  -e WANDB_API_KEY=dein_wandb_key \
  -e MAX_STEPS=30000 \
  -e GLOBAL_BATCH_SIZE=8 \
  -it lucam03/projekt-humanoider-roboter:latest
```

> **Niemals `--rm` mitgeben.** Das würde den Container nach Stop löschen und damit alle Daten verlieren.

Fortsetzen:
```bash
docker start -ai groot-train
```

Im Hintergrund laufen lassen (statt `-it`):
```bash
docker run -d --name groot-train --gpus all --ipc=host --shm-size=16g \
  -e HF_TOKEN=... -e WANDB_API_KEY=... \
  lucam03/projekt-humanoider-roboter:latest

docker logs -f groot-train       # Logs verfolgen
```

### Image selbst bauen statt zu pullen

```bash
git clone https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
git checkout training-luca-IKR-IS6.0
docker build -t projektarbeit-humanoider-roboter Training/   # Build-Context = Training/
```

Build dauert ~30 Minuten (PyTorch + flash-attn). Dann ein `docker run` wie oben mit dem lokalen Image-Namen.

---

## Daten retten

Alles im Container muss **vor dem Destroy** exportiert werden. Drei Wege:

### 1. `docker cp` auf den Host

```bash
docker cp groot-train:/data/g1_dex3_finetune ./checkpoints
docker cp groot-train:/data/logs ./logs
```

Geht bei laufendem oder gestopptem Container.

### 2. Aus dem Container heraus hochladen (HuggingFace)

```bash
docker exec -it groot-train bash
# Im Container:
cd /data/g1_dex3_finetune/blockstacking
huggingface-cli upload <dein-namespace>/g1-dex3-blockstacking . --repo-type=model
```

### 3. Auf vast.ai per SSH

Auf der vast.ai-Instanz per SSH:
```bash
docker ps                                                       # Container-Name finden
docker cp <container>:/data/g1_dex3_finetune /workspace/ckpt    # auf Host kopieren
# Von /workspace dann z. B. per scp lokal abholen
```

### 4. KISSKI: rsync vom Cluster

```bash
rsync -avz --progress \
    <username>@transfer.hpc.gwdg.de:/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/ \
    ./checkpoints/
```

---

## Konfigurationsreferenz (Env-Vars)

Alle Parameter werden über Umgebungsvariablen gesteuert — auf vast.ai, KISSKI und lokal
identisch. Die vollständige Tabelle (inkl. VRAM-Richtwerten und OOM-Hinweisen) steht in der
[Konfigurationsreferenz](env-vars.md).

Die wichtigsten:

| Variable | Default | Beschreibung |
|---|---|---|
| `HF_TOKEN` | — | **Pflicht.** HuggingFace-Token |
| `WANDB_API_KEY` | — | Optional. Ohne diesen läuft Training ohne W&B |
| `MAX_STEPS` | `20000` | Anzahl Trainings-Steps (KISSKI-Multi-GPU-Default: 44000) |
| `GLOBAL_BATCH_SIZE` | `8` | 8 für 8 GB VRAM, 16–32 für 16+ GB, 32+ für A100 80 GB |
| `SHELL_ON_ERROR` | `0` | `1` = bei Fehler in Shell fallen statt zu beenden |

Bei `CUDA out of memory`: zuerst `GLOBAL_BATCH_SIZE` halbieren.

---

## Was passiert intern?

Das Image enthält den GR00T-Code (`/app/Groot-1.6`), alle Python-Abhängigkeiten und die Skripte (`/scripts/`). Beim Containerstart führt `/scripts/entrypoint.sh` aus:

1. **GPU-Check** — bricht ab, wenn keine GPU sichtbar ist
2. **Token-Check** — bricht ab, wenn `HF_TOKEN` fehlt
3. **Download** — falls `/data/models/GR00T-N1.6-3B` und `/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset` leer sind: `huggingface-cli download` (~25 GB). Sonst übersprungen (idempotent).
4. **Konvertierung** — falls `modality.json` fehlt: `convert_v3_to_v2_standalone.py` + Kopieren der Modalitäts-Config. Sonst übersprungen.
5. **W&B-Login** — falls `WANDB_API_KEY` gesetzt: `wandb login` mit dem Key.
6. **Training** — `bash /scripts/run_finetuning.sh` → `uv run python gr00t/experiment/launch_finetune.py` mit den Parametern aus den Env-Vars.

Logs werden in `/data/logs/finetune-<timestamp>.log` geschrieben.

---

## Häufige Fragen

### Wo finde ich die Checkpoints?

Im Container unter `/data/g1_dex3_finetune/blockstacking/`. Zum Holen:
```bash
docker cp groot-train:/data/g1_dex3_finetune ./checkpoints
```

### Wie unterbreche ich das Training?

`Ctrl+C` (lokal) oder `docker stop groot-train`. Auf vast.ai: **Stop** der Instanz. Checkpoints werden alle 2000 Steps gespeichert (`SAVE_STEPS=2000`; im Vision-Lauf `TUNE_VISUAL=1` alle 1000), also geht maximal die letzte angefangene Periode verloren. **Wichtig:** der Container bleibt bestehen, Daten sind sicher.

### Wie setze ich das Training nach einem Abbruch fort?

Mit dem Launcher: `./Training/setup_and_train_DockerHub-pull.sh --resume`. Manuell: `docker start -ai groot-train`. Der Entrypoint sieht, dass Daten vorhanden sind, und überspringt Download + Konvertierung.

> **Wichtig — was beim Resume passiert:** `docker start` bzw. `--resume` startet den
> *Container* neu (Daten und Checkpoints bleiben erhalten), und **das Training setzt dabei
> automatisch am letzten Checkpoint fort** — nicht bei Step 0. Der Trainer löst das über
> `get_last_checkpoint(output_dir)` auf und stellt `TrainerState` (inkl. `global_step`) sowie
> den Optimizer wieder her ([`trainer.py:244`](../../app/Groot-1.6/gr00t/experiment/trainer.py));
> findet er **keinen** Checkpoint, loggt er eine Warnung und startet regulär bei Step 0. Nicht
> abschaltbar (kein `--resume_from_checkpoint`-CLI-Flag) — bewusst von vorn trainieren geht nur
> mit anderem `EXPERIMENT_NAME`/`OUTPUT_DIR` oder durch Wegräumen der vorhandenen
> `checkpoint-*`. Hintergrund und Fundstellen (kanonische Fassung, inkl. Job-15271760-Vorfall):
> [env-vars.md § Namespace des Laufs](env-vars.md#namespace-des-laufs---der-fork-setzt-ungefragt-fort).

### Was ist, wenn ich versehentlich `--rm` benutze?

Dann ist nach Stop alles weg. **Niemals `--rm` mit diesem Image verwenden.** Das Launcher-Skript schließt das aus.

### Wie ändere ich das Image?

1. Code/Skripte ändern.
2. `docker build -t lucam03/projekt-humanoider-roboter:latest Training/`
3. `docker push lucam03/projekt-humanoider-roboter:latest` (für vast.ai)

> `Training/scripts/*.sh` werden ins Image **kopiert** (kein Bind-Mount mehr). Änderungen erfordern einen Rebuild — sonst läuft die alte Version weiter.

### Wie komme ich in eine Shell im laufenden Container?

```bash
docker exec -it groot-train bash
```

Oder Container mit Shell-Override starten (überspringt den Entrypoint):
```bash
docker run -it --name groot-debug --gpus all --ipc=host --shm-size=16g \
  lucam03/projekt-humanoider-roboter:latest bash
```

### Wie lange darf der Container existieren?

Beliebig lange — solange du den Host nicht abschießt bzw. die vast.ai-Instanz nicht destroyst. Stop/Restart ist beliebig oft möglich.

### Training läuft, aber Loss bleibt hoch — was tun?

- `modality.json` korrekt? Sollte nach Schritt 4 in `/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/` liegen.
- Längeres Training nötig — siehe Empfehlungen in [`FINETUNING_GUIDE.md`](../../app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md).

### Wie evaluiere ich das fertige Modell?

Siehe [Train-Test-Split](train-test-split.md) und den Inference-Abschnitt in [`FINETUNING_GUIDE.md`](../../app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md).

---

## Weiterführende Dokumentation

- [Doku-Übersicht](../README.md) — Navigations-Hub aller Dokumente
- [Projekt-README](../../README.md) — Projekt-Übersicht & Schnellstart
- [HPC-Training auf KISSKI](kisski-hpc.md) — vollständige KISSKI-Anleitung
- [Konfigurationsreferenz (Env-Vars)](env-vars.md) — alle Parameter
- [Train-Test-Split](train-test-split.md) — 80/20-Datensatz-Split
- [W&B-Offline-Sync](wandb-offline-sync.md) — W&B auf KISSKI
- [`SETUP_DOCUMENTATION.md`](../../app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md)
- [`FINETUNING_GUIDE.md`](../../app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md)
