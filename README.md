# Projektarbeit Humanoider Roboter
## GR00T N1.6 Fine-tuning — Unitree G1 mit DEX3-Hand

Dieses Repository dokumentiert das Fine-tuning von NVIDIAs **GR00T N1.6** Vision-Language-Action-Modell auf den Unitree G1 Roboter mit DEX3-Hand für die Aufgabe Block-Stacking.

Die Trainingsumgebung läuft **vollständig autonom in einem Docker-Container**: Image starten → Training läuft. Der gleiche Container läuft lokal *und* auf Cloud-GPU-Plattformen wie [vast.ai](https://vast.ai) — alle Schritte (Daten-Download, Konvertierung, Training) passieren im Container.

> **Speicher-Modell:** Es gibt **keinen persistenten Storage auf dem Host**. Daten, Checkpoints und Logs leben ausschließlich im Container-Filesystem. Auf vast.ai entspricht ein Container genau einer Instanz — wird die Instanz/Container zerstört, ist alles weg. Das Container-Filesystem ist die einzige Wahrheit.

---

## Inhaltsverzeichnis

1. [Schnellstart](#1-schnellstart)
2. [Voraussetzungen](#2-voraussetzungen)
3. [Cloud-Training auf vast.ai](#3-cloud-training-auf-vastai)
4. [Lokales Training](#4-lokales-training)
5. [Konfiguration über Env-Vars](#5-konfiguration-über-env-vars)
6. [Daten aus dem Container holen](#6-daten-aus-dem-container-holen)
7. [Training beobachten](#7-training-beobachten)
8. [Projektstruktur](#8-projektstruktur)
9. [Häufige Probleme](#9-häufige-probleme)
10. [Train-Test-Split](#10-train-test-split)

---

## 1. Schnellstart

**Auf vast.ai:**
1. Instanz mit Image `lucam03/projekt-humanoider-roboter:latest` starten
2. In "Docker options" setzen: `-e HF_TOKEN=hf_… -e WANDB_API_KEY=…`
3. Instanz startet → Container startet automatisch → Training läuft autonom

**Lokal (Linux / WSL2):**
```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
./setup_and_train_DockerHub-pull.sh
```

Detaillierte Schritt-für-Schritt-Anleitung: [Anleitung.md](Anleitung.md)

---

## 2. Voraussetzungen

### Hardware

| Anforderung | Minimum | Getestet mit |
|---|---|---|
| GPU | NVIDIA GPU, ≥ 8 GB VRAM | RTX 4070 Laptop (8 GB) |
| VRAM | 8 GB | 8 GB |
| RAM | 32 GB | 32 GB |
| Speicherplatz | 80 GB im Container-Filesystem | SSD empfohlen |

> **Hinweis:** Mit 8 GB VRAM `GLOBAL_BATCH_SIZE=8` (oder kleiner). Mit 16 GB VRAM `GLOBAL_BATCH_SIZE=16`.

### Software (nur für lokales Training)

| Software | Version | Installationslink |
|---|---|---|
| Docker | ≥ 4.x | https://www.docker.com/products/docker-desktop |
| NVIDIA Container Toolkit | aktuell | https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html |
| NVIDIA Treiber | ≥ 570 (CUDA 12.8) | https://www.nvidia.com/drivers |

Auf vast.ai sind diese vorinstalliert — du brauchst lokal nichts davon.

### Accounts

- **HuggingFace** mit Zugriff auf:
  - [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) (Lizenz akzeptieren!)
  - [`unitreerobotics/G1_Dex3_BlockStacking_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset)
- **WandB** (optional, kostenlos) für Trainings-Logging: https://wandb.ai

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

Oder direkt aus dem Container heraus per `huggingface-cli upload`, `rclone`, `scp` etc. — siehe [Abschnitt 6](#6-daten-aus-dem-container-holen).

---

## 4. Lokales Training

### Variante A: Mit dem Launcher-Skript (empfohlen)

```bash
git clone https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
git checkout training-luca

export HF_TOKEN=hf_...
export WANDB_API_KEY=...        # optional

./setup_and_train_DockerHub-pull.sh
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
git clone --recurse-submodules https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
docker build -t projektarbeit-humanoider-roboter .
```

Build dauert ~30 Minuten (PyTorch, flash-attn). Anschließend ein `docker run` wie in Variante B mit dem lokalen Image-Namen.

### Interaktive Shell (für Debugging)

```bash
docker run -it --name groot-debug --gpus all --ipc=host --shm-size=16g \
  lucam03/projekt-humanoider-roboter:latest bash
```

Wird beim Aufruf ein Befehl wie `bash` übergeben, wird der Entrypoint übersprungen und der Befehl direkt ausgeführt.

---

## 5. Konfiguration über Env-Vars

Alle Parameter werden über Umgebungsvariablen gesteuert:

| Variable | Default | Beschreibung |
|---|---|---|
| `HF_TOKEN` | — | **Pflicht.** HuggingFace-Token (Lese-Berechtigung reicht) |
| `WANDB_API_KEY` | — | Optional. W&B-Key. Ohne diesen läuft Training ohne W&B. |
| `MAX_STEPS` | `30000` | Anzahl Trainings-Steps |
| `GLOBAL_BATCH_SIZE` | `8` | Globale Batch-Size (8 für 8 GB VRAM, 16–32 für 16 GB) |
| `NUM_GPUS` | `1` | Anzahl genutzter GPUs |
| `WANDB_PROJECT` | `gr00t-g1-dex3` | W&B-Projektname |
| `DATA_DIR` | `/data` | Datenverzeichnis im Container |
| `SKIP_DOWNLOAD` | `0` | Auf `1` setzen, wenn Daten schon im Container sind |
| `SKIP_CONVERT` | `0` | Auf `1` setzen, wenn `modality.json` schon existiert |
| `SKIP_TRAIN` | `0` | Auf `1` setzen für reines Setup (fällt in Shell) |
| `SHELL_ON_ERROR` | `0` | Auf `1` setzen für Debug-Shell bei Fehler |

### VRAM-Richtwerte

| VRAM | `GLOBAL_BATCH_SIZE` | `MAX_STEPS` |
|---|---|---|
| 8 GB  | 8  | 30 000 |
| 16 GB | 16–32 | 50 000 |

Reduziere bei `CUDA out of memory` zuerst `GLOBAL_BATCH_SIZE`. Volle Parameter-Referenz: [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md)

---

## 6. Daten aus dem Container holen

Da alles im Container lebt, musst du Checkpoints und Logs **vor dem Destroy** exportieren.

### Lokal: `docker cp`

```bash
docker cp groot-train:/data/g1_dex3_finetune ./checkpoints
docker cp groot-train:/data/logs ./logs
```

Funktioniert auch, während der Container läuft.

### Auf vast.ai

Per SSH auf die Instanz, dort:

```bash
# Container-Name herausfinden
docker ps

# Auf Host-Filesystem kopieren
docker cp <container>:/data/g1_dex3_finetune /workspace/checkpoints

# Von dort z. B. nach S3 / HuggingFace / lokal per scp
```

### Direkt aus dem Training-Container heraus

Mit interaktiver Shell (siehe oben) z. B. nach HuggingFace pushen:

```bash
cd /data/g1_dex3_finetune/blockstacking
huggingface-cli upload <dein-namespace>/g1-dex3-blockstacking . --repo-type=model
```

---

## 7. Training beobachten

### WandB (empfohlen)

Mit gesetztem `WANDB_API_KEY` läuft das Logging automatisch. Live verfolgen unter https://wandb.ai → Projekt `gr00t-g1-dex3` (bzw. dein `WANDB_PROJECT`).

### Live-Logs im Terminal

`docker run -it …` zeigt stdout direkt. Bei `--detach` oder vast.ai:

```bash
docker logs -f groot-train
```

### Logs im Container

Werden in `/data/logs/finetune-<timestamp>.log` geschrieben:

```bash
docker exec groot-train tail -f /data/logs/finetune-*.log
```

### Checkpoints

```bash
docker exec groot-train ls -lht /data/g1_dex3_finetune/blockstacking/
```

---

## 8. Projektstruktur

```
/
├── Dockerfile                          # Container-Definition mit ENTRYPOINT
├── docker-compose.yml                  # Optional (lokale Dev-Variante)
├── setup_and_train_DockerHub-pull.sh   # Thin host-launcher (Linux/macOS/WSL2)
├── setup_and_train_DockerHub-pull.ps1  # Thin host-launcher (Windows PowerShell)
├── scripts/                            # In das Image kopiert
│   ├── entrypoint.sh                   # ENTRYPOINT — orchestriert Download → Convert → Train
│   ├── download_data.sh                # HuggingFace-Download
│   ├── run_finetuning.sh               # Trainings-Launcher (im Container)
│   └── run_finetuning.ps1              # Trainings-Launcher (Windows-Variante)
└── app/                                # Git-Submodule (im Image bereits geklont)
    └── Groot-1.6/                      # GR00T N1.6 + eigene G1/DEX3-Configs
        └── examples/G1_DEX3/
            ├── g1_dex3_config.py
            ├── modality_4cam.json
            ├── modality_2cam.json
            ├── FINETUNING_GUIDE.md
            └── SETUP_DOCUMENTATION.md
```

Im Container (zur Laufzeit unter `/data/`, **kein Host-Mount**):
```
/data/
├── models/GR00T-N1.6-3B/    # Modellgewichte (~6 GB)
├── unitreerobotics/          # Rohdatensatz (~18 GB)
├── g1_dex3_finetune/        # Checkpoints
└── logs/                     # Trainings-Logs
```

---

## 9. Häufige Probleme

### `docker: Error response from daemon: could not select device driver "nvidia"`

NVIDIA Container Toolkit fehlt. Auf vast.ai ist es vorinstalliert; lokal:

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

Entrypoint bricht ab, wenn kein Token gesetzt ist. Token erstellen auf https://huggingface.co/settings/tokens (Read-Berechtigung reicht) und per `-e HF_TOKEN=hf_...` mitgeben.

### Container existiert bereits (`Conflict. The container name "/groot-train" is already in use`)

Du hast schon einen Container aus einem früheren Lauf:
```bash
./setup_and_train_DockerHub-pull.sh --resume    # Daten + Checkpoints behalten
./setup_and_train_DockerHub-pull.sh --destroy   # alles verwerfen und neu starten
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

---

## 10. Train-Test-Split

Der Datensatz ist in einen Trainings- und einen Test-Split aufgeteilt (80/20), damit das Modell nach dem Fine-tuning auf ungesehenen Episoden bewertet werden kann.

| Split | Episoden | Anteil |
|-------|----------|--------|
| `train` | 241 | 80 % |
| `test` | 60 | 20 % |

Vollständige Beschreibung: [Train-Test-split.md](Train-Test-split.md)

---

## Weiterführende Dokumentation

- [Anleitung.md](Anleitung.md) — Schritt-für-Schritt-Anleitung zur Nutzung der Umgebung
- [Train-Test-split.md](Train-Test-split.md) — Implementierung und Nutzung des 80/20-Splits
- [`app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md`](app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md)
- [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md)
- [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/groot)
- [LeRobot](https://github.com/huggingface/lerobot)
