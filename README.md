# Projektarbeit Humanoider Roboter
## GR00T N1.6 Fine-tuning — Unitree G1 mit DEX3-Hand

Dieses Repository dokumentiert das Fine-tuning von NVIDIAs **GR00T N1.6** Vision-Language-Action-Modell auf den Unitree G1 Roboter mit DEX3-Hand für die Aufgabe Block-Stacking. Die Entwicklungsumgebung läuft vollständig in Docker.

---

## Inhaltsverzeichnis

1. [Voraussetzungen](#1-voraussetzungen)
2. [Repository klonen](#2-repository-klonen)
3. [Container bauen](#3-container-bauen)
4. [Daten herunterladen](#4-daten-herunterladen)
5. [Datensatz vorbereiten](#5-datensatz-vorbereiten)
6. [Fine-tuning starten](#6-fine-tuning-starten)
7. [Training beobachten](#7-training-beobachten)
8. [Projektstruktur](#8-projektstruktur)
9. [Häufige Probleme](#9-häufige-probleme)

---

## 1. Voraussetzungen

### Hardware

| Anforderung | Minimum | Getestet mit |
|---|---|---|
| GPU | NVIDIA GPU, ≥ 8 GB VRAM | RTX 4070 Laptop (8 GB) |
| VRAM | 8 GB | 8 GB |
| RAM | 32 GB | 32 GB |
| Speicherplatz | 60 GB frei | SSD empfohlen |

> **Hinweis:** Mit 8 GB VRAM muss `--global_batch_size 8` oder kleiner verwendet werden.
> Mit 16 GB VRAM kann `--global_batch_size 16` genutzt werden.

### Software

| Software | Version | Installationslink |
|---|---|---|
| Docker Desktop | ≥ 4.x | https://www.docker.com/products/docker-desktop |
| NVIDIA Container Toolkit | aktuell | https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html |
| NVIDIA Treiber | ≥ 570 (CUDA 12.8) | https://www.nvidia.com/drivers |
| Git | ≥ 2.34 | https://git-scm.com |
| Git LFS | aktuell | https://git-lfs.com |

### Accounts

- **HuggingFace-Account** mit Zugriff auf:
  - [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) (Lizenz akzeptieren!)
  - [`unitreerobotics/G1_Dex3_BlockStacking_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset)
- **WandB-Account** (kostenlos) für Trainings-Logging: https://wandb.ai

---

## 2. Repository klonen

```bash
# Mit allen Submodulen klonen
git clone --recurse-submodules https://github.com/Docboter/projektarbeit_humanoider_roboter.git

cd projektarbeit_humanoider_roboter
git checkout training-luca
```

Falls das Repo bereits geklont wurde, Submodule nachträglich initialisieren:

```bash
git submodule update --init --recursive
```

### Submodule im Überblick

| Pfad | Quelle | Inhalt |
|---|---|---|
| `app/Groot-1.6/` | `lucam06/Isaac-GR00T` (Branch `luca/g1-dex3`) | GR00T N1.6 + eigene G1/DEX3-Configs |

---

## 3. Container bauen

```bash
# Image bauen (beim ersten Mal ~30 Minuten wegen PyTorch/flash-attn)
docker compose build
```

Was dabei passiert:
- Basis: `nvidia/cuda:12.8.0-devel-ubuntu22.04`
- Python 3.10, PyTorch 2.7.1+cu128, flash-attn werden installiert
- Alle Abhängigkeiten kommen aus dem eingefrorenen `uv.lock` → **exakt reproduzierbar**
- EGL/Vulkan wird für headless Rendering konfiguriert (MuJoCo, PyOpenGL)

> **Docker Desktop unter Windows (WSL2):** In den Docker-Einstellungen unter
> *Resources → WSL Integration* muss die WSL2-Distro aktiviert sein. Außerdem muss
> das NVIDIA Container Toolkit innerhalb der WSL2-Distro installiert sein.

---

## 4. Daten herunterladen

### HuggingFace-Token setzen

Da jeder `docker compose run --rm`-Aufruf einen frischen Container startet, muss der Token als Umgebungsvariable übergeben werden (nicht per `huggingface-cli login`):

```bash
# Token von https://huggingface.co/settings/tokens (read-Berechtigung reicht)
export HF_TOKEN=hf_...
```

Unter Windows (PowerShell):

```powershell
$env:HF_TOKEN = "hf_..."
```

### Download starten (~25 GB, einmalig)

```bash
docker compose run --rm groot-training bash /scripts/download_data.sh
```

Dies lädt herunter:

| Datensatz / Modell | Größe | Ziel im Container |
|---|---|---|
| `nvidia/GR00T-N1.6-3B` | ~6 GB | `/data/models/GR00T-N1.6-3B` |
| `unitreerobotics/G1_Dex3_BlockStacking_Dataset_v3.0` | ~18 GB | `/data/unitreerobotics/` |
| Dataset-Metadaten (ohne Videos) | ~1 MB | `/data/G1_Dex3_BlockStacking/meta/` |

Die Daten werden im lokalen `./data/`-Ordner gespeichert und beim nächsten `docker compose up`
automatisch wiederverwendet.

---

## 5. Datensatz vorbereiten

Der Datensatz liegt im LeRobot-Format v3.0, GR00T N1.6 benötigt v2.1. Dieser Schritt
muss **einmalig** ausgeführt werden:

```bash
docker compose run --rm groot-training bash -c "
  cd /app/Groot-1.6

  # 1. LeRobot v3.0 → v2.1 konvertieren
  python scripts/lerobot_conversion/convert_v3_to_v2_standalone.py \
    --repo-id unitreerobotics/G1_Dex3_BlockStacking_Dataset \
    --root /data

  # 2. GR00T-Modalitäts-Metadaten hinzufügen (4-Kamera-Konfiguration)
  cp examples/G1_DEX3/modality_4cam.json \
     /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset_v3.0/meta/modality.json
"
```

Konvertierung erfolgreich, wenn diese Datei existiert:
```
/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset_v3.0/meta/modality.json
```

---

## 6. Fine-tuning starten

```bash
docker compose run --rm groot-training bash -c "
  cd /app/Groot-1.6
  export NUM_GPUS=1

  CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path     /data/models/GR00T-N1.6-3B \
    --data-config         examples/G1_DEX3/modality_4cam.json \
    --dataset-path        /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset_v3.0 \
    --embodiment-tag      NEW_EMBODIMENT \
    --embodiment-config-module examples.G1_DEX3.g1_dex3_config \
    --output-dir          /data/g1_dex3_finetune/blockstacking/mein_run \
    --num-gpus            \${NUM_GPUS} \
    --global_batch_size   8 \
    --max_steps           30000 \
    --report_to           wandb
"
```

### Parameter anpassen

| Parameter | 8 GB VRAM | 16 GB VRAM | Beschreibung |
|---|---|---|---|
| `--global_batch_size` | 8 | 16–32 | Kleinere Werte bei OOM |
| `--max_steps` | 30 000 | 50 000 | Mehr Steps = besser, aber länger |
| `--learning_rate` | 1e-4 | 1e-4 | Standard-Wert |
| `--num-gpus` | 1 | 1–4 | Anzahl GPUs |

Detaillierte Erklärung aller Parameter: [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md)

---

## 7. Training beobachten

### WandB (empfohlen)

Beim ersten Start nach dem WandB-Login-Code fragen lassen oder Token vorab setzen:

```bash
docker compose run --rm -e WANDB_API_KEY=<dein-token> groot-training ...
```

Training live verfolgen unter: https://wandb.ai → Projekt `gr00t-g1-dex3`

### Lokal im Terminal

```bash
# Checkpoint-Verzeichnis beobachten
watch -n 30 'ls -lht /data/g1_dex3_finetune/blockstacking/mein_run/ | head -10'
```

---

## 8. Projektstruktur

```
/
├── Dockerfile                          # Reproduzierbare Container-Definition
├── docker-compose.yml                  # GPU-Setup, Volume-Mounts
├── scripts/
│   └── download_data.sh                # Modell & Datensatz von HuggingFace laden
├── app/                                # Submodul (via git clone --recurse-submodules)
│   └── Groot-1.6/                      # GR00T N1.6 + eigene G1/DEX3-Erweiterungen
│       └── examples/G1_DEX3/
│           ├── g1_dex3_config.py       # Embodiment-Konfiguration (Gelenke, Kameras)
│           ├── modality_4cam.json      # Modalitäts-Config (4 Kameras + DEX3-Hand)
│           ├── modality_2cam.json      # Modalitäts-Config (2 Kameras, leichtgewichtig)
│           ├── FINETUNING_GUIDE.md     # Schritt-für-Schritt Fine-tuning Anleitung
│           └── SETUP_DOCUMENTATION.md  # Vollständige Setup-Dokumentation
└── data/                               # NICHT im Git (lokal, per Volume gemountet)
    ├── models/GR00T-N1.6-3B/           # Modellgewichte (~6 GB)
    ├── unitreerobotics/                 # Rohdatensatz (~18 GB)
    ├── G1_Dex3_BlockStacking/          # Dataset-Metadaten
    └── g1_dex3_finetune/               # Fine-tuning Outputs & Checkpoints
```

---

## 9. Häufige Probleme

### `docker: Error response from daemon: could not select device driver "nvidia"`

NVIDIA Container Toolkit ist nicht installiert oder nicht für Docker konfiguriert.

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### `CUDA out of memory`

```bash
# global_batch_size halbieren
--global_batch_size 4
```

---

### `FileNotFoundError: Modality config path does not exist`

Fine-tuning muss aus `/app/Groot-1.6` gestartet werden (steht im Befehl oben bereits so).

---

### `KeyError: annotation.human.task_description not found`

`modality.json` fehlt im Datensatz:

```bash
docker compose run --rm groot-training \
  cp /app/Groot-1.6/examples/G1_DEX3/modality_4cam.json \
     /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/modality.json
```

---

### HuggingFace-Download bricht ab

Mit `--resume-download` fortsetzen:

```bash
huggingface-cli download nvidia/GR00T-N1.6-3B \
  --local-dir /data/models/GR00T-N1.6-3B \
  --resume-download
```

---

## Weiterführende Dokumentation

- [`app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md`](app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md) — Vollständige Analyse und Setup-Schritte
- [`app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md`](app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md) — Detaillierter Fine-tuning Guide inkl. Inference & Troubleshooting
- [NVIDIA Isaac GR00T Dokumentation](https://developer.nvidia.com/isaac/groot)
- [LeRobot Dokumentation](https://github.com/huggingface/lerobot)
