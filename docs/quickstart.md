# Quickstart

> **TL;DR:** Der schnellste Weg vom frischen Checkout zu laufendem Training bzw. laufender Simulation: Systemvoraussetzungen, Host-Software, Accounts/Tokens und die kürzeste Befehlskette pro Weg. Details stehen bewusst nicht hier, sondern in den verlinkten Anleitungen ([Training](training/anleitung.md), [Simulation](simulation/sim-eval-anleitung.md), [KISSKI](training/kisski-hpc.md)).

Alles läuft in Containern: **CUDA 12.8, Python 3.10 und `uv` stecken im Image** — auf dem
Host werden nur Docker, der NVIDIA-Treiber und Git gebraucht. Wer nur trainieren will,
braucht weder Isaac Sim noch ein lokales Python.

## Systemvoraussetzungen

| | Training (lokal) | Simulation | KISSKI (HPC) |
|---|---|---|---|
| **GPU** | NVIDIA, ≥ 8 GB VRAM (16+ besser); volles Fine-Tuning ≥ 40 GB | Ampere oder neuer **mit RT-Cores** (L40/L40S, RTX 4090, RTX 6000 Ada, A6000), ≥ 24 GB VRAM. A100/H100 (keine RT-Cores) und Turing-Karten scheiden aus | A100 80 GB (Partition `kisski`) / H100 94 GB (`kisski-h100`), max. 48 h Walltime |
| **Speicherplatz** | ≥ 80 GB frei (Image + Modell + Datensatz + Checkpoints) | ≥ 60 GB frei | VAST-Projektspeicher vorhanden |
| **OS** | Linux; Windows nur über WSL 2 | Linux | Login per SSH |
| **Treiber** | NVIDIA ≥ 570 | NVIDIA ≥ 570 | — |

Batch-Größen je VRAM-Klasse: Tabelle in [training/env-vars.md](training/env-vars.md).
Die Simulation läuft **nicht** auf KISSKI (A100/H100 haben keine RT-Cores) — dafür gibt es
den IKR-Server oder vast.ai, siehe [simulation/README.md](simulation/README.md).

## Host-Software installieren

```bash
# Docker Engine (Ubuntu; andere Distros: https://docs.docker.com/engine/install/)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER        # danach neu einloggen

# NVIDIA Container Toolkit
# (Repo-Setup: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Funktionstest: sieht ein Container die GPU?
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
# (--rm ist bei diesem Wegwerf-Test ok — beim Trainings-Container ist es verboten,
#  weil dort alle Daten im Container leben, s. training/anleitung.md, Speicher-Modell)
```

- Lokaler **Image-Build** (optional, normalerweise kommt das fertige Image von Docker Hub): Docker ≥ 23 (BuildKit).
- **Windows:** Docker Desktop mit WSL-2-Backend; einen nativen Windows-Pfad gibt es nicht mehr.
- **KISSKI:** nichts zu installieren — `module load apptainer` auf dem Login-Node genügt ([training/kisski-hpc.md](training/kisski-hpc.md)).

## Accounts und Tokens

| Was | Wofür | Pflicht? |
|---|---|---|
| Hugging-Face-Token (`hf_...`) | Modell- und Datensatz-Download. Vorher auf huggingface.co die Lizenz von `nvidia/GR00T-N1.6-3B` akzeptieren und Zugriff auf `unitreerobotics/G1_Dex3_BlockStacking_Dataset` sicherstellen | **Ja** (jeder Weg) |
| W&B-API-Key | Trainings-Monitoring | Lokal optional; auf KISSKI Pflicht (außer `USE_WANDB=0`) |
| vast.ai-Konto + Guthaben | Miet-GPU für Sim-Eval bzw. Training | Nur Weg vast.ai |
| Docker-Hub-Login, NGC-API-Key (nvcr.io) | Eigene Images bauen und pushen | Nur Maintainer |

Drei Konten-Namensräume tauchen im Projekt auf, nicht verwechseln:
`lucam03` = Docker-Hub-Konto (Images), `lucam06` = GitHub-Konto (Isaac-GR00T-Fork),
`luca-mue` = Hugging-Face-Konto (Checkpoints/Datasets).

## Repo holen

```bash
git clone https://github.com/Docboter/projektarbeit_humanoider_roboter.git
cd projektarbeit_humanoider_roboter
git checkout training-luca-IKR-IS6.0          # aktueller Arbeits-Branch
git submodule update --init data/unitree_ros  # nur für die USD-Generierung (Sim) nötig
```

Das GR00T-Submodul (`app/Groot-1.6`) muss **nicht** initialisiert werden — das Dockerfile
klont es beim Build selbst (gepinnter Commit).

## Los geht's

Der bequemste Einstieg ist überall das geführte Menü:

```bash
./run.sh          # fragt Domäne (Simulation / Training / KISSKI) und Aktion ab
```

Oder direkt — pro Weg die kürzeste Kette:

**Training lokal** (zieht das fertige Image von Docker Hub):

```bash
export HF_TOKEN=hf_...            # Pflicht
export WANDB_API_KEY=...          # optional
./Training/setup_and_train_dockerhub_pull.sh
```

→ Alle vier Trainingswege, Speicher-Modell und FAQ: [training/anleitung.md](training/anleitung.md)

**Simulation auf dem eigenen Server** (GPU mit RT-Cores):

```bash
cp .env.local.example .env.local              # Host-Pfade eintragen
./Simulation/server_rl_run.sh preflight       # prüft Docker/GPU/Konfiguration
./run.sh sim                                  # dann: setup → eval → …
```

→ Schritt für Schritt (IKR & vast.ai): [simulation/sim-eval-anleitung.md](simulation/sim-eval-anleitung.md) · RL: [weiterfuehrend/rl-anleitung.md](weiterfuehrend/rl-anleitung.md) · bei Problemen zuerst [simulation/umsetzungsnotizen.md](simulation/umsetzungsnotizen.md)

**KISSKI-Cluster** (nur Training, kein Sim):

```bash
# einmalig auf dem Login-Node (Repo-Klon dort: siehe kisski-hpc.md)
module load apptainer
apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
    docker://lucam03/projekt-humanoider-roboter:latest

export HF_TOKEN=hf_... WANDB_API_KEY=...
sbatch Training/kisski_submit.sh
```

→ Storage, Monitoring, Checkpoints zurückholen: [training/kisski-hpc.md](training/kisski-hpc.md)

## Weiterführend

| Thema | Doku |
|---|---|
| Trainings-Anleitung (alle Wege, FAQ) | [training/anleitung.md](training/anleitung.md) |
| Env-Variablen-Referenz | [training/env-vars.md](training/env-vars.md) |
| KISSKI/SLURM im Detail | [training/kisski-hpc.md](training/kisski-hpc.md) |
| Sim-Eval Schritt für Schritt (IKR-Server & vast.ai) | [simulation/sim-eval-anleitung.md](simulation/sim-eval-anleitung.md) |
| Bekannte Sim-Fallstricke (zuerst lesen!) | [simulation/umsetzungsnotizen.md](simulation/umsetzungsnotizen.md) |
| Live in die Simulation schauen | [simulation/live-ansicht.md](simulation/live-ansicht.md) |
| RL-Feintuning | [weiterfuehrend/rl-anleitung.md](weiterfuehrend/rl-anleitung.md) |
| Repo auf anderer Hardware betreiben | [portabilitaet.md](portabilitaet.md) |
| Troubleshooting quer durch alle Bereiche | [fehlerbehebung.md](fehlerbehebung.md) |
| Übersicht aller Doku-Seiten | [README.md](README.md) |
