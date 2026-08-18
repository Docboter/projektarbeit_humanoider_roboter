# Projektarbeit Humanoider Roboter
## GR00T N1.6 Fine-tuning — Unitree G1 mit DEX3-Hand

Dieses Repository dokumentiert das Fine-tuning von NVIDIAs **GR00T N1.6**
Vision-Language-Action-Modell auf den Unitree G1 Roboter mit DEX3-Hand für die Aufgabe
Block-Stacking — plus eine Closed-Loop-**Simulations-Evaluation** in Isaac Lab.

Die Trainingsumgebung läuft **vollständig autonom in einem Container**: Container starten →
Training läuft. Derselbe Container läuft lokal, auf Cloud-GPU-Plattformen wie
[vast.ai](https://vast.ai) und auf dem **KISSKI HPC-Cluster** (GWDG Göttingen). Alle Schritte
(Daten-Download, Konvertierung, Training) passieren im Container.

> **Speicher-Modell:** Es gibt **keinen persistenten Storage auf dem Host**. Daten, Checkpoints
> und Logs leben ausschließlich im Container-Filesystem. Auf vast.ai entspricht ein Container
> genau einer Instanz — wird sie zerstört, ist alles weg. Auf KISSKI wird
> `/mnt/vast-kisski/projects/kisski-humrob/data` als `/data` in den Container gemountet.
> **Niemals `docker run --rm`** mit diesem Image verwenden.

---

## Schnellstart

**KISSKI (HPC-Cluster, empfohlen für langes Training):**
```bash
# Einmalig auf dem Login-Knoten glogin-gpu.hpc.gwdg.de:
module load apptainer
apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
    docker://lucam03/projekt-humanoider-roboter:latest

# Job einreichen:
export HF_TOKEN=hf_...  WANDB_API_KEY=...
sbatch Training/kisski_submit.sh
```

**vast.ai:** Instanz mit Image `lucam03/projekt-humanoider-roboter:latest` starten, in den
Docker-Options `-e HF_TOKEN=hf_… -e WANDB_API_KEY=…` setzen → Training läuft autonom.

**Lokal (Linux / WSL2):**
```bash
export HF_TOKEN=hf_...  WANDB_API_KEY=...
./Training/setup_and_train_DockerHub-pull.sh
```

→ Ausführliche Schritt-für-Schritt-Anleitung: **[docs/training/anleitung.md](docs/training/anleitung.md)**

---

## Dokumentation

Die vollständige Dokumentation liegt unter **[docs/](docs/README.md)**:

| Bereich | Einstieg |
|---|---|
| **Training** | [docs/training/](docs/training/README.md) — Einstieg & die vier Wege zum Trainieren |
| ↳ KISSKI HPC | [docs/training/kisski-hpc.md](docs/training/kisski-hpc.md) |
| ↳ Env-Vars | [docs/training/env-vars.md](docs/training/env-vars.md) — alle Parameter + VRAM-Richtwerte |
| ↳ Train-Test-Split | [docs/training/train-test-split.md](docs/training/train-test-split.md) |
| ↳ W&B-Offline-Sync | [docs/training/wandb-offline-sync.md](docs/training/wandb-offline-sync.md) |
| **Simulation** | [docs/simulation/](docs/simulation/README.md) — Closed-Loop-Eval in Isaac Lab |
| ↳ Primärer Workflow | [docs/simulation/vastai-anleitung.md](docs/simulation/vastai-anleitung.md) |
| ↳ Lessons & Fixes | [docs/simulation/umsetzungsnotizen.md](docs/simulation/umsetzungsnotizen.md) |
| **Ergebnisse** | [docs/ergebnisse/](docs/ergebnisse/README.md) — Auswertungen, Domain-Gap, Methodik-Review, Baseline |
| **Weiterführend** | [docs/weiterfuehrend/](docs/weiterfuehrend/README.md) — RL-Plan, Lokomotion, Livestream (Konzepte) |
| **Portabilität** | [docs/portabilitaet.md](docs/portabilitaet.md) — eigener Rechner / eigenes KISSKI-Projekt: welche Knöpfe zu setzen sind |
| **Projektstruktur** | [docs/README.md#projektstruktur](docs/README.md#projektstruktur) |

---

## Voraussetzungen (Kurzfassung)

**Hardware:** NVIDIA-GPU mit ≥ 24 GB VRAM (Minimum, sehr langsam); empfohlen A100/H100 ≥ 40 GB.
Full Fine-tuning verbraucht ~25–31 GB VRAM (bs=8); NVIDIA empfiehlt **≥ 40 GB**.

**Accounts:** HuggingFace (Zugriff auf
[`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) — Lizenz akzeptieren! —
und [`unitreerobotics/G1_Dex3_BlockStacking_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset)),
optional [WandB](https://wandb.ai), für HPC ein [KISSKI-Account](https://docs.hpc.gwdg.de).

**Software (nur lokal):** Docker ≥ 4.x, NVIDIA Container Toolkit, NVIDIA-Treiber ≥ 570
(CUDA 12.8). Auf vast.ai und KISSKI vorinstalliert.

Details: [docs/training/anleitung.md](docs/training/anleitung.md#vorab-accounts--tokens).

---

## Weiterführende Ressourcen

- [GWDG HPC Dokumentation](https://docs.hpc.gwdg.de) — KISSKI/Grete-Cluster
- [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/groot)
- [LeRobot](https://github.com/huggingface/lerobot)
