# Training — Fine-tuning von GR00T N1.6

Fine-tuning von NVIDIAs GR00T-N1.6-VLA-Modell auf den Unitree G1 + DEX3-Hand (Block-Stacking).
Die Umgebung läuft autonom in einem Docker-Container: Env-Vars setzen → der Container erledigt
Download, Konvertierung und Training. Derselbe Container läuft lokal, auf vast.ai und auf dem
KISSKI HPC-Cluster.

## Einstieg

| Dokument | Wann lesen |
|---|---|
| [trainingsverfahren.md](trainingsverfahren.md) | **Was für ein Training ist das?** Modellarchitektur (GR00T N1.6 VLA), Lernverfahren (Imitation Learning, Flow-Matching), Embodiment, Ein-/Ausgaben — die konzeptionelle Einordnung. |
| [anleitung.md](anleitung.md) | **Zuerst.** Schritt-für-Schritt — die vier Wege zum Trainieren (vast.ai, lokal mit Launcher-Skript, lokal mit `docker run`, KISSKI), Daten retten, „Was passiert intern?", FAQ. |
| [kisski-hpc.md](kisski-hpc.md) | **HPC-Training auf KISSKI.** SIF-Konvertierung, VAST-Storage, SLURM-Job, Monitoring, Checkpoint-Export, KISSKI-Troubleshooting (die ausführliche Form von Weg D). |
| [env-vars.md](env-vars.md) | **Konfigurationsreferenz.** Alle Env-Vars + VRAM-Richtwerte — die Single Source of Truth, auf die `anleitung.md` und `kisski-hpc.md` verweisen. |
| [multi-gpu.md](multi-gpu.md) | **Multi-GPU (bis 4× A100).** Vorteile, ToDos (torchrun-Launcher + SLURM-Ressourcen) und Verifikationsschritte — geplant, noch nicht umgesetzt. |
| [train-test-split.md](train-test-split.md) | 80/20-Datensatz-Split — Implementierung und Nutzung für die Evaluation auf ungesehenen Episoden. |
| [wandb-offline-sync.md](wandb-offline-sync.md) | W&B-Offline-Sync auf KISSKI (Compute-Knoten ohne Internet). |
| [wandb-run-auswertung.md](wandb-run-auswertung.md) | Momentaufnahme-Auswertung eines laufenden Trainings-Runs (Health-Check, LR-Schedule, Batch-Size- & Eval-Empfehlungen). Interaktive Kurven: [wandb-run-charts.html](wandb-run-charts.html). |

## Die vier Wege im Überblick

Welcher Weg passt — die konkreten Befehle stehen jeweils in [anleitung.md](anleitung.md):

| Weg | Plattform | Wann | Anleitung |
|---|---|---|---|
| A | vast.ai (Cloud-GPU) | kein eigener GPU-Rechner / schneller trainieren | [anleitung.md → Weg A](anleitung.md#weg-a--cloud-training-auf-vastai) |
| B | Lokal, Launcher-Skript | eigene NVIDIA-GPU (≥ 8 GB), bequem | [anleitung.md → Weg B](anleitung.md#weg-b--lokales-training-mit-dem-launcher-skript) |
| C | Lokal, `docker run` | maximale Kontrolle, kein Skript | [anleitung.md → Weg C](anleitung.md#weg-c--lokales-training-mit-docker-run) |
| D | KISSKI HPC (A100/H100) | langes Training per SLURM | [kisski-hpc.md](kisski-hpc.md) |

> **Speicher-Modell:** keine Host-Persistenz — alles lebt im Container-Filesystem. **Niemals
> `docker run --rm`** verwenden (löscht Daten + Checkpoints). Details: [anleitung.md →
> Speicher-Modell](anleitung.md#speicher-modell--wichtig-vorab).

## Verwandte Dokumentation

- [Doku-Übersicht](../README.md) — globaler Navigations-Hub
- [Simulation](../simulation/README.md) — Closed-Loop-Eval des fertigen Checkpoints in Isaac Lab
- [`FINETUNING_GUIDE.md`](../../app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md) — volle Parameter-Referenz (Submodul)
- [`SETUP_DOCUMENTATION.md`](../../app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md) — Setup & Architektur des G1/DEX3-Embodiments (Submodul)
