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
| [multi-gpu.md](multi-gpu.md) | **Multi-GPU (bis 4× A100) — umgesetzt.** Warum sich Multi-GPU lohnt, was geändert wurde (torchrun-Launcher + SLURM-Ressourcen), KISSKI-Defaults (4× A100, `GLOBAL_BATCH_SIZE=32`) und Verifikationsschritte. |
| [train-test-split.md](train-test-split.md) | 80/20-Datensatz-Split **und Checkpoint-Auswahl** — Implementierung, Nutzung und warum die Validierung erst *nach* dem Lauf stattfinden kann (der Fork hat keine In-Training-Eval: `enable_open_loop_eval` ist tot, `factory.py` sperrt `eval_strategy != "no"`). Werkzeug: [`checkpoint_sweep.py`](../../Training/scripts/checkpoint_sweep.py). |
| [wandb-offline-sync.md](wandb-offline-sync.md) | W&B-Offline-Sync auf KISSKI (Compute-Knoten ohne Internet). |
| [fixes-aus-erstem-lauf.md](fixes-aus-erstem-lauf.md) | **Fixes aus dem ersten Lauf** — konkrete Maßnahmen aus der Auswertung (Domain-Gap: Würfelfarbe, Stapel-Band, schwarze Hände, `BLACK_HANDS`-Auto-Recolor), neue Werkzeuge, Anwenden/Verifizieren, offene Punkte. |

> **Auswertung der Läufe** liegt jetzt unter [`../ergebnisse/`](../ergebnisse/README.md)
> (W&B-Metriken, Abschluss-Auswertung des 1. Laufs, Domain-Gap, Baseline). Der **RL-Plan** als
> möglicher nächster Schritt liegt unter [`../weiterfuehrend/reinforcement-learning-plan.md`](../weiterfuehrend/reinforcement-learning-plan.md).

> **Variante — Vision-Encoder mittrainieren:** Mit `TUNE_VISUAL=1` wird zusätzlich der
> Vision-Encoder feingetunt (LLM bleibt eingefroren). Der Entrypoint startet dann
> `run_finetuning_vision.sh` mit eigenem Output-/Experiment-Namespace (`blockstacking_vision`).
> Funktioniert auf allen Wegen (Container, vast.ai, KISSKI). Details: [env-vars.md](env-vars.md)
> (`TUNE_VISUAL`), Anwendung in [anleitung.md](anleitung.md) und [kisski-hpc.md](kisski-hpc.md).

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
- [Ergebnisse](../ergebnisse/README.md) — Auswertungen & Messungen der Trainingsläufe
- [Weiterführende Arbeiten](../weiterfuehrend/README.md) — RL-Plan, Lokomotion, Livestream
- [Simulation](../simulation/README.md) — Closed-Loop-Eval des fertigen Checkpoints in Isaac Lab
- [`FINETUNING_GUIDE.md`](../../app/Groot-1.6/examples/G1_DEX3/FINETUNING_GUIDE.md) — volle Parameter-Referenz (Submodul)
- [`SETUP_DOCUMENTATION.md`](../../app/Groot-1.6/examples/G1_DEX3/SETUP_DOCUMENTATION.md) — Setup & Architektur des G1/DEX3-Embodiments (Submodul)
