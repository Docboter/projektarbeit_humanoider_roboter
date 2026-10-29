# Projektarbeit Humanoider Roboter

> **TL;DR:** Fine-Tuning des Vision-Language-Action-Modells NVIDIA GR00T (N1.6 und N1.7) auf
> den Unitree G1 mit DEX3-Händen für Würfelstapeln, mit Closed-Loop-Bewertung in Isaac Lab.
> Einstieg: [Quickstart](docs/quickstart.md) (Befehle) und
> [Schnittstellen](docs/schnittstellen.md) (Formate); alles Weitere unter [docs/](docs/README.md).

## Ergebnis (Stand 2026-10-29)

- **Training:** Bester N1.6-Checkpoint ist vision-v2 (Lauf 3) @44000; N1.7 liegt im Open Loop
  gleichauf und trainiert 1,6× schneller ([Lauf 4](docs/ergebnisse/lauf4-n17-vergleich.md)).
- **Zwei Blocker der Sim gemessen:** Die Würfel liegen in der Sim je Seite ~3 cm zu weit innen;
  mit Korrektur hebt die abgespielte menschliche Demo in 30 % statt ~0 % der Episoden (Läufe 62,
  69). Auf Sim-Bildern ist der Hand-Fehler der Policy ×13 (N1.6) bzw. ×36 (N1.7) gegenüber
  Echtbildern (Lauf 66).
- **Co-Training mit Sim-Renderbildern** halbiert diesen Fehler bei N1.6 und senkt ihn bei N1.7
  auf 18–30 %. Die Policy erreicht und berührt die Würfel; echte Griffe bleiben selten
  (`lifted_clear` 2/60 = 3 %, gegen 0/20 ohne Co-Training nicht signifikant), gestapelt wird nie
  (Läufe 73–97). Ein Test auf dem Roboter fehlt.

Laufende Statusquelle: [next-steps.md §0](next-steps.md); jede Zahl mit Herkunft:
[Diagnose-Chronik](docs/ergebnisse/diagnose-chronik.md).

## Repo-Karte

| Ordner | Inhalt |
|---|---|
| `run.sh` | Menü-Einstieg für alle Aktionen (`MENU=0` schaltet ab) |
| `Training/` | Trainings-Image (Dockerfile, `update_image.sh`), Launcher lokal, KISSKI-Skripte (`kisski_submit.sh`, `kisski_chain.sh`), `scripts/` = Entrypoint im Image |
| `Simulation/` | Isaac-Lab-Umgebung und Eval (`g1_dex3_sim/`), Sim-Image, `server_rl_run.sh` (alle Sim-Aktionen), Hilfsskripte im Container (`scripts/`) |
| `Augmentation/` | Cosmos-Transfer-Augmentierung (gebaut, nicht end-to-end geprüft) |
| `app/Groot-1.6`, `app/Groot-1.7` | GR00T-Fork als Submodule (`lucam06/Isaac-GR00T`), G1-Konfiguration unter `examples/G1_DEX3/` |
| `tools/` | Host-Werkzeuge: Menü, Doku-Prüfungen, Auswertung der Sim-Läufe |
| `data/` | Roboter-Assets (USD, `unitree_ros`-Submodul) |
| `docs/` | Doku (Deutsch), Einstieg [docs/README.md](docs/README.md) |
| `latex/` | schriftliche Ausarbeitung |

## Wichtig vor dem ersten Start

- Trainings-Container **nie mit `docker run --rm`** starten: Daten und Checkpoints liegen im
  Container. Stoppen und Starten erhält sie, erst `docker rm` löscht sie.
- Skripte unter `Training/scripts/` und `Simulation/` werden ins Image kopiert. Änderungen
  brauchen einen Neubau. Ausnahme: KISSKI bindet `Training/scripts` aus dem Repo ein.
- Die Sim braucht eine GPU mit RT-Cores. Sie läuft also nicht auf KISSKI (A100/H100), sondern auf
  dem IKR-Server oder einer gemieteten Karte.
