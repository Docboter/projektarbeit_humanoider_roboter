# Abbildungsdaten für die Ausarbeitung

> **TL;DR:** Aus den Mess-JSONs auf dem IKR-Server (`~/groot-rl-data/sim_results/`) exportierte
> Zahlen, damit die Abbildungen der Ausarbeitung ohne Serverzugang gebaut werden können. Stand
> 2026-10-29; Quelle und Lesart jeder Zahl steht in der
> [Diagnose-Chronik](../../../docs/ergebnisse/diagnose-chronik.md).

| Datei | Inhalt | Chronik |
|---|---|---|
| `gate_mse_haende.csv` | Visuelles Gate: MSE Hände (Mittel der Seed-Mediane, Min/Max über Seeds 0/1/2) auf Echtbild und den Sim-Prüfsets 5 cm / 4 cm, für Ausgangswerte, alle acht Co-Training-Checkpoints und die zwei Altmodelle | Läufe 80, 83, 86, 87, 92 |
| `closed_loop.csv` | Closed Loop je Lauf: Episoden, Episodenlänge, Stufen `reached` … `stacked` (`lifted`/`lifted_clear` ohne Artefakte), Artefakte, Fingerspanne Median/Min/Max, Medianabstand Kuppe↔Würfel, Quell-JSONs | Läufe 81–97 |
| `orakel_versatz_lauf62.csv` | Versatz-Sweep im Orakel: Würfel um `dx` (vom Roboter weg) und `dy` (seitlich nach außen) verschoben, `lifted_clear` k von 8 Episoden (240–247, 40 s); übertragen aus der Tabelle der Chronik | Lauf 62 |
| `lernkurven_lauf3_lauf4.csv` | `train/loss` je 10 Schritte von Lauf 3 (N1.6, W&B `tp1nc699`) und Lauf 4 (N1.7, `2v02ye81`), ausgelesen aus den Offline-Logs auf KISSKI (`/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune/wandb/offline-run-*`); Lauf 3 an 500 Punkten gegen W&B geprüft, identisch. Lauf 4 seit 2026-10-30 auch in W&B. Absolute Höhe zwischen N1.6 und N1.7 nicht vergleichbar | Läufe 3, 4 |
| `val_mse_lauf3_lauf4.csv` | Validierungs-MSE je Checkpoint, 20 Testepisoden × 750 Schritte | [lauf4-n17-vergleich.md §5](../../../docs/ergebnisse/lauf4-n17-vergleich.md#5-nachgeschärfter-vergleich-20-episoden--750-steps-volle-länge) |

Bilder unter `../sim/`:

| Datei | Inhalt |
|---|---|
| `sim_scene70_cam_left_high.png` | Kopfkamera links in der Szene aus Lauf 70 (erstes Beobachtungsbild aus Lauf 88, Testlage 240) |
| `griff_lauf88_ep05_80s.png`, `…_ausschnitt.png` | Erster echter Griff der Politik: Lauf 88, Ep. 5 (Testlage 244), t = 80 s, linke Hand hält den roten Würfel in der Luft |

Echtbilder derselben Kameras: `Simulation/camera_reference/dataset_cam_*.png`.
Neu erzeugen: `python3 tools/sim_visgate_seeds_summary.py` bzw. `tools/sim_cl_summary.py`
auf dem Server (Pfade in der Spalte `quellen_json`). Diagramme daraus:
`uv run --with matplotlib python latex/figures/make_figures.py` → `../plots/*.pdf`.
