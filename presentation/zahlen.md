# Zahlen der Präsentation und ihre Quellen

> **TL;DR:** Jede Zahl auf den Folien (und die wichtigsten aus den Sprechernotizen) mit Quelldatei,
> Abschnitt und Laufnummer, damit sie sich vor dem Vortrag prüfen lässt. Bedienung der
> Präsentation: [README.md](README.md); gepflegter Projektstatus: [next-steps.md §0](../next-steps.md).

Abkürzungen: **CL** = [`closed_loop.csv`](../latex/figures/daten/closed_loop.csv),
**Gate** = [`gate_mse_haende.csv`](../latex/figures/daten/gate_mse_haende.csv),
**Chronik** = [`diagnose-chronik.md`](../docs/ergebnisse/diagnose-chronik.md),
**Ergebnisse** = [`docs/ergebnisse/README.md`](../docs/ergebnisse/README.md) (Tabelle „auf einen Blick“).
Die Diagramme lesen ihre Werte direkt aus den CSVs unter `latex/figures/daten/`
(eingebettet über `tools/build_assets.py`), es gibt dort keine abgetippten Zahlen.

## Folien

| St. | Zahl auf der Folie | Quelle | Lauf |
|---|---|---|---|
| 2 | drei Würfel · 4 Kamerabilder | Ausarbeitung Kap. 1 (`latex/chapters/01_einleitung.tex`), Kap. 2 | — |
| 3 | 301 Episoden, Split 240 / 61 | Ausarbeitung Kap. 1; [train-test-split.md](../docs/training/train-test-split.md) | ab Lauf 3 |
| 3 | N1.6 und N1.7, Checkpoints nicht austauschbar | [CLAUDE.md](../CLAUDE.md) (Projektüberblick) | — |
| 3 | 4 Kameras: 2 Kopf, 2 Handgelenk | `Simulation/camera_reference/dataset_cam_*.png` | — |
| 4 | LeRobot v3.0 → v2.1 | Ausarbeitung Kap. 3 und 6 (`05b_cosmos.tex`, Pipeline-Abbildung) | — |
| 5 | Val-MSE 0,00482 (N1.6 @44 000), 0,00512 (N1.7 @35 000), n. s. | [`val_mse_lauf3_lauf4.csv`](../latex/figures/daten/val_mse_lauf3_lauf4.csv); [lauf4-n17-vergleich.md §5](../docs/ergebnisse/lauf4-n17-vergleich.md#5-nachgeschärfter-vergleich-20-episoden--750-steps-volle-länge) | 3, 4 |
| 5 | 1,6× schneller · n = 20 Test-Episoden × 750 Schritte | Ergebnisse (Zeile „Training“); CSV-README | 3, 4 |
| 6 | Szene aus Lauf 70 | Chronik [Läufe 70/71](../docs/ergebnisse/diagnose-chronik.md#läufe-7071-szene-ans-echtbild-angleichen-kamera-mischung) | 70 |
| 7 | 16/20 erreicht, 0/20 angehoben (N1.6 @44 000, 20 × 120 s) | CL, Zeile 82; Chronik [Lauf 82](../docs/ergebnisse/diagnose-chronik.md#lauf-82-closed-loop-n16-44000-szene-aus-lauf-70) | 82 |
| 7 | N1.7: 7/20 erreicht, 0/20 angehoben | CL, Zeile 81 | 81 |
| 8 | ~3 cm je Seite zu weit innen | Chronik [Lauf 62](../docs/ergebnisse/diagnose-chronik.md#lauf-62-versatz-sweep--die-würfel-liegen-3-cm-zu-weit-innen) | 62 |
| 8 | 0/24 ohne Seitenkorrektur (dy = 0: drei Zeilen à 0/8) | [`orakel_versatz_lauf62.csv`](../latex/figures/daten/orakel_versatz_lauf62.csv) | 62 |
| 8 | 6/20 = 30 % (KI 12–54 %) mit Korrektur | Chronik [Lauf 69](../docs/ergebnisse/diagnose-chronik.md#lauf-69-orakel-auf-dem-korrigierten-layout) | 69 |
| 8 | Heatmap k/8 je Versatz | `orakel_versatz_lauf62.csv` | 62 |
| 9 | ×13 (N1.6 vision-v2 @30 000), ×36 (N1.7 @44 000) | Chronik [Läufe 63–68](../docs/ergebnisse/diagnose-chronik.md#läufe-6368-ursache-des-versatzes-bild-ablation-sim-bilder-im-open-loop); Ergebnisse (Zeile „Blocker 2“) | 65, 66 |
| 9 | −40 % / −18 %, bleibt ×8 / ×29 | Ergebnisse (Zeile „Blocker 2“); next-steps §0 | 70 |
| 10 | Mischung 15 % (0,15) | next-steps §0; [co-training.md](../docs/training/co-training.md) | 73–76 |
| 10 | Sim-Fehler halbiert (N1.6), auf 18–30 % (N1.7) | Gate; next-steps §0 | 80, 83, 86, 92 |
| 10 | N1.6 Echtbild +50–60 % | Gate (0,0014 → 0,0021–0,0023); next-steps §0 | 80, 83 |
| 10 | Diagrammwerte, 3 Seeds, alle @44 000 | Gate, Zeilen „N1.6 vision-v2 @44000“, „C N1.6 cube4 @44000“, „N1.7 vision-v1 @44000“, „D N1.7 cube4 @44000“ | 80, 83, 92 |
| 11 | 2/60 = 3 % angehoben (KI 0,4–11,5 %) | CL, Zeile 88+97; Chronik [Lauf 97](../docs/ergebnisse/diagnose-chronik.md#lauf-97-closed-loop-des-besten-kandidaten-auf-allen-60-testlagen) | 88, 97 |
| 11 | 73 % der Demo-Fingerspanne, p = 5 · 10⁻⁶ | Ergebnisse (Zeile „Vorregistriertes Gate“); next-steps §0 | 96 |
| 11 | gegen 0/20 ohne Co-Training nicht trennbar | Ergebnisse (Zeile „Co-Training, Closed Loop“) | 82 |
| 11 | Standbild Lauf 88, Ep. 5, t = 80 s | [CSV-README](../latex/figures/daten/README.md), Bildtabelle | 88 |
| 12 | ohne Co-Training 16 / 17 / 0 / 0 von 20 | CL, Zeile 82 (`reached`, `touched`, `lifted_clear`, `stacked`) | 82 |
| 12 | mit Co-Training 58 / 60 / 2 / 0 von 60 | CL, Zeile 88+97 | 88, 97 |
| 12 | 5-cm- gegen 4-cm-Würfel | Chronik [Lauf 88](../docs/ergebnisse/diagnose-chronik.md#lauf-88-closed-loop-co-training-n16-cube4-mit-4-cm-würfeln): „Protokoll wie Lauf 82, zusätzlich `SIM_CUBE_EDGE=0.04`“ | 82, 88 |
| 13 | Pipeline lief einmal komplett, Training steht aus | next-steps §0 (Cosmos); Ausarbeitung Kap. 6 | — |
| 14 | gestapelt 0 · 20–60 Episoden | CL (`stacked` überall 0); Ausarbeitung Kap. 7 | alle |
| 14 | Nächste Schritte | [next-steps.md §2](../next-steps.md), Punkte 3, 4, 5, 7 | — |

## Sprechernotizen

| St. | Zahl | Quelle | Lauf |
|---|---|---|---|
| 5 | 7 h 53 min gegen 12 h 27 min; N1.7 bei 10 000 / 20 000 signifikant besser | Ergebnisse, Eintrag lauf4-n17-vergleich.md | 4 |
| 7 | RoboCasa 47,7 % gegen 47,8 % erwartet | [basismodell-referenz-eval.md](../docs/ergebnisse/basismodell-referenz-eval.md) | — |
| 8 | Orakel ohne Korrektur 1/16; 5/8 bei dy = 3 cm | Ergebnisse (Zeile „Blocker 1“); `orakel_versatz_lauf62.csv` | 57–62 |
| 9 | ×17 für N1.6 @44 000 im Gate-Protokoll | Ergebnisse (Zeile „Blocker 2“) | 80 |
| 10 | je 187 Render-Episoden | next-steps §2, Punkt 00 | 72 |
| 11 | 1,53 rad gegen 0,52 rad (ohne Renderdaten) | Chronik, Lauf-Tabelle Zeile 96 | 96 |
| 13 | 2 Quell-Episoden × 3 Varianten, ~65 GB Grafikspeicher | next-steps §0; Ausarbeitung Kap. 6 | — |
| 14 | 95-%-Intervall bis 17 % bei 0/20 | Ausarbeitung Kap. 7 (`06_grenzen.tex`) | — |
