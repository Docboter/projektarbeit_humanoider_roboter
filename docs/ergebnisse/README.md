# Ergebnisse & Evaluation

> **TL;DR:** Navigations-Hub für alle Auswertungen, Messungen und Methodik-Reviews aus Training und
> Simulation — die zentrale Anlaufstelle für die Frage „Was kam raus?“. Verweist unten auf die
> Einzeldokumente, chronologisch geordnet in der Zeitleiste.

Gebündelte Auswertungen, Messungen und Methodik-Reviews aus Training und Simulation — die
„Was kam raus?"-Anlaufstelle. Die operativen Anleitungen, die diese Ergebnisse erzeugt haben,
liegen unter [`../training/`](../training/README.md) und [`../simulation/`](../simulation/README.md).

| Dokument | Wann lesen |
|---|---|
| [lauf1-auswertung.md](lauf1-auswertung.md) | **Zuerst.** Abschluss-Auswertung des ersten kompletten Laufs (175k Steps, frozen Vision-Encoder) — W&B-Metriken + Verhaltens-Evaluation (Closed-Loop, Replay, Open-Loop), Diagnose (visueller Domain-Gap) und Empfehlungen. Verweist auf alle übrigen Dokumente hier. |
| [lauf2-vision-auswertung.md](lauf2-vision-auswertung.md) | **Gegenprobe.** Zweiter Lauf mit **mittrainiertem Vision-Encoder** (`tune_visual=true`, Run `ajgoskon`, 44k Steps @ 4 GPUs). Training gleich gesund (Loss ~0,009), Closed-Loop aber **schlechter**: Policy-Kollaps auf reines Arm-Zurückziehen. Bestätigt empirisch die Lauf-1-Prognose, dass Encoder-Tuning auf reinen Realdaten den Sim-Gap nicht schließt. |
| [lauf3-vision-split-auswertung.md](lauf3-vision-split-auswertung.md) | **Erste echte Validierung.** Dritter Lauf (`tp1nc699`, 44k Steps @ 4 GPUs) mit Vision-Encoder **und** Color-Jitter **und** 80/20-Split — der erste Lauf mit zurückgehaltenen Episoden. Der Checkpoint-Sweep zeigt eine U-Kurve: bester Checkpoint **30.000** (MSE 0,00716), der letzte (44.000) ist **25 % schlechter**. Damit ist Overfitting im Projekt erstmals gemessen statt vermutet — und rückwirkend belegt, dass Lauf 1 und 2 blind den falschen Checkpoint genommen haben. Mit Signifikanz-Vorbehalt (n=6). **Nachtrag 2026-08-14:** checkpoint-30000 ist im Closed Loop vermessen — Fingerspanne 27,6 % statt 20,5 % (vollständige Trennung, p = 3,3 · 10⁻⁴), `lifted` weiterhin 0/10 (§8.3). |
| [lauf1-zwischenstand.md](lauf1-zwischenstand.md) | **Zwischenstand des 1. Laufs** — Momentaufnahme-Auswertung des damals laufenden Trainings (Health-Check, LR-Schedule, Batch-Size- & Eval-Empfehlungen). Interaktive Kurven: [`wandb-run-charts.html`](wandb-run-charts.html). |
| [diagnose-chronik.md](diagnose-chronik.md) | **Diagnose-Chronik (fortlaufendes Protokoll ab Lauf 08)** — chronologisches Protokoll aller Sim-/RL-Diagnoseläufe (Kamera-Fixes, Domain-Gap-Messläufe, Greif-Physik Lauf 29, `span`-Gate Lauf 32, `TUNE_VISUAL` im Closed Loop Lauf 34). Hierher zeigen die projektweiten „Lauf N"-Verweise. Über 2000 Zeilen — Einstieg über das Lauf-Inhaltsverzeichnis/die Anker, nicht linear lesen. |
| [domain-gap-analyse.md](domain-gap-analyse.md) | **Domain-Gap-Messung** — Cosine-Distanz Real→Sim pro Kamera via frozen SigLIP-ViT. Gültiges Ergebnis (Neumessung 2026-08-08 nach Kamerakalibrierung + Albedo-Fixes): Mittelwert **0,22**, `cam_left_wrist` weiterhin Ausreißer bei **0,36** (Kriterium < 0,35 knapp verfehlt). Die Erstmessung (0,26 / 0,43) ist überholt. Drei Handlungsoptionen mit Aufwand/Risiko-Abwägung. |
| [sim-bewertung.md](sim-bewertung.md) | **Methodik-Review** — Ist Closed-Loop-Sim sinnvoll/korrekt? Belegt: 0-%-Ergebnis ist der erwartete Real→Sim-Gap (SIMPLER), `TUNE_VISUAL`/Eval-DR greifen nicht, Open-Loop-MSE ist die valide Metrik. Mit Code-Befunden + Quellen. |
| [basismodell-referenz-eval.md](basismodell-referenz-eval.md) | **Pipeline-Validierung.** Basismodell zero-shot auf RoboCasa GR-1 Tabletop (robosuite/MuJoCo, 2× RTX PRO 6000). Aggregat über 12 Tasks **47,7 % ≈ 47,8 %** erwartet (Δ 0,1 pp) → GR00T-Inferenz-Pipeline extern bestätigt; die 0 % im Closed-Loop sind der Domain-Gap, keine kaputte Harness. |

## Zeitleiste

Die Tabelle oben ordnet nach Wichtigkeit; chronologisch liegen die Auswertungen so:

| Wann | Was | Dokument |
|---|---|---|
| 2026-06-02/03 | Lauf 1, Zwischenstand (`i6n1t613`, noch laufend) | [lauf1-zwischenstand.md](lauf1-zwischenstand.md) |
| 2026-06-03/04 | Lauf 1, Abschluss (175k Steps, frozen Encoder) | [lauf1-auswertung.md](lauf1-auswertung.md) |
| 2026-06-05 | Lauf 2 (`ajgoskon`, `tune_visual` ohne Augmentierung) | [lauf2-vision-auswertung.md](lauf2-vision-auswertung.md) |
| 2026-06-05 | Methodik-Review Closed-Loop-Sim | [sim-bewertung.md](sim-bewertung.md) |
| 2026-07-19 | RoboCasa-Referenz-Eval (Pipeline-Validierung, 47,7 %) | [basismodell-referenz-eval.md](basismodell-referenz-eval.md) |
| 2026-08-08 | Domain-Gap-Neumessung (ersetzt Erstmessung vom 2026-06-04) | [domain-gap-analyse.md](domain-gap-analyse.md) |
| 2026-08-08 → 14 | Diagnose-Läufe ab 08, fortlaufend (Kamera, Greif-Physik, `span`-Gate, `TUNE_VISUAL`) | [diagnose-chronik.md](diagnose-chronik.md) |
| 2026-08-13/14 | Lauf 3 (`tp1nc699`, Vision + Jitter + Split, Checkpoint-Sweep) | [lauf3-vision-split-auswertung.md](lauf3-vision-split-auswertung.md) |

> Der **Baseline-Vergleich** (stock G1-Greifer, `SIM_MODE=baseline`) ist noch nicht gelaufen —
> die Vorbereitung liegt bei den operativen Sim-Anleitungen:
> [../simulation/baseline-eval.md](../simulation/baseline-eval.md).

## Verwandte Dokumentation

- [Doku-Übersicht](../README.md) — globaler Navigations-Hub
- [Training](../training/README.md) — die Läufe, die hier ausgewertet werden
- [Simulation](../simulation/README.md) — die Sim-Eval, deren Messungen hier liegen
- [Weiterführende Arbeiten](../weiterfuehrend/README.md) — Folgeschritte aus den Befunden (RL-Plan etc.)
