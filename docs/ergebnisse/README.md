# Ergebnisse & Evaluation

Gebündelte Auswertungen, Messungen und Methodik-Reviews aus Training und Simulation — die
„Was kam raus?"-Anlaufstelle. Die operativen Anleitungen, die diese Ergebnisse erzeugt haben,
liegen unter [`../training/`](../training/README.md) und [`../simulation/`](../simulation/README.md).

| Dokument | Wann lesen |
|---|---|
| [lauf1-auswertung.md](lauf1-auswertung.md) | **Zuerst.** Abschluss-Auswertung des ersten kompletten Laufs (175k Steps, frozen Vision-Encoder) — W&B-Metriken + Verhaltens-Evaluation (Closed-Loop, Replay, Open-Loop), Diagnose (visueller Domain-Gap) und Empfehlungen. Verweist auf alle übrigen Dokumente hier. |
| [lauf2-vision-auswertung.md](lauf2-vision-auswertung.md) | **Gegenprobe.** Zweiter Lauf mit **mittrainiertem Vision-Encoder** (`tune_visual=true`, Run `ajgoskon`, 44k Steps @ 4 GPUs). Training gleich gesund (Loss ~0,009), Closed-Loop aber **schlechter**: Policy-Kollaps auf reines Arm-Zurückziehen. Bestätigt empirisch die Lauf-1-Prognose, dass Encoder-Tuning auf reinen Realdaten den Sim-Gap nicht schließt. |
| [wandb-run-auswertung.md](wandb-run-auswertung.md) | Momentaufnahme-Auswertung des laufenden Trainings (Health-Check, LR-Schedule, Batch-Size- & Eval-Empfehlungen). Interaktive Kurven: [`wandb-run-charts.html`](wandb-run-charts.html). |
| [domain-gap-analyse.md](domain-gap-analyse.md) | **Domain-Gap-Messung** — Cosine-Distanz Real→Sim pro Kamera via frozen SigLIP-ViT. Ergebnis: Mittelwert 0.26, `cam_left_wrist` kritisch bei 0.43. Drei Handlungsoptionen mit Aufwand/Risiko-Abwägung. |
| [sim-bewertung.md](sim-bewertung.md) | **Methodik-Review** — Ist Closed-Loop-Sim sinnvoll/korrekt? Belegt: 0-%-Ergebnis ist der erwartete Real→Sim-Gap (SIMPLER), `TUNE_VISUAL`/Eval-DR greifen nicht, Open-Loop-MSE ist die valide Metrik. Mit Code-Befunden + Quellen. |
| [basismodell-referenz-eval.md](basismodell-referenz-eval.md) | **Pipeline-Validierung.** Basismodell zero-shot auf RoboCasa GR-1 Tabletop (robosuite/MuJoCo, 2× RTX PRO 6000). Aggregat über 12 Tasks **47,7 % ≈ 47,8 %** erwartet (Δ 0,1 pp) → GR00T-Inferenz-Pipeline extern bestätigt; die 0 % im Closed-Loop sind der Domain-Gap, keine kaputte Harness. |
| [baseline-unitree-g1.md](baseline-unitree-g1.md) | **Baseline-Vergleich** — un-finetuntes `GR00T-N1.6-3B` + stock G1-Greifer (`UNITREE_G1`) auf demselben Block-Stacking-Task. Parallele Pipeline (`SIM_MODE=baseline`), TODO-Checkliste vor dem ersten Run, OOD-Einordnung. |

## Verwandte Dokumentation

- [Doku-Übersicht](../README.md) — globaler Navigations-Hub
- [Training](../training/README.md) — die Läufe, die hier ausgewertet werden
- [Simulation](../simulation/README.md) — die Sim-Eval, deren Messungen hier liegen
- [Weiterführende Arbeiten](../weiterfuehrend/README.md) — Folgeschritte aus den Befunden (RL-Plan etc.)
