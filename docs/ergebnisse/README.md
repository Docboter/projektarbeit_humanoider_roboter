# Ergebnisse & Evaluation

> **TL;DR:** Was kam raus? Oben die Ergebnisse auf einen Blick (Stand 2026-10-29, bis Lauf 97)
> mit Lauf-Nummern, darunter die Einzelauswertungen. Herkunft jeder Zahl:
> [diagnose-chronik.md](diagnose-chronik.md); Abbildungsdaten: [`latex/figures/daten/`](../../latex/figures/daten/README.md).

Gebündelte Auswertungen, Messungen und Methodik-Reviews aus Training und Simulation — die
„Was kam raus?"-Anlaufstelle. Die operativen Anleitungen, die diese Ergebnisse erzeugt haben,
liegen unter [`../training/`](../training/README.md) und [`../simulation/`](../simulation/README.md).

## Ergebnisse auf einen Blick (Stand 2026-10-29)

| Schritt | Ergebnis | Läufe |
|---|---|---|
| Training, Checkpoint-Wahl | N1.6 vision-v2 @44000 ist der beste Checkpoint (Val-MSE 0,00482, monoton fallend). N1.7 bestes 0,00512 @35000, nicht signifikant verschieden, 1,6× schneller | 3, 4 ([lauf4 §5](lauf4-n17-vergleich.md#5-nachgeschärfter-vergleich-20-episoden--750-steps-volle-länge)) |
| Closed Loop ohne Co-Training | N1.6 @44000: `reached` 16/20, `lifted_clear` 0/20. N1.7 @44000: `reached` 7/20, 0/20 | 81, 82 |
| Blocker 1: Registrierung | Abgespielte Demo-Aktionen heben kaum (`lifted_clear` 1/16). Würfel je Seite ~3 cm zu weit innen; mit Korrektur 6/20 = 30 % (KI 12–54 %) | 57–62, 69 |
| Blocker 2: Domain-Gap | Gleicher Zustand, Sim- statt Echtbild: Hand-Fehler ×13 (N1.6), ×36 (N1.7) | 65, 66 |
| Co-Training, Gate | MSE Hände Echtbild / Sim 5 cm / Sim 4 cm: N1.6 0,0014 / 0,0239 / 0,0221 → cube4 0,0021 / 0,0115 / 0,0104. N1.7 0,0026 / 0,0758 / 0,0666 → cube4 0,0027 / 0,0217 / 0,0123 | 80, 83, 92 |
| Co-Training, Closed Loop | N1.6 cube4 @44000 auf allen 60 Testlagen: `touched` 60/60, `lifted_clear` 2/60 = 3 % (KI 0,4–11,5 %), `stacked` 0. N1.7: `lifted_clear` 0/20. Gegen 0/20 ohne Co-Training nicht signifikant | 88, 89, 93, 97 |
| Vorregistriertes Gate | Fingerspanne 1,53 rad = 73 % der Demo gegen 0,577 rad (Lauf 34) und 0,52 rad ohne Renderdaten (p = 5 · 10⁻⁶) | 96 |
| Offen | Kein Test auf dem Roboter; `stacked` in keinem Lauf | — |

## Einzelauswertungen

| Dokument | Wann lesen |
|---|---|
| [lauf1-auswertung.md](lauf1-auswertung.md) | **Zuerst.** Abschluss-Auswertung des ersten kompletten Laufs (175k Steps, frozen Vision-Encoder) — W&B-Metriken + Verhaltens-Evaluation (Closed-Loop, Replay, Open-Loop), Diagnose (visueller Domain-Gap) und Empfehlungen. Verweist auf alle übrigen Dokumente hier. |
| [lauf2-vision-auswertung.md](lauf2-vision-auswertung.md) | **Gegenprobe.** Zweiter Lauf mit **mittrainiertem Vision-Encoder** (`tune_visual=true`, Run `ajgoskon`, 44k Steps @ 4 GPUs). Training gleich gesund (Loss ~0,009), Closed-Loop aber **schlechter**: Policy-Kollaps auf reines Arm-Zurückziehen. Stützt die Lauf-1-Prognose, dass Encoder-Tuning auf reinen Realdaten den Sim-Gap nicht schließt (Vorbehalte im Dokument: alte Sim-Geometrie, Jitter-Zuschreibung ungeklärt). |
| [lauf3-vision-split-auswertung.md](lauf3-vision-split-auswertung.md) | **Erste echte Validierung.** Dritter Lauf (`tp1nc699`, 44k Steps @ 4 GPUs) mit Vision-Encoder **und** Color-Jitter **und** 80/20-Split — der erste Lauf mit zurückgehaltenen Episoden. Der erste Checkpoint-Sweep (6 × 300, n = 6) zeigte eine U-Kurve mit Optimum **30.000** — **überholt** durch den Nachtrag 2026-10-12 unten (bester Checkpoint 44.000). **Nachtrag 2026-08-14:** checkpoint-30000 ist im Closed Loop vermessen — Fingerspanne 27,6 % statt 20,5 % (vollständige Trennung, p = 3,3 · 10⁻⁴), `lifted` weiterhin 0/10 (§8.3). **Nachtrag 2026-10-12: Overfitting-Befund widerlegt.** Die Nachmessung mit 20 Episoden à 750 Steps zeigt keine U-Kurve — Validierungs-MSE fällt monoton bis 44.000 (0,00482), 44.000 ist signifikant besser als 30.000 (t = 2,15, n = 20). Zahlen in lauf4-n17-vergleich.md §5. |
| [lauf4-n17-vergleich.md](lauf4-n17-vergleich.md) | **GR00T N1.7 gegen N1.6.** Vierter Lauf (Job 16634862, N1.7, Vision + Jitter + Split, 44k Steps @ 4 GPUs, State-Dropout 0,2) gegen Lauf 3. **Open-Loop am Ende gleichauf, nicht besser:** bester N1.7-Checkpoint 0,00512 (35k) gegen 0,00482 (N1.6, 44k), n.s.; N1.7 lernt früher (signifikant besser bei 10k/20k), Plateau ab ~30k, trainiert 1,6× schneller (7 h 53 min statt 12 h 27 min). Enthält die Nachmessung von Lauf 3 (20 × 750, §5), die dessen Overfitting-Befund widerlegt. |
| [diagnose-chronik.md](diagnose-chronik.md) | **Diagnose-Chronik (fortlaufendes Protokoll ab Lauf 08)** — chronologisches Protokoll aller Sim-/RL-Diagnoseläufe (Kamera-Fixes, Domain-Gap-Messläufe, Greif-Physik Lauf 29, `span`-Gate Lauf 32, `TUNE_VISUAL` im Closed Loop Lauf 34, Würfellage-Rekonstruktion, Geometriefehler-Korrektur Läufe 38–51, N1.7 Läufe 54–56, Orakel und Registrierung 57–69, Domain-Gap 65–71, Co-Training v3 72–97). Hierher zeigen die projektweiten „Lauf N"-Verweise. Über 2000 Zeilen — Einstieg über das Lauf-Inhaltsverzeichnis/die Anker, nicht linear lesen. |
| [domain-gap-analyse.md](domain-gap-analyse.md) | **Domain-Gap-Messung** — Cosine-Distanz Real→Sim pro Kamera via vortrainiertes SigLIP (Näherung, nicht der Encoder der Policy). Gültiges Ergebnis (Neumessung 2026-08-08 nach Kamerakalibrierung + Albedo-Fixes): Mittelwert **0,22**, `cam_left_wrist` weiterhin Ausreißer bei **0,36** (Kriterium < 0,35 knapp verfehlt). Die Erstmessung (0,26 / 0,43) ist überholt; seit der Geometrie-Korrektur vom 2026-08-24 nicht neu gemessen. |
| [sim-bewertung.md](sim-bewertung.md) | **Methodik-Review** — Ist Closed-Loop-Sim sinnvoll/korrekt? Argumentiert: 0-%-Ergebnis ist ein erwarteter Real→Sim-Gap (SIMPLER), Eval-DR greift nicht, Open-Loop-MSE ist die valide Metrik; `TUNE_VISUAL` wurde durch Lauf 34 relativiert und die Sim-Korrektheit durch den Fund vom 2026-08-24 (Nachtrag im Dokument). Mit Code-Befunden + Quellen. |
| [basismodell-referenz-eval.md](basismodell-referenz-eval.md) | **Pipeline-Validierung.** Basismodell zero-shot auf RoboCasa GR-1 Tabletop (robosuite/MuJoCo, 2× RTX PRO 6000). Aggregat über 12 Tasks **47,7 % ≈ 47,8 %** erwartet (Δ 0,1 pp) → GR00T-Inferenz-Pipeline extern bestätigt; die 0 % im Closed-Loop sind der Domain-Gap, keine kaputte Harness. |

## Zeitleiste

Die Tabelle oben ordnet nach Wichtigkeit; chronologisch liegen die Auswertungen so:

| Wann | Was | Dokument |
|---|---|---|
| 2026-06-02/03 | Lauf 1, Zwischenstand (`i6n1t613`, noch laufend) | [lauf1-zwischenstand.md](../archiv/lauf1-zwischenstand.md) |
| 2026-06-03/04 | Lauf 1, Abschluss (175k Steps, frozen Encoder) | [lauf1-auswertung.md](lauf1-auswertung.md) |
| 2026-06-05 | Lauf 2 (`ajgoskon`, `tune_visual` ohne Augmentierung) | [lauf2-vision-auswertung.md](lauf2-vision-auswertung.md) |
| 2026-06-05 | Methodik-Review Closed-Loop-Sim | [sim-bewertung.md](sim-bewertung.md) |
| 2026-07-19 | RoboCasa-Referenz-Eval (Pipeline-Validierung, 47,7 %) | [basismodell-referenz-eval.md](basismodell-referenz-eval.md) |
| 2026-08-08 | Domain-Gap-Neumessung (ersetzt Erstmessung vom 2026-06-04) | [domain-gap-analyse.md](domain-gap-analyse.md) |
| 2026-08-08 → 26 | Diagnose-Läufe 08–53, fortlaufend (Kamera, Greif-Physik, `span`-Gate, `TUNE_VISUAL`, Würfellage, Geometriefehler) | [diagnose-chronik.md](diagnose-chronik.md) |
| 2026-08-13/14 | Lauf 3 (`tp1nc699`, Vision + Jitter + Split, Checkpoint-Sweep) | [lauf3-vision-split-auswertung.md](lauf3-vision-split-auswertung.md) |
| 2026-10-12/13 | Nachmessung Lauf 3 (20 × 750, Overfitting widerlegt); Lauf 4 (N1.7, Job 16634862) + Sweeps | [lauf4-n17-vergleich.md](lauf4-n17-vergleich.md) |
| 2026-10-12 → 29 | Diagnose-Läufe 54–97: N1.7 in der Sim, Orakel und Würfelversatz 3 cm, Szene Lauf 70, Co-Training v3 (Gate, Closed Loop bis `lifted_clear` 2/60) | [Läufe 54/55](diagnose-chronik.md#läufe-5455-die-ersten-n17-checkpoints-in-der-sim), [62](diagnose-chronik.md#lauf-62-versatz-sweep--die-würfel-liegen-3-cm-zu-weit-innen), [73–76](diagnose-chronik.md#läufe-7376-co-training-läufe-v3-auf-kisski), [97](diagnose-chronik.md#lauf-97-closed-loop-des-besten-kandidaten-auf-allen-60-testlagen) |

> Der **Baseline-Vergleich** (stock G1-Greifer, `SIM_MODE=baseline`) ist noch nicht gelaufen —
> die Vorbereitung liegt bei den operativen Sim-Anleitungen:
> [../archiv/baseline-eval.md](../archiv/baseline-eval.md).

## Verwandte Dokumentation

- [Doku-Übersicht](../README.md) — globaler Navigations-Hub
- [Training](../training/README.md) — die Läufe, die hier ausgewertet werden
- [Simulation](../simulation/README.md) — die Sim-Eval, deren Messungen hier liegen
- [Weiterführende Arbeiten](../weiterfuehrend/README.md) — Folgeschritte aus den Befunden (RL-Plan etc.)
