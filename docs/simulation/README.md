# Simulation — Closed-Loop-Eval in Isaac Lab

Closed-Loop-Evaluation des feingetunten GR00T-N1.6-Checkpoints in der Isaac-Lab-Simulation
auf dem Unitree G1 + DEX3-Hand (Block-Stacking). Ein GR00T-Policy-Server beantwortet die
ZMQ-Requests eines Isaac-Lab-Sim-Clients.

## Einstieg

| Dokument | Wann lesen |
|---|---|
| [umsetzungsnotizen.md](umsetzungsnotizen.md) | **Zuerst.** Lessons Learned, bekannte Fixes (Triton/gcc, flash-attn, zmq, SSH), Kamera-Rekonstruktion, Open-Loop-Replay-Diagnose. Stand bis Juni 2026 — die Sim-Erkenntnisse seit August (Isaac-Sim-6.0-Port, Kamera-Neukalibrierung, Greif-Diagnostik) stehen in [../weiterfuehrend/rl-anleitung.md](../weiterfuehrend/rl-anleitung.md). |
| [vastai-anleitung.md](vastai-anleitung.md) | **Primärer Workflow.** Schritt-für-Schritt-Anleitung für die Sim-Eval auf vast.ai (Image bauen → Checkpoint/USD bereitstellen → Instanz konfigurieren → überwachen). Enthält auch den Open-Loop-Replay als Diagnose-Lauf. |
| [basismodell-referenzaufgabe.md](basismodell-referenzaufgabe.md) | Recherche, was das **Basismodell GR00T N1.6** zero-shot kann + Plan, eine vom Basismodell beherrschte Referenzaufgabe (RoboCasa GR-1 Tabletop) zur **Validierung der Sim-Pipeline** zu reproduzieren. |
| [robocasa-referenz-eval.md](robocasa-referenz-eval.md) | **Bedienung** der Referenz-Eval: Basismodell zero-shot auf RoboCasa GR-1 Tabletop (Docker/vast.ai bzw. KISSKI), Env-Vars, Akzeptanzkriterien, Caveats. Skripte unter [`Simulation/robocasa_reference/`](../../Simulation/robocasa_reference/). |

> **Mess- und Methodik-Ergebnisse der Sim-Eval** liegen jetzt unter [`../ergebnisse/`](../ergebnisse/README.md):
> [Methodik-Review](../ergebnisse/sim-bewertung.md), [Domain-Gap-Messung](../ergebnisse/domain-gap-analyse.md)
> und [Baseline-Vergleich](../ergebnisse/baseline-unitree-g1.md). Der **Livestream-Plan** liegt unter
> [`../weiterfuehrend/livestream-plan.md`](../weiterfuehrend/livestream-plan.md).

## Archiv

Im Unterordner [archiv/](archiv/) liegen historische Planungs- und Analyse-Dokumente. Sie sind
durch die tatsächliche Umsetzung überholt (Details in [umsetzungsnotizen.md](umsetzungsnotizen.md)
§13), bleiben aber als Entscheidungs-Kontext erhalten:

| Dokument | Inhalt (historisch) |
|---|---|
| [archiv/isaac-lab-plan.md](archiv/isaac-lab-plan.md) | Ursprünglicher Implementierungsplan |
| [archiv/sim-docker-build.md](archiv/sim-docker-build.md) | Zwei-Container-Build-Plan (für KISSKI weiterhin gültig) |
| [archiv/kisski-desktop.md](archiv/kisski-desktop.md) | JupyterHPC-Desktop-Test auf KISSKI (RTX 5000) |
| [archiv/gpu-kompatibilitaet.md](archiv/gpu-kompatibilitaet.md) | GPU-Kompatibilitätsanalyse (RT-Cores) |

## GPU-Anforderung (Kurzfassung)

Die Sim braucht **Ampere+ mit RT-Cores** — L40, L40S, RTX 4090, RTX 6000 Ada, RTX 5000 Ada,
A6000. **Nicht** A100/H100 (keine RT-Cores), **nicht** Turing (RTX 5000 Quadro, Titan RTX, V100).
Vollständige Analyse: [archiv/gpu-kompatibilitaet.md](archiv/gpu-kompatibilitaet.md) (mit den
Korrekturen aus [umsetzungsnotizen.md](umsetzungsnotizen.md) §1).
