# Simulation — Closed-Loop-Eval in Isaac Lab

> **TL;DR:** Navigations-Hub der Sim-Doku: verlinkt alle operativen Anleitungen (Setup, vast.ai,
> Live-Ansicht, Inferenz-Optimierung, Würfellage) sowie das Archiv überholter Planungsdokumente.
> Einstiegspunkt, um sich in der Closed-Loop-Eval-Doku zu orientieren.

Closed-Loop-Evaluation des feingetunten GR00T-N1.6-Checkpoints in der Isaac-Lab-Simulation
auf dem Unitree G1 + DEX3-Hand (Block-Stacking). Ein GR00T-Policy-Server beantwortet die
ZMQ-Requests eines Isaac-Lab-Sim-Clients.

## Einstieg

| Dokument | Wann lesen |
|---|---|
| [umsetzungsnotizen.md](umsetzungsnotizen.md) | **Zuerst.** Lessons Learned, bekannte Fixes (Triton/gcc, flash-attn, zmq, SSH), Kamera-Rekonstruktion, Open-Loop-Replay-Diagnose. Stand bis Juni 2026 — die Sim-Erkenntnisse seit August (Isaac-Sim-6.0-Port, Kamera-Neukalibrierung, Greif-Diagnostik) stehen in [../ergebnisse/diagnose-chronik.md](../ergebnisse/diagnose-chronik.md). |
| [vastai-anleitung.md](vastai-anleitung.md) | **Primärer Workflow.** Schritt-für-Schritt-Anleitung für die Sim-Eval auf vast.ai (Image bauen → Checkpoint/USD bereitstellen → Instanz konfigurieren → überwachen). Enthält auch den Open-Loop-Replay als Diagnose-Lauf. |
| [live-ansicht.md](live-ansicht.md) | **Live zuschauen statt Videos holen.** `LIVESTREAM=2` lässt Isaac Sim seinen 3D-Viewport per WebRTC streamen; geöffnet wird er von der Desktop-App *Isaac Sim WebRTC Streaming Client* — freie Kamera, Szene drehen. Gilt für Sim-Eval, Baseline, Greif-Test und RL. Client-Installation, `livecheck`, Fehlersuche. Enthält außerdem `view` — die Szene **ohne Modell, ohne Checkpoint, ohne `HF_TOKEN`**, der billigste Weg zum ersten Bild — und `webview`, denselben Viewport **im Browser** samt Maus/Tastatur, ohne App-Installation. Gebaut 2026-08-13 (`view`/`webview`: 2026-08-17), **Hardware-Test offen**. |
| [inferenz-optimierung.md](inferenz-optimierung.md) | **Backends `eager`, `compile`, `tensorrt` + synchrones Kamerarendering.** Erklärt die Auswahl per `GROOT_INFERENCE_BACKEND`, baut den N1.6-DiT über ONNX als GPU-spezifische TensorRT-Engine und vergleicht Parität sowie Closed-Loop-Durchsatz. Enthält die vollständigen IKR-Server-Kommandos. |
| [wuerfellage-rekonstruktion.md](wuerfellage-rekonstruktion.md) | **Würfellage aus den Realbildern.** Wie die Position der drei Würfel aus den realen Trainingsaufnahmen bestimmt und in Sim-Koordinaten überführt wird: Farbsegmentierung → Kamerastrahl → Schnitt mit der Würfelebene. Enthält die vollständige Transformationskette samt Vorzeichenkonventionen, die beiden gescheiterten Vorgängerverfahren, die Liste der Annahmen und den offenen Abnahme-Test. **Übergabedokument** für alle, die den Co-Training-Renderer weiterbauen. |
| [basismodell-referenzaufgabe.md](basismodell-referenzaufgabe.md) | Recherche, was das **Basismodell GR00T N1.6** zero-shot kann + Plan, eine vom Basismodell beherrschte Referenzaufgabe (RoboCasa GR-1 Tabletop) zur **Validierung der Sim-Pipeline** zu reproduzieren. |
| [robocasa-referenz-eval.md](robocasa-referenz-eval.md) | **Bedienung** der Referenz-Eval: Basismodell zero-shot auf RoboCasa GR-1 Tabletop (Docker/vast.ai bzw. KISSKI), Env-Vars, Akzeptanzkriterien, Caveats. Skripte unter [`Simulation/robocasa_reference/`](../../Simulation/robocasa_reference/). |
| [baseline-eval.md](baseline-eval.md) | **Baseline-Eval (Vorbereitung).** Un-finetuntes `GR00T-N1.6-3B` + stock G1-Greifer (`SIM_MODE=baseline`) auf demselben Block-Stacking-Task: parallele Pipeline, TODO-Checkliste, OOD-Einordnung. **Erster Lauf steht aus.** |

> **Mess- und Methodik-Ergebnisse der Sim-Eval** liegen unter [`../ergebnisse/`](../ergebnisse/README.md):
> [Methodik-Review](../ergebnisse/sim-bewertung.md), [Domain-Gap-Messung](../ergebnisse/domain-gap-analyse.md)
> und die [Diagnose-Chronik der Läufe 08–34](../ergebnisse/diagnose-chronik.md). Der **Livestream-Plan**
> liegt unter [`../weiterfuehrend/livestream-plan.md`](../weiterfuehrend/livestream-plan.md).

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
