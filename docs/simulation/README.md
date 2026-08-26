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
| [sim-eval-anleitung.md](sim-eval-anleitung.md) | **Primärer Workflow.** Schritt-für-Schritt-Anleitung für die Sim-Eval: gemeinsame Einmal-Schritte (Image bauen → Checkpoint/USD bereitstellen), dann Weg A IKR-Server (`server_rl_run.sh`, Standard) oder Weg B vast.ai (Instanz konfigurieren → überwachen). Enthält auch den Open-Loop-Replay als Diagnose-Lauf. |
| [live-ansicht.md](live-ansicht.md) | **Live zuschauen statt Videos holen.** `LIVESTREAM=2` lässt Isaac Sim seinen 3D-Viewport per WebRTC streamen; geöffnet wird er von der Desktop-App *Isaac Sim WebRTC Streaming Client* — freie Kamera, Szene drehen. Gilt für Sim-Eval, Baseline, Greif-Test und RL. Client-Installation, `livecheck`, Fehlersuche. Enthält außerdem `view` — die Szene **ohne Modell, ohne Checkpoint, ohne `HF_TOKEN`**, der billigste Weg zum ersten Bild — und `webview`, denselben Viewport **im Browser** samt Maus/Tastatur, ohne App-Installation. Gebaut 2026-08-13 (`view`/`webview`: 2026-08-17), **Hardware-Test offen**. |
| [inferenz-optimierung.md](inferenz-optimierung.md) | **Backends `eager`, `compile`, `tensorrt` + synchrones Kamerarendering.** Erklärt die Auswahl per `GROOT_INFERENCE_BACKEND`, baut den N1.6-DiT über ONNX als GPU-spezifische TensorRT-Engine und vergleicht Parität sowie Closed-Loop-Durchsatz. Enthält die vollständigen IKR-Server-Kommandos. |
| [wuerfellage-rekonstruktion.md](wuerfellage-rekonstruktion.md) | **Würfellage aus den Realbildern.** Wie die Position der drei Würfel aus den realen Trainingsaufnahmen bestimmt und in Sim-Koordinaten überführt wird: Farbsegmentierung → Kamerastrahl → Schnitt mit der Würfelebene. Enthält die vollständige Transformationskette samt Vorzeichenkonventionen, die beiden gescheiterten Vorgängerverfahren, die Liste der Annahmen und den offenen Abnahme-Test. **Übergabedokument** für alle, die den Co-Training-Renderer weiterbauen. |
| [wuerfellage-rekonstruktion-bewertung.md](wuerfellage-rekonstruktion-bewertung.md) | **Code-Audit des Layout-Verfahrens.** Vergleicht die Doku mit der tatsächlichen Implementierung (Pfad A vs. Pfad B/v4), listet Abweichungen und priorisierte Empfehlungen. |
| [wuerfellage-rekonstruktion-lauf35-befund.md](wuerfellage-rekonstruktion-lauf35-befund.md) | **Abnahmelauf.** Befund des `pick_anchored_homography`-Kalibrierungslaufs; empfahl die Aufgabe von Pfad B zugunsten von Pfad A. |
| [replay-videos-aus-realdaten.md](replay-videos-aus-realdaten.md) | **Verfahren aufgegeben, als Protokoll erhalten.** Dokumentiert den früheren Kalibrierungsweg `pick_anchored_homography` (Pfad B): Schrittfolge `replay-prepare` → `replay-calibrate` → `replay-poses` → `replay-render`, standardmäßig auf zehn Episoden begrenzt. Schreibt nur MP4s; Würfel werden einmal gesetzt und danach ausschließlich von PhysX bewegt. |
| [basismodell-referenzaufgabe.md](basismodell-referenzaufgabe.md) | Recherche, was das **Basismodell GR00T N1.6** zero-shot kann + Plan, eine vom Basismodell beherrschte Referenzaufgabe (RoboCasa GR-1 Tabletop) zur **Validierung der Sim-Pipeline** zu reproduzieren. |
| [robocasa-referenz-eval.md](robocasa-referenz-eval.md) | **Bedienung** der Referenz-Eval: Basismodell zero-shot auf RoboCasa GR-1 Tabletop (Docker/vast.ai bzw. KISSKI), Env-Vars, Akzeptanzkriterien, Caveats. Skripte unter [`Simulation/robocasa_reference/`](../../Simulation/robocasa_reference/). |
| [baseline-eval.md](baseline-eval.md) | **Baseline-Eval (Vorbereitung).** Un-finetuntes `GR00T-N1.6-3B` + stock G1-Greifer (`SIM_MODE=baseline`) auf demselben Block-Stacking-Task: parallele Pipeline, TODO-Checkliste, OOD-Einordnung. **Erster Lauf steht aus.** |

> **Mess- und Methodik-Ergebnisse der Sim-Eval** liegen unter [`../ergebnisse/`](../ergebnisse/README.md):
> [Methodik-Review](../ergebnisse/sim-bewertung.md), [Domain-Gap-Messung](../ergebnisse/domain-gap-analyse.md)
> und die [Diagnose-Chronik](../ergebnisse/diagnose-chronik.md) (fortlaufend ab Lauf 08). Der **Livestream-Plan**
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

Die Sim braucht **Ampere+ mit RT-Cores** (z. B. L40, RTX 4090, A6000) — **nicht** A100/H100,
**nicht** Turing. Details und Korrekturen zur ursprünglichen Analyse:
[umsetzungsnotizen.md](umsetzungsnotizen.md) §1.
