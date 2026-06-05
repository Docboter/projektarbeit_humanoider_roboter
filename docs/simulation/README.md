# Simulation — Closed-Loop-Eval in Isaac Lab

Closed-Loop-Evaluation des feingetunten GR00T-N1.6-Checkpoints in der Isaac-Lab-Simulation
auf dem Unitree G1 + DEX3-Hand (Block-Stacking). Ein GR00T-Policy-Server beantwortet die
ZMQ-Requests eines Isaac-Lab-Sim-Clients.

## Einstieg

| Dokument | Wann lesen |
|---|---|
| [implementation-notes.md](implementation-notes.md) | **Zuerst.** Lessons Learned, bekannte Fixes (Triton/gcc, flash-attn, zmq, SSH), Kamera-Rekonstruktion, Open-Loop-Replay-Diagnose, aktueller Stand. |
| [sim-bewertung.md](sim-bewertung.md) | **Methodik-Review (2026-06-05).** Ist Closed-Loop-Sim sinnvoll/korrekt? Belegt: 0-%-Ergebnis ist der erwartete Real→Sim-Gap (SIMPLER), `TUNE_VISUAL`/Eval-DR greifen nicht, Open-Loop-MSE ist die valide Metrik. Mit Code-Befunden + Quellen. |
| [vastai-anleitung.md](vastai-anleitung.md) | **Primärer Workflow.** Schritt-für-Schritt-Anleitung für die Sim-Eval auf vast.ai (Image bauen → Checkpoint/USD bereitstellen → Instanz konfigurieren → überwachen). Enthält auch den Open-Loop-Replay als Diagnose-Lauf. |
| [baseline-unitree-g1.md](baseline-unitree-g1.md) | **Baseline-Vergleich.** Un-finetuntes `GR00T-N1.6-3B` + stock G1-Greifer (`UNITREE_G1`) auf demselben Block-Stacking-Task. Parallele Pipeline (`SIM_MODE=baseline`), TODO-Checkliste vor dem ersten Run, OOD-Einordnung. |
| [domain-gap-analyse.md](domain-gap-analyse.md) | **Domain-Gap-Messung (2026-06-04).** Cosine-Distanz Real→Sim pro Kamera via frozen SigLIP-ViT. Ergebnis: Mittelwert 0.26, `cam_left_wrist` kritisch bei 0.43. Drei Handlungsoptionen mit Aufwand/Risiko-Abwägung. |

## Archiv

Im Unterordner [archiv/](archiv/) liegen historische Planungs- und Analyse-Dokumente. Sie sind
durch die tatsächliche Umsetzung überholt (Details in [implementation-notes.md](implementation-notes.md)
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
Korrekturen aus [implementation-notes.md](implementation-notes.md) §1).
