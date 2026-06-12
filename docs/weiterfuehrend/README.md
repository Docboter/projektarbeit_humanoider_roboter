# Weiterführende Arbeiten

Konzept- und Planungs-Dokumente für mögliche Folgeschritte — überwiegend **recherchiert, noch nicht
end-to-end validiert** (Ausnahme: für RL existiert inzwischen eine erste Implementierung, siehe unten).
Sie grenzen sich bewusst von den operativen Anleitungen ([`../training/`](../training/README.md),
[`../simulation/`](../simulation/README.md)) ab und speisen das gleichnamige Kapitel der Projektarbeit.

| Dokument | Wann lesen |
|---|---|
| [rl-anleitung.md](rl-anleitung.md) | **RL-Bedienungsanleitung (operativ).** Schritt-für-Schritt: Image bauen, BC-Checkpoint hochladen, vast.ai-Instanz (RT-Core-GPU) konfigurieren, RL starten (`entrypoint_rl.sh` + `RL_*`-Env-Vars), überwachen (W&B-Erfolgsrate), RL-Checkpoints sichern. Inkl. Smoke-Test + LIVE-CHECK-Punkte. |
| [reinforcement-learning-plan.md](reinforcement-learning-plan.md) | **RL-Plan + erste Implementierung.** Bausteine (Reward, Flow-kompatibler RL-Algorithmus, Rollouts), Phasen-Plan und GPU-/Rendering-Konflikt. **Inzwischen gebaut:** Shaped Reward, batched Obs, FPO-Trainer ([`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py)) + Launch-Infra (`USE_RL`, `entrypoint_rl.sh`, `kisski_rl_submit.sh`) — Algorithmus-Kern verifiziert, End-to-End-Validierung offen (RT-Core-GPU). Motiviert durch den Domain-Gap-Befund aus [`../ergebnisse/lauf1-auswertung.md`](../ergebnisse/lauf1-auswertung.md). |
| [lokomotion-recherche.md](lokomotion-recherche.md) | **Lokomotions-Recherche.** Warum der Roboter aktuell fixiert ist (Code-Analyse), GR00T-N1.6-Whole-Body-Control (entkoppelt: RL-Beine + IK/VLA-Arme), Unitree-G1-Lokomotions-Stacks (`unitree_rl_gym`/`unitree_rl_lab`, SDK `LocoClient`), Loco-Manipulation-Forschung, konkrete Integrationspfade + Quellen. |
| [livestream-plan.md](livestream-plan.md) | **Livestream-Plan.** Live-Stream der Sim via WebRTC (Echtzeit-Viewport vom Remote-GPU). Additiv zur bestehenden Sim-Eval, noch nicht umgesetzt. |

> **Kurzfassung Lokomotion:** Der G1 steht heute fest, weil (1) der Sim-Root-Link hart fixiert
> ist (`fix_root_link=True`), (2) das GR00T-Embodiment nur 28 DOF (Arme + Hände, **keine Beine**)
> umfasst und (3) der Trainingsdatensatz reine Tischmanipulation ohne Basisbewegung enthält. Die
> Beine **existieren bereits** im USD-Asset (29-DOF-G1-URDF). Lokomotion muss aus einer **separaten
> Steuerung** kommen — entweder als entkoppelter RL-Lauf-Controller (Beine) neben GR00T
> (Arme/Hände), wie es GR00T N1.5/N1.6 und die Unitree-Hardware ohnehin tun, oder über ein
> Whole-Body-VLA-Retraining mit neuen Daten.

## Verwandte Dokumentation

- [Doku-Übersicht](../README.md) — globaler Navigations-Hub
- [Ergebnisse](../ergebnisse/README.md) — die Befunde, die diese Folgeschritte motivieren
- [Training](../training/README.md) · [Simulation](../simulation/README.md) — die aktuelle, umgesetzte Pipeline
