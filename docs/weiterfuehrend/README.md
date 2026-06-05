# Weiterführende Arbeiten

Konzept- und Planungs-Dokumente für mögliche Folgeschritte — **recherchiert, aber noch nicht
umgesetzt**. Sie grenzen sich bewusst von den operativen Anleitungen ([`../training/`](../training/README.md),
[`../simulation/`](../simulation/README.md)) ab und speisen das gleichnamige Kapitel der Projektarbeit.

| Dokument | Wann lesen |
|---|---|
| [reinforcement-learning-plan.md](reinforcement-learning-plan.md) | **RL-Plan (Konzept).** Was nötig wäre, um das Modell per Reinforcement Learning (statt nur Behavior Cloning) zu trainieren — Bausteine (Reward, Flow-kompatibler RL-Algorithmus, Rollouts), Phasen-Plan, projektspezifische Umsetzung und der GPU-/Rendering-Konflikt. Motiviert durch den Domain-Gap-Befund aus [`../ergebnisse/lauf1-auswertung.md`](../ergebnisse/lauf1-auswertung.md). |
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
