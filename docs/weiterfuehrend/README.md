# Weiterführende Arbeiten

> **TL;DR:** Navigations-Hub für Konzept- und Planungsdokumente zu möglichen Folgeschritten (RL,
> Lokomotion, Live-Ansicht, CLI-Menüführung) — größtenteils recherchiert, noch nicht end-to-end
> validiert. Ausnahmen: RL läuft seit 2026-08-08 end-to-end, ebenso die Live-Ansicht-Spur B.

Konzept- und Planungs-Dokumente für mögliche Folgeschritte — überwiegend **recherchiert, noch nicht
end-to-end validiert**. Ausnahmen: **RL** ist gebaut und läuft seit 2026-08-08 end-to-end auf
RT-Core-Hardware, und die **Live-Ansicht des RL-Laufs** („Spur B" des Livestream-Plans) ist seit
2026-08-08 umgesetzt — beides siehe unten.
Sie grenzen sich von den operativen Anleitungen ([`../training/`](../training/README.md),
[`../simulation/`](../simulation/README.md)) ab und speisen das gleichnamige Kapitel der
Projektarbeit. Einzige operative Ausnahme hier: [rl-anleitung.md](rl-anleitung.md) — sie bleibt
beim RL-Plan, weil RL erst nach dem Co-Training wieder ansteht; ihre Lauf-Historie liegt als
[Diagnose-Chronik](../ergebnisse/diagnose-chronik.md) bei den Ergebnissen.

| Dokument | Wann lesen |
|---|---|
| [rl-anleitung.md](rl-anleitung.md) | **RL-Bedienungsanleitung (operativ, schlank).** Schritt-für-Schritt: Image bauen, BC-Checkpoint hochladen, RT-Core-GPU (eigener Docker-Server als Pfad B oder vast.ai als Pfad A), RL starten (`entrypoint_rl.sh` + `RL_*`-Env-Vars), überwachen, Checkpoints sichern; plus die Diagnose-Werkzeuge `gap`/`eval`/`grasp`/`span`. Die komplette Lauf-Historie (Läufe 08–34) steht in der [Diagnose-Chronik](../ergebnisse/diagnose-chronik.md). |
| [reinforcement-learning-plan.md](reinforcement-learning-plan.md) | **RL-Plan + Implementierung.** Bausteine (Reward, Flow-kompatibler RL-Algorithmus, Rollouts), Phasen-Plan und GPU-/Rendering-Konflikt. **Gebaut und gelaufen:** Shaped Reward, batched Obs, FPO-Trainer ([`rl_finetune.py`](../../Simulation/g1_dex3_sim/rl_finetune.py)) + Launch-Infra (`USE_RL`, `entrypoint_rl.sh`, `kisski_rl_submit.sh`) — eine vollständige Iteration ist am 2026-08-08 auf RTX PRO 6000 Blackwell / Isaac Sim 6.0 durchgelaufen; offen bleibt die **Lernwirkung** (Erfolgsrate über viele Iterationen) und der Render-Durchsatz. Motiviert durch den Domain-Gap-Befund aus [`../ergebnisse/lauf1-auswertung.md`](../ergebnisse/lauf1-auswertung.md). |
| [lokomotion-recherche.md](lokomotion-recherche.md) | **Lokomotions-Recherche.** Warum der Roboter aktuell fixiert ist (Code-Analyse), GR00T-N1.6-Whole-Body-Control (entkoppelt: RL-Beine + IK/VLA-Arme), Unitree-G1-Lokomotions-Stacks (`unitree_rl_gym`/`unitree_rl_lab`, SDK `LocoClient`), Loco-Manipulation-Forschung, konkrete Integrationspfade + Quellen. |
| [cli-menuefuehrung.md](cli-menuefuehrung.md) | **Geführte CLI-Menüs für Training und Simulation — umgesetzt am 2026-08-20.** Startet man ein Skript ohne Parameter, fragt es die nötigen Werte ab und erklärt sie (`?` zeigt den Langtext); `MENU=0` schaltet es ab, jeder bisherige Aufruf läuft unverändert. Kern ist nicht das Menü, sondern eine **deklarative Parameter-Spezifikation als einzige Quelle** unter [`tools/menu/`](../../tools/menu/) — `tools/gen_docs.sh` gleicht sie gegen die echten `${VAR:-…}` und die Doku-Tabellen ab und schlägt bei Drift fehl. Reines Bash, nur auf dem Host, Container bleibt autonom. Enthält §12 mit den Abweichungen vom Plan und zwei dabei gefundenen Defekten (Trainings-Launcher las keine `.env.local` und reichte 6 von 19 Env-Vars durch). Prüfstand: `tools/test_menu.sh`, 30 Prüfungen. |
| [livestream-plan.md](livestream-plan.md) | **Live-Ansicht-Plan (v3).** Live-Äquivalent zu den MP4s, zweigleisig: **Spur A** WebRTC-Viewport (Isaac Sim nativ, für Sim-/Baseline-Eval) und **Spur B** leichtgewichtiger MJPEG-Frame-Stream im Browser. **Spur B ist für den RL-Lauf gebaut** ([`live_view.py`](../../Simulation/g1_dex3_sim/live_view.py), `LIVE_VIEW=1`) — Bedienung in [rl-anleitung.md](rl-anleitung.md) Schritt 6; dazu Option C (Rollout-Video ins W&B-Dashboard). **Spur A ist seit 2026-08-13 gebaut**: alle 6 Defekte behoben (Port 8211 raus, versionsabhängige Kit-Settings, gemeinsame [`lib_livestream.sh`](../../Simulation/scripts/lib_livestream.sh)), `LIVESTREAM=2` wirkt auf alle vier Läufe und ersetzt die MP4s. Seit 2026-08-17 hat Spur A **zwei Clients**: die native App oder den Browser (`server_rl_run.sh webview`, Port 8210, mit Maus/Tastatur) — Bedienung in [live-ansicht.md](../simulation/live-ansicht.md), **Hardware-Test offen**. |

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
