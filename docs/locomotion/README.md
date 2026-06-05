# Lokomotion — den G1 zum Laufen bringen

Recherche-Dokumentation zur Frage: **Wie lässt sich die Lokomotion des Unitree G1
freischalten**, sodass der Roboter sich bewegen kann und nicht nur fix an einer Position
am Tisch steht?

| Dokument | Inhalt |
|---|---|
| [lokomotion-recherche.md](lokomotion-recherche.md) | **Vollständige Recherche** — Warum der Roboter aktuell fixiert ist, GR00T-N1.6-Whole-Body-Architektur, Unitree-G1-Lokomotions-Stacks, Loco-Manipulation-Forschungslandschaft, konkrete Integrationspfade für dieses Projekt + Quellen |

**Kurzfassung:** Der G1 steht heute fest, weil (1) der Sim-Root-Link hart fixiert ist
(`fix_root_link=True`), (2) das GR00T-Embodiment nur 28 DOF (Arme + Hände, **keine Beine**)
umfasst und (3) der Trainingsdatensatz reine Tischmanipulation ohne Basisbewegung enthält.
Die Beine **existieren bereits** im USD-Asset (29-DOF-G1-URDF). Lokomotion muss aus einer
**separaten Steuerung** kommen — entweder als entkoppelter RL-Lauf-Controller (Beine) neben
GR00T (Arme/Hände), exakt wie es GR00T N1.5/N1.6 und die Unitree-Hardware ohnehin machen,
oder über ein Whole-Body-VLA-Retraining mit neuen Daten.
</content>
</invoke>
