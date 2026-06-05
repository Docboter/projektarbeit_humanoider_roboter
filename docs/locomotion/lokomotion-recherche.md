# Lokomotion des Unitree G1 freischalten — Recherche

> **Frage:** Aktuell steht der Roboter fix an einer Position und manipuliert nur den Tisch.
> Was ist nötig, damit er sich **bewegen** (laufen, drehen, neu positionieren) kann?
>
> **Stand:** 2026-06-05 · reine Recherche (Code-Analyse + Internetquellen), noch keine Umsetzung.

---

## 0. TL;DR — Die zentrale Erkenntnis

Lokomotion ist **kein Schalter im bestehenden GR00T-Modell**, sondern eine **zweite,
eigenständige Steuerebene**. Das gilt aus drei Gründen:

1. **Sim:** Der Roboter-Torso ist in Isaac Lab hart am Welt-Frame festgeschweißt
   (`fix_root_link=True`). Es gibt keinen Boden zum Laufen und keine Bein-Aktuatoren.
2. **Modell:** Das G1/DEX3-Embodiment von GR00T umfasst **28 DOF — nur Arme (14) und
   Hände (14), keine Beine**. Der Action-Head kann gar keine Beinbefehle erzeugen.
3. **Daten:** Alle Trainingsdatensätze (`G1_Dex3_*`) sind reine Tischmanipulation per
   AVP-Teleoperation. Die Basis bewegt sich in den Demos **nie** → das Modell hat
   Lokomotion nie gesehen.

**Gute Nachricht:** Die Beine sind **physisch schon da**. Das USD-Asset wird aus dem
**vollen 29-DOF-G1-URDF** (`g1_29dof_with_hand_rev_1_0.urdf`) mit `fix_base=False`
gebaut — Hüft-, Knie-, Sprung- und Taillengelenke sind in der Articulation vorhanden,
sie werden nur zur Laufzeit eingefroren und nicht angesteuert.

**Der etablierte Weg** (NVIDIA GR00T N1.5/N1.6 **und** die Unitree-Hardware machen es
genau so) ist **Entkopplung**: ein RL-Lauf-Controller steuert den **Unterkörper**
(Beine + Taille, hält Balance, folgt Geschwindigkeitsbefehlen), während der **Oberkörper**
(Arme + Hände) vom bestehenden GR00T-Policy bzw. per IK gesteuert wird. → siehe
[§5 Pfad A](#pfad-a--entkoppelt-empfohlen).

---

## 1. Warum der Roboter aktuell fixiert ist (Code-Analyse)

### 1.1 Der Sim-Root-Link ist festgeschweißt

In [`Simulation/g1_dex3_sim/g1_dex3_cfg.py`](../../Simulation/g1_dex3_sim/g1_dex3_cfg.py#L122-L129):

```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=1,
    # Unterkörper fixieren: Beine/Waist werden gesperrt; Roboter steht am Tisch.
    # fix_root_link=True  → stellt den Torso fest (kein Balance-Controller nötig)
    fix_root_link=True,
),
```

`fix_root_link=True` ersetzt das freie Floating-Base-Gelenk durch eine starre Verbindung
zur Welt. Der Torso schwebt bei `pos=(0.0, 0.0, 0.85)` ohne Schwerkrafteinfluss auf die
Basis. Das war eine **bewusste Vereinfachung** für die Manipulations-Eval: kein
Balance-Controller nötig, der Roboter kippt nicht um. Derselbe Trick steht auch in
[`phase_b_test.py`](../../Simulation/g1_dex3_sim/phase_b_test.py#L105).

### 1.2 Der Action-/State-Vektor hat keine Beine

Aus [`examples/G1_DEX3/README.md`](../../app/Groot-1.6/examples/G1_DEX3/README.md) und
[`g1_dex3_cfg.py`](../../Simulation/g1_dex3_sim/g1_dex3_cfg.py#L75-L78):

```
[0:7]   left_arm    — Shoulder P/R/Y, Elbow, Wrist R/P/Y
[7:14]  right_arm   — Shoulder P/R/Y, Elbow, Wrist R/P/Y
[14:21] left_dex3   — Thumb 0/1/2, Middle 0/1, Index 0/1
[21:28] right_dex3  — Thumb 0/1/2, Index 0/1, Middle 0/1
```

28 DOF, **null Beingelenke**. In der Sim sind auch nur vier Aktuator-Gruppen definiert
(`left_arm`, `right_arm`, `left_hand`, `right_hand`) — Hüfte/Knie/Sprunggelenk/Taille
haben **keinen Aktuator** und damit keinen Antrieb.

### 1.3 Die Beine sind aber im Asset vorhanden

[`convert_urdf_to_usd.py`](../../Simulation/g1_dex3_sim/convert_urdf_to_usd.py#L18-L52)
baut das USD aus:

```python
default="/data/assets/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf"
...
cfg = UrdfConverterCfg(
    fix_base=False,           # ← freie Basis im Asset!
    merge_fixed_joints=False,
    ...
)
```

Das **29-DOF-G1-URDF** enthält den vollständigen Beinsatz. `fix_base=False` heißt: das
Asset selbst hat eine bewegliche Basis. Das spätere `fix_root_link=True` in der
Env-Config überschreibt das nur zur **Laufzeit**. Es muss also kein neues Asset gebaut
werden — die Kinematik ist vollständig vorhanden.

### 1.4 Der reale Datensatz enthält keine Lokomotion

Alle `unitreerobotics/G1_Dex3_*`-Datensätze (BlockStacking, ToastedBread, PickApple, …)
sind per **Apple Vision Pro / AVP-Teleoperation** an einem **stehenden** G1 aufgenommen.
Der/die Operator:in bewegt nur Arme und Hände; die Beine laufen im Halte-/Dämpfungsmodus.
**Konsequenz:** Selbst ein perfekt trainiertes GR00T-Modell kann auf dieser Datenbasis
**niemals** Beinaktionen produzieren — die Information fehlt im Datensatz.

---

## 2. Wie der Unitree G1 mechanisch aufgebaut ist

| Körperteil | DOF | Gelenke |
|---|---|---|
| Bein (×2) | 6 je Bein = **12** | Hip Pitch/Roll/Yaw, Knee, Ankle Pitch/Roll |
| Taille (Waist) | 1–3 | Yaw (immer); Roll + Pitch optional (sperrbar → 1-DOF) |
| Arm (×2) | 5 oder 7 je Arm | Shoulder P/R/Y, Elbow, Wrist R/P/Y |
| Hand (Dex3, ×2) | 7 je Hand | 3-Finger, je 7 DOF |

Varianten: **G1-23-DOF** (1-DOF-Taille), **G1-29-DOF** (3-DOF-Taille). Dieses Projekt
nutzt das **29-DOF**-URDF + Dex3-Hände.

**Sprunggelenk-Besonderheit:** Parallelmechanismus mit zwei Modi — **PR-Mode**
(Pitch/Roll, Standard) und **AB-Mode** (direkte A/B-Motoren). Für Lokomotions-RL ist
das relevant, weil die Policy entweder im seriellen (P/R) oder parallelen (A/B) Gelenkraum
ausgeben muss.

> Quellen: [Unitree G1 Overview (QRE Docs)](https://www.docs.quadruped.de/projects/g1/html/g1_overview.html),
> [Unitree G1 EDU Specs (RoboStore)](https://robostore.com/blogs/news/unitree-g1-edu-ultimate-technical-specifications)

---

## 3. NVIDIAs eigene Antwort: GR00T N1.6 Whole-Body Control

Genau diese Frage — Manipulation **und** Fortbewegung im selben Humanoiden — adressiert
**NVIDIA GR00T N1.6** explizit als zentrale Neuerung ("perception, navigation and
**loco-manipulation**").

### 3.1 Entkoppelte Architektur (Decoupled WBC)

GR00T N1.5 und N1.6 verwenden einen **entkoppelten Whole-Body-Controller**:

> *"the decoupled controller (RL for lower body, and IK for upper body) was used in GR00T
> N1.5 and N1.6 models."* — [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)

- **Unterkörper (Beine + Taille):** Reinforcement-Learning-Policy, in Isaac Lab trainiert,
  hält Balance und Mehrfachkontakt, folgt **Geschwindigkeitsbefehlen**.
- **Oberkörper (Arme + Hände):** Inverse Kinematik (IK) bzw. die VLA-/Manipulations-Policy.

Die High-Level-VLA (GR00T) bzw. eine separate Navigations-Policy schickt **einfache
Geschwindigkeitsbefehle** an den WBC; der WBC kümmert sich um Balance und Kontakt:

> *"the navigation policy operates through simple velocity commands to the whole-body
> controller … the low-level whole-body RL policy handles balance and contact, while the
> navigation head focuses on obstacle avoidance."*
> — [NVIDIA Technical Blog: GR00T N1.6 Sim-to-Real](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow/)

### 3.2 Die unified-Variante (Latent-Action-Tokens)

Im voll integrierten N1.6-Workflow sagt die VLA **kompakte latente Action-Tokens** voraus,
die ein gelernter Whole-Body-Controller in **Ganzkörper-Gelenkbefehle (Beine, Arme, Hände)**
dekodiert. Das ist die ambitioniertere, end-to-end-Variante — sie braucht aber
Whole-Body-Trainingsdaten und den WBC-Decoder.

### 3.3 Das Repo: `NVlabs/GR00T-WholeBodyControl`

Offizielle Plattform mit drei Bausteinen:

- **Decoupled WBC** — die oben genannten RL-Beine + IK-Arme aus N1.5/N1.6.
- **GEAR-SONIC** — neuerer generalistischer Whole-Body-Controller.
- **MotionBricks** — Echtzeit-Latent-Generativmodell für Bewegung.

Unterstützt **explizit den Unitree G1**: trainiert auf dem **Bones-SEED-G1-Datensatz**
(142 K+ Motions, ~288 h, auf G1 retargetet). Liefert einen **C++-Inference-Stack** für
echte Hardware und VR-Teleoperation (PICO) zur Datensammlung. Training: Isaac Lab,
64+ GPUs für volles Fine-tuning empfohlen (für ein Studierendenprojekt eher Fine-tuning
vom Release-Checkpoint als from-scratch).

> Quellen: [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl),
> [NVIDIA GR00T N1.6 Blog](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow/),
> [HyperAI: GR00T N1.6 Announcement](https://hyper.ai/en/headlines/72c38e5f5753d41268a6d4f8d4e805f7)

---

## 4. Unitree-G1-Lokomotions-Stacks (was es fertig gibt)

Für die Beine muss **nichts from scratch** erfunden werden — es existieren mehrere
einsatzfähige RL-Lauf-Policies für den G1:

| Stack | Basis | G1-Variante | Notizen |
|---|---|---|---|
| [`unitreerobotics/unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym) | Isaac **Gym** | G1, H1, H1_2, Go2 | Offiziell, Train→Sim2Sim(MuJoCo)→Sim2Real, einfacher Einstieg |
| [`unitreerobotics/unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab) | Isaac **Lab** | **G1-29-DOF** | Offiziell, neuer; braucht **Isaac Lab 2.3.0 / Isaac Sim 5.1**, C++-Deploy |
| [`mintlabkorea/unitree_g1_rl_lab`](https://github.com/mintlabkorea/unitree_g1_rl_lab) | Isaac Lab | G1-29-DOF | Symmetrie-basiert, In-Place-Yaw-Turning |
| [Gait-Conditioned RL (arXiv 2505.20619)](https://arxiv.org/abs/2505.20619) | — | G1 | Stehen/Gehen/Laufen + Übergänge in **einer** rekurrenten Policy, zero-shot Sim2Real |

**Typische Lokomotions-Policy** (Velocity-Tracking):
- **Observation:** Basis-Lineargeschw./Winkelgeschw., Projektion der Schwerkraft,
  Velocity-Command `(vx, vy, ωyaw)`, Beingelenk-Positionen/-Geschwindigkeiten,
  letzte Aktion, ggf. Gangphase/Clock.
- **Action:** Zielpositionen der **12 Bein- (+ggf. Taillen-)Gelenke** (PD-Targets).
- **Export:** JIT/ONNX, läuft mit ~50 Hz auf dem realen Roboter.

> ⚠️ **Isaac-Lab-Versionskonflikt beachten:** `unitree_rl_lab` verlangt Isaac Lab 2.3 /
> Isaac Sim 5.1. Die Sim-Eval dieses Projekts läuft auf einer eigenen Isaac-Lab-Version
> (siehe Sim-Dockerfiles). Vor einer Integration die Versionen abgleichen.

### 4.1 Auf der realen Hardware ist es ebenfalls entkoppelt

Das Unitree-SDK2 spiegelt die Entkopplung 1:1 in **Hardware-APIs**:

- **High-Level `LocoClient`** — semantische Lokomotion: `Move(vx, vy, vyaw)`, `StandUp()`,
  `Damp()`, FSM-Gangsteuerung. Der **interne Unitree-Lauf-Controller** hält die Balance.
- **`arm_sdk` (`rt/arm_sdk`)** — direkte Motorbefehle für die **Arme/Oberkörper** mit
  Weight-Parameter zum sanften Einkoppeln (0 = aus, 1 = voll).
- **Low-Level `rt/lowcmd`** — rohe PD-Kontrolle aller Gelenke (volle Verantwortung selbst).

Heißt: Real kann man den **G1 mit `LocoClient` laufen lassen** und gleichzeitig per
`arm_sdk` die von GR00T berechneten Arm-/Hand-Targets aufspielen — **ohne** dass GR00T je
ein Bein ansteuert.

> Quellen: [G1 SDK Development Guide](https://support.unitree.com/home/en/G1_developer),
> [High-Level Sports Service Interface](https://support.unitree.com/home/en/developer/sports_services),
> [unitree_sdk2 G1 (DeepWiki)](https://deepwiki.com/unitreerobotics/unitree_sdk2/3-g1-humanoid-robot)

---

## 5. Konkrete Integrationspfade für dieses Projekt

### Pfad A — Entkoppelt (empfohlen)

**Idee:** Zwei Policies parallel. Beine = fertiger RL-Lauf-Controller (Velocity-Tracking).
Arme + Hände = bestehendes, fine-getuntes GR00T-Modell. Entspricht GR00T N1.5/N1.6
**und** der Unitree-Hardware.

**In der Simulation (Isaac Lab):**
1. `fix_root_link=True` → **`False`** in
   [`g1_dex3_cfg.py`](../../Simulation/g1_dex3_sim/g1_dex3_cfg.py#L128); Basis als
   Floating Base freigeben.
2. **Boden/Ground-Plane** + Schwerkraft auf die Basis; Start-`pos.z` auf reale Stehhöhe
   (~0.74 m Pelvis) statt 0.85 m schwebend.
3. **Aktuatoren für Beine (12) + Taille** ergänzen (Hip/Knee/Ankle/Waist) mit passenden
   PD-Gains aus `unitree_rl_lab`/`unitree_rl_gym`.
4. **Lokomotions-Policy laden** (JIT/ONNX) und im Control-Loop die Beingelenke ansteuern;
   GR00T steuert weiter Arme/Hände (relative Arm-Deltas, absolute Hand-Targets).
5. Velocity-Command-Quelle: zunächst Skript/Tastatur/Wegpunkte (`vx, vy, ωyaw`), später
   eine Navigations-Policy.
6. **Tisch-Anker beachten:** Block-Stacking setzt die Basis-Pose der Trainingsverteilung
   voraus. Beim Greifen den Roboter an die Tischpose fahren und Beine in Stand halten;
   Lokomotion nur **zwischen** den Manipulationsphasen.

**Auf der realen Hardware:** `LocoClient.Move(...)` für die Beine + `arm_sdk` für die von
GR00T berechneten Arm-/Hand-Targets. Beide Streams laufen unabhängig.

**Aufwand:** mittel · **Risiko:** niedrig · **Datenbedarf:** keine neuen Demos (Lauf-Policy
ist vortrainiert). **Bestes Preis-/Leistungsverhältnis für dieses Projekt.**

### Pfad B — Unified Whole-Body-VLA (GR00T-N1.6-nativ)

**Idee:** GR00T gibt latente Action-Tokens aus, die ein WBC in Ganzkörper-Gelenkbefehle
(inkl. Beine) dekodiert — eine einzige end-to-end-Policy.

**Nötig:**
- Adaption auf **GR00T N1.6** mit Whole-Body-Action-Head + Decoupled-WBC/SONIC aus
  [`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl).
- **Whole-Body-Trainingsdaten** mit Basisbewegung — die `G1_Dex3_*`-Datensätze taugen
  dafür **nicht** (statische Basis). Entweder Bones-SEED-Mocap (Lokomotion) + eigene
  Loco-Manipulation-Teleop (VR/PICO) sammeln, oder Sim-Rollouts.
- Deutlich mehr Compute (NVIDIA nennt 64+ GPUs für volles WBC-Training).

**Aufwand:** hoch · **Risiko:** hoch · **Datenbedarf:** hoch. Wissenschaftlich die
sauberste, aber für den Projektrahmen vermutlich zu groß. Eher als Ausblick.

### Pfad C — Reine Sim-Lokomotions-Demo (schneller Meilenstein)

**Idee:** Unabhängig von GR00T zuerst zeigen, dass der **G1 in Isaac Lab läuft** — mit
einer Stock-Lauf-Policy aus `unitree_rl_gym`/`unitree_rl_lab`. Liefert schnell ein
sichtbares Ergebnis (Roboter geht), validiert Asset/Boden/Aktuatoren und ist die
Vorstufe zu Pfad A. **Empfohlener erster Schritt.**

---

## 6. Forschungslandschaft Loco-Manipulation (Kontext)

Zwei Schulen, beide auf dem **Unitree G1** demonstriert:

**Entkoppelt** (Oberkörper IK/VLA, Unterkörper RL) — robust, einfacher zu trainieren:
- **Mobile-TeleVision (PMP)** — Predictive Motion Priors (CVAE) für Oberkörper; Lauf-Policy
  auf diese Repräsentation konditioniert. [arXiv 2412.07773](https://arxiv.org/pdf/2412.07773)
- **OmniH2O** — Whole-Body-Dexterous-Loco-Manipulation + VR-Teleop zur Datensammlung.
  [arXiv 2406.08858](https://arxiv.org/html/2406.08858v1)

**Unified / fein-granular** (ein Controller für alles) — größerer Workspace, externe Lasten:
- **ULC: Unified and Fine-Grained Controller** — auf G1 validiert, schlägt entkoppelte
  Methoden im Tracking, präzise Manipulation unter Last. [arXiv 2507.06905](https://arxiv.org/abs/2507.06905)
- **WholeBodyVLA** — unified latent VLA für Whole-Body-Loco-Manipulation.
  [arXiv 2512.11047](https://arxiv.org/pdf/2512.11047)
- **Kinematics-Aware Multi-Policy RL** — kraftfähige Loco-Manipulation (G1 trägt 4 kg,
  schiebt 112 kg). [arXiv 2511.21169](https://arxiv.org/pdf/2511.21169)

**Einordnung:** Für dieses Projekt ist der **entkoppelte** Ansatz (Pfad A) der pragmatische
Standard — er deckt sich mit GR00T N1.5/N1.6, der Unitree-Hardware und der Mehrheit der
robusten Demos. Unified-Ansätze sind die Forschungsspitze, brauchen aber mehr Daten/Compute.

---

## 7. Offene Punkte / nächste Schritte

- [ ] **Pfad C zuerst:** Stock-G1-Lauf-Policy (`unitree_rl_gym`/`unitree_rl_lab`) in Isaac
      Lab gegen das vorhandene 29-DOF-USD testen — läuft der Roboter mit Boden + freier Basis?
- [ ] **Isaac-Lab-Version abgleichen:** `unitree_rl_lab` (2.3/Sim 5.1) vs. Projekt-Sim-Version.
- [ ] **Aktuator-Gains** für Beine/Taille aus dem RL-Stack übernehmen.
- [ ] **Geschwindigkeitsbefehl-Quelle** definieren (Skript/Wegpunkte → später Navigation).
- [ ] **Tisch-Anker-Logik:** Manipulation nur an definierter Tischpose; Lokomotion dazwischen.
- [ ] **Hardware-Plan:** `LocoClient` (Beine) + `arm_sdk` (GR00T-Arme/Hände) verzahnen.
- [ ] **Ausblick Pfad B:** Falls Whole-Body-VLA gewünscht → `GR00T-WholeBodyControl`
      evaluieren + Datensammelplan (VR/PICO oder Sim-Rollouts).

---

## Quellen

**NVIDIA GR00T / Whole-Body Control**
- [NVIDIA Technical Blog — GR00T N1.6 Sim-to-Real Workflow](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow/)
- [NVlabs/GR00T-WholeBodyControl (GitHub)](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [NVIDIA/Isaac-GR00T (GitHub)](https://github.com/NVIDIA/Isaac-GR00T)
- [HyperAI — GR00T N1.6 Announcement](https://hyper.ai/en/headlines/72c38e5f5753d41268a6d4f8d4e805f7)

**Unitree G1 — Lokomotions-Stacks & SDK**
- [unitreerobotics/unitree_rl_gym (GitHub)](https://github.com/unitreerobotics/unitree_rl_gym)
- [unitreerobotics/unitree_rl_lab (GitHub)](https://github.com/unitreerobotics/unitree_rl_lab)
- [mintlabkorea/unitree_g1_rl_lab (GitHub)](https://github.com/mintlabkorea/unitree_g1_rl_lab)
- [Unitree G1 SDK Development Guide](https://support.unitree.com/home/en/G1_developer)
- [Unitree High-Level Sports Service Interface](https://support.unitree.com/home/en/developer/sports_services)
- [unitree_sdk2 — G1 (DeepWiki)](https://deepwiki.com/unitreerobotics/unitree_sdk2/3-g1-humanoid-robot)
- [Unitree G1 Overview (QRE Docs)](https://www.docs.quadruped.de/projects/g1/html/g1_overview.html)
- [Unitree G1 EDU Technical Specs (RoboStore)](https://robostore.com/blogs/news/unitree-g1-edu-ultimate-technical-specifications)

**Loco-Manipulation-Forschung**
- [ULC: Unified and Fine-Grained Controller (arXiv 2507.06905)](https://arxiv.org/abs/2507.06905)
- [OmniH2O (arXiv 2406.08858)](https://arxiv.org/html/2406.08858v1)
- [Mobile-TeleVision / PMP (arXiv 2412.07773)](https://arxiv.org/pdf/2412.07773)
- [WholeBodyVLA (arXiv 2512.11047)](https://arxiv.org/pdf/2512.11047)
- [Kinematics-Aware Multi-Policy RL (arXiv 2511.21169)](https://arxiv.org/pdf/2511.21169)
- [Gait-Conditioned RL with Multi-Phase Curriculum (arXiv 2505.20619)](https://arxiv.org/abs/2505.20619)
</content>
