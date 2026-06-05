# Reinforcement-Learning-Training — Recherche & Umsetzungsplan

> **Status: Recherche / Konzept.** Dieses Dokument klärt, **was nötig wäre**, um das Modell
> zusätzlich (oder anschließend) per **Reinforcement Learning (RL)** zu trainieren, wie ein
> Gesamt-Umsetzungsplan aussieht und wie eine konkrete Umsetzung **in diesem Projekt** aussehen
> könnte. Es ist **noch nicht implementiert** — es beschreibt den Weg dorthin.

Verwandte Dokumente:
- [trainingsverfahren.md](../training/trainingsverfahren.md) — das **aktuelle** Verfahren (Imitation Learning / Behavior Cloning per Flow-Matching). RL grenzt sich davon ab.
- [lauf1-auswertung.md](../ergebnisse/lauf1-auswertung.md) — der Befund (Closed-Loop-„Einfrieren", Domain-Gap), der RL überhaupt motiviert.
- [../simulation/README.md](../simulation/README.md) — die Sim-Umgebung, die für RL die Rolle des „Environments" übernimmt.

---

## 1. Warum überhaupt RL? — Motivation aus dem ersten Lauf

Das aktuelle Training ist reines **Behavior Cloning (BC)**: Das Modell ahmt ~301 teleoperierte
Demonstrationen nach (Flow-Matching-Loss auf Beobachtung→Aktion-Paaren). Es gibt **keine
Belohnung**, **keine Exploration** und **kein Feedback aus der Umgebung** (siehe
[trainingsverfahren.md §1](../training/trainingsverfahren.md)).

Die [Abschluss-Auswertung des ersten Laufs](../ergebnisse/lauf1-auswertung.md) hat genau die
**klassische BC-Schwäche** offengelegt:

- **Distribution Shift / Compounding Errors:** In der Closed-Loop-Sim „friert" die Policy ein. Das
  Modell hat die Aufgabe gelernt (Replay + Open-Loop funktionieren), kann den Regelkreis aber nicht
  schließen — sobald es in Zustände gerät, die leicht außerhalb der Demonstrationsverteilung liegen,
  akkumulieren sich Fehler.
- **Kein Erfolgssignal beim Training:** BC optimiert „Aktion nachahmen", **nicht** „Aufgabe lösen".
  Ob ein Klötzchen tatsächlich gestapelt wurde, fließt nie ins Training ein.

**Genau hier setzt RL an:** RL optimiert direkt auf **Aufgaben-Erfolg** (Reward) und exploriert
dabei selbst Zustände jenseits der Demonstrationen — es lernt also gerade die Korrekturen, die BC
fehlen. In der aktuellen Forschung hebt RL-Fine-tuning von VLA-Policies Erfolgsraten typischerweise
**massiv** an (z. B. π·RL: LIBERO 57,6 % → 97,6 %; ManiSkill-MultiTask 41,6 % → 85,7 %, siehe
[Quellen](#7-quellen)).

---

## 2. Zwei grundverschiedene „RL"-Ebenen — nicht verwechseln

In der GR00T-/Humanoid-Welt bezeichnet „RL" **zwei völlig verschiedene Dinge**. Für dieses Projekt
ist nur eine davon relevant.

| | **(A) Low-Level Whole-Body-Controller-RL** | **(B) RL-Fine-tuning der VLA-Policy** |
|---|---|---|
| Was wird gelernt? | Ein **separater** Regler für Balance/Locomotion/Kontaktdynamik (Motor-Primitive) | Der **Action-Head von GR00T selbst** (die Aufgaben-Policy) |
| Methode (NVIDIA) | RL in Isaac Lab, Teacher-Student-Distillation, Zero-Shot-Sim-to-Real | Online-RL auf der Flow-Matching-Policy (FPO / π·RL / ReinFlow) |
| Eingabe/Ausgabe | Propriozeption → Gelenk-Torques/-Targets | 4 Kameras + Zustand + Sprache → 28-dim Action-Chunk |
| Belohnung | Stabilität, Geschwindigkeit, Energie, Trackingfehler | **Aufgaben-Erfolg** (Klötzchen gestapelt) |
| Relevanz hier | **Out of Scope** — wir steuern keine Beine, der G1 sitzt/steht fix; NVIDIA liefert den WBC | **Das ist unser Ziel** — die Block-Stacking-Policy schließen |

> **Festlegung für dieses Projekt:** Wenn wir „RL-Training des Modells" sagen, meinen wir **(B)** —
> das **RL-Fine-tuning des GR00T-Action-Heads auf die Block-Stacking-Aufgabe**. (A) ist NVIDIAs
> Aufgabe und für unsere Tabletop-Manipulation nicht nötig.

---

## 3. Was ist technisch nötig? — Die Bausteine

RL braucht einen geschlossenen Regelkreis aus **Policy ↔ Environment ↔ Reward ↔ Optimierer**.
Konkret sind fünf Bausteine erforderlich; in Klammern jeweils der **Stand in diesem Projekt**.

### 3.1 Ein Environment mit `reset()` / `step()`
Eine Simulation, die Beobachtungen exakt im Modality-Format liefert (4 RGB-Kameras, 28-dim
Zustand, Sprach-Prompt), eine Aktion entgegennimmt und den nächsten Zustand zurückgibt.
→ **Bereits vorhanden:** [`Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py)
ist ein Isaac-Lab-Env mit Roboter, Tisch, Würfeln und 5 Kameras. Es hat bereits
`_get_rewards`, `_check_success`, `_get_dones` und `episode_success`.

### 3.2 Eine **Reward-Funktion**
Das Herzstück von RL. Aktuell liefert das Env nur einen **binären Success-Reward** — und der Code
sagt dazu wörtlich:
> *„Einfache binary-Success-Reward für Ablations; **nicht für RL-Training**."*
> ([`g1_dex3_blockstack_env.py:463`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py))

Ein binärer Endreward ist für RL extrem **spärlich** (sparse) — das Modell bekommt erst ganz am
Ende Feedback und lernt kaum. **Nötig ist ein „shaped reward"**, z. B. (Vorschlag):
- Distanz Greifer ↔ Zielklotz (negativ, kontinuierlich)
- Bonus für Greif-Kontakt (gripper closed on cube)
- Distanz gehaltener Klotz ↔ Stapelziel
- großer Bonus bei stabilem Stapel (`_check_success`)
- kleine Strafen für Energie / unrealistische Gelenkbewegungen (Smoothness — adressiert auch die
  beobachteten **verrauschten Finger-Aktionen**)

### 3.3 Ein RL-Algorithmus, der zur **Flow-Matching-Policy passt**
Das ist die zentrale technische Hürde. Klassisches PPO/GRPO braucht die **Log-Wahrscheinlichkeit**
`log π(a|s)` einer Aktion. Der GR00T-Action-Head ist aber eine **Flow-Matching-/Diffusions-Policy**
([`gr00t_n1d6.py`](../../app/Groot-1.6/gr00t/model/gr00t_n1d6/gr00t_n1d6.py)) — er erzeugt Aktionen
durch iteratives Entrauschen eines ODE, dessen Likelihood **nicht direkt berechenbar** ist.

Die Forschung 2025 hat dafür mehrere Verfahren entwickelt (siehe [Quellen](#7-quellen)):

| Verfahren | Kerntrick | Anmerkung |
|---|---|---|
| **π·RL (Flow-Noise / Flow-SDE)** | Lernbares Rauschnetz **oder** ODE→SDE-Umwandlung → exakte/handhabbare Likelihood; **PPO** | Direkt für **flow-basierte VLAs** (π₀-Familie) gebaut → architektonisch am nächsten an GR00T. PPO schlägt dort GRPO. |
| **FPO (Flow Policy Optimization)** | Ersetzt den Likelihood-Ratio durch einen **likelihood-freien Proxy** aus dem Flow-Matching-Loss → PPO-Style-Clipping | Kein explizites Likelihood-Modell nötig |
| **ReinFlow** | Injiziert lernbares Rauschen → diskrete Markov-Kette mit exakter Likelihood | Allgemein für Flow-Matching-Policies |
| **Flow-GRPO** | ODE→SDE für stochastische Exploration, dann GRPO | GRPO-Variante |

→ **Empfehlung für dieses Projekt: π·RL-Ansatz (Flow-SDE + PPO)**, weil er für genau diese
Modellklasse (flow-basierte VLA mit eingefrorenem VLM + trainierbarem Action-Expert) entworfen ist —
das deckt sich 1:1 mit unserem Setup (`tune_llm=False`, `tune_visual=False`, nur Projector +
Action-Head trainierbar, siehe [trainingsverfahren.md §2](../training/trainingsverfahren.md)).

### 3.4 Rollout-Infrastruktur (Policy ↔ Env-Kommunikation)
RL erzeugt Daten durch **eigenes Agieren** (Rollouts), nicht aus einem Datensatz. Policy-Inferenz
und Sim müssen pro Schritt kommunizieren.
→ **Teilweise vorhanden:** Der ZMQ-Policy-Client
[`Simulation/g1_dex3_sim/client.py`](../../Simulation/g1_dex3_sim/client.py) verbindet die Sim
bereits mit einem GR00T-Inferenz-Server. Für RL muss diese Schleife aber **Gradienten-fähig** und
**hochparallel** werden (s. u.) — der reine Eval-Client reicht nicht.

### 3.5 Rechenleistung — und ein projektspezifischer GPU-Konflikt ⚠️
RL ist **deutlich teurer** als BC: Pro Optimierungsschritt müssen **Sim + Rendering + Policy-
Inferenz (mehrfaches Entrauschen) + Backprop** laufen, und RL braucht **viele parallele
Environments** für Stichproben-Effizienz (π·RL nutzt z. B. **320 parallele Envs** auf **8× H100**).

**Der Haken in diesem Projekt** (siehe [umsetzungsnotizen.md](../simulation/umsetzungsnotizen.md)
und CLAUDE.md): Bildbasiertes RL braucht **Kamera-Rendering**, und Isaac Sims Raytracing braucht
**RT-Cores** → L40 / RTX 4090 / A6000. **A100 und H100 haben keine RT-Cores.** Genau die KISSKI-
GPUs, die fürs Training stark sind (A100/H100), können das **bildbasierte** RL-Rollout also **nicht
effizient rendern**, und die GPUs mit RT-Cores (RTX 5000 etc. auf der `jupyter`-Partition) sind
teils zu alt für Isaac Sim 4.x bzw. zu schwach fürs gleichzeitige Training.

→ **Das ist die größte praktische Hürde** und muss vor jedem Umsetzungsversuch geklärt werden
(Optionen in [§6](#6-realistische-einschätzung-für-dieses-projekt)).

---

## 4. Gesamt-Umsetzungsplan (Phasen)

Empfohlen ist **RL als Post-Training-Stufe nach BC** (nicht statt BC). Der BC-Checkpoint
(`checkpoint-175000`) ist der Startpunkt — RL verfeinert ihn. Das ist Standard: erst imitieren,
dann durch Belohnung verfeinern.

```
BC-Checkpoint (vorhanden)  ──►  RL-Fine-tuning (neu)  ──►  besserer Closed-Loop-Checkpoint
   tune nur Action-Head          tune nur Action-Head         (Ziel: kein Einfrieren mehr)
```

| Phase | Inhalt | Hauptartefakt |
|---|---|---|
| **0. Machbarkeit** | GPU-Frage klären (§3.5): Wo läuft Render **und** Training? Benchmark: wie viele parallele Envs schafft 1 GPU mit Kamera-Rendering? | Entscheidung Hardware-Pfad |
| **1. Reward-Design** | `_get_rewards` von binär → **shaped** umbauen (§3.2). Vektorisiert über alle Envs. Reward-Komponenten einzeln loggen. | neue `_get_rewards` |
| **2. Env als RL-Env** | Sicherstellen: `step`/`reset` voll vektorisiert (N Envs parallel), Beobachtung == Modality-Format, Domain-Randomization (Licht/Textur — adressiert den Domain-Gap) | gymnasium-/IsaacLab-`ManagerBasedRLEnv` |
| **3. RL-Algorithmus** | Flow-kompatiblen RL-Loop integrieren (π·RL / FPO). Entweder bestehendes Framework adaptieren (**RLinf-VLA** unterstützt PPO/GRPO für VLAs) oder FPO-Loss in den Trainer einklinken. VLM **eingefroren** lassen, nur Action-Head/Projector tunen. | RL-Trainer-Skript |
| **4. Rollout-Pipeline** | Hochparallele Policy↔Env-Schleife (Inferenz-Batching über N Envs), Replay-Buffer/Advantage-Schätzung (GAE), KL-Regularisierung gegen den BC-Checkpoint (gegen „Reward-Hacking"/Vergessen) | Rollout-Worker |
| **5. Training & Tuning** | Auf KISSKI/Cloud starten. Wichtige Hebel: Reward-Skalierung, # Denoising-Steps, Chunk-Größe, Noise-Level, KL-Coeff. Erfolgsrate statt Loss als Hauptmetrik in W&B. | RL-Checkpoint |
| **6. Eval & Vergleich** | Closed-Loop-Erfolgsrate RL vs. BC auf ungesehenen Episoden ([train-test-split.md](../training/train-test-split.md)). Friert es noch ein? | Auswertungs-Doc |

---

## 5. Wie eine konkrete Umsetzung *hier* aussehen könnte

Da viel Infrastruktur schon steht, ist der Delta-Aufwand überschaubarer als ein Greenfield-RL-Setup:

**Wiederverwendbar (bereits da):**
- Isaac-Lab-Env mit Roboter/Tisch/Würfeln/Kameras + `_check_success` (`g1_dex3_blockstack_env.py`)
- BC-Checkpoint als RL-Startpunkt (`checkpoint-175000`)
- GR00T-Inferenz + ZMQ-Client (`client.py`) — als Vorlage für die Rollout-Schleife
- Container-/SLURM-/W&B-Pipeline (KISSKI), USD-Asset, Modality-Definition

**Neu zu bauen (das eigentliche RL-Delta):**
1. **Shaped Reward** in `_get_rewards` (§3.2) — der mit Abstand wichtigste, aber gut umsetzbare
   Schritt; das Erfolgskriterium existiert schon.
2. **Vektorisierter RL-Wrapper** des Envs (N parallele Instanzen statt 1 Eval-Env).
3. **Flow-RL-Trainer**: bestehendes VLA-RL-Framework (z. B. RLinf-VLA) anbinden **oder** FPO-Loss in
   den GR00T-Trainer ([`launch_finetune.py`](../../app/Groot-1.6/gr00t/experiment/launch_finetune.py))
   integrieren — Action-Head trainierbar, VLM eingefroren (wie bisher).
4. **Neues Submit-Skript** `kisski_rl_submit.sh` analog zu
   [`kisski_submit.sh`](../../Training/kisski_submit.sh), aber auf einer **RT-Core-fähigen GPU**
   (§3.5).

**Pragmatischer erster Schritt (Minimal-Pilot):**
> Reward-Shaping + RL auf **State-only** (ohne Kamera-Rendering) als Machbarkeits-Pilot. Damit
> entfällt zunächst der RT-Core-Konflikt (§3.5), die Sim läuft schnell auf A100/H100, und man testet
> isoliert, ob Reward + Flow-RL-Loop überhaupt sauber lernen. Erst danach Kameras dazuschalten —
> denn die volle VLA-Policy braucht Vision.
>
> ⚠️ Einschränkung: GR00T **ist** bildkonditioniert. Ein rein zustandsbasierter Pilot validiert nur
> die RL-Mechanik (Reward, PPO/Flow-Loss, Stabilität), **nicht** das vision-basierte Schließen des
> Regelkreises — das eigentliche Ziel braucht zwingend Rendering.

---

## 6. Realistische Einschätzung für dieses Projekt

**Aufwand:** RL-Fine-tuning ist **deutlich** anspruchsvoller als das bisherige BC — sowohl im
Engineering (Reward-Design, vektorisierte Rollouts, Flow-kompatibler RL-Loss) als auch im
Tuning (RL ist notorisch instabil und hyperparameter-sensitiv). Es ist eher ein eigenes
Teilprojekt als ein Konfig-Flag.

**Der kritische Pfad** ist **nicht** der Algorithmus (dafür gibt es 2025 fertige Verfahren), sondern
die **GPU-/Rendering-Frage (§3.5)**: bildbasiertes RL braucht gleichzeitig RT-Core-Rendering **und**
Trainings-FLOPs. Mögliche Auflösungen:
- **Cloud (vast.ai):** L40 / A6000 (Ampere + RT-Cores, genug VRAM) — passt zur bestehenden
  vast.ai-Sim-Pipeline; vermutlich der **realistischste Weg**.
- **KISSKI:** prüfen, ob eine Partition Ada/L40-GPUs hat; sonst eignet sich KISSKI eher für die
  **State-only-Variante** (§5) oder reines BC.
- **Entkoppeln:** Render-Rollouts auf RT-GPU(s), Optimierung auf A100/H100 — maximaler
  Engineering-Aufwand (verteilte Rollout-/Lerner-Architektur).

**Erwarteter Nutzen:** Falls der Domain-Gap ([lauf1-auswertung.md](../ergebnisse/lauf1-auswertung.md))
die Hauptursache des Einfrierens ist, ist RL **das prinzipiell richtige Werkzeug**: Es trainiert die
Policy **in genau der Sim**, in der sie auch evaluiert wird → die Trainings- und Test-Verteilung
fallen zusammen, der Distribution-Shift verschwindet, und der Reward erzwingt aktives Handeln statt
Einfrieren. Genau dieser Effekt (BC-Plateau → RL-Sprung) ist in der Literatur breit belegt.

**Günstigere Alternativen vor RL** (falls der Aufwand zu hoch ist):
- **Vision-Encoder mit-tunen** (`tune_visual=True`) — billiger erster Hebel gegen den Domain-Gap.
- **Sim-Rendering an die echten Trainingsbilder angleichen** oder den Datensatz mit Sim-Bildern
  augmentieren — adressiert die Domain-Gap-Ursache direkt, ohne RL.
- **DAgger / Interactive Imitation** — schließt die Distribution-Shift-Lücke teilweise, ohne den
  vollen RL-Apparat.

---

## 7. Konkrete ToDo-Liste zur Vorbereitung

Ziel dieser Liste: die **spätere Umsetzung kurz und leicht halten**. Alle Punkte sind so geschnitten,
dass sie **vorbereitend** (Recherche, Design, Entscheidungen, kleine isolierte Code-Gerüste) erledigt
werden können, **ohne** schon den vollen RL-Lauf zu starten. Reihenfolge = empfohlener Pfad. Jede
Gruppe hat ein **Definition-of-Done (DoD)**, an dem man erkennt, dass der Vorbereitungsschritt
abgeschlossen ist.

Die gute Nachricht vorweg: Das Sim-Env ist **bereits eine `isaaclab.envs.DirectRLEnv`** mit
vektorisierten `_get_rewards` / `_get_dones` / `_check_success` / `_pre_physics_step`
([`g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py)). Es fehlen vor
allem **(a) ein echter Reward**, **(b) echte Parallelität (`num_envs>1`)** und **(c) der RL-Loss-Loop**
— der Rest ist Anpassung.

### Gruppe 0 — Machbarkeit & Entscheidungen *(blockiert alles Weitere)*

- [ ] **GPU-/Render-Pfad festlegen** (§3.5, §6): Wo laufen Render-Rollouts (RT-Cores) und wo das
  Training? Drei Optionen bewerten: vast.ai L40/A6000 · KISSKI Ada/L40-Partition prüfen · Render/Lerner
  entkoppeln. → **Eine** Option auswählen und begründen.
- [ ] **Render-Durchsatz-Benchmark** (Vorbereitung, kein RL): Auf der gewählten GPU messen, wie viele
  parallele Envs **mit** 4-Kamera-Rendering bei akzeptabler FPS laufen (Anhaltspunkt Literatur: 320
  Envs auf 8× H100, aber **ohne** RT-Cores). Ergebnis bestimmt realistische `num_envs`.
- [ ] **Algorithmus & Framework fixieren**: π·RL (Flow-SDE+PPO) vs. FPO vs. ReinFlow; eigenes
  RL-Framework (z. B. RLinf-VLA) anbinden **oder** Loss in GR00T-Trainer integrieren. → Entscheidung +
  1 Absatz Begründung in dieses Dokument.
- [ ] **Scope bestätigen**: Nur Action-Head/Projector tunen, VLM eingefroren (wie BC). Klären, ob ein
  **State-only-Pilot** (§5) als Zwischenschritt gefahren wird.

> **DoD:** Hardware-Pfad, `num_envs`-Richtwert, Algorithmus und Pilot-Frage sind schriftlich
> entschieden. Ab hier ist klar, *worauf* implementiert wird.

### Gruppe 1 — Reward-Design *(der wichtigste Vorbereitungsschritt, rein offline machbar)*

- [ ] **Shaped-Reward spezifizieren** (zunächst nur als Design-Tabelle in diesem Dokument, kein Code):
  Komponenten + Gewichte festlegen, z. B.
  - Greifer↔Zielklotz-Distanz (kontinuierlich, negativ)
  - Greif-Kontakt-Bonus
  - gehaltener Klotz ↔ Stapelziel-Distanz
  - Stapel-Erfolg-Bonus (nutzt bestehendes `_check_success`)
  - Smoothness-/Energie-Strafe (adressiert die verrauschten **Finger-Aktionen** aus dem ersten Lauf)
- [ ] **Benötigte Größen im Env identifizieren**: Welche Tensoren liefern Greifer-Pose, Kontakt,
  Klotz-Posen? (Block-Posen existieren bereits in `_check_success` via `block.data.root_pos_w`.) Lücken
  notieren (z. B. Kontakt-Sensor nötig?).
- [ ] **`cfg`-Felder planen**: neue Reward-Gewichte/Toleranzen analog zu den bestehenden
  `stack_xy_tol` / `stack_height_min` / `stack_vel_max` in `G1Dex3BlockstackEnvCfg`.
- [ ] **(optional, klein)** `_get_rewards` als **separate, getestete Funktion** vorbereiten, ohne den
  Eval-Pfad zu verändern (Stub bleibt nutzbar) — z. B. hinter einem `cfg.reward_mode`-Schalter.

> **DoD:** Eine vollständige Reward-Spezifikation (Komponenten, Gewichte, benötigte Env-Größen) liegt
> vor. Die spätere Implementierung ist dann „nur noch Abtippen".

### Gruppe 2 — Env RL-tauglich machen *(Anpassung, kein Neubau)*

- [ ] **Parallelität**: `num_envs` von hartkodiert **1** (`G1Dex3BlockstackSceneCfg(num_envs=1, …)`,
  Z. 224/241) auf einen Parameter heben; prüfen, dass Reset/Spawn/Domain-Randomization pro Env sauber
  vektorisiert sind.
- [ ] **Observation-Pfad verallgemeinern**: `get_obs_for_policy` liefert aktuell nur **Env-Index 0**
  (`val[0].cpu().numpy()`, Z. 537). Für RL einen **batched** Pfad ergänzen, der alle `num_envs`
  zurückgibt (der Single-Env-Eval-Pfad bleibt unangetastet).
- [ ] **Domain-Randomization** als Stichpunkte planen (Licht, Texturen, Klotz-Startposen) — adressiert
  direkt den **Domain-Gap** und damit das Closed-Loop-Einfrieren.
- [ ] **Aktions-Konvention dokumentieren** für den RL-Loop: `_pre_physics_step` erwartet **absolute**
  28-dim Gelenk-Targets (Z. 446–469). Klären, an welcher Stelle der RL-Sampler die Flow-Aktion in
  dieses Format bringt (im BC-Eval macht das der Server via `decode_action`).

> **DoD:** Es ist klar (und in kleinen, getrennten Commits vorbereitet), wie das Env N Envs parallel
> fährt und batched Observations liefert — ohne den bestehenden Eval-Pfad zu brechen.

### Gruppe 3 — RL-Loop & Anbindung *(das eigentliche Delta, hier nur Gerüst/Recherche)*

- [ ] **Flow-RL-Loss isoliert nachvollziehen**: Referenz-Implementierung (π·RL / FPO / ReinFlow) lesen
  und auf einem Mini-Toy-Beispiel zum Laufen bringen — getrennt vom Roboter, um die Mechanik zu
  verstehen.
- [ ] **Anbindungspunkt im GR00T-Code bestimmen**: Wo greift der RL-Loss in
  [`launch_finetune.py`](../../app/Groot-1.6/gr00t/experiment/launch_finetune.py) /
  [`gr00t_n1d6.py`](../../app/Groot-1.6/gr00t/model/gr00t_n1d6/gr00t_n1d6.py) ein? Stelle markieren, an
  der das Velocity-Field/Denoising für den Likelihood-Proxy abgegriffen wird.
- [ ] **Rollout-Schleife skizzieren**: batched Policy-Inferenz über N Envs (Vorlage:
  [`client.py:build_obs`](../../Simulation/g1_dex3_sim/client.py) + `_get_observations`), Advantage
  (GAE), **KL-Regularisierung gegen den BC-Checkpoint** (gegen Reward-Hacking/Vergessen).
- [ ] **W&B-Metriken definieren**: Hauptmetrik **Erfolgsrate** (nicht Loss), plus
  Reward-Komponenten einzeln — analog zur bestehenden W&B-Pipeline.

> **DoD:** Loss-Mechanik verstanden (Toy lauffähig), Anbindungsstelle im GR00T-Code lokalisiert,
> Rollout-Architektur als Diagramm/Pseudocode dokumentiert.

### Gruppe 4 — Lauf-Infrastruktur *(klein, am Ende der Vorbereitung)*

- [ ] **`kisski_rl_submit.sh`** (bzw. vast.ai-Pendant) als **Kopie** von
  [`kisski_submit.sh`](../../Training/kisski_submit.sh) entwerfen — Unterschiede: RT-Core-GPU,
  `--resume`-from-BC-Checkpoint, RL-Env-Vars (`num_envs`, Reward-Gewichte, KL-Coeff).
- [ ] **Env-Var-Referenz** für RL ergänzen ([env-vars.md](../training/env-vars.md)).
- [ ] **Eval-Vergleich vorbereiten**: bestehende Closed-Loop-Eval auf den
  [Train/Test-Split](../training/train-test-split.md) anwenden, damit RL- vs. BC-Erfolgsrate sauber vergleichbar
  ist (Baseline-Zahl des BC-Checkpoints **vorab** messen).

> **DoD:** Submit-Skript-Entwurf, RL-Env-Vars und eine **BC-Baseline-Erfolgsrate** liegen vor — der
> RL-Lauf ist dann „nur noch starten".

### Minimal-Pfad (wenn die Zeit knapp ist)

Reihenfolge, die mit **kleinstem Aufwand** zum ersten echten Lernsignal führt:

1. **G0** GPU-Pfad entscheiden → 2. **G1** Reward spezifizieren → 3. **G2** `num_envs>1` + batched Obs
→ 4. **G3** Flow-RL-Loss anbinden → 5. **G4** Submit-Skript + BC-Baseline → **Pilot starten**.

> Optional davor: **State-only-Pilot** (§5) — überspringt zunächst Gruppe-0-Render-Frage und Kameras,
> validiert nur Reward + RL-Mechanik auf A100/H100. Schneller erster Erfolg, aber **nicht** das
> vision-basierte Endziel.

---

## 8. Quellen

**NVIDIA / GR00T:**
- [Building Generalist Humanoid Capabilities with NVIDIA Isaac GR00T N1.6 Using a Sim-to-Real Workflow (NVIDIA Technical Blog)](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow/) — Zwei-Ebenen-Architektur, Whole-Body-RL in Isaac Lab.
- [GR00T N1.6 (NVIDIA GEAR Research)](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
- [Isaac Lab — GPU-beschleunigtes Robot-Learning-Framework](https://developer.nvidia.com/isaac/lab)

**RL-Fine-tuning von Flow-/Diffusions-VLAs (2025):**
- [πRL: Online RL Fine-tuning for Flow-based VLAs (arXiv 2510.25889)](https://arxiv.org/html/2510.25889v1) — Flow-Noise/Flow-SDE, PPO > GRPO, 320 parallele Envs, 8× H100.
- [Reinforcement Fine-Tuning of Flow-Matching Policies for VLA Models — FPO (arXiv 2510.09976)](https://arxiv.org/html/2510.09976v1) — likelihood-freier PPO-Proxy.
- [ReinFlow: Fine-tuning Flow Matching Policy with Online RL (arXiv 2505.22094)](https://arxiv.org/pdf/2505.22094)
- [Reinforcement Learning for Flow-Matching Policies (arXiv 2507.15073)](https://arxiv.org/pdf/2507.15073)
- [REFINE-DP: Diffusion Policy Fine-tuning for Humanoid Loco-manipulation via RL (arXiv 2603.13707)](https://arxiv.org/pdf/2603.13707)
- [What Can RL Bring to VLA Generalization? An Empirical Study (arXiv 2505.19789)](https://arxiv.org/html/2505.19789v4)
- [Awesome-RL-VLA — Survey & Framework-Übersicht (GitHub)](https://github.com/Denghaoyuan123/Awesome-RL-VLA) — u. a. RLinf-VLA (PPO/GRPO für VLAs).

**Projektinterne Referenzen:**
- [trainingsverfahren.md](../training/trainingsverfahren.md), [lauf1-auswertung.md](../ergebnisse/lauf1-auswertung.md)
- [`Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py`](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) (`_get_rewards`, `_check_success`)
- [`app/Groot-1.6/gr00t/model/gr00t_n1d6/gr00t_n1d6.py`](../../app/Groot-1.6/gr00t/model/gr00t_n1d6/gr00t_n1d6.py) (Flow-Matching-Action-Head)
