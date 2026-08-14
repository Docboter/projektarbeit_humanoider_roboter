# Auswertung — erster vollständiger Trainingsdurchlauf (`g1_dex3_blockstacking_v1`)

**Erstellt:** 2026-06-03 · **Run-ID:** `i6n1t613` (W&B-Projekt `gr00t-g1-dex3`, Entity
`projektarbeit_humanoider_roboter`) · **Modell:** GR00T N1.6 (3,29 Mrd. Parameter),
Partial-Finetune auf Unitree G1 + DEX3, Task „stack the blocks".

> Dies ist die **abschließende** Auswertung des ersten kompletten Laufs (175.000 Steps,
> abgeschlossen). Die mitlaufende Momentaufnahme während des Trainings steht in
> [`wandb-run-auswertung.md`](wandb-run-auswertung.md); dieses Dokument ergänzt sie um
> die **Verhaltens-Evaluation** (Sim-Closed-Loop, Replay, Open-Loop) und das Gesamtfazit.

---

## 1. Kurzfazit (TL;DR)

- **Training selbst: gesund und vollständig auskonvergiert.** Loss 0,10 → ~0,008, sauberer
  Cosine-LR-Abfall auf ~0, keine NaN. Aus reiner Optimierungssicht ist nichts kaputt.
- **Aufgabe (Stapeln) im Closed-Loop: nicht gelöst.** Der Roboter fährt grob an und
  **friert dann ein**; kein Greifen, kein Stapel.
- **Ursachenkette über drei Diagnosen eindeutig eingegrenzt:** Das Modell ist
  **grundsätzlich fähig** (sagt bei echten Beobachtungen gute Arm-Actions vorher), die
  Sim/Config **führt Aktionen korrekt aus** — der Closed-Loop scheitert dennoch. Der
  einzige Unterschied zwischen „funktioniert" und „friert ein" sind die **Beobachtungen**.
  → Wahrscheinliche Hauptursache: **visueller Domain-Gap** (reale Trainingsbilder vs.
  synthetische Isaac-Sim-Renderings) bei **eingefrorenem Vision-Encoder**.
- **Sekundär:** Die **Finger-Actions (DEX3)** werden auch bei perfekten Beobachtungen
  verrauscht vorhergesagt → begrenzt die Greifzuverlässigkeit.
- **Strukturelle Lücke:** Es gab **keine Validierungs-/Eval-Metrik** im Lauf → die
  Checkpoint-Auswahl war blind (am Ende einfach Step 175.000).

---

## 2. Lauf-Eckdaten (abgeschlossen)

| Feld | Wert |
|---|---|
| Run-ID / Name | `i6n1t613` / `g1_dex3_blockstacking_v1` |
| State | `finished` |
| Node / GPUs | `ggpu177` (KISSKI), `num_gpus = 1` (A100) |
| Steps | **175.000** (`num_train_epochs = 3` ist nur der Konfig-Wert und wird von `max_steps` überschrieben — effektiv ≈ 5 Datendurchläufe: 175.000 × 8 / 281.196) |
| Laufzeit | ~77.200 s (**~21,4 h**), ~2,27 Steps/s |
| `global_batch_size` / per-device | 8 / 8, `gradient_accumulation = 1` |
| Action-Horizon / Denoising-Steps | 16 / 4 |
| Trainierbare Teile | Projector + Diffusion-Head + oberste 4 LLM-Layer + VLLN |
| Eingefroren | LLM-Backbone **und Vision-Encoder** (`tune_visual = false`) |
| Action-Repräsentation | Arme `RELATIVE`, Hände (`*_dex3`) `ABSOLUTE`, `use_relative_action = true` |
| Datensatz | `unitreerobotics/G1_Dex3_BlockStacking_Dataset` (real, Teleop), 1 Mischung |

---

## 3. Trainingsdynamik — gesund ✅

| Metrik | Verlauf | Bewertung |
|---|---|---|
| `train/loss` | 1,37 (Start, ungeglättet; vgl. §8.1) → ~0,03 (Step ~25k) → ~0,01 (Step ~100k) → **~0,008 final (geglättet)**; ungeglätteter W&B-Endwert 0,1024, avg `train_loss` 0,030 | sauberer Abfall, kein NaN |
| `train/grad_norm` | früh ~0,3–0,8 → stabil ~0,1; vereinzelt kleine Peaks | gesund, kein Divergieren |
| `train/learning_rate` | Cosine, Peak 1e-4 (nach Warmup 5 %) → Ende ~0 (8,9e-15) | korrekt für 175k Steps |

**Wichtig:** Niedriger Flow-Matching-Loss ≠ gute Policy. Die Kurve sagt nur, dass das
Modell die **Trainingsverteilung** gut anpasst — kein Generalisierungs- und kein
Aufgaben-Erfolgssignal. Der Loss plateaut faktisch ab ~Step 100–120k; die letzten ~55k
Steps brachten kaum noch etwas. **Weitertrainieren mit gleichem Rezept bringt nichts.**

---

## 4. Verhaltens-Evaluation — drei Diagnosen

Die eigentliche Aussagekraft kommt nicht aus dem Loss, sondern aus drei komplementären
Tests am finalen Checkpoint (`checkpoint-175000`).

### 4.1 Closed-Loop-Sim-Eval (Modell steuert die Sim)
- Eval-Videos (`episode_0001/0002.mp4`, je 20 s; sowie eine 3-min-Version): Der Roboter
  fährt in den ersten ~15 s grob an die Würfel heran, **danach nahezu Stillstand**.
- Bewegungsenergie (Bild-zu-Bild-Differenz) fällt nach dem Start auf ein konstant
  niedriges Niveau (~0,36 von 255) und bleibt dort über die vollen 3 min flach → der
  Roboter **friert ein**, statt zu greifen/stapeln.
- Mehr Zeit hilft nicht — es ist **kein „zu kurz"-Problem**.

### 4.2 Open-Loop-Dataset-Replay (echte Aktionen, KEIN Modell)
Skript: [`Simulation/kisski_replay_submit.sh`](../../Simulation/kisski_replay_submit.sh)
bzw. `entrypoint_replay.sh` (vast.ai). Spielt die aufgezeichneten Dataset-Aktionen direkt
in die Isaac-Lab-Env.
- Ergebnis (`replay_episode0.mp4`, 39 s): Beide Arme fahren nach vorne, **senken sich auf
  Tischhöhe, die Finger schließen sich greif-artig** (z. B. linke Hand um den grünen
  Würfel). Bewegungsenergie hoch am Start (0,72) und sauber abklingend.
- **Interpretation:** Sim/Config (Kinematik, Action-Decoding relative Arme + absolute
  Hände, Tischhöhe, Reachability, Asset) führt die korrekten Aktionen **korrekt aus**.
  → Der Closed-Loop-Fehler liegt **nicht** an der Sim-Config.

### 4.3 Open-Loop-Modell-Eval (echte Dataset-Beobachtungen → predicted vs. GT-Actions)
Tool: `gr00t/eval/open_loop_eval.py` via
[`Training/kisski_open_loop_eval.sh`](../../Training/kisski_open_loop_eval.sh). Drei
Trajektorien (`checkpoint-175000_traj_0..2.jpeg`). Action-Dims:
`left_arm` 0–6, `right_arm` 7–13, `left_dex3` 14–20, `right_dex3` 21–27.
- **Arm-Dims (0–13): gut vorhergesagt** — Prediction folgt der GT-Kurve eng, vor allem
  die großen glatten Bewegungen tracken nahezu perfekt; die Inferenz-Punkte (Start jedes
  16er-Chunks) sitzen auf der GT.
- **Finger-Dims (14–27, DEX3): deutlich schlechter** — hochfrequentes Rauschen und
  vereinzelte große Spikes, die die GT verfehlen. Welche Hand am schlimmsten ist,
  variiert pro Episode, aber die `*_dex3`-Dims (die `ABSOLUTE`-Aktionen) sind konsistent
  der Schwachpunkt.

> Offen / nachzutragen: die quantitativen **MSE/MAE-Werte** aus dem Job-Log (pro Traj +
> Durchschnitt). Sie würden „Arme gut / Hände schlecht" zahlenmäßig untermauern.

---

## 5. Synthese & Diagnose

| Test | Beobachtungen des Modells | Ergebnis |
|---|---|---|
| Closed-Loop | synthetische Isaac-Sim-Renderings | Anfahren, dann **Einfrieren** |
| Replay | — (aufgezeichnete Actions, kein Modell) | Greifbewegung **korrekt ausgeführt** |
| Open-Loop | echte Roboterkamera-Bilder (Dataset) | **gute Arm-Actions** vorhergesagt |

Zusammengesetzt: Das **Modell ist fähig** (Open-Loop beweist gute Arm-Trajektorien bei
echten Bildern), die **Sim ist korrekt** (Replay beweist korrekte Ausführung) — und
trotzdem **friert der Closed-Loop ein**. Der einzige verbleibende Unterschied sind die
**Beobachtungen**.

**Hauptursache (Hypothese, hoch plausibel): visueller Domain-Gap.** Das Modell wurde auf
**echten Teleop-Kamerabildern** trainiert; im Closed-Loop bekommt es **synthetische
Isaac-Sim-Renderings** der Policy-Kameras (`cam_left_high`, `cam_right_high`,
`cam_left_wrist`, `cam_right_wrist`). Da der **Vision-Encoder eingefroren** ist, kann er
diese stark Out-of-Distribution-Bilder nicht sinnvoll kodieren → die Action-Head bekommt
unbrauchbare Features → nahezu „nichts tun" (= das beobachtete Einfrieren). Der Replay
umgeht das Modell (funktioniert deshalb), der Open-Loop nutzt echte Bilder (funktioniert
deshalb).

**Sekundärursache: Finger-Action-Qualität.** Selbst bei In-Distribution-Beobachtungen
sind die DEX3-Finger-Dims verrauscht → die Greifzuverlässigkeit ist auch ohne Domain-Gap
gedeckelt.

**Strukturelle Lücke: keine Eval-Metrik.** `eval_strategy = "no"`,
`enable_open_loop_eval = false` → kein Val-Signal, keine fundierte Checkpoint-Auswahl,
Overfitting prinzipiell unsichtbar (Details in [`wandb-run-auswertung.md`](wandb-run-auswertung.md), Abschnitt 3).

### Nachgemessen 2026-08-13: Step 1000 gegen Step 175000

Mit [`checkpoint_sweep.py`](../../Training/scripts/checkpoint_sweep.py) nachträglich erhoben
(Job 15271599, 3 Episoden × 300 Steps, Action-Horizon 16):

| Checkpoint | Action-MSE | Action-MAE |
|---|---|---|
| 1000 | 0,0703 | 0,1652 |
| 175000 | **0,0018** | **0,0203** |

**Faktor 39.** Der Lauf hat zwischen früh und spät massiv gelernt — die 174.000 Schritte
waren nicht umsonst. Nebenbefund: bei Step 1000 lag Episode 0 mit 0,104 beim Doppelten der
übrigen (0,050–0,057), bei Step 175000 ist die Spreizung verschwunden (0,0011–0,0025). Der
Ausreißer war eine Eigenschaft des untrainierten Modells, **nicht** der Episode — die
Replay-Diagnostik auf Episode 0 (`replay_episode0.npz`) ist davon unbelastet.

> ⚠️ **Gemessen auf Trainingsdaten.** Lauf 1 lief ohne Split, es gab keine ungesehenen
> Episoden. Bei 301 Episoden und 175k Schritten (Batch 8) hat das Modell den Datensatz
> vielfach gesehen; eine MSE nahe null ist dort das erwartete Bild und **belegt keine
> Generalisierung** — Memorierung ist damit nicht auszuschließen. Auch bleibt offen, ob
> Step 50000 genauso gut gewesen wäre; gemessen sind zwei Punkte, keine Kurve.
>
> **Nachtrag 2026-08-14 — die Kurve gibt es inzwischen, für Lauf 3.** Der erste Lauf mit
> Split ([`lauf3-vision-split-auswertung.md`](lauf3-vision-split-auswertung.md)) zeigt auf
> zurückgehaltenen Episoden eine **U-Kurve**: Minimum bei Step 30.000, danach steigt die
> Validierungs-MSE wieder um 25 %, während der Trainingsloss weiter fällt. Übertragen auf
> Lauf 1 heißt das: die Wahl von Step 175.000 war nicht nur unbelegt, sondern nach heutigem
> Kenntnisstand **wahrscheinlich falsch** — ein früherer Checkpoint dürfte besser
> generalisiert haben. Nachmessen ließe sich das nur mit einem Sweep über die Lauf-1-Checkpoints,
> und auch dann nur auf Trainingsdaten, weil Lauf 1 keine zurückgehaltenen Episoden hat.

**Was die Zahl trotzdem trägt:** Auf **echten** Bildern sagt die Policy die
Demonstrations-Aktionen mit MSE 0,0018 fast exakt vorher — in der **Sim** kommandiert
dieselbe Policy 19 % der Fingerspanne und erreicht 0/20. Das ist die schärfste
Formulierung des Domain-Gaps, die das Projekt hat, unabhängig gemessen und deckungsgleich
mit dem `span`-Gate aus Lauf 32 (Verhältnis 1,00 auf Realbildern, siehe
[rl-anleitung.md](../weiterfuehrend/rl-anleitung.md)). „Griff nie gelernt" ist damit ein
zweites Mal ausgeschlossen, und die Trainingspipeline ist entlastet.

---

## 6. Handlungsempfehlungen

### 6.1 Domain-Gap zuerst hart bestätigen (billig, entscheidend)
Ein **Policy-Kamerabild aus der Sim** (`cam_*_high/wrist`, **nicht** die 3rd-Person-
Szenenkamera) direkt neben ein **Dataset-Kamerabild** derselben Ansicht legen. Sehen die
sichtbar völlig anders aus (echtes Labor-RGB vs. weißer Roboter auf grauem Gitter), ist
die Diagnose belegt.

### 6.2 Wenn bestätigt — den Closed-Loop-Sim-Eval realistisch einordnen
Ein rein auf Realdaten trainiertes Modell kann mit der jetzigen Sim-Optik nicht ohne
visuelle Angleichung funktionieren. Optionen:
1. Evaluation auf **echten / held-out realen** Episoden statt Sim (Open-Loop zeigt bereits
   brauchbare Arm-Prädiktion).
2. **Visuellen Gap schließen:** Sim-Rendering an die echten Kameras angleichen (Texturen,
   Beleuchtung, Roboter-Appearance, Kamera-Intrinsics) und/oder **Domain-Randomization**.
3. **Vision-Encoder mitfinetunen** (`tune_visual = true`) bzw. auf sim-ähnlichen Bildern
   nachtrainieren.
4. **Nicht** „länger trainieren" — der Lauf ist auskonvergiert.

### 6.3 Finger-Qualität separat verbessern
Normalisierungs-Statistik der `*_dex3`-`ABSOLUTE`-Dims prüfen; ggf. mehr/sauberere
Greif-Demos oder Gewichtung der Finger-Dims.

### 6.4 Für den nächsten Lauf
- **Eval einschalten:** Val-MSE über die Zeit, fundierte Checkpoint-Auswahl statt „letzter".
  > **Korrektur 2026-08-13:** Hier standen `enable_open_loop_eval = true`,
  > `eval_strategy = "steps"` und `eval_set_split_ratio = 0.1`. Alle drei sind im Fork
  > wirkungslos — die ersten beiden werden nirgends gelesen, `eval_strategy = "steps"` läuft
  > sogar in eine `assert`-Sperre. Umgesetzt ist die Auswertung **nach** dem Lauf:
  > [`checkpoint_sweep.py`](../../Training/scripts/checkpoint_sweep.py), Begründung in
  > [train-test-split.md](../training/train-test-split.md) (Hinweis 1).
- **Batch-Size hochziehen** auf der A100 (siehe [`multi-gpu.md`](../training/multi-gpu.md)) — bei
  4× A100 / `global_batch_size = 32` reichen ~40–50k Steps für denselben Konvergenzstand
  (Loss plateaut ohnehin ab ~100k bei bs=8).

---

## 8. Update — Sim-Kalibrierung und Diagnose-Bestätigung (2026-06-04)

### 8.1 W&B-Analyse (via MCP-Server)

Die W&B-Trainingskurven wurden erstmals maschinell ausgewertet. Ergebnis bestätigt die
visuelle Inspektion aus §2:

| Metrik | Wert |
|---|---|
| Run-ID | `i6n1t613` |
| Loss (Start) | 1,3716 |
| Loss (Ende) | **0,1024** |
| Loss (Minimum) | 0,0661 |
| Tail-Mean (letzte Phase) | **0,0921** |
| W&B-Diagnose | **„converged"** / „plateaued" |
| Empfehlung des Diagnose-Tools | „Training appears converged — run can likely be stopped" |

**Fazit:** Das Modell hat gelernt, was die Demonstrations-Daten zeigen. Die Closed-Loop-Versagen
liegen **nicht** am Training.

### 8.2 Sim-Kalibrierung: Replay-Durchbruch

Nach einer systematischen Kalibrierungs-Session (Details in
[`umsetzungsnotizen.md §14`](../simulation/umsetzungsnotizen.md)) wurde das Replay-Ergebnis
von `max_cube_lift = 1,0 cm` (kein Greifen) auf **2,8 cm** (Greifen bestätigt) verbessert.

> ⚠️ **Einschränkung (2026-08-08) — inzwischen aufgelöst:** Dieser Befund gilt für Isaac Sim 4.x.
> Nach der Isaac-Sim-6.0-Migration hob derselbe Replay-Testtyp zunächst keinen Würfel mehr an
> (0,0 cm).
> **Aufgelöst mit Lauf 29 (2026-08-12):** Ursache war der Referenzpunkt der Messung — sie lief
> am distalen Gelenk statt an den Fingerspitzen. Korrigiert hebt die Hand einen Würfel **7,9 cm**;
> die Greif-Physik war nicht defekt. Siehe
> [`rl-anleitung.md`](../weiterfuehrend/rl-anleitung.md), Läufe 25–29.

Die wesentlichen Fixes:

1. **Sign-Convention-Fix** (wichtigster Fix): `middle_0` / `index_0` beider Hände hatten
   invertierte Achsen im USD → Proximal-Gelenke bewegten sich beim Greifkommando in die
   **falsche Richtung** (öffnen statt schließen). Fix: Negierung der Actions und Observations
   für Policy-Indices `[17, 19, 24, 26]`.

2. **Finger-Aktuatoren**: stiffness 20→60, effort_limit 5→20 N·m — Finger konnten 50 g
   Würfel gegen Schwerkraft nicht halten.

3. **Würfel-Reibung**: static 0,5→3,0 / dynamic 0,5→2,5 — Würfel glitt trotz Griff heraus.

4. **Würfelhöhe**: `block_z_surface` 0,895→0,915 — Würfel-Oberkante jetzt bei z=0,940,
   entspricht dem tiefsten gemessenen Handpunkt (links z=0,937, rechts z=0,944).

### 8.3 Closed-Loop-Diagnose bestätigt

Das Closed-Loop-Verhalten mit `checkpoint-175000` auf vast.ai (L40):

> **„Hände liegen auf dem Tisch, Roboter führt kleine ungerichtete Bewegungen aus,
> keine Greifaktion."**

Dies ist konsistent mit der in §5 formulierten Hauptursache (**visueller Domain Gap**).
Zusätzlich blockierte die Tischkollision die Arme (Tisch testweise auf 0,89 m angehoben
→ Arme steckten fest). Tischhöhe auf 0,87 m zurückgesetzt.

Die Diagnose ist damit **dreifach bestätigt**:
- Replay (kein Modell): Arme folgen korrekt, Greif-Physik funktioniert ✅
- W&B: Training konvergiert ✅
- Closed-Loop: Versagen nur wenn Modell + Sim-Bilder zusammen ❌ → Domain Gap

### 8.4 Bewertung der Verbesserungsoptionen

#### Option A: Vision Encoder mittrainieren (`tune_visual = true`)

> ✅ **Inzwischen empirisch getestet — und bestätigt.** Der zweite Lauf
> (`g1_dex3_blockstacking_vision_v1`, Run `ajgoskon`) hat genau das gemacht. Ergebnis:
> Training gleich gesund (Loss ~0,009), Closed-Loop aber **schlechter** — die Policy
> kollabierte auf reines Arm-Zurückziehen. Volle Auswertung:
> [`lauf2-vision-auswertung.md`](lauf2-vision-auswertung.md). Die folgende Prognose ist
> damit belegt.

**Nicht empfohlen** für dieses Projekt:

- Mit 301 Demonstrationen droht **Catastrophic Forgetting** — der Encoder verliert seine
  realen Repräsentationen, die er auf Milliarden echter Bilder gelernt hat.
- Sim-Bilder und echte Bilder unterscheiden sich so fundamental (Beleuchtung, Schatten,
  Texturen, Tiefenschärfe), dass der Encoder sich nicht sinnvoll auf Sim-Bilder adaptieren
  kann — er würde auf realen Bildern schlechter werden, ohne im Sim gut zu werden.
- Benötigt ~3–4× mehr VRAM (volle 3B-Parameter trainierbar) und mehr Demonstrations-Daten.
- Würde das Modell für eine spätere Verwendung am echten Roboter **schlechter** machen.

#### Option B: Reinforcement Learning (RL)

**Machbar, mittlerer bis hoher Aufwand** (~2–3 Wochen Engineering):

| Aspekt | Bewertung |
|---|---|
| Sim-Env (reset/step) | ✅ bereits gebaut und validiert |
| Reward-Funktion | ✅ `_check_success()` als Basis; Shaped Reward ~1–2 Tage Arbeit |
| BC-Checkpoint als Startpunkt | ✅ schnellere RL-Konvergenz |
| RL-Algorithmus für Flow-Matching | ⚠️ Standard-PPO passt nicht direkt; REINFORCE auf Trajektorien-Ebene ist der einfachste Einstieg |
| GPU-Anforderung | ⚠️ RT-Cores (Isaac Sim) + VRAM (3B Modell) gleichzeitig → vast.ai L40/A6000, nicht KISSKI |

**Realistischster Ansatz:** Trajectory-Level REINFORCE — Episode läuft durch, am Ende wird
der kumulierte Reward als Policy-Gradient-Signal verwendet. Keine Differenzierung durch den
Denoising-Prozess nötig, daher mit der aktuellen GR00T-Architektur umsetzbar.
Details: [`reinforcement-learning-plan.md`](../weiterfuehrend/reinforcement-learning-plan.md).

#### Option C: Mehr Demonstrations-Daten + erneutes BC-Training

Wenn der echte Roboter schlechte Ergebnisse zeigt (kein Domain Gap, aber schlechtes Verhalten):
mehr und vielfältigere Teleop-Episoden aufnehmen, insbesondere mit expliziten Greif- und
Platzier-Übergängen. Aktuell 301 Episoden — typische VLA-Trainings nutzen 500–2000.

### 8.5 Empfehlung

Für dieses Projekt (kein echter Roboter verfügbar):

1. **Sim-Diagnose als vollständig abgeschlossen betrachten:** Replay-Funktionalität und
   Domain-Gap-Ursache sind belegt — das ist ein valides wissenschaftliches Ergebnis.
2. **RL als Ausblick/Erweiterung formulieren** (falls kein Zeitrahmen mehr vorhanden).
3. **Falls Zeit:** REINFORCE-Ansatz auf vast.ai umsetzen — die Infrastruktur (Env,
   Reward-Grundlage, Docker-Pipeline) ist fertig.

---

## 7. Verwendete Artefakte

| Zweck | Datei |
|---|---|
| W&B-Trainingsmetriken (Mid-Run-Snapshot) | [`wandb-run-auswertung.md`](wandb-run-auswertung.md) · [`wandb-run-charts.html`](wandb-run-charts.html) |
| Diagnose 1 — Dataset-Replay (Sim/Config) | [`Simulation/kisski_replay_submit.sh`](../../Simulation/kisski_replay_submit.sh) · `Simulation/scripts/entrypoint_replay.sh` |
| Diagnose 2 — Open-Loop-Modell-Eval | [`Training/kisski_open_loop_eval.sh`](../../Training/kisski_open_loop_eval.sh) · `gr00t/eval/open_loop_eval.py` |
| Closed-Loop-Sim-Eval | `Simulation/kisski_sim_submit.sh` · `Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py` |
| GPU-Eignung der Sim (RT-Cores) | [`../simulation/umsetzungsnotizen.md`](../simulation/umsetzungsnotizen.md) |
