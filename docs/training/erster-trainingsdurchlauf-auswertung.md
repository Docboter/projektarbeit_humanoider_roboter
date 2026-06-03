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
| Steps | **175.000** (`num_train_epochs = 3` ≈ 3 Datendurchläufe) |
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
| `train/loss` | 0,10 → ~0,03 (Step ~25k) → ~0,01 (Step ~100k) → **~0,008 final** (avg `train_loss` 0,030) | sauberer Abfall, kein NaN |
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
- **Eval einschalten:** `enable_open_loop_eval = true` (+ ggf. `eval_strategy = "steps"`;
  `eval_set_split_ratio = 0.1` ist gesetzt) → Val-MSE über die Zeit, fundierte
  Checkpoint-Auswahl statt „letzter".
- **Batch-Size hochziehen** auf der A100 (siehe [`multi-gpu.md`](multi-gpu.md)) — bei
  4× A100 / `global_batch_size = 32` reichen ~40–50k Steps für denselben Konvergenzstand
  (Loss plateaut ohnehin ab ~100k bei bs=8).

---

## 7. Verwendete Artefakte

| Zweck | Datei |
|---|---|
| W&B-Trainingsmetriken (Mid-Run-Snapshot) | [`wandb-run-auswertung.md`](wandb-run-auswertung.md) · [`wandb-run-charts.html`](wandb-run-charts.html) |
| Diagnose 1 — Dataset-Replay (Sim/Config) | [`Simulation/kisski_replay_submit.sh`](../../Simulation/kisski_replay_submit.sh) · `Simulation/scripts/entrypoint_replay.sh` |
| Diagnose 2 — Open-Loop-Modell-Eval | [`Training/kisski_open_loop_eval.sh`](../../Training/kisski_open_loop_eval.sh) · `gr00t/eval/open_loop_eval.py` |
| Closed-Loop-Sim-Eval | `Simulation/kisski_sim_submit.sh` · `Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py` |
| GPU-Eignung der Sim (RT-Cores) | [`../simulation/implementation-notes.md`](../simulation/implementation-notes.md) |
