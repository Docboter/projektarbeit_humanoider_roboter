# Auswertung — zweiter Trainingsdurchlauf mit Vision-Encoder (`g1_dex3_blockstacking_vision_v1`)

> **TL;DR:** Auswertung des zweiten Trainingslaufs mit mittrainiertem Vision-Encoder
> (`tune_visual=true`, ohne Augmentierung) als Gegenprobe zu Lauf 1 — die Closed-Loop-Policy
> kollabiert auf reines Arm-Zurückziehen. Kurzfazit der Befunde in §1.

**Erstellt:** 2026-06-05 · **Run-ID:** `ajgoskon` (W&B-Projekt `gr00t-g1-dex3`, Entity
`projektarbeit_humanoider_roboter`) · **Modell:** GR00T N1.6 (3,29 Mrd. Parameter),
Finetune **mit aufgetautem Vision-Encoder** (`tune_visual = true`) auf Unitree G1 + DEX3,
Task „stack the blocks".

> Dies ist die Auswertung des **zweiten** vollständigen Laufs. Er ist die direkte
> Gegenprobe zur Hauptempfehlung aus dem ersten Lauf
> ([`lauf1-auswertung.md`](lauf1-auswertung.md) §8.4, Option A): „Was passiert, wenn man
> den Vision-Encoder mittrainiert?" Datengrundlage sind die maschinell ausgewerteten
> W&B-Kurven (`ajgoskon`) und die Closed-Loop-Beobachtung des resultierenden Checkpoints.

---

## 1. Kurzfazit (TL;DR)

- **Training selbst: erneut gesund und sauber auskonvergiert.** Loss 1,37 → ~0,009,
  Cosine-LR korrekt auf ~0, grad_norm gesund sinkend, keine NaN. Der **finale Loss ist
  praktisch identisch** zum ersten Lauf (0,0093 vs. 0,0081).
- **Verhalten im Closed-Loop: schlechter, nicht besser.** Statt des „Anfahren →
  Einfrieren" aus Lauf 1 kollabiert die Policy diesmal auf eine **degenerierte
  Einzelbewegung — der Roboter zieht nur noch die Arme zurück** (kein Anfahren, kein
  Greifen, kein Stapeln).
- **Damit ist die Vorhersage aus Lauf 1 empirisch bestätigt:** Den Vision-Encoder auf
  **reinen Realdaten** aufzutauen schließt den Sim-Domain-Gap **nicht** — der Encoder
  bekommt im Training nie ein Sim-Bild zu sehen — und bringt bei nur ~301 Demonstrationen
  das Risiko von **Catastrophic Forgetting / Über-Spezialisierung** mit, das sich hier als
  Politik-Kollaps materialisiert hat. Differenzierende Einordnung nach Lauf 3: siehe
  §7-Nachtrag bzw. [`lauf3-vision-split-auswertung.md`](lauf3-vision-split-auswertung.md).
- **Strukturelle Lücke wie in Lauf 1:** weiterhin **keine Validierungs-/Eval-Metrik**
  (`eval_strategy = "no"`) → Checkpoint-Auswahl blind, Overfitting prinzipiell unsichtbar.

---

## 2. Lauf-Eckdaten (abgeschlossen)

| Feld | Wert |
|---|---|
| Run-ID / Name | `ajgoskon` / `g1_dex3_blockstacking_vision_v1` |
| State | `finished` |
| Start | 2026-06-04 23:51 UTC |
| Laufzeit | ~30.172 s (**~8,4 h**) |
| GPUs / Global-Batch | **4 GPUs** (aus Durchsatz abgeleitet, s. u.) / per-device 8 → **global 32** |
| Steps | **44.000** (`num_train_epochs = 3`) |
| Durchsatz | **1,46 Steps/s**, **46,7 Samples/s** |
| Trainierbare Teile | **Vision-Encoder** + Projector + Diffusion-Head + oberste 4 LLM-Layer + VLLN |
| Eingefroren | nur noch das **LLM-Backbone** (`tune_llm = false`) |
| LR / Schedule | 1e-4, cosine, `warmup_ratio = 0.1` → Warmup 4.400 Steps |
| Action-Repräsentation | Arme `RELATIVE`, Hände (`*_dex3`) `ABSOLUTE`, `use_relative_action = true` (wie Lauf 1) |
| Datensatz | `unitreerobotics/G1_Dex3_BlockStacking_Dataset` (real, Teleop), identisch zu Lauf 1 |
| Output | `/data/g1_dex3_finetune/blockstacking_vision/g1_dex3_blockstacking_vision_v1` |

> **Warum nur 44.000 statt 175.000 Steps bei identischen 3 Epochen?** Nicht weil weniger
> trainiert wurde — die **gesehene Datenmenge ist gleich** (~1,4 Mio. Samples). Lauf 2 lief
> auf **4 GPUs** mit Global-Batch 32 (4 × per-device 8) statt auf 1 GPU mit Batch 8. Bei
> 4-fachem Batch reichen 4-fach weniger Schritte für dieselben 3 Datendurchläufe
> (175.000 / 4 ≈ 44.000). Belegt durch den Durchsatz: 46,7 Samples/s = 1,46 Steps/s ×
> Batch 8 × **4** → Welt-Größe 4. Trotz aufgetautem Encoder war der Lauf dank Multi-GPU
> mit ~8,4 h **schneller** als die ~21,4 h des Single-GPU-Laufs.

---

## 3. Trainingsdynamik — gesund ✅ (wie Lauf 1)

| Metrik | Verlauf | Bewertung |
|---|---|---|
| `train/loss` | 1,374 (Start) → 0,343 (1k) → 0,107 (2k) → 0,064 (5k) → 0,047 (10k) → 0,023 (20k) → 0,012 (30k) → 0,0083 (40k) → **0,0093 final**; Minimum 0,0037, Tail-Mittel ~0,0086 | sauberer Abfall, kein NaN, faktisches Plateau ab ~30k |
| `train/grad_norm` | ~2,29 (Start), Peak ~2,62 früh → **0,08 final** | gesund sinkend; Clipping (`max_grad_norm = 1`) nur in den ersten Steps aktiv |
| `train/learning_rate` | Cosine, Peak 1,0e-4 bei Step ~4.410 (= Warmup 0,1 × 44.000) → Ende ~0 (1,6e-13) | korrekt für 44.000 Steps |

**Befund:** Aus reiner Optimierungssicht ist der Lauf so sauber wie Lauf 1 — der
**finale Loss ist sogar praktisch deckungsgleich** (0,0093 vs. 0,0081). Das Auftauen des
Vision-Encoders hat das Training **nicht** destabilisiert. Wie schon in Lauf 1 gilt aber:
Niedriger Flow-Matching-Loss = gute Nachahmung der **Trainingsverteilung**, **kein**
Aufgabenerfolg und **kein** Generalisierungssignal. Die Aussage über die Policy kommt
ausschließlich aus der Verhaltens-Evaluation (Abschnitt 4).

---

## 4. Verhaltens-Evaluation — Regression statt Verbesserung ❌

Closed-Loop-Sim-Eval mit dem finalen Vision-Checkpoint:

> **Der Roboter zieht nur noch die Arme zurück (retraktiert) — eine degenerierte
> Einzelbewegung. Kein Anfahren an die Würfel, kein Greifen, kein Stapeln.**

Das ist eine **Verschlechterung gegenüber Lauf 1**: Dort fuhr der Roboter immerhin grob an
die Würfel heran, bevor er einfror. Hier kollabiert die Policy auf eine repetitive
Rückzugsbewegung — das klassische Bild eines **Policy-Kollaps** in einen degenerierten
Attraktor.

| | **Lauf 1 (state-only, frozen Encoder)** | **Lauf 2 (vision, getunter Encoder)** |
|---|---|---|
| Closed-Loop-Verhalten | Anfahren (~15 s), dann **Einfrieren** | nur **Arme zurückziehen** (Kollaps) |
| Erfolgsrate | 0 / 20 | 0 (kein Aufgabenfortschritt) |
| Interpretation | unbrauchbare Sim-Features → „nichts tun" | Politik auf degenerierte Bewegung kollabiert |

> **Hinweis zur Belastbarkeit:** Für Lauf 2 liegt (noch) **keine** separate Open-Loop-MSE-
> und keine erneute Domain-Gap-Messung des Vision-Checkpoints vor; die Bewertung stützt
> sich auf die Closed-Loop-Beobachtung. Ein Open-Loop-Eval auf echten Dataset-Bildern
> (`open_loop_eval.py`) wäre der nächste Schritt, um zu prüfen, ob der getunte Encoder
> **auch auf Realbildern** schlechter geworden ist (Erwartung bei Catastrophic Forgetting:
> ja). Siehe „Offene Punkte" unten.

---

## 5. Synthese & Diagnose

**Das Training ist nicht das Problem — wieder nicht.** Beide Läufe konvergieren auf
praktisch denselben Loss. Der einzige relevante Konfigurationsunterschied mit
Verhaltenswirkung ist `tune_visual` (von `false` auf `true`); der größere Global-Batch
(32 statt 8) ist unkritisch und erklärt einen Kollaps nicht.

**Warum `tune_visual = true` hier schadet statt hilft:**

1. **Der Domain-Gap wird nicht adressiert.** Der Encoder sieht im Training **ausschließlich
   echte Teleop-Kamerabilder** — nie ein Isaac-Sim-Rendering. Domain-Invarianz ist eine
   *im Training erlernte* Eigenschaft, kein zur Eval-Zeit umlegbarer Schalter. Ein nur auf
   Realbildern feinabgestimmter Encoder wird **nicht** robuster gegen Sim-Bilder — er
   **spezialisiert sich im Gegenteil noch stärker** auf das exakte Erscheinungsbild der
   Realbilder. Der Gap zu den Sim-Renderings bleibt oder wächst.

2. **Catastrophic Forgetting / Über-Spezialisierung.** Beim Auftauen des ~2-Mrd.-Parameter-
   Encoders auf nur **~301 Demonstrationen** verliert dieser die generischen,
   vortrainierten Merkmale, die einen eingefrorenen Encoder vergleichsweise sim-tolerant
   machen. Genau das erklärt, warum das Closed-Loop-Verhalten **schlechter** wird als mit
   frozen Encoder.

3. **Ergebnis: Politik-Kollaps.** Auf den (weiterhin out-of-distribution) Sim-Bildern
   liefert der nun über-spezialisierte Encoder unbrauchbare Features, und die Action-Head
   fällt in einen degenerierten Attraktor — hier die Arm-Rückzugsbewegung.

Damit ist die in Lauf 1 §8.4 (Option A) und in der Diskussion der Ausarbeitung
(§„Eingefrorener vs. mittrainierter Vision-Encoder") **vorab formulierte Hypothese
empirisch bestätigt**: Vision-Encoder-Tuning auf reinen Realdaten ist für die
Sim-Evaluation dieses Projekts **kontraproduktiv**.

---

## 6. Handlungsempfehlungen

### 6.1 `tune_visual = true` auf reinen Realdaten nicht weiterverfolgen
Für die Sim-Evaluation ist der **frozen Encoder (Lauf 1) die robustere Wahl**. Ein
getunter Encoder wäre nur dann sinnvoll, wenn er im Training **tatsächlich Sim- oder per
Domain-Randomization variierte Bilder** sieht — was hier nicht der Fall war. Ohne
sim-ähnliche Trainingsbilder ist `TUNE_VISUAL` gegen den Domain-Gap wirkungslos und
riskiert den beobachteten Kollaps.

> **Nachtrag 2026-08-13 — die Bedingung aus §6.2 ist inzwischen erfüllt.** Diese Empfehlung
> gilt für *reine* Realdaten. Lauf 2 lief am **04./05.06.**; die Augmentierung
> (`USE_AUGMENTATION`, Color-Jitter) kam erst am **12.06.** dazu (`c3a1cd6`) — Lauf 2 hatte
> also gar keinen Jitter. Ein neuer `TUNE_VISUAL`-Lauf mit `USE_AUGMENTATION=1` ist deshalb
> **keine Wiederholung**, sondern der erste, der die hier geforderte Voraussetzung erfüllt.
> Zwei Vorbehalte bleiben: Color-Jitter variiert Farbe, **nicht** Geometrie, Textur oder
> Rendering-Stil — und der Ausreißer der
> [Domain-Gap-Messung](domain-gap-analyse.md) ist `cam_left_wrist` (0,36), eine
> Nahbereichskamera. Der Lauf ist ein begründeter billiger Test, keine Erfolgsgarantie; das
> Gate dafür steht in [next-steps.md](../../next-steps.md).
>
> Bis 2026-08-13 kannte `run_finetuning_vision.sh` den Schalter ohnehin nicht — der
> Color-Jitter war dort **fest verdrahtet**, `USE_AUGMENTATION` wirkungslos. Beides ist
> jetzt behoben.

### 6.2 Wenn der Vision-Pfad weiterverfolgt wird
- **Sim-/DR-Bilder ins Training mischen** (Domain-Randomization oder co-Training auf
  gerenderten Bildern), damit das Auftauen des Encoders den Gap überhaupt schließen kann.
- **Mehr Daten:** ~301 Demos sind für ein Vollauftauen des Encoders zu wenig; typische
  VLA-Trainings nutzen 500–2000.

### 6.3 Eval endlich einschalten (gilt für beide Läufe)
Val-MSE über die Zeit statt „letzter Step" — ohne das ist auch dieser Lauf am Ende blind auf
Step 44.000 ausgewählt worden.

> **Korrektur 2026-08-13 — der hier ursprünglich genannte Weg funktioniert nicht.**
> Empfohlen war `enable_open_loop_eval = true` (+ `eval_strategy = "steps"`,
> `eval_set_split_ratio = 0.1`). Alle drei sind im Fork wirkungslos oder schädlich:
> `training_config.py:100–109` deklariert `enable_open_loop_eval` und die drei
> `open_loop_eval_*`-Felder, die **nirgends gelesen** werden; `eval_set_split_ratio` ebenso;
> und `data/dataset/factory.py:26` bricht mit `assert eval_strategy == "no"` ab, gibt dem
> Loader fest `split="train"` und liefert `eval_dataset=None`. Ein `eval_strategy="steps"`
> **stürzt ab**, statt zu evaluieren. In-Training-Validierung ist in diesem Fork
> strukturell nicht vorhanden, nicht bloß abgeschaltet.
>
> **Umgesetzt wurde stattdessen die Auswertung nach dem Lauf:**
> [`checkpoint_sweep.py`](../../Training/scripts/checkpoint_sweep.py) lädt jeden Checkpoint
> und misst MSE/MAE auf den zurückgehaltenen Episoden (`TRAIN_TEST_SPLIT=1` vorausgesetzt).
> Bedienung: [train-test-split.md](../training/train-test-split.md),
> [env-vars.md](../training/env-vars.md).

### 6.4 Diagnose nachschärfen (optional, billig)
**Open-Loop-Eval des Vision-Checkpoints** auf echten Dataset-Bildern ausführen. Sind die
Arm-Vorhersagen dort schlechter als beim frozen-Encoder-Checkpoint aus Lauf 1, ist das
Catastrophic Forgetting direkt zahlenmäßig belegt — und nicht nur aus dem Closed-Loop
erschlossen.

---

## 7. Offene Punkte / nachzutragen

> **Nachtrag 2026-08-14 — Lauf 3 liegt vor.** Die in §6.2 geforderte Voraussetzung
> (Encoder-Tuning **mit** Augmentierung) wurde am 13./14.08. eingelöst:
> [`lauf3-vision-split-auswertung.md`](lauf3-vision-split-auswertung.md) (`tp1nc699`,
> `TUNE_VISUAL=1` + `USE_AUGMENTATION=1` + `TRAIN_TEST_SPLIT=1`). Damit ist auch die hier
> unter §6.3 geforderte Validierung erstmals vorhanden — allerdings **nach** dem Lauf per
> [`checkpoint_sweep.py`](../../Training/scripts/checkpoint_sweep.py), nicht in-training.
> Ergebnis: bester Checkpoint bei Step 30.000, der letzte ist 25 % schlechter. Das erklärt
> rückwirkend auch einen Teil des Lauf-2-Befunds: der hier evaluierte finale Checkpoint war
> mit hoher Wahrscheinlichkeit **nicht der beste des Laufs** — der Politik-Kollaps ist damit
> nicht widerlegt, aber die Zuschreibung „liegt allein an `tune_visual`" wird schwächer.
> Ein Sweep über die Lauf-2-Checkpoints fehlt (Lauf 2 lief ohne Split, ergäbe also nur
> Trainings-MSE).

- Quantitative **Open-Loop-MSE/MAE** des Vision-Checkpoints (pro Trajektorie + Mittel),
  Vergleich gegen Lauf 1.
- **Erneute Domain-Gap-Messung** mit dem getunten Encoder (SigLIP-Cosine-Distanz) — ist
  der Gap zur Sim größer geworden?
- Bestätigung der **GPU-Anzahl** (hier aus dem Durchsatz auf 4 abgeleitet) und der
  konkreten Plattform (KISSKI A100/H100) aus dem SLURM-Log.

---

## 8. Verwendete Artefakte / Quellen

| Zweck | Quelle |
|---|---|
| W&B-Trainingsmetriken Lauf 2 | Run `ajgoskon`, Projekt `gr00t-g1-dex3` |
| Vergleichslauf (frozen Encoder) | [`lauf1-auswertung.md`](lauf1-auswertung.md) (`i6n1t613`) |
| Domain-Gap-Messung (frozen Encoder) | [`domain-gap-analyse.md`](domain-gap-analyse.md) |
| Methodik-Review Sim-Eval | [`sim-bewertung.md`](sim-bewertung.md) |
| Vision-Encoder-Routing (`TUNE_VISUAL`) | [`../training/env-vars.md`](../training/env-vars.md) · `Training/scripts/run_finetuning_vision.sh` |
