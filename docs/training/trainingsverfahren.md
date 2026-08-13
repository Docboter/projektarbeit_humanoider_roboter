# Was für ein Training wird hier durchgeführt?

Dieses Dokument beschreibt **die Art des Trainings** — Modellarchitektur, Lernverfahren und was
genau gelernt wird. Für die *Bedienung* (wie man einen Job startet) siehe
[anleitung.md](anleitung.md) und [kisski-hpc.md](kisski-hpc.md); für die Parameterliste
[env-vars.md](env-vars.md).

## Kurzfassung

> **Supervised Fine-tuning (Post-Training) eines vortrainierten Vision-Language-Action-Modells
> per Imitation Learning / Behavior Cloning.** Konkret: Das Foundation-Modell **NVIDIA Isaac
> GR00T N1.6 (3B)** wird auf eine **einzelne neue Roboter-Plattform** (Unitree G1 + Dex3-Hände)
> und **eine Aufgabe** (Klötzchen stapeln) nachtrainiert. Das Modell lernt aus ~301
> teleoperierten Demonstrationen, aus Kamerabildern + Gelenkzuständen + Sprachbeschreibung die
> nächsten **16 Aktionsschritte** vorherzusagen. Das Aktions-Ziel wird über einen
> **Flow-Matching-/Diffusions-Action-Head** gelernt.

Das ist **kein** Reinforcement Learning und **kein** Training von Grund auf — es ist überwachtes
Nachtrainieren eines bereits breit vortrainierten Modells (daher *Fine-tuning* bzw. *Post-Training*).

---

## 1. Lernparadigma: Imitation Learning (Behavior Cloning)

- **Datenquelle:** menschliche **Teleoperation** — ein Mensch steuert den realen Unitree G1, die
  Trajektorien werden aufgezeichnet (Bilder, Gelenkwinkel, ausgeführte Aktionen).
- **Lernziel:** Das Modell soll die demonstrierte Aktion möglichst gut **nachahmen** (Behavior
  Cloning). Es gibt **keine Belohnungsfunktion** und **keine Exploration** wie bei RL — rein
  überwachtes Lernen auf (Beobachtung → Aktion)-Paaren.
- **Konsequenz:** Qualität und Abdeckung der Demonstrationen bestimmen die erreichbare Leistung.
  Zustände außerhalb der Demonstrationsverteilung (Distribution Shift) sind die typische
  Schwäche dieses Ansatzes.

## 2. Basismodell: GR00T N1.6 (3B) — ein Vision-Language-Action-Modell

GR00T N1.6 (`Gr00tN1d6`) ist ein **VLA-Foundation-Modell** mit zwei Hauptkomponenten:

| Komponente | Rolle | Quelle |
|---|---|---|
| **VLM-Backbone (NVIDIA Eagle)** | verarbeitet Kamerabilder + Sprach-Prompt zu einem gemeinsamen Vision-Language-Embedding | `gr00t/model/modules/eagle_backbone.py` |
| **Action-Head (Flow-Matching-Diffusion-Policy)** | erzeugt aus dem VL-Embedding + Roboterzustand die Aktions-Sequenz; Cross-Attention auf die Backbone-Features | `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` (*„Action head component for flow matching diffusion policy"*) |

Das Modell wurde von NVIDIA auf großen, **multi-embodiment** Robotik-Datensätzen vortrainiert.
Wir bauen darauf auf und spezialisieren es auf unser Embodiment.

### Was wird trainiert, was bleibt eingefroren?

Standard der `FinetuneConfig` (`gr00t/configs/finetune_config.py`) — **selektives Tuning**, kein
LoRA:

| Modellteil | `tune_*` | Status |
|---|---|---|
| Sprach-/LLM-Backbone | `tune_llm = False` | **eingefroren** |
| Vision-Encoder | `tune_visual = False` | **eingefroren** |
| Projector | `tune_projector = True` | **trainiert** |
| Diffusion-/Action-Head | `tune_diffusion_model = True` | **trainiert** |

→ Das breite Welt-/Sprachwissen des VLM bleibt erhalten; angepasst werden nur **Projector** und
**Action-Head**, die das Wissen auf unsere konkrete Roboter-Aktion abbilden. Das spart Speicher
und reduziert Overfitting auf den kleinen Datensatz.

## 3. Das Embodiment: Unitree G1 + Dex3 (28 DOF)

Tag: `NEW_EMBODIMENT` — eine im Vortraining **nicht** enthaltene Plattform, daher der
New-Embodiment-Pfad. Zustands-/Aktionsvektor mit **28 Dimensionen**
(`examples/G1_DEX3/g1_dex3_config.py`, `modality_4cam.json`):

| Index | Gruppe | DOF | Aktions-Repräsentation |
|---|---|---|---|
| 0–6 | `left_arm` | 7 | **relative** Gelenkpositions-Deltas |
| 7–13 | `right_arm` | 7 | **relative** Gelenkpositions-Deltas |
| 14–20 | `left_dex3` (3-Finger-Hand) | 7 | **absolute** Gelenkwinkel-Ziele |
| 21–27 | `right_dex3` (3-Finger-Hand) | 7 | **absolute** Gelenkwinkel-Ziele |

## 4. Ein- und Ausgaben

**Beobachtung (Eingabe) pro Zeitschritt:**

- **4 RGB-Kameras:** `cam_left_high`, `cam_right_high` (Kopf/Szene) + `cam_left_wrist`,
  `cam_right_wrist` (Handgelenke)
- **Propriozeptiver Zustand:** 28-dim Gelenkzustand (siehe oben)
- **Sprache:** Aufgabenbeschreibung (`annotation.human.task_description`)

**Aktion (Ausgabe):**

- **Action Chunk** über **16 zukünftige Zeitschritte** (`delta_indices = range(16)`), je 28-dim.
  Das Modell sagt also nicht nur den nächsten, sondern einen ganzen kurzen **Aktions-Horizont**
  voraus (typisch für GR00T — stabiler, weniger Mikro-Ruckeln).

## 5. Trainingsobjektiv: Flow Matching

Der Action-Head ist eine **Flow-Matching-/Diffusions-Policy**:

- Beim Training wird das Modell darauf trainiert, das **Geschwindigkeitsfeld** (velocity field) zu
  regredieren, das verrauschte Aktionen in die echten Demonstrations-Aktionen überführt
  (Flow-Matching-Loss, konditioniert auf VL-Embedding + Zustand).
- Bei der **Inferenz** wird die Aktion iterativ **entrauscht** (Integration in
  `num_inference_timesteps` Schritten, `gr00t_n1d6.py`), bis die finale Aktions-Sequenz steht.
- Bilder werden zusätzlich per **Color-Jitter** augmentiert (`--color_jitter_params` in
  `run_finetuning.sh`), um Robustheit gegen Beleuchtungs-/Farbschwankungen zu erhöhen.

## 6. Datensatz

| Eigenschaft | Wert |
|---|---|
| Datensatz | `unitreerobotics/G1_Dex3_BlockStacking_Dataset` (HuggingFace, **LeRobot**-Format) |
| Aufgabe (`total_tasks`) | 1 — Klötzchen stapeln |
| Episoden | **301** |
| Frames | **281.196** |
| Aufnahmerate | 30 fps |

Vor dem Training konvertiert die Pipeline den Datensatz ins LeRobot-Format und ergänzt bei Bedarf
`meta/modality.json` aus `modality_4cam.json` (siehe `run_finetuning.sh`).

## 7. Trainings-Hyperparameter — Lauf 1 (1× A100, historische Referenz)

> ⚠️ **Nicht mehr der aktuelle KISSKI-Default.** Die Tabelle beschreibt den ersten Lauf auf
> **einer** A100. Seit der Multi-GPU-Umstellung gilt in
> [`kisski_submit.sh`](../../Training/kisski_submit.sh): `MAX_STEPS=44000`,
> `GLOBAL_BATCH_SIZE=32`, `NUM_GPUS=4`, `LEARNING_RATE=2e-4` (Nicht-Vision-Pfad).
> Hintergrund: [multi-gpu.md](multi-gpu.md).

Gesetzt in [Training/kisski_submit.sh](../../Training/kisski_submit.sh) bzw.
`Training/scripts/run_finetuning.sh`:

| Parameter | Wert (Lauf 1) | Anmerkung |
|---|---|---|
| `MAX_STEPS` | 175.000 | ~5 Epochen (281k Frames / Batch 8 ≈ 35k Schritte/Epoche); das Konfig-Feld `num_train_epochs = 3` wird davon überschrieben |
| `GLOBAL_BATCH_SIZE` | 8 | A100 80 GB verträgt mehr (siehe [env-vars.md](env-vars.md)) |
| `LEARNING_RATE` | 1e-4 | aktueller Default: `2e-4` |
| `WARMUP_RATIO` | 0.05 | |
| `WEIGHT_DECAY` | 1e-5 | |
| Optimierer | AdamW | (HF-Trainer-Default) |
| Aktions-Horizont | 16 Schritte | |
| Objektiv | Flow-Matching-Loss | |
| `SAVE_STEPS` / `SAVE_TOTAL_LIMIT` | 5.000 / 40 | 35 Checkpoints über den 175k-Lauf verteilt. Skript-Default ist inzwischen `SAVE_STEPS=2000` (Vision-Variante: 1000) |

## 8. Abgrenzung — was es *nicht* ist

- **Kein RL:** keine Belohnung, kein Environment-Feedback, keine Exploration während des Trainings.
- **Kein Training von Grund auf:** wir starten von NVIDIAs vortrainierten Gewichten.
- **Kein Full-Fine-tuning:** VLM-Backbone (Vision + Sprache) bleibt eingefroren; nur Projector +
  Action-Head werden trainiert.
- **Kein Multi-Task/Multi-Embodiment-Training:** ein Embodiment (G1+Dex3), eine Aufgabe
  (Block-Stacking).

Die spätere **Evaluation** des fertigen Checkpoints (Closed-Loop in Isaac Lab) ist ein separater
Schritt — siehe [Simulation](../simulation/README.md).

---

## Quellen im Code

- `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` — Modell + Flow-Matching-Action-Head
- `gr00t/model/modules/eagle_backbone.py` — Eagle-VLM-Backbone
- `gr00t/configs/finetune_config.py` — Tuning-Defaults (`tune_llm/visual/projector/diffusion`)
- `gr00t/experiment/launch_finetune.py` — Einstiegspunkt des Fine-tunings
- `examples/G1_DEX3/g1_dex3_config.py`, `examples/G1_DEX3/modality_4cam.json` — Embodiment-/Modality-Definition
- `Training/scripts/run_finetuning.sh` — Trainingsbefehl mit allen Argumenten
