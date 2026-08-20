# Train-Test-Split (80/20)

> **TL;DR:** Beschreibt den 80/20-Train-Test-Split des Datensatzes und die Checkpoint-Auswahl
> per Open-Loop-Eval nach dem Training (`checkpoint_sweep.py`), da In-Training-Validierung im
> Fork nicht möglich ist. Aktivierbar per `TRAIN_TEST_SPLIT=1`.

Dieses Dokument beschreibt, wie der Block-Stacking-Datensatz in einen Trainings- und einen Test-Split aufgeteilt wurde, damit das nachtrainierte Modell nach dem Fine-tuning auf ungesehenen Episoden bewertet werden kann.

---

## Status: per Env-Var scharf schaltbar (Standard AUS)

Der Split ist **im Code vorbereitet** (Filter `_apply_split_filter`, der den Bereich
aus `meta/info.json` liest) **und über einen Schalter aktivierbar**:

```bash
TRAIN_TEST_SPLIT=1 ...        # 80/20-Split scharf schalten
TRAIN_SPLIT_RATIO=0.8         # optional: Trainingsanteil (Default 0.8)
```

[`lib_split.sh`](../../Training/scripts/lib_split.sh) patcht dann **vor dem Trainingsstart**
automatisch den `splits`-Eintrag in `meta/info.json` (kein manuelles Editieren mehr nötig).
Die Logik teilen sich seit 2026-08-13 **beide** Trainings-Skripte — auch der Vision-Lauf
(`TUNE_VISUAL=1`), der den Schalter vorher gar nicht kannte:

```json
// TRAIN_TEST_SPLIT=1 (bei 301 Episoden, Ratio 0.8):
"splits": { "train": "0:240", "test": "240:301" }
```

Dadurch werden nur die ersten 240 Episoden als `train` geladen; die letzten 61
stehen als `test` für die Open-Loop-Eval auf **ungesehenen** Episoden bereit.

- **Default (`TRAIN_TEST_SPLIT=0`)**: kompletter Datensatz (`train: 0:301`) — bisheriges
  Verhalten. Ein zuvor gesetzter `test`-Split wird automatisch auf den vollen Datensatz
  zurückgesetzt, damit ein Folge-Lauf reproduzierbar alle Episoden sieht.
- Der Patch greift, **bevor** der Datensatz beim Job-Start gesharded wird — eine spätere
  Änderung an `info.json` wirkt nicht mehr auf einen laufenden Job.

> **Hinweis 1: Validierung während des Trainings ist im Fork strukturell nicht möglich.**
> Das ist kein fehlendes Flag, sondern eine Sperre im Code (geprüft 2026-08-13):
> `training_config.py:100–109` deklariert `enable_open_loop_eval` und drei
> `open_loop_eval_*`-Felder, die **nirgends gelesen** werden; `data/dataset/factory.py:26`
> bricht mit `assert eval_strategy == "no"` ab; derselbe Factory gibt dem Loader fest
> `split="train"` und liefert `eval_dataset=None`. Ein `eval_strategy="steps"` würde also
> **abstürzen**, nicht evaluieren. Der `test`-Split dient deshalb ausschließlich der
> Evaluation **nach** dem Fine-tuning.
>
> **Hinweis 2: Die Auswertung auf dem `test`-Split ist umgesetzt** —
> [`checkpoint_sweep.py`](../../Training/scripts/checkpoint_sweep.py), auf KISSKI über
> [`kisski_open_loop_eval.sh`](../../Training/kisski_open_loop_eval.sh). Früher stand hier die
> Auflage, die Trajektorien-IDs von Hand in den `test`-Bereich zu legen. **Das wäre falsch
> gewesen:** `_apply_split_filter` filtert die Episodenliste, und `loader[idx]` indiziert in
> die **gefilterte** Liste. Position 0 ist bei aktivem Split die Episode 240 — eine
> „traj_id 240" wäre also die Episode 480 gewesen, die es nicht gibt. Der Sweep übergibt
> Positionen im Split und protokolliert den absoluten `episode_index` dazu.

---

## Motivation

Der originale Datensatz (`unitreerobotics/G1_Dex3_BlockStacking_Dataset`) enthielt alle 301 Episoden in einem einzigen `train`-Split. Ohne eine separate Testmenge wäre eine objektive Bewertung nach dem Training nicht möglich — das Modell würde auf denselben Daten bewertet, auf denen es gelernt hat.

---

## Welche Validierung wofür? (Methodik)

**Lohnt sich der Split überhaupt?** Für eine Imitation-Learning-/VLA-Policy wie
GR00T nur bedingt — der entscheidende methodische Punkt:

> **Niedriger Action-Loss ≠ erfolgreiche Aufgabe.** Eine Policy kann auf
> Test-Episoden eine gute MSE erreichen und beim echten Stapeln trotzdem
> scheitern (Fehler akkumulieren über den Rollout, Demonstrationen sind
> multimodal). Der held-out Loss ist nur ein **schwacher Proxy** für die
> eigentlich relevante Größe: die **Task-Success-Rate**.

Die Validierungssignale, geordnet nach Aussagekraft (und was im Projekt dafür
existiert):

| Signal | Was es misst | Aussagekraft | Braucht Daten-Split? |
|---|---|---|---|
| **Trainings-Loss** (W&B) | Konvergenz | nur Sanity-Check | nein |
| **Open-Loop-Eval** (`gr00t/eval/open_loop_eval.py`, `Training/kisski_open_loop_eval.sh`) — Action-MSE/MAE | Aktions-Vorhersagefehler auf Trajektorien | **schwacher** Proxy; gut zum billigen Checkpoint-Vergleich | **ja** — sonst Bewertung auf Trainingsdaten |
| **Closed-Loop-Sim** (Isaac Lab, `Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py`) | tatsächliche **Task-Success-Rate** | **die** relevante Metrik | nein (Sim randomisiert Startbedingungen) |
| **Echter Roboter** | reale Erfolgsrate | Goldstandard | nein |

**Konsequenzen:**

- Der **Closed-Loop-Sim-Eval ist die eigentliche Validierung** — er beantwortet
  „funktioniert die Policy?". Er braucht **keinen** Daten-Split; ungesehene
  *Startbedingungen* (Block-Positionen) sind sogar der aussagekräftigere
  Generalisierungstest als zurückgehaltene Demo-Episoden.
- Der 80/20-Split nützt vor allem der **Open-Loop-Eval**: er macht deren
  MSE/MAE-Zahl zu einem sauberen *held-out* Wert (gut für den Bericht und zum
  schnellen Vorfiltern der Checkpoints).
- **Datenkosten beachten:** 20 % von 301 Episoden zurückzuhalten reduziert bei
  Behavior Cloning spürbar die Trainingsdaten — ein realer Trade-off, wenn man
  ohnehin primär in der Sim validiert.

**Empfohlene Pipeline:**

```
Train-Loss (Konvergenz)
  → Open-Loop-MSE (billiger Checkpoint-Filter, auf test-Split!)
  → Closed-Loop-Sim (Erfolgsrate, Hauptmetrik)
  → echter Roboter (final)
```

- **Primär** auf die **Closed-Loop-Sim-Success-Rate** validieren.
- Den Split **behalten**, aber als *sekundäres, billiges* Signal nutzen (Action-MSE
  zum Checkpoint-Vergleich, bevor die teure Sim-Eval nur auf die besten 2–3
  Checkpoints angewandt wird).
- **Erledigt seit 2026-08-13:** Der Checkpoint-Vergleich läuft über
  [`checkpoint_sweep.py`](../../Training/scripts/checkpoint_sweep.py) und misst standardmäßig
  auf `split="test"`. Vorher wertete `kisski_open_loop_eval.sh` einen **fest verdrahteten**
  `checkpoint-3000` auf Trajektorien-IDs über alle 301 Episoden aus — also Trainingsdaten.
  Die Ursache lag tiefer als der Aufruf: `gr00t/eval/open_loop_eval.py` reicht **kein**
  `split` an den `LeRobotEpisodeLoader` durch, dessen Default `split="train"` ist. Wer jenes
  Skript direkt benutzt, misst nach einem Split-Lauf weiterhin Trainings-MSE.
- **Vertretbare Alternative:** auf allen 301 Episoden trainieren (maximale Daten)
  und **ausschließlich in der Sim** validieren — methodisch sauber, solange die
  Sim-Startbedingungen nicht 1:1 aus den Demos stammen.

---

## Aufteilung

| Split | Episoden | Anteil | Episoden-Indices |
|-------|----------|--------|-----------------|
| `train` | 240 | 80 % | 0 – 239 |
| `test` | 61 | 20 % | 240 – 300 |

Die ersten 240 Episoden werden zum Training verwendet. Die letzten 61 Episoden werden während des Trainings **nie geladen** und stehen danach für die Evaluation zur Verfügung.

> **Korrektur 2026-08-14.** Hier stand vorher `train` = 241 Episoden (0–240) und `test` = 60
> (241–300). Das ist um eins verschoben: `split_apply()` rechnet
> `n_train = int(301 × 0.8) = int(240,8) = **240**`. Bestätigt durch den ersten realen
> Split-Lauf — der Sweep von Lauf 3 protokolliert `split_range: "240:301"` und löst
> Split-Position 0 auf `episode_index 240` auf
> ([`lauf3-vision-split-auswertung.md`](../ergebnisse/lauf3-vision-split-auswertung.md)).

---

## Geänderte Dateien

### 1. `data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/info.json`

Der `splits`-Eintrag wurde angepasst:

```json
// vorher
"splits": { "train": "0:301" }

// nachher
"splits": {
  "train": "0:240",
  "test":  "240:301"
}
```

### 2. `app/Groot-1.6/gr00t/data/dataset/lerobot_episode_loader.py`

Der `LeRobotEpisodeLoader` liest den Split-Bereich jetzt aus `info.json` und filtert `episodes_metadata` entsprechend. Hinzugefügt wurden:

- **Parameter `split`** (Standard: `"train"`) im Konstruktor
- **Methode `_apply_split_filter()`**: parst den `"start:end"`-String aus `info.json` und behält nur Episoden, deren `episode_index` im Bereich `[start, end)` liegt

```python
def _apply_split_filter(self) -> None:
    splits = self.info_meta.get("splits", {})
    if self.split not in splits:
        return
    start, end = (int(x) for x in splits[self.split].split(":"))
    self.episodes_metadata = [
        ep for ep in self.episodes_metadata
        if start <= ep["episode_index"] < end
    ]
```

### 3. `app/Groot-1.6/gr00t/data/dataset/sharded_single_step_dataset.py`

Der `split`-Parameter wird vom Konstruktor an `LeRobotEpisodeLoader` weitergegeben:

```python
# vorher
self.episode_loader = LeRobotEpisodeLoader(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    ...
)

# nachher
self.episode_loader = LeRobotEpisodeLoader(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    ...
    split=split,          # neu
)
```

### 4. `app/Groot-1.6/gr00t/data/dataset/factory.py`

Beim Aufbau des Trainingsdatensatzes wird `split="train"` explizit übergeben:

```python
dataset = ShardedSingleStepDataset(
    ...
    split="train",    # neu — stellt sicher, dass nur Episoden 0–240 geladen werden
)
```

---

## Test-Split nach dem Training verwenden

Um das Modell nach dem Fine-tuning auf den 61 Testepisoden auszuwerten, kann der `LeRobotEpisodeLoader` oder `ShardedSingleStepDataset` direkt mit `split="test"` instanziiert werden:

```python
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

loader = LeRobotEpisodeLoader(
    dataset_path="/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset",
    modality_configs=modality_configs,
    split="test",   # lädt nur Episoden 240–300
)

# Einzelne Episode laden (Index bezieht sich auf die gefilterte Liste)
episode_df = loader[0]   # entspricht episode_index 240 im Datensatz
```

Oder für ein vollständiges Evaluation-Dataset:

```python
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

test_dataset = ShardedSingleStepDataset(
    dataset_path="/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset",
    embodiment_tag=EmbodimentTag("new_embodiment"),
    modality_configs=modality_configs,
    split="test",
)
```

---

## Hinweis zur Reproduzierbarkeit

Der Split basiert auf dem **natürlichen Index** der Episoden in `episodes.jsonl` (nicht zufällig), damit er stabil und reproduzierbar ist. Episoden mit ähnlichem Aufzeichnungsdatum oder -kontext können dadurch im gleichen Split landen — dies ist für diese Aufgabe (Block Stacking, einzelne Aufgabe) unkritisch.
