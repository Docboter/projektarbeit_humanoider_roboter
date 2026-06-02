# Train-Test-Split (80/20)

Dieses Dokument beschreibt, wie der Block-Stacking-Datensatz in einen Trainings- und einen Test-Split aufgeteilt wurde, damit das nachtrainierte Modell nach dem Fine-tuning auf ungesehenen Episoden bewertet werden kann.

---

## ⚠️ Status: Split aktuell NICHT aktiv — vor dem nächsten Training beachten!

Der Split ist **im Code vorbereitet, aber nicht scharf geschaltet**. Der Filter
(`_apply_split_filter`) liest den Bereich aus `meta/info.json` — dort steht aber
weiterhin der originale Eintrag:

```json
"splits": { "train": "0:301" }
```

Dadurch werden **alle 301 Episoden** als `train` geladen; die 60 vorgesehenen
Test-Episoden werden mittrainiert. Bestätigt durch das Trainings-Log von Job
14058798 (`Total steps: 276681` = voller Datensatz; bei 80 % wären es ~221.000
Frames).

**Vor dem nächsten Training zwingend erledigen:**

1. In `data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/info.json` den
   `splits`-Eintrag auf `{ "train": "0:241", "test": "241:301" }` setzen
   (Details unter [Geänderte Dateien](#geänderte-dateien)).
2. Erst **danach** den Job einreichen — der Datensatz wird beim Job-Start
   gesharded; eine spätere Änderung wirkt nicht mehr auf einen laufenden Job.

> **Hinweis:** Auch mit korrektem Split gibt es **keine** Validierung *während*
> des Trainings (in `run_finetuning.sh` ist kein `eval`/`val`-Flag gesetzt). Der
> `test`-Split dient ausschließlich der Evaluation **nach** dem Fine-tuning.

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
- **Achtung:** Wer den Split nutzt, muss ihn *konsequent* nutzen — neben der
  `info.json` (siehe Status-Hinweis oben) auch `Training/kisski_open_loop_eval.sh`
  so anpassen, dass die Trajektorien-IDs aus dem **`test`-Bereich (241–300)**
  stammen. Aktuell zieht das Skript IDs über alle 301 Episoden → die Open-Loop-Eval
  läuft sonst weiter auf Trainingsdaten.
- **Vertretbare Alternative:** auf allen 301 Episoden trainieren (maximale Daten)
  und **ausschließlich in der Sim** validieren — methodisch sauber, solange die
  Sim-Startbedingungen nicht 1:1 aus den Demos stammen.

---

## Aufteilung

| Split | Episoden | Anteil | Episoden-Indices |
|-------|----------|--------|-----------------|
| `train` | 241 | 80 % | 0 – 240 |
| `test` | 60 | 20 % | 241 – 300 |

Die ersten 241 Episoden werden zum Training verwendet. Die letzten 60 Episoden werden während des Trainings **nie geladen** und stehen danach für die Evaluation zur Verfügung.

---

## Geänderte Dateien

### 1. `data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/info.json`

Der `splits`-Eintrag wurde angepasst:

```json
// vorher
"splits": { "train": "0:301" }

// nachher
"splits": {
  "train": "0:241",
  "test":  "241:301"
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

Um das Modell nach dem Fine-tuning auf den 60 Testepisoden auszuwerten, kann der `LeRobotEpisodeLoader` oder `ShardedSingleStepDataset` direkt mit `split="test"` instanziiert werden:

```python
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

loader = LeRobotEpisodeLoader(
    dataset_path="/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset",
    modality_configs=modality_configs,
    split="test",   # lädt nur Episoden 241–300
)

# Einzelne Episode laden (Index bezieht sich auf die gefilterte Liste)
episode_df = loader[0]   # entspricht episode_index 241 im Datensatz
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
