# Train-Test-Split (80/20)

Dieses Dokument beschreibt, wie der Block-Stacking-Datensatz in einen Trainings- und einen Test-Split aufgeteilt wurde, damit das nachtrainierte Modell nach dem Fine-tuning auf ungesehenen Episoden bewertet werden kann.

---

## Motivation

Der originale Datensatz (`unitreerobotics/G1_Dex3_BlockStacking_Dataset`) enthielt alle 301 Episoden in einem einzigen `train`-Split. Ohne eine separate Testmenge wäre eine objektive Bewertung nach dem Training nicht möglich — das Modell würde auf denselben Daten bewertet, auf denen es gelernt hat.

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
