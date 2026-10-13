# TL;DR: Sprachbefehl der Policy aus dem echten Datensatz lesen statt fest eintragen (Isaac-frei).
"""Task-Text, mit dem die Policy gefahren wird.

Der echte Datensatz `unitreerobotics/G1_Dex3_BlockStacking_Dataset` hat genau eine
Aufgabe: "stack three block" (aus `meta/tasks.parquet` auf HF gelesen, 2026-10-11).
Bis 2026-10-13 schickten alle Closed-Loop-Pfade fest "stack the blocks" — einen Text,
der im Training nie vorkam (projekt-review-2026-10-11.md §3.1). Deshalb hier eine
Stelle, die den Text aus dem Datensatz liest und nur ersatzweise auf die Konstante
zurückfällt.

Bewusst ohne Isaac- und Torch-Abhängigkeit, damit es überall importierbar ist
(Eval, RL, Render, Tests auf der CPU).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# Einziger Task des echten Datensatzes. Letzter Fallback, wenn kein Datensatz lesbar ist.
DEFAULT_TASK_DESCRIPTION = "stack three block"

# Ablageort des echten Datensatzes im Sim-Container (span/render/layout laden ihn dorthin).
DEFAULT_DATASET_PATH = "/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset"


def read_dataset_tasks(root: str | os.PathLike | None) -> dict[int, str]:
    """`meta/tasks.jsonl` (LeRobot v2.1) bzw. `meta/tasks.parquet` (v3.0) → {index: text}.

    Leeres Dict, wenn nichts lesbar ist. Fehler sind nie fatal: ein fehlender Datensatz
    darf keinen Eval-Lauf kosten, er kostet nur die Herkunftsangabe.
    """
    if not root:
        return {}
    meta = Path(root) / "meta"
    jsonl = meta / "tasks.jsonl"
    if jsonl.is_file():
        try:
            out = {}
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    out[int(rec["task_index"])] = str(rec["task"])
            return out
        except Exception:
            return {}
    parquet = meta / "tasks.parquet"
    if parquet.is_file():
        try:
            import pandas as pd

            df = pd.read_parquet(parquet)
            # v3.0: der Text steht im Index, task_index als Spalte.
            if "task" in df.columns:
                return {int(i): str(t) for i, t in zip(df["task_index"], df["task"])}
            return {int(i): str(t) for t, i in zip(df.index, df["task_index"])}
        except Exception:
            return {}
    return {}


def resolve_task_description(
    explicit: str | None = None,
    dataset_path: str | os.PathLike | None = None,
) -> tuple[str, str]:
    """Liefert (Text, Herkunft). Reihenfolge: explizit > Datensatz > Konstante.

    `dataset_path=None` heißt: `TASK_DATASET_PATH` aus der Umgebung, sonst
    DEFAULT_DATASET_PATH. Hat der Datensatz mehrere Tasks, gilt task_index 0.
    """
    if explicit and explicit.strip():
        return explicit.strip(), "explizit (--task-description / TASK_DESCRIPTION)"
    if dataset_path is None:
        dataset_path = os.environ.get("TASK_DATASET_PATH") or DEFAULT_DATASET_PATH
    tasks = read_dataset_tasks(dataset_path)
    if tasks:
        idx = 0 if 0 in tasks else min(tasks)
        return tasks[idx], f"Datensatz {dataset_path} (task_index {idx})"
    return DEFAULT_TASK_DESCRIPTION, "Fallback-Konstante (kein Datensatz lesbar)"
