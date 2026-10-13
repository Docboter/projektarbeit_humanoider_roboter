# TL;DR: Isaac-freie Split-Formel n_train = int(total*ratio), Spiegel von lib_split.sh.
"""Eine Quelle für die Split-Grenze der Sim-Werkzeuge.

Die Formel MUSS exakt der in ``Training/scripts/lib_split.sh`` (``split_apply``)
entsprechen, sonst landen zurückgehaltene Episoden im Co-Training-Datensatz oder
eine Eval-Lage wird falsch als "held-out" beschriftet:

    n_train = int(total * float(ratio))       # train = 0:n_train, test = n_train:total
    gültig nur, wenn 0 < n_train < total

Bis 2026-10-13 stand die Formel einzeln in extract_block_layout.py,
reconstruct_cube_poses.py, render_cotrain_dataset.py, run_dataset_replay_videos.py,
dazu die feste Konstante 240 in run_g1_dex3_sim_eval.py (Review 2026-10-13, R4/S6).
Ändert sich lib_split.sh, muss diese Datei mitziehen (test_split_math.py prüft das
gegen den Quelltext von lib_split.sh, solange der im Repo liegt).

Isaac-frei, nur Standardbibliothek.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# Default wie TRAIN_SPLIT_RATIO in Training/ (Dockerfile-ENV / lib_split.sh-Aufrufer).
DEFAULT_TRAIN_RATIO = 0.8


def train_count(total: int, ratio: float, *, validate: bool = True) -> int:
    """Anzahl Trainings-Episoden = Index der ersten zurückgehaltenen Episode.

    validate=True bricht wie lib_split.sh ab, wenn der Split leer oder voll wäre.
    """
    total = int(total)
    if total <= 0:
        raise ValueError(f"Ungültiges total_episodes ({total}).")
    n_train = int(total * float(ratio))
    if validate and not (0 < n_train < total):
        raise ValueError(f"Ungültiger Split: ratio={ratio} -> n_train={n_train} von {total}.")
    return n_train


def held_out_range(total: int, ratio: float) -> range:
    """Zurückgehaltene Test-Episoden: range(n_train, total)."""
    return range(train_count(total, ratio), int(total))


def split_record(total: int, ratio: float) -> dict:
    """Felder für layout.json & Co.: woraus die Grenze stammt, damit Leser sie nicht raten."""
    n_train = train_count(total, ratio)
    return {"total_episodes": int(total), "train_split_ratio": float(ratio),
            "n_train": n_train, "formula": "int(total*ratio), wie Training/scripts/lib_split.sh"}


def ratio_from_env(default: float = DEFAULT_TRAIN_RATIO) -> float:
    """TRAIN_SPLIT_RATIO aus der Umgebung (leer/ungültig = Default)."""
    raw = os.environ.get("TRAIN_SPLIT_RATIO", "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def total_from_info(dataset_root: str | os.PathLike) -> int | None:
    """total_episodes aus <dataset>/meta/info.json, None wenn nicht lesbar."""
    try:
        info = json.loads((Path(dataset_root) / "meta" / "info.json").read_text())
        total = int(info.get("total_episodes") or 0)
        return total if total > 0 else None
    except (OSError, ValueError, TypeError):
        return None


def layout_train_count(layout: dict,
                       dataset_fallback: str | os.PathLike | None = None
                       ) -> tuple[int | None, str]:
    """Split-Grenze n_train einer layout.json → (n_train oder None, Herkunft/Warnung).

    Neue Dateien (extract_block_layout.py ab 2026-10-13) tragen ``n_train`` im Kopf.
    Ältere nicht: dann aus meta/info.json des Datensatzes (erst ``source_dataset`` der
    Datei, sonst ``dataset_fallback``) und TRAIN_SPLIT_RATIO (Default 0.8) nachrechnen —
    mit Warnung, denn ob die Datei mit derselben Ratio erzeugt wurde, steht nicht darin.
    None = nicht bestimmbar; der Aufrufer darf dann keine Train/Test-Aussage treffen.
    """
    n = layout.get("n_train")
    if isinstance(n, int) and n > 0:
        return n, "layout.json"
    ratio = ratio_from_env()
    for root in (layout.get("source_dataset"), dataset_fallback):
        if not root:
            continue
        total = total_from_info(root)
        if total is None:
            continue
        try:
            n_train = train_count(total, ratio)
        except ValueError:
            continue
        return n_train, (f"WARNUNG: layout.json ohne n_train — nachgerechnet aus {root}/meta/"
                         f"info.json (total {total}) und TRAIN_SPLIT_RATIO={ratio}")
    return None, ("WARNUNG: layout.json ohne n_train und kein lesbares meta/info.json — "
                  "Train/Test-Zuordnung der Lagen unbekannt")
