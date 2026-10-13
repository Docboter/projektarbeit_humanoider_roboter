# TL;DR: CPU-Tests für den Eval-Messaufbau: Task-Text, gemeinsame Geometrie, Held-out-Layout.
"""Prüft die Isaac-freien Teile der Korrekturen aus dem Review vom 2026-10-11, §3.

    python -m pytest Simulation/g1_dex3_sim/test_eval_setup.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_block_layout import Z_CUBE_CENTER, header_mismatch, select_episodes
import pytest
from scene_geometry import CUBE_Z_CENTER, TABLE_HEIGHT, TABLE_POS, TABLE_SIZE
from task_text import DEFAULT_TASK_DESCRIPTION, resolve_task_description


def _dataset(tmp_path: Path, total: int = 301, task: str | None = "stack three block") -> Path:
    meta = tmp_path / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": total}))
    if task is not None:
        (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    return tmp_path


# --- §3.1 Task-Text ------------------------------------------------------------------

def test_default_is_the_only_task_of_the_real_dataset():
    assert DEFAULT_TASK_DESCRIPTION == "stack three block"


def test_task_is_read_from_the_dataset(tmp_path):
    root = _dataset(tmp_path, task="stack three block")
    text, source = resolve_task_description("", root)
    assert text == "stack three block"
    assert "Datensatz" in source


def test_explicit_task_wins(tmp_path):
    root = _dataset(tmp_path)
    assert resolve_task_description("stack the blocks", root)[0] == "stack the blocks"


def test_missing_dataset_falls_back_to_the_constant(tmp_path):
    text, source = resolve_task_description(None, tmp_path / "fehlt")
    assert text == DEFAULT_TASK_DESCRIPTION
    assert "Fallback" in source


# --- §3.5 Gemeinsame Geometrie ---------------------------------------------------------

def test_cube_rests_on_the_table():
    assert CUBE_Z_CENTER == pytest.approx(0.895)
    assert TABLE_SIZE[2] == TABLE_HEIGHT
    assert TABLE_POS[2] == pytest.approx(TABLE_HEIGHT / 2)


def test_layout_extraction_uses_the_shared_cube_plane():
    assert Z_CUBE_CENTER == CUBE_Z_CENTER


# --- §3.2 Held-out-Startlagen ----------------------------------------------------------

def _args(**kw) -> argparse.Namespace:
    base = {"train_ratio": 0.8, "num_episodes": 60, "episode_ids": None, "held_out": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_held_out_selects_only_test_episodes(tmp_path):
    root = _dataset(tmp_path)
    eps = select_episodes(root, _args(held_out=True, num_episodes=100))
    assert eps == list(range(240, 301))


def test_held_out_rejects_training_episodes(tmp_path):
    root = _dataset(tmp_path)
    with pytest.raises(SystemExit):
        select_episodes(root, _args(held_out=True, episode_ids=[10, 250]))


def test_render_selection_still_never_touches_test_episodes(tmp_path):
    root = _dataset(tmp_path)
    assert max(select_episodes(root, _args())) < 240
    with pytest.raises(SystemExit):
        select_episodes(root, _args(episode_ids=[250]))


def test_train_and_test_layouts_do_not_mix():
    want = {"bias_cm": [0.0, 0.0], "z_plane": 0.895, "cameras": ["a"], "source_dataset": "x",
            "split": "test"}
    old_train = {**want}
    del old_train["split"]   # Dateien von vor 2026-10-13 haben keinen Split = train
    assert "Split" in header_mismatch(old_train, want)
    assert header_mismatch({**want}, want) == ""
