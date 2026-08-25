"""Eine vorhandene layout.json darf ein Teillauf nicht leeren.

Die Bewegungsbeginne in dieser Datei kosten je Episode und Kamera rund vier Sekunden
Videodekodierung. Ein Probelauf über drei Episoden, der die anderen 57 mitnimmt, ist
deshalb kein Schönheitsfehler, sondern verlorene Rechenzeit — und schlimmer: er fällt
nicht auf, weil die Datei danach gültig aussieht.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_block_layout import Z_CUBE_CENTER, header_mismatch, run_extract


def make_dataset(tmp_path: Path, total_episodes: int = 10) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": total_episodes}))
    return root


def extract_args(root: Path, out: Path, **over) -> argparse.Namespace:
    args = argparse.Namespace(
        dataset_path=str(root), out=str(out), num_episodes=3, episode_ids=None,
        train_ratio=0.8, cameras="cam_left_high,cam_right_high", overwrite=False,
        fresh=False, no_motion_onset=True, frame=0, min_area=120, bias=(0.0, 0.0),
        debug_dir="", yaw_tolerance=8.0, min_squareness=1.25)
    for k, v in over.items():
        setattr(args, k, v)
    return args


def seeded_layout(out: Path, root: Path, episodes=("0", "1", "2")) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "episodes": {e: {"cubes": [[0.3, 0.0], [0.35, 0.1], [0.4, -0.1]],
                         "motion_onset": {"first": 40}} for e in episodes},
        "cameras": ["cam_left_high", "cam_right_high"], "bias_cm": [0.0, 0.0],
        "z_plane": Z_CUBE_CENTER, "source_dataset": str(root)}))


# ── Zusammenführen ──────────────────────────────────────────────────────────

def test_partial_rerun_keeps_the_other_episodes(tmp_path: Path) -> None:
    root, out = make_dataset(tmp_path), tmp_path / "layout.json"
    seeded_layout(out, root)
    run_extract(extract_args(root, out, episode_ids=[5], overwrite=True))
    got = json.loads(out.read_text())["episodes"]
    assert {"0", "1", "2"} <= set(got), sorted(got)
    assert got["1"]["motion_onset"]["first"] == 40, "alter Eintrag wurde überschrieben"


def test_fresh_starts_from_nothing(tmp_path: Path) -> None:
    root, out = make_dataset(tmp_path), tmp_path / "layout.json"
    seeded_layout(out, root)
    run_extract(extract_args(root, out, episode_ids=[5], fresh=True))
    assert set(json.loads(out.read_text())["episodes"]) == {"5"}


def test_rerun_without_overwrite_skips_existing_episodes(tmp_path: Path) -> None:
    root, out = make_dataset(tmp_path), tmp_path / "layout.json"
    seeded_layout(out, root)
    run_extract(extract_args(root, out, episode_ids=[1]))
    assert json.loads(out.read_text())["episodes"]["1"]["motion_onset"]["first"] == 40


# ── Kopf-Prüfung ────────────────────────────────────────────────────────────

def test_a_different_bias_refuses_instead_of_mixing(tmp_path: Path) -> None:
    """Zwei Bias-Werte in einer Datei sind nicht unterscheidbar — also gar nicht erst."""
    root, out = make_dataset(tmp_path), tmp_path / "layout.json"
    seeded_layout(out, root)
    assert run_extract(extract_args(root, out, episode_ids=[5], bias=(0.01, 0.0))) == 1
    assert set(json.loads(out.read_text())["episodes"]) == {"0", "1", "2"}


def test_header_mismatch_names_what_differs() -> None:
    want = {"bias_cm": [0.0, 0.0], "z_plane": 0.895,
            "cameras": ["a", "b"], "source_dataset": "/d"}
    assert header_mismatch(dict(want), want) == ""
    assert "Bias" in header_mismatch({**want, "bias_cm": [1.0, 0.0]}, want)
    assert "Würfelebene" in header_mismatch({**want, "z_plane": 0.915}, want)
    assert "Kameras" in header_mismatch({**want, "cameras": ["a"]}, want)


def test_header_check_tolerates_a_file_written_before_the_header_existed() -> None:
    want = {"bias_cm": [0.0, 0.0], "z_plane": 0.895,
            "cameras": ["a"], "source_dataset": "/d"}
    assert header_mismatch({"episodes": {}}, want) == ""


# ── Bewegungsbeginn ─────────────────────────────────────────────────────────

def test_skipping_the_onset_measurement_keeps_an_existing_one(tmp_path: Path) -> None:
    """--no-motion-onset heißt „nicht messen", nicht „löschen".

    Sonst macht ein schneller Teillauf die Datei für den Renderer unbrauchbar, ohne
    es zu sagen: das Fenster stünde dann auf None, und die vier Sekunden je Video
    und Kamera wären umsonst ausgegeben worden.
    """
    root, out = make_dataset(tmp_path), tmp_path / "layout.json"
    seeded_layout(out, root)
    run_extract(extract_args(root, out, episode_ids=[1], overwrite=True,
                             no_motion_onset=True))
    got = json.loads(out.read_text())["episodes"]["1"]
    assert got["motion_onset"]["first"] == 40
    assert got["cubes"] == [None, None, None], "Würfel sollten neu gerechnet sein"


def test_a_new_episode_without_onset_measurement_records_none(tmp_path: Path) -> None:
    root, out = make_dataset(tmp_path), tmp_path / "layout.json"
    seeded_layout(out, root)
    run_extract(extract_args(root, out, episode_ids=[5], no_motion_onset=True))
    assert json.loads(out.read_text())["episodes"]["5"]["motion_onset"]["first"] is None
