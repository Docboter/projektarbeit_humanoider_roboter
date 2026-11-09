# TL;DR: CPU-Tests für eval_metrics.py (lifted_clear-Geometrie, Clopper-Pearson) und split_math.py.
"""python -m pytest Simulation/g1_dex3_sim/test_eval_metrics.py"""

from __future__ import annotations

import math
from pathlib import Path
import re

from eval_metrics import (
    HAND_CONTACT_PROXY_M,
    clopper_pearson,
    cube_lowest_z,
    format_rate_ci,
    lifted_clear_step,
)
import numpy as np
import pytest
from scene_geometry import CUBE_EDGE, CUBE_Z_CENTER, TABLE_HEIGHT
from split_math import held_out_range, split_record, total_from_info, train_count


HALF = CUBE_EDGE / 2
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    return (math.cos(angle / 2), *(axis * s))


# ---------------------------------------------------------------------------
# Würfel-Unterseite
# ---------------------------------------------------------------------------

def test_upright_cube_rests_on_table():
    assert float(cube_lowest_z((0.3, 0.0, CUBE_Z_CENTER), IDENTITY, HALF)) == pytest.approx(
        TABLE_HEIGHT, abs=1e-9)


def test_yaw_does_not_change_the_lowest_point():
    q = _quat_axis_angle((0, 0, 1), math.radians(37))
    assert float(cube_lowest_z((0, 0, CUBE_Z_CENTER), q, HALF)) == pytest.approx(TABLE_HEIGHT)


def test_tilt_onto_edge_raises_centre_but_not_the_bottom():
    """Kante: Mitte steigt um 2,5·(√2−1) ≈ 1,04 cm, Unterseite bleibt auf dem Tisch."""
    q = _quat_axis_angle((1, 0, 0), math.radians(45))
    centre = TABLE_HEIGHT + HALF * math.sqrt(2)
    assert (centre - CUBE_Z_CENTER) * 100 == pytest.approx(1.036, abs=0.01)
    assert float(cube_lowest_z((0, 0, centre), q, HALF)) == pytest.approx(TABLE_HEIGHT)
    ok, clearance = lifted_clear_step(
        np.array([[0, 0, centre]]), np.array([q]), np.array([0.0]), TABLE_HEIGHT, HALF)
    assert not ok and clearance == pytest.approx(0.0, abs=1e-9)


def test_tilt_onto_corner_does_not_count():
    """Ecke: Mitte steigt um ≈ 1,83 cm (knapp unter 2 cm) — Unterseite trotzdem 0."""
    # Körperdiagonale (1,1,1) auf Welt-z drehen.
    d = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
    axis = np.cross(d, [0, 0, 1])
    q = _quat_axis_angle(axis, math.acos(d[2]))
    centre = TABLE_HEIGHT + HALF * math.sqrt(3)
    assert (centre - CUBE_Z_CENTER) * 100 == pytest.approx(1.83, abs=0.01)
    assert float(cube_lowest_z((0, 0, centre), q, HALF)) == pytest.approx(TABLE_HEIGHT)
    ok, _ = lifted_clear_step(
        np.array([[0, 0, centre]]), np.array([q]), np.array([0.0]), TABLE_HEIGHT, HALF)
    assert not ok


def test_corner_tilt_plus_small_lift_would_fool_the_old_metric_only():
    """Ecke + 0,3 cm Abheben: Schwerpunkt +2,1 cm (alte Metrik: lifted), Unterseite 0,3 cm."""
    d = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
    q = _quat_axis_angle(np.cross(d, [0, 0, 1]), math.acos(d[2]))
    centre = TABLE_HEIGHT + HALF * math.sqrt(3) + 0.003
    assert centre - CUBE_Z_CENTER > 0.02          # alte Metrik würde zählen
    ok, clearance = lifted_clear_step(
        np.array([[0, 0, centre]]), np.array([q]), np.array([0.0]), TABLE_HEIGHT, HALF)
    assert not ok and clearance == pytest.approx(0.003, abs=1e-6)


def test_real_lift_with_hand_counts():
    pos = np.array([[0.3, 0, CUBE_Z_CENTER + 0.03], [0.3, 0.15, CUBE_Z_CENTER]])
    quats = np.array([IDENTITY, IDENTITY])
    ok, clearance = lifted_clear_step(pos, quats, np.array([0.03, 0.5]), TABLE_HEIGHT, HALF)
    assert ok and clearance == pytest.approx(0.03)


def test_lift_without_hand_contact_does_not_count():
    """Geworfener / abprallender Würfel: hoch, aber keine Hand dran."""
    pos = np.array([[0.3, 0, CUBE_Z_CENTER + 0.05]])
    ok, clearance = lifted_clear_step(
        pos, np.array([IDENTITY]), np.array([HAND_CONTACT_PROXY_M + 0.01]), TABLE_HEIGHT, HALF)
    assert not ok and clearance == 0.0


def test_unnormalised_quaternion_is_tolerated():
    q = np.array(IDENTITY) * 3.0
    assert float(cube_lowest_z((0, 0, CUBE_Z_CENTER), q, HALF)) == pytest.approx(TABLE_HEIGHT)


# ---------------------------------------------------------------------------
# Clopper-Pearson
# ---------------------------------------------------------------------------

def test_cp_zero_successes_match_the_review():
    assert clopper_pearson(0, 10) == (0.0, pytest.approx(0.3085, abs=5e-4))
    assert clopper_pearson(0, 20) == (0.0, pytest.approx(0.1684, abs=5e-4))


def test_cp_closed_form_edges():
    # k = 0: obere Grenze = 1 − (α/2)^(1/n); k = n: untere Grenze = (α/2)^(1/n).
    for n in (1, 5, 10, 37):
        assert clopper_pearson(0, n)[1] == pytest.approx(1 - 0.025 ** (1 / n), abs=1e-9)
        assert clopper_pearson(n, n) == (pytest.approx(0.025 ** (1 / n), abs=1e-9), 1.0)


def test_cp_known_value_and_symmetry():
    lo, hi = clopper_pearson(5, 10)                 # Literaturwert 0,1871–0,8129
    assert lo == pytest.approx(0.1871, abs=5e-4) and hi == pytest.approx(0.8129, abs=5e-4)
    lo3, hi3 = clopper_pearson(3, 20)
    lo17, hi17 = clopper_pearson(17, 20)
    assert lo3 == pytest.approx(1 - hi17, abs=1e-9) and hi3 == pytest.approx(1 - lo17, abs=1e-9)


def test_cp_rejects_impossible_counts_and_handles_empty():
    with pytest.raises(ValueError):
        clopper_pearson(11, 10)
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    assert "keine gültigen" in format_rate_ci(0, 0)
    assert format_rate_ci(0, 10).startswith("0/10 = 0% [KI95 0.0%–30.8%]")


# ---------------------------------------------------------------------------
# Split-Formel
# ---------------------------------------------------------------------------

def test_split_matches_the_real_dataset():
    assert train_count(301, 0.8) == 240           # int(240.8)
    assert train_count(300, 0.8) == 240
    assert list(held_out_range(301, 0.8)) == list(range(240, 301))
    assert split_record(301, 0.8)["n_train"] == 240


@pytest.mark.parametrize("total,ratio", [(10, 0.0), (10, 1.0), (1, 0.8), (0, 0.8)])
def test_split_rejects_empty_or_full(total, ratio):
    with pytest.raises(ValueError):
        train_count(total, ratio)


def test_split_float_semantics_match_python_in_lib_split():
    # lib_split.sh rechnet in Python: int(total * float(ratio)). 0.7 * 10 = 7.000…001.
    for total in range(2, 400):
        for ratio in (0.7, 0.8, 0.9, 0.75, 0.85):
            n = int(total * float(ratio))
            if 0 < n < total:
                assert train_count(total, ratio) == n


def test_lib_split_still_uses_the_same_formula():
    lib = Path(__file__).resolve().parents[2] / "Training" / "scripts" / "lib_split.sh"
    if not lib.is_file():
        pytest.skip("lib_split.sh nicht im Baum (Sim-Image)")
    src = lib.read_text()
    assert re.search(r"n_train\s*=\s*int\(total \* ratio\)", src)
    assert "float(sys.argv[2])" in src
    assert "0 < n_train < total" in src


def test_total_from_info(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text('{"total_episodes": 120}')
    assert total_from_info(tmp_path) == 120
    assert total_from_info(tmp_path / "fehlt") is None


def test_layout_train_count_prefers_the_file_then_the_dataset(tmp_path, monkeypatch):
    from split_math import layout_train_count
    monkeypatch.delenv("TRAIN_SPLIT_RATIO", raising=False)
    assert layout_train_count({"n_train": 96}) == (96, "layout.json")
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text('{"total_episodes": 120}')
    n, note = layout_train_count({"source_dataset": str(tmp_path)})
    assert n == 96 and "WARNUNG" in note
    n, _ = layout_train_count({"source_dataset": "/fehlt"}, dataset_fallback=tmp_path)
    assert n == 96
    monkeypatch.setenv("TRAIN_SPLIT_RATIO", "0.9")
    assert layout_train_count({}, dataset_fallback=tmp_path)[0] == 108
    assert layout_train_count({}, dataset_fallback=tmp_path / "x")[0] is None
