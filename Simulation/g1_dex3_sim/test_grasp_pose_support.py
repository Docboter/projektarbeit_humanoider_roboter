# TL;DR: Prüft Hilfsfunktionen für Greif-Positionsunterstützung.

from __future__ import annotations

import numpy as np

from grasp_pose_support import closure_candidates, fingertip_measurement


def test_closure_candidates_require_coordinated_fingers() -> None:
    states = np.zeros((40, 28), dtype=float)
    states[12:20, 14] = np.linspace(0.0, 0.4, 8)
    states[20:, 14] = 0.4
    assert closure_candidates(states, "left") == []
    states[12:20, 15] = np.linspace(0.0, -0.3, 8)
    states[20:, 15] = -0.3
    candidates = closure_candidates(states, "left")
    assert candidates
    assert candidates[0]["coordinated_joints"] >= 2


def test_opening_is_rejected_by_sparse_fk_spread_check() -> None:
    open_tips = np.asarray([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]])
    closed_tips = open_tips * 0.5
    open_spread, _ = fingertip_measurement(open_tips)
    closed_spread, _ = fingertip_measurement(closed_tips)
    assert open_spread - closed_spread > 0.006
    assert closed_spread - open_spread < 0.006
