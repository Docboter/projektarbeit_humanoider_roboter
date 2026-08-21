from __future__ import annotations

import numpy as np

from grasp_pose_support import apply_grasp_support, closure_candidates, fingertip_measurement


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


def blocks() -> list[dict]:
    return [
        {
            "name": "block_0",
            "status": "ok",
            "cv_position_m": [0.34, -0.12, 0.915],
            "position_m": [0.34, -0.12, 0.915],
        },
        {
            "name": "block_1",
            "status": "ok",
            "cv_position_m": [0.34, 0.12, 0.915],
            "position_m": [0.34, 0.12, 0.915],
        },
    ]


def event(x: float, y: float) -> dict:
    return {
        "hand": "left",
        "grasp_frame": 100,
        "closure_start_frame": 90,
        "fingertip_centroid_m": [x, y, 0.93],
    }


def test_unique_grasp_support_uses_75_25_fusion() -> None:
    values = blocks()
    assert apply_grasp_support(values, [event(0.36, -0.10)]) == 1
    expected = 0.75 * np.asarray([0.36, -0.10]) + 0.25 * np.asarray([0.34, -0.12])
    assert np.allclose(values[0]["position_m"][:2], expected)
    assert values[0]["position_source"] == "cv_grasp_fused"


def test_ambiguous_grasp_support_is_rejected() -> None:
    values = blocks()
    values[0]["cv_position_m"][1] = -0.03
    values[0]["position_m"][1] = -0.03
    values[1]["cv_position_m"][1] = 0.03
    values[1]["position_m"][1] = 0.03
    assert apply_grasp_support(values, [event(0.34, 0.0)]) == 0
    assert all(block["position_source"] == "cv" for block in values)


def test_grasp_support_over_eight_centimetres_is_rejected() -> None:
    values = blocks()
    assert apply_grasp_support(values, [event(0.44, -0.02)]) == 0
    assert all(block["position_source"] == "cv" for block in values)
