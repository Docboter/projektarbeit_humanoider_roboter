from __future__ import annotations

import numpy as np

from camera_geometry import PinholeCamera
from replay_calibration import action_sha256, best_top_face_detection, top_face_blob


def test_action_hash_detects_any_value_change() -> None:
    actions = np.arange(84, dtype=np.float32).reshape(3, 28)
    original_hash = action_sha256(actions)
    copied = actions.copy()
    assert action_sha256(copied) == original_hash
    copied[1, 7] += 1e-3
    assert action_sha256(copied) != original_hash


def test_top_face_detection_prefers_bright_surface() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[40:90, 50:110] = [150, 10, 10]
    image[40:62, 50:110] = [245, 20, 20]
    blob = top_face_blob(image, "rot", min_area=80)
    assert blob is not None
    assert blob["source"] == "top_face"
    assert blob["v"] < 65
    assert len(blob["axis_uv"]) == 2


def test_best_top_face_detection_uses_largest_stable_candidate() -> None:
    small = np.zeros((100, 120, 3), dtype=np.uint8)
    large = small.copy()
    small[30:45, 40:60] = [240, 20, 20]
    large[30:60, 40:80] = [240, 20, 20]
    index, _, blob = best_top_face_detection([small, large], "rot", min_area=50)
    assert index == 1
    assert blob is not None


def test_projection_backprojection_on_fixed_cube_top_plane() -> None:
    camera = PinholeCamera.from_cfg("cam_left_high")
    expected = np.asarray([[0.34, 0.08, 0.94]], dtype=float)
    u, v = camera.project(expected)[0]
    recovered = camera.backproject_to_plane(float(u), float(v), 0.94)
    assert np.allclose(recovered, expected[0], atol=1e-8)
