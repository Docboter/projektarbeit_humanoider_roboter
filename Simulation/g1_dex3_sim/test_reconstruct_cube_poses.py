from __future__ import annotations

import math

import numpy as np

from camera_geometry import PinholeCamera, quat_to_matrix
from reconstruct_cube_poses import (
    cube_vertices,
    fit_cube_pose,
    fit_cube_pose_3d,
    matrix_to_quat,
    projected_bbox,
)
from replay_calibration import (
    action_sha256,
    apply_homography,
    closing_hand_scores,
    find_motion_onset,
    fit_homography_ransac,
    match_closing_hand,
    top_face_blob,
)


def test_cube_vertices_have_requested_edge() -> None:
    vertices = cube_vertices(np.array([0.3, -0.1, 0.9]), edge=0.05, yaw=0.37)
    assert np.isclose(vertices[:, 2].max() - vertices[:, 2].min(), 0.05)
    assert np.isclose(np.linalg.norm(vertices[0] - vertices[1]), 0.05)


def test_two_camera_bbox_fit_recovers_synthetic_pose() -> None:
    cameras = {
        "cam_left_high": PinholeCamera.from_cfg("cam_left_high"),
        "cam_right_high": PinholeCamera.from_cfg("cam_right_high"),
    }
    expected = np.array([0.34, 0.08, 0.915])
    edge, yaw = 0.05, 0.24
    blobs = {}
    for name, camera in cameras.items():
        bbox = projected_bbox(camera, expected, edge, yaw)
        uv = camera.project(expected)
        blobs[name] = {"bbox": bbox.tolist(), "u": float(uv[0]), "v": float(uv[1])}

    center, fitted_yaw, error = fit_cube_pose(blobs, cameras, expected[2], edge)

    assert np.linalg.norm(center - expected) < 0.002
    yaw_error = abs((fitted_yaw - yaw + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0)
    assert yaw_error < math.radians(2.0)
    assert error < 1.0


def test_two_camera_3d_bbox_fit_recovers_height() -> None:
    cameras = {
        "cam_left_high": PinholeCamera.from_cfg("cam_left_high"),
        "cam_right_high": PinholeCamera.from_cfg("cam_right_high"),
    }
    expected = np.array([0.37, -0.06, 0.912])
    edge, yaw = 0.05, 0.31
    blobs = {}
    for name, camera in cameras.items():
        bbox = projected_bbox(camera, expected, edge, yaw)
        uv = camera.project(expected)
        blobs[name] = {"bbox": bbox.tolist(), "u": float(uv[0]), "v": float(uv[1])}

    center, fitted_yaw, error = fit_cube_pose_3d(blobs, cameras, 0.93, edge)

    assert np.linalg.norm(center - expected) < 0.003
    yaw_error = abs((fitted_yaw - yaw + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0)
    assert yaw_error < math.radians(3.0)
    assert error < 1.0


def test_homography_ransac_rejects_outlier() -> None:
    pixels = np.array([
        [100, 100], [200, 100], [300, 100], [100, 200], [200, 200],
        [300, 200], [100, 300], [200, 300], [300, 300], [550, 20],
    ], dtype=float)
    expected = np.column_stack((0.001 * pixels[:, 0] + 0.1,
                                -0.0015 * pixels[:, 1] + 0.3))
    expected[-1] = [2.0, 2.0]
    matrix, inliers = fit_homography_ransac(pixels, expected, threshold_m=0.005)
    recovered = apply_homography(matrix, pixels[:-1])
    assert inliers.sum() == 9
    assert np.max(np.linalg.norm(recovered - expected[:-1], axis=1)) < 1e-6


def test_motion_onset_and_closing_hand_are_aligned() -> None:
    track = [[100.0, 100.0] for _ in range(20)] + [[112.0, 100.0] for _ in range(10)]
    onset = find_motion_onset(track, threshold_px=8.0, stable_frames=5)
    assert onset == 20
    spreads = np.full((30, 2), 0.08, dtype=float)
    spreads[12:, 0] = np.linspace(0.08, 0.04, 18)
    assert match_closing_hand(onset, spreads) == 0
    assert closing_hand_scores(onset, spreads)[0] > 0.02


def test_hand_closure_may_precede_cube_motion() -> None:
    spreads = np.full((120, 2), 0.08, dtype=float)
    spreads[35:56, 1] = np.linspace(0.08, 0.045, 21)
    spreads[56:, 1] = 0.045
    assert match_closing_hand(70, spreads) == 1


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


def test_matrix_to_quaternion_round_trip() -> None:
    quaternion = np.array([0.81, -0.21, 0.42, 0.35])
    quaternion /= np.linalg.norm(quaternion)
    rotation = quat_to_matrix(quaternion)
    recovered = matrix_to_quat(rotation)
    assert np.allclose(quat_to_matrix(recovered), rotation, atol=1e-8)
