from __future__ import annotations

import math

import numpy as np

from camera_geometry import PinholeCamera
from reconstruct_cube_poses import (
    cube_vertices,
    fit_cube_pose,
    fit_cube_pose_3d,
    projected_bbox,
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
