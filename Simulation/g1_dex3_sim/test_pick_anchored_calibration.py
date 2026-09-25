# TL;DR: Prüft Funktionen der anker-basierten Kalibrierung.

from __future__ import annotations

import json

import numpy as np
import pytest
import replay_calibration

from replay_calibration import (
    CUBE_CENTER_Z_M,
    CUBE_TOP_Z_M,
    FINGERTIP_Z_WINDOW_M,
    TABLE_TOP_Z_M,
    apply_homography,
    build_calibration,
    combine_camera_estimates,
    coverage_defects,
    deterministic_episode_split,
    episode_diagnostic_pixels,
    find_motion_onset,
    fit_homography_ransac,
    pose_resume_matches,
    require_calibration_dataset,
    require_pick_homography_calibration,
    select_unique_closing_hand,
    stable_top_face_measurement,
    stationary_camera_measurements,
    validate_anchor_document,
)


def test_motion_onset_requires_persistent_displacement() -> None:
    track = [[10.0, 20.0] for _ in range(15)]
    track += [[30.0, 20.0], [10.0, 20.0]]
    track += [[30.0, 20.0] for _ in range(5)]
    assert find_motion_onset(track) == 17
    assert find_motion_onset([[10.0, 20.0] for _ in range(30)]) is None


def test_stable_measurement_rejects_short_and_accepts_top_face_run() -> None:
    frames = []
    for index in range(8):
        image = np.zeros((100, 140, 3), dtype=np.uint8)
        image[30:62, 45 + index % 2 : 85 + index % 2] = [245, 20, 20]
        frames.append(image)
    assert stable_top_face_measurement(frames, "rot", 4, min_area=50) is None
    result = stable_top_face_measurement(frames, "rot", 8, min_area=50)
    assert result is not None
    assert result["source"] == "top_face"
    assert result["num_samples"] >= 5
    assert result["top_face_count"] >= 5
    assert len(result["top_face_frames"]) == result["top_face_count"]
    assert result["top_face_uv"] is not None


def test_stationary_measurements_use_each_cameras_own_onset() -> None:
    frames = []
    for index in range(12):
        image = np.zeros((100, 140, 3), dtype=np.uint8)
        x = 45 if index < 7 else 75
        image[30:62, x : x + 40] = [245, 20, 20]
        frames.append(image)
    measurements, stops = stationary_camera_measurements(
        {"left": frames, "right": frames}, "rot", {"left": 6, "right": 9}, min_area=50
    )
    assert stops == {"left": 6, "right": 9}
    assert max(measurements["left"]["frames"]) < 6
    assert max(measurements["right"]["frames"]) < 9


def test_measurement_counts_only_actual_top_face_detections(monkeypatch) -> None:
    sources = iter(["top_face"] * 4 + ["full_blob"] * 2)

    def fake_blob(_frame, _color, min_area=80):
        del min_area
        return {
            "u": 60.0,
            "v": 40.0,
            "bbox": [40, 30, 80, 62],
            "extent": 41.0,
            "source": next(sources),
        }

    monkeypatch.setattr(replay_calibration, "top_face_blob", fake_blob)
    frames = [np.zeros((100, 140, 3), dtype=np.uint8) for _ in range(6)]
    result = stable_top_face_measurement(
        frames, "rot", 6, min_samples=5, allow_full_blob=True
    )
    assert result is not None
    assert result["num_samples"] == 6
    assert result["source"] == "top_face"
    assert result["top_face_count"] == 4
    assert result["top_face_frames"] == [0, 1, 2, 3]
    assert result["top_face_uv"] == [60.0, 40.0]


def test_diagnostics_include_measurements_from_rejected_anchors() -> None:
    measurement = {
        "top_uv": [60.0, 40.0],
        "bbox": [40, 30, 80, 62],
        "frame": 3,
        "frames": [0, 1, 2, 3, 4],
        "num_samples": 5,
        "source": "top_face",
        "extent_px": 41.0,
    }
    diagnostics = {
        "7": {
            "colors": {
                "rot": {
                    "status": "rejected",
                    "reason": "keine eindeutige physische Handschließung",
                    "motion_onsets": {"cam_left_high": 12, "cam_right_high": None},
                    "measurement_stop_frames": {
                        "cam_left_high": 12,
                        "cam_right_high": 30,
                    },
                    "pixels": {"cam_left_high": measurement, "cam_right_high": None},
                }
            }
        }
    }
    samples = episode_diagnostic_pixels(diagnostics)
    assert len(samples) == 1
    assert samples[0]["episode"] == 7
    assert samples[0]["camera"] == "cam_left_high"
    assert samples[0]["motion_onset_frame"] == 12
    assert samples[0]["measurement_stop_frame"] == 12
    assert samples[0]["anchor_status"] == "rejected"
    assert samples[0]["extent_px"] == 41.0


def test_ransac_recovers_mapping_with_outlier() -> None:
    src = np.asarray([[u, v] for u in range(5) for v in range(4)], dtype=float)
    dst = np.column_stack((0.25 + src[:, 0] * 0.04, -0.2 + src[:, 1] * 0.08))
    dst[-1] = [2.0, 2.0]
    matrix, inliers = fit_homography_ransac(src, dst, threshold_m=0.005)
    assert inliers.sum() == len(src) - 1
    assert np.allclose(apply_homography(matrix, src[:-1]), dst[:-1], atol=1e-7)


def _anchors() -> list[dict]:
    anchors = []
    for episode in range(40):
        for color_index, color in enumerate(("rot", "gruen", "gelb")):
            x = 0.25 + 0.20 * ((episode % 8) / 7)
            y = -0.25 + 0.50 * (((episode // 8) * 3 + color_index) % 7) / 6
            pixels = {}
            for camera, offset in (("cam_left_high", 0.0), ("cam_right_high", 7.0)):
                pixels[camera] = {"top_uv": [(x - 0.2) * 1000 + offset,
                                                  (y + 0.3) * 800 - offset]}
            anchors.append({"episode": episode, "color": color,
                            "world_xy_m": [x, y], "pixels": pixels})
    return anchors


def test_split_is_deterministic_and_has_no_leakage() -> None:
    first = deterministic_episode_split(list(range(40)), 0.2, 17)
    second = deterministic_episode_split(list(range(40)), 0.2, 17)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert len(first[1]) == 8


def test_calibration_uses_episode_holdout_and_passes_exact_mapping() -> None:
    result = build_calibration(_anchors(), list(range(40)))
    assert result["valid"], result["failures"]
    assert set(result["fit_episodes"]).isdisjoint(result["holdout_episodes"])
    assert result["quality"]["camera_disagreement_p90_m"] < 1e-8


def test_coverage_accepts_a_spread_anchor_set() -> None:
    assert coverage_defects(np.asarray([a["world_xy_m"] for a in _anchors()])) == []


def test_coverage_flags_two_clusters_that_pass_the_span_gate() -> None:
    """Der Abnahmelauf 2026-08-22: zwei Griffwolken, 17 cm x und 34 cm y Spannweite."""
    left = np.column_stack((np.linspace(0.33, 0.50, 5), np.full(5, 0.18)))
    right = np.column_stack((np.linspace(0.33, 0.50, 5), np.full(5, -0.09)))
    points = np.vstack((left, right))
    assert np.ptp(points[:, 0]) >= 0.12 and np.ptp(points[:, 1]) >= 0.12
    assert any("geklumpt" in defect for defect in coverage_defects(points))


def test_coverage_flags_near_collinear_anchors() -> None:
    points = np.column_stack((np.linspace(0.25, 0.45, 12), np.linspace(-0.2, 0.2, 12)))
    assert any("kollinear" in defect for defect in coverage_defects(points))


def test_fingertip_window_excludes_a_hand_above_the_cube() -> None:
    """Neun von zehn Ankern des Abnahmelaufs lagen 3,5-9,9 cm über der Würfeloberseite."""
    low, high = FINGERTIP_Z_WINDOW_M
    assert low < CUBE_CENTER_Z_M < high
    assert not (low <= CUBE_TOP_Z_M + 0.035 <= high)
    assert not (low <= TABLE_TOP_Z_M - 0.02 <= high)


def test_old_cv_scale_calibration_is_rejected() -> None:
    with pytest.raises(ValueError, match="Version 4"):
        require_pick_homography_calibration({"version": 2, "method": "cv_cube_scale"})


def test_temporal_hand_assignment_requires_six_mm_and_two_mm_margin() -> None:
    assert select_unique_closing_hand([0.009, 0.005]) == 0
    assert select_unique_closing_hand([0.009, 0.008]) is None
    assert select_unique_closing_hand([0.005, None]) is None


def test_one_and_two_camera_pose_policy() -> None:
    left = {
        "world_xy_m": [0.34, 0.05],
        "top_face_world_xy_m": [0.33, 0.04],
        "source": "top_face",
        "top_face_count": 5,
    }
    right = {
        "world_xy_m": [0.35, 0.05],
        "top_face_world_xy_m": None,
        "source": "full_blob",
        "top_face_count": 0,
    }
    xy, disagreement, reason = combine_camera_estimates({"left": left, "right": right})
    assert reason == "" and np.allclose(xy, [0.345, 0.05]) and disagreement == pytest.approx(0.01)
    xy, _, reason = combine_camera_estimates({"right": right})
    assert xy is None and "Top-Face" in reason
    xy, _, reason = combine_camera_estimates({"left": left})
    assert reason == "" and np.allclose(xy, [0.33, 0.04])


def test_single_camera_requires_five_actual_top_face_detections() -> None:
    estimate = {
        "world_xy_m": [0.34, 0.05],
        "top_face_world_xy_m": [0.33, 0.04],
        "source": "top_face",
        "top_face_count": 4,
        "num_samples": 9,
    }
    xy, _, reason = combine_camera_estimates({"left": estimate})
    assert xy is None
    assert "mindestens fünf echte" in reason


def _anchor_document(source_dataset, episodes=(0, 1)) -> dict:
    pixel = {
        "top_uv": [60.0, 40.0],
        "top_face_uv": [60.0, 40.0],
        "bbox": [40, 30, 80, 62],
        "frame": 3,
        "frames": [0, 1, 2, 3, 4],
        "top_face_frames": [0, 1, 2, 3, 4],
        "num_samples": 5,
        "top_face_count": 5,
        "source": "top_face",
        "extent_px": 41.0,
    }
    return {
        "version": 4,
        "method": "sparse_pick_anchors",
        "source_dataset": str(source_dataset),
        "episodes": list(episodes),
        "anchors": [{
            "episode": episodes[0],
            "color": "rot",
            "frame": 12,
            "hand": "left",
            "world_xy_m": [0.34, 0.05],
            "pixels": {
                "cam_left_high": dict(pixel),
                "cam_right_high": dict(pixel),
            },
        }],
        "episode_diagnostics": {str(episode): {} for episode in episodes},
        "action_hashes": {str(episode): "hash" for episode in episodes},
    }


def test_anchor_document_matches_dataset_and_selection(tmp_path) -> None:
    document = _anchor_document(tmp_path)
    validate_anchor_document(document, tmp_path, [0, 1])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"version": 3}, "Version 4"),
        ({"method": "cv_cube_scale"}, "sparse_pick_anchors"),
        ({"episodes": [0, 2]}, "ausgewählt"),
    ],
)
def test_anchor_document_rejects_old_or_wrong_selection(tmp_path, change, message) -> None:
    document = {**_anchor_document(tmp_path), **change}
    with pytest.raises(ValueError, match=message):
        validate_anchor_document(document, tmp_path, [0, 1])


def test_anchor_document_rejects_foreign_or_partial_dataset(tmp_path) -> None:
    foreign = _anchor_document(tmp_path / "foreign")
    with pytest.raises(ValueError, match="fremdem Datensatz"):
        validate_anchor_document(foreign, tmp_path, [0, 1])
    partial = _anchor_document(tmp_path)
    partial["episode_diagnostics"].pop("1")
    with pytest.raises(ValueError, match="unvollständig"):
        validate_anchor_document(partial, tmp_path, [0, 1])


def test_anchor_document_rejects_malformed_anchor_fields(tmp_path) -> None:
    document = _anchor_document(tmp_path)
    document["anchors"][0]["world_xy_m"] = [float("nan"), 0.05]
    with pytest.raises(ValueError, match="world_xy_m"):
        validate_anchor_document(document, tmp_path, [0, 1])

    document = _anchor_document(tmp_path)
    document["anchors"][0]["pixels"].pop("cam_right_high")
    with pytest.raises(ValueError, match="beide Kopfkamera"):
        validate_anchor_document(document, tmp_path, [0, 1])

    document = _anchor_document(tmp_path)
    document["anchors"][0]["pixels"]["cam_left_high"]["source"] = "full_blob"
    with pytest.raises(ValueError, match="source muss top_face"):
        validate_anchor_document(document, tmp_path, [0, 1])

    document = _anchor_document(tmp_path)
    document["anchors"][0]["pixels"]["cam_left_high"]["top_uv"] = [float("inf"), 40.0]
    with pytest.raises(ValueError, match="top_uv"):
        validate_anchor_document(document, tmp_path, [0, 1])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("color", "blau", "Farbe"),
        ("hand", "middle", "Hand"),
        ("frame", -1, "frame"),
    ],
)
def test_anchor_document_rejects_invalid_anchor_identity(
    tmp_path, field, value, message
) -> None:
    document = _anchor_document(tmp_path)
    document["anchors"][0][field] = value
    with pytest.raises(ValueError, match=message):
        validate_anchor_document(document, tmp_path, [0, 1])


def test_invalid_calibration_quality_serializes_without_nonfinite_values() -> None:
    result = build_calibration([], list(range(4)))
    assert not result["valid"]
    assert result["quality"]["camera_disagreement_median_m"] is None
    json.dumps(result, allow_nan=False)


def test_calibration_dataset_and_pose_resume_binding(tmp_path) -> None:
    calibration = {"source_dataset": str(tmp_path)}
    require_calibration_dataset(calibration, tmp_path)
    with pytest.raises(ValueError, match="aktuellen Datensatz"):
        require_calibration_dataset(calibration, tmp_path / "foreign")

    pose_document = {
        "version": 4,
        "method": "stationary_top_face_homography",
        "source_dataset": str(tmp_path),
        "calibration_sha256": "abc123",
    }
    assert pose_resume_matches(pose_document, tmp_path, "abc123")
    assert not pose_resume_matches(pose_document, tmp_path, "changed")
    assert not pose_resume_matches(
        {**pose_document, "method": "cv_top_face_stereo"}, tmp_path, "abc123"
    )
