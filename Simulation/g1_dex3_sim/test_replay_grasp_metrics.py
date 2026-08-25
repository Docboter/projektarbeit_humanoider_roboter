import numpy as np
import pytest

from replay_grasp_metrics import (
    artifact_validation_errors,
    infer_expected_first_pick,
    mark_episode_existing,
    manifest_compatibility_errors,
    summarize_grasp_trace,
)


def traces(frames: int = 8) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.zeros((frames, 3, 3), dtype=float)
    blocks[:, :, 2] = 0.915
    blocks[:, 0, 0] = 0.30
    blocks[:, 1, 0] = 0.35
    blocks[:, 2, 0] = 0.40
    tips = np.full((frames, 6, 3), 2.0, dtype=float)
    tips[:, :3] = blocks[:, 0:1] + np.array([0.01, 0.0, 0.0])
    return blocks, tips


def test_sustained_expected_cube_lift_passes() -> None:
    blocks, tips = traces()
    blocks[3:, 0, 2] += 0.021
    tips[3:, :3] = blocks[3:, 0:1] + np.array([0.01, 0.0, 0.0])
    summary = summarize_grasp_trace(
        blocks, tips, {"color": "rot", "hand": "left", "frame": 3}
    )
    assert summary["status"] == "passed"
    assert summary["grasp_success"] is True
    assert summary["blocks"]["rot"]["max_consecutive_lift_frames"] == 5
    assert summary["expected_hand_cube_min_distance_m"] == pytest.approx(0.01)


def test_short_or_wrong_cube_lift_fails_without_rejecting_render() -> None:
    blocks, tips = traces()
    blocks[2:6, 0, 2] += 0.03
    blocks[1:, 1, 2] += 0.04
    summary = summarize_grasp_trace(
        blocks, tips, {"color": "rot", "hand": "right", "frame": 2}
    )
    assert summary["status"] == "failed"
    assert summary["grasp_success"] is False
    assert summary["blocks"]["rot"]["max_consecutive_lift_frames"] == 4
    assert summary["blocks"]["gruen"]["max_consecutive_lift_frames"] == 7


def test_lift_before_expected_pick_does_not_count_as_grasp() -> None:
    blocks, tips = traces()
    blocks[1:5, 0, 2] += 0.03
    summary = summarize_grasp_trace(
        blocks, tips, {"color": "rot", "hand": "left", "frame": 5}
    )
    assert summary["status"] == "failed"
    assert summary["expected_cube_max_consecutive_lift_frames"] == 0


def test_expected_hand_distance_ignores_early_contact() -> None:
    blocks, tips = traces()
    tips[:4, :3] = blocks[:4, 0:1] + np.array([0.001, 0.0, 0.0])
    tips[4:, :3] = blocks[4:, 0:1] + np.array([0.10, 0.0, 0.0])
    summary = summarize_grasp_trace(
        blocks, tips, {"color": "rot", "hand": "left", "frame": 4}
    )
    assert summary["blocks"]["rot"]["min_left_fingertip_distance_m"] == pytest.approx(
        0.001
    )
    assert summary["expected_hand_cube_min_distance_m"] == pytest.approx(0.10)


def test_missing_expected_pick_keeps_physics_metrics() -> None:
    blocks, tips = traces(2)
    summary = summarize_grasp_trace(blocks, tips, None)
    assert summary["status"] == "unavailable"
    assert summary["reason"] == "expected_first_pick_missing"
    assert summary["min_hand_cube_distance_m"] == pytest.approx(0.01)


def test_pick_after_short_render_is_unavailable_not_failed() -> None:
    blocks, tips = traces(3)
    summary = summarize_grasp_trace(
        blocks, tips, {"color": "rot", "hand": "left", "frame": 30}
    )
    assert summary["status"] == "unavailable"
    assert summary["reason"] == "expected_first_pick_not_observed"


def test_summary_counts_final_post_action_sample() -> None:
    # Acht Actions besitzen neun Physik-Samples: vor jeder Action und eines danach.
    blocks, tips = traces(9)
    summary = summarize_grasp_trace(blocks, tips, None)
    assert summary["frames_observed"] == 9


def test_first_pick_prefers_earliest_valid_pose_diagnostic() -> None:
    pose_episode = {
        "pick_anchor_diagnostics": [
            {"color": "gruen", "hand": "right", "frame": 40},
            {"color": "rot", "hand": "left", "frame": 20},
        ]
    }
    calibration = {
        "anchors": [
            {"episode": 7, "color": "gelb", "hand": "right", "frame": 30},
            {"episode": 8, "color": "rot", "hand": "right", "frame": 1},
        ]
    }
    assert infer_expected_first_pick(pose_episode, calibration, 7) == {
        "color": "rot",
        "hand": "left",
        "frame": 20,
        "source": "cube_poses.pick_anchor_diagnostics",
    }


def test_first_pick_falls_back_to_accepted_calibration_diagnostic() -> None:
    calibration = {
        "episode_diagnostics": {
            "3": {
                "colors": {
                    "gelb": {
                        "status": "accepted",
                        "matched_hand": "right",
                        "motion_onsets": {"cam_left_high": 31, "cam_right_high": 33},
                    }
                }
            }
        }
    }
    pick = infer_expected_first_pick({}, calibration, 3)
    assert pick == {
        "color": "gelb",
        "hand": "right",
        "frame": 32,
        "source": "geometry_calibration.episode_diagnostics",
    }


def test_trace_shape_is_checked() -> None:
    with pytest.raises(ValueError, match="T,6,3"):
        summarize_grasp_trace(np.zeros((2, 3, 3)), np.zeros((2, 2, 3)), None)


def test_manifest_compatibility_requires_complete_v2_identity() -> None:
    expected = {
        "version": 2,
        "source_dataset": "/data/source",
        "calibration_sha256": "cal",
        "poses_sha256": "poses",
    }
    compatible = {
        **expected,
        "episodes": {
            "1": {
                "grasp_validation": {
                    "status": "failed",
                    "grasp_success": False,
                    "reason": "expected_cube_not_lifted_long_enough",
                }
            }
        },
    }
    assert manifest_compatibility_errors(compatible, expected) == []
    incompatible = {
        **compatible,
        "version": 1,
        "poses_sha256": "old",
        "episodes": {"1": {"status": "ok"}},
    }
    errors = manifest_compatibility_errors(incompatible, expected)
    assert any("version=" in error for error in errors)
    assert any("poses_sha256=" in error for error in errors)
    assert any("ohne v2-Griffvalidierung" in error for error in errors)


def test_artifact_validation_requires_matching_v4_pair() -> None:
    pose = {
        "version": 4,
        "method": "stationary_top_face_homography",
        "source_dataset": "/data/source",
        "calibration_sha256": "abc",
    }
    calibration = {
        "version": 4,
        "method": "pick_anchored_homography",
        "source_dataset": "/data/source",
    }
    assert artifact_validation_errors(pose, calibration, "/data/source", "abc") == []
    pose["method"] = "cv_cube_scale"
    calibration["source_dataset"] = "/data/other"
    errors = artifact_validation_errors(pose, calibration, "/data/source", "wrong")
    assert any("Würfelpose: method=" in error for error in errors)
    assert any("Kalibrierung: source_dataset=" in error for error in errors)
    assert any("calibration_sha256=" in error for error in errors)


def test_resume_preserves_complete_episode_record() -> None:
    record = {
        "status": "ok",
        "render_status": "ok",
        "task_index": 7,
        "renderer_projection": {"head": {"median_px": 2.0}},
        "cube_poses": [{"name": "block_0"}],
        "grasp_validation": {"status": "passed", "grasp_success": True},
    }
    resumed = mark_episode_existing(record)
    assert resumed == {**record, "existing": True, "render_status": "existing"}
    assert record["render_status"] == "ok"
