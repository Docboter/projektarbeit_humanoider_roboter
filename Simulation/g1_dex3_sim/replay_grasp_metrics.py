"""Pure helpers for physical grasp validation during dataset replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


BLOCK_COLORS = ("rot", "gruen", "gelb")
VALID_HANDS = ("left", "right")


def artifact_validation_errors(
    pose_document: dict[str, Any],
    calibration: dict[str, Any],
    source_dataset: str | Path,
    calibration_sha256: str,
) -> list[str]:
    """Validate the exact v4 artifact pair accepted by the renderer."""
    errors = []
    expected = (
        ("Würfelpose", pose_document, 4, "stationary_top_face_homography"),
        ("Kalibrierung", calibration, 4, "pick_anchored_homography"),
    )
    expected_source = Path(source_dataset).resolve()
    for label, document, version, method in expected:
        if document.get("version") != version:
            errors.append(f"{label}: version={document.get('version')!r}, erwartet {version}")
        if document.get("method") != method:
            errors.append(f"{label}: method={document.get('method')!r}, erwartet {method!r}")
        recorded_source = document.get("source_dataset")
        if not isinstance(recorded_source, str):
            errors.append(f"{label}: source_dataset fehlt")
        elif Path(recorded_source).resolve() != expected_source:
            errors.append(
                f"{label}: source_dataset={recorded_source!r}, erwartet "
                f"{str(source_dataset)!r}"
            )
    recorded_hash = pose_document.get("calibration_sha256")
    if recorded_hash != calibration_sha256:
        errors.append(
            f"Würfelpose: calibration_sha256={recorded_hash!r}, "
            f"erwartet {calibration_sha256!r}"
        )
    return errors


def mark_episode_existing(record: dict[str, Any]) -> dict[str, Any]:
    """Preserve a complete v2 episode record while marking it as resumed."""
    resumed = dict(record)
    resumed.update(existing=True, render_status="existing")
    return resumed


def manifest_compatibility_errors(
    existing: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    """Describe why a persisted replay manifest cannot be safely resumed."""
    errors = []
    for field in ("version", "source_dataset", "calibration_sha256", "poses_sha256"):
        if existing.get(field) != expected.get(field):
            errors.append(
                f"{field}={existing.get(field)!r}, erwartet {expected.get(field)!r}"
            )
    episodes = existing.get("episodes")
    if not isinstance(episodes, dict):
        errors.append("episodes fehlt oder ist kein Objekt")
    else:
        incomplete = []
        for episode, record in episodes.items():
            validation = record.get("grasp_validation") if isinstance(record, dict) else None
            if not isinstance(validation, dict) or not {
                "status", "grasp_success", "reason"
            }.issubset(validation):
                incomplete.append(str(episode))
        if incomplete:
            errors.append(
                "Episoden ohne v2-Griffvalidierung: " + ", ".join(sorted(incomplete))
            )
    return errors


def _candidate_pick(record: dict[str, Any], source: str) -> dict[str, Any] | None:
    color = record.get("color")
    hand = record.get("hand") or record.get("matched_hand")
    frame = record.get("frame")
    if frame is None:
        event = record.get("pick_event", {})
        frame = event.get("end_frame", event.get("start_frame"))
    if color not in BLOCK_COLORS or hand not in VALID_HANDS or frame is None:
        return None
    try:
        frame = int(frame)
    except (TypeError, ValueError):
        return None
    if frame < 0:
        return None
    return {"color": color, "hand": hand, "frame": frame, "source": source}


def infer_expected_first_pick(
    pose_episode: dict[str, Any], calibration: dict[str, Any], episode: int
) -> dict[str, Any] | None:
    """Return the earliest valid pick diagnostic without using it as a cube pose."""
    candidates = []
    for record in pose_episode.get("pick_anchor_diagnostics", []):
        candidate = _candidate_pick(record, "cube_poses.pick_anchor_diagnostics")
        if candidate is not None:
            candidates.append(candidate)
    for record in calibration.get("anchors", []):
        try:
            same_episode = int(record.get("episode", -1)) == int(episode)
        except (TypeError, ValueError):
            same_episode = False
        if same_episode:
            candidate = _candidate_pick(record, "geometry_calibration.anchors")
            if candidate is not None:
                candidates.append(candidate)

    # A calibration diagnostic can retain the accepted association even when the
    # compact anchor list was removed for transport. Use the median camera onset.
    colors = (
        calibration.get("episode_diagnostics", {})
        .get(str(episode), {})
        .get("colors", {})
    )
    for color, record in colors.items():
        if record.get("status") != "accepted":
            continue
        onsets = [value for value in record.get("motion_onsets", {}).values()
                  if value is not None]
        if not onsets:
            continue
        candidate = _candidate_pick(
            {
                "color": color,
                "matched_hand": record.get("matched_hand"),
                "frame": int(round(float(np.median(onsets)))),
            },
            "geometry_calibration.episode_diagnostics",
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None
    first_frame = min(item["frame"] for item in candidates)
    first = [item for item in candidates if item["frame"] == first_frame]
    identities = {(item["color"], item["hand"]) for item in first}
    if len(identities) != 1:
        return None
    # Prefer the pose document because it is the exact artifact used for this replay.
    return min(first, key=lambda item: item["source"] != "cube_poses.pick_anchor_diagnostics")


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def summarize_grasp_trace(
    block_positions_m: np.ndarray,
    fingertip_positions_m: np.ndarray,
    expected_pick: dict[str, Any] | None,
    *,
    lift_threshold_m: float = 0.02,
    success_frames: int = 5,
) -> dict[str, Any]:
    """Summarize read-only PhysX traces and validate the expected first pick."""
    blocks = np.asarray(block_positions_m, dtype=float)
    tips = np.asarray(fingertip_positions_m, dtype=float)
    if blocks.ndim != 3 or blocks.shape[1:] != (3, 3):
        raise ValueError(f"block_positions_m muss (T,3,3) sein, erhalten {blocks.shape}")
    if tips.ndim != 3 or tips.shape[1:] != (6, 3):
        raise ValueError(f"fingertip_positions_m muss (T,6,3) sein, erhalten {tips.shape}")
    if len(blocks) == 0 or len(tips) != len(blocks):
        raise ValueError("Block- und Fingertraces müssen gleich lang und nicht leer sein")
    if not np.all(np.isfinite(blocks)) or not np.all(np.isfinite(tips)):
        raise ValueError("Block- und Fingertraces müssen endlich sein")

    distances = np.linalg.norm(
        tips[:, :, None, :] - blocks[:, None, :, :], axis=-1
    )
    lifts = blocks[:, :, 2] - blocks[0, :, 2]
    block_metrics = {}
    for index, color in enumerate(BLOCK_COLORS):
        block_metrics[color] = {
            "block_index": index,
            "min_fingertip_distance_m": float(np.min(distances[:, :, index])),
            "min_left_fingertip_distance_m": float(np.min(distances[:, :3, index])),
            "min_right_fingertip_distance_m": float(np.min(distances[:, 3:, index])),
            "max_lift_m": float(max(0.0, np.max(lifts[:, index]))),
            "max_consecutive_lift_frames": _longest_true_run(
                lifts[:, index] >= lift_threshold_m
            ),
        }

    result: dict[str, Any] = {
        "status": "unavailable",
        "grasp_success": False,
        "reason": "expected_first_pick_missing",
        "lift_threshold_m": float(lift_threshold_m),
        "required_consecutive_frames": int(success_frames),
        "frames_observed": int(len(blocks)),
        "min_hand_cube_distance_m": float(np.min(distances)),
        "blocks": block_metrics,
        "expected_first_pick": expected_pick,
    }
    if expected_pick is None:
        return result
    color = expected_pick.get("color")
    hand = expected_pick.get("hand")
    if color not in block_metrics or hand not in VALID_HANDS:
        result["reason"] = "expected_first_pick_invalid"
        return result
    expected_frame = int(expected_pick.get("frame", -1))
    if expected_frame < 0:
        result["reason"] = "expected_first_pick_invalid"
        return result
    if expected_frame >= len(blocks):
        result["reason"] = "expected_first_pick_not_observed"
        return result

    metric = block_metrics[color]
    color_index = BLOCK_COLORS.index(color)
    expected_lifts = lifts[expected_frame:, color_index]
    run = _longest_true_run(expected_lifts >= lift_threshold_m)
    success = run >= int(success_frames)
    hand_slice = slice(0, 3) if hand == "left" else slice(3, 6)
    expected_hand_distance = float(
        np.min(distances[expected_frame:, hand_slice, color_index])
    )
    result.update(
        status="passed" if success else "failed",
        grasp_success=success,
        reason="lift_sustained" if success else "expected_cube_not_lifted_long_enough",
        expected_cube_max_lift_m=float(max(0.0, np.max(expected_lifts))),
        expected_cube_max_consecutive_lift_frames=run,
        expected_hand_cube_min_distance_m=expected_hand_distance,
    )
    return result
