# TL;DR: Hilfsfunktionen für sparse, zustandsbasierte Greif-Unterstützung bei Würfelposenschätzung.
"""Pure helpers for sparse, state-based grasp support in cube pose reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np

HAND_SLICES = {"left": slice(14, 21), "right": slice(21, 28)}


def closure_candidates(
    states: np.ndarray,
    hand: str,
    *,
    min_frames: int = 3,
    max_candidates: int = 4,
    min_joint_motion_rad: float = 0.015,
) -> list[dict[str, Any]]:
    """Find a few coordinated finger-motion intervals without assuming joint signs."""
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 28:
        raise ValueError(f"states must have shape (T, 28), got {values.shape}")
    fingers = values[:, HAND_SLICES[hand]]
    if len(fingers) < min_frames + 2:
        return []
    smooth = fingers.copy()
    if len(fingers) >= 3:
        smooth[1:-1] = np.median(
            np.stack((fingers[:-2], fingers[1:-1], fingers[2:])), axis=0
        )
    velocity = np.linalg.norm(np.diff(smooth, axis=0), axis=1)
    median = float(np.median(velocity))
    mad = float(np.median(np.abs(velocity - median)))
    threshold = max(0.01, median + 3.0 * max(mad, 1e-4))
    active = velocity >= threshold

    runs: list[tuple[int, int]] = []
    start: int | None = None
    last = -10
    for index, is_active in enumerate(active):
        if is_active:
            if start is None or index - last > 2:
                if start is not None:
                    runs.append((start, last + 1))
                start = index
            last = index
    if start is not None:
        runs.append((start, last + 1))

    candidates = []
    for run_start, run_stop in runs:
        before = max(0, run_start - 2)
        after = min(len(fingers) - 1, max(run_stop + 2, run_start + min_frames))
        delta = np.abs(smooth[after] - smooth[before])
        coordinated = int(np.count_nonzero(delta >= min_joint_motion_rad))
        if coordinated < 2 or after - before < min_frames:
            continue
        candidates.append(
            {
                "start_frame": int(before),
                "end_frame": int(after),
                "coordinated_joints": coordinated,
                "joint_motion_rad": float(np.linalg.norm(delta)),
            }
        )
    candidates.sort(key=lambda item: item["start_frame"])
    return candidates[:max_candidates]


def fingertip_measurement(tips: np.ndarray) -> tuple[float, np.ndarray]:
    """Return mean pairwise fingertip spread and centroid for one three-tip hand."""
    points = np.asarray(tips, dtype=float)
    if points.shape != (3, 3):
        raise ValueError(f"tips must have shape (3, 3), got {points.shape}")
    spread = float(
        np.linalg.norm(points[[0, 0, 1]] - points[[1, 2, 2]], axis=1).mean()
    )
    return spread, points.mean(axis=0)


def is_closing_spread(
    before_m: float, after_m: float, min_decrease_m: float = 0.006
) -> bool:
    """Return true only when the physical fingertip opening decreased enough."""
    return float(before_m) - float(after_m) >= float(min_decrease_m)


def apply_grasp_support(
    blocks: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    max_distance_m: float = 0.08,
    uniqueness_margin_m: float = 0.03,
    hand_weight: float = 0.75,
) -> int:
    """Associate sparse grasp events to CV cubes and fuse accepted XY positions."""
    usable = [block for block in blocks if block.get("status") == "ok"]
    for block in usable:
        block.setdefault("cv_position_m", list(block["position_m"]))
        block["position_m"] = list(block["cv_position_m"])
        block["position_source"] = "cv"
        block["grasp_support"] = {
            "accepted": False,
            "reason": "kein eindeutiger Griff",
        }
    used_names: set[str] = set()
    accepted = 0
    for event in sorted(events, key=lambda item: int(item["grasp_frame"])):
        centroid = np.asarray(event["fingertip_centroid_m"], dtype=float)
        available = [block for block in usable if block["name"] not in used_names]
        if not available:
            break
        distances = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(block["cv_position_m"], dtype=float)[:2] - centroid[:2]
                )
                for block in available
            ]
        )
        order = np.argsort(distances)
        nearest = int(order[0])
        distance = float(distances[nearest])
        second = float(distances[order[1]]) if len(order) > 1 else float("inf")
        block = available[nearest]
        support = {
            **event,
            "accepted": False,
            "cv_distance_m": distance,
            "reason": "",
        }
        if distance > max_distance_m:
            support["reason"] = "Hand mehr als 8 cm von CV-Position entfernt"
            block["grasp_support"] = support
            continue
        if second - distance < uniqueness_margin_m:
            support["reason"] = "Handposition zwischen mehreren Würfeln mehrdeutig"
            block["grasp_support"] = support
            continue
        cv = np.asarray(block["cv_position_m"], dtype=float)
        fused = cv.copy()
        fused[:2] = hand_weight * centroid[:2] + (1.0 - hand_weight) * cv[:2]
        block["position_m"] = [float(value) for value in fused]
        block["position_source"] = "cv_grasp_fused"
        support["accepted"] = True
        support["reason"] = ""
        block["grasp_support"] = support
        used_names.add(block["name"])
        accepted += 1
    return accepted
