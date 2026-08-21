"""Pure-numpy helpers for CV-based cube replay calibration."""

from __future__ import annotations

import hashlib

import numpy as np

from extract_block_layout import color_mask, largest_blob, rgb_to_hsv


def action_sha256(actions: np.ndarray) -> str:
    """Return a stable hash of the unchanged float32 action tensor."""
    canonical = np.ascontiguousarray(actions, dtype=np.float32)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def top_face_blob(rgb: np.ndarray, color: str, min_area: int = 80) -> dict | None:
    """Return the bright upper cube face, falling back to the full color blob."""
    mask = color_mask(rgb, color)
    full = largest_blob(mask, min_area=min_area)
    if full is None:
        return None
    x0, y0, x1, y1 = full["bbox"]
    _, _, value = rgb_to_hsv(rgb)
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    values = value[y0 : y1 + 1, x0 : x1 + 1][crop_mask]
    if values.size < min_area:
        full["source"] = "full_blob"
        return full
    threshold = float(np.percentile(values, 70.0))
    bright = np.zeros_like(mask)
    bright[y0 : y1 + 1, x0 : x1 + 1] = crop_mask & (
        value[y0 : y1 + 1, x0 : x1 + 1] >= threshold
    )
    top = largest_blob(bright, min_area=max(30, min_area // 2))
    if top is None:
        full["source"] = "full_blob"
        return full
    top["source"] = "top_face"
    tx0, ty0, tx1, ty1 = top["bbox"]
    _ys, xs = np.nonzero(bright[ty0 : ty1 + 1, tx0 : tx1 + 1])
    if len(xs) >= 4:
        points = np.column_stack((xs + tx0, _ys + ty0)).astype(float)
        centered = points - points.mean(axis=0)
        best_angle, best_area = 0.0, float("inf")
        for angle in np.linspace(0.0, np.pi / 2.0, 181, endpoint=False):
            direction = np.array([np.cos(angle), np.sin(angle)])
            normal = np.array([-direction[1], direction[0]])
            area = float(np.ptp(centered @ direction) * np.ptp(centered @ normal))
            if area < best_area:
                best_angle, best_area = angle, area
        direction = np.array([np.cos(best_angle), np.sin(best_angle)])
        half = max(4.0, float(top["extent"]) / 2.0)
        center = np.array([top["u"], top["v"]])
        top["axis_uv"] = [
            (center - half * direction).tolist(),
            (center + half * direction).tolist(),
        ]
    return top


def best_top_face_detection(
    frames: list[np.ndarray] | np.ndarray, color: str, min_area: int = 80
) -> tuple[int, np.ndarray, dict | None]:
    """Select the largest, least occluded top-face detection in early frames."""
    candidates: list[tuple[int, np.ndarray, dict]] = []
    for index, frame in enumerate(frames):
        rgb = np.asarray(frame, dtype=np.uint8)[..., :3]
        blob = top_face_blob(rgb, color, min_area=min_area)
        if blob is None:
            continue
        candidates.append((index, rgb, blob))
    if not candidates:
        return 0, np.asarray(frames[0], dtype=np.uint8)[..., :3], None
    centers = np.asarray([[item[2]["u"], item[2]["v"]] for item in candidates])
    median = np.median(centers, axis=0)
    stable = [
        item
        for item, center in zip(candidates, centers)
        if np.linalg.norm(center - median) <= 12.0
    ]
    pool = stable if len(stable) >= 3 else candidates
    best = max(
        pool,
        key=lambda item: (
            1 if item[2].get("source") == "top_face" else 0,
            int(item[2]["area"]),
        ),
    )
    return best
