"""Pure-numpy helpers for trajectory-anchored cube replay calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from extract_block_layout import (
    CUBE_COLORS,
    HSV_WINDOWS,
    color_mask,
    largest_blob,
    rgb_to_hsv,
)


def action_sha256(actions: np.ndarray) -> str:
    canonical = np.ascontiguousarray(actions, dtype=np.float32)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def top_face_blob(rgb: np.ndarray, color: str, min_area: int = 80) -> dict | None:
    """Return the bright upper face, falling back to the full color blob."""
    mask = color_mask(rgb, color)
    full = largest_blob(mask, min_area=min_area)
    if full is None:
        return None
    x0, y0, x1, y1 = full["bbox"]
    _, _, value = rgb_to_hsv(rgb)
    crop_mask = mask[y0:y1 + 1, x0:x1 + 1]
    values = value[y0:y1 + 1, x0:x1 + 1][crop_mask]
    if values.size < min_area:
        return full
    threshold = float(np.percentile(values, 70.0))
    bright = np.zeros_like(mask)
    bright[y0:y1 + 1, x0:x1 + 1] = crop_mask & (
        value[y0:y1 + 1, x0:x1 + 1] >= threshold
    )
    top = largest_blob(bright, min_area=max(30, min_area // 2))
    if top is None:
        return full
    top["source"] = "top_face"
    tx0, ty0, tx1, ty1 = top["bbox"]
    ys, xs = np.nonzero(bright[ty0:ty1 + 1, tx0:tx1 + 1])
    if len(xs) >= 4:
        points = np.column_stack((xs + tx0, ys + ty0)).astype(float)
        centered = points - points.mean(axis=0)
        best_angle, best_area = 0.0, float("inf")
        for angle in np.linspace(0.0, np.pi / 2.0, 181, endpoint=False):
            direction = np.array([np.cos(angle), np.sin(angle)])
            normal = np.array([-direction[1], direction[0]])
            width = np.ptp(centered @ direction)
            height = np.ptp(centered @ normal)
            if width * height < best_area:
                best_angle, best_area = angle, float(width * height)
        direction = np.array([np.cos(best_angle), np.sin(best_angle)])
        half = max(4.0, float(top["extent"]) / 2.0)
        center = np.array([top["u"], top["v"]])
        top["axis_uv"] = [
            (center - half * direction).tolist(),
            (center + half * direction).tolist(),
        ]
    return top


def best_top_face_detection(frames: list[np.ndarray] | np.ndarray, color: str,
                            min_area: int = 80) -> tuple[int, np.ndarray, dict | None]:
    """Select the least occluded top-face detection for one cube color."""
    best: tuple[tuple[int, int], int, np.ndarray, dict] | None = None
    for index, frame in enumerate(frames):
        rgb = np.asarray(frame, dtype=np.uint8)[..., :3]
        blob = top_face_blob(rgb, color, min_area=min_area)
        if blob is None:
            continue
        score = (1 if blob.get("source") == "top_face" else 0, int(blob["area"]))
        candidate = (score, index, rgb, blob)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return 0, np.asarray(frames[0], dtype=np.uint8)[..., :3], None
    return best[1], best[2], best[3]


def track_color(path: Path, color: str, min_area: int = 80) -> list[list[float] | None]:
    import imageio.v2 as imageio

    track: list[list[float] | None] = []
    with imageio.get_reader(str(path), format="FFMPEG") as reader:
        for frame in reader:
            blob = largest_blob(color_mask(np.asarray(frame)[..., :3], color), min_area=min_area)
            track.append(None if blob is None else [float(blob["u"]), float(blob["v"])])
    return track


def track_colors(path: Path, min_area: int = 80) -> dict[str, list[list[float] | None]]:
    import imageio.v2 as imageio

    scale = 2
    reduced_min_area = max(20, min_area // (scale * scale))
    tracks: dict[str, list[list[float] | None]] = {color: [] for color in CUBE_COLORS}
    with imageio.get_reader(str(path), format="FFMPEG") as reader:
        for frame in reader:
            rgb = np.asarray(frame)[::scale, ::scale, :3]
            hue, saturation, value = rgb_to_hsv(rgb)
            for color in CUBE_COLORS:
                window = HSV_WINDOWS[color]
                hue_mask = np.zeros(hue.shape, dtype=bool)
                for low, high in window["h"]:
                    hue_mask |= (hue >= low) & (hue <= high)
                mask = hue_mask & (saturation >= window["s"]) & (value >= window["v"])
                y_pixels, x_pixels = np.nonzero(mask)
                if len(x_pixels) < reduced_min_area:
                    tracks[color].append(None)
                else:
                    tracks[color].append([
                        float(x_pixels.mean() * scale),
                        float(y_pixels.mean() * scale),
                    ])
            frame_count = len(tracks[CUBE_COLORS[0]])
            if frame_count >= 30 and frame_count % 30 == 0 \
                    and all(find_motion_onset(tracks[color]) is not None
                            for color in CUBE_COLORS):
                break
    return tracks


def find_motion_onset(track: list[list[float] | None], threshold_px: float = 8.0,
                      stable_frames: int = 5) -> int | None:
    valid_initial = [p for p in track[:30] if p is not None]
    if len(valid_initial) < 5:
        return None
    baseline = np.median(np.asarray(valid_initial[:10], dtype=float), axis=0)
    run = 0
    for idx, point in enumerate(track[10:], 10):
        moved = point is not None and np.linalg.norm(np.asarray(point) - baseline) >= threshold_px
        run = run + 1 if moved else 0
        if run >= stable_frames:
            return idx - stable_frames + 1
    return None


def closing_hand_scores(onset: int, spreads: np.ndarray) -> np.ndarray:
    """Hand closure before cube motion; contact usually starts after fingers close."""
    open_start = max(0, onset - 60)
    open_stop = max(open_start + 1, onset - 5)
    closed_start = max(0, onset - 18)
    closed_stop = min(len(spreads), onset + 6)
    if open_stop - open_start < 4 or closed_stop - closed_start < 3:
        return np.full(2, np.nan)
    open_level = np.nanmax(spreads[open_start:open_stop], axis=0)
    decreases = open_level - np.nanmin(spreads[closed_start:closed_stop], axis=0)
    return decreases


def match_closing_hand(onset: int, spreads: np.ndarray, min_close_m: float = 0.006,
                       margin_m: float = 0.002) -> int | None:
    decreases = closing_hand_scores(onset, spreads)
    order = np.argsort(decreases)[::-1]
    best, second = int(order[0]), int(order[1])
    if not np.isfinite(decreases[best]) or decreases[best] < min_close_m:
        return None
    if decreases[best] - decreases[second] < margin_m:
        return None
    return best


def apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = homogeneous @ np.asarray(matrix, dtype=float).T
    return mapped[:, :2] / mapped[:, 2:3]


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if len(src) < 4:
        raise ValueError("Mindestens vier Korrespondenzen sind erforderlich.")
    rows = []
    for (u, v), (x, y) in zip(src, dst):
        rows.extend((
            [-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x],
            [0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y],
        ))
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=float))
    matrix = vh[-1].reshape(3, 3)
    return matrix / matrix[2, 2]


def fit_homography_ransac(src: np.ndarray, dst: np.ndarray, threshold_m: float = 0.02,
                          iterations: int = 1000, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if len(src) < 4:
        raise ValueError("Mindestens vier Korrespondenzen sind erforderlich.")
    rng = np.random.default_rng(seed)
    best = np.zeros(len(src), dtype=bool)
    best_error = float("inf")
    for _ in range(iterations):
        sample = rng.choice(len(src), 4, replace=False)
        try:
            matrix = fit_homography(src[sample], dst[sample])
            errors = np.linalg.norm(apply_homography(matrix, src) - dst, axis=1)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        inliers = errors <= threshold_m
        score = float(errors[inliers].sum()) if inliers.any() else float("inf")
        if inliers.sum() > best.sum() or (inliers.sum() == best.sum() and score < best_error):
            best, best_error = inliers, score
    if best.sum() < 4:
        raise ValueError("RANSAC fand weniger als vier konsistente Anker.")
    return fit_homography(src[best], dst[best]), best


def homography_report(anchors: list[dict[str, Any]], camera: str,
                      min_anchors: int = 8) -> dict[str, Any]:
    usable = [a for a in anchors if camera in a.get("pixels", {})]
    if len(usable) < min_anchors:
        raise ValueError(f"{camera}: nur {len(usable)} statt {min_anchors} Anker.")
    src = np.asarray([a["pixels"][camera]["top_uv"] for a in usable], dtype=float)
    dst = np.asarray([a["world_xy_m"] for a in usable], dtype=float)
    if np.ptp(dst[:, 0]) < 0.12 or np.ptp(dst[:, 1]) < 0.12:
        raise ValueError(f"{camera}: Anker decken den Arbeitsraum nicht ausreichend ab.")
    matrix, inliers = fit_homography_ransac(src, dst)
    errors = np.linalg.norm(apply_homography(matrix, src) - dst, axis=1)
    held_out_errors = []
    episode_ids = np.asarray([int(a["episode"]) for a in usable])
    for episode in sorted(set(episode_ids.tolist())):
        train = episode_ids != episode
        test = ~train
        if train.sum() < 4:
            continue
        try:
            held_matrix, _ = fit_homography_ransac(
                src[train], dst[train], threshold_m=0.02, iterations=400, seed=episode + 31
            )
        except ValueError:
            continue
        held_out_errors.extend(
            np.linalg.norm(apply_homography(held_matrix, src[test]) - dst[test], axis=1)
        )
    validation_errors = np.asarray(held_out_errors if held_out_errors else errors[inliers])
    return {
        "pixel_to_world": matrix.tolist(),
        "num_samples": len(usable),
        "num_inliers": int(inliers.sum()),
        "median_error_m": float(np.median(validation_errors)),
        "p90_error_m": float(np.percentile(validation_errors, 90)),
        "validation": "leave_one_episode_out" if held_out_errors else "ransac_inliers",
        "samples": [
            {"episode": int(a["episode"]), "color": a["color"],
             "error_m": float(error), "inlier": bool(inside)}
            for a, error, inside in zip(usable, errors, inliers)
        ],
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
