"""Pure-numpy helpers for pick-anchored cube replay calibration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from extract_block_layout import CUBE_COLORS, HSV_WINDOWS, color_mask, largest_blob, rgb_to_hsv


def action_sha256(actions: np.ndarray) -> str:
    """Return a stable hash of the unchanged float32 action tensor."""
    canonical = np.ascontiguousarray(actions, dtype=np.float32)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def touches_image_border(blob: dict, shape: tuple[int, ...], margin: int = 2) -> bool:
    x0, y0, x1, y1 = (int(value) for value in blob["bbox"])
    height, width = shape[:2]
    return x0 <= margin or y0 <= margin or x1 >= width - 1 - margin or y1 >= height - 1 - margin


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


def track_colors(path, min_area: int = 80) -> dict[str, list[list[float] | None]]:
    """Track the color centroids cheaply; this is used only for motion timing."""
    import imageio.v2 as imageio

    scale = 2
    reduced_min_area = max(20, min_area // (scale * scale))
    tracks = {color: [] for color in CUBE_COLORS}
    with imageio.get_reader(str(path), format="FFMPEG") as reader:
        for frame in reader:
            rgb = np.asarray(frame, dtype=np.uint8)[::scale, ::scale, :3]
            hue, saturation, value = rgb_to_hsv(rgb)
            for color in CUBE_COLORS:
                window = HSV_WINDOWS[color]
                hue_mask = np.zeros(hue.shape, dtype=bool)
                for low, high in window["h"]:
                    hue_mask |= (hue >= low) & (hue <= high)
                mask = hue_mask & (saturation >= window["s"]) & (value >= window["v"])
                ys, xs = np.nonzero(mask)
                tracks[color].append(
                    None
                    if len(xs) < reduced_min_area
                    else [float(xs.mean() * scale), float(ys.mean() * scale)]
                )
    return tracks


def find_motion_onset(
    track: list[list[float] | None], threshold_px: float = 8.0, stable_frames: int = 5
) -> int | None:
    """Return the first run that stays displaced from the first ten valid samples."""
    valid = [(index, point) for index, point in enumerate(track[:30]) if point is not None]
    if len(valid) < 10:
        return None
    baseline = np.median(np.asarray([point for _, point in valid[:10]], dtype=float), axis=0)
    search_start = valid[9][0] + 1
    run = 0
    for index, point in enumerate(track[search_start:], search_start):
        moved = point is not None and np.linalg.norm(np.asarray(point) - baseline) >= threshold_px
        run = run + 1 if moved else 0
        if run >= stable_frames:
            return index - stable_frames + 1
    return None


def stable_top_face_measurement(
    frames: list[np.ndarray], color: str, stop: int, min_area: int = 80,
    min_samples: int = 5, allow_full_blob: bool = False,
) -> dict | None:
    """Robust top-face measurement from stationary frames strictly before motion."""
    measurements = []
    for frame_index, frame in enumerate(frames[: max(0, stop)]):
        blob = top_face_blob(frame, color, min_area=min_area)
        if (blob is None or (blob.get("source") != "top_face" and not allow_full_blob)
                or touches_image_border(blob, frame.shape)):
            continue
        measurements.append((frame_index, blob))
    if len(measurements) < min_samples:
        return None
    centers = np.asarray([[blob["u"], blob["v"]] for _, blob in measurements], dtype=float)
    median = np.median(centers, axis=0)
    keep = np.linalg.norm(centers - median, axis=1) <= 12.0
    if int(keep.sum()) < min_samples:
        return None
    accepted = [item for item, valid in zip(measurements, keep) if valid]
    representative = min(accepted, key=lambda item: np.linalg.norm(
        np.asarray([item[1]["u"], item[1]["v"]]) - median
    ))
    frame_index, blob = representative
    top_faces = [
        item for item in accepted if item[1].get("source") == "top_face"
    ]
    top_face_centers = np.asarray(
        [[item_blob["u"], item_blob["v"]] for _, item_blob in top_faces], dtype=float
    )
    return {
        "top_uv": [float(value) for value in median],
        "bbox": [int(value) for value in blob["bbox"]],
        "frame": int(frame_index),
        "frames": [int(item[0]) for item in accepted],
        "num_samples": len(accepted),
        "top_face_count": len(top_faces),
        "top_face_uv": (
            [float(value) for value in np.median(top_face_centers, axis=0)]
            if top_faces else None
        ),
        "top_face_frames": [int(item[0]) for item in top_faces],
        "source": blob.get("source", "full_blob"),
        "extent_px": float(blob["extent"]),
    }


def stationary_camera_measurements(
    frames_by_camera: dict[str, list[np.ndarray]],
    color: str,
    onsets: dict[str, int | None],
    *,
    min_area: int = 80,
    diagnostic_fallback_frames: int = 30,
) -> tuple[dict[str, dict | None], dict[str, int]]:
    """Measure each camera strictly before its own onset.

    A missing onset uses a bounded early window for diagnostics only. Callers must
    still reject such a sample as a calibration anchor.
    """
    measurements = {}
    stops = {}
    for camera, frames in frames_by_camera.items():
        onset = onsets.get(camera)
        stop = int(onset) if onset is not None else min(
            len(frames), diagnostic_fallback_frames
        )
        stops[camera] = stop
        measurements[camera] = stable_top_face_measurement(
            frames, color, stop, min_area=min_area
        )
    return measurements, stops


def episode_diagnostic_pixels(
    episode_diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten every recorded camera/color detection, including rejected anchors."""
    samples = []
    for episode_text, episode_record in episode_diagnostics.items():
        for color, color_record in episode_record.get("colors", {}).items():
            for camera, pixel in color_record.get("pixels", {}).items():
                if pixel is None:
                    continue
                samples.append({
                    "episode": int(episode_text),
                    "camera": camera,
                    "color": color,
                    "motion_onset_frame": color_record.get("motion_onsets", {}).get(camera),
                    "measurement_stop_frame": color_record.get(
                        "measurement_stop_frames", {}
                    ).get(camera),
                    "anchor_status": color_record.get("status", "rejected"),
                    "anchor_rejection_reason": color_record.get("reason", ""),
                    **pixel,
                })
    return samples


def deterministic_episode_split(
    episodes: list[int], holdout_ratio: float = 0.2, seed: int = 17
) -> tuple[list[int], list[int]]:
    unique = np.asarray(sorted(set(int(ep) for ep in episodes)), dtype=int)
    if len(unique) < 2:
        return unique.tolist(), []
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    count = max(1, int(round(len(unique) * holdout_ratio)))
    count = min(count, len(unique) - 1)
    return sorted(shuffled[count:].tolist()), sorted(shuffled[:count].tolist())


def apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
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
            [-u, -v, -1, 0, 0, 0, x * u, x * v, x],
            [0, 0, 0, -u, -v, -1, y * u, y * v, y],
        ))
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=float))
    matrix = vh[-1].reshape(3, 3)
    if abs(matrix[2, 2]) < 1e-12:
        raise ValueError("Degenerierte Homographie.")
    return matrix / matrix[2, 2]


def fit_homography_ransac(
    src: np.ndarray, dst: np.ndarray, threshold_m: float = 0.02,
    iterations: int = 1000, seed: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
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
        inliers = np.isfinite(errors) & (errors <= threshold_m)
        score = float(errors[inliers].sum()) if inliers.any() else float("inf")
        if inliers.sum() > best.sum() or (inliers.sum() == best.sum() and score < best_error):
            best, best_error = inliers, score
    if best.sum() < 4:
        raise ValueError("RANSAC fand weniger als vier konsistente Anker.")
    return fit_homography(src[best], dst[best]), best


def build_calibration(
    anchors: list[dict[str, Any]], episodes: list[int], *, holdout_ratio: float = 0.2,
    seed: int = 17, cameras: tuple[str, ...] = ("cam_left_high", "cam_right_high"),
) -> dict[str, Any]:
    """Fit on complete episodes and evaluate on untouched holdout episodes."""
    fit_eps, holdout_eps = deterministic_episode_split(episodes, holdout_ratio, seed)
    fit_set, holdout_set = set(fit_eps), set(holdout_eps)
    fit_anchors = [a for a in anchors if int(a["episode"]) in fit_set]
    holdout = [a for a in anchors if int(a["episode"]) in holdout_set]
    failures = []
    if len(fit_anchors) < 24:
        failures.append(f"nur {len(fit_anchors)} statt 24 Fit-Anker")
    if len({int(a['episode']) for a in fit_anchors}) < 8:
        failures.append("weniger als acht Fit-Episoden")
    if len(holdout) < 6:
        failures.append(f"nur {len(holdout)} statt sechs Holdout-Anker")
    if len({int(a['episode']) for a in holdout}) < 3:
        failures.append("weniger als drei Holdout-Episoden")
    reports = {}
    for camera in cameras:
        usable_fit = [a for a in fit_anchors if camera in a.get("pixels", {})]
        usable_holdout = [a for a in holdout if camera in a.get("pixels", {})]
        if len(usable_fit) < 4:
            failures.append(f"{camera}: weniger als vier Fit-Anker")
            continue
        src = np.asarray([a["pixels"][camera]["top_uv"] for a in usable_fit])
        dst = np.asarray([a["world_xy_m"] for a in usable_fit])
        if np.ptp(dst[:, 0]) < 0.12 or np.ptp(dst[:, 1]) < 0.12:
            failures.append(f"{camera}: Arbeitsraumabdeckung unter 12 cm")
        try:
            matrix, inliers = fit_homography_ransac(src, dst, seed=seed)
        except ValueError as exc:
            failures.append(f"{camera}: {exc}")
            continue
        held_errors = np.asarray([
            np.linalg.norm(apply_homography(matrix, [a["pixels"][camera]["top_uv"]])[0]
                           - np.asarray(a["world_xy_m"])) for a in usable_holdout
        ])
        median = _finite_statistic(held_errors, np.median)
        p90 = _finite_statistic(held_errors, lambda values: np.percentile(values, 90))
        if median is None or p90 is None:
            failures.append(f"{camera}: keine auswertbaren Holdout-Anker")
        elif median > 0.015 or p90 > 0.03:
            failures.append(f"{camera}: Holdout Median/P90 {median:.3f}/{p90:.3f} m")
        reports[camera] = {
            "pixel_to_table_xy": matrix.tolist(), "num_fit_samples": len(usable_fit),
            "num_inliers": int(inliers.sum()), "num_holdout_samples": len(usable_holdout),
            "holdout_median_error_m": median, "holdout_p90_error_m": p90,
        }
    stereo_diffs = []
    if all(camera in reports for camera in cameras):
        for anchor in holdout:
            if all(camera in anchor.get("pixels", {}) for camera in cameras):
                estimates = [apply_homography(reports[camera]["pixel_to_table_xy"],
                    [anchor["pixels"][camera]["top_uv"]])[0] for camera in cameras]
                stereo_diffs.append(float(np.linalg.norm(estimates[0] - estimates[1])))
    stereo_median = _finite_statistic(stereo_diffs, np.median)
    stereo_p90 = _finite_statistic(stereo_diffs, lambda values: np.percentile(values, 90))
    if stereo_median is None or stereo_p90 is None:
        failures.append("keine auswertbare Kameradifferenz im Holdout")
    elif stereo_median > 0.02 or stereo_p90 > 0.03:
        failures.append(f"Kameradifferenz Median/P90 {stereo_median:.3f}/{stereo_p90:.3f} m")
    return {
        "valid": not failures, "failures": failures, "fit_episodes": fit_eps,
        "holdout_episodes": holdout_eps, "cameras": reports,
        "quality": {"num_fit_anchors": len(fit_anchors), "num_holdout_anchors": len(holdout),
                    "camera_disagreement_median_m": stereo_median,
                    "camera_disagreement_p90_m": stereo_p90},
    }


def _finite_statistic(values: Any, statistic) -> float | None:
    if len(values) == 0:
        return None
    result = float(statistic(values))
    return result if np.isfinite(result) else None


def require_pick_homography_calibration(document: dict[str, Any]) -> None:
    if (document.get("version") != 4
            or document.get("method") != "pick_anchored_homography"
            or not document.get("valid")):
        raise ValueError(
            "Kalibrierung inkompatibel oder ungültig; erwartet Version 4 mit "
            "method=pick_anchored_homography."
        )


def require_calibration_dataset(document: dict[str, Any], source_dataset: Path) -> None:
    recorded_source = document.get("source_dataset")
    if (
        not isinstance(recorded_source, str)
        or Path(recorded_source).resolve() != source_dataset.resolve()
    ):
        raise ValueError(
            f"Kalibrierung gehört nicht zum aktuellen Datensatz {str(source_dataset)!r}."
        )


def pose_resume_matches(
    document: dict[str, Any], source_dataset: Path, calibration_sha256: str
) -> bool:
    recorded_source = document.get("source_dataset")
    return (
        document.get("version") == 4
        and document.get("method") == "stationary_top_face_homography"
        and isinstance(recorded_source, str)
        and Path(recorded_source).resolve() == source_dataset.resolve()
        and document.get("calibration_sha256") == calibration_sha256
    )


def validate_anchor_document(
    document: dict[str, Any], source_dataset: Path, episodes: list[int]
) -> None:
    """Reject stale, foreign, partial, or selection-mismatched anchor collections."""
    if document.get("version") != 4 or document.get("method") != "sparse_pick_anchors":
        raise ValueError(
            "Anchor-Dokument inkompatibel; erwartet Version 4 mit "
            "method=sparse_pick_anchors."
        )
    recorded_source = document.get("source_dataset")
    if not isinstance(recorded_source, str) or not recorded_source:
        raise ValueError("Anchor-Dokument enthält kein source_dataset.")
    if Path(recorded_source).resolve() != source_dataset.resolve():
        raise ValueError(
            f"Anchor-Dokument gehört zu fremdem Datensatz: {recorded_source!r}; "
            f"erwartet {str(source_dataset)!r}."
        )
    expected = sorted(set(int(episode) for episode in episodes))
    try:
        declared = sorted(set(int(episode) for episode in document.get("episodes", [])))
        anchor_episodes = {int(anchor["episode"]) for anchor in document.get("anchors", [])}
        diagnostic_episodes = {
            int(episode) for episode in document.get("episode_diagnostics", {})
        }
        hash_episodes = {int(episode) for episode in document.get("action_hashes", {})}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Anchor-Dokument enthält ungültige Episodenangaben.") from exc
    if declared != expected:
        raise ValueError(
            f"Anchor-Dokument deckt Episoden {declared} ab; ausgewählt sind {expected}."
        )
    unexpected_anchors = sorted(anchor_episodes - set(expected))
    if unexpected_anchors:
        raise ValueError(
            f"Anchor-Dokument enthält Anker außerhalb der Auswahl: {unexpected_anchors}."
        )
    if diagnostic_episodes != set(expected) or hash_episodes != set(expected):
        raise ValueError(
            "Anchor-Dokument ist unvollständig: Diagnose- und Action-Hash-Episoden "
            "müssen exakt der aktuellen Auswahl entsprechen."
        )
    cameras = ("cam_left_high", "cam_right_high")
    for index, anchor in enumerate(document.get("anchors", [])):
        prefix = f"Anchor {index}"
        if anchor.get("color") not in CUBE_COLORS:
            raise ValueError(f"{prefix}: ungültige Farbe {anchor.get('color')!r}.")
        if anchor.get("hand") not in ("left", "right"):
            raise ValueError(f"{prefix}: ungültige Hand {anchor.get('hand')!r}.")
        frame = anchor.get("frame")
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise ValueError(f"{prefix}: frame muss eine nichtnegative Ganzzahl sein.")
        _require_finite_vector(anchor.get("world_xy_m"), 2, f"{prefix}: world_xy_m")
        pixels = anchor.get("pixels")
        if not isinstance(pixels, dict) or any(camera not in pixels for camera in cameras):
            raise ValueError(f"{prefix}: beide Kopfkamera-Pixelrecords fehlen.")
        for camera in cameras:
            pixel = pixels[camera]
            pixel_prefix = f"{prefix}/{camera}"
            if not isinstance(pixel, dict) or pixel.get("source") != "top_face":
                raise ValueError(f"{pixel_prefix}: source muss top_face sein.")
            _require_finite_vector(pixel.get("top_uv"), 2, f"{pixel_prefix}: top_uv")
            _require_finite_vector(pixel.get("top_face_uv"), 2, f"{pixel_prefix}: top_face_uv")
            _require_finite_vector(pixel.get("bbox"), 4, f"{pixel_prefix}: bbox")
            pixel_frame = pixel.get("frame")
            if (
                not isinstance(pixel_frame, int)
                or isinstance(pixel_frame, bool)
                or pixel_frame < 0
            ):
                raise ValueError(f"{pixel_prefix}: frame ist ungültig.")
            pixel_frames = pixel.get("frames")
            top_face_frames = pixel.get("top_face_frames")
            if not _valid_frame_list(pixel_frames) or not _valid_frame_list(top_face_frames):
                raise ValueError(f"{pixel_prefix}: frames/top_face_frames sind ungültig.")
            top_face_count = pixel.get("top_face_count")
            num_samples = pixel.get("num_samples")
            if num_samples != len(pixel_frames):
                raise ValueError(f"{pixel_prefix}: num_samples passt nicht zu frames.")
            if (
                not isinstance(top_face_count, int)
                or isinstance(top_face_count, bool)
                or top_face_count < 5
                or top_face_count != len(top_face_frames)
            ):
                raise ValueError(f"{pixel_prefix}: mindestens fünf echte Top-Face-Frames nötig.")


def _require_finite_vector(value: Any, length: int, label: str) -> None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not np.isfinite(item)
            for item in value
        )
    ):
        raise ValueError(f"{label} muss {length} endliche Zahlen enthalten.")


def _valid_frame_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in value
        )
    )


def select_unique_closing_hand(
    decreases_m: list[float | None], min_close_m: float = 0.006, margin_m: float = 0.002
) -> int | None:
    ranked = sorted(
        ((float(value), index) for index, value in enumerate(decreases_m) if value is not None),
        reverse=True,
    )
    if not ranked or ranked[0][0] < min_close_m:
        return None
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    return ranked[0][1] if ranked[0][0] - second >= margin_m else None


def combine_camera_estimates(
    estimates: dict[str, dict[str, Any]], max_disagreement_m: float = 0.03,
) -> tuple[np.ndarray | None, float | None, str]:
    """Apply the explicit stereo/single-top-face acceptance policy."""
    if not estimates:
        return None, None, "in keiner Kopfkamera stabil erkannt"
    records = list(estimates.values())
    points = np.asarray([record["world_xy_m"] for record in records], dtype=float)
    disagreement = float(np.linalg.norm(points[0] - points[1])) if len(points) == 2 else None
    xy = np.median(points, axis=0)
    if disagreement is not None and disagreement > max_disagreement_m:
        return None, disagreement, f"Kopfkameras widersprechen sich um {disagreement:.3f} m"
    if len(points) == 1:
        top_face_count = int(records[0].get("top_face_count", 0))
        if top_face_count < 5:
            return None, None, (
                "Einzelkamera-Fallback erfordert mindestens fünf echte "
                f"Top-Face-Detektionen; vorhanden: {top_face_count}"
            )
        top_face_xy = records[0].get("top_face_world_xy_m")
        if top_face_xy is None:
            return None, None, "Einzelkamera-Fallback hat keine Top-Face-only Position"
        xy = np.asarray(top_face_xy, dtype=float)
    if not (0.25 <= xy[0] <= 0.45 and -0.25 <= xy[1] <= 0.25):
        return None, disagreement, "Position außerhalb des Würfel-Arbeitsbereichs"
    return xy, disagreement, ""
