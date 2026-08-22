#!/usr/bin/env python3
"""Collect sparse real-pixel/FK pick anchors without replaying actions."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--debug-dir", required=True)
parser.add_argument("--asset-path", default="")
parser.add_argument("--num-episodes", type=int, default=40)
parser.add_argument("--start-episode", type=int, default=0)
parser.add_argument("--episode-ids", type=int, nargs="*", default=None)
parser.add_argument("--train-ratio", type=float, default=0.8)
parser.add_argument("--min-area", type=int, default=80)
parser.add_argument("--onset-tolerance-frames", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

os.environ.setdefault("DR_ENABLED", "0")
os.environ["SCENE_CAM"] = "0"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg  # noqa: E402
from grasp_pose_support import closure_candidates, fingertip_measurement  # noqa: E402
from extract_block_layout import color_mask  # noqa: E402
from reconstruct_cube_poses import (  # noqa: E402
    CUBE_COLORS, HEAD_CAMS, data_path, read_info, select_episodes, video_path,
)
from replay_calibration import (  # noqa: E402
    CUBE_WORKSPACE_X_M, CUBE_WORKSPACE_Y_M, FINGERTIP_Z_WINDOW_M,
    action_sha256, find_motion_onset, select_unique_closing_hand,
    stationary_camera_measurements, track_colors,
)


def load_episode(root: Path, info: dict, episode: int) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_parquet(data_path(root, info, episode))
    actions = np.stack(table["action"].to_numpy()).astype(np.float32)
    states = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    if actions.shape != states.shape or states.ndim != 2 or states.shape[1] != 28:
        raise ValueError(
            f"Episode {episode}: action/state nicht (T,28): "
            f"{actions.shape}/{states.shape}"
        )
    return actions, states


def load_frames(path: Path, limit: int) -> list[np.ndarray]:
    frames = []
    with imageio.get_reader(str(path), format="FFMPEG") as reader:
        for index, frame in enumerate(reader):
            if index >= limit:
                break
            frames.append(np.asarray(frame, dtype=np.uint8)[..., :3])
    return frames


def set_robot_state(env: G1Dex3BlockstackEnv, state: np.ndarray) -> None:
    if env._joint_ids is None:
        env._joint_ids = env._build_joint_id_mapping()
    q = np.asarray(state, dtype=np.float32).copy()
    q[env._SIGN_FLIP_IDX] *= -1
    full = env.robot.data.joint_pos.clone()
    for policy_index, isaac_index in enumerate(env._joint_ids):
        full[:, isaac_index] = float(q[policy_index])
    env.robot.write_joint_state_to_sim(full, torch.zeros_like(full))
    env.robot.set_joint_position_target(full)
    env.sim.forward()
    env.robot.update(float(env.cfg.sim.dt))


def hand_measurement(env: G1Dex3BlockstackEnv, hand: int) -> tuple[float, np.ndarray]:
    tips = env.get_contact_points_w()[0].detach().cpu().numpy()
    if tips.shape != (6, 3) or env._reach_frame != "fingertip":
        raise RuntimeError(
            f"Sechs echte Fingerkuppen benötigt; frame={env._reach_frame}, "
            f"shape={tips.shape}"
        )
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    return fingertip_measurement(tips[hand * 3 : hand * 3 + 3] - origin)


def evaluate_hand(env, states: np.ndarray, onset: int, hand: int) -> dict | None:
    start, stop = max(0, onset - 18), min(len(states), onset + 7)
    candidates = closure_candidates(states[start:stop], "left" if hand == 0 else "right")
    candidates = [{**item, "start_frame": item["start_frame"] + start,
                   "end_frame": item["end_frame"] + start} for item in candidates]
    if not candidates:
        return None
    best = None
    for candidate in candidates:
        set_robot_state(env, states[candidate["start_frame"]])
        opened, _ = hand_measurement(env, hand)
        set_robot_state(env, states[candidate["end_frame"]])
        closed, _ = hand_measurement(env, hand)
        decrease = opened - closed
        if best is None or decrease > best["finger_closure_m"]:
            best = {**candidate, "finger_opening_before_m": opened,
                    "finger_opening_after_m": closed, "finger_closure_m": decrease}
    return best


def pick_anchor(env, states: np.ndarray, onset: int) -> tuple[dict | None, list[dict], str]:
    measurements = [evaluate_hand(env, states, onset, hand) for hand in range(2)]
    diagnostics = [
        item or {"reason": "keine koordinierte Schließbewegung"}
        for item in measurements
    ]
    hand = select_unique_closing_hand(
        [item["finger_closure_m"] if item is not None else None for item in measurements]
    )
    if hand is None:
        return None, diagnostics, "keine eindeutige physische Handschließung"
    event = measurements[hand]
    # Gemessen wird am ENDE der Schließbewegung, nicht am Bewegungsbeginn. Zwischen beiden
    # lagen im Abnahmelauf 2026-08-22 neun bis achtzehn Frames, in denen die Hand den Würfel
    # bereits anhebt: alle zehn damals akzeptierten Anker waren dadurch nach vorne und oben
    # versetzt (neun davon 3,5–9,9 cm über der Würfeloberseite, im Mittel 11,5 cm weiter
    # vorne als die unabhängige Rückprojektion desselben Würfelpixels). Der Anker soll den
    # Würfel in seiner RUHELAGE treffen, also im letzten Moment vor dem Anheben.
    reference = min(int(event["end_frame"]), onset)
    frames = sorted({
        max(0, min(len(states) - 1, onset, reference + offset)) for offset in (-2, 0, 2)
    })
    centers = []
    for frame in frames:
        set_robot_state(env, states[frame])
        _, center = hand_measurement(env, hand)
        centers.append(center)
    center = np.median(np.asarray(centers), axis=0)
    if not (CUBE_WORKSPACE_X_M[0] <= center[0] <= CUBE_WORKSPACE_X_M[1]
            and CUBE_WORKSPACE_Y_M[0] <= center[1] <= CUBE_WORKSPACE_Y_M[1]
            and FINGERTIP_Z_WINDOW_M[0] <= center[2] <= FINGERTIP_Z_WINDOW_M[1]):
        return None, diagnostics, (
            "Fingerkuppen liegen nicht an einem ruhenden Würfel: "
            f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m"
        )
    return ({**event, "hand": "left" if hand == 0 else "right",
             "measurement_frames": frames,
             "fingertip_centroid_m": center.tolist()}, diagnostics, "")


def save_detection_overlay(frame: np.ndarray, measurement: dict, color: str, path: Path) -> None:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    display_color = {"rot": "red", "gruen": "green", "gelb": "yellow"}[color]
    x0, y0, x1, y1 = measurement["bbox"]
    draw.rectangle((x0, y0, x1, y1), outline="white", width=3)
    u, v = measurement["top_uv"]
    draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=display_color, width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    mask = (color_mask(frame, color) * 255).astype(np.uint8)
    Image.fromarray(mask).save(path.with_name(path.stem + "_color_mask.png"))


def main() -> int:
    root, out, debug = Path(args.dataset_path), Path(args.out), Path(args.debug_dir)
    info = read_info(root)
    episodes = select_episodes(info, args)
    cfg = G1Dex3BlockstackEnvCfg()
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    cfg.sim.render_interval = 1_000_000
    cfg.episode_length_s = 3600.0
    cfg.terminate_on_success = False
    env = G1Dex3BlockstackEnv(cfg=cfg, render_mode=None)
    payload = {
        "version": 4,
        "method": "sparse_pick_anchors",
        "source_dataset": str(root.resolve()),
        "episodes": episodes,
        "anchors": [],
        "episode_diagnostics": {},
        "action_hashes": {},
    }
    try:
        env.reset()
        for ordinal, episode in enumerate(episodes, 1):
            actions, states = load_episode(root, info, episode)
            payload["action_hashes"][str(episode)] = action_sha256(actions)
            paths = {camera: video_path(root, info, episode, camera) for camera in HEAD_CAMS}
            with ThreadPoolExecutor(max_workers=2) as executor:
                tracks = dict(zip(paths, executor.map(track_colors, paths.values())))
            onsets = {color: {camera: find_motion_onset(tracks[camera][color])
                              for camera in HEAD_CAMS} for color in CUBE_COLORS}
            max_needed = max([value for values in onsets.values() for value in values.values()
                              if value is not None] + [30])
            frames = {camera: load_frames(path, max_needed + 1) for camera, path in paths.items()}
            diagnostics = {"colors": {}}
            accepted = 0
            for color in CUBE_COLORS:
                values = onsets[color]
                record = {
                    "motion_onsets": values,
                    "status": "rejected",
                    "reason": "",
                    "pixels": {},
                }
                diagnostics["colors"][color] = record
                measurements, measurement_stops = stationary_camera_measurements(
                    frames, color, values, min_area=args.min_area
                )
                record["measurement_stop_frames"] = measurement_stops
                pixels = {}
                for camera, measurement in measurements.items():
                    record["pixels"][camera] = measurement
                    if measurement is None:
                        continue
                    pixels[camera] = measurement
                    frame = frames[camera][measurement["frame"]]
                    overlay_path = debug / (
                        f"ep{episode:06d}_{camera}_{color}_f"
                        f"{measurement['frame']:04d}.png"
                    )
                    save_detection_overlay(frame, measurement, color, overlay_path)
                    measurement["overlay_path"] = str(overlay_path)
                    measurement["color_mask_path"] = str(
                        overlay_path.with_name(overlay_path.stem + "_color_mask.png")
                    )
                if any(values[camera] is None for camera in HEAD_CAMS):
                    record["reason"] = "Bewegungsbeginn fehlt in einer Kopfkamera"
                    continue
                if abs(values[HEAD_CAMS[0]] - values[HEAD_CAMS[1]]) > args.onset_tolerance_frames:
                    record["reason"] = "Kamera-Onsets unterscheiden sich um mehr als 12 Frames"
                    continue
                onset = int(round(np.median(list(values.values()))))
                if len(pixels) != 2:
                    record["reason"] = "keine stabilen Top-Face-Messungen in beiden Kameras"
                    continue
                event, hand_diagnostics, anchor_reason = pick_anchor(env, states, onset)
                record["hand_diagnostics"] = hand_diagnostics
                if event is None:
                    record["reason"] = anchor_reason
                    continue
                anchor = {"episode": episode, "color": color, "frame": onset,
                          "hand": event["hand"], "world_xy_m": event["fingertip_centroid_m"][:2],
                          "world_xyz_m": event["fingertip_centroid_m"], "pixels": pixels,
                          "pick_event": event}
                payload["anchors"].append(anchor)
                record.update(status="accepted", reason="", matched_hand=event["hand"])
                accepted += 1
            if action_sha256(actions) != payload["action_hashes"][str(episode)]:
                raise RuntimeError(f"Episode {episode}: Originalaktionen wurden verändert")
            payload["episode_diagnostics"][str(episode)] = diagnostics
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            print(
                f"[replay-calibrate] ({ordinal}/{len(episodes)}) Episode {episode}: "
                f"{accepted} Anker.",
                flush=True,
            )
        print(f"[replay-anchor-collection] {len(payload['anchors'])} Anker; {out}", flush=True)
        return 0
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
