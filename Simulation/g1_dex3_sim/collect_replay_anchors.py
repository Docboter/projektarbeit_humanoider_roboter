#!/usr/bin/env python3
"""Collect trajectory-anchored real-pixel to simulated-table correspondences."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--asset-path", default="")
parser.add_argument("--num-episodes", type=int, default=10)
parser.add_argument("--start-episode", type=int, default=0)
parser.add_argument("--episode-ids", type=int, nargs="*", default=None)
parser.add_argument("--train-ratio", type=float, default=0.8)
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

os.environ.setdefault("DR_ENABLED", "0")
os.environ["SCENE_CAM"] = "0"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg  # noqa: E402
from reconstruct_cube_poses import (  # noqa: E402
    CUBE_COLORS,
    HEAD_CAMS,
    load_video_frames,
    read_info,
    select_episodes,
    video_path,
)
from replay_calibration import (  # noqa: E402
    action_sha256,
    best_top_face_detection,
    find_motion_onset,
    match_closing_hand,
    top_face_blob,
    track_colors,
)


def episode_path(root: Path, info: dict, episode: int) -> Path:
    template = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    return root / template.format(
        episode_chunk=episode // int(info.get("chunks_size", 1000)), episode_index=episode
    )


def load_episode(root: Path, info: dict, episode: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(episode_path(root, info, episode))
    actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
    states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    if actions.shape != states.shape or actions.ndim != 2 or actions.shape[1] != 28:
        raise ValueError(f"Episode {episode}: action/state nicht (T,28): {actions.shape}")
    return actions, states


def set_robot_state(env: G1Dex3BlockstackEnv, state: np.ndarray) -> None:
    if env._joint_ids is None:
        env._joint_ids = env._build_joint_id_mapping()
    q = np.asarray(state, dtype=np.float32).copy()
    q[env._SIGN_FLIP_IDX] *= -1
    full = env.robot.data.joint_pos.clone()
    for policy_idx, isaac_idx in enumerate(env._joint_ids):
        full[:, isaac_idx] = float(q[policy_idx])
    zeros = torch.zeros_like(full)
    env.robot.write_joint_state_to_sim(full, zeros)
    env.robot.set_joint_position_target(full)


def stash_cubes(env: G1Dex3BlockstackEnv) -> None:
    for index, block in enumerate(env.blocks):
        pose = torch.tensor(
            [[5.0 + index, 5.0, 0.1, 1.0, 0.0, 0.0, 0.0]],
            device=env.device,
            dtype=torch.float32,
        )
        block.write_root_pose_to_sim(pose)
        block.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))


def hand_measurement(env: G1Dex3BlockstackEnv) -> tuple[np.ndarray, np.ndarray]:
    tips = env.get_contact_points_w()[0].detach().cpu().numpy()
    if tips.shape != (6, 3) or env._reach_frame != "fingertip":
        raise RuntimeError(
            "Kalibrierung benötigt sechs echte Fingerkuppen; "
            f"erhalten: frame={env._reach_frame}, shape={tips.shape}."
        )
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    tips = tips - origin
    spreads, centroids = np.zeros(2), np.zeros((2, 3))
    for hand in range(2):
        three = tips[hand * 3:hand * 3 + 3]
        spreads[hand] = np.linalg.norm(
            three[[0, 0, 1]] - three[[1, 2, 2]], axis=1
        ).mean()
        centroids[hand] = three.mean(axis=0)
    return spreads, centroids


def replay_measurements(env: G1Dex3BlockstackEnv, actions: np.ndarray,
                        state0: np.ndarray) -> tuple[np.ndarray, ...]:
    env.reset()
    set_robot_state(env, state0)
    stash_cubes(env)
    forward = getattr(env.sim, "forward", None)
    if callable(forward):
        forward()
    wrist_ids, _ = env.robot.find_bodies(
        ["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True
    )
    spreads, centroids, wrist_positions, wrist_quaternions = [], [], [], []
    for frame_index in range(len(actions)):
        spread, centroid = hand_measurement(env)
        spreads.append(spread)
        centroids.append(centroid)
        origin = env.scene.env_origins[0]
        wrist_positions.append(
            (env.robot.data.body_pos_w[0, wrist_ids] - origin).detach().cpu().numpy()
        )
        wrist_quaternions.append(
            env.robot.data.body_quat_w[0, wrist_ids].detach().cpu().numpy()
        )
        action_before = actions[frame_index].copy()
        action = torch.from_numpy(action_before).to(env.device).unsqueeze(0)
        env.step(action)
        if not np.array_equal(action_before, actions[frame_index]):
            raise RuntimeError("Originalaktion wurde während der Kalibrierung verändert.")
    return (
        np.asarray(spreads), np.asarray(centroids),
        np.asarray(wrist_positions), np.asarray(wrist_quaternions),
    )


def initial_pixels(root: Path, info: dict, episode: int) -> dict[str, dict]:
    result: dict[str, dict] = {color: {} for color in CUBE_COLORS}
    for camera in HEAD_CAMS:
        frames = load_video_frames(video_path(root, info, episode, camera), 30)
        for color in CUBE_COLORS:
            frame_index, _, blob = best_top_face_detection(frames, color)
            if blob is not None:
                result[color][camera] = {
                    "top_uv": [float(blob["u"]), float(blob["v"])],
                    "bbox": [int(v) for v in blob["bbox"]],
                    "frame": int(frame_index),
                    "source": blob.get("source", "full_blob"),
                }
    return result


def wrist_observations(root: Path, info: dict, episode: int, color: str, hand: int,
                       onset: int, wrist_positions: np.ndarray,
                       wrist_quaternions: np.ndarray, world_xy: np.ndarray) -> list[dict]:
    camera = "cam_left_wrist" if hand == 0 else "cam_right_wrist"
    frames = load_video_frames(video_path(root, info, episode, camera), onset + 1)
    candidates = []
    start = max(0, onset - 90)
    for frame_index in range(start, min(onset, len(frames)), 3):
        blob = top_face_blob(frames[frame_index], color, min_area=40)
        if blob is None:
            continue
        candidates.append({
            "camera": camera,
            "episode": int(episode),
            "color": color,
            "frame": int(frame_index),
            "top_uv": [float(blob["u"]), float(blob["v"])],
            "world_xy_m": [float(v) for v in world_xy],
            "link_position_m": [float(v) for v in wrist_positions[frame_index, hand]],
            "link_quat_wxyz": [float(v) for v in wrist_quaternions[frame_index, hand]],
        })
    return candidates[-12:]


def camera_diagnostics(env: G1Dex3BlockstackEnv) -> dict[str, dict]:
    diagnostics = {}
    for name, camera in env.cameras.items():
        data = camera.data
        record = {}
        for source_name, target_name in (
            ("intrinsic_matrices", "intrinsic_matrix"),
            ("pos_w", "reported_position_w"),
            ("quat_w_world", "reported_quat_wxyz"),
        ):
            value = getattr(data, source_name, None)
            if value is not None:
                array = value[0].detach().cpu().numpy()
                record[target_name] = np.asarray(array).tolist()
        diagnostics[name] = record
    return diagnostics


def main() -> int:
    root = Path(args.dataset_path)
    info = read_info(root)
    episodes = select_episodes(info, args)
    longest = max(len(pd.read_parquet(episode_path(root, info, ep))) for ep in episodes)
    cfg = G1Dex3BlockstackEnvCfg()
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    cfg.decimation = 7
    cfg.sim.dt = 1.0 / (30.0 * cfg.decimation)
    cfg.sim.render_interval = 1000000
    cfg.episode_length_s = longest / 30.0 + 10.0
    cfg.terminate_on_success = False
    env = G1Dex3BlockstackEnv(cfg=cfg, render_mode=None)

    env.reset()
    diagnostics = camera_diagnostics(env)
    payload = {
        "version": 2,
        "episodes": episodes,
        "anchors": [],
        "wrist_observations": [],
        "action_hashes": {},
        "actual_camera_diagnostics": diagnostics,
    }
    try:
        for ordinal, episode in enumerate(episodes, 1):
            actions, states = load_episode(root, info, episode)
            before_hash = action_sha256(actions)
            spreads, centroids, wrist_positions, wrist_quaternions = replay_measurements(
                env, actions, states[0]
            )
            after_hash = action_sha256(actions)
            if before_hash != after_hash:
                raise RuntimeError(f"Episode {episode}: Action-Hash hat sich geändert.")
            payload["action_hashes"][str(episode)] = before_hash
            pixels = initial_pixels(root, info, episode)
            tracks_by_camera = {
                camera: track_colors(video_path(root, info, episode, camera))
                for camera in HEAD_CAMS
            }

            candidates = []
            for color in CUBE_COLORS:
                onsets = [
                    find_motion_onset(tracks_by_camera[camera][color])
                    for camera in HEAD_CAMS
                ]
                if any(value is None for value in onsets):
                    continue
                if max(onsets) - min(onsets) > 5:
                    continue
                onset = int(round(float(np.median(onsets))))
                if onset >= len(centroids):
                    continue
                hand = match_closing_hand(onset, spreads)
                if hand is None or len(pixels[color]) != len(HEAD_CAMS):
                    continue
                candidates.append((onset, color, hand))

            # Je Hand nur das erste eindeutige Pick-Ereignis. Spätere Bewegungen können
            # Ablegen oder Kontakt mit einem bereits transportierten Würfel sein.
            used_hands: set[int] = set()
            for onset, color, hand in sorted(candidates):
                if hand in used_hands:
                    continue
                used_hands.add(hand)
                xy = centroids[onset, hand, :2]
                payload["anchors"].append({
                    "episode": int(episode),
                    "color": color,
                    "hand": "left" if hand == 0 else "right",
                    "frame": int(onset),
                    "world_xy_m": [float(xy[0]), float(xy[1])],
                    "world_xyz_m": [float(v) for v in centroids[onset, hand]],
                    "pixels": pixels[color],
                })
                payload["wrist_observations"].extend(
                    wrist_observations(
                        root, info, episode, color, hand, onset,
                        wrist_positions, wrist_quaternions, xy,
                    )
                )
            print(
                f"[replay-calibrate] ({ordinal}/{len(episodes)}) Episode {episode}: "
                f"{len(used_hands)} Bewegungsanker.",
                flush=True,
            )

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[replay-calibrate] Anker: {out}")
        print("[replay-anchor-collection] fertig.", flush=True)
    finally:
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
