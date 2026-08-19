#!/usr/bin/env python3
"""Validate trajectory anchors against cubes rendered by the actual Isaac camera."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--calibration", required=True)
parser.add_argument("--anchors", required=True)
parser.add_argument("--report-dir", required=True)
parser.add_argument("--asset-path", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

os.environ.setdefault("DR_ENABLED", "0")
os.environ["SCENE_CAM"] = "0"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg  # noqa: E402
from reconstruct_cube_poses import data_path, read_info  # noqa: E402
from replay_calibration import read_json, top_face_blob  # noqa: E402


HEAD_CAMS = ("cam_left_high", "cam_right_high")
COLOR_INDEX = {"rot": 0, "gruen": 1, "gelb": 2}


def configure(cfg: G1Dex3BlockstackEnvCfg, calibration: dict) -> None:
    edge = float(calibration["cube_edge_m"])
    table_top = float(calibration["table_top_z_m"])
    table_size = list(cfg.scene.table.spawn.size)
    table_size[2] = table_top
    cfg.scene.table.spawn.size = tuple(table_size)
    table_pos = list(cfg.scene.table.init_state.pos)
    table_pos[2] = table_top / 2.0
    cfg.scene.table.init_state.pos = tuple(table_pos)
    for name in ("block_0", "block_1", "block_2"):
        getattr(cfg.scene, name).spawn.size = (edge, edge, edge)
    for name in HEAD_CAMS:
        record = calibration["cameras"][name]
        camera_cfg = getattr(cfg.scene, name)
        camera_cfg.offset.pos = tuple(record["eye"])
        camera_cfg.offset.rot = tuple(record["quat_wxyz"])
        camera_cfg.spawn.focal_length = float(record["focal_mm"])
        camera_cfg.spawn.horizontal_aperture = float(record["aperture_mm"])


def set_robot_state(env: G1Dex3BlockstackEnv, state: np.ndarray) -> None:
    if env._joint_ids is None:
        env._joint_ids = env._build_joint_id_mapping()
    policy = state.astype(np.float32).copy()
    policy[env._SIGN_FLIP_IDX] *= -1
    full = env.robot.data.joint_pos.clone()
    for policy_index, isaac_index in enumerate(env._joint_ids):
        full[:, isaac_index] = float(policy[policy_index])
    env.robot.write_joint_state_to_sim(full, torch.zeros_like(full))
    env.robot.set_joint_position_target(full)


def place_markers(env: G1Dex3BlockstackEnv, episode_anchors: list[dict], z: float) -> None:
    by_color = {record["color"]: record for record in episode_anchors}
    origin = env.scene.env_origins[0]
    for color, index in COLOR_INDEX.items():
        if color in by_color:
            xy = by_color[color]["world_xy_m"]
            position = [float(xy[0]), float(xy[1]), z]
        else:
            position = [5.0 + index, 5.0, 0.1]
        pose = torch.tensor(
            [[*position, 1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32
        )
        pose[:, :3] += origin
        env.blocks[index].write_root_pose_to_sim(pose)
        env.blocks[index].write_root_velocity_to_sim(
            torch.zeros((1, 6), device=env.device, dtype=torch.float32)
        )


def render(env: G1Dex3BlockstackEnv) -> dict:
    forward = getattr(env.sim, "forward", None)
    if callable(forward):
        forward()
    env.sim.render()
    for camera in env.cameras.values():
        camera.update(0.0)
    return env._get_observations()


def frame_from_obs(obs: dict, camera: str) -> np.ndarray:
    return obs[f"video.{camera}"][0].detach().cpu().numpy().astype(np.uint8)


def main() -> int:
    root = Path(args.dataset_path)
    info = read_info(root)
    calibration = read_json(Path(args.calibration))
    anchor_doc = read_json(Path(args.anchors))
    by_episode: dict[int, list[dict]] = {}
    for anchor in anchor_doc["anchors"]:
        by_episode.setdefault(int(anchor["episode"]), []).append(anchor)

    cfg = G1Dex3BlockstackEnvCfg()
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    configure(cfg, calibration)
    cfg.terminate_on_success = False
    cfg.episode_length_s = 10.0
    env = G1Dex3BlockstackEnv(cfg=cfg, render_mode=None)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    details = []
    try:
        for episode, episode_anchors in sorted(by_episode.items()):
            frame = pd.read_parquet(data_path(root, info, episode))
            state0 = np.asarray(frame["observation.state"].iloc[0], dtype=np.float32)
            env.reset()
            set_robot_state(env, state0)
            place_markers(env, episode_anchors, float(calibration["cube_center_z_m"]))
            obs = render(env)
            for camera in HEAD_CAMS:
                rgb = frame_from_obs(obs, camera)
                image = Image.fromarray(rgb)
                draw = ImageDraw.Draw(image)
                for anchor in episode_anchors:
                    expected = anchor["pixels"][camera]["top_uv"]
                    detected = top_face_blob(rgb, anchor["color"])
                    record = {
                        "episode": episode,
                        "camera": camera,
                        "color": anchor["color"],
                        "expected_uv": expected,
                    }
                    if detected is None:
                        details.append({**record, "status": "missing"})
                        continue
                    rendered = [float(detected["u"]), float(detected["v"])]
                    error = float(np.linalg.norm(np.asarray(rendered) - expected))
                    details.append({
                        **record,
                        "status": "ok",
                        "rendered_uv": rendered,
                        "error_px": error,
                    })
                    u, v = rendered
                    draw.ellipse((u - 6, v - 6, u + 6, v + 6), outline="white", width=3)
                image.save(report_dir / f"marker_ep{episode:06d}_{camera}.png")

        errors = [record["error_px"] for record in details if record["status"] == "ok"]
        expected_count = sum(len(records) for records in by_episode.values()) * len(HEAD_CAMS)
        report = {
            "samples": len(errors),
            "expected_samples": expected_count,
            "median_pixel_error": float(np.median(errors)) if errors else 1e9,
            "p90_pixel_error": float(np.percentile(errors, 90)) if errors else 1e9,
            "details": details,
        }
        (report_dir / "marker_renderer_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        if len(errors) != expected_count or report["median_pixel_error"] > 5.0 \
                or report["p90_pixel_error"] > 10.0:
            raise RuntimeError(
                "Marker-Rendererprüfung fehlgeschlagen: "
                f"{len(errors)}/{expected_count} Punkte, Median/P90 "
                f"{report['median_pixel_error']:.1f}/{report['p90_pixel_error']:.1f} px."
            )
        print(
            "[replay-calibration-render] Marker Median/P90 "
            f"{report['median_pixel_error']:.1f}/{report['p90_pixel_error']:.1f} px.",
            flush=True,
        )
        print("[replay-calibration-render] fertig.", flush=True)
    finally:
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
