#!/usr/bin/env python3
"""Rendert echte G1-DEX3-Dataset-Bewegungen mit einmalig gesetzten Würfeln.

Es gibt absichtlich keine Policy, kein Würfeltracking und keinen kinematischen Attach.
Nach dem einmaligen Spawn bestimmt ausschließlich PhysX die Würfelbewegung.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--poses", required=True)
parser.add_argument("--out-dir", default="/data/cube_replay/videos")
parser.add_argument("--asset-path", default="")
parser.add_argument("--num-episodes", type=int, default=10)
parser.add_argument("--start-episode", type=int, default=0)
parser.add_argument("--episode-ids", type=int, nargs="*", default=None)
parser.add_argument("--train-ratio", type=float, default=0.8)
parser.add_argument("--max-frames", type=int, default=0,
                    help="0 = vollständige Episode; >0 = Techniktest")
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.num_episodes < 1:
    raise SystemExit("--num-episodes muss mindestens 1 sein.")

# Vor dem Import der Env setzen: deren Config liest diese Werte in __post_init__.
os.environ.setdefault("DR_ENABLED", "0")
os.environ["SCENE_CAM"] = "1"
os.environ["CAM_RES_SCALE"] = "1"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg  # noqa: E402


CAMS = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist", "scene")
DEFAULT_DATA_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


def read_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def select_episodes(info: dict) -> list[int]:
    train_stop = int(int(info["total_episodes"]) * float(args.train_ratio))
    if args.episode_ids:
        selected = sorted(set(int(ep) for ep in args.episode_ids))
    else:
        selected = list(range(
            int(args.start_episode),
            min(int(args.start_episode) + int(args.num_episodes), train_stop),
        ))
    bad = [ep for ep in selected if ep < 0 or ep >= train_stop]
    if bad:
        raise SystemExit(f"Episoden außerhalb des Trainingsbereichs 0:{train_stop}: {bad}")
    return selected


def episode_path(root: Path, info: dict, ep: int) -> Path:
    template = info.get("data_path", DEFAULT_DATA_TEMPLATE)
    try:
        return root / template.format(
            episode_chunk=ep // int(info.get("chunks_size", 1000)), episode_index=ep
        )
    except KeyError as exc:
        raise SystemExit(
            "Quelldatensatz ist nicht LeRobot v2.1. Zuerst replay-prepare ausführen."
        ) from exc


def load_episode(root: Path, info: dict, ep: int) -> tuple[np.ndarray, np.ndarray]:
    path = episode_path(root, info, ep)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
    states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    if actions.ndim != 2 or states.ndim != 2 or actions.shape[1:] != (28,) \
            or states.shape[1:] != (28,):
        raise ValueError(
            f"Episode {ep}: erwartet action/state (T,28), erhalten "
            f"{actions.shape}/{states.shape}"
        )
    n = min(len(actions), len(states))
    if args.max_frames > 0:
        n = min(n, int(args.max_frames))
    return actions[:n], states[:n]


def configure_geometry(cfg: G1Dex3BlockstackEnvCfg, calibration: dict) -> None:
    edge = float(calibration["cube_edge_m"])
    table_top = float(calibration["table_top_z_m"])
    center_z = float(calibration["cube_center_z_m"])

    table_size = list(cfg.scene.table.spawn.size)
    table_size[2] = table_top
    cfg.scene.table.spawn.size = tuple(table_size)
    table_pos = list(cfg.scene.table.init_state.pos)
    table_pos[2] = table_top / 2.0
    cfg.scene.table.init_state.pos = tuple(table_pos)
    band_pos = list(cfg.scene.stack_band.init_state.pos)
    band_pos[2] = table_top + float(cfg.scene.stack_band.spawn.size[2]) / 2.0
    cfg.scene.stack_band.init_state.pos = tuple(band_pos)
    cfg.block_z_surface = center_z

    for name in ("block_0", "block_1", "block_2"):
        block_cfg = getattr(cfg.scene, name)
        block_cfg.spawn.size = (edge, edge, edge)
        position = list(block_cfg.init_state.pos)
        position[2] = center_z
        block_cfg.init_state.pos = tuple(position)

    for name in ("cam_left_high", "cam_right_high"):
        record = calibration["cameras"][name]
        camera_cfg = getattr(cfg.scene, name)
        camera_cfg.offset.pos = tuple(float(v) for v in record["eye"])
        camera_cfg.offset.rot = tuple(float(v) for v in record["quat_wxyz"])
        camera_cfg.spawn.focal_length = float(record["focal_mm"])
        camera_cfg.spawn.horizontal_aperture = float(record["aperture_mm"])
        camera_cfg.width = int(record["width"])
        camera_cfg.height = int(record["height"])


def set_robot_state(env: G1Dex3BlockstackEnv, state: np.ndarray) -> None:
    if env._joint_ids is None:
        env._joint_ids = env._build_joint_id_mapping()
    q = state.astype(np.float32).copy()
    q[env._SIGN_FLIP_IDX] *= -1
    full = env.robot.data.joint_pos.clone()
    for policy_idx, isaac_idx in enumerate(env._joint_ids):
        full[:, isaac_idx] = float(q[policy_idx])
    velocity = torch.zeros_like(full)
    env.robot.write_joint_state_to_sim(full, velocity)
    env.robot.set_joint_position_target(full)


def place_cubes_once(env: G1Dex3BlockstackEnv, blocks: list[dict]) -> None:
    by_name = {block["name"]: block for block in blocks if block.get("status") == "ok"}
    origin = env.scene.env_origins[0]
    for index, name in enumerate(("block_0", "block_1", "block_2")):
        if name not in by_name:
            raise ValueError(f"Pose für {name} fehlt")
        record = by_name[name]
        position = torch.tensor(record["position_m"], device=env.device, dtype=torch.float32)
        position = position + origin
        quat = torch.tensor(
            record["orientation_wxyz"], device=env.device, dtype=torch.float32
        )
        pose = torch.cat((position, quat)).unsqueeze(0)
        env.blocks[index].write_root_pose_to_sim(pose)
        env.blocks[index].write_root_velocity_to_sim(
            torch.zeros((1, 6), device=env.device, dtype=torch.float32)
        )


def render_current(env: G1Dex3BlockstackEnv) -> dict:
    """Aktualisiert Render-Produkte ohne einen Physikschritt auszuführen."""
    forward = getattr(env.sim, "forward", None)
    if callable(forward):
        forward()
    env.sim.render()
    for camera in env.cameras.values():
        camera.update(0.0)
    return env._get_observations()


def frame_from_obs(obs: dict, cam: str) -> np.ndarray:
    key = "video.cam_scene" if cam == "scene" else f"video.{cam}"
    frame = obs[key][0].detach().cpu().numpy()
    while frame.ndim > 3:
        frame = frame[0]
    return frame.astype(np.uint8)


def output_paths(out_dir: Path, ep: int) -> dict[str, Path]:
    return {cam: out_dir / f"episode_{ep:06d}_{cam}.mp4" for cam in CAMS}


def open_writers(paths: dict[str, Path], fps: float) -> tuple[dict, dict[str, Path]]:
    writers, temporary = {}, {}
    for cam, final_path in paths.items():
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_name(final_path.stem + ".tmp.mp4")
        temporary[cam] = tmp_path
        writers[cam] = imageio.get_writer(
            str(tmp_path), format="FFMPEG", mode="I", fps=fps, codec="libx264",
            pixelformat="yuv420p", macro_block_size=8, ffmpeg_log_level="error",
        )
    return writers, temporary


def append_observation(writers: dict, obs: dict) -> None:
    for cam, writer in writers.items():
        writer.append_data(frame_from_obs(obs, cam))


def main() -> int:
    dataset = Path(args.dataset_path)
    pose_doc = json.loads(Path(args.poses).read_text(encoding="utf-8"))
    calibration = json.loads(Path(pose_doc["calibration"]).read_text(encoding="utf-8"))
    info = read_info(dataset)
    fps = float(info.get("fps", 30.0))
    if abs(fps - 30.0) > 1e-6:
        raise SystemExit(f"Datensatz hat {fps} Hz statt der erwarteten 30 Hz.")

    selected = select_episodes(info)
    loaded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ep in selected:
        pose_entry = pose_doc.get("episodes", {}).get(str(ep))
        if not pose_entry or pose_entry.get("status") != "ok":
            print(f"[replay-render] Episode {ep}: keine vollständige Würfelpose — übersprungen.")
            continue
        try:
            loaded[ep] = load_episode(dataset, info, ep)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[replay-render] Episode {ep}: {exc} — übersprungen.")
    if not loaded:
        raise SystemExit("Keine renderbare Episode mit vollständiger Würfelpose.")

    longest = max(len(actions) for actions, _ in loaded.values())
    cfg = G1Dex3BlockstackEnvCfg()
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    configure_geometry(cfg, calibration)
    cfg.decimation = 7
    cfg.sim.dt = 1.0 / (fps * cfg.decimation)
    cfg.sim.render_interval = cfg.decimation
    cfg.policy_hz = fps
    cfg.episode_length_s = longest / fps + 10.0
    cfg.terminate_on_success = False

    print("=" * 72)
    print("G1+DEX3 — physikbasierter Dataset-Video-Replay")
    print("=" * 72)
    print(f"  Dataset:       {dataset}")
    print(f"  Episoden:      {list(loaded)}")
    print(f"  Zeitbasis:     {fps:.0f} Hz, dt={cfg.sim.dt:.8f}, decimation={cfg.decimation}")
    print(f"  Würfelkante:   {calibration['cube_edge_m'] * 100:.2f} cm")
    print(f"  Tischoberkante:{calibration['table_top_z_m']:.3f} m")
    print("  Würfelmodus:   einmaliger Spawn, danach ausschließlich PhysX")

    env = G1Dex3BlockstackEnv(cfg=cfg, render_mode=None)
    out_dir = Path(args.out_dir)
    try:
        for ordinal, (ep, (actions, states)) in enumerate(loaded.items(), 1):
            paths = output_paths(out_dir, ep)
            if not args.overwrite and all(path.is_file() for path in paths.values()):
                print(f"[replay-render] ({ordinal}/{len(loaded)}) Episode {ep}: vorhanden.")
                continue

            env.reset()
            set_robot_state(env, states[0])
            place_cubes_once(env, pose_doc["episodes"][str(ep)]["blocks"])
            obs = render_current(env)
            writers, temporary = open_writers(paths, fps)
            try:
                try:
                    # Frame 0 entspricht dem echten Anfangszustand. Action[t] erzeugt danach
                    # Frame t+1. So bleiben Anzahl und zeitliche Bedeutung der Frames erhalten.
                    append_observation(writers, obs)
                    for frame_idx in range(1, len(actions)):
                        action = torch.tensor(
                            actions[frame_idx - 1], device=env.device, dtype=torch.float32
                        ).unsqueeze(0)
                        obs, _, _, _, _ = env.step(action)
                        append_observation(writers, obs)
                        if frame_idx % 200 == 0:
                            print(
                                f"      Episode {ep}: Frame {frame_idx}/{len(actions)}",
                                flush=True,
                            )
                finally:
                    for writer in writers.values():
                        writer.close()
            except Exception:
                for tmp_path in temporary.values():
                    tmp_path.unlink(missing_ok=True)
                raise

            for cam, tmp_path in temporary.items():
                if not tmp_path.is_file():
                    raise RuntimeError(f"MP4-Schreiber lieferte keine Datei: {tmp_path}")
                tmp_path.replace(paths[cam])
            print(f"[replay-render] ({ordinal}/{len(loaded)}) Episode {ep}: "
                  f"{len(actions)} Frames, 5 MP4s.", flush=True)
    finally:
        env.close()
        simulation_app.close()

    print(f"[replay-render] Videos: {out_dir}")
    print("[replay-render] fertig.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
