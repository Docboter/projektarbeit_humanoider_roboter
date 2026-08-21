#!/usr/bin/env python3
"""Rendert echte G1-DEX3-Dataset-Bewegungen mit einmalig gesetzten Würfeln.

Es gibt absichtlich keine Policy, kein Würfeltracking und keinen kinematischen Attach.
Nach dem einmaligen Spawn bestimmt ausschließlich PhysX die Würfelbewegung.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--poses", required=True)
parser.add_argument("--out-dir", default="/data/cube_replay/videos")
parser.add_argument("--output-mode", choices=("videos", "dataset"), default="videos")
parser.add_argument("--dataset-out", default="/data/cube_replay/dataset")
parser.add_argument("--report-dir", default="/data/cube_replay/work/render_reports")
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
from replay_calibration import action_sha256, top_face_blob  # noqa: E402
from replay_grasp_metrics import (  # noqa: E402
    artifact_validation_errors,
    infer_expected_first_pick,
    mark_episode_existing,
    manifest_compatibility_errors,
    summarize_grasp_trace,
)


CAMS = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist", "scene")
POLICY_CAMS = CAMS[:4]
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


def load_episode(root: Path, info: dict, ep: int) -> tuple[np.ndarray, np.ndarray, int]:
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
    if len(actions) != len(states):
        raise ValueError(
            f"Episode {ep}: Action-/State-Länge weicht ab: {len(actions)}/{len(states)}"
        )
    n = len(actions)
    if args.max_frames > 0:
        n = min(n, int(args.max_frames))
    task_index = int(frame["task_index"].iloc[0]) if "task_index" in frame else 0
    return actions[:n], states[:n], task_index


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
    for name in ("cam_left_wrist", "cam_right_wrist"):
        if name not in calibration.get("cameras", {}):
            continue
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


def read_physics_positions(env: G1Dex3BlockstackEnv) -> tuple[np.ndarray, np.ndarray]:
    """Read env-local fingertip and block positions without mutating simulation state."""
    origin = env.scene.env_origins[0]
    fingertips = env.get_contact_points_w()
    blocks = torch.stack([block.data.root_pos_w for block in env.blocks], dim=1)
    if fingertips.shape != (1, 6, 3) or blocks.shape != (1, 3, 3):
        raise RuntimeError(
            "Griffdiagnose benötigt Fingerkuppen (1,6,3) und Würfel (1,3,3), "
            f"erhalten {tuple(fingertips.shape)}/{tuple(blocks.shape)}"
        )
    if tuple(origin.shape) != (3,):
        raise RuntimeError(f"Unerwartete Scene-Origin-Form: {tuple(origin.shape)}")
    return (
        (blocks[0] - origin).detach().cpu().numpy(),
        (fingertips[0] - origin).detach().cpu().numpy(),
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


def dataset_paths(out_dir: Path, ep: int) -> tuple[Path, dict[str, Path]]:
    chunk = ep // 1000
    parquet = out_dir / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
    videos = {
        cam: out_dir / (
            f"videos/chunk-{chunk:03d}/observation.images.{cam}/episode_{ep:06d}.mp4"
        )
        for cam in POLICY_CAMS
    }
    return parquet, videos


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


def validate_initial_projection(obs: dict, blocks: list[dict], report_dir: Path,
                                episode: int) -> dict:
    errors = []
    details = []
    for camera in POLICY_CAMS[:2]:
        rgb = frame_from_obs(obs, camera)
        from PIL import Image

        report_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(report_dir / f"ep{episode:06d}_{camera}_initial.png")
        for block in blocks:
            expected = block.get("camera_estimates", {}).get(camera, {}).get("top_uv")
            if expected is None:
                continue
            detected = top_face_blob(rgb, block["color"])
            if detected is None:
                details.append({"camera": camera, "color": block["color"], "status": "missing"})
                continue
            error = float(np.linalg.norm(
                np.asarray([detected["u"], detected["v"]]) - np.asarray(expected)
            ))
            errors.append(error)
            details.append({
                "camera": camera,
                "color": block["color"],
                "status": "ok",
                "expected_uv": expected,
                "rendered_uv": [float(detected["u"]), float(detected["v"])],
                "error_px": error,
            })
    return {
        "details": details,
        "samples": len(errors),
        "median_px": float(np.median(errors)) if errors else 1e9,
        "p90_px": float(np.percentile(errors, 90)) if errors else 1e9,
    }


def wrist_schedule(blocks: list[dict]) -> dict[int, list[dict]]:
    schedule: dict[int, list[dict]] = {}
    for block in blocks:
        for camera, record in block.get("wrist_observations", {}).items():
            schedule.setdefault(int(record["frame"]), []).append({
                "camera": camera,
                "color": block["color"],
                "expected_uv": record["top_uv"],
            })
    return schedule


def measure_wrist_projection(obs: dict, expected: list[dict], report_dir: Path,
                             episode: int, frame_index: int) -> list[dict]:
    from PIL import Image

    details = []
    for record in expected:
        camera = record["camera"]
        rgb = frame_from_obs(obs, camera)
        Image.fromarray(rgb).save(
            report_dir / f"ep{episode:06d}_{camera}_frame{frame_index:04d}.png"
        )
        detected = top_face_blob(rgb, record["color"], min_area=40)
        if detected is None:
            details.append({**record, "frame": frame_index, "status": "missing"})
            continue
        rendered = [float(detected["u"]), float(detected["v"])]
        error = float(np.linalg.norm(np.asarray(rendered) - record["expected_uv"]))
        details.append({
            **record,
            "frame": frame_index,
            "status": "ok",
            "rendered_uv": rendered,
            "error_px": error,
        })
    return details


def summarize_wrist_projection(details: list[dict]) -> dict:
    errors = [item["error_px"] for item in details if item.get("status") == "ok"]
    cameras = {item["camera"] for item in details if item.get("status") == "ok"}
    return {
        "details": details,
        "samples": len(errors),
        "validated_cameras": sorted(cameras),
        "median_px": float(np.median(errors)) if errors else 1e9,
        "p90_px": float(np.percentile(errors, 90)) if errors else 1e9,
    }


def write_dataset_parquet(path: Path, episode: int, achieved: np.ndarray,
                          actions: np.ndarray, task_index: int, fps: float,
                          index_offset: int) -> None:
    length = len(actions)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "observation.state": list(achieved.astype(np.float32)),
        "action": list(actions.astype(np.float32)),
        "timestamp": np.arange(length, dtype=np.float32) / fps,
        "frame_index": np.arange(length, dtype=np.int64),
        "episode_index": np.full(length, episode, dtype=np.int64),
        "index": np.arange(index_offset, index_offset + length, dtype=np.int64),
        "task_index": np.full(length, task_index, dtype=np.int64),
    }).to_parquet(path, index=False)


def finalize_dataset(out: Path, source: Path, info: dict, manifest: dict, fps: float) -> None:
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    task_map = {0: "stack the blocks"}
    source_tasks = source / "meta" / "tasks.jsonl"
    if source_tasks.is_file():
        task_map = {}
        for line in source_tasks.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            task_map[int(record["task_index"])] = record["task"]

    episodes = []
    numeric = {"observation.state": [], "action": []}
    next_index = 0
    for ep, record in sorted(manifest["episodes"].items(), key=lambda item: int(item[0])):
        if record.get("status") != "ok":
            continue
        episode = int(ep)
        parquet, videos = dataset_paths(out, episode)
        if not parquet.is_file() or not all(path.is_file() for path in videos.values()):
            continue
        frame = pd.read_parquet(parquet)
        length = len(frame)
        frame["index"] = np.arange(next_index, next_index + length, dtype=np.int64)
        frame.to_parquet(parquet, index=False)
        next_index += length
        task_index = int(frame["task_index"].iloc[0])
        record.update(frames=length, task_index=task_index)
        episodes.append({
            "episode_index": episode,
            "tasks": [task_map.get(task_index, "stack the blocks")],
            "length": length,
        })
        for key in numeric:
            numeric[key].append(np.stack(frame[key].to_numpy()).astype(np.float32))
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in episodes),
        encoding="utf-8",
    )
    if source_tasks.is_file():
        shutil.copyfile(source_tasks, meta / "tasks.jsonl")
    else:
        (meta / "tasks.jsonl").write_text(
            json.dumps(
                {"task_index": 0, "task": "stack the blocks"}, allow_nan=False
            ) + "\n",
            encoding="utf-8",
        )
    source_modality = source / "meta" / "modality.json"
    if source_modality.is_file():
        shutil.copyfile(source_modality, meta / "modality.json")
    features = {
        key: value for key, value in info["features"].items()
        if key not in {f"observation.images.{cam}" for cam in CAMS}
    }
    for camera in POLICY_CAMS:
        feature = json.loads(json.dumps(
            info["features"][f"observation.images.{camera}"], allow_nan=False
        ))
        feature["info"].update({
            "video.fps": fps,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "has_audio": False,
        })
        features[f"observation.images.{camera}"] = feature
    highest = max((record["episode_index"] for record in episodes), default=-1) + 1
    output_info = {
        **info,
        "codebase_version": "v2.1",
        "robot_type": "Unitree_G1_sim",
        "total_episodes": len(episodes),
        "total_frames": sum(record["length"] for record in episodes),
        "total_videos": len(episodes) * len(POLICY_CAMS),
        "total_chunks": (
            max(record["episode_index"] // 1000 for record in episodes) + 1
            if episodes else 0
        ),
        "splits": {"train": f"0:{highest}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }
    (meta / "info.json").write_text(
        json.dumps(output_info, indent=2, allow_nan=False), encoding="utf-8"
    )
    stats = {}
    for key, parts in numeric.items():
        values = np.concatenate(parts, axis=0)
        stats[key] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0, dtype=np.float64).tolist(),
            "std": values.std(axis=0, dtype=np.float64).tolist(),
            "count": [len(values)],
        }
    (meta / "stats.json").write_text(
        json.dumps(stats, indent=2, allow_nan=False), encoding="utf-8"
    )
    (out / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )


def validate_dataset(out: Path, manifest: dict) -> None:
    info = json.loads((out / "meta/info.json").read_text(encoding="utf-8"))
    good = [int(ep) for ep, record in manifest["episodes"].items()
            if record.get("status") == "ok"]
    if info["total_episodes"] != len(good) or info["fps"] != 30:
        raise RuntimeError("LeRobot-Metadaten stimmen nicht mit dem Replay-Manifest überein.")
    for episode in good:
        parquet, videos = dataset_paths(out, episode)
        frame = pd.read_parquet(parquet)
        actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        expected = manifest["episodes"][str(episode)]["action_sha256"]
        if actions.shape != states.shape or actions.shape[1:] != (28,):
            raise RuntimeError(f"Episode {episode}: Dataset-State/Action nicht (T,28).")
        if action_sha256(actions) != expected:
            raise RuntimeError(f"Episode {episode}: Action-Hash im Dataset weicht ab.")
        if not all(path.is_file() for path in videos.values()):
            raise RuntimeError(f"Episode {episode}: mindestens ein Policy-Video fehlt.")
        for camera, path in videos.items():
            with imageio.get_reader(str(path), format="FFMPEG") as reader:
                video_frames = int(reader.count_frames())
            if video_frames != len(frame):
                raise RuntimeError(
                    f"Episode {episode}/{camera}: {video_frames} Video- statt "
                    f"{len(frame)} Parquet-Frames."
                )
    print(f"[replay-render] LeRobot-Validierung: {len(good)} Episoden, 30 Hz, 28 DoF, "
          "Action-Hashes unverändert.", flush=True)


def main() -> int:
    if args.output_mode == "dataset" and args.max_frames > 0:
        raise SystemExit(
            "--max-frames ist im Dataset-Modus verboten; Techniktests im Modus videos ausführen."
        )
    dataset = Path(args.dataset_path)
    pose_path = Path(args.poses)
    try:
        pose_doc = json.loads(pose_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Würfelpose ist nicht lesbar: {pose_path}: {exc}") from exc
    calibration_reference = pose_doc.get("calibration")
    if not isinstance(calibration_reference, str):
        raise SystemExit("Würfelpose enthält keinen gültigen calibration-Pfad.")
    calibration_path = Path(calibration_reference)
    try:
        calibration_bytes = calibration_path.read_bytes()
        calibration = json.loads(calibration_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Kalibrierung ist nicht lesbar: {calibration_path}: {exc}"
        ) from exc
    calibration_hash = hashlib.sha256(calibration_bytes).hexdigest()
    artifact_errors = artifact_validation_errors(
        pose_doc, calibration, dataset, calibration_hash
    )
    if artifact_errors:
        raise SystemExit("Ungültige Replay-Artefakte: " + "; ".join(artifact_errors))
    if args.output_mode == "dataset" \
            and not calibration.get("wrist_calibration", {}).get("dataset_ready"):
        raise SystemExit(
            "Wrist-Kalibrierung ist nicht dataset-tauglich; mehr Kalibrierungsepisoden "
            "verwenden. Der Videomodus bleibt verfügbar."
        )
    info = read_info(dataset)
    fps = float(info.get("fps", 30.0))
    if abs(fps - 30.0) > 1e-6:
        raise SystemExit(f"Datensatz hat {fps} Hz statt der erwarteten 30 Hz.")

    selected = select_episodes(info)
    loaded: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for ep in selected:
        pose_entry = pose_doc.get("episodes", {}).get(str(ep))
        if not pose_entry or pose_entry.get("status") != "ok":
            print(
                f"[replay-render] Episode {ep}: keine vollständige Würfelpose — "
                "übersprungen."
            )
            continue
        try:
            loaded[ep] = load_episode(dataset, info, ep)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[replay-render] Episode {ep}: {exc} — übersprungen.")
    if not loaded:
        raise SystemExit("Keine renderbare Episode mit vollständiger Würfelpose.")

    longest = max(len(actions) for actions, _, _ in loaded.values())
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

    out_dir = Path(args.dataset_out if args.output_mode == "dataset" else args.out_dir)
    report_dir = Path(args.report_dir)
    manifest_path = (
        out_dir / "replay_manifest.json"
        if args.output_mode == "dataset"
        else report_dir / "replay_manifest.json"
    )
    pose_hash = hashlib.sha256(pose_path.read_bytes()).hexdigest()
    manifest = {
        "version": 2,
        "output_mode": args.output_mode,
        "source_dataset": str(dataset),
        "calibration": pose_doc["calibration"],
        "calibration_sha256": calibration_hash,
        "poses_sha256": pose_hash,
        "episodes": {},
    }
    output_files = list(out_dir.rglob("episode_*.mp4"))
    if args.output_mode == "dataset":
        output_files.extend(out_dir.rglob("episode_*.parquet"))
    episodes_with_outputs = set()
    for path in output_files:
        suffix = path.name.split("episode_", maxsplit=1)[-1]
        try:
            episodes_with_outputs.add(int(suffix[:6]))
        except ValueError:
            continue
    if manifest_path.is_file():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if not args.overwrite:
                raise RuntimeError(
                    f"Vorhandenes Replay-Manifest ist nicht lesbar: {manifest_path}. "
                    "REPLAY_OVERWRITE=1 oder neues Ausgabeverzeichnis verwenden."
                ) from exc
            existing_manifest = {}
        compatibility_errors = manifest_compatibility_errors(existing_manifest, manifest)
        existing_episodes = existing_manifest.get("episodes")
        untracked_outputs = sorted(
            episode for episode in episodes_with_outputs
            if not isinstance(existing_episodes, dict)
            or existing_episodes.get(str(episode), {}).get("status") != "ok"
        )
        if untracked_outputs:
            compatibility_errors.append(
                "Replay-Dateien ohne Manifest-Episode: "
                + ", ".join(str(episode) for episode in untracked_outputs)
            )
        if compatibility_errors and not args.overwrite:
            raise RuntimeError(
                f"Vorhandene Replay-Ausgabe ist nicht mit Manifest v2 kompatibel: "
                f"{'; '.join(compatibility_errors)}. REPLAY_OVERWRITE=1 oder neues "
                "Ausgabeverzeichnis verwenden."
            )
        if not compatibility_errors:
            manifest["episodes"].update(existing_manifest.get("episodes", {}))
    elif output_files and not args.overwrite:
        raise RuntimeError(
            f"Replay-Dateien existieren, aber das zugehörige Manifest v2 fehlt: "
            f"{manifest_path}. REPLAY_OVERWRITE=1 oder neues Ausgabeverzeichnis verwenden."
        )

    env = G1Dex3BlockstackEnv(cfg=cfg, render_mode=None)
    try:
        for ordinal, (ep, (actions, states, task_index)) in enumerate(loaded.items(), 1):
            parquet, paths = (dataset_paths(out_dir, ep) if args.output_mode == "dataset"
                              else (None, output_paths(out_dir, ep)))
            complete = all(path.is_file() for path in paths.values())
            complete = complete and (parquet is None or parquet.is_file())
            if not args.overwrite and complete:
                previous_record = manifest["episodes"].get(str(ep))
                if not isinstance(previous_record, dict):
                    raise RuntimeError(f"Episode {ep}: Manifest-v2-Eintrag fehlt.")
                manifest["episodes"][str(ep)] = mark_episode_existing(previous_record)
                print(f"[replay-render] ({ordinal}/{len(loaded)}) Episode {ep}: vorhanden.")
                continue

            source_hash = action_sha256(actions)
            source_actions = actions.copy()
            env.reset()
            set_robot_state(env, states[0])
            pose_episode = pose_doc["episodes"][str(ep)]
            blocks = pose_episode["blocks"]
            expected_pick = infer_expected_first_pick(pose_episode, calibration, ep)
            place_cubes_once(env, blocks)
            obs = render_current(env)
            head_projection = validate_initial_projection(obs, blocks, report_dir, ep)
            projection_ok = head_projection["samples"] == 6 \
                and head_projection["median_px"] <= 5.0 \
                and head_projection["p90_px"] <= 10.0
            if not projection_ok:
                message = (
                    f"Episode {ep}: Renderer-Abweichung Median/P90 Head "
                    f"{head_projection['median_px']:.1f}/"
                    f"{head_projection['p90_px']:.1f} px"
                )
                if args.output_mode == "dataset":
                    manifest["episodes"][str(ep)] = {
                        "status": "rejected",
                        "render_status": "rejected",
                        "reason": "renderer_projection_mismatch",
                        "renderer_projection": {"head": head_projection},
                        "grasp_validation": {
                            "status": "unavailable",
                            "grasp_success": False,
                            "reason": "render_rejected_before_replay",
                            "expected_first_pick": expected_pick,
                        },
                    }
                    print(f"[replay-render] {message}; Episode verworfen.", flush=True)
                    continue
                print(
                    f"[replay-render] WARNUNG: {message}; Videomodus läuft zur "
                    "manuellen Sichtprüfung weiter.",
                    flush=True,
                )
            writers, temporary = open_writers(paths, fps)
            achieved = np.zeros_like(actions, dtype=np.float32)
            # T Video-/State-Beobachtungen plus der Physikzustand direkt nach action[T-1].
            block_position_trace = np.zeros((len(actions) + 1, 3, 3), dtype=np.float32)
            fingertip_position_trace = np.zeros((len(actions) + 1, 6, 3), dtype=np.float32)
            wrist_expected = wrist_schedule(blocks)
            wrist_details = []
            try:
                try:
                    # Zeile t enthält Beobachtung/State vor action[t], exakt wie im
                    # Quelldatensatz. Auch die letzte Action wird gesendet; nur ihr
                    # Folgezustand liegt definitionsgemäß außerhalb der Episode.
                    for frame_idx in range(len(actions)):
                        append_observation(writers, obs)
                        if frame_idx in wrist_expected:
                            wrist_details.extend(measure_wrist_projection(
                                obs, wrist_expected[frame_idx], report_dir, ep, frame_idx
                            ))
                        block_positions, fingertip_positions = read_physics_positions(env)
                        block_position_trace[frame_idx] = block_positions
                        fingertip_position_trace[frame_idx] = fingertip_positions
                        achieved[frame_idx] = obs["joint_pos"][0].detach().cpu().numpy()
                        action_row = actions[frame_idx].copy()
                        action = torch.from_numpy(action_row).to(env.device).unsqueeze(0)
                        obs, _, _, _, _ = env.step(action)
                        if not np.array_equal(action_row, actions[frame_idx]):
                            raise RuntimeError(f"Episode {ep}: action[{frame_idx}] verändert")
                        if frame_idx % 200 == 0:
                            print(
                                f"      Episode {ep}: Frame {frame_idx}/{len(actions)}",
                                flush=True,
                            )
                    final_blocks, final_fingertips = read_physics_positions(env)
                    block_position_trace[-1] = final_blocks
                    fingertip_position_trace[-1] = final_fingertips
                finally:
                    for writer in writers.values():
                        writer.close()
            except Exception:
                for tmp_path in temporary.values():
                    tmp_path.unlink(missing_ok=True)
                raise

            wrist_projection = summarize_wrist_projection(wrist_details)
            grasp_validation = summarize_grasp_trace(
                block_position_trace, fingertip_position_trace, expected_pick
            )
            projection = {"head": head_projection, "wrist": wrist_projection}
            wrist_cameras_ok = all(
                camera in wrist_projection["validated_cameras"]
                for camera in ("cam_left_wrist", "cam_right_wrist")
            )
            wrist_ok = wrist_projection["samples"] >= 2 and wrist_cameras_ok \
                and wrist_projection["median_px"] <= 10.0 \
                and wrist_projection["p90_px"] <= 20.0
            if not wrist_ok:
                message = (
                    f"Episode {ep}, Wrist-Projektion Median/P90 "
                    f"{wrist_projection['median_px']:.1f}/"
                    f"{wrist_projection['p90_px']:.1f} px"
                )
                if args.output_mode == "dataset":
                    for tmp_path in temporary.values():
                        tmp_path.unlink(missing_ok=True)
                    manifest["episodes"][str(ep)] = {
                        "status": "rejected",
                        "render_status": "rejected",
                        "reason": "wrist_projection_mismatch",
                        "renderer_projection": projection,
                        "grasp_validation": grasp_validation,
                    }
                    print(f"[replay-render] {message}; Episode verworfen.", flush=True)
                    continue
                print(
                    f"[replay-render] WARN: {message}; Videomodus läuft weiter.",
                    flush=True,
                )
            for tmp_path in temporary.values():
                if not tmp_path.is_file():
                    raise RuntimeError(f"MP4-Schreiber lieferte keine Datei: {tmp_path}")
                with imageio.get_reader(str(tmp_path), format="FFMPEG") as reader:
                    video_frames = int(reader.count_frames())
                if video_frames != len(actions):
                    for candidate in temporary.values():
                        candidate.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Episode {ep}/{tmp_path.name}: {video_frames} statt "
                        f"{len(actions)} Frames."
                    )
            for cam, tmp_path in temporary.items():
                tmp_path.replace(paths[cam])
            final_hash = action_sha256(actions)
            if source_hash != final_hash or not np.array_equal(actions, source_actions):
                raise RuntimeError(f"Episode {ep}: Originalaktionen wurden verändert.")
            if parquet is not None:
                write_dataset_parquet(
                    parquet, ep, achieved, actions, task_index, fps, 0
                )
            manifest["episodes"][str(ep)] = {
                "status": "ok",
                "render_status": "ok",
                "frames": len(actions),
                "action_sha256": source_hash,
                "actions_unchanged": True,
                "task_index": task_index,
                "renderer_projection": projection,
                "grasp_validation": grasp_validation,
                "cube_poses": blocks,
            }
            print(f"[replay-render] ({ordinal}/{len(loaded)}) Episode {ep}: "
                  f"{len(actions)} Frames, {len(paths)} MP4s.", flush=True)

        successful = sum(
            record.get("status") == "ok" for record in manifest["episodes"].values()
        )
        if successful == 0:
            raise RuntimeError(
                "Keine Episode bestand die Renderer-Projektionsprüfung; Berichte unter "
                f"{report_dir}."
            )
        if args.output_mode == "dataset":
            finalize_dataset(out_dir, dataset, info, manifest, fps)
            validate_dataset(out_dir, manifest)
        else:
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "replay_manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
            )

        # Isaac Lab 3 / Isaac Sim 6 kann den Python-Ablauf bereits in env.close() beenden.
        # Der Host-Wrapper erkennt Erfolg deshalb an einem Marker VOR dem Cleanup. Zu diesem
        # Zeitpunkt sind alle Writer geschlossen und alle .tmp.mp4 atomar umbenannt.
        label = "LeRobot-Datensatz" if args.output_mode == "dataset" else "Videos"
        print(f"[replay-render] {label}: {out_dir}")
        print("[replay-render] fertig.", flush=True)
    finally:
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
