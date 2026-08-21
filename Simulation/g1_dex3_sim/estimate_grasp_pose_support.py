#!/usr/bin/env python3
"""Add sparse grasp-position support to already reconstructed CV cube poses."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--poses", required=True)
parser.add_argument("--asset-path", default="")
parser.add_argument("--num-episodes", type=int, default=10)
parser.add_argument("--start-episode", type=int, default=0)
parser.add_argument("--episode-ids", type=int, nargs="*", default=None)
parser.add_argument("--train-ratio", type=float, default=0.8)
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
from grasp_pose_support import (  # noqa: E402
    apply_grasp_support,
    closure_candidates,
    fingertip_measurement,
)
from reconstruct_cube_poses import data_path, read_info, select_episodes  # noqa: E402
from replay_calibration import action_sha256  # noqa: E402


def load_episode(root: Path, info: dict, episode: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(data_path(root, info, episode))
    actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
    states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    if actions.shape != states.shape or actions.ndim != 2 or actions.shape[1:] != (28,):
        raise ValueError(
            f"Episode {episode}: action/state nicht (T,28): {actions.shape}/{states.shape}"
        )
    return actions, states


def set_robot_state(env: G1Dex3BlockstackEnv, state: np.ndarray) -> None:
    if env._joint_ids is None:
        env._joint_ids = env._build_joint_id_mapping()
    q = np.asarray(state, dtype=np.float32).copy()
    q[env._SIGN_FLIP_IDX] *= -1
    full = env.robot.data.joint_pos.clone()
    for policy_index, isaac_index in enumerate(env._joint_ids):
        full[:, isaac_index] = float(q[policy_index])
    velocity = torch.zeros_like(full)
    env.robot.write_joint_state_to_sim(full, velocity)
    env.robot.set_joint_position_target(full)
    env.sim.forward()
    env.robot.update(float(env.cfg.sim.dt))


def stash_cubes(env: G1Dex3BlockstackEnv) -> None:
    for index, block in enumerate(env.blocks):
        pose = torch.tensor(
            [[5.0 + index, 5.0, 0.1, 1.0, 0.0, 0.0, 0.0]],
            device=env.device,
            dtype=torch.float32,
        )
        block.write_root_pose_to_sim(pose)
        block.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))


def hand_measurement(
    env: G1Dex3BlockstackEnv, hand_index: int
) -> tuple[float, np.ndarray]:
    tips = env.get_contact_points_w()[0].detach().cpu().numpy()
    if tips.shape != (6, 3) or env._reach_frame != "fingertip":
        raise RuntimeError(
            "Greifstütze benötigt sechs echte Fingerkuppen; "
            f"erhalten: frame={env._reach_frame}, shape={tips.shape}."
        )
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    selected = tips[hand_index * 3 : hand_index * 3 + 3] - origin
    return fingertip_measurement(selected)


def sparse_grasp_events(
    env: G1Dex3BlockstackEnv, states: np.ndarray
) -> tuple[list[dict], int, dict]:
    events = []
    evaluated = 0
    diagnostics = {}
    for hand_index, hand in enumerate(("left", "right")):
        candidates = closure_candidates(states, hand)
        hand_diagnostics = {"candidates": candidates, "rejections": []}
        diagnostics[hand] = hand_diagnostics
        for candidate in candidates:
            set_robot_state(env, states[candidate["start_frame"]])
            open_spread, _ = hand_measurement(env, hand_index)
            set_robot_state(env, states[candidate["end_frame"]])
            closed_spread, centroid = hand_measurement(env, hand_index)
            evaluated += 2
            decrease = open_spread - closed_spread
            reason = ""
            if decrease < 0.006:
                reason = f"Fingeröffnung nimmt nur um {decrease:.4f} m ab"
            elif not (
                0.20 <= centroid[0] <= 0.50
                and -0.30 <= centroid[1] <= 0.30
                and 0.82 <= centroid[2] <= 1.05
            ):
                reason = "Fingerkuppenschwerpunkt außerhalb des Arbeitsbereichs"
            if reason:
                hand_diagnostics["rejections"].append({**candidate, "reason": reason})
                continue
            event = {
                "hand": hand,
                "closure_start_frame": int(candidate["start_frame"]),
                "grasp_frame": int(candidate["end_frame"]),
                "coordinated_joints": int(candidate["coordinated_joints"]),
                "finger_opening_before_m": float(open_spread),
                "finger_opening_at_grasp_m": float(closed_spread),
                "finger_closure_m": float(decrease),
                "fingertip_centroid_m": [float(value) for value in centroid],
            }
            events.append(event)
            hand_diagnostics["accepted_candidate"] = event
            break
    return events, evaluated, diagnostics


def main() -> int:
    root = Path(args.dataset_path)
    info = read_info(root)
    episodes = select_episodes(info, args)
    pose_path = Path(args.poses)
    document = json.loads(pose_path.read_text(encoding="utf-8"))
    selected = [
        episode
        for episode in episodes
        if document.get("episodes", {}).get(str(episode), {}).get("status") == "ok"
    ]
    if not selected:
        print("[replay-grasp-support] Keine CV-gültige Episode ausgewählt.")
        print("[replay-grasp-support] fertig.", flush=True)
        simulation_app.close()
        return 0

    cfg = G1Dex3BlockstackEnvCfg()
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    cfg.sim.render_interval = 1_000_000
    cfg.episode_length_s = 3600.0
    cfg.terminate_on_success = False
    env = G1Dex3BlockstackEnv(cfg=cfg, render_mode=None)
    total_accepted = 0
    total_fk = 0
    try:
        env.reset()
        stash_cubes(env)
        for ordinal, episode in enumerate(selected, 1):
            started = time.perf_counter()
            actions, states = load_episode(root, info, episode)
            source_hash = action_sha256(actions)
            source_copy = actions.copy()
            events, evaluated, diagnostics = sparse_grasp_events(env, states)
            episode_record = document["episodes"][str(episode)]
            accepted = apply_grasp_support(episode_record["blocks"], events)
            episode_record["grasp_support"] = {
                "enabled": True,
                "accepted": accepted,
                "fk_states": evaluated,
                "events": events,
                "diagnostics": diagnostics,
                "action_sha256": source_hash,
                "actions_unchanged": bool(
                    np.array_equal(actions, source_copy)
                    and source_hash == action_sha256(actions)
                ),
            }
            if not episode_record["grasp_support"]["actions_unchanged"]:
                raise RuntimeError(f"Episode {episode}: Originalaktionen wurden verändert.")
            total_accepted += accepted
            total_fk += evaluated
            print(
                f"[replay-grasp-support] ({ordinal}/{len(selected)}) Episode {episode}: "
                f"CV 3/3; {accepted} Greifstützen; {evaluated} FK-Zustände; "
                f"{time.perf_counter() - started:.1f} s.",
                flush=True,
            )
        temporary = pose_path.with_suffix(pose_path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
        temporary.replace(pose_path)
        print(
            f"[replay-grasp-support] Gesamt: {total_accepted} Greifstützen, "
            f"{total_fk} FK-Zustände."
        )
        print(f"[replay-grasp-support] {pose_path}")
        print("[replay-grasp-support] fertig.", flush=True)
    finally:
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())