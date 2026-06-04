"""
Closed-Loop-Baseline-Eval für **un-finetuntes GR00T-N1.6-3B** auf dem
**stock Unitree G1 + Dex1-Greifer** (Embodiment ``UNITREE_G1``) in Isaac Lab.

Spiegelt ``g1_dex3_sim/run_g1_dex3_sim_eval.py``; nutzt aber das Gripper-Env,
client_g1 und EINE ego_view-Kamera. Eigenständig — der DEX3-Pfad bleibt unberührt.

Zweck: Baseline-Vergleich (Success-Rate) gegen den DEX3-Fine-Tune auf demselben
Block-Stacking-Task. Das Basismodell läuft hier zero-shot und das UNITREE_G1-
Embodiment (Loco-Manip, eine Ego-Kamera) wird auf eine fixierte Tabletop-Szene
gezwungen → stark out-of-distribution; eine Erfolgsrate ≈ 0 % ist erwartbar und
genau der gesuchte untere Baseline-Wert.

Verwendung (im Sim-Container, GR00T-Server muss mit --embodiment-tag UNITREE_G1 laufen):
    ${ISAACLAB_PATH}/isaaclab.sh -p g1_gripper_sim/run_g1_gripper_sim_eval.py \\
        --headless --server tcp://localhost:5555 \\
        --dims-file /data/g1_baseline_dims.json \\
        --num-episodes 20 --asset-path /workspace/assets/g1_gripper.usd
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI-Argumente
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="GR00T UNITREE_G1 Baseline Closed-Loop Sim Eval")
parser.add_argument("--server", type=str, default="tcp://localhost:5555",
                    help="ZMQ-Adresse des GR00T-Policy-Servers")
parser.add_argument("--dims-file", type=str, default="/data/g1_baseline_dims.json",
                    help="JSON mit Pro-Gruppe-State-/Action-Dims (dump_unitree_g1_dims.py)")
parser.add_argument("--num-episodes", type=int, default=20, help="Anzahl Eval-Episoden")
parser.add_argument("--execution-horizon", type=int, default=8,
                    help="Wie viele Steps eines Chunks ausgeführt werden, bevor neu geplant wird")
parser.add_argument("--task-description", type=str, default="stack the blocks",
                    help="Language-Prompt für die Policy")
parser.add_argument("--video-dir", type=str, default="/data/sim_videos",
                    help="Verzeichnis für Rollout-Videos (leer = keine Videos)")
parser.add_argument("--results-file", type=str, default="/data/sim_results/results.json",
                    help="JSON-Datei für Eval-Ergebnisse")
parser.add_argument("--ping-retries", type=int, default=20,
                    help="Wie oft auf den Server gewartet wird (je 5 s)")
parser.add_argument("--asset-path", type=str, default="",
                    help="Pfad zum G1+Dex1 USD-Asset (überschreibt den cfg-Default)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports (nach AppLauncher-Start)
# ---------------------------------------------------------------------------

import numpy as np
import torch

from client_g1 import PolicyClient, build_obs, load_dims
from g1_gripper_blockstack_env import G1GripperBlockstackEnv, G1GripperBlockstackEnvCfg
from g1_gripper_cfg import CAMERA_CFG


def save_episode_video(frames: list[np.ndarray], episode: int, video_dir: str) -> None:
    if not frames:
        return
    try:
        import imageio
        os.makedirs(video_dir, exist_ok=True)
        path = os.path.join(video_dir, f"episode_{episode:04d}.mp4")
        imageio.mimwrite(path, frames, fps=30, quality=8)
        print(f"[Video] Gespeichert: {path}")
    except Exception as e:
        print(f"[Video] Fehler beim Speichern: {e}")


def run_episode(env, client, dims, execution_horizon, task_description,
                record_video, episode, video_dir) -> dict:
    env.reset()
    client.reset()

    frames: list[np.ndarray] = []
    step = 0
    success = False
    success_step = -1
    t_start = time.perf_counter()
    max_steps = int(env.cfg.episode_length_s * env.cfg.policy_hz)

    while step < max_steps:
        obs_np = env.get_obs_for_policy()
        policy_obs = build_obs(
            ego_view=obs_np["video.ego_view"],
            arm_pos=obs_np["arm_pos"],
            gripper=obs_np["gripper"],
            dims=dims,
            task_description=task_description,
        )

        # Debug (einmalig): erste ego_view-Obs dumpen, um die Kamera-Pose zu prüfen.
        if episode == 1 and step == 0 and video_dir:
            try:
                import imageio
                os.makedirs(video_dir, exist_ok=True)
                arr = np.asarray(obs_np["video.ego_view"]).astype(np.uint8)
                imageio.imwrite(os.path.join(video_dir, "_debug_obs_ego_view.png"), arr)
                print("[Debug] Erste ego_view-Obs gespeichert: _debug_obs_ego_view.png", flush=True)
            except Exception:
                pass

        try:
            chunk = client.get_action(policy_obs)  # (T, 16)
        except Exception as e:
            print(f"[Episode] Fehler bei get_action (Step {step}): {e}")
            break

        for t in range(min(execution_horizon, len(chunk))):
            action_t = torch.tensor(chunk[t], dtype=torch.float32,
                                    device=env.device).unsqueeze(0)  # (1, 16)
            obs_step, _, terminated, time_out, _ = env.step(action_t)

            if record_video:
                cam_key = "video.cam_scene" if "video.cam_scene" in obs_step else "video.ego_view"
                frame = obs_step[cam_key][0].cpu().numpy().astype(np.uint8)
                frames.append(frame)

            step += 1
            if step % 25 == 0:
                print(f"    … Step {step}/{max_steps} ({time.perf_counter() - t_start:.0f}s)",
                      flush=True)

            if terminated.any() and not success:
                success = True
                success_step = step
                print(f"[Episode] Erfolg bei Step {step}!")

            if terminated.any() or time_out.any():
                break

        if terminated.any() or time_out.any():
            break

    duration = time.perf_counter() - t_start
    if record_video and video_dir:
        save_episode_video(frames, episode, video_dir)

    return {
        "success": success,
        "num_steps": step,
        "duration_s": round(duration, 2),
        "success_step": success_step,
    }


def main():
    dims = load_dims(args.dims_file)
    print(f"[dims] Geladen: state={dims['state']} action={dims['action']} "
          f"horizon={dims.get('action_horizon')}")

    cfg = G1GripperBlockstackEnvCfg()
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    cfg.execution_horizon = args.execution_horizon
    cfg.task_description = args.task_description

    print("=" * 60)
    print("GR00T N1.6 — UNITREE_G1 Baseline Closed-Loop Sim Eval (stock G1 + Dex1)")
    print("=" * 60)
    print(f"  Server:           {args.server}")
    print(f"  Episoden:         {args.num_episodes}")
    print(f"  Exec-Horizon:     {args.execution_horizon}")
    print(f"  Task:             {args.task_description}")
    print(f"  Video-Dir:        {args.video_dir or '(kein Video)'}")
    print(f"  Asset:            {cfg.scene.robot.spawn.usd_path}")
    print()

    print("[Phase A] Isaac-Lab-Sim wird initialisiert …")
    env = G1GripperBlockstackEnv(cfg=cfg, render_mode="rgb_array" if args.video_dir else None)
    print("[Phase A] Sim läuft.")

    print(f"[Phase D] Verbinde mit GR00T-Server: {args.server}")
    client = PolicyClient(dims=dims, server_url=args.server, timeout_ms=20_000)

    if not client.ping(retries=args.ping_retries, delay=5.0):
        print("[FEHLER] Server nicht erreichbar. Abbruch.")
        env.close()
        sys.exit(1)
    print("[Phase D] Server erreichbar — starte Eval-Schleife.")
    print()

    # Dummy-Test: synthetische Obs → Action-Chunk
    dummy_rgb = np.zeros((CAMERA_CFG.height, CAMERA_CFG.width, 3), dtype=np.uint8)
    dummy_arm = np.zeros(14, dtype=np.float32)
    dummy_grip = np.zeros(2, dtype=np.float32)
    dummy_obs = build_obs(dummy_rgb, dummy_arm, dummy_grip, dims)
    dummy_chunk = client.get_action(dummy_obs)
    assert dummy_chunk.shape[-1] == 16, f"Unerwartete Chunk-Shape: {dummy_chunk.shape}"
    print(f"[Phase D] Dummy-get_action OK: shape={dummy_chunk.shape}")
    print()

    results = []
    record = bool(args.video_dir)
    for ep in range(args.num_episodes):
        print(f"--- Episode {ep + 1}/{args.num_episodes} ---")
        ep_result = run_episode(env, client, dims, args.execution_horizon,
                                args.task_description, record, ep + 1, args.video_dir)
        ep_result["episode"] = ep + 1
        results.append(ep_result)
        status = "ERFOLG" if ep_result["success"] else "misslungen"
        print(f"  Episode {ep + 1}: {status} | {ep_result['num_steps']} Steps | "
              f"{ep_result['duration_s']:.1f}s")

    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / len(results) if results else 0.0
    print()
    print("=" * 60)
    print(f"Ergebnis: {n_success}/{len(results)} Erfolge — Success-Rate: {success_rate:.1%}")
    print("=" * 60)

    summary = {
        "embodiment": "unitree_g1",
        "baseline": True,
        "num_episodes": len(results),
        "num_success": n_success,
        "success_rate": success_rate,
        "server": args.server,
        "task_description": args.task_description,
        "execution_horizon": args.execution_horizon,
        "episodes": results,
    }
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Ergebnisse gespeichert: {args.results_file}")

    client.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
