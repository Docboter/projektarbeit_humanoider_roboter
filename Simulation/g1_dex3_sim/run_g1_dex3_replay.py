"""
Open-Loop-REPLAY-Diagnose für G1 + Dex3 in Isaac Lab.

Zweck: trennt eindeutig "Sim/Config-Fehler" von "Modell zu wenig trainiert".
Statt das GR00T-Modell zu befragen, werden die ECHTEN aufgezeichneten Dataset-Aktionen
(Episode 0 von unitreerobotics/G1_Dex3_BlockStacking_Dataset) direkt in dieselbe Env
gespeist. Kein Server, kein Modell.

Interpretation:
  - Fährt der Roboter die Arme NACH VORNE zum Tisch und schließt die Finger (greif-artige
    Bewegung im Tischbereich) → Sim/Config führt korrekte Aktionen korrekt aus → das
    Wegdriften im Closed-Loop liegt am Modell (Training), NICHT an der Config.
  - Driftet der Roboter auch beim Replay weg / bewegt sich unsinnig → es steckt doch ein
    Sim-Problem drin (Reachability, Konvention, Skalierung), das gezielt zu fixen ist.

Hinweis: Die Würfel-Startpositionen der Env entsprechen NICHT exakt der realen Episode 0
(Objekt-Posen sind im Dataset nicht gespeichert). Aussagekräftig ist daher die ARM-/FINGER-
BEWEGUNG (vorwärts-greifen vs. wegdriften), nicht zwingend ein erfolgreicher Stapel.

Vollständig additiv: überschreibt nichts vom Modell-Eval. Eigene Default-Ausgabepfade
(/data/sim_videos_replay, /data/sim_results_replay). Parallel zum Modell-Eval nutzbar.

Verwendung (im Sim-Container):
    unset VIRTUAL_ENV
    ${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/run_g1_dex3_replay.py \\
        --headless --enable_cameras \\
        --asset-path /data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI-Argumente (eigene, getrennte Defaults — kein Konflikt mit dem Modell-Eval)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="G1-Dex3 Open-Loop Dataset-Replay (Diagnose)")
parser.add_argument(
    "--actions-file", type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_episode0.npz"),
    help="npz mit 'action' (T,28) — echte Dataset-Aktionen (default: gebündelte Episode 0)",
)
parser.add_argument("--video-dir", type=str, default="/data/sim_videos_replay",
                    help="Verzeichnis für das Replay-Video (getrennt vom Modell-Eval)")
parser.add_argument("--results-file", type=str, default="/data/sim_results_replay/results.json")
parser.add_argument("--max-steps", type=int, default=0, help="0 = alle aufgezeichneten Frames")
parser.add_argument("--asset-path", type=str, default="/workspace/assets/g1_dex3.usd")
parser.add_argument("--grasp-test", action="store_true",
                    help="Würfel exakt an die aufgezeichneten Greifpunkte setzen (statt Zufall), "
                         "um die Greif-Physik zu prüfen: wird ein Würfel angehoben?")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports nach AppLauncher-Start
# ---------------------------------------------------------------------------

import numpy as np
import torch

from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg


def save_video(frames, path):
    if not frames:
        print("[Replay] Keine Frames — kein Video gespeichert.")
        return
    try:
        import imageio
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimwrite(path, frames, fps=30, quality=8)
        print(f"[Replay] Video gespeichert: {path}")
    except Exception as e:
        print(f"[Replay] Fehler beim Video-Speichern: {e}")


def main():
    data = np.load(args.actions_file)
    actions = data["action"].astype(np.float32)  # (T, 28) absolute Ziel-Gelenkpositionen
    n = actions.shape[0] if args.max_steps <= 0 else min(args.max_steps, actions.shape[0])

    cfg = G1Dex3BlockstackEnvCfg()
    if args.asset_path != "/workspace/assets/g1_dex3.usd":
        cfg.scene.robot.spawn.usd_path = args.asset_path
    # Episodenlänge hochsetzen, damit der Replay nicht durch Time-Out auto-resettet.
    cfg.episode_length_s = (n / cfg.policy_hz) + 5.0

    print("=" * 60)
    print("G1+Dex3 — OPEN-LOOP DATASET-REPLAY (Diagnose Config vs. Training)")
    print("=" * 60)
    print(f"  Aktionen:     {args.actions_file}")
    print(f"  Frames:       {n} / {actions.shape[0]}")
    print(f"  Asset:        {cfg.scene.robot.spawn.usd_path}")
    print(f"  Video-Dir:    {args.video_dir}")
    print()

    record = bool(args.video_dir)
    env = G1Dex3BlockstackEnv(cfg=cfg, render_mode="rgb_array" if record else None)
    print("[Replay] Sim läuft. Spiele aufgezeichnete Aktionen ab …", flush=True)

    env.reset()

    # Greif-Test: Würfel exakt an die im Diagnose-Lauf gemessenen Greifpunkte setzen
    # (linke Hand x≈0.35,y≈0.20 / rechte Hand x≈0.36,y≈-0.18), auf der neuen Tischhöhe.
    # Damit schließen die aufgezeichneten Finger genau um die Würfel → testet die Greif-Physik.
    if args.grasp_test:
        z = float(env.cfg.block_z_surface)
        grasp_pts = [(0.35, 0.20, z), (0.36, -0.18, z), (0.35, 0.0, z)]
        for blk, (gx, gy, gz) in zip(env.blocks, grasp_pts):
            pose = torch.tensor([[gx, gy, gz, 1.0, 0.0, 0.0, 0.0]],
                                device=env.device, dtype=torch.float32)
            blk.write_root_pose_to_sim(pose)
            blk.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
        print("[Replay] GRASP-TEST: Würfel an die aufgezeichneten Greifpunkte gesetzt.", flush=True)

    frames = []
    success = False
    success_step = -1
    arm_err = []          # mittlerer |kommandiert - erreicht| pro Step (Arm-Gelenke 0:14)
    per_joint_max = np.zeros(28, dtype=np.float32)

    # Greif-Diagnose: kommt eine Hand nah an einen Würfel, und hebt sich ein Würfel?
    body_names = list(env.robot.data.body_names)
    palm_idx = [body_names.index(b) for b in ("left_hand_palm_link", "right_hand_palm_link")
                if b in body_names]
    init_cube_z = np.array([float(b.data.root_pos_w[0, 2].cpu()) for b in env.blocks])
    min_hand_cube = float("inf")
    max_cube_lift = 0.0
    # tiefster Greifpunkt je Hand (zeigt, WO ein Würfel liegen müsste, um die Greif-Physik zu testen)
    hand_low = [np.array([0.0, 0.0, 1e9]) for _ in palm_idx]

    for i in range(n):
        action_t = torch.tensor(actions[i], dtype=torch.float32, device=env.device).unsqueeze(0)
        obs_step, _, terminated, time_out, info = env.step(action_t)

        # Tracking-Fehler: folgt der Roboter den kommandierten Gelenkwinkeln?
        achieved = obs_step["joint_pos"][0].cpu().numpy()
        err = np.abs(achieved - actions[i])
        arm_err.append(float(err[:14].mean()))
        per_joint_max = np.maximum(per_joint_max, err)

        # Greif-Diagnose: min Hand→Würfel-Distanz + maximale Würfel-Anhebung
        if palm_idx:
            hands = env.robot.data.body_pos_w[0, palm_idx].cpu().numpy()           # (H,3)
            cubes = np.array([b.data.root_pos_w[0].cpu().numpy() for b in env.blocks])  # (3,3)
            d = np.linalg.norm(hands[:, None, :] - cubes[None, :, :], axis=-1).min()
            min_hand_cube = min(min_hand_cube, float(d))
            max_cube_lift = max(max_cube_lift, float((cubes[:, 2] - init_cube_z).max()))
            for h in range(len(palm_idx)):
                if hands[h, 2] < hand_low[h][2]:
                    hand_low[h] = hands[h].copy()

        if record:
            cam_key = "video.cam_scene" if "video.cam_scene" in obs_step else "video.cam_left_high"
            frames.append(obs_step[cam_key][0].cpu().numpy().astype(np.uint8))

        if terminated.any() and not success:
            success = True
            success_step = i
            print(f"[Replay] _check_success bei Step {i}! (Würfel gestapelt)", flush=True)

        if i % 100 == 0:
            print(f"    … Step {i}/{n}", flush=True)

    mean_arm_err = float(np.mean(arm_err)) if arm_err else 0.0
    print(f"\n[Replay] TRACKING-FEHLER Arm-Gelenke: mittel={mean_arm_err:.3f} rad")
    print(f"[Replay]   worst joints (max |Ziel-Ist| rad): "
          f"L_sh_pitch={per_joint_max[0]:.2f} L_elbow={per_joint_max[3]:.2f} "
          f"R_sh_pitch={per_joint_max[7]:.2f} R_elbow={per_joint_max[10]:.2f}", flush=True)
    print("[Replay]   > ~0.3 rad mittel = Arme folgen NICHT (PD-Gains zu schwach = Sim-Bug);"
          " < ~0.1 = Tracking ok (dann Geometrie/Modell).", flush=True)
    print(f"[Replay] GREIF-DIAGNOSE: min Hand→Würfel-Distanz = {min_hand_cube*100:.1f} cm | "
          f"max Würfel-Anhebung = {max_cube_lift*100:.1f} cm", flush=True)
    print("[Replay]   Distanz groß (>~15cm) → Greifbewegung trifft unsere Würfel nicht (Platzierung);"
          " Distanz klein + Anhebung≈0 → Greif-Physik prüfen; Anhebung>~2cm → Greifen FUNKTIONIERT.",
          flush=True)
    lbl = ["left", "right"]
    for h in range(len(palm_idx)):
        p = hand_low[h]
        print(f"[Replay]   tiefster {lbl[h] if h < 2 else h}-Hand-Punkt (Welt): "
              f"x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}", flush=True)
    cz = float(init_cube_z[0])
    print(f"[Replay]   (Würfel-Oberseite liegt bei z≈{cz:.3f}; aktuelle Würfel-XY: "
          f"{[tuple(round(float(c),2) for c in b.data.root_pos_w[0,:2].cpu()) for b in env.blocks]})",
          flush=True)

    if record:
        save_video(frames, os.path.join(args.video_dir, "replay_episode0.mp4"))

    result = {
        "mode": "open_loop_dataset_replay",
        "episode": 0,
        "num_steps": n,
        "success": success,
        "success_step": success_step,
        "mean_arm_tracking_error_rad": round(mean_arm_err, 4),
        "per_joint_max_error_rad": [round(float(x), 3) for x in per_joint_max],
        "min_hand_cube_dist_cm": round(min_hand_cube * 100, 1),
        "max_cube_lift_cm": round(max_cube_lift * 100, 1),
        "note": "Würfel-Posen entsprechen nicht exakt der realen Episode; "
                "aussagekräftig ist die Arm-/Finger-Bewegung (vorwärts-greifen vs. wegdriften).",
    }
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Replay] Ergebnis: success={success} (Step {success_step}) → {args.results_file}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
