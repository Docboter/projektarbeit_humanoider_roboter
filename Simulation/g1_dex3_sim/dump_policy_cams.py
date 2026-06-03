"""
Dump-Skript: speichert je ein RGB-Standbild der vier POLICY-Kameras der Isaac-Lab-Env.

Zweck: Domain-Gap-Diagnose. Die so erzeugten Sim-Frames (cam_left_high, cam_right_high,
cam_left_wrist, cam_right_wrist) werden neben die ECHTEN Dataset-Referenzframes aus
Simulation/camera_reference/dataset_cam_*.png gelegt. Sehen sie sichtbar völlig anders aus
(synthetischer weißer Roboter / graue Szene vs. reales Labor-RGB mit schwarzen Händen),
ist der visuelle Domain-Gap belegt — die Erklärung dafür, dass die auf Realbildern
trainierte Policy im Closed-Loop einfriert (eingefrorener Vision-Encoder, OOD-Eingaben).

Kein Modell, kein Server. Reset + ein paar Settle-Steps, dann Frames speichern.

Verwendung (im Sim-Container, RT-Core-GPU):
    unset VIRTUAL_ENV
    ${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/dump_policy_cams.py \\
        --headless --enable_cameras \\
        --asset-path /data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd \\
        --out-dir /data/sim_cam_frames
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump der vier Policy-Kamera-Standbilder (Sim)")
parser.add_argument("--asset-path", type=str, default="/data/assets/g1_dex3_blackhands.usd",
                    help="Pfad zum G1+Dex3 USD-Asset (Default: schwarzhändiges Asset)")
parser.add_argument("--out-dir", type=str, default="/data/sim_cam_frames",
                    help="Ausgabeverzeichnis für die PNG-Standbilder")
parser.add_argument("--settle-steps", type=int, default=8,
                    help="Physik-/Render-Steps nach dem Reset, bevor das Frame gespeichert wird")

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

POLICY_CAMS = ["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]


def main():
    cfg = G1Dex3BlockstackEnvCfg()
    cfg.scene.robot.spawn.usd_path = args.asset_path

    print("=" * 60)
    print("G1+Dex3 — POLICY-KAMERA-DUMP (Domain-Gap-Diagnose)")
    print("=" * 60)
    print(f"  Asset:    {cfg.scene.robot.spawn.usd_path}")
    print(f"  Out-Dir:  {args.out_dir}")
    print(f"  Settle:   {args.settle_steps} Steps")
    print()

    env = G1Dex3BlockstackEnv(cfg=cfg, render_mode=None)
    env.reset()

    # Ein paar Steps mit Null-Aktion, damit die Kameras gerendert sind und die Physik settlet.
    zero_action = torch.zeros((env.num_envs, 28), dtype=torch.float32, device=env.device)
    obs_step = None
    for _ in range(max(1, args.settle_steps)):
        obs_step, _, _, _, _ = env.step(zero_action)

    os.makedirs(args.out_dir, exist_ok=True)
    try:
        import imageio
    except Exception as e:  # noqa: BLE001
        print(f"[Dump] imageio nicht verfügbar: {e}", flush=True)
        env.close()
        return

    for cam in POLICY_CAMS:
        key = f"video.{cam}"
        if key not in obs_step:
            print(f"[Dump] WARN: {key} nicht in Observations — übersprungen.", flush=True)
            continue
        frame = obs_step[key][0].cpu().numpy().astype(np.uint8)  # (H, W, 3)
        path = os.path.join(args.out_dir, f"sim_{cam}.png")
        imageio.imwrite(path, frame)
        print(f"[Dump] gespeichert: {path}  shape={frame.shape}", flush=True)

    env.close()
    print("[Dump] fertig.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
