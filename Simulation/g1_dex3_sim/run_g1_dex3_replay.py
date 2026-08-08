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
parser.add_argument("--asset-path", type=str, default="",
                    help="Pfad zum G1+Dex3 USD-Asset (leer = cfg-Default verwenden)")
parser.add_argument("--grasp-test", action="store_true",
                    help="Würfel exakt an die aufgezeichneten Greifpunkte setzen (statt Zufall), "
                         "um die Greif-Physik zu prüfen: wird ein Würfel angehoben?")
parser.add_argument("--grasp-hold", action="store_true",
                    help="Würfel im Moment des Zugreifens direkt zwischen die Fingerspitzen "
                         "setzen. Nimmt jede Platzierungs- und Timing-Annahme aus dem Test "
                         "heraus und misst nur noch, ob die Hand einen Würfel halten kann.")

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
    if args.asset_path:
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
        # Positionen aus dem tiefsten Handpunkt + ~5 cm Finger-Offset in +x:
        #   links:  Palm(0.294, 0.203) → Würfel (0.35, 0.20)
        #   rechts: Palm(0.333,-0.160) → Würfel (0.37,-0.16)  [war -0.18, zu weit innen]
        grasp_pts = [(0.35, 0.20, z), (0.37, -0.16, z), (0.35, 0.00, z)]
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
    finger_err = []       # dito für die 14 Dex3-Fingergelenke (14:28)
    per_joint_max = np.zeros(28, dtype=np.float32)
    # Spannweite der Fingergelenke, kommandiert gegen erreicht. Bleibt "erreicht" deutlich
    # hinter "kommandiert" zurück, klemmt der Sim die Greifbewegung an einer Gelenkgrenze ab
    # (genau der Fehler, den _widen_finger_joint_limits im Env behebt) — dann liegt es nicht
    # an Reibung. Sind beide groß und der Würfel bleibt trotzdem liegen, ist es Kontakt/
    # Reibung oder der Würfel liegt gar nicht in der Greiföffnung.
    cmd_lo, cmd_hi = np.full(14, np.inf), np.full(14, -np.inf)
    ach_lo, ach_hi = np.full(14, np.inf), np.full(14, -np.inf)

    # Greif-Diagnose: kommt eine Hand nah an einen Würfel, und hebt sich ein Würfel?
    body_names = list(env.robot.data.body_names)
    palm_idx = [body_names.index(b) for b in ("left_hand_palm_link", "right_hand_palm_link")
                if b in body_names]
    init_cube_z = np.array([float(b.data.root_pos_w[0, 2].cpu()) for b in env.blocks])
    min_hand_cube = float("inf")
    max_cube_lift = 0.0
    # tiefster Greifpunkt je Hand (zeigt, WO ein Würfel liegen müsste, um die Greif-Physik zu testen)
    hand_low = [np.array([0.0, 0.0, 1e9]) for _ in palm_idx]

    # Fingerspitzen — die Flächen, die den Würfel tatsächlich berühren. Der Handflächen-
    # Abstand allein führt in die Irre: der Bezugskörper liegt innerhalb der Handgeometrie,
    # genau dieser Fehler hat in Lauf 24 schon einmal die falsche Schlussfolgerung erzeugt.
    # Namen kommen aus dem Env, damit Eval und Replay denselben Bezugsrahmen messen.
    tip_names = list(G1Dex3BlockstackEnv._REACH_BODY_SETS[0][1])
    try:
        tip_ids, _ = env.robot.find_bodies(tip_names, preserve_order=True)
    except ValueError:
        tip_ids = []
    if len(tip_ids) != len(tip_names):
        tip_ids = []
        print("[Replay] WARNUNG: Fingerspitzen-Links nicht gefunden — nur Handflächen-Diagnose.",
              flush=True)
    min_tip_cube, min_tip_step = float("inf"), -1
    spread_max = [0.0, 0.0]                 # größte Fingeröffnung je Hand (links, rechts)
    spread_min, close_step = [float("inf")] * 2, [-1, -1]
    # --grasp-hold: Einsetz-Step, Halte-Zähler und Höhen je Hand
    hold_step, hold_ok, hold_total = [-1, -1], [0, 0], [0, 0]
    hold_insert_z, hold_max_z, hold_final_z = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]

    for i in range(n):
        action_t = torch.tensor(actions[i], dtype=torch.float32, device=env.device).unsqueeze(0)
        obs_step, _, terminated, time_out, info = env.step(action_t)

        # Tracking-Fehler: folgt der Roboter den kommandierten Gelenkwinkeln?
        achieved = obs_step["joint_pos"][0].cpu().numpy()
        err = np.abs(achieved - actions[i])
        arm_err.append(float(err[:14].mean()))
        finger_err.append(float(err[14:].mean()))
        per_joint_max = np.maximum(per_joint_max, err)
        np.minimum(cmd_lo, actions[i][14:], out=cmd_lo)
        np.maximum(cmd_hi, actions[i][14:], out=cmd_hi)
        np.minimum(ach_lo, achieved[14:], out=ach_lo)
        np.maximum(ach_hi, achieved[14:], out=ach_hi)

        # Greif-Diagnose: min Hand→Würfel-Distanz + maximale Würfel-Anhebung
        cubes = np.array([b.data.root_pos_w[0].cpu().numpy() for b in env.blocks])  # (3,3)
        if palm_idx:
            hands = env.robot.data.body_pos_w[0, palm_idx].cpu().numpy()           # (H,3)
            d = np.linalg.norm(hands[:, None, :] - cubes[None, :, :], axis=-1).min()
            min_hand_cube = min(min_hand_cube, float(d))
            max_cube_lift = max(max_cube_lift, float((cubes[:, 2] - init_cube_z).max()))
            for h in range(len(palm_idx)):
                if hands[h, 2] < hand_low[h][2]:
                    hand_low[h] = hands[h].copy()

        if tip_ids:
            tips = env.robot.data.body_pos_w[0, tip_ids].cpu().numpy()             # (6,3)
            d_tip = float(np.linalg.norm(tips[:, None, :] - cubes[None, :, :], axis=-1).min())
            if d_tip < min_tip_cube:
                min_tip_cube, min_tip_step = d_tip, i

            for h in range(2):
                t3 = tips[3 * h:3 * h + 3]                                         # Daumen/Zeige/Mittel
                # mittlerer paarweiser Fingerspitzen-Abstand = Weite der Greiföffnung
                spread = float(np.linalg.norm(t3[[0, 0, 1]] - t3[[1, 2, 2]], axis=-1).mean())
                spread_max[h] = max(spread_max[h], spread)
                if spread < spread_min[h]:
                    spread_min[h], close_step[h] = spread, i

                if h >= len(env.blocks):
                    continue
                # Der Würfel wird genau dann eingesetzt, wenn die Hand aus ihrer weitesten
                # Öffnung heraus zugreift (< 70 %). Der Mittelpunkt der drei Fingerspitzen
                # IST die Greiföffnung — damit ist Platzierung und Timing per Konstruktion
                # richtig und es bleibt nur die Frage, ob die Hand überhaupt halten kann.
                if (args.grasp_hold and hold_step[h] < 0 and i >= 60
                        and spread_max[h] > 0.05 and spread < 0.7 * spread_max[h]):
                    c = t3.mean(axis=0)
                    pose = torch.tensor([[float(c[0]), float(c[1]), float(c[2]),
                                          1.0, 0.0, 0.0, 0.0]],
                                        device=env.device, dtype=torch.float32)
                    env.blocks[h].write_root_pose_to_sim(pose)
                    env.blocks[h].write_root_velocity_to_sim(
                        torch.zeros((1, 6), device=env.device))
                    hold_step[h] = i
                    hold_insert_z[h] = float(c[2])
                    # ab hier wird die Anhebung gegen den Einsetzpunkt gemessen, nicht
                    # gegen den Tisch — sonst zählte das Einsetzen selbst als "Anhebung".
                    init_cube_z[h] = float(c[2])
                    print(f"[Replay] GRASP-HOLD: Würfel {h} bei Step {i} in die "
                          f"{'linke' if h == 0 else 'rechte'} Greiföffnung gesetzt "
                          f"(Spanne {spread * 100:.1f} cm von max. {spread_max[h] * 100:.1f} cm).",
                          flush=True)
                elif hold_step[h] >= 0:
                    hold_total[h] += 1
                    if float(np.linalg.norm(cubes[h] - t3.mean(axis=0))) < 0.04:
                        hold_ok[h] += 1
                    hold_max_z[h] = max(hold_max_z[h], float(cubes[h, 2]))
                    hold_final_z[h] = float(cubes[h, 2])

        # Debug (einmalig, Step 0): alle 4 Kamera-Frames dumpen (wie im Modell-Eval)
        if i == 0 and args.video_dir:
            import imageio as _iio
            os.makedirs(args.video_dir, exist_ok=True)
            for _ck in ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"):
                _key = f"video.{_ck}"
                if _key in obs_step:
                    _arr = obs_step[_key][0].cpu().numpy()
                    while _arr.ndim > 3:
                        _arr = _arr[0]
                    _iio.imwrite(os.path.join(args.video_dir, f"_debug_obs_{_ck}.png"),
                                 _arr.astype(np.uint8))
            print("[Replay] Debug-Bilder aller 4 Kameras gespeichert: _debug_obs_*.png", flush=True)

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
    mean_finger_err = float(np.mean(finger_err)) if finger_err else 0.0
    cmd_span, ach_span = cmd_hi - cmd_lo, ach_hi - ach_lo
    print(f"[Replay] FINGER-TRACKING: mittel={mean_finger_err:.3f} rad | "
          f"max={per_joint_max[14:].max():.2f} rad")
    print(f"[Replay]   Fingerspanne kommandiert={cmd_span.max():.2f} rad, "
          f"erreicht={ach_span.max():.2f} rad", flush=True)
    print("[Replay]   erreicht ≪ kommandiert → Griff wird an einer Gelenkgrenze abgeklemmt;"
          " beide groß + Anhebung 0 → Kontakt/Reibung oder Würfel nicht in der Greiföffnung.",
          flush=True)
    print(f"[Replay] GREIF-DIAGNOSE: min Hand(Handfläche)→Würfelmitte = {min_hand_cube*100:.1f} cm | "
          f"max Würfel-Anhebung = {max_cube_lift*100:.1f} cm", flush=True)
    print("[Replay]   Distanz groß (>~15cm) → Greifbewegung trifft unsere Würfel nicht (Platzierung);"
          " Distanz klein + Anhebung≈0 → Greif-Physik prüfen; Anhebung>~2cm → Greifen FUNKTIONIERT.",
          flush=True)
    if tip_ids:
        print(f"[Replay]   min Fingerspitze→Würfelmitte = {min_tip_cube * 100:.1f} cm "
              f"(Step {min_tip_step}) | engste Greiföffnung: "
              f"links {spread_min[0] * 100:.1f} cm (Step {close_step[0]}), "
              f"rechts {spread_min[1] * 100:.1f} cm (Step {close_step[1]})", flush=True)
        print("[Replay]   Liegt der Step der engsten Greiföffnung weit weg vom Step des "
              "kleinsten Fingerspitzen-Abstands, greift die Hand ins Leere — dann stimmt die "
              "Würfel-Platzierung des Greif-Tests nicht, und die Physik ist nicht widerlegt.",
              flush=True)

    if args.grasp_hold:
        for h in range(2):
            if hold_step[h] < 0:
                print(f"[Replay] GRASP-HOLD {'links' if h == 0 else 'rechts'}: kein Zugreifen "
                      "erkannt (Finger schließen nie unter 70 % ihrer größten Öffnung).",
                      flush=True)
                continue
            ratio = hold_ok[h] / max(hold_total[h], 1)
            print(f"[Replay] GRASP-HOLD {'links' if h == 0 else 'rechts'}: Würfel ab Step "
                  f"{hold_step[h]} zu {ratio * 100:.0f} % in der Hand "
                  f"({hold_ok[h]}/{hold_total[h]} Steps), max. {(hold_max_z[h] - hold_insert_z[h]) * 100:+.1f} cm "
                  f"über dem Einsetzpunkt, Endhöhe z={hold_final_z[h]:.3f}", flush=True)
        print("[Replay]   Haltequote >~50 % → die Greif-Physik trägt, das Problem ist "
              "Platzierung/Timing des Tests; ~0 % und Endhöhe ≈ Tischauflage → der Würfel "
              "rutscht aus der geschlossenen Hand: Kontakt/Reibung.", flush=True)

    lbl = ["left", "right"]
    for h in range(len(palm_idx)):
        p = hand_low[h]
        print(f"[Replay]   tiefster {lbl[h] if h < 2 else h}-Hand-Punkt (Welt): "
              f"x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}", flush=True)
    # init_cube_z ist die Würfel-MITTE (block_z_surface = Tisch 0.89 + halbe Kantenlänge
    # 0.025), nicht die Oberkante — die Zeile war bis Lauf 26 falsch beschriftet, und genau
    # solche Beschriftungen haben hier schon einmal zu einem falschen Schluss geführt.
    cz = float(init_cube_z[0])
    print(f"[Replay]   (Würfel-MITTE z≈{cz:.3f}, Oberkante z≈{cz + 0.025:.3f}; "
          f"tiefster Handflächenpunkt darüber = Hand bleibt über dem Würfel. Würfel-XY: "
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
        "mean_finger_tracking_error_rad": round(mean_finger_err, 4),
        "finger_span_commanded_rad": round(float(cmd_span.max()), 3),
        "finger_span_achieved_rad": round(float(ach_span.max()), 3),
        "per_joint_max_error_rad": [round(float(x), 3) for x in per_joint_max],
        "min_hand_cube_dist_cm": round(min_hand_cube * 100, 1),
        "max_cube_lift_cm": round(max_cube_lift * 100, 1),
        "grasp_mode": "hold" if args.grasp_hold else ("test" if args.grasp_test else "none"),
        "min_fingertip_cube_dist_cm": (round(min_tip_cube * 100, 1)
                                       if np.isfinite(min_tip_cube) else None),
        "min_fingertip_step": min_tip_step,
        "finger_spread_min_cm": [round(s * 100, 1) if np.isfinite(s) else None
                                 for s in spread_min],
        "finger_close_step": close_step,
        "hold_step": hold_step if args.grasp_hold else None,
        "hold_ratio": ([round(hold_ok[h] / max(hold_total[h], 1), 3) for h in range(2)]
                       if args.grasp_hold else None),
        "hold_rise_cm": ([round((hold_max_z[h] - hold_insert_z[h]) * 100, 1) for h in range(2)]
                         if args.grasp_hold else None),
        "hold_final_z": ([round(z, 3) for z in hold_final_z] if args.grasp_hold else None),
        "note": "Würfel-Posen entsprechen nicht exakt der realen Episode; "
                "aussagekräftig ist die Arm-/Finger-Bewegung (vorwärts-greifen vs. wegdriften).",
    }
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Replay] Ergebnis: success={success} (Step {success_step}) → {args.results_file}")
    # Erfolgsmarker für server_rl_run.sh: isaaclab.sh verschluckt den Exit-Code, der
    # Wrapper erkennt einen sauberen Durchlauf deshalb nur an dieser Zeile (wie
    # '[eval] fertig.' / '[dump] fertig.' / '[gap] fertig.').
    print("[replay] fertig.", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
