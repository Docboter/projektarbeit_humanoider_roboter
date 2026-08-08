#!/usr/bin/env python3
"""Diagnose: KONFIGURIERTE gegen TATSAECHLICH gerenderte Kamera-Pose.

Anlass (2026-08-08): Die Live-Ansicht zeigte fuer `cam_scene` fast nur den hellen
Dome-Hintergrund plus eine Ecke Bodengitter — kein Tisch, kein Roboter. Nachrechnen der
konfigurierten Pose (Auge (1.8, 1.6, 1.7), Ziel (0.30, 0, 0.80), HFOV 60.4 Grad) ergab
aber, dass Tisch, Wuerfel und Roboter vollstaendig im Bild liegen muessten. Die Zahlen in
g1_dex3_cfg.py sind also nicht das Problem — irgendetwas zwischen Konfiguration und
Render weicht ab. Dieses Skript misst genau diese Luecke, statt an Koordinaten zu raten.

Ausgegeben wird je Kamera:
  * konfigurierter Offset (cfg.scene.<cam>.offset)
  * tatsaechliche Weltpose nach dem Reset (camera.data.pos_w / quat_w_world)
  * die Pose RELATIV zum Env-Ursprung — der entscheidende Vergleich, denn die Kameras
    haengen unter {ENV_REGEX_NS} und werden je Env geklont
  * der Winkel zwischen konfigurierter und tatsaechlicher Blickrichtung

Ausserdem ein PNG je Kamera. Bewusst mit Pillow statt imageio: Pillow ist im
Isaac-Sim-Kit-Python nachweislich vorhanden (die Live-Ansicht nutzt es), imageio nicht.

--num-envs 4 ist Default, weil der Befund am Vier-Env-RL-Lauf auftrat und ein
Ein-Env-Lauf die Klon-Frage per Konstruktion nicht beantworten kann.

Verwendung (im Sim-Container, RT-Core-GPU):
    unset VIRTUAL_ENV
    ${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/dump_camera_poses.py \\
        --headless --enable_cameras \\
        --asset-path /data/checkpoints/groot-g1dex3-checkpoint/g1_dex3.usd
Bequemer:  ./Simulation/server_rl_run.sh cams
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Kamera-Posen: konfiguriert vs. gerendert")
parser.add_argument("--asset-path", type=str, default=None, help="G1+Dex3 USD-Asset")
parser.add_argument("--out-dir", type=str, default="/data/cam_dump", help="PNG-Ausgabe")
parser.add_argument("--num-envs", type=int, default=4, help="Envs (Klon-Verhalten pruefen)")
parser.add_argument("--settle-steps", type=int, default=8, help="Steps vor der Messung")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- erst nach AppLauncher importierbar ---------------------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402
from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg  # noqa: E402


def quat_rotate(q, v):
    """Vektor v mit Quaternion q=(w, x, y, z) drehen."""
    w, x, y, z = [float(c) for c in q]
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def first_attr(obj, names):
    """Erstes vorhandenes Attribut zurueckgeben (Isaac Lab 3.0 ist Beta — Namen wandern)."""
    for n in names:
        if hasattr(obj, n):
            return n, getattr(obj, n)
    return None, None


def main() -> None:
    cfg = G1Dex3BlockstackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path

    print("=" * 72)
    print(f"KAMERA-POSEN — konfiguriert vs. gerendert   (num_envs={args.num_envs})")
    print("=" * 72, flush=True)

    env = G1Dex3BlockstackEnv(cfg)
    env.reset()
    zero = torch.zeros((env.num_envs, 28), dtype=torch.float32, device=env.device)
    obs = None
    for _ in range(max(1, args.settle_steps)):
        obs, _, _, _, _ = env.step(zero)

    origins = env.scene.env_origins.detach().cpu().numpy()
    print("\nEnv-Ursprünge (env_spacing="
          f"{getattr(cfg.scene, 'env_spacing', '?')}):")
    for e, o in enumerate(origins):
        print(f"  env {e}: {np.round(o, 3)}")

    for name, cam in env.cameras.items():
        cam_cfg = getattr(cfg.scene, name, None)
        off = getattr(cam_cfg, "offset", None) if cam_cfg is not None else None
        cfg_pos = np.asarray(off.pos, dtype=float) if off is not None else None
        cfg_rot = np.asarray(off.rot, dtype=float) if off is not None else None

        pos_key, pos_w = first_attr(cam.data, ["pos_w"])
        quat_key, quat_w = first_attr(
            cam.data, ["quat_w_world", "quat_w", "quat_w_ros", "quat_w_opengl"]
        )
        print("\n" + "-" * 72)
        print(f"{name}   (Prim: {getattr(cam_cfg, 'prim_path', '?')})")
        if pos_w is None or quat_w is None:
            print(f"  !! Pose nicht auslesbar. Verfuegbar: {sorted(dir(cam.data))[:40]}")
            continue
        pos_w = pos_w.detach().cpu().numpy()
        quat_w = quat_w.detach().cpu().numpy()
        print(f"  gelesen aus: data.{pos_key} / data.{quat_key}")
        if cfg_pos is not None:
            print(f"  konfiguriert: pos={np.round(cfg_pos, 3)}  rot={np.round(cfg_rot, 4)}")

        for e in range(min(len(pos_w), args.num_envs)):
            rel = pos_w[e] - origins[e]
            line = (f"  env {e}: welt={np.round(pos_w[e], 3)}  "
                    f"rel. zum Ursprung={np.round(rel, 3)}")
            if cfg_pos is not None and off is not None and "ENV_REGEX_NS" in str(
                getattr(cam_cfg, "prim_path", "")
            ):
                d = float(np.linalg.norm(rel - cfg_pos))
                line += f"  Δ zum Offset={d:.3f} m {'OK' if d < 0.02 else '<-- WEICHT AB'}"
            print(line)

        # Blickrichtung. In convention="world" ist die Blickachse +X.
        view = quat_rotate(quat_w[0], [1.0, 0.0, 0.0])
        print(f"  Blickrichtung (env 0, +X): {np.round(view, 3)}  "
              f"Pitch={np.degrees(np.arcsin(np.clip(view[2], -1, 1))):+.1f}°")
        if cfg_rot is None:
            continue
        want = quat_rotate(cfg_rot, [1.0, 0.0, 0.0])
        print(f"  SOLL-Richtung (aus der Config): {np.round(want, 3)}")

        # Welche Konvention/Achse trifft die Sollrichtung? Isaac Lab meldet dieselbe
        # physische Orientierung in mehreren Konventionen (world/ros/opengl), und die
        # "Vorwaerts"-Achse ist je Konvention eine andere: world +X, ROS +Z, OpenGL -Z.
        # Zeigt eine ANDERE Kombination als (world, +X) auf das Ziel, dann wendet Isaac Lab
        # unsere rot anders an als gemeint — dann ist `convention=` der Fehler, nicht die Pose.
        axes = {"+X": [1, 0, 0], "-X": [-1, 0, 0], "+Y": [0, 1, 0],
                "-Y": [0, -1, 0], "+Z": [0, 0, 1], "-Z": [0, 0, -1]}
        variants = {}
        for qname in ("quat_w_world", "quat_w_ros", "quat_w_opengl", "quat_w"):
            if hasattr(cam.data, qname):
                variants[qname] = getattr(cam.data, qname).detach().cpu().numpy()[0]
        best = None
        print("  Treffer-Matrix (Winkel zur SOLL-Richtung, kleinster Wert gewinnt):")
        for qname, q in variants.items():
            row, cells = [], []
            for aname, ax in axes.items():
                a = np.degrees(np.arccos(np.clip(np.dot(quat_rotate(q, ax), want), -1, 1)))
                cells.append(f"{aname} {a:6.1f}°")
                row.append((a, qname, aname))
            print(f"    {qname:<14} " + "  ".join(cells))
            cand = min(row)
            best = cand if best is None or cand[0] < best[0] else best
        if best is not None:
            ang, qname, aname = best
            verdict = "OK" if (qname.endswith("world") and aname == "+X" and ang < 2) else \
                      "<-- Konvention passt NICHT zu convention='world'"
            print(f"  bester Treffer: {qname} / {aname} bei {ang:.1f}°   {verdict}")

    # Standbilder — belegen, was die Kamera wirklich sieht.
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        print(f"\n[dump] Pillow fehlt ({e}) — keine PNGs.", flush=True)
    else:
        print()
        for key in sorted(k for k in (obs or {}) if k.startswith("video.")):
            frame = obs[key][0].detach().cpu().numpy().astype(np.uint8)[..., :3]
            path = os.path.join(args.out_dir, f"{key.split('.', 1)[1]}.png")
            Image.fromarray(frame).save(path)
            print(f"[dump] {path}  {frame.shape}", flush=True)

    env.close()
    print("\n[dump] fertig.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
