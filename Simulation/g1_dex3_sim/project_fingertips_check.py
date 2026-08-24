#!/usr/bin/env python3
"""Fingerkuppen aus der FK ins REALBILD projizieren — Kamerapose gegen Roboter-FK.

WOZU
────
Der Render-Lauf vom 2026-08-23 zeigt einen konstanten Widerspruch: über alle Frames und
beide Hände kommt keine Fingerkuppe je näher als **10 cm** an den Würfel, den das
Bild-Layout gesetzt hat — obwohl sich der reale Würfel am Fensterende nachweislich bewegt,
die reale Hand ihn also hält. Dieselben rund 10 cm zeigten schon die v4-Pick-Anker (4–11 cm)
und die Pinhole-Gegenprobe (11,5 cm). Genau eines von beiden ist falsch:

  (a) die Würfellage aus dem Realbild — also die Kamerapose für REALE Bilder. In der
      Simulation ist das Modell gegen Grundwahrheit auf 0,3 cm bestätigt (Lauf 37), für
      Realbilder ist es unbewiesen: die Pose wurde aus Datensatzframes rekonstruiert.
  (b) die Handposition aus dem aufgezeichneten Zustand — FK, Basispose, oder die im
      28-dim-State FEHLENDEN Hüftgelenke. 13° Rumpfneigung reichen für 10 cm.

Dieses Werkzeug trennt beide Fälle in einem Bild. Es setzt den Roboter auf den
aufgezeichneten Zustand, liest die echten Fingerkuppen (``get_contact_points_w``) und
projiziert sie mit demselben Kameramodell ins REALE Kamerabild desselben Frames. Dazu die
Würfel aus ``layout.json``.

  * Kreuze liegen auf den realen Händen  →  Kamera UND FK stimmen. Dann kann der Fehler nur
    in der Würfellage stecken, obwohl sie in der Sim geprüft ist — und das hieße, dass die
    reale Kamera anders steht als die simulierte.
  * Kreuze liegen daneben  →  Kamera oder FK. Die Richtung des Versatzes sagt, welches:
    ein Kamerafehler verschiebt Hände UND Würfel gleichsinnig, ein FK-Fehler nur die Hände.

Der zweite Punkt ist der eigentliche Wert: Würfelmarken und Handmarken im selben Bild
verschieben sich bei einem Kamerafehler gemeinsam, bei einem FK-Fehler gegeneinander.

VERWENDUNG
    ./Simulation/server_rl_run.sh tipcheck
oder direkt im Container:
    $ISAAC_PY project_fingertips_check.py --headless \\
        --dataset-path /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset \\
        --layout /data/cotrain/layout.json --out-dir /data/cotrain/tipcheck

Der Prüfframe ist standardmäßig der Bewegungsbeginn aus ``layout.json`` — der Moment, in dem
die Hand den Würfel nachweislich hält. ``--frame N`` prüft stattdessen einen festen Frame.
Es werden keine Aktionen abgespielt; nur Zustände werden direkt gesetzt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--layout", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--asset-path", default="")
parser.add_argument("--num-episodes", type=int, default=4)
parser.add_argument("--start-episode", type=int, default=0)
parser.add_argument("--episode-ids", type=int, nargs="*", default=None)
parser.add_argument("--train-ratio", type=float, default=0.8)
parser.add_argument("--frame", type=int, default=-1,
                    help="-1 = Bewegungsbeginn aus layout.json (empfohlen), sonst fester Frame")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

os.environ.setdefault("DR_ENABLED", "0")
os.environ["SCENE_CAM"] = "0"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from camera_geometry import PinholeCamera  # noqa: E402
from extract_block_layout import CUBE_COLORS  # noqa: E402
from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from reconstruct_cube_poses import (  # noqa: E402
    HEAD_CAMS,
    data_path,
    read_info,
    select_episodes,
    video_path,
)
from replay_calibration import CUBE_CENTER_Z_M  # noqa: E402

DRAW_COLOR = {"rot": (255, 60, 60), "gruen": (60, 220, 60), "gelb": (255, 220, 40)}
HAND_COLOR = {"left": (80, 160, 255), "right": (255, 140, 0)}


def set_robot_state(env: G1Dex3BlockstackEnv, state: np.ndarray) -> None:
    """Aufgezeichneten 28-DoF-Zustand direkt setzen. Kopie aus collect_replay_anchors.py.

    Bewusst dupliziert statt importiert: jedes Isaac-Skript startet seine eigene App, ein
    Import zöge die fremde ``AppLauncher``-Initialisierung mit.
    """
    if env._joint_ids is None:
        env._joint_ids = env._build_joint_id_mapping()
    q = np.asarray(state, dtype=np.float32).copy()
    q[env._SIGN_FLIP_IDX] *= -1
    full = env.robot.data.joint_pos.clone()
    for policy_index, isaac_index in enumerate(env._joint_ids):
        full[:, isaac_index] = float(q[policy_index])
    env.robot.write_joint_state_to_sim(full, torch.zeros_like(full))
    env.robot.set_joint_position_target(full)
    env.sim.forward()
    env.robot.update(float(env.cfg.sim.dt))


def fingertips_env_local(env: G1Dex3BlockstackEnv) -> np.ndarray:
    """Sechs Fingerkuppen (links 0–2, rechts 3–5) env-lokal, wie das Kameramodell rechnet."""
    tips = env.get_contact_points_w()[0].detach().cpu().numpy()
    if tips.shape != (6, 3) or env._reach_frame != "fingertip":
        raise RuntimeError(
            f"Sechs echte Fingerkuppen benötigt; frame={env._reach_frame}, shape={tips.shape}"
        )
    return tips - env.scene.env_origins[0].detach().cpu().numpy()


def real_frame(root: Path, info: dict, episode: int, camera: str, index: int) -> np.ndarray:
    path = video_path(root, info, episode, camera)
    with imageio.get_reader(str(path), format="FFMPEG") as reader:
        for i, frame in enumerate(reader):
            if i == index:
                return np.asarray(frame, dtype=np.uint8)[..., :3]
    raise SystemExit(f"Episode {episode}: Frame {index} fehlt in {path}.")


def marker(draw: ImageDraw.ImageDraw, uv, color, label: str, size: int = 7) -> None:
    if uv is None or not np.all(np.isfinite(uv)):
        return
    u, v = float(uv[0]), float(uv[1])
    draw.line([(u - size, v), (u + size, v)], fill=color, width=2)
    draw.line([(u, v - size), (u, v + size)], fill=color, width=2)
    draw.text((u + size + 2, v - size), label, fill=color)


def annotate(frame: np.ndarray, cam: PinholeCamera, tips: np.ndarray,
             cubes: list, path: Path) -> None:
    """Handmarken und Würfelmarken in dasselbe Realbild zeichnen."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for i, color in enumerate(CUBE_COLORS):
        if i >= len(cubes) or not cubes[i]:
            continue
        world = np.array([cubes[i][0], cubes[i][1], CUBE_CENTER_Z_M], dtype=float)
        # project() liefert fuer EINEN Punkt bereits (2,) — kein [0] davor.
        marker(draw, cam.project(world), DRAW_COLOR[color], color, size=10)
    for hand, offset in (("left", 0), ("right", 3)):
        pixels = cam.project(tips[offset : offset + 3])
        for k, uv in enumerate(pixels):
            marker(draw, uv, HAND_COLOR[hand], f"{hand[0].upper()}{k}", size=5)
        marker(draw, cam.project(tips[offset : offset + 3].mean(axis=0)),
               HAND_COLOR[hand], f"{hand}-Mitte", size=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> int:
    root = Path(args.dataset_path)
    info = read_info(root)
    episodes = select_episodes(root, args)
    layout = json.loads(Path(args.layout).read_text(encoding="utf-8")).get("episodes", {})
    out_dir = Path(args.out_dir)
    cams = {name: PinholeCamera.from_cfg(name) for name in HEAD_CAMS}

    cfg = G1Dex3BlockstackEnvCfg()
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    cfg.sim.render_interval = 1_000_000
    cfg.episode_length_s = 3600.0
    cfg.terminate_on_success = False
    env = G1Dex3BlockstackEnv(cfg=cfg, render_mode=None)
    env.reset()

    summary = []
    for ordinal, episode in enumerate(episodes, 1):
        head = f"[tipcheck] ({ordinal}/{len(episodes)}) Episode {episode}"
        record = layout.get(str(episode))
        if not record:
            print(f"{head}: kein Layout-Eintrag — übersprungen.", flush=True)
            continue
        frame_index = (int(args.frame) if args.frame >= 0
                       else (record.get("motion_onset") or {}).get("first"))
        if frame_index is None:
            print(f"{head}: kein Bewegungsbeginn im Layout — übersprungen.", flush=True)
            continue

        states = np.stack(
            pd.read_parquet(data_path(root, info, episode))["observation.state"].to_numpy()
        ).astype(np.float32)
        frame_index = int(min(frame_index, len(states) - 1))
        set_robot_state(env, states[frame_index])
        tips = fingertips_env_local(env)
        cubes = record.get("cubes") or []

        # Abstand jeder Handmitte zum nächsten Würfel — dieselbe Größe wie im Renderbericht.
        distances = {}
        for hand, offset in (("left", 0), ("right", 3)):
            center = tips[offset : offset + 3].mean(axis=0)
            best = None
            for i, color in enumerate(CUBE_COLORS):
                if i >= len(cubes) or not cubes[i]:
                    continue
                world = np.array([cubes[i][0], cubes[i][1], CUBE_CENTER_Z_M])
                d = float(np.linalg.norm(world - center))
                if best is None or d < best[1]:
                    best = (color, d)
            distances[hand] = best

        for name, cam in cams.items():
            annotate(real_frame(root, info, episode, name, frame_index), cam, tips, cubes,
                     out_dir / f"ep{episode:06d}_{name}_f{frame_index:04d}.png")

        left, right = distances["left"], distances["right"]
        summary.append({"episode": episode, "frame": frame_index,
                        "left_cm": round(left[1] * 100, 1) if left else None,
                        "left_cube": left[0] if left else None,
                        "right_cm": round(right[1] * 100, 1) if right else None,
                        "right_cube": right[0] if right else None,
                        "fingertips_env_local_m": tips.round(4).tolist(),
                        "cubes_xy_m": cubes})
        parts = [f"links {left[1] * 100:.1f} cm ({left[0]})" if left else "links —",
                 f"rechts {right[1] * 100:.1f} cm ({right[0]})" if right else "rechts —"]
        print(f"{head}: Frame {frame_index}, Handmitte→nächster Würfel: "
              + ", ".join(parts), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tipcheck.json").write_text(
        json.dumps({"frame_source": "motion_onset" if args.frame < 0 else f"fest:{args.frame}",
                    "cube_plane_z_m": CUBE_CENTER_Z_M, "episodes": summary}, indent=2),
        encoding="utf-8")
    print(f"[tipcheck] Bilder und tipcheck.json in {out_dir}")
    print("[tipcheck] Lesart: Handmarken auf den realen Händen => Kamera und FK stimmen.")
    print("[tipcheck]         Hand- UND Würfelmarken gleichsinnig daneben => Kamerapose.")
    print("[tipcheck]         nur die Handmarken daneben => FK bzw. fehlende Hüftgelenke.")
    print("[tipcheck] fertig.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
