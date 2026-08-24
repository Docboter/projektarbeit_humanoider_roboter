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

Das Werkzeug setzt den Roboter auf den aufgezeichneten Zustand und projiziert mit dem
Kameramodell ins REALE Bild desselben Frames: die Fingerkuppen
(``get_contact_points_w``), die Kette vom Becken zum Handgelenk und die Würfel aus
``layout.json``.

WIE ES ZU LESEN IST
───────────────────
Die **Würfelmarken sagen nichts aus**. Das Layout hat den Würfelpixel mit genau diesem
Modell auf die Tischebene rückprojiziert; ihn damit zurückzuprojizieren trifft immer, per
Konstruktion. (Eine frühere Fassung dieses Kommentars behauptete das Gegenteil.)

Aussagekräftig ist allein, wo die Roboter-Marken gegenüber dem realen Roboter im Bild
liegen — sie verbinden die FK, die vom Kameramodell unabhängig ist, mit dem Modell:

  * Becken sitzt richtig, der Fehler wächst die Kette hinunter zur Hand
        →  Arm-FK oder Gelenkzuordnung.
  * schon das Becken versetzt, alle Marken gleichsinnig daneben
        →  Kamerapose relativ zum Roboter.

Beides sind Eigenschaften des Robotermodells. Erst diese Unterscheidung sagt, ob die
Würfellage aus dem Realbild überhaupt in Frage steht.

VERWENDUNG
    ./Simulation/server_rl_run.sh tipcheck
oder direkt im Container:
    $ISAAC_PY project_fingertips_check.py --headless --enable_cameras \\
        --dataset-path /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset \\
        --layout /data/cotrain/layout.json --out-dir /data/cotrain/tipcheck

``--enable_cameras`` ist Pflicht, obwohl hier nichts gerendert wird: die Env legt die
Policy-Kameras beim Aufbau an, und Isaac Lab bricht ohne das Flag beim Szenenaufbau ab.

Der Prüfframe ist standardmäßig der Bewegungsbeginn aus ``layout.json`` — der Moment, in dem
die Hand den Würfel nachweislich hält. ``--frame N`` prüft stattdessen einen festen Frame.
Es werden keine Aktionen abgespielt; nur Zustände werden direkt gesetzt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
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
CHAIN_COLOR = (255, 0, 255)


# Vorzeichen-Varianten fuer die 14 Handdimensionen (links 14–20, rechts 21–27).
# Der Kommentar an _SIGN_FLIP_IDX begruendet die Spiegelung nur fuer die LINKE Hand
# ("USD: links negativ = schliessen, rechts positiv = schliessen", Datensatz positiv =
# schliessen), gespiegelt werden aber beide Seiten. Statt darueber zu streiten, wird hier
# gemessen: die richtige Variante muss am Greifframe eine Kuppenoeffnung nahe der
# Wuerfelkante liefern, nicht die knapp 11 cm einer offenen Hand.
SIGN_VARIANTS: dict[str, list[int]] = {
    "aktuell": [17, 19, 24, 26],
    "ohne": [],
    "nur_links_0": [17, 19],
    "nur_rechts_0": [24, 26],
    "alle_finger": list(range(14, 28)),
    "alle_links": list(range(14, 21)),
    "alle_rechts": list(range(21, 28)),
}


def set_robot_state(env: G1Dex3BlockstackEnv, state: np.ndarray,
                    flip_idx: list[int] | None = None) -> None:
    """Aufgezeichneten 28-DoF-Zustand direkt setzen. Kopie aus collect_replay_anchors.py.

    Bewusst dupliziert statt importiert: jedes Isaac-Skript startet seine eigene App, ein
    Import zöge die fremde ``AppLauncher``-Initialisierung mit. ``flip_idx`` überschreibt
    die Vorzeichenspiegelung der Env, damit Varianten vergleichbar werden.
    """
    if env._joint_ids is None:
        env._joint_ids = env._build_joint_id_mapping()
    q = np.asarray(state, dtype=np.float32).copy()
    q[env._SIGN_FLIP_IDX if flip_idx is None else flip_idx] *= -1
    full = env.robot.data.joint_pos.clone()
    for policy_index, isaac_index in enumerate(env._joint_ids):
        full[:, isaac_index] = float(q[policy_index])
    env.robot.write_joint_state_to_sim(full, torch.zeros_like(full))
    env.robot.set_joint_position_target(full)
    env.sim.forward()
    env.robot.update(float(env.cfg.sim.dt))


# Kette vom Rumpf zur Hand. Sie ist der eigentliche Trenner: liegt das Becken im Realbild
# richtig und wandert der Fehler erst die Kette hinunter, steckt er in der Arm-FK bzw. der
# Gelenkzuordnung. Ist schon das Becken versetzt, ist die Kamerapose relativ zum Roboter
# falsch — dann verschieben sich alle Marken gemeinsam. Nur diese Unterscheidung beantwortet,
# ob die Wuerfellage aus dem Realbild ueberhaupt in Frage steht.
CHAIN_LINKS = (
    "pelvis", "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_yaw_link",
)


def chain_env_local(env: G1Dex3BlockstackEnv) -> dict[str, np.ndarray]:
    """Weltpositionen der vorhandenen Kettenglieder, env-lokal. Fehlende werden ausgelassen."""
    data = env.robot.data
    names = list(getattr(data, "body_names", []) or [])
    positions = data.body_pos_w[0].detach().cpu().numpy()
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    return {name: positions[names.index(name)] - origin
            for name in CHAIN_LINKS if name in names}


def fingertips_env_local(env: G1Dex3BlockstackEnv) -> np.ndarray:
    """Sechs Fingerkuppen (links 0–2, rechts 3–5) env-lokal, wie das Kameramodell rechnet."""
    tips = env.get_contact_points_w()[0].detach().cpu().numpy()
    if tips.shape != (6, 3) or env._reach_frame != "fingertip":
        raise RuntimeError(
            f"Sechs echte Fingerkuppen benötigt; frame={env._reach_frame}, shape={tips.shape}"
        )
    return tips - env.scene.env_origins[0].detach().cpu().numpy()


def hand_spread_and_center(tips: np.ndarray, offset: int) -> tuple[float, np.ndarray]:
    """Mittlerer paarweiser Kuppenabstand und Schwerpunkt einer Hand — wie grasp_pose_support."""
    points = tips[offset : offset + 3]
    spread = float(np.linalg.norm(points[[0, 0, 1]] - points[[1, 2, 2]], axis=1).mean())
    return spread, points.mean(axis=0)


def hand_joint_ranges(env: G1Dex3BlockstackEnv, states: np.ndarray) -> list[dict]:
    """Aufgezeichnete Spannweite je Handdimension gegen die Gelenkgrenzen der Sim.

    Die Vorzeichen-Probe setzt die Gelenke mit ``write_joint_state_to_sim`` und umgeht damit
    die Grenzen. Im geschlossenen Regelkreis wirkt dagegen ``set_joint_position_target``, das
    klemmt. Ein Wert ausserhalb der Grenze bewegt das Gelenk dort also nicht — genau so wurde
    die rechte Hand bis 2026-08-24 stillgelegt. Deshalb hier beides nebeneinander.
    """
    if env._joint_ids is None:
        env._joint_ids = env._build_joint_id_mapping()
    names = list(env.robot.data.joint_names)
    lower = env.robot.data.joint_pos_limits[0, :, 0].detach().cpu().numpy()
    upper = env.robot.data.joint_pos_limits[0, :, 1].detach().cpu().numpy()
    rows = []
    for dim in range(14, 28):
        isaac = env._joint_ids[dim]
        recorded = states[:, dim]
        sign = -1.0 if dim in env._SIGN_FLIP_IDX else 1.0
        sent = sign * recorded
        rows.append({
            "dim": dim, "joint": names[isaac], "gespiegelt": sign < 0,
            "aufgezeichnet": [round(float(recorded.min()), 3), round(float(recorded.max()), 3)],
            "gesendet": [round(float(sent.min()), 3), round(float(sent.max()), 3)],
            "grenze": [round(float(lower[isaac]), 3), round(float(upper[isaac]), 3)],
            "geklemmt_frames": int(((sent < lower[isaac]) | (sent > upper[isaac])).sum()),
        })
    return rows


def sign_sweep(env: G1Dex3BlockstackEnv, state: np.ndarray,
               cubes: list) -> list[dict]:
    """Je Vorzeichenvariante die Kuppenoeffnung und den Abstand zum naechsten Wuerfel."""
    targets = [np.array([c[0], c[1], CUBE_CENTER_Z_M]) for c in cubes if c]
    rows = []
    for name, flip in SIGN_VARIANTS.items():
        set_robot_state(env, state, flip_idx=flip)
        tips = fingertips_env_local(env)
        row = {"variante": name, "flip": flip}
        for hand, offset in (("left", 0), ("right", 3)):
            spread, center = hand_spread_and_center(tips, offset)
            row[f"{hand}_spread_cm"] = round(spread * 100, 1)
            row[f"{hand}_dist_cm"] = (
                round(min(float(np.linalg.norm(t - center)) for t in targets) * 100, 1)
                if targets else None)
        rows.append(row)
    return rows


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
             cubes: list, chain: dict[str, np.ndarray], path: Path) -> None:
    """Ketten-, Hand- und Würfelmarken in dasselbe Realbild zeichnen."""
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
    for name, position in chain.items():
        marker(draw, cam.project(position), CHAIN_COLOR, name.replace("_link", ""), size=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> int:
    root = Path(args.dataset_path)
    info = read_info(root)
    layout = json.loads(Path(args.layout).read_text(encoding="utf-8")).get("episodes", {})
    if args.episode_ids:
        episodes = select_episodes(info, args)
    else:
        # Ohne ausdrueckliche Auswahl aus dem LAYOUT waehlen, nicht fortlaufend: der
        # Layout-Lauf streut seine Episoden ueber den Trainingsbereich (linspace, also
        # 0, 4, 8, …), fortlaufende Nummern treffen ihn fast nie. Testepisoden koennen
        # dabei nicht auftauchen, die sperrt bereits extract_block_layout.
        episodes = [e for e in sorted(int(k) for k in layout)
                    if e >= int(args.start_episode)][: int(args.num_episodes)]
        if not episodes:
            raise SystemExit(
                f"{args.layout} enthaelt keine Episode ab {args.start_episode}. "
                "Erst 'server_rl_run.sh layout' fahren oder --episode-ids setzen."
            )
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
        cubes = record.get("cubes") or []
        ranges = hand_joint_ranges(env, states)
        sweep = sign_sweep(env, states[frame_index], cubes)
        # Zuletzt die produktive Variante setzen, damit Bilder und Zahlen sie zeigen.
        set_robot_state(env, states[frame_index])
        tips = fingertips_env_local(env)
        chain = chain_env_local(env)

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
                     chain, out_dir / f"ep{episode:06d}_{name}_f{frame_index:04d}.png")

        left, right = distances["left"], distances["right"]
        summary.append({"episode": episode, "frame": frame_index,
                        "left_cm": round(left[1] * 100, 1) if left else None,
                        "left_cube": left[0] if left else None,
                        "right_cm": round(right[1] * 100, 1) if right else None,
                        "right_cube": right[0] if right else None,
                        "fingertips_env_local_m": tips.round(4).tolist(),
                        "chain_env_local_m": {k: v.round(4).tolist() for k, v in chain.items()},
                        "sign_sweep": sweep,
                        "hand_joint_ranges": ranges,
                        "cubes_xy_m": cubes})
        parts = [f"links {left[1] * 100:.1f} cm ({left[0]})" if left else "links —",
                 f"rechts {right[1] * 100:.1f} cm ({right[0]})" if right else "rechts —"]
        print(f"{head}: Frame {frame_index}, Handmitte→nächster Würfel: "
              + ", ".join(parts), flush=True)
        print(f"      {'Variante':<14}{'links Öffn.':>12}{'links Δ':>10}"
              f"{'rechts Öffn.':>13}{'rechts Δ':>10}   (cm; Würfelkante 5,0)", flush=True)
        for row in sweep:
            print(f"      {row['variante']:<14}{row['left_spread_cm']:>12}"
                  f"{row['left_dist_cm']:>10}{row['right_spread_cm']:>13}"
                  f"{row['right_dist_cm']:>10}", flush=True)
        clamped = [r for r in ranges if r["geklemmt_frames"]]
        if clamped:
            print("      GEKLEMMTE Handgelenke (gesendeter Wert ausserhalb der Grenze):",
                  flush=True)
            for r in clamped:
                print(f"        dim {r['dim']:>2} {r['joint']:<26} gesendet {r['gesendet']} "
                      f"Grenze {r['grenze']}  {r['geklemmt_frames']}/{len(states)} Frames",
                      flush=True)
        else:
            print("      keine Handdimension wird geklemmt.", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tipcheck.json").write_text(
        json.dumps({"frame_source": "motion_onset" if args.frame < 0 else f"fest:{args.frame}",
                    "cube_plane_z_m": CUBE_CENTER_Z_M, "episodes": summary}, indent=2),
        encoding="utf-8")
    print(f"[tipcheck] Bilder und tipcheck.json in {out_dir}")
    print("[tipcheck] Lesart — die WÜRFELmarken sagen nichts aus: das Layout hat den Pixel")
    print("[tipcheck]   mit demselben Modell rückprojiziert, das ihn hier zurückprojiziert.")
    print("[tipcheck]   Entscheidend ist die magenta Kette gegen den realen Roboter:")
    print("[tipcheck]     Becken sitzt richtig, Fehler wächst zur Hand  => Arm-FK/Gelenke.")
    print("[tipcheck]     schon das Becken versetzt, alles gleichsinnig => Kamerapose.")
    print("[tipcheck] fertig.", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        # simulation_app.close() beendet den Prozess hart. Ohne dieses ausdrueckliche
        # Ausgeben verschwindet jeder Fehler spurlos — genau daran scheiterte der erste
        # tipcheck-Lauf am 2026-08-24: kein Traceback, nur "ohne Erfolgsmarker beendet".
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    simulation_app.close()
    raise SystemExit(exit_code)
