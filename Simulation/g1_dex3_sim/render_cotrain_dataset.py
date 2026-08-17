#!/usr/bin/env python3
"""
render_cotrain_dataset.py — erzeugt den GERENDERTEN Datensatz für das Co-Training (Schritt 4).

Aufgabe: (Sim-Bild, echte Aktion)-Paare. Die Aktionen stammen unverändert aus dem realen
Datensatz, die Bilder rendert Isaac Lab in derselben Env, in der später evaluiert wird.
Ein Encoder, der auf beiden Domänen dieselben Aktionen produzieren muss, kann sich nicht
mehr auf die Oberflächen-Statistik der Realbilder verlassen — genau das fordert
`docs/ergebnisse/lauf2-vision-auswertung.md` §6.2.

Ausgabe ist ein vollwertiger LeRobot-**v2.1**-Datensatz (data/, videos/, meta/), also
dasselbe Format, das `LeRobotEpisodeLoader` schon für den echten Datensatz liest. Damit
braucht das Training keinen Sonderpfad, sondern nur einen zweiten `dataset_path` mit
Mischungsverhältnis (`Training/scripts/launch_cotrain.py`).

────────────────────────────────────────────────────────────────────────────────
ZWEI STUFEN, und warum
────────────────────────────────────────────────────────────────────────────────
Der Datensatz speichert **keine Objektposen**. Würde man einfach die aufgezeichneten
Aktionen abspielen, während die Env ihre Würfel zufällig auslegt, entstünden Bild-Aktions-
Paare, in denen der Arm an eine Stelle greift, an der kein Würfel liegt. Ein Encoder lernt
daraus, den Würfel zu IGNORIEREN — das Gegenteil des Ziels. Deshalb:

  1. ``--stage scan``   Episode ohne nennenswertes Rendering abspielen (Kameras auf
                        ~1/10 Auflösung, Übersichtskamera aus) und je Hand den Punkt
                        bestimmen, an dem sich die Finger schließen. Das ist der Ort,
                        an dem in der echten Aufnahme ein Würfel lag.
                        Ergebnis: ``scan.json``.
  2. ``--stage render`` Dieselben Episoden noch einmal, diesmal mit den Würfeln an
                        genau diesen Punkten und in kalibrierter Auflösung (640×480).
                        Ergebnis: der Datensatz.

────────────────────────────────────────────────────────────────────────────────
WAS DER GRIFF KAPUTT MACHT — und warum --stop-at-grasp existiert  (2026-08-17)
────────────────────────────────────────────────────────────────────────────────
Der Scan liefert je Hand GENAU EINEN Greifpunkt (``np.argmin`` über die Öffnungsspur).
Die Aufgabe heißt "stack three block" und braucht zwei bis vier Pick-and-Place-Zyklen.
Für jeden Griff außer einem pro Hand liegt also kein Würfel — genau der Failure-Mode, den
der Zwei-Stufen-Aufbau vermeiden sollte. Dazu kommt: der Würfel wird EINMAL gesetzt, danach
entscheidet die Kontaktphysik. Und die greift im Replay meist nicht — im Lauf vom 2026-08-17
lag die engste erreichte Kuppenöffnung bei 101 von 116 Griffen über 6 cm, bei 5 cm Würfel-
kante. Die Hand schließt sich neben dem Würfel, nicht darum.

Folge für die Beschriftung: Frames VOR dem Griff sind brauchbar (der Würfel liegt dort, wo
der Arm hinfährt), Frames AB dem Griff sind FALSCH beschriftet (Bild: Würfel liegt auf dem
Tisch, Aktion: Würfel transportieren). ``--stop-at-grasp`` schneidet genau dort.

Das ist die Zwischenlösung, nicht das Ziel. Richtig wird es mit Greif-INTERVALLEN statt
-Punkten und kinematischem Attach (Würfelpose zwischen close und release auf den Kuppen-
Schwerpunkt schreiben); dann sind auch Transport- und Stapelphasen verwendbar.

Zwei Isaac-Starts statt einem. Der Scan ist billiger als das Rendern, aber NICHT
vernachlässigbar: im Rauchtest 2026-08-14 lief er mit ~12 Steps/s (Kameras 64×64) gegen
~8,6 Steps/s im Eval bei voller Auflösung. Die Physik (7 Sub-Steps à 5 ms je Policy-Step)
dominiert also, nicht das Rendering — die bekannte 94-%-Zahl aus live-ansicht.md meint
"Sim UND Render" zusammen und sagt über die Aufteilung darin nichts. Rechne mit rund
3 Minuten je Episode für beide Stufen zusammen.

────────────────────────────────────────────────────────────────────────────────
Verwendung (im Sim-Container; der Wrapper `server_rl_run.sh render` macht beides)
────────────────────────────────────────────────────────────────────────────────
    unset VIRTUAL_ENV
    ${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/render_cotrain_dataset.py \\
        --headless --enable_cameras --stage scan \\
        --dataset-path /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset \\
        --out-path    /data/cotrain/g1_dex3_rendered \\
        --num-episodes 60 --asset-path /data/checkpoints/.../g1_dex3.usd
    # danach dasselbe mit --stage render

Rauchtest vor dem langen Lauf (zwei Episoden, je 60 Frames, ~2 min):
    --num-episodes 2 --max-frames-per-episode 60
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI (vor dem AppLauncher — Isaac konsumiert eigene Argumente)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Gerenderten Co-Training-Datensatz erzeugen")
parser.add_argument("--stage", choices=("scan", "render"), required=True,
                    help="scan = Greifpunkte suchen (billig), render = Datensatz schreiben")
parser.add_argument("--dataset-path", type=str, required=True,
                    help="ECHTER Datensatz in LeRobot v2.1 (liefert Aktionen, State, Task)")
parser.add_argument("--out-path", type=str, default="/data/cotrain/g1_dex3_rendered",
                    help="Zielverzeichnis des gerenderten Datensatzes")
parser.add_argument("--num-episodes", type=int, default=60,
                    help="Wie viele Episoden. Gleichmäßig über den TRAIN-Bereich verteilt.")
parser.add_argument("--episode-ids", type=int, nargs="*", default=None,
                    help="Explizite Episoden-Indices (überschreibt --num-episodes)")
parser.add_argument("--train-ratio", type=float, default=0.8,
                    help="Muss dem Trainings-Split entsprechen (lib_split.sh: int(total*ratio)). "
                         "Episoden ab dieser Grenze sind ZURÜCKGEHALTEN und werden nie "
                         "gerendert — sonst wäre die Validierungs-MSE kontaminiert.")
parser.add_argument("--max-frames-per-episode", type=int, default=0,
                    help="0 = ganze Episode. >0 kürzt (Rauchtest).")
parser.add_argument("--layout", type=str, default="",
                    help="layout.json aus extract_block_layout.py — Würfelpositionen, die "
                         "aus dem REALBILD gelesen wurden (Farbblob → Strahl auf die "
                         "Würfelebene). Das ist die richtige Quelle; der Greifpunkt aus "
                         "scan.json ist nur der Notnagel und liegt bei knapp der Hälfte der "
                         "Griffe auf dem Transportweg statt am Pick.")
parser.add_argument("--stop-at-grasp", action="store_true",
                    help="Nur bis zum ersten Zugreifen rendern (Fensterende = kleinstes "
                         "close_step aus scan.json). Bis dorthin liegt der Würfel dort, wo "
                         "der Arm hinfährt; danach entscheidet die Kontaktphysik über seine "
                         "Lage und das Bild zeigt etwas anderes, als die Aktion beschreibt. "
                         "Solche Paare sind FALSCH beschriftet, nicht bloß unscharf.")
parser.add_argument("--grasp-window", type=int, default=0,
                    help="Mit --stop-at-grasp: nur die letzten N Frames vor dem Griff "
                         "rendern (0 = ab Frame 0). Schneidet den Leerlauf-Kopf langer "
                         "Aufnahmen weg und vereinheitlicht das Gewicht der Episoden — "
                         "sonst stellt eine 6791-Frame-Episode ein Sechstel des Satzes.")
parser.add_argument("--min-window", type=int, default=60,
                    help="Mit --stop-at-grasp: Episoden mit kürzerem Fenster überspringen. "
                         "Ein Griff in den ersten Frames ist keine Greifbewegung, sondern "
                         "eine Hand, die schon geschlossen startet.")
parser.add_argument("--asset-path", type=str, default="",
                    help="G1+Dex3 USD-Asset (leer = cfg-Default)")
parser.add_argument("--tracking-error-max", type=float, default=0.15,
                    help="Episoden verwerfen, deren Arme den Aktionen im Mittel schlechter "
                         "als dieser Wert (rad) folgen. Ihr Bild zeigt dann eine andere "
                         "Pose als die Aktion beschreibt — solche Paare sind Gift.")
parser.add_argument("--settle-steps", type=int, default=30,
                    help="Karenz nach dem Setzen der Würfel, bevor aufgezeichnet wird. "
                         "Der Kontakt-Solver schiebt frisch gesetzte Körper sonst noch "
                         "sichtbar (s. run_g1_dex3_sim_eval.py, settle_steps).")
parser.add_argument("--no-place-cubes", action="store_true",
                    help="Würfel NICHT an die Greifpunkte setzen (Ablation). Erzeugt "
                         "Bild-Aktions-Paare ohne visuellen Bezug — nur zum Vergleich.")
parser.add_argument("--overwrite", action="store_true",
                    help="Bereits gerenderte Episoden neu rendern (Default: überspringen)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.stop_at_grasp and args.no_place_cubes:
    raise SystemExit(
        "--stop-at-grasp braucht die Greifpunkte aus scan.json, --no-place-cubes wirft sie "
        "gerade weg. Beides zusammen ergäbe ein Fenster ohne Inhalt."
    )
if args.grasp_window > 0 and not args.stop_at_grasp:
    raise SystemExit("--grasp-window wirkt nur mit --stop-at-grasp (das Fensterende fehlt sonst).")

# Der Scan braucht keine Bildqualität, nur Physik: Kameras klein, Übersichtskamera weg.
# Beides liest G1Dex3BlockstackEnvCfg.__post_init__ aus der Umgebung, deshalb VOR dem
# Import/Bau der Config setzen. Im render-Stage ist die kalibrierte Auflösung Pflicht.
if args.stage == "scan":
    os.environ["CAM_RES_SCALE"] = os.environ.get("SCAN_CAM_RES_SCALE", "0.1")
    os.environ["SCENE_CAM"] = "0"
else:
    scale = os.environ.get("CAM_RES_SCALE", "1").strip() or "1"
    if abs(float(scale) - 1.0) > 1e-6:
        raise SystemExit(
            f"CAM_RES_SCALE={scale} im render-Stage. Die 640×480 sind gegen die "
            "Dataset-Referenzframes kalibriert; ein anders skalierter Datensatz trainiert "
            "auf einer Optik, die im Eval nicht vorkommt. Variable entfernen."
        )
    os.environ.setdefault("SCENE_CAM", "0")   # Übersichtskamera geht nie ins Training

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports nach AppLauncher-Start
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg  # noqa: E402

# Die vier Kameras, die die Policy sieht. Reihenfolge und Namen wie in modality_4cam.json.
CAMS = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist")

# v2.1-Pfadschablonen (identisch zu convert_v3_to_v2.py — der Loader liest sie aus info.json)
DATA_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNK_SIZE = 1000


# ---------------------------------------------------------------------------
# Quelldatensatz lesen
# ---------------------------------------------------------------------------

def read_source_meta(root: Path) -> dict:
    """info.json des echten Datensatzes einlesen (Pfadschablonen, Feature-Liste, fps)."""
    with open(root / "meta" / "info.json") as fh:
        return json.load(fh)


def read_source_tasks(root: Path) -> dict[int, str]:
    """tasks.jsonl → {task_index: task}. Ohne Datei: leer (dann greift der cfg-Default)."""
    path = root / "meta" / "tasks.jsonl"
    if not path.exists():
        return {}
    with open(path) as fh:
        return {int(json.loads(line)["task_index"]): json.loads(line)["task"] for line in fh}


def read_source_lengths(root: Path) -> dict[int, int]:
    """episodes.jsonl → {episode_index: length}. Für die Auslegung der Episodenlänge der Env."""
    path = root / "meta" / "episodes.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            out[int(rec["episode_index"])] = int(rec["length"])
    return out


def load_episode(root: Path, info: dict, ep_idx: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Aktionen (T,28), State (T,28) und task_index einer Episode aus dem Parquet."""
    template = info.get("data_path", DATA_TEMPLATE)
    path = root / template.format(episode_chunk=ep_idx // int(info.get("chunks_size", CHUNK_SIZE)),
                                  episode_index=ep_idx)
    df = pd.read_parquet(path)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
    return action, state, task_index


def select_episodes(info: dict, args) -> list[int]:
    """Welche Episoden gerendert werden — und die Zusicherung, dass keine Test-Episode dabei ist.

    Die Grenze wird mit **derselben** Formel gezogen wie in `Training/scripts/lib_split.sh`
    (``int(total * ratio)``). Weicht sie ab, landen zurückgehaltene Episoden im
    Trainingsmaterial und die Validierungs-MSE des Checkpoint-Sweeps misst nichts mehr.
    """
    total = int(info["total_episodes"])
    n_train = int(total * args.train_ratio)

    if args.episode_ids:
        chosen = sorted(set(args.episode_ids))
        bad = [e for e in chosen if e >= n_train]
        if bad:
            raise SystemExit(
                f"--episode-ids enthält zurückgehaltene Test-Episoden: {bad} "
                f"(train = 0:{n_train} bei ratio={args.train_ratio}). Abbruch."
            )
        return chosen

    n = max(1, min(args.num_episodes, n_train))
    # Gleichmäßig über den Trainingsbereich streuen statt die ersten n zu nehmen: die
    # Episoden sind chronologisch aufgenommen, ein Präfix wäre eine Tageszeit.
    return sorted({int(round(i * (n_train - 1) / max(n - 1, 1))) for i in range(n)})


# ---------------------------------------------------------------------------
# Env-Helfer
# ---------------------------------------------------------------------------

def build_env(args, n_frames: int):
    """Env bauen. episode_length_s so hoch, dass kein Auto-Reset in die Episode fällt.

    ``n_frames`` MUSS die längste zu fahrende Episode sein, nicht die erste: DirectRLEnv
    setzt bei Time-Out selbsttätig zurück, und ein Reset mitten in der Aufnahme legt die
    Würfel neu aus, während die Aktionen weiterlaufen. Der Rest der Episode wäre dann
    stumm entkoppelt — genau der Fehler, den dieses Skript verhindern soll.
    """
    cfg = G1Dex3BlockstackEnvCfg()
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    cfg.episode_length_s = (n_frames + args.settle_steps) / cfg.policy_hz + 10.0
    return G1Dex3BlockstackEnv(cfg=cfg, render_mode=None)


def set_robot_to_state(env, state_row: np.ndarray) -> None:
    """Roboter hart auf die aufgezeichnete Anfangspose setzen.

    Ohne das startet jede Episode in der Home-Pose und die ersten ~30 Frames zeigen den
    Flug dorthin, wo die echte Aufnahme schon steht — Bild und Aktion passen dort nicht
    zusammen. Die Vorzeichen-Konvention ist die des Envs (``_SIGN_FLIP_IDX``): die
    Beobachtung negiert beim LESEN, hier muss beim SCHREIBEN zurückgedreht werden.
    """
    if env._joint_ids is None:
        env._joint_ids = env._build_joint_id_mapping()
    q = state_row.astype(np.float32).copy()
    q[env._SIGN_FLIP_IDX] *= -1

    full = env.robot.data.joint_pos.clone()
    for policy_idx, isaac_idx in enumerate(env._joint_ids):
        full[:, isaac_idx] = float(q[policy_idx])
    zeros = torch.zeros_like(full)
    env.robot.write_joint_state_to_sim(full, zeros)
    env.robot.set_joint_position_target(full)


def hand_spreads_and_centroids(env) -> tuple[np.ndarray, np.ndarray] | None:
    """Je Hand: Fingeröffnung (mittlerer paarweiser Kuppenabstand) und Kuppen-Schwerpunkt.

    Bezugspunkt sind die Fingerkuppen aus dem Env (``get_contact_points_w``), nicht die
    Handfläche — der Versatz bis zur Kuppe war in den Läufen 25–28 die Ursache dreier
    falscher Schlüsse.
    """
    try:
        tips = env.get_contact_points_w()
    except RuntimeError:
        return None
    if tips.shape[1] != 6:
        return None
    tips = tips[0].cpu().numpy()                       # (6,3): 3 links, 3 rechts
    spreads = np.zeros(2, dtype=np.float32)
    centroids = np.zeros((2, 3), dtype=np.float32)
    for h in range(2):
        t3 = tips[3 * h:3 * h + 3]
        spreads[h] = float(np.linalg.norm(t3[[0, 0, 1]] - t3[[1, 2, 2]], axis=-1).mean())
        centroids[h] = t3.mean(axis=0)
    return spreads, centroids


def block_positions(env) -> np.ndarray:
    """Würfelmittelpunkte in Env-Koordinaten, (num_blocks, 3)."""
    pos = torch.stack([b.data.root_pos_w[0] for b in env.blocks], dim=0)
    return (pos - env.scene.env_origins[0]).cpu().numpy()


def play_episode(env, actions: np.ndarray, collect_images: bool, writers=None,
                 track_blocks: bool = False):
    """Eine Episode abspielen.

    Rückgabe: erreichte States, mittlerer Arm-Tracking-Fehler, Fingeröffnung je Step,
    Kuppen-Schwerpunkt je Step, Würfelposen je Step (nur mit ``track_blocks``, sonst None)
    und die tatsächliche Bildgröße (H, W) — Letztere gemessen statt angenommen, damit
    info.json nicht behauptet, was der Renderer nicht geliefert hat.
    """
    n = actions.shape[0]
    achieved = np.zeros((n, 28), dtype=np.float32)
    spread_trace = np.full((n, 2), np.nan, dtype=np.float32)
    centroid_trace = np.full((n, 2, 3), np.nan, dtype=np.float32)
    block_trace = (np.full((n, len(env.blocks), 3), np.nan, dtype=np.float32)
                   if track_blocks else None)
    arm_err = np.zeros(n, dtype=np.float32)
    frame_hw: tuple[int, int] | None = None

    for i in range(n):
        act = torch.tensor(actions[i], dtype=torch.float32, device=env.device).unsqueeze(0)
        obs, _, _, _, _ = env.step(act)

        joint_pos = obs["joint_pos"][0].cpu().numpy()
        achieved[i] = joint_pos
        arm_err[i] = float(np.abs(joint_pos[:14] - actions[i][:14]).mean())

        hands = hand_spreads_and_centroids(env)
        if hands is not None:
            spread_trace[i], centroid_trace[i] = hands
        if block_trace is not None:
            block_trace[i] = block_positions(env)

        if collect_images:
            for cam in CAMS:
                frame = obs[f"video.{cam}"][0].cpu().numpy().astype(np.uint8)
                if frame_hw is None:
                    frame_hw = (int(frame.shape[0]), int(frame.shape[1]))
                writers[cam].append_data(frame)

        if i % 200 == 0:
            print(f"      … Frame {i}/{n}", flush=True)

    return achieved, float(arm_err.mean()), spread_trace, centroid_trace, block_trace, frame_hw


def find_grasp_points(spread_trace: np.ndarray, centroid_trace: np.ndarray,
                      skip: int = 20, min_close_m: float = 0.02) -> list[dict]:
    """Je Hand den Moment des Zugreifens und den dortigen Kuppen-Schwerpunkt bestimmen.

    „Zugreifen" = das Minimum der Fingeröffnung, sofern sie sich über die Episode
    überhaupt um ``min_close_m`` verändert hat. Eine Hand, die nie schließt (im Datensatz
    ist häufig nur eine aktiv), liefert ``ok=False`` und bekommt keinen Würfel gesetzt.
    """
    out = []
    for h in range(2):
        s = spread_trace[skip:, h]
        valid = np.isfinite(s)
        if valid.sum() < 10:
            out.append({"ok": False, "reason": "keine Fingerkuppen-Daten"})
            continue
        s_valid = s[valid]
        idx_valid = np.flatnonzero(valid)
        i_min = int(idx_valid[int(np.argmin(s_valid))]) + skip
        s_min, s_max = float(np.nanmin(s_valid)), float(np.nanmax(s_valid))
        entry = {
            "close_step": i_min,
            "spread_min_cm": round(s_min * 100, 2),
            "spread_max_cm": round(s_max * 100, 2),
            "xy": [round(float(centroid_trace[i_min, h, 0]), 4),
                   round(float(centroid_trace[i_min, h, 1]), 4)],
        }
        if s_max - s_min < min_close_m:
            entry.update(ok=False, reason=f"Hand schließt nicht ({(s_max - s_min) * 100:.1f} cm)")
        else:
            entry["ok"] = True
        out.append(entry)
    return out


def consistency_report(spread: np.ndarray, centroid: np.ndarray,
                       blocks: np.ndarray | None) -> dict | None:
    """Bild-Aktions-Konsistenz als Zahl, statt Endframes anzusehen.

    Die Frage ist NICHT, ob die Episode die Aufgabe löst — die Aktionen stammen aus einer
    echten Aufnahme, ihr Erfolg ist Eigenschaft des realen Datensatzes und hier nicht
    messbar. Die Frage ist, ob das gerenderte BILD zeigt, was die Aktion tut. Deshalb je
    Hand: liegt im Moment des engsten Griffs ein Würfel zwischen den Fingerkuppen? Und je
    Würfel: bewegt er sich überhaupt, hebt er ab?

    ``dist_cm`` ist das Kernmaß. Der Würfel wurde per Konstruktion unter den Greifpunkt
    gelegt, also gehört dort ein kleiner Wert hin. Ein großer Wert heißt: die Hand schließt
    sich neben dem Würfel, und ab diesem Frame beschreibt die Aktion einen Transport, den
    das Bild nicht zeigt.
    """
    if blocks is None or spread.shape[0] == 0:
        return None
    hands: list[dict | None] = []
    for h in range(2):
        s = spread[:, h]
        valid = np.isfinite(s)
        if valid.sum() < 10:
            hands.append(None)
            continue
        idx_valid = np.flatnonzero(valid)
        i_min = int(idx_valid[int(np.argmin(s[valid]))])
        c, b = centroid[i_min, h], blocks[i_min]
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(b)):
            hands.append(None)
            continue
        d3 = np.linalg.norm(b - c, axis=-1)
        j = int(np.argmin(d3))
        hands.append({
            "close_step": i_min,
            "spread_cm": round(float(s[i_min]) * 100, 2),
            "cube": j,
            "dist_cm": round(float(d3[j]) * 100, 2),
            "dist_xy_cm": round(float(np.linalg.norm(b[j][:2] - c[:2])) * 100, 2),
        })

    cubes = []
    for j in range(blocks.shape[1]):
        t = blocks[:, j]
        ok = np.all(np.isfinite(t), axis=-1)
        if ok.sum() < 2:
            cubes.append(None)
            continue
        t = t[ok]
        cubes.append({
            "moved_cm": round(float(np.linalg.norm(t[-1, :2] - t[0, :2])) * 100, 2),
            "lift_cm": round(float(t[:, 2].max() - t[0, 2]) * 100, 2),
        })
    return {"hands": hands, "cubes": cubes}


def summarize_consistency(manifest: dict) -> None:
    """Konsistenz über alle fertigen Episoden — die Zahl, die den Datensatz beurteilt.

    Bewusst am Ende des Laufs und aus dem Manifest: derselbe Überblick entsteht so auch
    nach einem fortgesetzten Lauf, ohne die Episoden erneut abzuspielen.
    """
    recs = [r for r in manifest.get("episodes", {}).values()
            if r.get("status") == "ok" and r.get("consistency")]
    if not recs:
        return
    dists = [h["dist_cm"] for r in recs for h in r["consistency"]["hands"] if h]
    lifts = [c["lift_cm"] for r in recs for c in r["consistency"]["cubes"] if c]
    moved = [c["moved_cm"] for r in recs for c in r["consistency"]["cubes"] if c]
    if not dists:
        return
    near = sum(1 for d in dists if d <= 4.0)
    print(f"\n[render] Konsistenz über {len(recs)} Episoden:")
    print(f"  Abstand Kuppen↔Würfel beim Griff: Median {float(np.median(dists)):.1f} cm, "
          f"p90 {float(np.percentile(dists, 90)):.1f} cm, "
          f"≤ 4 cm bei {near}/{len(dists)} Händen")
    print(f"  Würfel bewegt   > 2 cm: {sum(1 for m in moved if m > 2.0)}/{len(moved)}")
    print(f"  Würfel angehoben> 1 cm: {sum(1 for m in lifts if m > 1.0)}/{len(lifts)}")
    print("  Das misst NICHT Aufgabenerfolg, sondern ob das Bild zur Aktion passt.",
          flush=True)


def stash_cubes(env) -> None:
    """Würfel für den Scan aus dem Weg räumen.

    Der Scan sucht den Punkt, an dem sich die Finger schließen. Läge dort ein zufällig
    ausgelegter Würfel, könnte er die Hand mechanisch blockieren und den gesuchten Punkt
    verfälschen — die Messung würde also von genau dem beeinflusst, was sie erst bestimmen
    soll. Weit weg legen kostet nichts: im Scan wird nichts aufgezeichnet.
    """
    for i, block in enumerate(env.blocks):
        pose = torch.tensor([[5.0 + i, 5.0, 0.05, 1.0, 0.0, 0.0, 0.0]],
                            device=env.device, dtype=torch.float32)
        block.write_root_pose_to_sim(pose)
        block.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))


def place_cubes(env, grasp: list[dict], rng: np.random.Generator,
                layout: list | None = None) -> tuple[list[list[float]], list[str]]:
    """Würfel auslegen — nach Vorrang: Bild-Layout, dann Greifpunkt, dann zufällig.

    **Das Bild-Layout ist die einzige Quelle, die wirklich weiß, wo die Würfel lagen**
    (``extract_block_layout.py``: Farbblob im Realbild → Strahl auf die Würfelebene). Der
    Greifpunkt aus ``scan.json`` ist nur ein Notnagel und ein schlechter: er ist das Minimum
    der Fingeröffnung über die ganze Episode, und weil die Hand beim Pick-and-Place vom
    Zugreifen bis zum Ablegen geschlossen bleibt, liegt dieses Minimum irgendwo auf dem
    Transportweg — 48 von 116 Griffen des Laufs vom 2026-08-17 jenseits von 60 % der
    Episode. Der Würfel landete dann fern vom echten Pick, und der Arm griff ins Leere.

    Nur x/y kommen aus der Quelle, die Höhe ist immer die Tischauflage.
    """
    z = float(env.cfg.block_z_surface)
    origin = env.scene.env_origins[0].cpu().numpy()
    placed: list[np.ndarray] = []
    record: list[list[float]] = []
    source: list[str] = []

    for i, block in enumerate(env.blocks):
        pos, src = None, "random"
        if layout and i < len(layout) and layout[i]:
            x, y = float(layout[i][0]), float(layout[i][1])
            if 0.15 <= x <= 0.70 and -0.35 <= y <= 0.35:
                pos, src = np.array([x, y, z], dtype=np.float32), "layout"
            else:
                print(f"      Würfel {i}: Layout-Punkt ({x:.2f}, {y:.2f}) außerhalb des "
                      f"Tischs — verworfen.", flush=True)
        if pos is None and i < len(grasp) and grasp[i].get("ok"):
            x, y = grasp[i]["xy"]
            # Plausibilitätsfenster um den Tisch: alles weiter draußen ist ein Ausreißer
            # der Kuppen-Rekonstruktion, kein Greifpunkt.
            if 0.20 <= x <= 0.60 and -0.40 <= y <= 0.40:
                pos, src = np.array([x, y, z], dtype=np.float32), "grasp"
            else:
                grasp[i]["ok"] = False
                grasp[i]["reason"] = f"Greifpunkt außerhalb des Tischs ({x:.2f}, {y:.2f})"

        if pos is None:
            # Kein Greifpunkt → zufällig im konfigurierten Band, aber mit Abstand zu den
            # bereits gesetzten Würfeln (zwei 5-cm-Würfel im selben Punkt schießen
            # auseinander, s. _reset_idx).
            for _ in range(50):
                cand = np.array([rng.uniform(*env.cfg.block_x_range),
                                 rng.uniform(*env.cfg.block_y_range), z], dtype=np.float32)
                if all(np.linalg.norm(cand[:2] - p[:2]) > 0.08 for p in placed):
                    pos = cand
                    break
            if pos is None:
                pos = np.array([env.cfg.block_x_range[0], env.cfg.block_y_range[0], z],
                               dtype=np.float32)

        placed.append(pos)
        record.append([round(float(v), 4) for v in pos])
        source.append(src)
        world = torch.tensor([[float(pos[0] + origin[0]), float(pos[1] + origin[1]),
                               float(pos[2]), 1.0, 0.0, 0.0, 0.0]],
                             device=env.device, dtype=torch.float32)
        block.write_root_pose_to_sim(world)
        block.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))

    return record, source


# ---------------------------------------------------------------------------
# Ausgabe-Datensatz schreiben
# ---------------------------------------------------------------------------

def episode_paths(out: Path, ep_idx: int) -> tuple[Path, dict[str, Path]]:
    chunk = ep_idx // CHUNK_SIZE
    parquet = out / DATA_TEMPLATE.format(episode_chunk=chunk, episode_index=ep_idx)
    videos = {
        cam: out / VIDEO_TEMPLATE.format(episode_chunk=chunk,
                                         video_key=f"observation.images.{cam}",
                                         episode_index=ep_idx)
        for cam in CAMS
    }
    return parquet, videos


def open_writers(videos: dict[str, Path], fps: float):
    import imageio
    writers = {}
    for cam, path in videos.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        writers[cam] = imageio.get_writer(
            str(path), format="FFMPEG", mode="I", fps=fps,
            codec="libx264", pixelformat="yuv420p", macro_block_size=8,
            ffmpeg_log_level="error",
        )
    return writers


def write_parquet(path: Path, ep_idx: int, achieved: np.ndarray, actions: np.ndarray,
                  real_state: np.ndarray, task_index: int, fps: float, index_offset: int) -> None:
    """Ein Episoden-Parquet im v2.1-Schema.

    ``observation.state`` ist der in der Sim ERREICHTE Zustand, nicht der aufgezeichnete:
    nur so gehören Bild und State zusammen. Der aufgezeichnete Zustand bleibt als
    ``observation.state_real`` daneben liegen — nicht in info.json deklariert, also für
    Loader und Statistik unsichtbar, aber für die Nachprüfung da.
    """
    n = achieved.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "observation.state": list(achieved.astype(np.float32)),
        "action": list(actions[:n].astype(np.float32)),
        "observation.state_real": list(real_state[:n].astype(np.float32)),
        "timestamp": np.arange(n, dtype=np.float32) / float(fps),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, ep_idx, dtype=np.int64),
        "index": np.arange(index_offset, index_offset + n, dtype=np.int64),
        "task_index": np.full(n, task_index, dtype=np.int64),
    })
    df.to_parquet(path, index=False)


def finalize_meta(out: Path, src_info: dict, tasks: dict[int, str], manifest: dict,
                  fps: float) -> None:
    """meta/ aus dem schreiben, was tatsächlich auf der Platte liegt.

    Bewusst aus dem Dateibestand und nicht aus dem Lauf-Zustand: so ist ein abgebrochener
    und fortgesetzter Lauf am Ende genauso konsistent wie ein durchgelaufener.
    """
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    episodes = []
    total_frames = 0
    for ep_idx in sorted(int(k) for k in manifest["episodes"]):
        rec = manifest["episodes"][str(ep_idx)]
        if rec.get("status") != "ok":
            continue
        parquet, videos = episode_paths(out, ep_idx)
        if not parquet.exists() or not all(v.exists() for v in videos.values()):
            continue
        length = int(rec["length"])
        total_frames += length
        episodes.append({
            "episode_index": ep_idx,
            "tasks": [rec.get("task", "stack the blocks")],
            "length": length,
        })

    with open(meta / "episodes.jsonl", "w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep) + "\n")

    task_map = tasks or {0: "stack the blocks"}
    with open(meta / "tasks.jsonl", "w") as fh:
        for idx, task in sorted(task_map.items()):
            fh.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # modality.json unverändert vom echten Datensatz übernehmen — dieselben Kameranamen,
    # dieselbe Aufteilung der 28 Dimensionen. Ein abweichender Zuschnitt wäre ein stiller
    # Trainingsfehler, deshalb kopieren statt nachbauen.
    src_modality = Path(args.dataset_path) / "meta" / "modality.json"
    if src_modality.exists():
        (meta / "modality.json").write_text(src_modality.read_text())

    features = {
        "observation.state": {"dtype": "float32", "shape": [28], "fps": fps},
        "action": {"dtype": "float32", "shape": [28], "fps": fps},
        "timestamp": {"dtype": "float32", "shape": [1], "fps": fps},
        "frame_index": {"dtype": "int64", "shape": [1], "fps": fps},
        "episode_index": {"dtype": "int64", "shape": [1], "fps": fps},
        "index": {"dtype": "int64", "shape": [1], "fps": fps},
        "task_index": {"dtype": "int64", "shape": [1], "fps": fps},
    }
    for cam in CAMS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [3, manifest["frame_height"], manifest["frame_width"]],
            "info": {
                "video.fps": fps,
                "video.height": manifest["frame_height"],
                "video.width": manifest["frame_width"],
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    # Der höchste Episoden-Index +1, NICHT die Anzahl: die Indices sind bewusst die des
    # Quelldatensatzes (dünn besetzt), damit ein fortgesetzter Lauf dieselbe Episode
    # wiedererkennt. Der Split-Filter des Loaders vergleicht gegen genau diesen Bereich.
    highest = (max(ep["episode_index"] for ep in episodes) + 1) if episodes else 0
    info = {
        "codebase_version": "v2.1",
        "robot_type": "Unitree_G1_sim",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(task_map),
        "total_videos": len(episodes) * len(CAMS),
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{highest}"},
        "data_path": DATA_TEMPLATE,
        "video_path": VIDEO_TEMPLATE,
        "features": features,
    }
    with open(meta / "info.json", "w") as fh:
        json.dump(info, fh, indent=4)

    with open(out / "render_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[render] meta/ geschrieben: {len(episodes)} Episoden, {total_frames} Frames.")
    print(f"[render] Datensatz: {out}")


# ---------------------------------------------------------------------------
# Stufen
# ---------------------------------------------------------------------------

def episode_window(info: dict, n_src: int) -> tuple[int, int, str]:
    """Welcher Frame-Bereich gerendert wird — und warum. Rückgabe ``(start, stop, Grund)``.

    Ohne ``--stop-at-grasp`` die ganze Episode. Mit dem Flag endet das Fenster am
    **frühesten** Griff beider Hände, nicht am spätesten: sobald eine Hand zugreift, ist
    ihr Würfel der Physik überlassen — und er ist auch in der Kamera der anderen Hand zu
    sehen. Das späteste close_step zu nehmen hieße, für die eine Hand konsistente Frames mit
    für die andere schon falschen zu erkaufen.

    ``start == stop`` heißt „diese Episode liefert kein brauchbares Fenster".
    """
    if not args.stop_at_grasp:
        return 0, n_src, "ganze Episode"
    closes = [int(h["close_step"]) for h in info.get("hands", [])
              if h.get("ok") and h.get("close_step") is not None]
    if not closes:
        return 0, 0, "kein Greifpunkt im Scan"
    stop = min(min(closes), n_src)
    start = max(0, stop - args.grasp_window) if args.grasp_window > 0 else 0
    if stop - start < args.min_window:
        return start, start, f"Fenster {stop - start} < {args.min_window} Frames"
    return start, stop, f"Frames {start}–{stop}, Griff bei {stop}"


def expected_length(src_lengths: dict[int, int], ep_idx: int, info: dict | None = None) -> int:
    """Wie viele Frames diese Episode im Ergebnis haben MUSS (0 = unbekannt).

    Der Grund ist der Rauchtest: er läuft mit ``--max-frames-per-episode 60``, schreibt
    also 2-Sekunden-Stummel. Ohne diesen Vergleich hielte der Fortsetz-Mechanismus sie
    danach für fertig und der echte Lauf würde sie überspringen — der Trainingsdatensatz
    enthielte stumm zwei abgeschnittene Episoden.

    ``info`` ist der Scan-Eintrag und darf nur im render-Stage mitkommen: mit
    ``--stop-at-grasp`` ist die Soll-Länge das Fenster, nicht die Quell-Länge. Ohne diese
    Unterscheidung würde jeder fortgesetzte Lauf alles neu rendern, weil gekürzte Episoden
    per Definition kürzer sind als die Quelle.
    """
    n = int(src_lengths.get(ep_idx, 0))
    if args.max_frames_per_episode > 0:
        n = min(n, args.max_frames_per_episode) if n else args.max_frames_per_episode
    if args.stop_at_grasp and info is not None and n:
        start, stop, _ = episode_window(info, n)
        return stop - start
    return n


def run_scan(env_builder, src_root: Path, src_info: dict, episodes: list[int],
             out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    scan_path = out / "scan.json"
    scan = {"episodes": {}}
    if scan_path.exists() and not args.overwrite:
        scan = json.loads(scan_path.read_text())

    src_lengths = read_source_lengths(src_root)
    env = None
    for k, ep_idx in enumerate(episodes, 1):
        old = scan["episodes"].get(str(ep_idx))
        want = expected_length(src_lengths, ep_idx)
        if old and not args.overwrite:
            if want and int(old.get("length", 0)) != want:
                print(f"[scan] ({k}/{len(episodes)}) Episode {ep_idx}: alter Eintrag über "
                      f"{old.get('length')} statt {want} Frames — wird neu gescannt.", flush=True)
            else:
                print(f"[scan] ({k}/{len(episodes)}) Episode {ep_idx}: schon gescannt.",
                      flush=True)
                continue
        actions, state, task_index = load_episode(src_root, src_info, ep_idx)
        if args.max_frames_per_episode > 0:
            actions, state = actions[:args.max_frames_per_episode], \
                state[:args.max_frames_per_episode]

        if env is None:
            env = env_builder()
        env.reset()
        stash_cubes(env)
        set_robot_to_state(env, state[0])
        for _ in range(args.settle_steps):
            env.step(torch.tensor(state[0], dtype=torch.float32,
                                  device=env.device).unsqueeze(0))

        print(f"[scan] ({k}/{len(episodes)}) Episode {ep_idx}: {actions.shape[0]} Frames",
              flush=True)
        _, arm_err, spread, centroid, _, _ = play_episode(env, actions, collect_images=False)
        grasp = find_grasp_points(spread, centroid)
        scan["episodes"][str(ep_idx)] = {
            "length": int(actions.shape[0]),
            "task_index": task_index,
            "arm_tracking_error_rad": round(arm_err, 4),
            "hands": grasp,
        }
        for h, g in enumerate(grasp):
            side = "links" if h == 0 else "rechts"
            if g.get("ok"):
                print(f"        {side}: Greifpunkt {g['xy']} bei Step {g['close_step']} "
                      f"(Öffnung {g['spread_max_cm']}→{g['spread_min_cm']} cm)", flush=True)
            else:
                print(f"        {side}: kein Greifpunkt — {g.get('reason', '?')}", flush=True)
        print(f"        Arm-Tracking-Fehler {arm_err:.3f} rad", flush=True)
        scan_path.write_text(json.dumps(scan, indent=2))

    scan_path.write_text(json.dumps(scan, indent=2))
    ok = sum(1 for v in scan["episodes"].values()
             if v["arm_tracking_error_rad"] <= args.tracking_error_max)
    grasped = sum(1 for v in scan["episodes"].values()
                  if any(h.get("ok") for h in v["hands"]))
    print(f"\n[scan] {len(scan['episodes'])} Episoden gescannt, {ok} unter der "
          f"Tracking-Schwelle, {grasped} mit mindestens einem Greifpunkt.")
    print(f"[scan] {scan_path}")
    print("[scan] fertig.", flush=True)


def run_render(env_builder, src_root: Path, src_info: dict, episodes: list[int],
               out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    scan_path = out / "scan.json"
    scan = {"episodes": {}}
    if scan_path.exists():
        scan = json.loads(scan_path.read_text())
    elif args.stop_at_grasp:
        raise SystemExit(
            f"{scan_path} fehlt, --stop-at-grasp braucht daraus aber das close_step, um das "
            "Fenster zu schneiden. Erst `--stage scan` fahren."
        )
    elif not args.no_place_cubes and not args.layout:
        raise SystemExit(
            f"{scan_path} fehlt und kein --layout — dann lägen die Würfel zufällig und die "
            "Bild-Aktions-Paare wären visuell entkoppelt. Empfohlen ist --layout "
            "(extract_block_layout.py); --no-place-cubes erzwingt die Entkopplung als Ablation."
        )

    # Bild-Layout: die einzige Quelle, die weiß, wo die Würfel wirklich lagen. Fehlt es,
    # fällt place_cubes auf den Greifpunkt zurück — laut, weil das der Modus ist, in dem der
    # Arm ins Leere greift.
    layout: dict = {}
    if args.layout:
        lp = Path(args.layout)
        if not lp.exists():
            raise SystemExit(f"--layout {lp} existiert nicht. Erst extract_block_layout.py fahren.")
        layout = json.loads(lp.read_text()).get("episodes", {})
        print(f"[render] Layout aus {lp}: {len(layout)} Episoden.", flush=True)
    else:
        print("[render] WARNUNG: kein --layout. Die Würfel landen am Greifpunkt aus "
              "scan.json, und der liegt bei knapp der Hälfte der Griffe auf dem "
              "Transportweg statt am Pick.", flush=True)

    manifest_path = out / "render_manifest.json"
    manifest = {"episodes": {}, "frame_width": 640, "frame_height": 480,
                "source_dataset": str(src_root), "place_cubes": not args.no_place_cubes}
    if manifest_path.exists() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("episodes", {})

    fps = float(src_info.get("fps", 30.0))
    tasks = read_source_tasks(src_root)
    src_lengths = read_source_lengths(src_root)
    rng = np.random.default_rng(0)
    env = None
    index_offset = sum(int(r.get("length", 0)) for r in manifest["episodes"].values()
                       if r.get("status") == "ok")

    for k, ep_idx in enumerate(episodes, 1):
        head = f"[render] ({k}/{len(episodes)}) Episode {ep_idx}"
        info = scan["episodes"].get(str(ep_idx), {})
        parquet, videos = episode_paths(out, ep_idx)
        done = parquet.exists() and all(v.exists() for v in videos.values())
        if done and not args.overwrite:
            want = expected_length(src_lengths, ep_idx, info)
            have = int(manifest["episodes"].get(str(ep_idx), {}).get("length", 0))
            if want and have and have != want:
                print(f"{head}: vorhandene Fassung hat {have} statt {want} Frames "
                      f"(Rauchtest? anderes Fenster?) — wird neu gerendert.", flush=True)
            else:
                print(f"{head}: liegt schon vor — übersprungen.", flush=True)
                continue

        if not info and not args.no_place_cubes and str(ep_idx) not in layout:
            # Weder Scan-Eintrag noch Layout: die Wuerfel landen zufaellig, das
            # Bild-Aktions-Paar ist visuell entkoppelt. Laut, nicht still — meist steht ein
            # anderes RENDER_EPISODES dahinter als beim Scan bzw. beim Layout.
            print(f"{head}: WARNUNG — weder Scan-Eintrag noch Layout, Würfel liegen "
                  f"zufällig. Mit denselben Episoden nachholen.", flush=True)
        err = info.get("arm_tracking_error_rad")
        if err is not None and err > args.tracking_error_max:
            print(f"{head}: VERWORFEN — Arm-Tracking {err:.3f} rad > "
                  f"{args.tracking_error_max} rad.", flush=True)
            manifest["episodes"][str(ep_idx)] = {"status": "rejected_tracking",
                                                 "arm_tracking_error_rad": err}
            continue

        actions, state, task_index = load_episode(src_root, src_info, ep_idx)
        if args.max_frames_per_episode > 0:
            actions, state = actions[:args.max_frames_per_episode], \
                state[:args.max_frames_per_episode]

        start, stop, why = episode_window(info, actions.shape[0])
        if stop - start <= 0:
            print(f"{head}: ÜBERSPRUNGEN — {why}.", flush=True)
            manifest["episodes"][str(ep_idx)] = {"status": "skipped_window", "reason": why}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            continue
        if args.stop_at_grasp:
            # Der Startzustand wird gleich per set_robot_to_state hart gesetzt, ein Fenster
            # mitten in der Episode ist also kein Sonderfall — genau dafür existiert die
            # Funktion schon.
            actions, state = actions[start:stop], state[start:stop]

        if env is None:
            env = env_builder()
        env.reset()
        cubes, cube_src = None, None
        if not args.no_place_cubes:
            cubes, cube_src = place_cubes(env, info.get("hands", []), rng,
                                          layout=layout.get(str(ep_idx), {}).get("cubes"))
        set_robot_to_state(env, state[0])
        for _ in range(args.settle_steps):
            env.step(torch.tensor(state[0], dtype=torch.float32,
                                  device=env.device).unsqueeze(0))

        print(f"{head}: {actions.shape[0]} Frames ({why}), Würfel {cubes} "
              f"aus {cube_src}", flush=True)
        writers = open_writers(videos, fps)
        try:
            achieved, arm_err, spread, centroid, block_trace, frame_hw = play_episode(
                env, actions, collect_images=True, writers=writers,
                track_blocks=not args.no_place_cubes)
        finally:
            for w in writers.values():
                w.close()
        if frame_hw:
            manifest["frame_height"], manifest["frame_width"] = frame_hw

        if arm_err > args.tracking_error_max:
            # Erst hier messbar, wenn kein Scan-Eintrag vorlag. Dateien wieder entfernen,
            # damit keine halbgute Episode im Datensatz landet.
            print(f"{head}: VERWORFEN nach dem Rendern — Tracking {arm_err:.3f} rad.",
                  flush=True)
            for v in videos.values():
                v.unlink(missing_ok=True)
            manifest["episodes"][str(ep_idx)] = {"status": "rejected_tracking",
                                                 "arm_tracking_error_rad": round(arm_err, 4)}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            continue

        cons = consistency_report(spread, centroid, block_trace)
        write_parquet(parquet, ep_idx, achieved, actions, state, task_index, fps, index_offset)
        index_offset += achieved.shape[0]
        manifest["episodes"][str(ep_idx)] = {
            "status": "ok",
            "length": int(achieved.shape[0]),
            "source_window": [start, stop],
            "task": tasks.get(task_index, "stack the blocks"),
            "task_index": task_index,
            "arm_tracking_error_rad": round(arm_err, 4),
            "cubes_xyz": cubes,
            "cube_source": cube_src,
            "grasp_points": info.get("hands"),
            "consistency": cons,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        note = ""
        if cons:
            d = [h["dist_cm"] for h in cons["hands"] if h]
            if d:
                note = f", Kuppen↔Würfel beim Griff {min(d):.1f} cm"
        print(f"{head}: geschrieben (Tracking {arm_err:.3f} rad{note}).", flush=True)

    finalize_meta(out, src_info, tasks, manifest, fps)
    summarize_consistency(manifest)
    print("[render] fertig.", flush=True)


def main():
    src_root = Path(args.dataset_path)
    out = Path(args.out_path)
    src_info = read_source_meta(src_root)
    episodes = select_episodes(src_info, args)

    # Die Env wird EINMAL gebaut und für alle Episoden wiederverwendet (ein Isaac-Kontext
    # je Prozess). Ihre Episodenlänge muss deshalb für die längste Episode reichen.
    lengths = read_source_lengths(src_root)
    max_len = max((lengths.get(e, 0) for e in episodes), default=0)
    if args.max_frames_per_episode > 0:
        max_len = min(max_len or args.max_frames_per_episode, args.max_frames_per_episode)
    if max_len <= 0:
        # Kein episodes.jsonl → auf die längste bekannte Teleop-Demo aufrunden.
        max_len = 2000
        print("[warn] meta/episodes.jsonl liefert keine Längen — Env auf 2000 Frames "
              "ausgelegt. Längere Episoden würden mitten im Lauf zurückgesetzt.", flush=True)

    print("=" * 72)
    print(f"G1+Dex3 — Co-Training-Datensatz, Stufe '{args.stage}'")
    print("=" * 72)
    print(f"  Quelle:      {src_root}")
    print(f"  Ziel:        {out}")
    print(f"  Episoden:    {len(episodes)} → {episodes[:8]}{' …' if len(episodes) > 8 else ''}")
    print(f"  Train-Grenze: 0:{int(int(src_info['total_episodes']) * args.train_ratio)} "
          f"von {src_info['total_episodes']} (Test-Episoden bleiben ungerendert)")
    print(f"  Längste Ep.: {max_len} Frames → episode_length_s entsprechend gesetzt")
    cube_mode = "zufällig (ABLATION)" if args.no_place_cubes else "an den Greifpunkten"
    print(f"  Würfel:      {cube_mode}")
    if args.stop_at_grasp:
        win = f"letzte {args.grasp_window} Frames vor dem Griff" if args.grasp_window \
            else "Frame 0 bis zum Griff"
        print(f"  Fenster:     {win}, mind. {args.min_window} Frames "
              f"(ab dem Griff wäre das Bild-Aktions-Paar falsch beschriftet)")
    else:
        print("  Fenster:     ganze Episode — Frames AB dem Griff sind falsch beschriftet, "
              "solange kein Attach existiert (--stop-at-grasp)")
    print()

    def env_builder():
        return build_env(args, max_len)

    if args.stage == "scan":
        run_scan(env_builder, src_root, src_info, episodes, out)
    else:
        run_render(env_builder, src_root, src_info, episodes, out)

    simulation_app.close()


if __name__ == "__main__":
    main()
