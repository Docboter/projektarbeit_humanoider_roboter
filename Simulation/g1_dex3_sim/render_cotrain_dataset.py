#!/usr/bin/env python3
# TL;DR: Rendert den Co-Training-Datensatz (Sim-Bild + echte Aktion); `server_rl_run.sh render`.
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
parser.add_argument("--cube-source", choices=("layout", "grasp"), default="layout",
                    help="layout = Wuerfel an die Lage aus dem Realbild (Standard); "
                         "grasp = den gegriffenen Wuerfel zusaetzlich auf den Kuppen-"
                         "Schwerpunkt der Hand ziehen, damit der Griff im Bild aufgeht")
parser.add_argument("--ignore-yaw", action="store_true",
                    help="Gierwinkel aus layout.json NICHT verwenden, alle Wuerfel "
                         "achsparallel setzen. Nur fuer den A/B-Vergleich: derselbe "
                         "Episodensatz einmal mit und einmal ohne Drehung, sonst ist "
                         "nicht zu trennen, ob eine Verbesserung vom Winkel kommt "
                         "oder von der Episodenauswahl")
parser.add_argument("--dr-seed", type=int, default=0,
                    help="Seed der Domain Randomization (Licht- und Materialfarben). "
                         "Jede Episode zieht aus default_rng([seed, episode_index]), das "
                         "Aussehen haengt also an der Episode und nicht daran, an welcher "
                         "Stelle des Laufs sie gerendert wurde — reproduzierbar auch nach "
                         "Resume oder bei anders ausgelassenen Fenstern. NEGATIV = wie "
                         "frueher pro Prozess neu auswuerfeln (dann sind zwei Laeufe nicht "
                         "vergleichbar)")
parser.add_argument("--no-dr", action="store_true",
                    help="Domain Randomization ganz aus: festes Licht, feste Farben. Fuer "
                         "den A/B-Vergleich die schaerfste Variante — die Bildpaare "
                         "unterscheiden sich dann NUR im geprueften Faktor")
parser.add_argument("--max-anchor-shift", type=float, default=0.08,
                    help="Wieviel der Greifanker hoechstens von der Bild-Lage abweichen darf "
                         "(m). Darueber wird die Episode verworfen statt geraten")
parser.add_argument("--layout", type=str, default="",
                    help="layout.json aus extract_block_layout.py — Würfelpositionen, die "
                         "aus dem REALBILD gelesen wurden (Farbblob → Strahl auf die "
                         "Würfelebene) UND je Episode der Bewegungsbeginn, der das "
                         "Renderfenster begrenzt. Ohne Eintrag wird die Episode verworfen — "
                         "geraten wird nicht.")
parser.add_argument("--stop-at-grasp", action="store_true",
                    help="Nur bis zur ersten Würfelbewegung rendern (Fensterende = "
                         "motion_onset.first aus layout.json). Bis dorthin liegt der Würfel "
                         "dort, wo ihn das Layout hinsetzt; danach hat die reale Hand ihn "
                         "mitgenommen, der simulierte bleibt liegen, und das Bild zeigt etwas "
                         "anderes, als die Aktion beschreibt. Solche Paare sind FALSCH "
                         "beschriftet, nicht bloß unscharf.")
parser.add_argument("--grasp-window", type=int, default=0,
                    help="Mit --stop-at-grasp: nur die letzten N Frames vor der "
                         "Würfelbewegung "
                         "rendern (0 = ab Frame 0). Schneidet den Leerlauf-Kopf langer "
                         "Aufnahmen weg und vereinheitlicht das Gewicht der Episoden — "
                         "sonst stellt eine 6791-Frame-Episode ein Sechstel des Satzes.")
parser.add_argument("--min-window", type=int, default=60,
                    help="Mit --stop-at-grasp: Episoden mit kürzerem Fenster überspringen. "
                         "Bewegt sich ein Würfel schon in den ersten Frames, ist das keine "
                         "Greifbewegung, sondern eine Fehlauslösung oder eine bereits "
                         "gestörte Szene.")
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
        "--stop-at-grasp schneidet das Fenster an der Würfelbewegung, --no-place-cubes setzt "
        "gar keine Würfel. Beides zusammen ergäbe ein Fenster ohne Inhalt."
    )
if args.stop_at_grasp and not args.layout:
    raise SystemExit(
        "--stop-at-grasp braucht --layout: das Fensterende (motion_onset) steht in "
        "layout.json. Erst extract_block_layout.py fahren."
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
    # DR ist fuer den Datensatz erwuenscht (sie ist der halbe Zweck der gerenderten
    # Bilder), aber sie muss reproduzierbar sein: sonst wuerfeln zwei Laeufe Licht und
    # Wuerfelfarben unabhaengig neu und ein A/B laesst sich nicht mehr auf den geprueften
    # Faktor zurueckfuehren. Deshalb hier per Default ein fester Seed statt keinem.
    cfg.dr_enabled = not args.no_dr
    cfg.dr_seed = None if args.dr_seed < 0 else int(args.dr_seed)
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


def hand_spreads_and_centroids(env) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Je Hand: Fingeröffnung, Kuppen-Schwerpunkt und Greifachsen-Gierwinkel.

    Bezugspunkt sind die Fingerkuppen aus dem Env (``get_contact_points_w``), nicht die
    Handfläche — der Versatz bis zur Kuppe war in den Läufen 25–28 die Ursache dreier
    falscher Schlüsse. Reihenfolge je Hand ist [Zeigefinger, Mittelfinger, Daumen], siehe
    ``_REACH_BODY_SETS``.

    Der Gierwinkel ist die Richtung Daumen → Mitte der Gegenfinger, auf die Tischebene
    projiziert und mod 90° genommen. Er ist die BILDUNABHÄNGIGE Gegenprobe zum Winkel aus
    ``extract_block_layout.py``: wer einen Würfel greift, legt die Greifachse quer zu einer
    Fläche, nicht zu einer Ecke. Stimmen beide Zahlen überein, bestätigen sich zwei
    Verfahren, die nichts miteinander zu tun haben. Er ist kein Ersatz für den Bildwinkel —
    er belegt nur, ob die Hand zur angenommenen Würfellage passt.
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
    yaws = np.zeros(2, dtype=np.float32)
    for h in range(2):
        t3 = tips[3 * h:3 * h + 3]
        spreads[h] = float(np.linalg.norm(t3[[0, 0, 1]] - t3[[1, 2, 2]], axis=-1).mean())
        centroids[h] = t3.mean(axis=0)
        axis = t3[2] - 0.5 * (t3[0] + t3[1])            # Daumen → Mitte Zeige/Mittel
        yaws[h] = float(np.degrees(np.arctan2(axis[1], axis[0])) % 90.0)
    return spreads, centroids, yaws


def block_positions(env) -> np.ndarray:
    """Würfelmittelpunkte in Env-Koordinaten, (num_blocks, 3)."""
    pos = torch.stack([b.data.root_pos_w[0] for b in env.blocks], dim=0)
    return (pos - env.scene.env_origins[0]).cpu().numpy()


def play_episode(env, actions: np.ndarray, collect_images: bool, writers=None,
                 track_blocks: bool = False):
    """Eine Episode abspielen.

    Rückgabe: erreichte States, Tracking-Fehler (Arm gemittelt, Hand je Step), Fingeröffnung
    je Step,
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
    # Handfehler je Step und je Hand. Der Arm wird gemittelt, die Hand NICHT: entscheidend
    # ist der Greifmoment am Fensterende, und ein Mittel ueber die ganze Episode verduennt
    # ihn mit den Frames, in denen die Hand frei in der Luft steht und muehelos folgt.
    hand_err = np.zeros((n, 2), dtype=np.float32)
    frame_hw: tuple[int, int] | None = None

    for i in range(n):
        act = torch.tensor(actions[i], dtype=torch.float32, device=env.device).unsqueeze(0)
        obs, _, _, _, _ = env.step(act)

        joint_pos = obs["joint_pos"][0].cpu().numpy()
        achieved[i] = joint_pos
        arm_err[i] = float(np.abs(joint_pos[:14] - actions[i][:14]).mean())
        hand_err[i, 0] = float(np.abs(joint_pos[14:21] - actions[i][14:21]).mean())
        hand_err[i, 1] = float(np.abs(joint_pos[21:28] - actions[i][21:28]).mean())

        hands = hand_spreads_and_centroids(env)
        if hands is not None:
            spread_trace[i], centroid_trace[i], _ = hands
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

    return (achieved, float(arm_err.mean()), hand_err, spread_trace, centroid_trace,
            block_trace, frame_hw)


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
    messbar. Die Frage ist, ob das gerenderte BILD zeigt, was die Aktion tut.

    Gemessen wird am **letzten Frame des Fensters**. Das ist der Bewegungsbeginn aus
    ``layout.json``: der Moment, in dem sich der Würfel im Realvideo nachweislich zu bewegen
    beginnt, also von einer Hand mitgenommen wird. Passt die Kette aus Würfellage,
    Kameramodell und Roboter-FK zusammen, muss dort eine Fingerkuppe am Würfel stehen.

    Bis 2026-08-22 wurde stattdessen am Minimum der Fingeröffnung gemessen. Das ist für die
    DEX3 keine sinnvolle Stelle — bei 101 von 116 Griffen bleibt die engste Kuppenöffnung
    über 6 cm bei 5 cm Würfelkante —, und weil das Minimum innerhalb des Fensters gesucht
    wird, verschob ein kürzeres (richtigeres) Fenster die Messstelle nach vorne und ließ die
    Zahl schlechter aussehen, obwohl der Datensatz besser wurde.

    ``dist_end_cm`` ist das Kernmaß, ``dist_min_cm`` die beste Annäherung im Fenster
    überhaupt. Sind beide groß, beschreibt die Aktion einen Griff, den das Bild nicht zeigt.
    """
    if blocks is None or spread.shape[0] == 0:
        return None
    hands: list[dict | None] = []
    for h in range(2):
        c_all = centroid[:, h]
        ok = np.all(np.isfinite(c_all), axis=-1) & np.all(np.isfinite(blocks), axis=(1, 2))
        if ok.sum() < 2:
            hands.append(None)
            continue
        idx = np.flatnonzero(ok)
        # (Frames, Würfel) — Abstand jeder Kuppenmitte zu jedem Würfel
        d = np.linalg.norm(blocks[idx] - c_all[idx][:, None, :], axis=-1)
        end = int(idx[-1])
        j_end = int(np.argmin(d[-1]))
        f_min, j_min = np.unravel_index(int(np.argmin(d)), d.shape)
        hands.append({
            "end_step": end,
            "cube": j_end,
            "dist_end_cm": round(float(d[-1, j_end]) * 100, 2),
            "dist_end_xy_cm": round(float(np.linalg.norm(
                blocks[end][j_end][:2] - c_all[end][:2])) * 100, 2),
            "dist_min_cm": round(float(d[f_min, j_min]) * 100, 2),
            "dist_min_step": int(idx[f_min]),
            "spread_end_cm": (round(float(spread[end, h]) * 100, 2)
                              if np.isfinite(spread[end, h]) else None),
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
    ends = [h["dist_end_cm"] for r in recs for h in r["consistency"]["hands"] if h]
    mins = [h["dist_min_cm"] for r in recs for h in r["consistency"]["hands"] if h]
    lifts = [c["lift_cm"] for r in recs for c in r["consistency"]["cubes"] if c]
    moved = [c["moved_cm"] for r in recs for c in r["consistency"]["cubes"] if c]
    if not ends:
        return
    near = sum(1 for d in ends if d <= 4.0)
    print(f"\n[render] Konsistenz über {len(recs)} Episoden:")
    print(f"  Kuppen↔Würfel am Fensterende (Bewegungsbeginn): Median "
          f"{float(np.median(ends)):.1f} cm, p90 {float(np.percentile(ends, 90)):.1f} cm, "
          f"≤ 4 cm bei {near}/{len(ends)} Händen")
    print(f"  beste Annäherung im Fenster: Median {float(np.median(mins)):.1f} cm, "
          f"min {float(np.min(mins)):.1f} cm")
    print(f"  Würfel bewegt   > 2 cm: {sum(1 for m in moved if m > 2.0)}/{len(moved)}")
    print(f"  Würfel angehoben> 1 cm: {sum(1 for m in lifts if m > 1.0)}/{len(lifts)}")
    print("  Das misst NICHT Aufgabenerfolg, sondern ob das Bild zur Aktion passt: am "
          "Fensterende bewegt sich der reale Würfel, dort MUSS eine Kuppe an ihm stehen.",
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


def grasp_anchor(env, states: np.ndarray, layout: list, until: int,
                 max_shift_m: float) -> dict | None:
    """Wo die HAND den Würfel greift — als Korrektur zur Lage aus dem Bild.

    Warum überhaupt: für Co-Training zählt nicht, ob der Würfel dort steht, wo er im
    Realbild lag, sondern ob er dort steht, wo die Hand ihn greift. Sonst zeigt das
    gerenderte Bild einen Griff, der danebengeht — als Aufsicht schlimmer als gar keine.
    Nach den Geometriekorrekturen vom 2026-08-24 bleiben zwischen Kuppen und Bild-Lage rund
    4,5 cm; bei 5 cm Würfelkante reicht das zum Anstoßen, nicht zum Greifen (Lauf 53).

    Gesucht wird der Frame VOR ``until`` (dem Bewegungsbeginn), an dem eine Handmitte einem
    Layout-Würfel am nächsten kommt — davor gilt dessen Ruhelage, danach trägt die Hand ihn
    bereits. Der Kuppen-Schwerpunkt dort ist der Anker.

    Nicht zu verwechseln mit dem alten ``find_grasp_points``: das nahm das Minimum der
    FINGERÖFFNUNG über die GANZE Episode und landete damit auf dem Transportweg.

    ``None`` heißt: kein Anker verwendbar — entweder gibt es keine Kuppen-Daten, oder der
    Anker liegt weiter als ``max_shift_m`` von der Bild-Lage entfernt. Der zweite Fall ist
    die Plausibilitätsschranke: zwei unabhängige Quellen, die weit auseinanderliegen,
    bezeugen einander nicht.
    """
    targets = [(i, np.asarray(xy[:2], dtype=np.float64))
               for i, xy in enumerate(layout or []) if xy]
    if not targets:
        return None
    best = None
    for frame in range(0, min(int(until) + 1, len(states)), 2):
        set_robot_to_state(env, states[frame])
        env.sim.forward()
        env.robot.update(float(env.cfg.sim.dt))
        sc = hand_spreads_and_centroids(env)
        if sc is None:
            continue
        _, centroids, hand_yaws = sc
        for hand in range(2):
            for block, xy in targets:
                dist = float(np.linalg.norm(centroids[hand][:2] - xy))
                if best is None or dist < best["dist_m"]:
                    best = {"dist_m": dist, "frame": frame, "hand": hand, "block": block,
                            "xy": [float(centroids[hand][0]), float(centroids[hand][1])],
                            "hand_yaw_deg": round(float(hand_yaws[hand]), 2)}
    if best is None:
        return None
    if best["dist_m"] > max_shift_m:
        print(f"      Greifanker {best['dist_m'] * 100:.1f} cm von der Bild-Lage entfernt "
              f"(Grenze {max_shift_m * 100:.0f} cm) — Episode wird ausgelassen.", flush=True)
        return None
    return best


def yaw_to_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    """Gierwinkel um z in Grad → Quaternion (w, x, y, z) in Isaacs Reihenfolge."""
    half = np.radians(float(yaw_deg)) / 2.0
    return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def place_cubes(env, layout: list | None = None, anchor: dict | None = None,
                yaws: list | None = None
                ) -> tuple[list[list[float]], list[str], list[float]] | None:
    """Würfel dorthin setzen, wo sie im REALBILD lagen — oder die Episode auslassen.

    Einzige Quelle ist das Bild-Layout aus ``extract_block_layout.py`` (Farbblob im
    Realbild → Strahl auf die Würfelebene). Bis 2026-08-22 gab es zwei stille
    Rückfallebenen, und beide setzten den Würfel nachweislich falsch:

    * Der Greifpunkt aus ``scan.json`` ist das Minimum der Fingeröffnung über die ganze
      Episode. Weil die Hand beim Pick-and-Place vom Zugreifen bis zum Ablegen geschlossen
      bleibt, liegt dieses Minimum irgendwo auf dem Transportweg — 48 von 116 Griffen des
      Laufs vom 2026-08-17 jenseits von 60 % der Episode. Zudem ist die Fingeröffnung für
      diese Hand gar kein Greifdetektor: bei 101 von 116 Griffen blieb die engste Öffnung
      über 6 cm, bei 5 cm Würfelkante.
    * Die Zufallsplatzierung erfindet eine Lage, die mit den Realaktionen nichts zu tun hat.

    Beides erzeugt Trainingsbilder, auf denen der Arm an einem Würfel vorbeigreift, der dort
    nie lag — der teuerste Fehler, den ein Co-Training-Datensatz machen kann, weil er wie
    gültige Aufsicht aussieht. Der Layout-Lauf vom 2026-08-22 findet in 60 von 60 Episoden
    alle drei Würfel; ein Rückfall ist also auch praktisch nicht nötig.

    ``anchor`` (aus ``grasp_anchor``) verschiebt GENAU EINEN Würfel — den gegriffenen —
    auf den Kuppen-Schwerpunkt der Hand. Die übrigen bleiben auf ihrer Bild-Lage: nur
    einer wird angefasst, für die anderen ist das Bild die bessere Quelle.

    ``yaws`` sind die Gierwinkel aus ``layout.json`` (``cubes_yaw_deg``), in Grad und mod
    90°. ``None`` je Würfel heißt „nicht belastbar gemessen", nicht „liegt gerade" — dann
    bleibt es bei 0°. Vor dem 2026-08-25 gab es den Wert nicht und jeder Würfel stand
    achsparallel; im Realdatensatz liegen sie aber schräg zur Tischkante. Ein falsch
    gedrehter Würfel ist derselbe Fehler wie ein falsch platzierter, nur im Drehfreiheits-
    grad: die Realaktion greift dann eine Ecke statt einer Fläche (5,0 cm über die Fläche,
    7,1 cm über die Diagonale), und das Bild-Aktions-Paar lehrt eine Zuordnung, die es
    nicht gibt.

    Rückgabe ``None`` heißt: keine vollständige Lage bekannt, Episode überspringen.
    Nur x/y kommen aus Layout bzw. Anker, die Höhe ist immer die Tischauflage.
    """
    z = float(env.cfg.block_z_surface)
    origin = env.scene.env_origins[0].cpu().numpy()
    positions: list[np.ndarray] = []

    for i, _block in enumerate(env.blocks):
        if not (layout and i < len(layout) and layout[i]):
            print(f"      Würfel {i}: keine Lage im Bild-Layout — Episode wird ausgelassen.",
                  flush=True)
            return None
        x, y = float(layout[i][0]), float(layout[i][1])
        if not (0.15 <= x <= 0.70 and -0.35 <= y <= 0.35):
            print(f"      Würfel {i}: Layout-Punkt ({x:.2f}, {y:.2f}) außerhalb des Tischs "
                  f"— Episode wird ausgelassen.", flush=True)
            return None
        if anchor and anchor["block"] == i:
            x, y = float(anchor["xy"][0]), float(anchor["xy"][1])
        positions.append(np.array([x, y, z], dtype=np.float32))

    record, source, used_yaw = [], [], []
    for i, (block, pos) in enumerate(zip(env.blocks, positions)):
        record.append([round(float(v), 4) for v in pos])
        source.append("greifanker" if anchor and anchor["block"] == i else "layout")
        yaw = 0.0
        if yaws and i < len(yaws) and yaws[i] is not None:
            yaw = float(yaws[i])
        used_yaw.append(round(yaw, 2))
        quat = yaw_to_quat(yaw)
        world = torch.tensor([[float(pos[0] + origin[0]), float(pos[1] + origin[1]),
                               float(pos[2]), *quat]],
                             device=env.device, dtype=torch.float32)
        block.write_root_pose_to_sim(world)
        block.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))

    return record, source, used_yaw


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

def episode_window(layout_entry: dict | None, n_src: int) -> tuple[int, int, str]:
    """Welcher Frame-Bereich gerendert wird — und warum. Rückgabe ``(start, stop, Grund)``.

    Ohne ``--stop-at-grasp`` die ganze Episode. Mit dem Flag endet das Fenster, sobald sich
    der **erste** Würfel im Realvideo bewegt: ab da hat die reale Hand ihn mitgenommen,
    während der simulierte liegen bleibt, und das Bild beschreibt nicht mehr die Aktion. Den
    frühesten Würfel zu nehmen und nicht den spätesten ist Absicht — ein bewegter Würfel ist
    in beiden Kopfkameras zu sehen, also verdirbt er auch die Frames der anderen Hand.

    Die Grenze kommt aus ``layout.json`` (``motion_onset.first``, gemessen von
    ``extract_block_layout.episode_motion_onsets``). Bis 2026-08-22 stand hier das
    ``close_step`` aus ``scan.json``, das Minimum der Fingeröffnung. Das war zweimal falsch:
    der Detektor greift für die DEX3 nicht (101 von 116 Griffen schließen nie unter 6 cm bei
    5 cm Würfelkante), und wo er etwas fand, lag es zu spät — in Episode 0 achtundzwanzig
    Frames, also 21 % der Episode falsch beschriftet.

    ``start == stop`` heißt „diese Episode liefert kein brauchbares Fenster".
    """
    if not args.stop_at_grasp:
        return 0, n_src, "ganze Episode"
    onset = (layout_entry or {}).get("motion_onset", {}).get("first")
    if onset is None:
        return 0, 0, ("kein belastbarer Bewegungsbeginn im Layout "
                      "(beide Kopfkameras müssen sich einig sein)")
    stop = min(int(onset), n_src)
    start = max(0, stop - args.grasp_window) if args.grasp_window > 0 else 0
    if stop - start < args.min_window:
        return start, start, f"Fenster {stop - start} < {args.min_window} Frames"
    return start, stop, f"Frames {start}–{stop}, Würfel bewegt sich ab {stop}"


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
        env.set_dr_key(ep_idx)
        env.reset()
        stash_cubes(env)
        set_robot_to_state(env, state[0])
        for _ in range(args.settle_steps):
            env.step(torch.tensor(state[0], dtype=torch.float32,
                                  device=env.device).unsqueeze(0))

        print(f"[scan] ({k}/{len(episodes)}) Episode {ep_idx}: {actions.shape[0]} Frames",
              flush=True)
        _, arm_err, _, spread, centroid, _, _ = play_episode(
            env, actions, collect_images=False)
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
    elif not args.no_place_cubes and not args.layout:
        raise SystemExit(
            f"{scan_path} fehlt und kein --layout — ohne Layout weiß der Renderer weder, wo "
            "die Würfel lagen, noch bis zu welchem Frame das Bild zur Aktion passt. Erst "
            "extract_block_layout.py fahren; --no-place-cubes erzwingt die Entkopplung als "
            "Ablation."
        )

    # Bild-Layout: die einzige Quelle, die weiß, wo die Würfel lagen UND ab wann sie sich
    # bewegen. Ohne Eintrag verwirft der Renderer die Episode, statt zu raten.
    layout: dict = {}
    if args.layout:
        lp = Path(args.layout)
        if not lp.exists():
            raise SystemExit(f"--layout {lp} existiert nicht. Erst extract_block_layout.py fahren.")
        layout = json.loads(lp.read_text()).get("episodes", {})
        print(f"[render] Layout aus {lp}: {len(layout)} Episoden.", flush=True)
    else:
        print("[render] WARNUNG: kein --layout — nur mit --no-place-cubes sinnvoll "
              "(Ablation ohne Würfel).", flush=True)

    manifest_path = out / "render_manifest.json"
    manifest = {"episodes": {}, "frame_width": 640, "frame_height": 480,
                "source_dataset": str(src_root), "place_cubes": not args.no_place_cubes}
    if manifest_path.exists() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("episodes", {})

    fps = float(src_info.get("fps", 30.0))
    tasks = read_source_tasks(src_root)
    src_lengths = read_source_lengths(src_root)
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

        if not args.no_place_cubes and str(ep_idx) not in layout:
            # Ohne Layout gibt es seit 2026-08-22 keine Platzierung mehr — place_cubes
            # verwirft die Episode. Meist steht ein anderes RENDER_EPISODES dahinter als
            # beim Layout-Lauf.
            print(f"{head}: kein Layout-Eintrag — Episode wird verworfen. Mit denselben "
                  f"Episoden 'server_rl_run.sh layout' nachholen.", flush=True)
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

        start, stop, why = episode_window(layout.get(str(ep_idx)), actions.shape[0])
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
        env.set_dr_key(ep_idx)
        env.reset()
        cubes, cube_src, cube_yaw = None, None, None
        if not args.no_place_cubes:
            entry = layout.get(str(ep_idx), {})
            entry_cubes = entry.get("cubes")
            # Fehlt der Schlüssel, stammt das layout.json von vor dem 2026-08-25: dann gibt
            # es keine Gierwinkel und jeder Würfel steht achsparallel — wie bisher.
            entry_yaws = None if args.ignore_yaw else entry.get("cubes_yaw_deg")
            anchor = None
            if args.cube_source == "grasp" and entry_cubes:
                # Das Fenster endet per Konstruktion am Bewegungsbeginn (episode_window),
                # `state` ist bereits darauf geschnitten — sein letzter Frame IST der Onset.
                anchor = grasp_anchor(env, state, entry_cubes, len(state) - 1,
                                      float(args.max_anchor_shift))
                if anchor:
                    # Gegenprobe, absichtlich nur als Bericht: der Greifachsen-Winkel wird
                    # NICHT als Ersatz für einen verworfenen Bildwinkel eingesetzt. Er
                    # beruht auf der Annahme, dass quer zu einer Fläche gegriffen wurde,
                    # und die ist unbelegt. place_cubes hat aus genau diesem Grund seine
                    # stillen Rückfallebenen verloren; hier eine neue einzubauen wäre
                    # derselbe Fehler. Erst wenn beide Zahlen über viele Episoden
                    # zusammenfallen, ist die Annahme belegt.
                    img_yaw = (entry_yaws[anchor["block"]]
                               if entry_yaws and anchor["block"] < len(entry_yaws) else None)
                    if img_yaw is not None:
                        d = abs((float(img_yaw) - anchor["hand_yaw_deg"] + 45.0) % 90.0 - 45.0)
                        cmp_txt = (f", Bildwinkel {float(img_yaw):.0f}° vs. Greifachse "
                                   f"{anchor['hand_yaw_deg']:.0f}° (Δ {d:.0f}°)")
                    else:
                        cmp_txt = (f", Greifachse {anchor['hand_yaw_deg']:.0f}° "
                                   f"(kein Bildwinkel zum Vergleich)")
                    print(f"      Greifanker: Hand {'links' if anchor['hand'] == 0 else 'rechts'}"
                          f", Wuerfel {anchor['block']}, Frame {anchor['frame']}, "
                          f"{anchor['dist_m'] * 100:.1f} cm von der Bild-Lage{cmp_txt}.",
                          flush=True)
                env.reset()
            placement = place_cubes(env, layout=entry_cubes, anchor=anchor,
                                    yaws=entry_yaws)
            if placement is None:
                print(f"{head}: VERWORFEN — keine vollständige Würfellage im Bild-Layout. "
                      f"Fehlt sie für viele Episoden, zuerst 'server_rl_run.sh layout' "
                      f"mit denselben Episoden nachfahren.", flush=True)
                manifest["episodes"][str(ep_idx)] = {"status": "rejected_layout"}
                continue
            cubes, cube_src, cube_yaw = placement
        set_robot_to_state(env, state[0])
        for _ in range(args.settle_steps):
            env.step(torch.tensor(state[0], dtype=torch.float32,
                                  device=env.device).unsqueeze(0))

        print(f"{head}: {actions.shape[0]} Frames ({why}), Würfel {cubes} "
              f"aus {cube_src}", flush=True)
        writers = open_writers(videos, fps)
        try:
            (achieved, arm_err, hand_err, spread, centroid, block_trace,
             frame_hw) = play_episode(
                env, actions, collect_images=True, writers=writers,
                track_blocks=not args.no_place_cubes)
        finally:
            for w in writers.values():
                w.close()
        if frame_hw:
            manifest["frame_height"], manifest["frame_width"] = frame_hw

        # Handfehler AM GREIFMOMENT (letzte 20 Frames des Fensters, dort schliesst die Hand
        # um den Wuerfel). Trennt zwei Ursachen, die im Video gleich aussehen: folgen die
        # Finger ihren Sollwinkeln nicht, fehlt Kraft (stiffness/effort_limit) — folgen sie
        # und die Hand steht trotzdem seitlich am Wuerfel, stimmt die Handorientierung nicht.
        grip = hand_err[-min(20, len(hand_err)):]
        hand_err_end = [round(float(grip[:, h].mean()), 3) for h in range(2)]

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
            "hand_tracking_error_rad_end": hand_err_end,
            "cubes_xyz": cubes,
            "cube_source": cube_src,
            # Grundwahrheit für `extract_block_layout.py detect --expect-yaw`: gegen einen
            # Frame aus DIESEM Lauf gemessen, ist das die einzige echte Abnahme des
            # Winkelschätzers am Renderer.
            "cubes_yaw_deg": cube_yaw,
            "grasp_points": info.get("hands"),
            "consistency": cons,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        note = ""
        if cons:
            d = [h["dist_end_cm"] for h in cons["hands"] if h]
            if d:
                note = f", Kuppen↔Würfel am Fensterende {min(d):.1f} cm"
        print(f"{head}: geschrieben (Tracking Arm {arm_err:.3f} rad, Hand am Greifmoment "
              f"links {hand_err_end[0]:.3f} / rechts {hand_err_end[1]:.3f} rad{note}).",
              flush=True)

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
    cube_mode = ("keine (ABLATION)" if args.no_place_cubes
                 else "aus dem Bild-Layout (einzige Quelle; sonst wird die Episode verworfen)")
    print(f"  Würfel:      {cube_mode}")
    if args.no_dr:
        dr_txt = "AUS (--no-dr) — festes Licht, feste Farben"
    elif args.dr_seed < 0:
        dr_txt = "AN, OHNE Seed — zwei Läufe sind visuell NICHT vergleichbar"
    else:
        dr_txt = f"AN, Seed {args.dr_seed} je Episode — reproduzierbar"
    print(f"  Domain Rand.: {dr_txt}")
    if args.stop_at_grasp:
        win = (f"letzte {args.grasp_window} Frames vor der Würfelbewegung"
               if args.grasp_window else "Frame 0 bis zur ersten Würfelbewegung")
        print(f"  Fenster:     {win}, mind. {args.min_window} Frames "
              f"(Quelle: motion_onset aus layout.json; ab da wäre das Bild-Aktions-Paar "
              f"falsch beschriftet)")
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
