#!/usr/bin/env python3
# TL;DR: Bringt einen fremden Sim-Datensatz (57-Dim-State, 50 fps) auf das Schema des echten
# G1-Dex3-Datensatzes (28 Dims, 30 fps) — Gelenk-Zuordnung wird belegt, nicht geraten.
"""harmonize_synth_dataset.py — fremden Sim-Datensatz co-training-fähig machen.

Warum es das gibt
─────────────────
``Fichtl00/Cube_Stacking_synth`` ist in der Sim teleoperiert worden — also genau das
Material, das dem Vision-Encoder seit Lauf 34 fehlt. Es passt aber nicht ins Schema des
echten Datensatzes:

    Feld                 echt                              synth
    observation.state    28  (left_arm/right_arm/…_dex3)   57  (robot_joint_pos[43] + eef)
    action               28  (…_dex3)                      28  (…_hand)   ← nur Namen
    fps                  30                                50
    annotation           human.task_description←task_index human.action.task_description
                                                           + human.validity

Drei davon sind reine Umbenennung, eines ist echte Arbeit: die 28 Arm- und Handgelenke
müssen aus den 43 Vollkörper-Gelenken herausgeschnitten werden — in **genau** der
Reihenfolge des echten Datensatzes. Diese Reihenfolge zu raten wäre ein stiller
Trainingsfehler der Sorte, die erst nach 30 Stunden Cluster-Zeit auffällt. Deshalb wird
sie hier gemessen:

  1. **Zuordnung aus dem Datensatz selbst.** Die Aktion ist ein Gelenk-Sollwert, der
     gemessene Zustand folgt ihm. Für das richtige Paar (i, j) ist darum
     ``rmse(action[:, i], robot_joint_pos[:, j])`` winzig, für jedes andere Paar groß.
     ``inspect`` rechnet die volle 28×43-Matrix, wählt je Aktionsdimension das Minimum
     und verlangt einen klaren Abstand zum Zweitbesten.
  2. **Gegenprobe gegen den echten Datensatz.** Schritt 1 vergleicht synth mit synth und
     könnte eine *systematisch* andere Aktionsreihenfolge nicht sehen — z. B. wenn der
     Generator beide Hände symmetrisch sortiert hat, der echte Datensatz rechts aber
     Index vor Middle führt. Deshalb wird zusätzlich der Wertebereich jeder Dimension
     gegen ``meta/stats.json`` des echten Datensatzes gehalten; passt eine synth-Achse
     besser zu einer *anderen* echten Achse, ist das ein Permutationsverdacht.

Ablauf
──────
    # 1. Prüfen und Zuordnung belegen (lädt nur meta/ + data/, ~4 MB, keine Videos)
    python Training/scripts/harmonize_synth_dataset.py inspect \\
        --source Fichtl00/Cube_Stacking_synth --work-dir data/cotrain_synth

    # 2. Umschreiben (lädt zusätzlich die Videos, ~150 MB)
    python Training/scripts/harmonize_synth_dataset.py convert \\
        --work-dir data/cotrain_synth --out data/cotrain_synth_v21 \\
        --modality-json app/Groot-1.6/examples/G1_DEX3/modality_4cam.json

    # 3. Ergebnis gegenprüfen
    python Training/scripts/harmonize_synth_dataset.py verify \\
        --dataset data/cotrain_synth_v21 \\
        --modality-json app/Groot-1.6/examples/G1_DEX3/modality_4cam.json

Ergebnis ist ein LeRobot-**v2.1**-Datensatz mit demselben Zuschnitt wie der echte, dessen
``meta/modality.json`` byteidentisch ist — genau das prüft ``run_finetuning_cotrain.sh``
vor dem Start (``cmp -s``).

Abhängigkeiten: numpy, pandas, pyarrow, huggingface_hub, imageio, imageio-ffmpeg.
    uv pip install numpy pandas pyarrow huggingface_hub imageio imageio-ffmpeg
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reihenfolge wie in modality_4cam.json — sie bestimmt die Spalten von info.json.
CAMS = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist")

# v2.1-Pfadschablonen (identisch zu render_cotrain_dataset.py)
DATA_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNK_SIZE = 1000

TARGET_FPS = 30.0
REFERENCE_REPO = "unitreerobotics/G1_Dex3_BlockStacking_Dataset"

# Die 28 Achsen des echten Datensatzes, in genau dieser Reihenfolge (aus dessen info.json).
# Wird nur für den Bericht gebraucht — der Loader liest sie nicht.
REAL_JOINT_NAMES = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
]


# ─────────────────────────────────────────────────────────────────────────────
# Ausgabe
# ─────────────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"\033[1;34m==>\033[0m {msg}")


def ok(msg: str) -> None:
    print(f"\033[1;32m v \033[0m{msg}")


def warn(msg: str) -> None:
    print(f"\033[1;33m ! \033[0m{msg}")


def die(msg: str) -> "NoReturn":  # noqa: F821
    print(f"\033[1;31m!! \033[0m{msg}", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Laden
# ─────────────────────────────────────────────────────────────────────────────
def fetch(source: str, work_dir: Path, patterns: list[str], token: str | None) -> Path:
    """HF-Repo nach work_dir spiegeln — oder einen lokalen Pfad unverändert nehmen."""
    local = Path(source)
    if local.is_dir():
        log(f"Quelle ist ein lokales Verzeichnis: {local}")
        return local

    from huggingface_hub import snapshot_download

    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"Lade {source} nach {work_dir}  (Muster: {', '.join(patterns)})")
    snapshot_download(
        repo_id=source,
        repo_type="dataset",
        local_dir=str(work_dir),
        allow_patterns=patterns,
        token=token,
    )
    return work_dir


def read_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def episode_indices(root: Path, info: dict) -> list[int]:
    """Episodenliste aus episodes.jsonl, ersatzweise aus total_episodes."""
    records = read_jsonl(root / "meta" / "episodes.jsonl")
    if records:
        return sorted(int(r["episode_index"]) for r in records)
    return list(range(int(info.get("total_episodes", 0))))


def parquet_path(root: Path, info: dict, ep: int) -> Path:
    template = info.get("data_path", DATA_TEMPLATE)
    return root / template.format(
        episode_chunk=ep // int(info.get("chunks_size", CHUNK_SIZE)), episode_index=ep
    )


def video_path(root: Path, info: dict, ep: int, cam: str) -> Path:
    template = info.get("video_path", VIDEO_TEMPLATE)
    return root / template.format(
        episode_chunk=ep // int(info.get("chunks_size", CHUNK_SIZE)),
        episode_index=ep,
        video_key=f"observation.images.{cam}",
    )


def stack_column(df: pd.DataFrame, column: str) -> np.ndarray:
    """Eine LeRobot-Listenspalte in ein (T, D)-Array holen."""
    if column not in df.columns:
        die(f"Spalte '{column}' fehlt im Parquet. Vorhanden: {list(df.columns)}")
    return np.stack(df[column].to_numpy()).astype(np.float64)


def load_episodes(root: Path, info: dict) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """(Episodenliste, State-Arrays, Action-Arrays) — alle Episoden, ungefiltert."""
    eps, states, actions = [], [], []
    for ep in episode_indices(root, info):
        path = parquet_path(root, info, ep)
        if not path.exists():
            warn(f"Episode {ep}: {path} fehlt — übersprungen.")
            continue
        df = pd.read_parquet(path)
        eps.append(ep)
        states.append(stack_column(df, "observation.state"))
        actions.append(stack_column(df, "action"))
    if not eps:
        die(f"Keine Episode gefunden unter {root}.")
    return eps, states, actions


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 1a — Zuordnung robot_joint_pos → Aktionsachsen
# ─────────────────────────────────────────────────────────────────────────────
def joint_block(modality: dict) -> tuple[int, int]:
    """Anfang/Ende des Vollkörper-Gelenkblocks im 57-Dim-State."""
    state = modality.get("state", {})
    for key in ("robot_joint_pos", "joint_pos", "robot_joint_positions"):
        if key in state:
            return int(state[key]["start"]), int(state[key]["end"])
    die(
        "In meta/modality.json des Quelldatensatzes gibt es keinen Gelenkblock "
        f"(gesucht: robot_joint_pos). Vorhandene State-Schlüssel: {list(state)}"
    )


def find_mapping(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    jstart: int,
    jend: int,
    max_lag: int,
    dims: list[int] | None = None,
) -> dict:
    """Je Aktionsachse das Gelenk mit dem kleinsten RMSE — über alle Episoden gepoolt.

    ``dims`` schränkt auf die Achsen ein, die überhaupt Gelenkwinkel *sind*. Das ist kein
    Detail: liegt auf den Armachsen eine kartesische EEF-Pose, zieht ihr Rauschen sonst die
    Zeitversatz-Suche in die Irre und belegt Gelenke, die den Händen gehören.

    Der Sollwert und der erreichte Zustand desselben Gelenks liegen in derselben Einheit
    dicht beieinander; jedes andere Gelenk ist um Größenordnungen weiter weg. Der
    Zeitversatz zwischen Kommando und Messung wird mitgesucht und für alle Achsen
    gemeinsam festgelegt — er ist eine Eigenschaft des Reglers, nicht der Achse.
    """
    n_act = actions[0].shape[1]
    n_joint = jend - jstart
    dims = list(range(n_act)) if dims is None else list(dims)

    def pooled(lag: int) -> tuple[np.ndarray, np.ndarray]:
        a_parts, s_parts = [], []
        for st, ac in zip(states, actions):
            t = min(st.shape[0], ac.shape[0])
            if t <= lag + 1:
                continue
            a_parts.append(ac[: t - lag] if lag else ac[:t])
            s_parts.append(st[lag:t, jstart:jend] if lag else st[:t, jstart:jend])
        return np.concatenate(a_parts), np.concatenate(s_parts)

    best_lag, best_score, best_rmse = 0, np.inf, None
    for lag in range(max_lag + 1):
        a_cat, s_cat = pooled(lag)
        # rmse[i, j] über alle Frames — (T,n_act,1) gegen (T,1,n_joint) wäre speicherhungrig,
        # deshalb spaltenweise.
        rmse = np.empty((n_act, n_joint))
        for j in range(n_joint):
            diff = a_cat - s_cat[:, j : j + 1]
            rmse[:, j] = np.sqrt(np.mean(diff * diff, axis=0))
        score = float(np.sum(np.min(rmse[dims], axis=1)))
        if score < best_score:
            best_lag, best_score, best_rmse = lag, score, rmse

    rmse = best_rmse
    a_cat, s_cat = pooled(best_lag)

    rows = []
    for i in dims:
        order = np.argsort(rmse[i])
        j = int(order[0])
        j2 = int(order[1]) if n_joint > 1 else j
        x, y = s_cat[:, j], a_cat[:, i]
        sx = float(np.std(x))
        if sx > 1e-9:
            slope = float(np.cov(x, y, bias=True)[0, 1] / (sx * sx))
            corr = float(np.corrcoef(x, y)[0, 1])
        else:
            slope, corr = float("nan"), float("nan")
        rows.append(
            {
                "action_dim": i,
                "action_name": REAL_JOINT_NAMES[i] if i < len(REAL_JOINT_NAMES) else f"dim{i}",
                "joint_index": j,
                "rmse": float(rmse[i, j]),
                "runner_up_joint": j2,
                "runner_up_rmse": float(rmse[i, j2]),
                "margin": float(rmse[i, j2] / rmse[i, j]) if rmse[i, j] > 1e-12 else float("inf"),
                "corr": corr,
                "slope": slope,
            }
        )
    return {"lag": best_lag, "rows": rows, "n_joint": n_joint, "joint_offset": jstart,
            "dims": dims}


def report_mapping(mapping: dict, max_rmse: float, min_margin: float) -> list[str]:
    """Tabelle drucken, Beanstandungen zurückgeben."""
    print()
    log(f"Gelenk-Zuordnung (Zeitversatz Kommando→Messung: {mapping['lag']} Frames)")
    print(f"    {'#':>2}  {'Achse':<20} {'→ joint':>7} {'RMSE':>9} {'2.-bester':>10} "
          f"{'Abstand':>8} {'corr':>6} {'Steigung':>9}")
    problems: list[str] = []
    seen: dict[int, int] = {}
    for row in mapping["rows"]:
        flag = " "
        if row["rmse"] > max_rmse:
            flag = "!"
            problems.append(
                f"Achse {row['action_dim']} ({row['action_name']}): RMSE {row['rmse']:.4f} "
                f"> Grenze {max_rmse}"
            )
        if row["margin"] < min_margin:
            flag = "!"
            problems.append(
                f"Achse {row['action_dim']} ({row['action_name']}): Abstand zum Zweitbesten "
                f"nur {row['margin']:.2f}× (< {min_margin}×) — Zuordnung nicht eindeutig"
            )
        if row["joint_index"] in seen:
            flag = "!"
            problems.append(
                f"Gelenk {row['joint_index']} doppelt vergeben: Achsen {seen[row['joint_index']]} "
                f"und {row['action_dim']}"
            )
        seen[row["joint_index"]] = row["action_dim"]
        print(
            f"  {flag} {row['action_dim']:>2}  {row['action_name']:<20} "
            f"{row['joint_index']:>7} {row['rmse']:>9.5f} {row['runner_up_rmse']:>10.5f} "
            f"{row['margin']:>7.1f}× {row['corr']:>6.3f} {row['slope']:>9.4f}"
        )
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 1a2 — Aktionsraum der Arme bestimmen
# ─────────────────────────────────────────────────────────────────────────────
# Verdachtsstellen für die beiden Arm-Quaternionen im 28er-Aktionsvektor, wenn die
# Armachsen eine EEF-Pose tragen: je 3 Positions- gefolgt von 4 Quaternionskomponenten.
ARM_QUAT_SLOTS = ((3, 7), (10, 14))


def detect_arm_action_space(actions: list[np.ndarray]) -> dict:
    """Sind die 14 Armachsen Gelenkwinkel oder je eine EEF-Pose (3 Position + 4 Quaternion)?

    Der Test braucht keine Ermessensschwelle: ein Einheitsquaternion hat die Norm 1, exakt,
    in jedem Frame. Vierzehn Gelenkwinkel tun das nicht. Fällt der Test auf ``eef_pose``,
    ist die Arm-Aktion mit dem echten Datensatz nicht vergleichbar und muss aus den
    gemessenen Armgelenken rekonstruiert werden (siehe ``reconstruct_arm_action``).
    """
    a = np.concatenate(actions)
    stats = []
    for qs, qe in ARM_QUAT_SLOTS:
        if a.shape[1] < qe:
            return {"mode": "joint", "quat_norms": []}
        n = np.linalg.norm(a[:, qs:qe], axis=1)
        stats.append({"slot": [qs, qe], "mean": float(n.mean()), "std": float(n.std()),
                      "max_abs_dev": float(np.abs(n - 1.0).max())})
    mode = "eef_pose" if all(st["max_abs_dev"] < 1e-3 for st in stats) else "joint"
    return {"mode": mode, "quat_norms": stats}


def _fk_design(joints: np.ndarray) -> np.ndarray:
    """Entwurfsmatrix einer FK-Näherung: [q, sin q, cos q, 1].

    Die echte Vorwärtskinematik ist ein Produkt von Rotationsmatrizen; deren Einträge sind
    Polynome in sin und cos der Gelenkwinkel. Diese Basis trifft den ersten Term davon —
    genug, um zu *unterscheiden*, welcher Gelenksatz welchen Endeffektor bewegt, und darum
    geht es hier. Ein exaktes FK-Modell bräuchte die URDF und wäre für diese Frage Aufwand
    ohne Ertrag.
    """
    return np.hstack([joints, np.sin(joints), np.cos(joints), np.ones((len(joints), 1))])


def find_arm_joints(
    states: list[np.ndarray],
    modality: dict,
    hand_joints: list[int],
    jstart: int,
    jend: int,
    move_threshold: float,
    max_frames: int = 4000,
) -> dict:
    """Die 14 Armgelenke finden — ohne Annahme über die Gelenkreihenfolge.

    Drei Schritte, jeder einzeln nachprüfbar:

    1. **Kandidaten** sind alle Gelenke, die nicht schon einer Hand gehören. Davon bleiben
       die *bewegten* übrig — in einer Tischaufgabe stehen Beine, Taille und Sprunggelenke
       still, die Arme nicht. Es müssen genau 14 sein, sonst Abbruch.
    2. **Seitenzuordnung** wird nicht geraten, sondern erschöpfend gesucht: alle
       C(14,7) = 3432 Aufteilungen, bewertet mit R²(linker Satz → ``left_eef_pos``) +
       R²(rechter Satz → ``right_eef_pos``). Ein Endeffektor ist eine Funktion *seiner*
       Kette und keiner anderen — das trennt sauber.
    3. **Reihenfolge innerhalb eines Arms** ist der aufsteigende Gelenkindex. In der
       Breitensuche über den Gelenkbaum (so ordnet Isaac Lab) wächst der Index mit der
       Baumtiefe, und die Tiefe ist genau die Kette Schulter-Pitch → -Roll → -Yaw →
       Ellenbogen → Handgelenk-Roll → -Pitch → -Yaw. Das ist dieselbe Reihenfolge, die der
       echte Datensatz führt.

    Der teure Teil ist billig gemacht: die Gram-Matrix aller 43 Merkmalsspalten wird EINMAL
    gerechnet, jede der 3432 Aufteilungen ist danach nur noch ein 22×22-Gleichungssystem.
    """
    from itertools import combinations

    state_mod = modality.get("state", {})

    def block(*names: str) -> slice | None:
        for n in names:
            if n in state_mod:
                return slice(int(state_mod[n]["start"]), int(state_mod[n]["end"]))
        return None

    l_eef, r_eef = block("left_eef_pos"), block("right_eef_pos")
    if l_eef is None or r_eef is None:
        die(
            "Für die Armsuche werden left_eef_pos und right_eef_pos im State gebraucht; "
            f"vorhanden sind: {list(state_mod)}"
        )

    sc = np.concatenate(states)
    if len(sc) > max_frames:
        sc = sc[:: max(1, len(sc) // max_frames)]
    joints = sc[:, jstart:jend]
    span = joints.max(axis=0) - joints.min(axis=0)

    hand_local = {j - jstart for j in hand_joints}
    candidates = [j for j in range(jend - jstart) if j not in hand_local]
    moving = [j for j in candidates if span[j] > move_threshold]
    frozen = [j for j in candidates if span[j] <= move_threshold]

    result = {
        "moving": [j + jstart for j in moving],
        "frozen": [j + jstart for j in frozen],
        "move_threshold": move_threshold,
        "span": {int(j + jstart): float(span[j]) for j in candidates},
    }
    if len(moving) != 14:
        result["problems"] = [
            f"{len(moving)} bewegte Kandidaten-Gelenke (Spanne > {move_threshold} rad), "
            "erwartet werden 14 Armgelenke. Schwelle mit --joint-move-threshold anpassen "
            "oder den Datensatz von Hand prüfen."
        ]
        return result

    # Gram-Matrix einmal über alle 14 Kandidaten: Spalten [q ×14 | sin ×14 | cos ×14 | 1]
    q = joints[:, moving]
    design = _fk_design(q)
    gram = design.T @ design
    targets = {"left": sc[:, l_eef], "right": sc[:, r_eef]}
    cross = {k: design.T @ v for k, v in targets.items()}
    tss = {k: float(((v - v.mean(axis=0)) ** 2).sum()) for k, v in targets.items()}
    yty = {k: float((v * v).sum()) for k, v in targets.items()}
    n_c = len(moving)
    bias = 3 * n_c  # Index der Einsen-Spalte

    def r2(local_idx: list[int], side: str) -> float:
        cols = [i for i in local_idx] + [i + n_c for i in local_idx] \
             + [i + 2 * n_c for i in local_idx] + [bias]
        g = gram[np.ix_(cols, cols)]
        b = cross[side][cols]
        try:
            beta = np.linalg.solve(g, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(g, b, rcond=None)[0]
        rss = yty[side] - float((beta * b).sum())
        return 1.0 - rss / tss[side]

    scored = []
    for left in combinations(range(n_c), 7):
        right = [i for i in range(n_c) if i not in left]
        rl, rr = r2(list(left), "left"), r2(right, "right")
        scored.append((rl + rr, list(left), right, rl, rr))
    scored.sort(key=lambda x: -x[0])
    best, second = scored[0], scored[1]

    left_joints = sorted(moving[i] + jstart for i in best[1])
    right_joints = sorted(moving[i] + jstart for i in best[2])
    # Aussagekräftiger als der Abstand zur zweitbesten Aufteilung (die sich meist nur um ein
    # kaum bewegtes Gelenk unterscheidet): der KONTRAST — wie viel schlechter erklärt der
    # jeweils andere Armsatz denselben Endeffektor? Genau das trennt links von rechts.
    cross_left = r2(best[2], "left")
    cross_right = r2(best[1], "right")
    result.update(
        {
            "left_joints": left_joints,
            "right_joints": right_joints,
            "r2_left": float(best[3]),
            "r2_right": float(best[4]),
            "r2_left_with_right_set": float(cross_left),
            "r2_right_with_left_set": float(cross_right),
            "contrast": float(min(best[3] - cross_left, best[4] - cross_right)),
            "score": float(best[0]),
            "runner_up_score": float(second[0]),
            "margin": float(best[0] - second[0]),
            "problems": [],
        }
    )
    return result


def report_arm_joints(arm: dict, min_r2: float, min_margin: float) -> list[str]:
    """Armsuche berichten, Beanstandungen zurückgeben."""
    print()
    log("Armgelenke — bewegte Kandidaten gegen blockierte")
    print(f"    blockiert (Spanne ≤ {arm['move_threshold']} rad): {arm['frozen']}")
    print(f"    bewegt:                                {arm['moving']}")
    problems = list(arm.get("problems", []))
    if "left_joints" not in arm:
        return problems

    print()
    log("Seitenzuordnung — beste von 3432 Aufteilungen, bewertet an der Vorwärtskinematik")
    print(f"    linker Arm  → {arm['left_joints']}   R²(→ left_eef_pos)  = {arm['r2_left']:.4f}")
    print(f"    rechter Arm → {arm['right_joints']}   R²(→ right_eef_pos) = {arm['r2_right']:.4f}")
    print(f"    Gegenprobe: derselbe Endeffektor mit dem ANDEREN Armsatz — "
          f"links {arm['r2_left_with_right_set']:.4f}, rechts {arm['r2_right_with_left_set']:.4f}")
    print(f"    Kontrast (eigener minus fremder Satz): {arm['contrast']:.4f}")
    print(f"    (Abstand zur zweitbesten Aufteilung {arm['margin']:.4f} — kein Tor: die "
          "zweitbeste\n     unterscheidet sich meist nur um ein kaum bewegtes Gelenk.)")
    print("    Reihenfolge je Arm = aufsteigender Index = Baumtiefe = Kette Schulter-Pitch,")
    print("    -Roll, -Yaw, Ellenbogen, Handgelenk-Roll, -Pitch, -Yaw (wie im echten Datensatz).")

    if arm["r2_left"] < min_r2 or arm["r2_right"] < min_r2:
        problems.append(
            f"FK-Bestimmtheitsmaß zu niedrig (links {arm['r2_left']:.3f}, rechts "
            f"{arm['r2_right']:.3f}, gefordert ≥ {min_r2}) — die Seitenzuordnung trägt nicht."
        )
    if arm["contrast"] < min_margin:
        problems.append(
            f"Kontrast zwischen eigenem und fremdem Armsatz nur {arm['contrast']:.4f} "
            f"(< {min_margin}) — die Seitenzuordnung trennt nicht."
        )
    return problems


def check_state_ranges(states: list[np.ndarray], idx28: list[int], stats: dict,
                       tolerance: float) -> list[str]:
    """Wertebereich der 28 gewählten Gelenke gegen ``observation.state`` des echten Satzes.

    Zweite, unabhängige Meinung zur Zuordnung: ein Ellenbogen, der im echten Datensatz
    zwischen −1,0 und +1,3 rad läuft, kann im synth-Satz nicht bei 0,00…0,01 stehen.
    """
    entry = stats.get("observation.state")
    if not entry:
        warn("stats.json enthält keinen 'observation.state'-Block — Bereichsprüfung entfällt.")
        return []
    rmin = np.asarray(entry["min"], dtype=float).ravel()
    rmax = np.asarray(entry["max"], dtype=float).ravel()
    if rmin.size != len(idx28):
        warn(f"Referenz hat {rmin.size} State-Dims, gewählt sind {len(idx28)} — Prüfung entfällt.")
        return []

    sc = np.concatenate(states)
    print()
    log("Bereichsprüfung: gewählte Gelenke gegen den echten Datensatz (observation.state)")
    print(f"    {'#':>2}  {'Achse':<20}{'→ j':>5}{'synth min…max':>24}{'echt min…max':>24}  drin")
    inside = 0
    for i, j in enumerate(idx28):
        a0, a1 = float(sc[:, j].min()), float(sc[:, j].max())
        hit = (a0 >= rmin[i] - tolerance) and (a1 <= rmax[i] + tolerance)
        inside += hit
        print(f"    {i:>2}  {REAL_JOINT_NAMES[i]:<20}{j:>5}{a0:>11.3f}…{a1:<12.3f}"
              f"{rmin[i]:>11.3f}…{rmax[i]:<12.3f}  {'ja' if hit else 'NEIN'}")
    print(f"    → {inside}/{len(idx28)} innerhalb (Toleranz ±{tolerance} rad)")
    if inside < len(idx28) - 4:
        return [
            f"Nur {inside}/{len(idx28)} Gelenkbereiche liegen im Bereich des echten "
            "Datensatzes — die Zuordnung ist vermutlich falsch."
        ]
    if inside < len(idx28):
        warn(f"{len(idx28) - inside} Achse(n) außerhalb — bei zwei Datensätzen, die dieselbe "
             "Aufgabe unterschiedlich ausführen, ist das normal. Eine falsche Zuordnung "
             "fällt mit zweistelligen Zahlen auf, nicht mit ein bis vier.")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 1a3 — Gelenknamen des Quelldatensatzes auswerten (falls beschafft)
# ─────────────────────────────────────────────────────────────────────────────
# Die 28 Achsen des echten Datensatzes als URDF-Gelenknamen, in genau seiner Reihenfolge.
# Beachte die Asymmetrie: links Middle vor Index, rechts Index vor Middle. Die steckt so im
# Datensatz (meta/info.json, names) und ist keine Nachlässigkeit hier.
REAL_URDF_NAMES = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    "left_hand_thumb_0", "left_hand_thumb_1", "left_hand_thumb_2",
    "left_hand_middle_0", "left_hand_middle_1", "left_hand_index_0", "left_hand_index_1",
    "right_hand_thumb_0", "right_hand_thumb_1", "right_hand_thumb_2",
    "right_hand_index_0", "right_hand_index_1", "right_hand_middle_0", "right_hand_middle_1",
]


def normalize_joint_name(name: str) -> str:
    """Gelenknamen auf eine vergleichbare Form bringen.

    Deckt die üblichen Schreibweisen ab: ``left_elbow_joint``, ``L_elbow``,
    ``kLeftElbow``, ``left_elbow_pitch_joint``. Bewusst großzügig — was übrig bleibt,
    meldet ``apply_joint_names`` namentlich, statt still danebenzugreifen.
    """
    n = name.strip().lower()
    for suffix in ("_joint", "_jnt", "joint"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    n = n.strip("_")
    # kLeftShoulderPitch → left_shoulder_pitch (CamelCase auftrennen)
    if "_" not in n and any(c.isupper() for c in name):
        out, prev_lower = [], False
        for ch in name.lstrip("k"):
            if ch.isupper() and prev_lower:
                out.append("_")
            out.append(ch.lower())
            prev_lower = ch.islower() or ch.isdigit()
        n = "".join(out)
    n = n.replace("__", "_")
    for a, b in (("l_", "left_"), ("r_", "right_")):
        if n.startswith(a):
            n = b + n[len(a):]
    # G1-Ellenbogen heißt je nach URDF-Fassung elbow oder elbow_pitch
    n = n.replace("elbow_pitch", "elbow").replace("elbow_joint", "elbow")
    return n


def read_joint_names(spec: str) -> list[str]:
    """Namensliste aus einer Datei (eine je Zeile oder JSON-Liste) oder als Kommaliste."""
    path = Path(spec)
    if path.exists():
        text = path.read_text().strip()
        if text.startswith("["):
            return [str(x) for x in json.loads(text)]
        return [ln.strip() for ln in text.splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
    return [x.strip() for x in spec.split(",") if x.strip()]


def apply_joint_names(names: list[str], jstart: int, jend: int) -> dict:
    """Aus der Namensliste des Quelldatensatzes die 28 Indizes in echter Achsenreihenfolge.

    Das ist die Antwort auf die Frage, die weder RMSE noch Vorwärtskinematik beantworten
    konnten (siehe docs/training/synth-datensatz.md § 3). Sie wird trotzdem nicht blind
    übernommen: der Aufrufer hält das Ergebnis gegen die selbst gemessene Armzuordnung und
    gegen die Wertebereiche des echten Datensatzes.
    """
    n_joint = jend - jstart
    if len(names) != n_joint:
        die(f"Namensliste hat {len(names)} Einträge, der Gelenkblock aber {n_joint}.")
    lookup: dict[str, int] = {}
    for i, raw in enumerate(names):
        key = normalize_joint_name(raw)
        if key in lookup:
            die(f"Gelenkname doppelt nach Normalisierung: '{raw}' → '{key}'")
        lookup[key] = jstart + i

    idx28, missing = [], []
    for want in REAL_URDF_NAMES:
        if want in lookup:
            idx28.append(lookup[want])
        else:
            missing.append(want)
    if missing:
        die(
            "In der Namensliste fehlen Achsen des echten Datensatzes:\n    "
            + ", ".join(missing)
            + "\n    Erkannt wurden: "
            + ", ".join(sorted(lookup))
        )
    return {"state_index_map": idx28, "normalized": lookup}


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 1b — Gegenprobe gegen den echten Datensatz
# ─────────────────────────────────────────────────────────────────────────────
def reference_stats(work_dir: Path, token: str | None, reference: str) -> dict | None:
    """meta/stats.json des echten Datensatzes holen (14 KB) — oder None."""
    local = Path(reference)
    if local.is_dir():
        path = local / "meta" / "stats.json"
        return read_json(path) if path.exists() else None
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=reference,
            repo_type="dataset",
            filename="meta/stats.json",
            token=token,
            local_dir=str(work_dir / "_reference"),
        )
        return read_json(Path(path))
    except Exception as exc:  # noqa: BLE001 — Gegenprobe ist optional, nicht tragend
        warn(f"meta/stats.json des echten Datensatzes nicht ladbar ({exc}) — Gegenprobe entfällt.")
        return None


def cross_check_axes(actions: list[np.ndarray], stats: dict) -> list[str]:
    """Wertebereich je Aktionsachse gegen den echten Datensatz halten.

    Sucht nach einer *Permutation*: Wenn synth-Achse i besser zu echt-Achse k≠i passt,
    hat der Generator anders sortiert. Das kann Schritt 1a nicht sehen, weil dort nur
    synth mit synth verglichen wird.
    """
    entry = stats.get("action")
    if not entry:
        warn("stats.json enthält keinen 'action'-Block — Gegenprobe entfällt.")
        return []
    real_mean = np.asarray(entry["mean"], dtype=float).ravel()
    real_std = np.asarray(entry["std"], dtype=float).ravel()
    real_min = np.asarray(entry["min"], dtype=float).ravel()
    real_max = np.asarray(entry["max"], dtype=float).ravel()

    a_cat = np.concatenate(actions)
    if a_cat.shape[1] != real_mean.size:
        warn(
            f"Aktionsbreite {a_cat.shape[1]} ≠ echt {real_mean.size} — Gegenprobe entfällt."
        )
        return []

    syn_mean, syn_std = a_cat.mean(axis=0), a_cat.std(axis=0)
    syn_min, syn_max = a_cat.min(axis=0), a_cat.max(axis=0)

    # Abstandsmaß zwischen synth-Achse i und echt-Achse k: normierter Abstand von
    # Mittelwert und Streuung. Skalenfrei genug, um Gelenke zu unterscheiden.
    scale = np.maximum(real_std, 1e-3)
    dist = (
        np.abs(syn_mean[:, None] - real_mean[None, :]) / scale[None, :]
        + np.abs(syn_std[:, None] - real_std[None, :]) / scale[None, :]
    )

    print()
    log("Gegenprobe: Wertebereich synth gegen echt (meta/stats.json)")
    print(f"    {'#':>2}  {'Achse':<20} {'synth min…max':>22} {'echt min…max':>22} "
          f"{'Δμ/σ':>7}  bester Treffer")
    problems: list[str] = []
    for i in range(a_cat.shape[1]):
        best = int(np.argmin(dist[i]))
        dmu = abs(syn_mean[i] - real_mean[i]) / scale[i]
        flag = " "
        note = "eigene Achse" if best == i else f"\033[1;33mAchse {best}\033[0m"
        if best != i and dist[i, best] < dist[i, i] * 0.5:
            flag = "!"
            problems.append(
                f"Achse {i} ({REAL_JOINT_NAMES[i]}): passt deutlich besser zu echt-Achse "
                f"{best} ({REAL_JOINT_NAMES[best]}) — Permutationsverdacht"
            )
        print(
            f"  {flag} {i:>2}  {REAL_JOINT_NAMES[i]:<20} "
            f"{syn_min[i]:>10.3f}…{syn_max[i]:<10.3f} "
            f"{real_min[i]:>10.3f}…{real_max[i]:<10.3f} {dmu:>7.2f}  {note}"
        )
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# inspect
# ─────────────────────────────────────────────────────────────────────────────
def cmd_inspect(args: argparse.Namespace) -> int:
    work = Path(args.work_dir)
    root = fetch(args.source, work, ["meta/*", "data/*", "README.md"], args.token)

    info = read_json(root / "meta" / "info.json")
    modality = read_json(root / "meta" / "modality.json")

    print()
    log("Quelldatensatz")
    print(f"    codebase_version : {info.get('codebase_version')}")
    print(f"    robot_type       : {info.get('robot_type')}")
    print(f"    Episoden/Frames  : {info.get('total_episodes')} / {info.get('total_frames')}")
    print(f"    fps              : {info.get('fps')}")
    print(f"    State-Schlüssel  : {list(modality.get('state', {}))}")
    print(f"    Action-Schlüssel : {list(modality.get('action', {}))}")
    print(f"    Video-Schlüssel  : {list(modality.get('video', {}))}")
    print(f"    Annotation       : {list(modality.get('annotation', {}))}")

    eps, states, actions = load_episodes(root, info)
    first = pd.read_parquet(parquet_path(root, info, eps[0]))
    print(f"    Parquet-Spalten  : {list(first.columns)}")
    print(f"    State-Breite     : {states[0].shape[1]}   Action-Breite: {actions[0].shape[1]}")

    cams_src = set(modality.get("video", {}))
    missing_cams = set(CAMS) - cams_src
    if missing_cams:
        die(f"Kameras fehlen im Quelldatensatz: {sorted(missing_cams)}")
    ok(f"Alle vier Policy-Kameras vorhanden: {', '.join(CAMS)}")

    jstart, jend = joint_block(modality)
    log(f"Gelenkblock im State: [{jstart}:{jend}]  ({jend - jstart} Gelenke)")

    # ── Aktionsraum der Arme ─────────────────────────────────────────────────
    space = detect_arm_action_space(actions)
    print()
    log("Aktionsraum der Armachsen")
    for st in space["quat_norms"]:
        print(f"    Norm von action[{st['slot'][0]}:{st['slot'][1]}]  mean={st['mean']:.6f}  "
              f"std={st['std']:.6f}  max|·−1|={st['max_abs_dev']:.2e}")
    if space["mode"] == "eef_pose":
        warn("Die Armachsen tragen eine EEF-POSE (3 Position + 4 Einheitsquaternion je Arm),")
        warn("keine Gelenkwinkel. Der echte Datensatz kommandiert Gelenke — die Arm-Aktion")
        warn("wird deshalb aus den GEMESSENEN Armgelenken bei t+lag rekonstruiert.")
        act_dims = list(range(14, 28))
    else:
        ok("Die Armachsen sind Gelenkwinkel — alle 28 Achsen werden direkt zugeordnet.")
        act_dims = list(range(28))

    # ── Handgelenke aus der Aktion zuordnen (dort ist sie Gelenkraum) ────────
    mapping = find_mapping(states, actions, jstart, jend, args.max_lag, dims=act_dims)
    problems = report_mapping(mapping, args.max_rmse, args.min_margin)
    joint_of = {r["action_dim"]: mapping["joint_offset"] + r["joint_index"]
                for r in mapping["rows"]}

    arm: dict = {}
    hand_idx = [joint_of[i] for i in sorted(joint_of)]
    measured_arm: list[int] = []
    if space["mode"] == "eef_pose":
        arm = find_arm_joints(states, modality, hand_idx, jstart, jend,
                              args.joint_move_threshold)
        problems += report_arm_joints(arm, args.min_fk_r2, args.min_fk_margin)
        measured_arm = arm.get("left_joints", []) + arm.get("right_joints", [])

    names_map: dict = {}
    if args.joint_names:
        names_map = apply_joint_names(read_joint_names(args.joint_names), jstart, jend)
        idx28 = names_map["state_index_map"]
        print()
        log("Gelenknamen des Quelldatensatzes ausgewertet — Abgleich mit den eigenen Messungen")
        if measured_arm:
            agree = idx28[:14] == measured_arm
            print(f"    Arme laut Namen : {idx28[:14]}")
            verdict = "✓ gleich" if agree else "✗ ABWEICHUNG"
            print(f"    Arme gemessen   : {measured_arm}   {verdict}")
            if not agree:
                problems.append(
                    "Die Armzuordnung aus den Gelenknamen weicht von der gemessenen ab "
                    f"(Namen {idx28[:14]}, FK-Messung {measured_arm}). Eine von beiden ist "
                    "falsch — das muss geklärt werden, bevor irgendetwas trainiert wird."
                )
        agree_h = sorted(idx28[14:]) == sorted(hand_idx)
        print(f"    Hände laut Namen: {idx28[14:]}")
        print(f"    Handblock gemessen (Reihenfolge offen): {sorted(hand_idx)}   "
              f"{'✓ gleiche Menge' if agree_h else '✗ ABWEICHUNG'}")
        if not agree_h:
            problems.append(
                f"Die Handgelenke aus den Namen {sorted(idx28[14:])} sind nicht dieselbe "
                f"Menge wie der gemessene Handblock {sorted(hand_idx)}."
            )
    elif space["mode"] == "eef_pose":
        idx28 = []
        problems.append(
            "Die Reihenfolge der 7 Gelenke INNERHALB jeder Hand ist unbestimmt. Der "
            "Handblock steht fest (links/rechts, siehe Tabelle), die Rollen darin nicht: "
            "die URDF-Grenzen lassen mehrere Dutzend Reihenfolgen zu, und weder die "
            "Korrelationsstruktur der Finger noch die Vorwärtskinematik trennen sie — "
            "das ist am ECHTEN Datensatz mit bekannter Wahrheit nachgemessen "
            "(docs/training/synth-datensatz.md § 3). Lösung: die Gelenknamen beim "
            "Ersteller erfragen und mit --joint-names übergeben."
        )
    else:
        idx28 = hand_idx

    stats = reference_stats(work, args.token, args.reference)
    if stats:
        if idx28:
            problems += check_state_ranges(states, idx28, stats, args.range_tolerance)
        if space["mode"] != "eef_pose":
            problems += cross_check_axes(actions, stats)

    if idx28 and len(idx28) != 28:
        problems.append(f"state_index_map hat {len(idx28)} Einträge statt 28.")

    mapping_file = work / "mapping.json"
    payload = {
        "source": args.source,
        "source_fps": float(info.get("fps", 0.0)),
        "target_fps": TARGET_FPS,
        "joint_offset": mapping["joint_offset"],
        "lag": mapping["lag"],
        "arm_action_mode": space["mode"],
        "quat_norms": space["quat_norms"],
        "arm_joint_indices": arm.get("left_joints", []) + arm.get("right_joints", []),
        "hand_joint_indices": [joint_of[i] for i in sorted(joint_of)],
        "arm_search": {k: v for k, v in arm.items() if k != "span"},
        "hand_block_left": hand_idx[:7],
        "hand_block_right": hand_idx[7:],
        "hand_order_resolved": bool(names_map),
        "joint_names_source": args.joint_names or None,
        "state_index_map": idx28,
        "action_key_rename": {"left_hand": "left_dex3", "right_hand": "right_dex3"},
        "rows": mapping["rows"],
        "problems": problems,
        "accepted": bool(idx28) and (not problems or bool(args.force)),
    }
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_file, "w") as fh:
        json.dump(payload, fh, indent=2)

    print()
    if problems:
        for p in problems:
            warn(p)
        print()
        if args.force:
            warn(f"--force gesetzt — {mapping_file} trotzdem als gültig markiert.")
        else:
            die(
                f"{len(problems)} Beanstandung(en). {mapping_file} geschrieben, aber NICHT "
                "als gültig markiert.\n    Erst klären, dann entweder das Mapping von Hand "
                "korrigieren (state_index_map) oder mit --force durchwinken."
            )
    ok(f"Zuordnung eindeutig und belegt. Geschrieben: {mapping_file}")
    if space["mode"] == "eef_pose":
        ok(f"Arm-Aktion wird rekonstruiert: robot_joint_pos[t+{mapping['lag']}] der Gelenke "
           f"{payload['arm_joint_indices']}")
        ok("Handaktion bleibt das echte Kommando aus action[14:28].")
    ok("Weiter mit:  harmonize_synth_dataset.py convert --work-dir "
       f"{args.work_dir} --out <ziel>")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# convert
# ─────────────────────────────────────────────────────────────────────────────
def count_video_frames(path: Path) -> int:
    import imageio_ffmpeg

    n, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    return int(n)


def resample_indices(n_src: int, src_fps: float, dst_fps: float) -> np.ndarray:
    """Quell-Frameindizes für ein Ziel-fps-Raster — zeittreu, nicht einfach jeder n-te.

    Ausgabe-Frame k liegt bei t = k/dst_fps, also beim Quell-Frame round(k·src_fps/dst_fps).
    Damit bleibt die Bewegung pro Schritt vergleichbar mit dem echten 30-fps-Datensatz —
    was zählt, weil ``delta_indices`` in Frames zählt, nicht in Sekunden.
    """
    if abs(src_fps - dst_fps) < 1e-6:
        return np.arange(n_src, dtype=int)
    n_out = int(round(n_src * dst_fps / src_fps))
    idx = np.round(np.arange(n_out) * src_fps / dst_fps).astype(int)
    idx = np.clip(idx, 0, n_src - 1)
    # Strikt aufsteigend erzwingen (bei Downsampling ohnehin gegeben, bei fps-Gleichheit
    # nicht erreichbar — dann greift der Kurzschluss oben).
    keep = np.concatenate([[True], np.diff(idx) > 0])
    return idx[keep]


def convert_video(src: Path, dst: Path, keep: np.ndarray, dst_fps: float) -> int:
    import imageio.v2 as imageio

    dst.parent.mkdir(parents=True, exist_ok=True)
    wanted = set(int(i) for i in keep)
    reader = imageio.get_reader(str(src), format="FFMPEG")
    writer = imageio.get_writer(
        str(dst), format="FFMPEG", mode="I", fps=dst_fps,
        codec="libx264", pixelformat="yuv420p", macro_block_size=8,
        ffmpeg_log_level="error",
    )
    written = 0
    try:
        for k, frame in enumerate(reader):
            if k in wanted:
                writer.append_data(frame)
                written += 1
    finally:
        writer.close()
        reader.close()
    return written


def cmd_convert(args: argparse.Namespace) -> int:
    work = Path(args.work_dir)
    mapping_file = work / "mapping.json"
    if not mapping_file.exists():
        die(f"{mapping_file} fehlt — erst 'inspect' fahren.")
    mapping = read_json(mapping_file)
    if not mapping.get("accepted") and not args.force:
        die(
            f"{mapping_file} ist nicht als gültig markiert (offene Beanstandungen aus "
            "'inspect'). Erst klären oder mit --force fortfahren."
        )

    source = args.source or mapping.get("source")
    root = fetch(source, work, ["meta/*", "data/*", "videos/*", "README.md"], args.token)
    info = read_json(root / "meta" / "info.json")
    src_fps = float(info.get("fps", mapping["source_fps"]))
    dst_fps = float(args.fps)

    modality_src = Path(args.modality_json)
    if not modality_src.exists():
        die(
            f"modality.json fehlt: {modality_src}\n"
            "    Erwartet wird die des ECHTEN Datensatzes bzw.\n"
            "    examples/G1_DEX3/modality_4cam.json.\n"
            "    Sie wird byteidentisch übernommen — genau das prüft run_finetuning_cotrain.sh."
        )

    idx_map = np.asarray(mapping["state_index_map"], dtype=int)
    if idx_map.size != 28:
        die(f"state_index_map hat {idx_map.size} Einträge, erwartet 28.")
    arm_mode = mapping.get("arm_action_mode", "joint")
    lag = int(mapping.get("lag", 0))
    if arm_mode == "eef_pose":
        log(f"Arm-Aktion wird rekonstruiert: robot_joint_pos[t+{lag}] der Gelenke "
            f"{idx_map[:14].tolist()}")
        log("Handaktion bleibt das echte Kommando aus action[14:28].")
        log(f"Je Episode entfallen dadurch die letzten {lag} Quellframes — für sie gibt es "
            "kein t+lag mehr.")
    else:
        log("Arm-Aktion wird unverändert aus action[0:14] übernommen (Gelenkraum).")

    out = Path(args.out)
    if out.exists() and args.overwrite:
        warn(f"--overwrite: lösche {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    src_tasks = {int(r["task_index"]): r["task"] for r in read_jsonl(root / "meta" / "tasks.jsonl")}
    task_text = args.task_text or src_tasks.get(0) or "stack the blocks"
    log(f"Sprach-Annotation für alle Episoden: \"{task_text}\"")

    eps = episode_indices(root, info)
    manifest, episodes_meta = [], []
    total_frames, index_offset = 0, 0
    height = width = None

    for ep in eps:
        pq = parquet_path(root, info, ep)
        if not pq.exists():
            warn(f"Episode {ep}: Parquet fehlt — übersprungen.")
            continue
        df = pd.read_parquet(pq)
        state57 = stack_column(df, "observation.state")
        action = stack_column(df, "action")

        srcs = {cam: video_path(root, info, ep, cam) for cam in CAMS}
        missing = [c for c, p in srcs.items() if not p.exists()]
        if missing:
            warn(f"Episode {ep}: Videos fehlen ({', '.join(missing)}) — übersprungen.")
            manifest.append({"episode_index": ep, "status": "missing_video", "cams": missing})
            continue

        counts = {cam: count_video_frames(p) for cam, p in srcs.items()}
        n_avail = min([len(df), state57.shape[0], action.shape[0]] + list(counts.values()))
        if n_avail < len(df):
            warn(
                f"Episode {ep}: Parquet {len(df)} Frames, kürzestes Video "
                f"{min(counts.values())} — auf {n_avail} gekürzt."
            )

        # Im EEF-Modus braucht jeder Ausgabeframe t noch den Quellframe t+lag: die letzten
        # lag Frames haben keinen mehr und fallen weg. Das MUSS vor dem Resampling
        # geschehen, sonst zeigen Bild und Aktion auf verschiedene Zeitpunkte.
        n_use = n_avail - lag if arm_mode == "eef_pose" else n_avail
        if n_use <= 0:
            warn(f"Episode {ep}: kürzer als der Zeitversatz ({n_avail} ≤ {lag}) — übersprungen.")
            manifest.append({"episode_index": ep, "status": "too_short", "frames": int(n_avail)})
            continue
        keep = resample_indices(n_use, src_fps, dst_fps)
        n_out = keep.size
        if n_out < args.min_frames:
            warn(f"Episode {ep}: nur {n_out} Frames nach Resampling — übersprungen.")
            manifest.append({"episode_index": ep, "status": "too_short", "frames": int(n_out)})
            continue

        written = {}
        for cam, src in srcs.items():
            out_info = {"video_path": VIDEO_TEMPLATE, "chunks_size": CHUNK_SIZE}
            dst = video_path(out, out_info, ep, cam)
            written[cam] = convert_video(src, dst, keep, dst_fps)
        bad = {c: n for c, n in written.items() if n != n_out}
        if bad:
            die(
                f"Episode {ep}: Video-Framezahl weicht ab {bad}, erwartet {n_out}. "
                "Bild und Aktion wären verschoben — Abbruch statt stiller Fehlausrichtung."
            )

        if height is None:
            import imageio.v2 as imageio

            probe = imageio.get_reader(str(next(iter(srcs.values()))), format="FFMPEG")
            frame = probe.get_next_data()
            height, width = int(frame.shape[0]), int(frame.shape[1])
            probe.close()

        state28 = state57[keep][:, idx_map].astype(np.float32)
        if arm_mode == "eef_pose":
            # Ersatz für das nicht aufgezeichnete Gelenk-Kommando: die tatsächlich
            # erreichte Armstellung lag Frames später. An den HÄNDEN gegen das echte
            # Kommando geprüft — dort liegen beide vor (siehe mapping.json, rows).
            arm_act = state57[keep + lag][:, idx_map[:14]]
            hand_act = action[keep][:, 14:28]
            action_out = np.concatenate([arm_act, hand_act], axis=1).astype(np.float32)
        else:
            action_out = action[keep].astype(np.float32)

        out_pq = parquet_path(out, {"data_path": DATA_TEMPLATE, "chunks_size": CHUNK_SIZE}, ep)
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "observation.state": list(state28),
                "action": list(action_out),
                # Der volle 57-Dim-Zustand bleibt zur Nachprüfung liegen — in info.json
                # nicht deklariert, also für Loader und Statistik unsichtbar.
                "observation.state_full": list(state57[keep].astype(np.float32)),
                "timestamp": np.arange(n_out, dtype=np.float32) / dst_fps,
                "frame_index": np.arange(n_out, dtype=np.int64),
                "episode_index": np.full(n_out, ep, dtype=np.int64),
                "index": np.arange(index_offset, index_offset + n_out, dtype=np.int64),
                "task_index": np.zeros(n_out, dtype=np.int64),
            }
        ).to_parquet(out_pq, index=False)

        index_offset += n_out
        total_frames += n_out
        episodes_meta.append({"episode_index": ep, "tasks": [task_text], "length": int(n_out)})
        manifest.append(
            {
                "episode_index": ep,
                "status": "ok",
                "frames_src": int(len(df)),
                "frames_out": int(n_out),
                "video_frames_src": {c: int(n) for c, n in counts.items()},
            }
        )
        ok(f"Episode {ep:>3}: {len(df)} → {n_out} Frames @ {dst_fps:g} fps")

    if not episodes_meta:
        die("Keine Episode konvertiert.")

    # ── meta/ ────────────────────────────────────────────────────────────────
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    with open(meta / "episodes.jsonl", "w") as fh:
        for rec in episodes_meta:
            fh.write(json.dumps(rec) + "\n")
    with open(meta / "tasks.jsonl", "w") as fh:
        fh.write(json.dumps({"task_index": 0, "task": task_text}) + "\n")

    # Byteidentisch übernehmen, nicht nachbauen: run_finetuning_cotrain.sh vergleicht mit cmp -s.
    shutil.copyfile(modality_src, meta / "modality.json")

    features = {
        "observation.state": {"dtype": "float32", "shape": [28], "fps": dst_fps},
        "action": {"dtype": "float32", "shape": [28], "fps": dst_fps},
        "timestamp": {"dtype": "float32", "shape": [1], "fps": dst_fps},
        "frame_index": {"dtype": "int64", "shape": [1], "fps": dst_fps},
        "episode_index": {"dtype": "int64", "shape": [1], "fps": dst_fps},
        "index": {"dtype": "int64", "shape": [1], "fps": dst_fps},
        "task_index": {"dtype": "int64", "shape": [1], "fps": dst_fps},
    }
    for cam in CAMS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [3, height, width],
            "info": {
                "video.fps": dst_fps,
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    highest = max(rec["episode_index"] for rec in episodes_meta) + 1
    with open(meta / "info.json", "w") as fh:
        json.dump(
            {
                "codebase_version": "v2.1",
                "robot_type": "Unitree_G1_sim_teleop",
                "total_episodes": len(episodes_meta),
                "total_frames": total_frames,
                "total_tasks": 1,
                "total_videos": len(episodes_meta) * len(CAMS),
                "total_chunks": 1,
                "chunks_size": CHUNK_SIZE,
                "fps": dst_fps,
                # Alle Episoden ins Training: 16 Episoden geben keine belastbare
                # Validierungszahl her, die kommt aus den zurückgehaltenen ECHTEN Episoden.
                "splits": {"train": f"0:{highest}"},
                "data_path": DATA_TEMPLATE,
                "video_path": VIDEO_TEMPLATE,
                "features": features,
            },
            fh,
            indent=4,
        )

    with open(out / "harmonize_report.json", "w") as fh:
        json.dump(
            {
                "source": source,
                "source_fps": src_fps,
                "target_fps": dst_fps,
                "state_index_map": idx_map.tolist(),
                "state_axis_names": REAL_JOINT_NAMES,
                "arm_action_mode": arm_mode,
                "arm_action_lag_frames": lag,
                "arm_joint_indices": idx_map[:14].tolist(),
                "hand_joint_indices": idx_map[14:].tolist(),
                "task_text": task_text,
                "modality_json_from": str(modality_src),
                "episodes": manifest,
            },
            fh,
            indent=2,
        )

    print()
    ok(f"Datensatz geschrieben: {out}")
    ok(f"    {len(episodes_meta)} Episoden, {total_frames} Frames @ {dst_fps:g} fps, "
       f"{width}×{height}")
    ok(f"    Bericht: {out / 'harmonize_report.json'}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi(token=args.token)
        api.create_repo(args.push_to_hub, repo_type="dataset", private=True, exist_ok=True)
        log(f"Lade nach https://huggingface.co/datasets/{args.push_to_hub} …")
        api.upload_folder(
            folder_path=str(out), repo_id=args.push_to_hub, repo_type="dataset"
        )
        ok(f"Hochgeladen: {args.push_to_hub}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# verify
# ─────────────────────────────────────────────────────────────────────────────
def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.dataset)
    info = read_json(root / "meta" / "info.json")
    problems: list[str] = []

    log(f"Prüfe {root}")
    print(f"    codebase_version : {info.get('codebase_version')}")
    print(f"    Episoden/Frames  : {info.get('total_episodes')} / {info.get('total_frames')}")
    print(f"    fps              : {info.get('fps')}")
    print(f"    splits           : {info.get('splits')}")

    if info.get("codebase_version") != "v2.1":
        problems.append(f"codebase_version ist {info.get('codebase_version')}, erwartet v2.1")
    if abs(float(info.get("fps", 0)) - TARGET_FPS) > 1e-6:
        problems.append(f"fps ist {info.get('fps')}, erwartet {TARGET_FPS}")
    for key in ("observation.state", "action"):
        shape = info.get("features", {}).get(key, {}).get("shape")
        if shape != [28]:
            problems.append(f"{key} hat shape {shape}, erwartet [28]")
    for cam in CAMS:
        if f"observation.images.{cam}" not in info.get("features", {}):
            problems.append(f"Kamera fehlt in info.json: {cam}")

    mod_ours = (root / "meta" / "modality.json").read_bytes()
    mod_ref = Path(args.modality_json).read_bytes()
    if mod_ours == mod_ref:
        ok(f"modality.json byteidentisch zu {args.modality_json}")
    else:
        problems.append(
            f"modality.json weicht von {args.modality_json} ab — run_finetuning_cotrain.sh "
            "bricht damit ab (cmp -s)."
        )

    eps = episode_indices(root, info)
    lengths = {int(r["episode_index"]): int(r["length"]) for r in
               read_jsonl(root / "meta" / "episodes.jsonl")}
    total = 0
    for ep in eps:
        pq = parquet_path(root, info, ep)
        if not pq.exists():
            problems.append(f"Episode {ep}: Parquet fehlt")
            continue
        df = pd.read_parquet(pq)
        n = len(df)
        total += n
        if lengths.get(ep) != n:
            problems.append(f"Episode {ep}: episodes.jsonl sagt {lengths.get(ep)}, Parquet hat {n}")
        st = stack_column(df, "observation.state")
        ac = stack_column(df, "action")
        if st.shape[1] != 28 or ac.shape[1] != 28:
            problems.append(
                f"Episode {ep}: State {st.shape[1]}, Action {ac.shape[1]} — erwartet 28"
            )
        if args.check_videos:
            for cam in CAMS:
                vp = video_path(root, info, ep, cam)
                if not vp.exists():
                    problems.append(f"Episode {ep}: Video fehlt ({cam})")
                    continue
                nv = count_video_frames(vp)
                if nv != n:
                    problems.append(
                        f"Episode {ep}/{cam}: {nv} Videoframes gegen {n} Parquetzeilen"
                    )
    if total != int(info.get("total_frames", -1)):
        problems.append(f"total_frames sagt {info.get('total_frames')}, gezählt {total}")

    print()
    if problems:
        for p in problems:
            warn(p)
        die(f"{len(problems)} Beanstandung(en).")
    ok(f"Alles konsistent: {len(eps)} Episoden, {total} Frames.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ins = sub.add_parser("inspect", help="Schema prüfen und Gelenk-Zuordnung belegen")
    ins.add_argument("--source", default="Fichtl00/Cube_Stacking_synth",
                     help="HF-Repo-ID oder lokales Verzeichnis")
    ins.add_argument("--work-dir", default="data/cotrain_synth")
    ins.add_argument("--reference", default=REFERENCE_REPO,
                     help="Echter Datensatz (HF-Repo oder lokaler Pfad) für die Gegenprobe")
    ins.add_argument("--token", default=None, help="HF-Token (sonst HF_TOKEN aus der Umgebung)")
    ins.add_argument("--max-lag", type=int, default=6,
                     help="Maximal gesuchter Versatz Kommando→Messung in Frames")
    ins.add_argument("--max-rmse", type=float, default=0.05,
                     help="Obergrenze für den RMSE des besten Treffers (rad)")
    ins.add_argument("--min-margin", type=float, default=3.0,
                     help="Geforderter Faktor zwischen zweitbestem und bestem RMSE")
    ins.add_argument("--joint-move-threshold", type=float, default=0.05,
                     help="Ab welcher Spanne (rad) ein Gelenk als bewegt gilt — trennt die "
                          "Arme von Beinen, Taille und Sprunggelenken")
    ins.add_argument("--min-fk-r2", type=float, default=0.95,
                     help="Geforderte Güte der FK-Näherung je Arm (Seitenzuordnung)")
    ins.add_argument("--min-fk-margin", type=float, default=0.20,
                     help="Geforderter Kontrast zwischen eigenem und fremdem Armsatz "
                          "(R² der Vorwärtskinematik) — das Tor der Seitenzuordnung")
    ins.add_argument("--joint-names", default=None,
                     help="Gelenknamen des Quelldatensatzes in der Reihenfolge von "
                          "robot_joint_pos — Datei (eine je Zeile oder JSON-Liste) oder "
                          "Kommaliste. Beantwortet die Handreihenfolge, die aus den Daten "
                          "allein nicht bestimmbar ist; wird gegen die eigenen Messungen "
                          "gegengeprüft, nicht blind übernommen.")
    ins.add_argument("--range-tolerance", type=float, default=0.15,
                     help="Toleranz (rad), um die ein synth-Gelenkbereich den echten "
                          "überschreiten darf")
    ins.add_argument("--force", action="store_true",
                     help="Beanstandungen nur melden, Mapping trotzdem als gültig markieren")
    ins.set_defaults(func=cmd_inspect)

    cv = sub.add_parser("convert", help="Datensatz auf das echte Schema umschreiben")
    cv.add_argument("--work-dir", default="data/cotrain_synth")
    cv.add_argument("--source", default=None, help="Überschreibt die Quelle aus mapping.json")
    cv.add_argument("--out", required=True, help="Zielverzeichnis des harmonisierten Datensatzes")
    cv.add_argument("--modality-json",
                    default="app/Groot-1.6/examples/G1_DEX3/modality_4cam.json",
                    help="Wird byteidentisch übernommen (die des ECHTEN Datensatzes)")
    cv.add_argument("--fps", type=float, default=TARGET_FPS)
    cv.add_argument("--task-text", default=None,
                    help="Sprachbefehl für alle Episoden; ohne Angabe der aus tasks.jsonl")
    cv.add_argument("--min-frames", type=int, default=30)
    cv.add_argument("--overwrite", action="store_true")
    cv.add_argument("--push-to-hub", default=None, help="HF-Repo-ID, privat angelegt")
    cv.add_argument("--token", default=None)
    cv.add_argument("--force", action="store_true")
    cv.set_defaults(func=cmd_convert)

    vf = sub.add_parser("verify", help="Erzeugten Datensatz gegenprüfen")
    vf.add_argument("--dataset", required=True)
    vf.add_argument("--modality-json",
                    default="app/Groot-1.6/examples/G1_DEX3/modality_4cam.json")
    vf.add_argument("--check-videos", action="store_true", default=True)
    vf.add_argument("--no-check-videos", dest="check_videos", action="store_false")
    vf.set_defaults(func=cmd_verify)

    args = p.parse_args()
    if getattr(args, "token", None) is None:
        import os

        args.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
