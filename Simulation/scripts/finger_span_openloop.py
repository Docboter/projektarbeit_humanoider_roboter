#!/usr/bin/env python3
# TL;DR: Container-Diagnose: prüft kommandierte Fingerspanne der Policy auf echten Dataset-Bildern.
"""Kommandierte Fingerspanne der BC-Policy auf ECHTEN Datensatz-Bildern.

Wozu
----
Lauf 30 (Closed Loop, Sim) hat gemessen: die Policy kommandiert **0,39 rad**
Fingerspanne, während die menschliche Demonstration **2,09 rad** enthält — rund
19 %. Die Hand greift also nicht, sie zuckt. Drei Erklärungen kamen dafür in Frage:

  (b) Die De-Normalisierung staucht den Ausgang.
      → WIDERLEGT durch check_action_norm.py: die Statistiken im Checkpoint
        stimmen auf 1e-5 mit dem Datensatz überein, die maximal ausgebbare
        Spanne ist exakt 2,0944 rad.
  (a1) Domain-Gap: die Policy kann greifen, aber die SIM-Bilder brechen sie.
  (a2) Die Policy hat den Griff nie gelernt (oder das Training ist schuld) —
        dann wäre sie auf echten Bildern genauso gestaucht.

Dieses Skript trennt (a1) von (a2), und das ist die teuerste offene Frage im
Projekt: bei (a1) ist ein TUNE_VISUAL=1-Lauf über ~48 GPU-Stunden begründet,
bei (a2) wäre er verschwendet.

Vorgehen
--------
Dieselbe Policy, dieselbe Metrik wie im Closed Loop (Spannweite je Fingergelenk
über die Episode, Maximum über die Gelenke), aber auf den **echten Bildern des
Test-Splits** statt auf Sim-Renderings. Als Bezugsgröße läuft dieselbe Rechnung
über die Ground-Truth-Aktionen derselben Episode mit — damit ist der Vergleich
nicht gegen eine notierte Zahl aus einem anderen Lauf, sondern gegen die
Demonstration, die das Modell an dieser Stelle nachahmen soll.

Lesart des Ergebnisses
----------------------
  Vorhersage ≈ Ground Truth   → Das Modell KANN greifen. Die Sim-Bilder brechen
                                es. Domain-Gap bestätigt, TUNE_VISUAL begründet.
  Vorhersage ≪ Ground Truth   → Das Modell greift auch auf echten Bildern nicht.
                                Dann ist es kein Wahrnehmungs-, sondern ein
                                Trainings-/Datenproblem — ein ViT-Lauf ginge an
                                der Ursache vorbei.

Voraussetzung: der ECHTE Datensatz **mit Videos** (nicht die Sim, und nicht der
reine meta/-Download). Braucht gr00t + torch, läuft also im Container, nicht auf
dem Host.

Aufruf
------
Normalerweise nicht direkt, sondern über den Wrapper — der kopiert das Skript in
den laufenden Container (``/scripts`` ist ins Image gebacken) und nutzt das
GR00T-venv statt des Isaac-Python:

    HF_TOKEN=hf_... ./Simulation/server_rl_run.sh span

Direkt im Container:

    /app/Groot-1.6/.venv/bin/python finger_span_openloop.py \\
        --model-path /data/checkpoints/groot-g1dex3-checkpoint \\
        --dataset-path /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset \\
        --traj-ids 0 1 2 3 4
"""

import argparse
import json
import logging
import os
import sys
from copy import deepcopy

import numpy as np
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy

# Welche Aktionsgruppen sind die Finger? Namen statt Indizes, damit eine geänderte
# Gruppen-Reihenfolge das Ergebnis nicht still verfälscht — genau diese Sorte
# stiller Fehlzuordnung hat in den Läufen 24-28 dreimal die Diagnose verdorben.
FINGER_GROUPS = ("left_dex3", "right_dex3")


def span_per_group(actions: np.ndarray, group_slices: dict[str, slice]) -> dict[str, float]:
    """Größte Spannweite (max-min über die Episode) je Gruppe, über deren Gelenke."""
    out = {}
    for name, sl in group_slices.items():
        block = actions[:, sl]
        out[name] = float((block.max(axis=0) - block.min(axis=0)).max()) if block.size else 0.0
    return out


def run_trajectory(policy, loader, traj_id, embodiment_tag, steps, action_horizon):
    """Gibt (pred, gt, group_slices) für eine Trajektorie zurück."""
    traj = loader[traj_id]
    actual_steps = min(steps, len(traj))
    action_keys = loader.modality_configs["action"].modality_keys

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    pred_rows, group_widths = [], None
    for step_count in range(0, actual_steps, action_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text

        _chunk, _ = policy.get_action(parse_observation_gr00t(obs, loader.modality_configs))
        chunk = parse_action_gr00t(_chunk)

        if group_widths is None:
            group_widths = {
                k: int(np.atleast_1d(np.atleast_1d(chunk[f"action.{k}"])[0]).shape[0])
                for k in action_keys
            }

        for j in range(action_horizon):
            pred_rows.append(
                np.concatenate(
                    [
                        np.atleast_1d(np.atleast_1d(chunk[f"action.{key}"])[j])
                        for key in action_keys
                    ],
                    axis=0,
                )
            )

    pred = np.array(pred_rows)[:actual_steps]

    # Ground Truth derselben Episode, in derselben Gruppen-Reihenfolge
    gt = np.concatenate(
        [np.vstack([a for a in traj[f"action.{key}"]]) for key in action_keys], axis=-1
    )[:actual_steps]

    # Gruppen-Slices aus den gemessenen Breiten aufbauen (nicht hartkodiert)
    group_slices, off = {}, 0
    for key in action_keys:
        group_slices[key] = slice(off, off + group_widths[key])
        off += group_widths[key]

    return pred, gt, group_slices


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="Checkpoint-Verzeichnis")
    p.add_argument("--dataset-path", required=True, help="ECHTER Datensatz (mit Videos)")
    p.add_argument("--traj-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--steps", type=int, default=400, help="Steps je Trajektorie")
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--json-out", default="", help="Ergebnis zusätzlich als JSON ablegen")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)

    import torch

    tag = EmbodimentTag(args.embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=tag,
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    print(f"[openloop] Datensatz: {len(loader)} Episoden | Checkpoint: {args.model_path}\n")

    rows = []
    for traj_id in args.traj_ids:
        if traj_id >= len(loader):
            print(f"[openloop] Trajektorie {traj_id} existiert nicht — übersprungen.")
            continue
        pred, gt, slices = run_trajectory(
            policy, loader, traj_id, tag, args.steps, args.action_horizon
        )
        s_pred, s_gt = span_per_group(pred, slices), span_per_group(gt, slices)
        finger_pred = max(s_pred[g] for g in FINGER_GROUPS if g in s_pred)
        finger_gt = max(s_gt[g] for g in FINGER_GROUPS if g in s_gt)
        rows.append(
            {
                "traj_id": traj_id,
                "steps": int(pred.shape[0]),
                "finger_span_pred_rad": round(finger_pred, 4),
                "finger_span_gt_rad": round(finger_gt, 4),
                "ratio": round(finger_pred / finger_gt, 4) if finger_gt > 1e-6 else None,
                "mse": round(float(np.mean((gt - pred) ** 2)), 6),
                "span_pred_per_group": {k: round(v, 4) for k, v in s_pred.items()},
                "span_gt_per_group": {k: round(v, 4) for k, v in s_gt.items()},
            }
        )
        if rows[-1]["ratio"] is None:
            # Ohne Fingerbewegung in der Demonstration gibt es nichts zu vergleichen —
            # die Episode zeigt an dieser Stelle keinen Griff.
            print(f"  Traj {traj_id:>3}: Ground Truth ohne Fingerbewegung — übersprungen")
        else:
            print(
                f"  Traj {traj_id:>3}: Fingerspanne Vorhersage {finger_pred:.2f} rad | "
                f"Ground Truth {finger_gt:.2f} rad | "
                f"{rows[-1]['ratio']:.0%} | MSE {rows[-1]['mse']:.4f}"
            )

    if not rows:
        print("[openloop] Keine Trajektorie ausgewertet.")
        return 1

    ratios = sorted(r["ratio"] for r in rows if r["ratio"] is not None)
    median = ratios[len(ratios) // 2] if ratios else 0.0
    print("\n" + "=" * 60)
    print(f"Median Vorhersage/Ground-Truth der Fingerspanne: {median:.0%}")
    print("Zum Vergleich Closed Loop in der Sim (Lauf 30):  19 %")
    print()
    # Die Auswertung steht hier im Klartext, damit die Zahl nicht später passend
    # gedeutet wird — die Schwellen sind vor dem Lauf festgelegt.
    if median >= 0.70:
        print("→ Die Policy greift auf ECHTEN Bildern. Die Sim-Bilder brechen sie.")
        print("  Domain-Gap bestätigt. TUNE_VISUAL=1 ist begründet.")
    elif median <= 0.35:
        print("→ Die Policy greift auch auf ECHTEN Bildern nicht — die Stauchung")
        print("  ist NICHT die Sim. Ein ViT-Lauf ginge an der Ursache vorbei;")
        print("  zu prüfen sind Training (Checkpoint, Schritte, Loss auf den")
        print("  Finger-Dims) und die Datenaufbereitung.")
    else:
        print("→ Zwischenbereich: teilweise Stauchung schon auf echten Bildern.")
        print("  Domain-Gap ist ein Faktor, aber nicht der einzige. Vor einem")
        print("  48-h-Lauf mehr Trajektorien messen (--traj-ids).")
    print("=" * 60)

    if args.json_out:
        # Der Checkpoint gehört IN die Datei, nicht nur ins Stdout: liegen mehrere unter
        # /data/checkpoints und wurde CHECKPOINT_PATH beim Aufruf vergessen, ist einer
        # gespeicherten Messung sonst nicht mehr anzusehen, welches Modell sie beschreibt.
        # run_id/global_step aus dem Checkpoint selbst, weil ein Pfad umbenannt sein kann.
        ckpt = {"path": args.model_path}
        for key, fname in (("run_id", "wandb_config.json"), ("global_step", "trainer_state.json")):
            try:
                with open(os.path.join(args.model_path, fname)) as fh:
                    ckpt[key] = json.load(fh).get(key)
            except Exception as e:
                ckpt[f"{key}_error"] = str(e)
        with open(args.json_out, "w") as fh:
            json.dump(
                {"checkpoint": ckpt, "episodes": rows, "median_ratio": median}, fh, indent=2
            )
        print(f"\nGespeichert: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
