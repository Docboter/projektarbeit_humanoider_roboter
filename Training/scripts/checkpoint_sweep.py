#!/usr/bin/env python3
"""checkpoint_sweep.py — Open-Loop-Eval über ALLE Checkpoints eines Laufs.

Beantwortet die Frage, die Lauf 1 und Lauf 2 offen gelassen haben: **welcher
Checkpoint ist der beste?** Beide Läufe haben blind den letzten Step genommen,
weil es keine Validierungs-Zahl gab.

Was das Skript tut: für jeden ``checkpoint-*`` unterhalb von ``--run-dir`` wird
die Policy geladen und auf **zurückgehaltenen** Episoden open-loop ausgewertet
(MSE/MAE der vorhergesagten gegen die echten Aktionen). Ergebnis ist eine
Tabelle über die Steps, eine JSON-Datei und der Name des besten Checkpoints.

Warum ein eigenes Skript und nicht ``gr00t/eval/open_loop_eval.py``:

1. Jenes wertet **einen** Checkpoint aus — für eine Auswahl braucht es alle.
2. Jenes reicht **kein** ``split`` an den ``LeRobotEpisodeLoader`` durch. Dessen
   Default ist ``split="train"``. Nach einem Lauf mit ``TRAIN_TEST_SPLIT=1``
   sieht es damit ausschließlich **Trainings**-Episoden — die gemeldete MSE ist
   dann Trainings-MSE und kann Overfitting prinzipiell nicht zeigen.
3. ``loader[idx]`` indiziert in die **gefilterte** Episodenliste. Eine
   „traj_id 250" ist nach dem Filtern nicht Episode 250. Dieses Skript
   protokolliert deshalb immer den absoluten ``episode_index`` mit.

Die eigentliche Auswertung kommt unverändert aus dem Fork
(``evaluate_single_trajectory``) — es wird also dieselbe Metrik gerechnet, nur
über die richtigen Episoden und über alle Checkpoints.

Beispiel (im Container):

    python /scripts/checkpoint_sweep.py \\
        --run-dir /data/g1_dex3_finetune/blockstacking_vision \\
        --dataset-path /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset \\
        --num-trajectories 6

    python /scripts/checkpoint_sweep.py --run-dir ... --dry-run   # nur Plan zeigen
"""

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

GROOT_ROOT = os.environ.get("GROOT_ROOT", "/app/Groot-1.6")


# ── Argumente ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Open-Loop-Eval über alle Checkpoints eines Laufs (Checkpoint-Auswahl).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        default=os.environ.get("RUN_DIR", "/data/g1_dex3_finetune/blockstacking"),
        help="Verzeichnis des Laufs; wird rekursiv nach checkpoint-* durchsucht.",
    )
    p.add_argument(
        "--dataset-path",
        default=os.environ.get(
            "DATASET_PATH", "/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset"
        ),
        help="LeRobot-Datensatz (enthält meta/info.json mit dem Split).",
    )
    p.add_argument(
        "--split",
        default=os.environ.get("EVAL_SPLIT", "test"),
        help='Datensatz-Split. "test" = zurückgehaltene Episoden (das ist der Sinn der Übung).',
    )
    p.add_argument(
        "--num-trajectories",
        type=int,
        default=int(os.environ.get("EVAL_NUM_TRAJ", "6")),
        help="Anzahl gleichmäßig über den Split verteilter Episoden.",
    )
    p.add_argument(
        "--traj-positions",
        default=os.environ.get("EVAL_TRAJ_POSITIONS", ""),
        help="Optional: explizite Positionen IM SPLIT (kommagetrennt), überschreibt "
        "--num-trajectories.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("EVAL_STEPS", "300")),
        help="Maximale Steps je Episode (wird auf die Episodenlänge gedeckelt).",
    )
    p.add_argument(
        "--action-horizon",
        type=int,
        default=int(os.environ.get("EVAL_ACTION_HORIZON", "16")),
        help="Action-Horizon (muss zum Training passen).",
    )
    p.add_argument(
        "--embodiment-tag",
        default=os.environ.get("EMBODIMENT_TAG", "NEW_EMBODIMENT"),
        help="Embodiment-Tag des Checkpoints.",
    )
    p.add_argument(
        "--checkpoints",
        default=os.environ.get("EVAL_CHECKPOINTS", ""),
        help="Optional: explizite Step-Nummern (kommagetrennt), sonst alle gefundenen.",
    )
    p.add_argument(
        "--out",
        default=os.environ.get("EVAL_OUT", ""),
        help="JSON-Ausgabe. Leer = <run-dir>/checkpoint_sweep.json",
    )
    p.add_argument(
        "--plot-dir",
        default=os.environ.get("EVAL_PLOT_DIR", ""),
        help="Verzeichnis für die Trajektorien-Plots. Leer = <run-dir>/open_loop_plots",
    )
    p.add_argument(
        "--device",
        default=os.environ.get("EVAL_DEVICE", ""),
        help='Torch-Device, z. B. "cuda:1". Leer = cuda falls verfügbar, sonst cpu.',
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur zeigen, was ausgewertet würde (lädt weder Torch noch Checkpoints).",
    )
    return p.parse_args()


# ── Checkpoints finden ────────────────────────────────────────────────────────
def find_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Alle checkpoint-<step>-Verzeichnisse unterhalb von run_dir, nach Step sortiert.

    Rekursiv, weil die Checkpoints je nach Lauf entweder direkt im OUTPUT_DIR
    liegen oder unter <experiment>/checkpoints/<datum>/.
    """
    found: dict[int, Path] = {}
    for path in run_dir.rglob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match:
            continue
        step = int(match.group(1))
        # Bei Duplikaten (mehrere Läufe im selben Baum) gewinnt der zuletzt geänderte.
        if step in found and found[step].stat().st_mtime >= path.stat().st_mtime:
            continue
        found[step] = path
    return sorted(found.items())


def resolve_positions(args: argparse.Namespace, n_episodes: int) -> list[int]:
    """Positionen IM SPLIT — nicht zu verwechseln mit absoluten Episoden-Indizes."""
    if args.traj_positions.strip():
        positions = [int(x) for x in args.traj_positions.replace(",", " ").split()]
    else:
        count = max(1, min(args.num_trajectories, n_episodes))
        if count == 1:
            positions = [0]
        else:
            step = (n_episodes - 1) / (count - 1)
            positions = sorted({int(round(i * step)) for i in range(count)})
    out_of_range = [p for p in positions if not (0 <= p < n_episodes)]
    if out_of_range:
        sys.exit(
            f"FEHLER: Position(en) {out_of_range} liegen außerhalb des Splits "
            f"(0..{n_episodes - 1})."
        )
    return positions


def read_split_record(run_dir: Path) -> dict | None:
    """split.json, das run_finetuning*.sh beim Aktivieren des Splits ablegt."""
    for candidate in [run_dir / "split.json", *sorted(run_dir.rglob("split.json"))]:
        if candidate.is_file():
            try:
                with open(candidate) as f:
                    record = json.load(f)
                record["_path"] = str(candidate)
                return record
            except (OSError, json.JSONDecodeError):
                return None
    return None


# ── Hauptlauf ─────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"FEHLER: --run-dir existiert nicht: {run_dir}")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_dir():
        sys.exit(f"FEHLER: --dataset-path existiert nicht: {dataset_path}")

    checkpoints = find_checkpoints(run_dir)
    if args.checkpoints.strip():
        wanted = {int(x) for x in args.checkpoints.replace(",", " ").split()}
        checkpoints = [(s, p) for s, p in checkpoints if s in wanted]
    if not checkpoints:
        sys.exit(f"FEHLER: keine checkpoint-*-Verzeichnisse unter {run_dir} gefunden.")

    # Split gegen das Protokoll des Trainingslaufs prüfen.
    info_json = dataset_path / "meta" / "info.json"
    splits = {}
    if info_json.is_file():
        with open(info_json) as f:
            splits = json.load(f).get("splits", {})
    if args.split not in splits:
        print(
            f"!! WARNUNG: info.json kennt keinen Split '{args.split}' "
            f"(vorhanden: {list(splits) or 'keine'}).",
            flush=True,
        )
        print(
            "   Ohne test-Split wird gegen TRAININGS-Episoden gemessen — die Zahl zeigt dann",
            flush=True,
        )
        print(
            "   kein Overfitting. Trainingslauf mit TRAIN_TEST_SPLIT=1 wiederholen.",
            flush=True,
        )
    record = read_split_record(run_dir)
    if record and args.split in splits:
        recorded = record.get("splits", {}).get(args.split)
        if recorded and recorded != splits[args.split]:
            print(
                f"!! WARNUNG: info.json sagt {args.split}={splits[args.split]}, das Protokoll "
                f"des Laufs sagt {recorded}.",
                flush=True,
            )
            print(
                f"   info.json wurde seit dem Training geändert ({record['_path']}). "
                "Die Auswertung träfe andere Episoden als zurückgehalten wurden.",
                flush=True,
            )

    out_path = Path(args.out) if args.out else run_dir / "checkpoint_sweep.json"
    plot_dir = Path(args.plot_dir) if args.plot_dir else run_dir / "open_loop_plots"

    print("=" * 78, flush=True)
    print("Checkpoint-Sweep — Open-Loop-Eval auf zurückgehaltenen Episoden", flush=True)
    print("=" * 78, flush=True)
    print(f"  Lauf         : {run_dir}", flush=True)
    print(f"  Datensatz    : {dataset_path}", flush=True)
    split_range = splits.get(args.split, "NICHT DEFINIERT")
    step_list = ", ".join(str(s) for s, _ in checkpoints)
    print(f"  Split        : {args.split}  ({split_range})", flush=True)
    print(f"  Checkpoints  : {len(checkpoints)}  ({step_list})", flush=True)
    print(f"  Steps/Episode: {args.steps}   Action-Horizon: {args.action_horizon}", flush=True)
    print(f"  JSON         : {out_path}", flush=True)
    print(f"  Plots        : {plot_dir}", flush=True)
    print("", flush=True)

    # Jeder Checkpoint bedeutet, ein 3-Mrd.-Parameter-Modell frisch zu laden. Bei einem
    # Lauf mit SAVE_TOTAL_LIMIT=40 kommen schnell 38 Checkpoints zusammen — das sprengt
    # die Walltime, ohne dass es jemand vorher merkt.
    if len(checkpoints) > 12:
        print(
            f"!! ACHTUNG: {len(checkpoints)} Checkpoints × ~{args.num_trajectories} Episoden. "
            "Jeder Checkpoint wird",
            flush=True,
        )
        print(
            "   einzeln geladen (~1–2 min allein dafür). Für einen schnellen Durchlauf "
            "einschränken, z. B.",
            flush=True,
        )
        print("   --checkpoints 1000,50000,100000,175000   oder   EVAL_CHECKPOINTS=…", flush=True)
        print("", flush=True)

    if args.dry_run:
        print("Dry-Run — nichts geladen, nichts gerechnet.", flush=True)
        return 0

    # Erst hier importieren: --dry-run soll ohne Torch/GR00T laufen.
    if GROOT_ROOT not in sys.path:
        sys.path.insert(0, GROOT_ROOT)
    import numpy as np
    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.open_loop_eval import evaluate_single_trajectory
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Membername ("NEW_EMBODIMENT") und Wert ("new_embodiment") beide akzeptieren:
    # die Trainings-Skripte übergeben den Membername, EmbodimentTag(...) löst aber
    # über den Wert auf.
    try:
        embodiment_tag = EmbodimentTag[args.embodiment_tag]
    except KeyError:
        try:
            embodiment_tag = EmbodimentTag(args.embodiment_tag)
        except ValueError:
            valid = ", ".join(t.name for t in EmbodimentTag)
            sys.exit(
                f"FEHLER: Unbekannter Embodiment-Tag '{args.embodiment_tag}'. Gültig: {valid}"
            )
    plot_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    positions: list[int] = []
    episode_indices: list[int] = []

    for idx, (step, ckpt_path) in enumerate(checkpoints, start=1):
        print(f"── [{idx}/{len(checkpoints)}] checkpoint-{step} ────────────────", flush=True)
        policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=str(ckpt_path),
            device=device,
        )
        loader = LeRobotEpisodeLoader(
            dataset_path=str(dataset_path),
            modality_configs=policy.get_modality_config(),
            video_backend="torchcodec",
            video_backend_kwargs=None,
            split=args.split,
        )

        if not positions:
            n_episodes = len(loader)
            if n_episodes == 0:
                sys.exit(
                    f"FEHLER: Split '{args.split}' enthält keine Episoden — "
                    "wurde der Lauf ohne TRAIN_TEST_SPLIT=1 gefahren?"
                )
            positions = resolve_positions(args, n_episodes)
            episode_indices = [
                int(loader.episodes_metadata[p]["episode_index"]) for p in positions
            ]
            print(
                f"    Split '{args.split}': {n_episodes} Episoden, ausgewertet werden "
                f"{len(positions)}",
                flush=True,
            )
            print(f"    Position im Split : {positions}", flush=True)
            print(f"    Absoluter Episode-Index: {episode_indices}", flush=True)
            print("", flush=True)

        per_traj = []
        for position, episode_index in zip(positions, episode_indices):
            mse, mae = evaluate_single_trajectory(
                policy,
                loader,
                position,
                embodiment_tag,
                None,
                steps=args.steps,
                action_horizon=args.action_horizon,
                save_plot_path=str(plot_dir / f"ckpt{step}_ep{episode_index}.jpeg"),
            )
            per_traj.append(
                {
                    "split_position": position,
                    "episode_index": episode_index,
                    "mse": float(mse),
                    "mae": float(mae),
                }
            )
            print(
                f"    Episode {episode_index:>4}  MSE {float(mse):.6f}  MAE {float(mae):.6f}",
                flush=True,
            )

        mean_mse = float(np.mean([t["mse"] for t in per_traj]))
        mean_mae = float(np.mean([t["mae"] for t in per_traj]))
        results.append(
            {
                "step": step,
                "checkpoint_path": str(ckpt_path),
                "mean_mse": mean_mse,
                "mean_mae": mean_mae,
                "trajectories": per_traj,
            }
        )
        print(f"    → Mittel  MSE {mean_mse:.6f}  MAE {mean_mae:.6f}", flush=True)
        print("", flush=True)

        del loader
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Ergebnis ──────────────────────────────────────────────────────────────
    best = min(results, key=lambda r: r["mean_mse"])
    last = max(results, key=lambda r: r["step"])

    print("=" * 78, flush=True)
    print("Ergebnis — Validierungs-MSE über die Checkpoints", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Step':>8}  {'MSE':>12}  {'MAE':>12}", flush=True)
    for entry in results:
        marker = "  ← bester" if entry["step"] == best["step"] else ""
        print(
            f"{entry['step']:>8}  {entry['mean_mse']:>12.6f}  {entry['mean_mae']:>12.6f}{marker}",
            flush=True,
        )
    print("", flush=True)
    best_path = best["checkpoint_path"]
    best_line = f"checkpoint-{best['step']}  (MSE {best['mean_mse']:.6f})"
    print(f"Bester Checkpoint : {best_line}", flush=True)
    print(f"                    {best_path}", flush=True)

    if len(results) == 1:
        # Mit einem einzigen Checkpoint ist "bester == letzter" trivial wahr und sagt
        # über die Auswahl gar nichts — das darf nicht wie ein Ergebnis aussehen.
        print("", flush=True)
        print(
            "Nur EIN Checkpoint ausgewertet — das ist ein Funktionstest, keine Auswahl. "
            "Für eine Aussage",
            flush=True,
        )
        print(
            "über den besten Checkpoint braucht es mindestens zwei (--checkpoints / "
            "EVAL_CHECKPOINTS).",
            flush=True,
        )
    elif best["step"] != last["step"]:
        delta = (last["mean_mse"] - best["mean_mse"]) / best["mean_mse"] * 100.0
        print("", flush=True)
        print(
            f"Der letzte Checkpoint (checkpoint-{last['step']}, MSE {last['mean_mse']:.6f}) ist "
            f"um {delta:.1f} % schlechter",
            flush=True,
        )
        print(
            "als der beste. Genau diese Auswahl war in Lauf 1 und Lauf 2 blind.",
            flush=True,
        )
    else:
        print("", flush=True)
        print(
            "Der letzte Checkpoint ist zugleich der beste — hier hätte die blinde Auswahl "
            "zufällig gestimmt.",
            flush=True,
        )

    payload = {
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "split": args.split,
        "split_range": splits.get(args.split),
        "steps_per_trajectory": args.steps,
        "action_horizon": args.action_horizon,
        "split_positions": positions,
        "episode_indices": episode_indices,
        "checkpoints": results,
        "best_step": best["step"],
        "best_checkpoint_path": best["checkpoint_path"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print("", flush=True)
    print(f"JSON: {out_path}", flush=True)
    print(f"Plots: {plot_dir}", flush=True)
    print("[sweep] fertig.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
