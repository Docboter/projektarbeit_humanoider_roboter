#!/usr/bin/env python3
"""Reine Policy-Latenz: wie lange braucht GR00T für EINEN Action-Chunk?

Wozu
----
Die Sim-Eval misst 3,5-fache Zeitlupe, aber **94 % davon sind Kamera-Rendering**
(`run_g1_dex3_sim_eval.py` gibt die Aufteilung seit 2026-08-13 aus). Auf echter
Hardware gibt es dieses Rendering nicht — dort liefern die Kameras ihre Bilder
selbst. Übrig bleibt genau die Zahl, die dieses Skript misst: die Zeit vom
fertigen Observation-Dict bis zum Action-Chunk.

Das ist die Zahl, die über den Weg auf einen realen Roboter entscheidet, und sie
lässt sich aus dem Sim-Lauf nicht sauber ablesen: dort steckt zusätzlich der
ZMQ-Roundtrip mit ~3,7 MB unkomprimierten Bildern drin (4 × 640×480×3). Die
Differenz zwischen beiden Messungen IST der Transport-Overhead — deshalb misst
dieses Skript **in-process**, ohne Server und ohne Sim.

Maßstab
-------
Ein Aufruf liefert einen Chunk über 16 Schritte. Ausgeführt werden davon
`--execution-horizon` Schritte, das ergibt das Zeitbudget:

    budget_ms = 1000 * execution_horizon / policy_hz      (8 / 30 Hz = 267 ms)

Solange die Latenz darunter liegt, kann die Politik den Takt halten. Berichtet
wird bewusst auch das **Maximum**, nicht nur der Mittelwert: im Echtzeitbetrieb
ist ein einzelner Ausreißer über dem Budget eine Lücke in der Aktionsfolge, und
genau die fällt beim Zuschauen als Ruckeln auf.

Was hier NICHT gemessen wird: Kamera-Aufnahme und -Vorverarbeitung auf dem
echten Roboter, Netzwerk zum Roboter, und die Low-Level-Regelung. Die Balance
hängt ohnehin nicht an dieser Schleife — der Whole-Body-Controller läuft
entkoppelt und mit eigener, viel höherer Rate (docs/weiterfuehrend/
lokomotion-recherche.md §3.1).

Aufruf
------
Normalerweise über den Wrapper, der das Skript in den laufenden Container kopiert:

    HF_TOKEN=hf_... ./Simulation/server_rl_run.sh latency

Direkt im Container (GR00T-venv, NICHT das Isaac-Python):

    /app/Groot-1.6/.venv/bin/python /workspace/policy_latency.py \\
        --model-path /data/checkpoints/groot-g1dex3-checkpoint \\
        --iterations 50
"""

import argparse
import json
import statistics
import time

import numpy as np
import torch
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

# Gelenk-Aufteilung wie in Simulation/g1_dex3_sim/client.py build_obs(): dieselben
# Gruppen, dieselben Schlüssel. Weicht das ab, misst der Bench einen anderen
# Codepfad als der Sim-Client — und wäre wertlos.
STATE_GROUPS = (
    ("state.left_arm", 7),
    ("state.right_arm", 7),
    ("state.left_dex3", 7),
    ("state.right_dex3", 7),
)
CAMERAS = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist")


def build_synthetic_obs(width: int, height: int, task: str, rng) -> dict:
    """Observation im flat-key Format des Gr00tSimPolicyWrapper (wie client.build_obs).

    Rauschbilder statt echter Frames: für die LAUFZEIT ist der Bildinhalt
    bedeutungslos — der Vision-Encoder rechnet unabhängig davon dieselben
    Faltungen. Für die AKTIONEN wäre er entscheidend, aber die interessieren
    hier nicht (dafür gibt es finger_span_openloop.py auf echten Bildern).
    """
    obs = {}
    for cam in CAMERAS:
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        obs[f"video.{cam}"] = frame[np.newaxis, np.newaxis]        # (1,1,H,W,3)
    for key, dim in STATE_GROUPS:
        obs[key] = rng.standard_normal((1, 1, dim), dtype=np.float32) * 0.1
    obs["annotation.human.task_description"] = (task,)
    return obs


def main() -> None:
    p = argparse.ArgumentParser(description="Policy-Latenz von GR00T (ohne Sim, ohne ZMQ)")
    p.add_argument("--model-path", required=True, help="Checkpoint-Verzeichnis")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--iterations", type=int, default=50, help="Gemessene Aufrufe")
    p.add_argument("--warmup", type=int, default=5,
                   help="Ungemessene Aufrufe vorab. Der erste Forward zieht Kernel-Autotuning "
                        "und Speicher-Allokation nach sich und ist um ein Vielfaches langsamer "
                        "— er gehört nicht in den Mittelwert.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--task-description", default="stack the blocks")
    p.add_argument("--execution-horizon", type=int, default=8,
                   help="Wie viele Schritte eines Chunks ausgeführt werden (Budget-Bezug)")
    p.add_argument("--policy-hz", type=float, default=30.0)
    p.add_argument("--json-out", default="", help="Ergebnisse zusätzlich als JSON ablegen")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "(CPU)"
    print("=" * 70)
    print("GR00T — Policy-Latenz (in-process, ohne Sim und ohne ZMQ)")
    print("=" * 70)
    print(f"  Checkpoint:   {args.model_path}")
    print(f"  Gerät:        {device}  |  {gpu}")
    print(f"  torch:        {torch.__version__}")
    print(f"  Eingabe:      {len(CAMERAS)} Kameras à {args.width}×{args.height}")
    print(f"  Aufrufe:      {args.iterations} (+{args.warmup} Warmlauf)")
    print()

    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device=device,
    )
    # Derselbe Wrapper, den run_gr00t_server.py mit --use-sim-policy-wrapper benutzt.
    # Ohne ihn würde hier ein anderer Pfad gemessen als der, den die Sim tatsächlich ruft.
    wrapped = Gr00tSimPolicyWrapper(policy)

    rng = np.random.default_rng(0)
    obs = build_synthetic_obs(args.width, args.height, args.task_description, rng)

    action, _info = wrapped.get_action(obs)
    horizon = None
    for key, val in action.items():
        arr = np.asarray(val)
        if arr.ndim >= 1:
            horizon = int(arr.shape[0]) if horizon is None else horizon
            print(f"  Chunk:        '{key}' → {arr.shape}")
            break

    for _ in range(max(args.warmup - 1, 0)):
        wrapped.get_action(obs)
    if device == "cuda":
        torch.cuda.synchronize()

    # Jede Messung einzeln synchronisieren: CUDA-Aufrufe sind asynchron, ohne synchronize()
    # misst perf_counter nur, wie schnell die Arbeit in die Queue geschoben wurde.
    samples = []
    for i in range(args.iterations):
        t0 = time.perf_counter()
        wrapped.get_action(obs)
        if device == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
        if (i + 1) % 10 == 0:
            print(f"    … {i + 1}/{args.iterations}  (letzte {samples[-1]:.1f} ms)", flush=True)

    samples.sort()
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    p95 = samples[min(int(0.95 * len(samples)), len(samples) - 1)]
    budget_ms = 1000.0 * args.execution_horizon / args.policy_hz

    print()
    print("-" * 70)
    print(f"  Mittel   {mean:8.1f} ms      Median {median:7.1f} ms")
    print(f"  Minimum  {samples[0]:8.1f} ms      p95    {p95:7.1f} ms")
    print(f"  Maximum  {samples[-1]:8.1f} ms")
    print()
    print(f"  Maximale Regelrate:      {1000.0 / mean:.1f} Hz  (1 / Mittelwert)")
    print(f"  Budget je Chunk:         {budget_ms:.0f} ms "
          f"(execution_horizon {args.execution_horizon} / {args.policy_hz:.0f} Hz)")
    print(f"  Ausgelastet:             {100.0 * mean / budget_ms:.0f} % im Mittel, "
          f"{100.0 * samples[-1] / budget_ms:.0f} % im schlechtesten Aufruf")
    if samples[-1] > budget_ms:
        print("  !! Mindestens ein Aufruf lag ÜBER dem Budget — im Echtzeitbetrieb wäre das")
        print("     eine Lücke in der Aktionsfolge. Asynchrone Inferenz überdeckt so etwas.")
    print("-" * 70)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "model_path": args.model_path,
                    "device": device,
                    "gpu": gpu,
                    "torch": torch.__version__,
                    "cameras": len(CAMERAS),
                    "resolution": [args.width, args.height],
                    "iterations": args.iterations,
                    "action_horizon": horizon,
                    "latency_ms": {
                        "mean": round(mean, 2),
                        "median": round(median, 2),
                        "min": round(samples[0], 2),
                        "p95": round(p95, 2),
                        "max": round(samples[-1], 2),
                    },
                    "budget_ms": round(budget_ms, 2),
                    "max_rate_hz": round(1000.0 / mean, 2),
                },
                f,
                indent=2,
            )
        print(f"  JSON: {args.json_out}")

    # Erfolgsmarker wie in den anderen Diagnose-Skripten: der Wrapper greppt darauf,
    # weil ein Traceback im Container trotzdem mit Exit 0 enden kann.
    print("[latency] fertig.", flush=True)


if __name__ == "__main__":
    main()
