#!/usr/bin/env python3
# TL;DR: Container-Diagnose: misst reine Policy-Latenz pro Action-Chunk, ohne Sim und ohne Server.
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

Direkt im Container (GR00T-venv, NICHT das Isaac-Python). $GROOT_ROOT ist je nach
GROOT_VERSION /app/Groot-1.6 oder /app/Groot-1.7 — das Skript selbst ist versionsneutral:

    "$GROOT_ROOT/.venv/bin/python" /workspace/policy_latency.py \\
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


def resolve_embodiment_tag(name: str) -> EmbodimentTag:
    """String -> EmbodimentTag, in N1.6 wie in N1.7.

    N1.7 hat dafuer die Klassenmethode ``resolve`` (Name ODER Wert, case-insensitive);
    N1.6 kennt nur ``EmbodimentTag(wert)`` bzw. ``EmbodimentTag[NAME]``.
    """
    resolve = getattr(EmbodimentTag, "resolve", None)
    if resolve is not None:
        return resolve(name)
    try:
        return EmbodimentTag(name)
    except ValueError:
        return EmbodimentTag[name.upper()]

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
    # Zerlegung der Latenz. Der Flow-Matching-Kopf iteriert num_inference_timesteps-mal
    # (Default 4), der Backbone — Vision-Tower + LLM, der teure Teil — läuft davor GENAU
    # EINMAL (gr00t_n1d6.py bzw. gr00t_n1d7.py: die backbone_features werden vor der
    # Denoising-Schleife berechnet; in beiden Versionen gleich aufgebaut).
    # Misst man mehrere Werte, trennt die Steigung den iterativen Kopf vom festen Backbone:
    #   t(n) ≈ backbone + n * kopf     -> Steigung = Kopf je Schritt, Achsenabschnitt = Backbone
    # Das sagt, wo eine Optimierung für echte Hardware überhaupt ansetzen müsste.
    p.add_argument("--denoising-sweep", default="",
                   help="Kommagetrennte Werte für num_inference_timesteps, z. B. '1,2,4'. "
                        "Reine LATENZ-Diagnose — weniger Schritte heißt auch gröbere "
                        "Aktionen, das ist keine Empfehlung.")
    p.add_argument("--json-out", default="", help="Ergebnisse zusätzlich als JSON ablegen")
    # Fremdlast auf derselben Karte verfälscht die Messung massiv: dieselbe Policy kam am
    # 2026-08-13 einmal auf 78 ms (ruhige GPU) und einmal auf 108 ms mit Ausreißern bis
    # 188 ms, während nebenher gerendert wurde. Für eine belastbare Zahl gehört der Bench
    # auf eine freie Karte — deshalb hier wählbar.
    p.add_argument("--device", default="",
                   help="z. B. 'cuda:1', um auf eine freie Karte auszuweichen "
                        "(leer = cuda:0 bzw. CPU)")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(device) if device.startswith("cuda") else "(CPU)"
    print("=" * 70)
    print("GR00T — Policy-Latenz (in-process, ohne Sim und ohne ZMQ)")
    print("=" * 70)
    print(f"  Checkpoint:   {args.model_path}")
    print(f"  Gerät:        {device}  |  {gpu}")
    print(f"  torch:        {torch.__version__}")
    print(f"  Eingabe:      {len(CAMERAS)} Kameras à {args.width}×{args.height}")
    print(f"  Aufrufe:      {args.iterations} (+{args.warmup} Warmlauf)")
    # Belegung VOR dem Laden des Modells: was hier schon weg ist, gehört jemand anderem.
    # Eine mitlaufende Sim oder ein llama-server erklärt Ausreißer, die sonst wie
    # Modell-Jitter aussehen — und die Zahl wäre dann nicht die des Roboters.
    if device.startswith("cuda"):
        free_b, total_b = torch.cuda.mem_get_info(device)
        used_gb = (total_b - free_b) / 1024**3
        print(f"  VRAM vorher:  {used_gb:.1f} GB belegt von {total_b / 1024**3:.1f} GB")
        if used_gb > 1.0:
            print("                ^ von anderen Prozessen. Die Messung teilt sich die Karte")
            print("                  mit ihnen — fuer eine belastbare Zahl --device auf eine")
            print("                  freie Karte legen (z. B. --device cuda:1).")
    print()

    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=resolve_embodiment_tag(args.embodiment_tag),
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

    def timed_run(n_iter: int, progress: bool = True) -> list[float]:
        """n_iter Aufrufe messen, jeden einzeln synchronisiert.

        CUDA-Aufrufe sind asynchron — ohne synchronize() misst perf_counter nur, wie
        schnell die Arbeit in die Queue geschoben wurde, nicht wie lange sie dauert.
        """
        out = []
        for i in range(n_iter):
            t0 = time.perf_counter()
            wrapped.get_action(obs)
            if device == "cuda":
                torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1000.0)
            if progress and (i + 1) % 10 == 0:
                print(f"    … {i + 1}/{n_iter}  (letzte {out[-1]:.1f} ms)", flush=True)
        return out

    timed_run(max(args.warmup - 1, 0), progress=False)
    samples = timed_run(args.iterations)
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

    # ── Zerlegung: fester Backbone gegen iterativen Aktionskopf ────────────────
    sweep_rows = []
    if args.denoising_sweep:
        # Rekursiv suchen statt den Pfad zu raten: das Attribut sitzt auf dem Action-Head
        # (Gr00tN1d6ActionHead bzw. Gr00tN1d7ActionHead), also eine Ebene unter
        # policy.model — ein fest verdrahtetes policy.model.num_inference_timesteps lief
        # 2026-08-13 ins Leere. named_modules() findet es unabhängig davon, wie tief es
        # hängt, und funktioniert deshalb auch über Modellversionen hinweg.
        owners = [
            (name or "<root>", mod)
            for name, mod in getattr(policy, "model", None).named_modules()
            if hasattr(mod, "num_inference_timesteps")
        ] if getattr(policy, "model", None) is not None else []

        if not owners:
            print("  !! num_inference_timesteps nirgends im Modell gefunden — Sweep entfällt.")
        else:
            defaults = [(m, m.num_inference_timesteps) for _n, m in owners]
            default_n = defaults[0][1]
            print()
            print(f"  Zerlegung über num_inference_timesteps (Default {default_n}, gesetzt auf "
                  f"{', '.join(n for n, _m in owners)}):")
            try:
                for raw in args.denoising_sweep.split(","):
                    raw = raw.strip()
                    if not raw:
                        continue
                    n = int(raw)
                    for _name, mod in owners:
                        mod.num_inference_timesteps = n
                    timed_run(2, progress=False)                  # kurzer Warmlauf
                    s = sorted(timed_run(max(args.iterations // 2, 5), progress=False))
                    m = statistics.fmean(s)
                    sweep_rows.append({"denoising_steps": n, "mean_ms": round(m, 2)})
                    print(f"    {n:>2} Schritte → {m:7.1f} ms")
            finally:
                # Zustand IMMER zurücksetzen: das Policy-Objekt lebt weiter, und ein
                # verstellter Wert würde jede folgende Messung still verfälschen.
                for mod, val in defaults:
                    mod.num_inference_timesteps = val

            if len(sweep_rows) >= 2:
                lo, hi = sweep_rows[0], sweep_rows[-1]
                d_steps = hi["denoising_steps"] - lo["denoising_steps"]
                if d_steps:
                    per_step = (hi["mean_ms"] - lo["mean_ms"]) / d_steps
                    backbone = lo["mean_ms"] - per_step * lo["denoising_steps"]
                    print(f"    -> Aktionskopf ~{per_step:.1f} ms je Denoising-Schritt, "
                          f"fester Anteil (Vision+LLM+Transformation) ~{backbone:.1f} ms")
                    print("       Der feste Anteil ist die Untergrenze: er faellt an, egal "
                          "wie wenige Schritte der Kopf laeuft.")

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
                    "denoising_sweep": sweep_rows,
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
