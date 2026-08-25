#!/usr/bin/env python3
# TL;DR: Fasst Baseline- vs. optimierten Policy-Benchmark zu einem Speedup-Report zusammen.
"""Fuegt dem Policy-Benchmark den Closed-Loop-A/B-Durchsatz hinzu."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize_run(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rates = [float(ep.get("steps_per_s", 0.0)) for ep in data.get("episodes", [])]
    if not rates or any(rate <= 0 for rate in rates):
        raise RuntimeError(f"Keine gueltigen steps_per_s in {path}")
    time_share = {}
    for key in ("obs", "inference", "sim_render"):
        values = [float(ep.get("time_share", {}).get(key, 0.0)) for ep in data["episodes"]]
        time_share[key] = round(mean(values), 4)
    inference_mean = mean(
        [float(ep.get("inference_ms", {}).get("mean", 0.0)) for ep in data["episodes"]]
    )
    steps_per_s = mean(rates)
    policy_hz = float(data.get("policy_hz", 30.0))
    summary = {
        "backend": data.get("inference_backend"),
        "camera_render_every_n": data.get("camera_render_every_n"),
        "scene_camera_enabled": data.get("scene_camera_enabled"),
        "mean_inference_ms": round(inference_mean, 3),
        "mean_steps_per_s": round(steps_per_s, 3),
        "realtime_factor": round(steps_per_s / policy_hz, 3),
        "mean_time_share": time_share,
    }
    return data, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-speedup", type=float, default=2.0)
    args = parser.parse_args()

    _baseline, baseline_summary = summarize_run(Path(args.baseline))
    _optimized, optimized_summary = summarize_run(Path(args.optimized))
    base_sps = baseline_summary["mean_steps_per_s"]
    opt_sps = optimized_summary["mean_steps_per_s"]
    speedup = opt_sps / base_sps
    remaining_bottleneck = max(
        optimized_summary["mean_time_share"],
        key=optimized_summary["mean_time_share"].get,
    )
    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["closed_loop_benchmark"] = {
        "baseline": baseline_summary,
        "optimized": optimized_summary,
        "speedup": round(speedup, 3),
        "target_speedup": args.target_speedup,
        "target_reached": speedup >= args.target_speedup,
        "remaining_bottleneck": remaining_bottleneck,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    status = "ERREICHT" if speedup >= args.target_speedup else "NICHT ERREICHT"
    print(
        f"[optimize] Closed Loop: {base_sps:.2f} -> {opt_sps:.2f} Steps/s "
        f"({speedup:.2f}x; Ziel {args.target_speedup:.2f}x {status})"
    )
    print(f"[optimize] Gesamtbericht: {report_path}")


if __name__ == "__main__":
    main()
