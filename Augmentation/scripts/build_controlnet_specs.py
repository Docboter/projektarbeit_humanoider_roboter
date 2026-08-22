#!/usr/bin/env python3
# TL;DR: Erzeugt Edge-Control-Videos (Canny) + Cosmos-Transfer2.5-Spec-JSONs für die Trainings-Episoden.
"""build_controlnet_specs.py — laeuft im Container (venv-tools), aufgerufen von entrypoint.sh.

Waehlt Episoden aus den TRAIN-Split-Grenzen des echten Datensatzes (siehe select_episodes()),
erzeugt pro (Episode, Kamera, Variante) ein Edge-Control-Video (klassisches Canny, keine
zusaetzliche ML-Abhaengigkeit) und eine Cosmos-Transfer2.5-Spec-JSON gemaess dem in
docs/inference.md dokumentierten Schema. Schreibt zusaetzlich work_list.json, das
entrypoint.sh fuer die Inferenz-Schleife und assemble_dataset.py fuer den Datensatz-Bau nutzt.

WICHTIG — dupliziertes Split-Grenzformel-Kontrakt (kein gemeinsames Artefakt, siehe
Training/scripts/lib_split.sh und Simulation/g1_dex3_sim/render_cotrain_dataset.py):
n_train = int(total_episodes * AUGMENT_TRAIN_RATIO). AUGMENT_TRAIN_RATIO MUSS mit
TRAIN_SPLIT_RATIO im Training-Image uebereinstimmen — sonst koennen Test-Episoden hier
augmentiert und ueber COTRAIN_DATASET_PATH doch ins Training gelangen.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

# Kleine Rotationsliste fuer Domain-Randomization-Prompts, falls AUGMENT_VARIANTS > 1 und
# AUGMENT_PROMPT_TEMPLATE keinen eigenen {variant}-Platzhalter fuellt. Erster konkreter
# Formulierungs-Durchgang — siehe "Offene Punkte" in docs/augmentation/anleitung.md.
DEFAULT_STYLE_DESCRIPTORS = [
    "warm indoor lighting, wooden table, neutral gray background",
    "cool overcast daylight, matte gray table, plain white background",
    "soft evening lamp lighting, dark wood table, cluttered office background",
    "bright overhead fluorescent lighting, light blue table, brick wall background",
]

DEFAULT_PROMPT_TEMPLATE = (
    "A photorealistic video of a humanoid robot with two hands stacking colored "
    "blocks on a table, {style}. Keep the robot's shape, proportions and motion "
    "exactly as shown; only change lighting, background and surface texture."
)


def select_episodes(total_episodes: int, train_ratio: float, limit: int, explicit_ids: list[int]) -> list[int]:
    n_train = int(total_episodes * train_ratio)
    if explicit_ids:
        bad = [i for i in explicit_ids if i >= n_train]
        if bad:
            raise SystemExit(
                f"AUGMENT_EPISODE_IDS enthaelt Episoden >= n_train ({n_train}): {bad} — "
                "das waeren Test-Split-Episoden. Abgebrochen."
            )
        return sorted(set(explicit_ids))
    if limit <= 0 or limit >= n_train:
        return list(range(n_train))
    # Gleichmaessig ueber [0, n_train) verteilt statt chronologisches Praefix — dieselbe
    # Begruendung wie in render_cotrain_dataset.py::select_episodes.
    step = n_train / limit
    return sorted({int(i * step) for i in range(limit)})


def generate_edge_control(source_video: Path, out_video: Path) -> None:
    """Erzeugt ein Edge-Control-Video per Canny — ein Kanal, auf 3 Kanaele repliziert,
    anschliessend per ffmpeg nach h264/yuv420p re-encodiert (gleiche Codec-Konvention wie
    die Original-Videos im Datensatz)."""
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Konnte Quellvideo nicht oeffnen: {source_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                writer.write(edges_bgr)
        finally:
            writer.release()
            cap.release()

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(raw_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out_video),
            ],
            check=True,
        )


def build_spec(
    name: str,
    video_path: Path,
    control_path: Path,
    prompt_path: Path,
    num_steps: int,
) -> dict:
    return {
        "name": name,
        "prompt_path": str(prompt_path),
        "video_path": str(video_path),
        "guidance": 3,
        "num_steps": num_steps,
        "edge": {
            "control_path": str(control_path),
            "control_weight": 1.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--controls-dir", required=True, type=Path)
    parser.add_argument("--specs-dir", required=True, type=Path)
    parser.add_argument("--raw-output-dir", required=True, type=Path)
    parser.add_argument("--work-list", required=True, type=Path)
    args = parser.parse_args()

    train_ratio = float(os.environ.get("AUGMENT_TRAIN_RATIO", "0.8"))
    limit = int(os.environ.get("AUGMENT_EPISODE_LIMIT", "2"))
    explicit_ids_raw = os.environ.get("AUGMENT_EPISODE_IDS", "").strip()
    explicit_ids = [int(x) for x in explicit_ids_raw.split()] if explicit_ids_raw else []
    cameras = [c.strip() for c in os.environ.get(
        "AUGMENT_CAMERAS", "cam_left_high,cam_right_high,cam_left_wrist,cam_right_wrist"
    ).split(",") if c.strip()]
    variants = int(os.environ.get("AUGMENT_VARIANTS", "1"))
    num_steps = int(os.environ.get("AUGMENT_NUM_STEPS", "4"))
    prompt_template = os.environ.get("AUGMENT_PROMPT_TEMPLATE", "").strip() or DEFAULT_PROMPT_TEMPLATE

    info = json.loads((args.dataset_dir / "meta" / "info.json").read_text())
    total_episodes = int(info["total_episodes"])
    episode_ids = select_episodes(total_episodes, train_ratio, limit, explicit_ids)

    print(
        f"total_episodes={total_episodes} train_ratio={train_ratio} "
        f"n_train={int(total_episodes * train_ratio)} ausgewaehlt={len(episode_ids)} "
        f"episoden={episode_ids}",
        file=sys.stderr,
    )

    args.specs_dir.mkdir(parents=True, exist_ok=True)
    args.controls_dir.mkdir(parents=True, exist_ok=True)
    args.raw_output_dir.mkdir(parents=True, exist_ok=True)

    work_items = []
    for ep in episode_ids:
        for cam in cameras:
            source_video = (
                args.dataset_dir / "videos" / "chunk-000"
                / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
            )
            if not source_video.exists():
                print(f"WARNUNG: Quellvideo fehlt, uebersprungen: {source_video}", file=sys.stderr)
                continue

            control_path = args.controls_dir / "edge" / f"episode_{ep:06d}_{cam}.mp4"
            if not control_path.exists():
                generate_edge_control(source_video, control_path)

            for variant in range(variants):
                style = DEFAULT_STYLE_DESCRIPTORS[variant % len(DEFAULT_STYLE_DESCRIPTORS)]
                prompt_text = prompt_template.format(style=style) if "{style}" in prompt_template else prompt_template

                name = f"ep{ep:06d}_{cam}_v{variant}"
                prompt_path = args.specs_dir / f"{name}.prompt.txt"
                prompt_path.write_text(prompt_text)

                spec = build_spec(name, source_video, control_path, prompt_path, num_steps)
                spec_path = args.specs_dir / f"{name}.json"
                spec_path.write_text(json.dumps(spec, indent=2))

                raw_output_dir = args.raw_output_dir / name
                work_items.append({
                    "episode": ep,
                    "camera": cam,
                    "variant": variant,
                    "name": name,
                    "spec_path": str(spec_path),
                    "raw_output_dir": str(raw_output_dir),
                    "source_video": str(source_video),
                })

    args.work_list.write_text(json.dumps(work_items, indent=2))
    print(f"{len(work_items)} Arbeitseintraege geschrieben nach {args.work_list}", file=sys.stderr)


if __name__ == "__main__":
    main()
