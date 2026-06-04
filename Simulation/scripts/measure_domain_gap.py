"""
Domain-Gap Quantification: Real (teleop) vs. Sim (Isaac Sim) frames through frozen GR00T ViT.

Since tune_visual=false during GR00T fine-tuning, the ViT weights are identical to the
pretrained google/siglip-so400m-patch14-224 checkpoint. We can measure the domain gap
without downloading the full 6GB GR00T model.

Usage (inside lucam03/projekt-humanoider-roboter-sim-vastai container):
  /app/Groot-1.6/.venv/bin/python /workspace/scripts/measure_domain_gap.py

Mounts expected:
  /workspace/camera_reference/  — real dataset frames (dataset_cam_*.png)
  /workspace/sim_debug/         — Isaac Sim debug frames (_debug_obs_cam_*.png)
  /output/                      — (optional) JSON results output
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import SiglipImageProcessor, SiglipVisionModel

SIGLIP_MODEL = "google/siglip-so400m-patch14-224"

REAL_DIR = Path("/workspace/camera_reference")
SIM_DIR = Path("/workspace/sim_debug")

CAMERA_PAIRS = [
    ("dataset_cam_left_high.png",   "_debug_obs_cam_left_high.png",   "cam_left_high"),
    ("dataset_cam_right_high.png",  "_debug_obs_cam_right_high.png",  "cam_right_high"),
    ("dataset_cam_left_wrist.png",  "_debug_obs_cam_left_wrist.png",  "cam_left_wrist"),
    ("dataset_cam_right_wrist.png", "_debug_obs_cam_right_wrist.png", "cam_right_wrist"),
]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - float(np.dot(a, b)))


def extract_embedding(model, processor, image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output[0].cpu().float().numpy()


def main():
    print(f"=== Domain Gap Quantification ===")
    print(f"Vision backbone: {SIGLIP_MODEL}")
    print(f"Note: GR00T was trained with tune_visual=false → ViT weights == pretrained SigLIP\n")

    print(f"Loading {SIGLIP_MODEL} ...")
    processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL)
    model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL)
    model.eval()
    print("Model loaded.\n")

    results: dict[str, float] = {}
    real_embeddings: dict[str, np.ndarray] = {}
    sim_embeddings: dict[str, np.ndarray] = {}

    print("Extracting embeddings per camera ...")
    for real_file, sim_file, cam_name in CAMERA_PAIRS:
        real_path = REAL_DIR / real_file
        sim_path = SIM_DIR / sim_file

        if not real_path.exists():
            print(f"  SKIP {cam_name}: {real_path} not found")
            continue
        if not sim_path.exists():
            print(f"  SKIP {cam_name}: {sim_path} not found")
            continue

        real_emb = extract_embedding(model, processor, real_path)
        sim_emb = extract_embedding(model, processor, sim_path)
        real_embeddings[cam_name] = real_emb
        sim_embeddings[cam_name] = sim_emb

        dist = cosine_distance(real_emb, sim_emb)
        results[cam_name] = dist
        print(f"  {cam_name:<20}  cosine_dist(real, sim) = {dist:.4f}")

    if not results:
        print("\nNo image pairs found. Check mount paths.")
        sys.exit(1)

    # --- Summary table ---
    mean_dist = float(np.mean(list(results.values())))

    print("\n" + "=" * 60)
    print(f"{'Camera':<22} {'Dist':>8}  Severity")
    print("-" * 60)
    for cam, dist in results.items():
        if dist < 0.10:
            severity = "minimal"
        elif dist < 0.20:
            severity = "moderate"
        elif dist < 0.35:
            severity = "large"
        else:
            severity = "SEVERE"
        print(f"  {cam:<20} {dist:>8.4f}  {severity}")
    print("-" * 60)
    print(f"  {'MEAN':<20} {mean_dist:>8.4f}")

    # --- Baseline: real-vs-real cross-camera distances ---
    cams = list(real_embeddings.keys())
    rr_dists = []
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            d = cosine_distance(real_embeddings[cams[i]], real_embeddings[cams[j]])
            rr_dists.append(d)
    mean_rr = float(np.mean(rr_dists)) if rr_dists else float("nan")

    ss_dists = []
    sim_cams = list(sim_embeddings.keys())
    for i in range(len(sim_cams)):
        for j in range(i + 1, len(sim_cams)):
            d = cosine_distance(sim_embeddings[sim_cams[i]], sim_embeddings[sim_cams[j]])
            ss_dists.append(d)
    mean_ss = float(np.mean(ss_dists)) if ss_dists else float("nan")

    print(f"\n  Baseline: real-vs-real cross-camera  mean = {mean_rr:.4f}")
    print(f"  Baseline: sim-vs-sim cross-camera    mean = {mean_ss:.4f}")
    print(f"  (random unrelated images ≈ 0.5-0.7, same image ≈ 0.0)")

    # --- Conclusion ---
    print("\n--- Conclusion ---")
    if mean_dist > 0.35:
        verdict = "SEVERE domain gap"
        action = "tune_visual=true OR domain randomization REQUIRED"
    elif mean_dist > 0.20:
        verdict = "Large domain gap"
        action = "ViT fine-tuning strongly recommended"
    elif mean_dist > 0.10:
        verdict = "Moderate domain gap"
        action = "ViT fine-tuning likely needed; zero-shot transfer borderline"
    else:
        verdict = "Low domain gap"
        action = "Zero-shot real→sim transfer may be feasible"

    print(f"  {verdict} (mean cosine distance = {mean_dist:.4f})")
    print(f"  Recommendation: {action}")

    if mean_rr > 0 and mean_dist > mean_rr:
        ratio = mean_dist / mean_rr
        print(f"  Real→Sim gap is {ratio:.1f}x larger than Real→Real cross-view gap")

    # --- Save JSON ---
    out_dir = Path("/output") if Path("/output").exists() else Path("/tmp")
    out_path = out_dir / "domain_gap_results.json"
    payload = {
        "model": SIGLIP_MODEL,
        "per_camera": results,
        "mean_real_sim": mean_dist,
        "mean_real_real_crossview": mean_rr,
        "mean_sim_sim_crossview": mean_ss,
        "verdict": verdict,
        "action": action,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
