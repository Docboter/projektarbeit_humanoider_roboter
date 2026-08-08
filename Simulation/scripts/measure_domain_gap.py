"""
Domain-Gap-Quantifizierung: Real- (Teleop-) gegen Sim-Frames durch den eingefrorenen GR00T-ViT.

Da das BC-Fine-tuning mit tune_visual=false lief, sind die ViT-Gewichte identisch mit dem
pretrained-Checkpoint google/siglip-so400m-patch14-224. Der Domain-Gap laesst sich damit
messen, ohne das vollstaendige 6-GB-GR00T-Modell zu laden.

Aufruf im Sim-Container (Isaac-Sim-Kit-Python, dort liegt transformers):
  unset VIRTUAL_ENV
  /workspace/isaaclab/_isaac_sim/python.sh /workspace/measure_domain_gap.py \
      --real-dir /workspace/camera_reference --sim-dir /data/cam_dump

Bequemer vom Host aus:
  ./Simulation/server_rl_run.sh gap        # kopiert Skript + Referenzbilder rein, misst

Sim-Frames erzeugt `server_rl_run.sh cams` (dump_camera_poses.py) nach /data/cam_dump/.
Es werden mehrere Namensschemata akzeptiert, siehe SIM_PATTERNS.

WAS DIE ZAHL MISST — und was nicht:
  Die Kosinus-Distanz vergleicht komplette Bild-Embeddings. Sie enthaelt damit BEIDES:
  den Erscheinungs-Gap (Materialien, Beleuchtung, Rausch-/Schaerfe-Charakteristik) und
  den Inhalts-Gap (Armpose, Wuerfelplatzierung, Roboter-Tisch-Abstand). Die Referenz-
  frames zeigen den Roboter mitten in der Aufgabe, der Kamera-Dump die Reset-Pose — ein
  Teil der Distanz ist also Szeneninhalt, nicht Renderqualitaet. Die Real-gegen-Real-
  Grundlinie unten ordnet die Groessenordnung ein.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionModel

SIGLIP_MODEL = "google/siglip-so400m-patch14-224"

CAMERAS = ["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]

# Sim-Dateinamen je Erzeuger — in dieser Reihenfolge gesucht:
#   {cam}.png             dump_camera_poses.py  (server_rl_run.sh cams)  ← aktueller Weg
#   _debug_obs_{cam}.png  capture_camera_frames.sh / run_g1_dex3_sim_eval.py
#   sim_{cam}.png         dump_policy_cams.py
SIM_PATTERNS = ["{cam}.png", "_debug_obs_{cam}.png", "sim_{cam}.png"]
REAL_PATTERN = "dataset_{cam}.png"

# Referenzmessung vom 2026-06-04 unter Isaac Sim 4.x (docs/ergebnisse/domain-gap-analyse.md).
# Sie stammt aus der Zeit VOR dem Isaac-Sim-6.0-Port und vor der Kamera-Kalibrierung
# (Iteration 13/14) — die Deltas unten sagen also, was diese beiden Eingriffe bewirkt haben.
BASELINE_JUNE = {
    "cam_left_high": 0.1477,
    "cam_right_high": 0.2136,
    "cam_left_wrist": 0.4275,
    "cam_right_wrist": 0.2491,
    "_mean": 0.2595,
    "_real_real": 0.2726,
    "_sim_sim": 0.2111,
}


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - float(np.dot(a, b)))


def severity(dist: float) -> str:
    if dist < 0.10:
        return "minimal"
    if dist < 0.20:
        return "moderat"
    if dist < 0.35:
        return "gross"
    return "KRITISCH"


def find_frame(directory: Path, cam: str, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        candidate = directory / pattern.format(cam=cam)
        if candidate.exists():
            return candidate
    return None


def extract_embedding(model, processor, image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output[0].cpu().float().numpy()


def crossview_mean(embeddings: dict[str, np.ndarray]) -> float:
    """Mittlere Kosinus-Distanz zwischen allen Kamerapaaren derselben Domaene."""
    names = list(embeddings)
    dists = [
        cosine_distance(embeddings[names[i]], embeddings[names[j]])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    return float(np.mean(dists)) if dists else float("nan")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Domain-Gap real gegen sim messen")
    parser.add_argument(
        "--real-dir",
        default="/workspace/camera_reference"
        if Path("/workspace/camera_reference").is_dir()
        else str(script_dir.parent / "camera_reference"),
        help="Verzeichnis mit den Real-Referenzbildern (dataset_cam_*.png)",
    )
    parser.add_argument(
        "--sim-dir", default="/data/cam_dump",
        help="Verzeichnis mit den Sim-Frames (cam_*.png bzw. _debug_obs_cam_*.png)",
    )
    parser.add_argument("--out", default=None, help="Ziel der JSON-Ergebnisse")
    parser.add_argument("--model", default=SIGLIP_MODEL, help="Vision-Backbone")
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    sim_dir = Path(args.sim_dir)
    out_path = Path(args.out) if args.out else sim_dir / "domain_gap_results.json"

    print("=== Domain-Gap-Quantifizierung ===")
    print(f"Vision-Backbone: {args.model}")
    print("BC lief mit tune_visual=false -> ViT-Gewichte == pretrained SigLIP\n")
    print(f"Real-Dir: {real_dir}")
    print(f"Sim-Dir:  {sim_dir}\n")

    print(f"Lade {args.model} ...")
    processor = SiglipImageProcessor.from_pretrained(args.model)
    model = SiglipVisionModel.from_pretrained(args.model)
    model.eval()
    print("Modell geladen.\n")

    results: dict[str, float] = {}
    real_embeddings: dict[str, np.ndarray] = {}
    sim_embeddings: dict[str, np.ndarray] = {}

    print("Embeddings je Kamera ...")
    for cam in CAMERAS:
        real_path = find_frame(real_dir, cam, [REAL_PATTERN])
        sim_path = find_frame(sim_dir, cam, SIM_PATTERNS)
        if real_path is None:
            print(f"  UEBERSPRUNGEN {cam}: kein Real-Frame in {real_dir}")
            continue
        if sim_path is None:
            names = ", ".join(p.format(cam=cam) for p in SIM_PATTERNS)
            print(f"  UEBERSPRUNGEN {cam}: kein Sim-Frame in {sim_dir} (gesucht: {names})")
            continue

        real_embeddings[cam] = extract_embedding(model, processor, real_path)
        sim_embeddings[cam] = extract_embedding(model, processor, sim_path)
        results[cam] = cosine_distance(real_embeddings[cam], sim_embeddings[cam])
        print(f"  {cam:<20} {results[cam]:.4f}   <- {sim_path.name}")

    if not results:
        print("\nKein einziges Bildpaar gefunden. Pfade pruefen.")
        raise SystemExit(1)

    mean_dist = float(np.mean(list(results.values())))
    mean_rr = crossview_mean(real_embeddings)
    mean_ss = crossview_mean(sim_embeddings)

    # --- Tabelle mit Juni-Vergleich ------------------------------------------------
    print("\n" + "=" * 68)
    print(f"{'Kamera':<20} {'jetzt':>8} {'Juni':>8} {'Delta':>8}  Bewertung")
    print("-" * 68)
    for cam, dist in results.items():
        base = BASELINE_JUNE.get(cam)
        base_s = f"{base:.4f}" if base is not None else "     --"
        delta_s = f"{dist - base:+.4f}" if base is not None else "     --"
        print(f"  {cam:<18} {dist:>8.4f} {base_s:>8} {delta_s:>8}  {severity(dist)}")
    print("-" * 68)
    print(
        f"  {'MITTEL':<18} {mean_dist:>8.4f} {BASELINE_JUNE['_mean']:>8.4f} "
        f"{mean_dist - BASELINE_JUNE['_mean']:>+8.4f}  {severity(mean_dist)}"
    )

    print(f"\n  Grundlinie real-gegen-real (andere Kamera): {mean_rr:.4f} "
          f"(Juni {BASELINE_JUNE['_real_real']:.4f})")
    print(f"  Grundlinie sim-gegen-sim   (andere Kamera): {mean_ss:.4f} "
          f"(Juni {BASELINE_JUNE['_sim_sim']:.4f})")
    print("  (voellig unverwandte Bilder ~0.5-0.7, identisches Bild 0.0)")

    # --- Einordnung ----------------------------------------------------------------
    if mean_dist > 0.35:
        verdict = "KRITISCHER Domain-Gap"
        action = "tune_visual=true ODER Domain-Randomisierung noetig"
    elif mean_dist > 0.20:
        verdict = "grosser Domain-Gap"
        action = "ViT-Fine-tuning dringend empfohlen"
    elif mean_dist > 0.10:
        verdict = "moderater Domain-Gap"
        action = "ViT-Fine-tuning wahrscheinlich noetig, Zero-Shot grenzwertig"
    else:
        verdict = "geringer Domain-Gap"
        action = "Zero-Shot-Transfer real->sim plausibel"

    print("\n--- Einordnung ---")
    print(f"  {verdict} (Mittel {mean_dist:.4f})")
    print(f"  Empfehlung: {action}")
    if mean_rr > 0:
        print(f"  Real->Sim ist das {mean_dist / mean_rr:.1f}-fache des Real->Real-Gaps")
    print("  Achtung: die Zahl enthaelt auch den Szeneninhalt (Armpose, Wuerfel,")
    print("  Tischabstand), nicht nur die Renderqualitaet — siehe Modulkopf.")

    payload = {
        "model": args.model,
        "real_dir": str(real_dir),
        "sim_dir": str(sim_dir),
        "per_camera": results,
        "mean_real_sim": mean_dist,
        "mean_real_real_crossview": mean_rr,
        "mean_sim_sim_crossview": mean_ss,
        "baseline_20260604": BASELINE_JUNE,
        "verdict": verdict,
        "action": action,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nErgebnisse -> {out_path}")
    print("[gap] fertig.")


if __name__ == "__main__":
    main()
