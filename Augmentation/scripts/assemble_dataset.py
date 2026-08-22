#!/usr/bin/env python3
# TL;DR: Baut aus den Cosmos-Rohvideos einen COTRAIN_DATASET_PATH-kompatiblen LeRobot-v2.1-Datensatz.
"""assemble_dataset.py — laeuft im Container (venv-tools), aufgerufen von entrypoint.sh.

Kernannahme (siehe Plan/Risiko-Liste in docs/augmentation/anleitung.md): Cosmos restyled
dieselben Episoden real->real, ohne Physik-Replay — bei erhaltener Framezahl/FPS kann die
Original-observation.state/action-Parquet-Datei UNVERAENDERT uebernommen werden (nur
episode_index/index umnummeriert). AUGMENT_STRICT_FRAME_CHECK ist das Sicherheitsnetz dafuer.

Nur run_finetuning_cotrain.sh's TATSAECHLICH gepruefter Vertrag zaehlt: meta/info.json muss
existieren, meta/modality.json muss byte-identisch zum echten Datensatz sein (cmp -s). Beides
wird hier hart erfuellt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def ffprobe_frame_count(video: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return int(out)


def find_produced_video(raw_output_dir: Path) -> Path | None:
    # Cosmos' genaue Ausgabe-Dateibenennung unter -o <out_dir> ist bis zum ersten echten
    # Lauf nicht verifiziert (offener Punkt) — deshalb glob statt hartcodierter Name.
    candidates = sorted(raw_output_dir.rglob("*.mp4"))
    return candidates[0] if candidates else None


def reconcile_frame_count(produced: Path, target_frames: int, strict: bool, dest: Path) -> bool:
    """Kopiert `produced` nach `dest`, ggf. auf target_frames getrimmt. Gibt True zurueck,
    wenn das Ergebnis fuer den Datensatz brauchbar ist."""
    produced_frames = ffprobe_frame_count(produced)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if produced_frames == target_frames:
        shutil.copy2(produced, dest)
        return True

    if strict:
        return False

    if produced_frames > target_frames:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(produced),
                "-frames:v", str(target_frames), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(dest),
            ],
            check=True,
        )
        return True

    # Weniger Frames als das Original: kein sicherer Weg, Frames zu erfinden — als
    # nicht behebbar behandeln, auch bei AUGMENT_STRICT_FRAME_CHECK=0.
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--work-list", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--strict-frame-check", required=True, type=int)
    args = parser.parse_args()
    strict = bool(args.strict_frame_check)

    src_meta = args.dataset_dir / "meta"
    src_info = json.loads((src_meta / "info.json").read_text())
    src_episodes = {
        (row := json.loads(line))["episode_index"]: row
        for line in (src_meta / "episodes.jsonl").read_text().splitlines() if line.strip()
    }
    work_items = json.loads(args.work_list.read_text())

    out_meta = args.out_dir / "meta"
    out_data = args.out_dir / "data" / "chunk-000"
    out_meta.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    # meta/modality.json UNVERAENDERT uebernehmen — das ist der einzige Vertrag, den
    # run_finetuning_cotrain.sh tatsaechlich prueft (cmp -s).
    shutil.copy2(src_meta / "modality.json", out_meta / "modality.json")
    # tasks.jsonl als ganzes uebernehmen — unsere Episoden referenzieren dieselben
    # task_index-Werte wie ihre Quell-Episode (Actions/Labels werden unveraendert kopiert).
    shutil.copy2(src_meta / "tasks.jsonl", out_meta / "tasks.jsonl")

    # Work-Items nach (episode, variant) gruppieren — eine Ausgabe-Episode pro Gruppe,
    # nur wenn ALLE geforderten Kameras erfolgreich waren.
    groups: dict[tuple[int, int], list[dict]] = {}
    for item in work_items:
        groups.setdefault((item["episode"], item["variant"]), []).append(item)

    manifest = []
    new_idx = 0
    global_row_index = 0
    episodes_jsonl_lines = []
    total_frames = 0

    for (src_ep, variant), items in sorted(groups.items()):
        cam_status: dict[str, str] = {}
        cam_videos: dict[str, Path] = {}
        src_frames = None
        ok = True

        for item in items:
            cam = item["camera"]
            raw_dir = Path(item["raw_output_dir"])
            produced = find_produced_video(raw_dir)
            if produced is None:
                cam_status[cam] = "missing_output"
                ok = False
                continue

            if src_frames is None:
                src_frames = ffprobe_frame_count(Path(item["source_video"]))

            dest = out_data.parent.parent / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{new_idx:06d}.mp4"
            if reconcile_frame_count(produced, src_frames, strict, dest):
                cam_status[cam] = "ok"
                cam_videos[cam] = dest
            else:
                cam_status[cam] = "rejected_frame_mismatch"
                ok = False

        if not ok:
            manifest.append({
                "source_episode": src_ep, "variant": variant,
                "status": "missing_camera" if "missing_output" in cam_status.values() else "rejected_frame_mismatch",
                "cameras": cam_status,
            })
            # Bereits geschriebene Videos dieser (unvollstaendigen) Gruppe wieder entfernen.
            for v in cam_videos.values():
                v.unlink(missing_ok=True)
            continue

        # Quell-Parquet UNVERAENDERT kopieren (nur episode_index/index umnummeriert) —
        # observation.state/action/timestamp/task_index bleiben exakt wie im echten Datensatz.
        src_parquet = args.dataset_dir / "data" / "chunk-000" / f"episode_{src_ep:06d}.parquet"
        df = pd.read_parquet(src_parquet)
        n_rows = len(df)
        df["episode_index"] = new_idx
        df["index"] = range(global_row_index, global_row_index + n_rows)
        out_parquet = out_data / f"episode_{new_idx:06d}.parquet"
        df.to_parquet(out_parquet, index=False)

        src_ep_meta = src_episodes.get(src_ep, {})
        episodes_jsonl_lines.append(json.dumps({
            "episode_index": new_idx,
            "tasks": src_ep_meta.get("tasks", []),
            "length": n_rows,
        }))

        manifest.append({
            "source_episode": src_ep, "variant": variant, "new_episode_index": new_idx,
            "status": "ok", "cameras": cam_status,
        })

        global_row_index += n_rows
        total_frames += n_rows
        new_idx += 1

    (out_meta / "episodes.jsonl").write_text("\n".join(episodes_jsonl_lines) + ("\n" if episodes_jsonl_lines else ""))

    out_info = dict(src_info)
    out_info["total_episodes"] = new_idx
    out_info["total_frames"] = total_frames
    out_info["splits"] = {"train": f"0:{new_idx}"}
    (out_meta / "info.json").write_text(json.dumps(out_info, indent=2))

    manifest_path = args.out_dir / "augment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    n_ok = sum(1 for m in manifest if m["status"] == "ok")
    print(
        f"{n_ok}/{len(manifest)} Episoden-Varianten uebernommen, {total_frames} Frames gesamt. "
        f"Manifest: {manifest_path}"
    )
    if n_ok == 0:
        raise SystemExit("Keine einzige Episode hat den Frame-/Kamera-Check bestanden — abgebrochen.")


if __name__ == "__main__":
    main()
