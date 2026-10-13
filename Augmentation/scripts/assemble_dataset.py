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

Zwei Modi:
  * Normal: Rohvideos aller Work-Items -> Datensatz unter --out-dir.
  * --dry-validate <raw_dir>: Vorabpruefung eines kurzen Cosmos-Clips (entrypoint.sh,
    AUGMENT_PREFLIGHT=1), BEVOR der stundenlange Volllauf startet. Vergleicht Framezahl und
    fps des Clips mit der Quelle — genau das, woran die strikte Pruefung im Volllauf sonst erst
    am Ende scheitern wuerde.

meta/stats.json wird hier aus den AUSGABE-Parquets neu berechnet (dieselbe Formel wie
gr00t/data/stats.py::calculate_dataset_statistics), NICHT aus der Quelle kopiert: die
stats.json der Quelle umfasst alle Episoden inkl. Test-Split, und N1.6' check_stats_validity
wuerde sie ungeprueft akzeptieren. relative_stats.json erzeugt die DatasetFactory beim
Trainingsstart selbst (braucht die Embodiment-Konfiguration, die es hier nicht gibt).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


# Marker, an dem prepare_out_dir() erkennt, dass ein Ausgabeordner von diesem Skript stammt
# und gefahrlos neu aufgebaut werden darf.
OUT_DIR_MARKER = ".augment_dataset"
MANIFEST_NAME = "augment_manifest.json"
# Was prepare_out_dir() in einem eigenen Ausgabeordner vor dem Neuaufbau entfernt.
OWNED_ENTRIES = ("meta", "data", "videos", MANIFEST_NAME)

# Nebenprodukte, die cosmos-transfer2.5 (inference.py::_generate_sample) neben das eigentliche
# Ergebnis <name>.mp4 legt: <name>_control_<key>.mp4 und <name>_mask_<key>.mp4.
AUX_VIDEO_MARKERS = ("_control_", "_mask_")

FPS_TOLERANCE = 0.01

DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


@dataclass(frozen=True)
class VideoInfo:
    frames: int
    fps: float
    width: int
    height: int


def _parse_rate(rate: str) -> float:
    try:
        value = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(value) if value.denominator else 0.0


def ffprobe_video(video: Path) -> VideoInfo:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,avg_frame_rate,r_frame_rate,width,height",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    fps = _parse_rate(stream.get("avg_frame_rate", "0/0")) or _parse_rate(
        stream.get("r_frame_rate", "0/0")
    )
    return VideoInfo(
        frames=int(stream["nb_read_frames"]),
        fps=fps,
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
    )


def find_produced_video(raw_output_dir: Path, name: str | None = None) -> Path | None:
    """Das eigentliche Cosmos-Ergebnis in raw_output_dir finden.

    cosmos-transfer2.5 schreibt <name>.mp4, daneben <name>_control_edge.mp4 (und ggf.
    _mask_-Videos) und — NACH dem Ergebnisvideo — <name>.txt mit dem Prompt. Die .txt ist
    damit ein Fertig-Signal. Nebenprodukte und leere Dateien fallen raus; unter dem Rest gewinnt
    (1) ein Video mit .txt daneben (vollstaendig geschrieben), (2) der exakte Name <name>.mp4,
    (3) die juengste Datei. Frueher: sorted(rglob)[0], das bei einem Neulauf ein altes oder
    halb geschriebenes MP4 greifen konnte.
    """
    if not raw_output_dir.is_dir():
        return None
    candidates = []
    for path in raw_output_dir.rglob("*.mp4"):
        if any(marker in path.stem for marker in AUX_VIDEO_MARKERS):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size == 0:
            continue
        complete = path.with_suffix(".txt").exists()
        exact = name is not None and path.stem == name
        candidates.append(((complete, exact, stat.st_mtime), path))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])[1]


def _fps_matches(a: float, b: float) -> bool:
    return abs(a - b) <= FPS_TOLERANCE


def reconcile_video(
    produced: Path,
    source: VideoInfo,
    strict: bool,
    tolerance: int,
    dest: Path,
) -> tuple[str, VideoInfo]:
    """Kopiert `produced` nach `dest`, ggf. auf source.frames angepasst.

    Rueckgabe (status, produced_info); status "ok" heisst brauchbar.
      * fps weicht ab                    -> "rejected_fps_mismatch" (nicht behebbar, ohne die
                                            Zeitachse und damit die Zuordnung zu den Aktionen
                                            zu verschieben)
      * Framezahl gleich                 -> Kopie
      * |Differenz| <= tolerance         -> am Ende kuerzen bzw. letztes Frame wiederholen,
                                            auch bei strict (bewusst begrenzte Abweichung)
      * mehr Frames, strict=0            -> auf source.frames kuerzen
      * sonst                            -> "rejected_frame_mismatch"
    """
    info = ffprobe_video(produced)
    if not _fps_matches(info.fps, source.fps):
        return "rejected_fps_mismatch", info

    target = source.frames
    diff = info.frames - target
    dest.parent.mkdir(parents=True, exist_ok=True)
    if diff == 0:
        shutil.copy2(produced, dest)
        return "ok", info

    within_tolerance = abs(diff) <= tolerance
    if diff > 0 and (within_tolerance or not strict):
        filters = []
    elif diff < 0 and within_tolerance:
        # Fehlende Frames am Ende durch Wiederholen des letzten Frames auffuellen.
        filters = ["-vf", f"tpad=stop_mode=clone:stop={-diff}"]
    else:
        return "rejected_frame_mismatch", info

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(produced), *filters,
            "-frames:v", str(target), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest),
        ],
        check=True,
    )
    return "ok", info


def prepare_out_dir(out_dir: Path) -> None:
    """Ausgabeordner fuer einen Neuaufbau vorbereiten.

    Leer/nicht vorhanden -> anlegen. Stammt er von diesem Skript (Marker oder Manifest) ->
    nur die eigenen Eintraege (meta/, data/, videos/, Manifest) entfernen; sonst blieben alte
    Episoden ueber dem neuen total_episodes liegen. Fremder, nicht leerer Ordner -> Abbruch,
    damit ein falsch gesetztes AUGMENT_OUT_DIR nie etwas anderes loescht.
    """
    if out_dir.exists() and not out_dir.is_dir():
        raise SystemExit(f"AUGMENT_OUT_DIR={out_dir} ist kein Verzeichnis — abgebrochen.")
    if out_dir.is_dir() and any(out_dir.iterdir()):
        ours = (out_dir / OUT_DIR_MARKER).exists() or (out_dir / MANIFEST_NAME).exists()
        if not ours:
            raise SystemExit(
                f"AUGMENT_OUT_DIR={out_dir} ist nicht leer und stammt nicht von "
                f"assemble_dataset.py (kein {OUT_DIR_MARKER}/{MANIFEST_NAME}). Abgebrochen, "
                "damit nichts Fremdes ueberschrieben wird — leeren Ordner waehlen oder den "
                "Inhalt selbst entfernen."
            )
        for entry in OWNED_ENTRIES:
            path = out_dir / entry
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUT_DIR_MARKER).touch()


def compute_stats(parquet_paths: list[Path], features: list[str]) -> dict:
    """Wie gr00t/data/stats.py::calculate_dataset_statistics (N1.6 und N1.7 gleich)."""
    import numpy as np
    import pandas as pd

    frames = [pd.read_parquet(p) for p in sorted(parquet_paths)]
    data = pd.concat(frames, axis=0)
    stats = {}
    for feature in features:
        if feature not in data.columns:
            continue
        arr = np.vstack([np.asarray(x, dtype=np.float32) for x in data[feature]])
        stats[feature] = dict(
            mean=np.mean(arr, axis=0).tolist(),
            std=np.std(arr, axis=0).tolist(),
            min=np.min(arr, axis=0).tolist(),
            max=np.max(arr, axis=0).tolist(),
            q01=np.quantile(arr, 0.01, axis=0).tolist(),
            q99=np.quantile(arr, 0.99, axis=0).tolist(),
        )
    return stats


def _fmt(info: VideoInfo | None) -> str:
    if info is None:
        return "-"
    return f"{info.frames} Frames @ {info.fps:.3f} fps ({info.width}x{info.height})"


def dry_validate(raw_dir: Path, name: str | None, source_video: Path, max_frames: int) -> int:
    """Vorabpruefung eines kurzen Cosmos-Clips. Rueckgabe: Exit-Code (0 = passt)."""
    produced = find_produced_video(raw_dir, name)
    if produced is None:
        print(f"PREFLIGHT FEHLGESCHLAGEN: kein Cosmos-Ergebnisvideo unter {raw_dir}", file=sys.stderr)
        return 1
    src = ffprobe_video(source_video)
    got = ffprobe_video(produced)
    expected_frames = min(max_frames, src.frames) if max_frames > 0 else src.frames
    print(f"  Quelle:   {source_video}\n            {_fmt(src)}")
    print(f"  Cosmos:   {produced}\n            {_fmt(got)}")
    print(f"  erwartet: {expected_frames} Frames @ {src.fps:.3f} fps")
    problems = []
    if got.frames != expected_frames:
        problems.append(f"Framezahl {got.frames} statt {expected_frames}")
    if not _fps_matches(got.fps, src.fps):
        problems.append(f"fps {got.fps:.3f} statt {src.fps:.3f}")
    if (got.width, got.height) != (src.width, src.height):
        # Nur Warnung: GR00T skaliert die Bilder ohnehin; info.json nennt aber die Quellgroesse.
        print(
            f"  WARNUNG: Aufloesung {got.width}x{got.height} statt {src.width}x{src.height}",
            file=sys.stderr,
        )
    if problems:
        print("PREFLIGHT FEHLGESCHLAGEN: " + "; ".join(problems), file=sys.stderr)
        return 1
    print("PREFLIGHT OK: Framezahl und fps passen zur Quelle.")
    return 0


def _path(pattern: str, chunks_size: int, episode_index: int, video_key: str | None = None) -> str:
    kwargs = {"episode_chunk": episode_index // chunks_size, "episode_index": episode_index}
    if video_key is not None:
        kwargs["video_key"] = video_key
    return pattern.format(**kwargs)


def assemble(dataset_dir: Path, work_list: Path, out_dir: Path, strict: bool, tolerance: int) -> None:
    import pandas as pd

    src_meta = dataset_dir / "meta"
    src_info = json.loads((src_meta / "info.json").read_text())
    src_episodes = {
        (row := json.loads(line))["episode_index"]: row
        for line in (src_meta / "episodes.jsonl").read_text().splitlines() if line.strip()
    }
    work_items = json.loads(work_list.read_text())

    chunks_size = int(src_info.get("chunks_size", 1000))
    data_pattern = src_info.get("data_path", DEFAULT_DATA_PATH)
    video_pattern = src_info.get("video_path") or DEFAULT_VIDEO_PATH
    features = src_info.get("features", {})
    info_video_keys = sorted(k for k, v in features.items() if v.get("dtype") == "video")

    prepare_out_dir(out_dir)
    out_meta = out_dir / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)

    # meta/modality.json UNVERAENDERT uebernehmen — das ist der einzige Vertrag, den
    # run_finetuning_cotrain.sh tatsaechlich prueft (cmp -s).
    shutil.copy2(src_meta / "modality.json", out_meta / "modality.json")
    # tasks.jsonl als ganzes uebernehmen — unsere Episoden referenzieren dieselben
    # task_index-Werte wie ihre Quell-Episode (Actions/Labels werden unveraendert kopiert).
    shutil.copy2(src_meta / "tasks.jsonl", out_meta / "tasks.jsonl")
    n_tasks = sum(1 for line in (src_meta / "tasks.jsonl").read_text().splitlines() if line.strip())

    # Work-Items nach (episode, variant) gruppieren — eine Ausgabe-Episode pro Gruppe,
    # nur wenn ALLE geforderten Kameras erfolgreich waren.
    groups: dict[tuple[int, int], list[dict]] = {}
    for item in work_items:
        groups.setdefault((item["episode"], item["variant"]), []).append(item)

    requested_cams = sorted({f"observation.images.{item['camera']}" for item in work_items})
    if info_video_keys and requested_cams != info_video_keys:
        print(
            f"WARNUNG: augmentierte Kameras {requested_cams} != Video-Features der info.json "
            f"{info_video_keys} — der Datensatz ist so NICHT als COTRAIN-Datensatz ladbar "
            "(nur fuer Sichtpruefung).",
            file=sys.stderr,
        )

    manifest = []
    mismatches = []
    new_idx = 0
    global_row_index = 0
    episodes_jsonl_lines = []
    total_frames = 0
    total_videos = 0
    written_parquets: list[Path] = []

    for (src_ep, variant), items in sorted(groups.items()):
        cam_status: dict[str, str] = {}
        cam_videos: dict[str, Path] = {}
        ok = True

        for item in items:
            cam = item["camera"]
            produced = find_produced_video(Path(item["raw_output_dir"]), item.get("name"))
            if produced is None:
                cam_status[cam] = "missing_output"
                ok = False
                continue

            # Pro Kamera die eigene Quelle messen (frueher: nur die erste Kamera der Gruppe).
            src_video = ffprobe_video(Path(item["source_video"]))
            dest = out_dir / _path(video_pattern, chunks_size, new_idx, f"observation.images.{cam}")
            status, got = reconcile_video(produced, src_video, strict, tolerance, dest)
            cam_status[cam] = status
            if status == "ok":
                cam_videos[cam] = dest
            else:
                ok = False
            if got.frames != src_video.frames or not _fps_matches(got.fps, src_video.fps):
                mismatches.append({
                    "source_episode": src_ep, "variant": variant, "camera": cam,
                    "expected_frames": src_video.frames, "got_frames": got.frames,
                    "expected_fps": src_video.fps, "got_fps": got.fps, "status": status,
                })

        if not ok:
            statuses = set(cam_status.values()) - {"ok"}
            manifest.append({
                "source_episode": src_ep, "variant": variant,
                "status": "missing_camera" if "missing_output" in statuses else sorted(statuses)[0],
                "cameras": cam_status,
            })
            # Bereits geschriebene Videos dieser (unvollstaendigen) Gruppe wieder entfernen.
            for v in cam_videos.values():
                v.unlink(missing_ok=True)
            continue

        # Quell-Parquet UNVERAENDERT kopieren (nur episode_index/index umnummeriert) —
        # observation.state/action/timestamp/task_index bleiben exakt wie im echten Datensatz.
        src_parquet = dataset_dir / _path(data_pattern, chunks_size, src_ep)
        df = pd.read_parquet(src_parquet)
        n_rows = len(df)
        df["episode_index"] = new_idx
        df["index"] = range(global_row_index, global_row_index + n_rows)
        out_parquet = out_dir / _path(data_pattern, chunks_size, new_idx)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)
        written_parquets.append(out_parquet)

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
        total_videos += len(cam_videos)
        new_idx += 1

    (out_meta / "episodes.jsonl").write_text(
        "\n".join(episodes_jsonl_lines) + ("\n" if episodes_jsonl_lines else "")
    )

    out_info = dict(src_info)
    out_info["total_episodes"] = new_idx
    out_info["total_frames"] = total_frames
    out_info["total_videos"] = total_videos
    out_info["total_chunks"] = math.ceil(new_idx / chunks_size) if new_idx else 0
    out_info["total_tasks"] = n_tasks
    out_info["splits"] = {"train": f"0:{new_idx}"}
    (out_meta / "info.json").write_text(json.dumps(out_info, indent=2))

    if written_parquets:
        float_features = [k for k, v in features.items() if "float" in str(v.get("dtype", ""))]
        stats = compute_stats(written_parquets, float_features)
        (out_meta / "stats.json").write_text(json.dumps(stats, indent=4))

    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps({"episodes": manifest, "mismatches": mismatches}, indent=2))

    if mismatches:
        print("Abweichungen Cosmos-Ausgabe vs. Quelle (erwartet -> erhalten):", file=sys.stderr)
        for m in mismatches:
            print(
                f"  ep {m['source_episode']:>4} v{m['variant']} {m['camera']:<16} "
                f"{m['expected_frames']} -> {m['got_frames']} Frames, "
                f"{m['expected_fps']:.3f} -> {m['got_fps']:.3f} fps  [{m['status']}]",
                file=sys.stderr,
            )

    n_ok = sum(1 for m in manifest if m["status"] == "ok")
    print(
        f"{n_ok}/{len(manifest)} Episoden-Varianten uebernommen, {total_frames} Frames gesamt. "
        f"Manifest: {manifest_path}"
    )
    if n_ok == 0:
        raise SystemExit("Keine einzige Episode hat den Frame-/Kamera-Check bestanden — abgebrochen.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--work-list", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--strict-frame-check", type=int, default=1)
    parser.add_argument(
        "--frame-tolerance", type=int, default=0,
        help="Abweichung in Frames, die durch Kuerzen/Auffuellen am Ende ausgeglichen wird",
    )
    parser.add_argument(
        "--dry-validate", type=Path, metavar="RAW_DIR",
        help="Nur einen Preflight-Clip unter RAW_DIR gegen --source-video pruefen",
    )
    parser.add_argument("--name", help="Spec-Name des Preflight-Clips (fuer --dry-validate)")
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    if args.dry_validate is not None:
        if args.source_video is None:
            parser.error("--dry-validate braucht --source-video")
        raise SystemExit(dry_validate(args.dry_validate, args.name, args.source_video, args.max_frames))

    missing = [f for f in ("dataset_dir", "work_list", "out_dir") if getattr(args, f) is None]
    if missing:
        parser.error("fehlt: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    if args.frame_tolerance < 0:
        parser.error("--frame-tolerance muss >= 0 sein")
    assemble(
        args.dataset_dir, args.work_list, args.out_dir,
        bool(args.strict_frame_check), args.frame_tolerance,
    )


if __name__ == "__main__":
    main()
