#!/usr/bin/env python3
# TL;DR: Isaac-/Cosmos-freie Tests fuer Split-Guard, Ausgabeordner, Videoauswahl und Frame-Abgleich.
"""Tests fuer Augmentation/scripts (laufen auf dem Host, nicht im Image).

    python3 -m unittest discover -s Augmentation/tests      # oder: pytest Augmentation/tests

Braucht weder GPU noch Cosmos noch pandas. Die Frame-/fps-Tests erzeugen winzige Testvideos
mit ffmpeg und werden uebersprungen, wenn ffmpeg/ffprobe fehlen.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import assemble_dataset as ad  # noqa: E402
import build_controlnet_specs as bcs  # noqa: E402


HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def make_video(path: Path, frames: int, fps: int = 30) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x48:rate={fps}",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


class SplitGuardTest(unittest.TestCase):
    def test_regular_ratio(self):
        self.assertEqual(bcs.compute_n_train(301, 0.8), 240)

    def test_ratio_one_rejected(self):
        with self.assertRaises(SystemExit):
            bcs.compute_n_train(301, 1.0)

    def test_ratio_zero_rejected(self):
        with self.assertRaises(SystemExit):
            bcs.compute_n_train(301, 0.0)

    def test_tiny_dataset_rejected(self):
        with self.assertRaises(SystemExit):
            bcs.compute_n_train(1, 0.8)

    def test_select_uses_guard(self):
        with self.assertRaises(SystemExit):
            bcs.select_episodes(10, 1.0, 0, [])

    def test_explicit_ids_outside_train_rejected(self):
        with self.assertRaises(SystemExit):
            bcs.select_episodes(10, 0.8, 0, [8])
        with self.assertRaises(SystemExit):
            bcs.select_episodes(10, 0.8, 0, [-1])
        self.assertEqual(bcs.select_episodes(10, 0.8, 0, [3, 1, 3]), [1, 3])

    def test_limit_spread(self):
        self.assertEqual(bcs.select_episodes(10, 0.8, 0, []), list(range(8)))
        self.assertEqual(bcs.select_episodes(10, 0.8, 2, []), [0, 4])


class FindProducedVideoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def touch(self, rel: str, content: bytes = b"x", mtime: float | None = None) -> Path:
        p = self.tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        if mtime is not None:
            os.utime(p, (mtime, mtime))
        return p

    def test_none_when_empty(self):
        self.assertIsNone(ad.find_produced_video(self.tmp, "ep"))
        self.assertIsNone(ad.find_produced_video(self.tmp / "missing", "ep"))

    def test_skips_control_and_mask(self):
        self.touch("ep_control_edge.mp4")
        self.touch("ep_mask_seg.mp4")
        main = self.touch("ep.mp4")
        self.assertEqual(ad.find_produced_video(self.tmp, "ep"), main)

    def test_skips_empty_file(self):
        self.touch("ep.mp4", b"")
        self.assertIsNone(ad.find_produced_video(self.tmp, "ep"))

    def test_prefers_complete_over_newer_partial(self):
        now = time.time()
        done = self.touch("old/ep.mp4", mtime=now - 100)
        self.touch("old/ep.txt")
        self.touch("new/ep.mp4", mtime=now)  # keine .txt -> evtl. halb geschrieben
        self.assertEqual(ad.find_produced_video(self.tmp, "ep"), done)

    def test_prefers_newest_among_complete(self):
        now = time.time()
        self.touch("a/ep.mp4", mtime=now - 100)
        self.touch("a/ep.txt")
        newest = self.touch("b/ep.mp4", mtime=now)
        self.touch("b/ep.txt")
        self.assertEqual(ad.find_produced_video(self.tmp, "ep"), newest)

    def test_prefers_exact_name(self):
        now = time.time()
        exact = self.touch("ep.mp4", mtime=now - 100)
        self.touch("other.mp4", mtime=now)
        self.assertEqual(ad.find_produced_video(self.tmp, "ep"), exact)


class PrepareOutDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_creates_and_marks(self):
        out = self.tmp / "out"
        ad.prepare_out_dir(out)
        self.assertTrue((out / ad.OUT_DIR_MARKER).exists())

    def test_refuses_foreign_dir(self):
        out = self.tmp / "out"
        out.mkdir()
        (out / "important.txt").write_text("nicht loeschen")
        with self.assertRaises(SystemExit):
            ad.prepare_out_dir(out)
        self.assertTrue((out / "important.txt").exists())

    def test_clears_own_previous_output(self):
        out = self.tmp / "out"
        ad.prepare_out_dir(out)
        stale = out / "data" / "chunk-000" / "episode_000009.parquet"
        stale.parent.mkdir(parents=True)
        stale.write_text("alt")
        (out / ad.MANIFEST_NAME).write_text("{}")
        (out / "README.md").write_text("bleibt")
        ad.prepare_out_dir(out)
        self.assertFalse(stale.exists())
        self.assertFalse((out / ad.MANIFEST_NAME).exists())
        self.assertTrue((out / "README.md").exists())


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe fehlen")
class ReconcileVideoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = make_video(self.tmp / "src.mp4", 20)
        self.src_info = ad.ffprobe_video(self.src)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_case(self, frames: int, fps: int = 30, strict: bool = True, tolerance: int = 0):
        produced = make_video(self.tmp / f"p_{frames}_{fps}.mp4", frames, fps)
        dest = self.tmp / "out" / f"d_{frames}_{fps}_{int(strict)}_{tolerance}.mp4"
        status, _ = ad.reconcile_video(produced, self.src_info, strict, tolerance, dest)
        return status, dest

    def test_probe(self):
        self.assertEqual(self.src_info.frames, 20)
        self.assertAlmostEqual(self.src_info.fps, 30.0, places=2)

    def test_exact_match(self):
        status, dest = self.run_case(20)
        self.assertEqual(status, "ok")
        self.assertEqual(ad.ffprobe_video(dest).frames, 20)

    def test_strict_rejects_mismatch(self):
        self.assertEqual(self.run_case(22)[0], "rejected_frame_mismatch")
        self.assertEqual(self.run_case(18)[0], "rejected_frame_mismatch")

    def test_non_strict_trims_longer(self):
        status, dest = self.run_case(25, strict=False)
        self.assertEqual(status, "ok")
        self.assertEqual(ad.ffprobe_video(dest).frames, 20)

    def test_tolerance_pads_shorter(self):
        status, dest = self.run_case(18, tolerance=2)
        self.assertEqual(status, "ok")
        self.assertEqual(ad.ffprobe_video(dest).frames, 20)

    def test_tolerance_bounded(self):
        self.assertEqual(self.run_case(17, tolerance=2)[0], "rejected_frame_mismatch")

    def test_fps_mismatch_rejected(self):
        self.assertEqual(
            self.run_case(20, fps=24, strict=False, tolerance=5)[0], "rejected_fps_mismatch"
        )

    def test_dry_validate(self):
        raw = self.tmp / "raw"
        make_video(raw / "clip.mp4", 10)
        (raw / "clip.txt").write_text("prompt")
        self.assertEqual(ad.dry_validate(raw, "clip", self.src, 10), 0)
        self.assertEqual(ad.dry_validate(raw, "clip", self.src, 12), 1)
        self.assertEqual(ad.dry_validate(self.tmp / "leer", "clip", self.src, 10), 1)


if __name__ == "__main__":
    unittest.main()
