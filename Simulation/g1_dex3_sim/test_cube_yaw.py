"""Gierwinkel der Würfel: Schätzer, Tor und die Umrechnung ins Quaternion.

Der Schätzer wird gegen SYNTHETISCHE Würfel bekannter Drehung geprüft, gerendert durch
dasselbe Kameramodell, das ihn später auswertet. Das ist keine Rundum-Bestätigung — beide
Seiten teilen sich die Kameraannahme —, aber es fängt genau die Klasse Fehler ab, an der
der Vorgänger scheiterte: Einrasten auf das Pixelraster und die perspektivische Verzerrung
der Deckfläche. Die Abnahme gegen den echten Renderer ist ``detect --expect-yaw``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from camera_geometry import PinholeCamera
from extract_block_layout import (
    accept_yaw,
    locate_cubes,
    mean_yaw_deg,
    synthetic_cube_image,
    yaw_delta_deg,
)


def measure(yaw_deg: float, camera: str = "cam_left_high",
            center=(0.35, 0.0), noise: float = 0.0) -> dict:
    cam = PinholeCamera.from_cfg(camera)
    rgb = synthetic_cube_image(cam, center, yaw_deg, noise=noise,
                               rng=np.random.default_rng(0))
    found = locate_cubes(rgb, cam)[0]
    assert found is not None, f"roter Würfel bei {yaw_deg}° nicht gefunden"
    return found


# ── Schätzer ────────────────────────────────────────────────────────────────

def test_recovers_known_yaw_over_the_full_quarter_turn() -> None:
    for truth in (0.0, 7.0, 22.5, 40.0, 63.0, 89.0):
        got = measure(truth)["yaw_deg"]
        assert yaw_delta_deg(got, truth) < 1.0, f"{truth}° → {got}°"


def test_yaw_is_reported_modulo_ninety_degrees() -> None:
    """Ein Würfel ist 4-zählig: 10° und 100° sind dieselbe Lage."""
    assert yaw_delta_deg(measure(10.0)["yaw_deg"], measure(100.0)["yaw_deg"]) < 1.0


def test_both_high_cameras_agree_on_the_same_cube() -> None:
    for truth in (5.0, 30.0, 70.0):
        left = measure(truth, "cam_left_high")["yaw_deg"]
        right = measure(truth, "cam_right_high")["yaw_deg"]
        assert yaw_delta_deg(left, right) < 2.0, f"{truth}°: {left} vs {right}"


def test_top_face_is_square_shaped_and_five_centimetres_wide() -> None:
    """Formprobe und Breite müssen beim sauberen Würfel den Sollwerten nahekommen.

    Sie sind das Tor gegen verunreinigte Masken — wenn sie schon beim idealen Würfel
    danebenlägen, würde das Tor die falschen Fälle durchlassen.
    """
    found = measure(25.0)
    assert 1.30 <= found["top_squareness"] <= 1.55, found["top_squareness"]
    assert 4.0 <= found["top_width_cm"] <= 5.6, found["top_width_cm"]


def test_estimate_does_not_snap_to_the_image_axes() -> None:
    """Der Vorgänger lieferte auf 101 von 120 Realframes exakt 0,0°.

    Ein Schätzer, der auf das Pixelraster einrastet, gibt für viele verschiedene wahre
    Winkel denselben Wert zurück. Also: viele Winkel messen und zählen, wie viele
    verschiedene Antworten herauskommen.
    """
    got = [measure(t)["yaw_deg"] for t in np.arange(0.0, 90.0, 6.0)]
    assert len(got) == len({round(g) for g in got}), got


# ── Tor ─────────────────────────────────────────────────────────────────────

def test_gate_rejects_a_single_camera() -> None:
    """Ein Winkel ohne Gegenprobe wird nicht geglaubt, egal wie gut er aussieht."""
    assert not accept_yaw([measure(20.0)], tolerance_deg=8.0, min_squareness=1.25)


def test_gate_rejects_disagreeing_cameras() -> None:
    a, b = measure(20.0), dict(measure(20.0))
    b["yaw_deg"] = (a["yaw_deg"] + 25.0) % 90.0
    assert not accept_yaw([a, b], tolerance_deg=8.0, min_squareness=1.25)


def test_gate_rejects_a_non_square_top_face() -> None:
    """Formprobe nahe 1 heißt: die Punktwolke hat gar keine Kantenrichtung."""
    a, b = dict(measure(20.0)), dict(measure(20.0))
    a["top_squareness"] = b["top_squareness"] = 1.02
    assert not accept_yaw([a, b], tolerance_deg=8.0, min_squareness=1.25)


def test_gate_accepts_two_agreeing_square_cameras() -> None:
    recs = [measure(20.0, "cam_left_high"), measure(20.0, "cam_right_high")]
    assert accept_yaw(recs, tolerance_deg=8.0, min_squareness=1.25)


# ── Winkelarithmetik ────────────────────────────────────────────────────────

def test_circular_mean_does_not_average_across_the_wrap() -> None:
    """1° und 89° liegen 2° auseinander, ihr Mittel ist 0° — nicht 45°."""
    assert mean_yaw_deg([1.0, 89.0]) < 0.01 or mean_yaw_deg([1.0, 89.0]) > 89.99
    assert abs(mean_yaw_deg([10.0, 20.0]) - 15.0) < 1e-6
    assert mean_yaw_deg([None]) is None


def test_delta_wraps_at_ninety_degrees() -> None:
    assert abs(yaw_delta_deg(1.0, 89.0) - 2.0) < 1e-9
    assert abs(yaw_delta_deg(0.0, 45.0) - 45.0) < 1e-9
    assert yaw_delta_deg(10.0, 100.0) < 1e-9


# ── Quaternion ──────────────────────────────────────────────────────────────

def load_yaw_to_quat():
    """``render_cotrain_dataset`` importiert Isaac beim Laden — also nur die Funktion holen."""
    src = Path(__file__).with_name("render_cotrain_dataset.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "yaw_to_quat")
    ns: dict = {"np": np}
    exec(compile(ast.Module([fn], []), "<yaw_to_quat>", "exec"), ns)
    return ns["yaw_to_quat"]


def test_quaternion_is_a_pure_z_rotation_of_unit_length() -> None:
    yaw_to_quat = load_yaw_to_quat()
    for deg in (0.0, 30.0, 45.0, 90.0):
        w, x, y, z = yaw_to_quat(deg)
        assert (x, y) == (0.0, 0.0)
        assert abs(np.linalg.norm([w, x, y, z]) - 1.0) < 1e-9
        assert abs(2.0 * np.degrees(np.arctan2(z, w)) - deg) < 1e-9


def test_zero_yaw_is_the_identity_quaternion() -> None:
    """Der Rückfall muss exakt das alte Verhalten sein, nicht beinahe."""
    assert load_yaw_to_quat()(0.0) == (1.0, 0.0, 0.0, 0.0)


# ── Diagnosebericht ─────────────────────────────────────────────────────────

def layout_with(records_per_episode: list[list[dict | None]], tmp_path) -> str:
    """Minimale layout.json mit genau einem (roten) Würfel je Episode."""
    import json
    episodes = {
        str(i): {"cubes": [[0.35, 0.0], None, None],
                 "per_camera": {"cam_left_high": [recs[0], None, None],
                                "cam_right_high": [recs[1], None, None]}}
        for i, recs in enumerate(records_per_episode)}
    out = tmp_path / "layout.json"
    out.write_text(json.dumps({"episodes": episodes}))
    return str(out)


def report_counts(path: str, capsys) -> dict[str, int]:
    import argparse

    from extract_block_layout import run_report
    assert run_report(argparse.Namespace(layout=path, yaw_tolerance=8.0,
                                         min_squareness=1.25)) == 0
    out = capsys.readouterr().out
    keys = {"durchgelassen": "pass", "an der Formprobe": "square",
            "an der Uneinigkeit": "agree", "ohne zwei Messungen": "none"}
    counts = {}
    for line in out.splitlines():
        for needle, key in keys.items():
            if line.strip().startswith(needle):
                counts[key] = int(next(t for t in line.split() if t.isdigit()))
    return counts


def rec(yaw: float, squareness: float = 1.41) -> dict:
    return {"color": "rot", "xy": [0.35, 0.0], "yaw_deg": yaw,
            "top_squareness": squareness, "top_px": 400, "fill": 0.9,
            "top_width_cm": 5.0, "yaw_coherence": 0.5}


def test_report_blames_the_shape_check_when_the_face_is_not_square(tmp_path, capsys) -> None:
    path = layout_with([[rec(20.0, 1.05), rec(20.0, 1.05)]], tmp_path)
    assert report_counts(path, capsys)["square"] == 1


def test_report_blames_disagreement_when_both_faces_are_square(tmp_path, capsys) -> None:
    path = layout_with([[rec(20.0), rec(50.0)]], tmp_path)
    assert report_counts(path, capsys)["agree"] == 1


def test_report_counts_a_cube_without_two_measurements_separately(tmp_path, capsys) -> None:
    path = layout_with([[rec(20.0), {"color": "rot", "xy": [0.35, 0.0]}]], tmp_path)
    assert report_counts(path, capsys)["none"] == 1


def test_report_counts_a_good_cube_as_passed(tmp_path, capsys) -> None:
    path = layout_with([[rec(20.0), rec(22.0)]], tmp_path)
    assert report_counts(path, capsys)["pass"] == 1


# ── Deckflächenschnitt ──────────────────────────────────────────────────────

def test_otsu_finds_the_valley_between_two_brightness_peaks() -> None:
    from extract_block_layout import otsu_threshold
    rng = np.random.default_rng(0)
    dark, bright = rng.normal(0.35, 0.05, 500), rng.normal(0.85, 0.05, 600)
    thr = otsu_threshold(np.concatenate([dark, bright]))
    assert 0.45 < thr < 0.75, thr


def test_otsu_survives_a_single_peak_and_a_constant_image() -> None:
    from extract_block_layout import otsu_threshold
    flat = np.full(200, 0.5)
    assert abs(otsu_threshold(flat) - 0.5) < 1e-6
    one = np.random.default_rng(1).normal(0.6, 0.03, 400)
    assert one.min() <= otsu_threshold(one) <= one.max()


def test_measured_threshold_beats_the_old_fixed_quota_under_oblique_light() -> None:
    """Der Grund, warum die feste Quote weg ist — und der Riegel dagegen, sie zurückzuholen.

    Bei schrägem Licht liegt die hellste Seitenfläche dicht an der Deckfläche. Die alte
    Regel „hellste 30 %" schneidet dann mitten in die Deckfläche hinein, die Fläche wird
    schmaler als 5 cm und die Formprobe bricht ein.
    """
    from extract_block_layout import color_mask, largest_blob, measure_yaw
    cam = PinholeCamera.from_cfg("cam_left_high")
    scores = {}
    for name, quantile in (("gemessen", None), ("hellste 30 %", 70.0)):
        errors, widths = [], []
        for truth in np.arange(0.0, 90.0, 6.0):
            rgb = synthetic_cube_image(cam, (0.35, 0.0), truth,
                                       rng=np.random.default_rng(0))
            mask = color_mask(rgb, "rot")
            blob = largest_blob(mask, 120)
            found = measure_yaw(rgb, mask, blob, cam, quantile=quantile)
            if found.get("yaw_deg") is None:
                continue
            errors.append(yaw_delta_deg(found["yaw_deg"], truth))
            widths.append(found["top_width_cm"])
        scores[name] = (float(np.percentile(errors, 90)), float(np.median(widths)))
    assert scores["gemessen"][0] < 1.0, scores
    assert scores["gemessen"][0] < scores["hellste 30 %"][0], scores
    assert abs(scores["gemessen"][1] - 5.0) < 0.5, scores


def test_the_synthetic_cube_does_not_hand_the_estimator_a_free_split() -> None:
    """Eine Abnahme, die jede Schwelle bestehen lässt, prüft nichts.

    Steht die hellste Seitenfläche zu weit unter der Deckfläche, trennt sie jede Quote,
    und der Test oben wäre wertlos — deshalb hier festgehalten, dass sie dicht liegt.
    """
    from extract_block_layout import SYNTH_AMBIENT, SYNTH_LIGHT
    def lit(normal):
        return SYNTH_AMBIENT + (1.0 - SYNTH_AMBIENT) * max(0.0, float(np.dot(normal, SYNTH_LIGHT)))
    top = lit(np.array([0.0, 0.0, 1.0]))
    side = max(lit(np.array(n)) for n in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]))
    assert side / top > 0.6, (top, side)


def test_default_gate_accepts_a_real_world_rounded_cube() -> None:
    """Echte Klötzchen erreichen die Formprobe eines scharfkantigen Würfels nie.

    Auf den Realframes liegt sie bei 1,21 (p10 1,06) statt 1,37, weil die Kanten gerundet
    sind. Mit der alten Vorgabe 1,25 überlebte 1 von 15 Würfeln, mit 1,00 sind es 9 — und
    die Uneinigkeit beider Kameras stieg dabei nicht (Maximum über alle Stufen 3,7°).
    """
    rounded = [{"yaw_deg": 30.0, "top_squareness": 1.21},
               {"yaw_deg": 32.0, "top_squareness": 1.06}]
    assert accept_yaw(rounded, tolerance_deg=8.0, min_squareness=1.0)
    assert not accept_yaw(rounded, tolerance_deg=8.0, min_squareness=1.25)


def test_disagreeing_cameras_are_still_rejected_with_the_shape_check_off() -> None:
    """Die Einigkeit ist jetzt das einzige Tor — sie muss allein tragen."""
    apart = [{"yaw_deg": 10.0, "top_squareness": 1.40},
             {"yaw_deg": 35.0, "top_squareness": 1.40}]
    assert not accept_yaw(apart, tolerance_deg=8.0, min_squareness=1.0)
