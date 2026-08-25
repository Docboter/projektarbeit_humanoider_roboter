from __future__ import annotations

import extract_block_layout
from extract_block_layout import (
    CHUNK_SIZE,
    CUBE_COLORS,
    VIDEO_TEMPLATE,
    episode_motion_onsets,
    find_motion_onset,
)

CAMS = ["cam_left_high", "cam_right_high"]


def _track(onset: int | None, length: int = 80) -> list[list[float] | None]:
    """Ein Farbverlauf, der genau ab ``onset`` um mehr als die Schwelle abweicht."""
    if onset is None:
        return [[100.0, 100.0]] * length
    return [[100.0, 100.0] if i < onset else [140.0, 100.0] for i in range(length)]


def _prepare(tmp_path, per_cam: dict[str, dict[str, int | None]], episode: int = 0):
    """Leere Videodateien anlegen und ``track_colors`` durch feste Spuren ersetzen."""
    for cam in per_cam:
        video = tmp_path / VIDEO_TEMPLATE.format(
            episode_chunk=episode // CHUNK_SIZE,
            video_key=f"observation.images.{cam}", episode_index=episode)
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"")
    by_cam = {cam: {c: _track(o) for c, o in colors.items()}
              for cam, colors in per_cam.items()}

    def _cam_of(path) -> str:
        """Kameraname aus dem Videopfad — er steckt im Verzeichnis, nicht im Dateinamen."""
        return str(path).split("observation.images.")[1].split("/")[0]

    return lambda path, min_area=80: by_cam[_cam_of(path)]


def test_onset_needs_both_cameras_to_agree(tmp_path, monkeypatch) -> None:
    fake = _prepare(tmp_path, {
        "cam_left_high": {"rot": 30, "gruen": 10, "gelb": 20},
        # rot einig (2 Frames), gruen weit auseinander, gelb nur in einer Kamera
        "cam_right_high": {"rot": 32, "gruen": 60, "gelb": None},
    })
    monkeypatch.setattr(extract_block_layout, "track_colors", fake)
    result = episode_motion_onsets(tmp_path, 0, CAMS)

    assert result["per_color"] == {"rot": 31}
    assert result["first"] == 31
    assert "uneins" in result["rejected"]["gruen"]
    assert "fehlt" in result["rejected"]["gelb"]


def test_first_onset_is_the_earliest_reliable_cube(tmp_path, monkeypatch) -> None:
    fake = _prepare(tmp_path, {
        "cam_left_high": {"rot": 50, "gruen": 20, "gelb": 70},
        "cam_right_high": {"rot": 50, "gruen": 20, "gelb": 70},
    })
    monkeypatch.setattr(extract_block_layout, "track_colors", fake)
    result = episode_motion_onsets(tmp_path, 0, CAMS)

    assert set(result["per_color"]) == set(CUBE_COLORS)
    assert result["first"] == 20, "das Fenster muss am FRÜHESTEN bewegten Würfel enden"


def test_no_reliable_onset_leaves_first_none(tmp_path, monkeypatch) -> None:
    fake = _prepare(tmp_path, {
        "cam_left_high": {"rot": 10, "gruen": 10, "gelb": 10},
        "cam_right_high": {"rot": 60, "gruen": 60, "gelb": 60},
    })
    monkeypatch.setattr(extract_block_layout, "track_colors", fake)
    result = episode_motion_onsets(tmp_path, 0, CAMS)

    assert result["per_color"] == {}
    assert result["first"] is None, "ohne belastbaren Onset darf kein Fenster entstehen"


def test_missing_videos_do_not_invent_an_onset(tmp_path) -> None:
    result = episode_motion_onsets(tmp_path, 0, CAMS)
    assert result["first"] is None
    assert set(result["rejected"]) == set(CUBE_COLORS)


def test_motion_onset_ignores_a_single_stray_frame() -> None:
    track = _track(None)
    track[40] = [200.0, 100.0]
    assert find_motion_onset(track) is None
