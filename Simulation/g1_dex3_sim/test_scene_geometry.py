# TL;DR: Isaac-freier Test: die stacked-Schwelle folgt der Würfelkante (SIM_CUBE_EDGE).
import importlib

import scene_geometry


def _reload(monkeypatch, edge):
    if edge is None:
        monkeypatch.delenv("SIM_CUBE_EDGE", raising=False)
    else:
        monkeypatch.setenv("SIM_CUBE_EDGE", edge)
    return importlib.reload(scene_geometry)


def test_stack_height_default_5cm_unchanged(monkeypatch):
    assert _reload(monkeypatch, None).STACK_HEIGHT_MIN == 0.08


def test_stack_height_4cm_tower_counts(monkeypatch):
    g = _reload(monkeypatch, "0.04")
    assert g.STACK_HEIGHT_MIN == 0.06
    assert 2 * g.CUBE_EDGE > g.STACK_HEIGHT_MIN  # perfekter Dreierturm zählt
    _reload(monkeypatch, None)
