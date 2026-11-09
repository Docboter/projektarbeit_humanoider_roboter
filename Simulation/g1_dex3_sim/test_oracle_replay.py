# TL;DR: CPU-Tests für den Orakel-Replay-Client (Chunks, Cursor, Halten, Tracking).
"""Isaac-freie Tests für oracle_replay.OracleReplayClient (nur numpy + pytest)."""

from pathlib import Path

import numpy as np
from oracle_replay import CHUNK_LEN, OracleReplayClient, episode_parquet, state_from_obs
import pytest


def _obs(state: np.ndarray) -> dict:
    s = np.asarray(state, dtype=np.float32)
    return {
        "state.left_arm": s[0:7][None, None],
        "state.right_arm": s[7:14][None, None],
        "state.left_dex3": s[14:21][None, None],
        "state.right_dex3": s[21:28][None, None],
    }


def _episode(n: int) -> tuple[np.ndarray, np.ndarray]:
    actions = np.arange(n, dtype=np.float32)[:, None].repeat(28, axis=1)
    states = actions - 0.5
    return actions, states


def test_state_from_obs_roundtrip():
    s = np.linspace(-1, 1, 28)
    np.testing.assert_allclose(state_from_obs(_obs(s)), s, rtol=0, atol=1e-6)


def test_chunks_follow_cursor_and_hold_last_action():
    client = OracleReplayClient(execution_horizon=8)
    actions, states = _episode(20)
    client.load(240, actions, states)
    client.reset()
    first = client.get_action(_obs(states[0]))
    assert first.shape == (CHUNK_LEN, 28)
    assert first[0, 0] == 0 and first[-1, 0] == 15
    second = client.get_action(_obs(states[8]))
    assert second[0, 0] == 8
    # Ab Index 19 (letzter Demo-Frame) wird gehalten.
    assert second[-1, 0] == 19 and second[11, 0] == 19
    third = client.get_action(_obs(states[16]))
    assert third[0, 0] == 16 and (third[3:, 0] == 19).all()
    late = client.get_action(_obs(states[-1]))      # Cursor 24 > Demo: nur noch halten
    assert (late[:, 0] == 19).all()


def test_reset_sets_recorded_start_state_and_rewinds():
    seen = []
    client = OracleReplayClient(execution_horizon=4, on_reset=seen.append)
    actions, states = _episode(10)
    client.load(250, actions, states)
    client.reset()
    client.get_action(_obs(states[0]))
    client.reset()
    assert client.cursor == 0
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], states[0])


def test_tracking_only_during_demo_and_split_by_group():
    client = OracleReplayClient(execution_horizon=8)
    actions, states = _episode(16)
    client.load(260, actions, states)
    client.reset()
    for _ in range(4):          # Cursor 0, 8, 16, 24 → nur 0 und 8 liegen in der Demo
        sim = states[min(client.cursor, 15)].copy()
        sim[14:28] += 0.1       # Hände 0,1 rad daneben, Arme exakt
        client.get_action(_obs(sim))
    trk = client.tracking_summary()
    assert trk["samples"] == 2
    assert trk["rmse_arm"] == pytest.approx(0.0, abs=1e-6)
    assert trk["rmse_hand"] == pytest.approx(0.1, abs=1e-4)
    assert trk["final_hand_rmse"] == pytest.approx(0.1, abs=1e-4)
    assert len(trk["per_dim_rmse"]) == 28


def test_rejects_bad_shapes_and_unloaded_use():
    client = OracleReplayClient(execution_horizon=8)
    with pytest.raises(RuntimeError):
        client.reset()
    with pytest.raises(ValueError):
        client.load(1, np.zeros((5, 27)), np.zeros((5, 27)))
    with pytest.raises(ValueError):
        OracleReplayClient(execution_horizon=0)


def test_episode_parquet_falls_back_to_v21_layout(tmp_path: Path):
    v21 = tmp_path / "data" / "chunk-000" / "episode_000240.parquet"
    v21.parent.mkdir(parents=True)
    v21.write_bytes(b"")
    v30_info = {"data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "chunks_size": 1000}
    assert episode_parquet(tmp_path, v30_info, 240) == v21
    assert episode_parquet(tmp_path, {}, 240) == v21
