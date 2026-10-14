# TL;DR: Isaac-freier Orakel-Client: spielt aufgezeichnete Aktionen statt Policy-Chunks ab.
"""Orakel-Replay für die Closed-Loop-Eval (Review 2026-10-13 §2.3, Experiment 1).

Ersetzt in ``run_g1_dex3_sim_eval.py`` den ZMQ-``PolicyClient`` durch einen Client, der die
aufgezeichneten Ground-Truth-Aktionen einer Datensatz-Episode in Chunks ausgibt. Alles
andere bleibt identisch: Layout, 30 Hz, Exec-Horizon, Meilenstein-Leiter, Konfidenzintervalle.
Damit ist das Ergebnis die **obere Schranke** der Eval: greift schon die Demo in der Sim
nicht, liegt der Engpass in der Sim und nicht in der Policy.

Zusätzlich misst der Client bei jedem Aufruf den **Tracking-Fehler** zwischen dem
Sim-Zustand (``state.*`` der Observation) und dem aufgezeichneten Zustand desselben Frames.
Der Vergleich gilt nur, solange die Demo läuft; danach hält der Client die letzte Aktion
(``hold``), damit ein fertiger Stapel noch als stabil erkannt werden kann.

Isaac-frei und ohne pandas (nur numpy), damit die Tests in der CI laufen; das Lesen der
Parquet-Datei steht in ``load_episode_arrays`` mit spätem Import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


CHUNK_LEN = 16
STATE_KEYS = ("state.left_arm", "state.right_arm", "state.left_dex3", "state.right_dex3")
# Policy-Reihenfolge (client.build_obs): 0:14 Arme, 14:28 Dex3-Finger.
ARM_DIMS = slice(0, 14)
HAND_DIMS = slice(14, 28)


def state_from_obs(obs: dict) -> np.ndarray:
    """(28,) Zustand aus dem flachen Observation-Dict von ``client.build_obs``."""
    return np.concatenate([np.asarray(obs[k], dtype=np.float64).reshape(-1) for k in STATE_KEYS])


class OracleReplayClient:
    """Gleiche Schnittstelle wie ``client.PolicyClient`` (ping/reset/get_action/close).

    Args:
        execution_horizon: Schritte, die die Eval je Chunk ausführt. Der Cursor rückt um
            genau diesen Wert vor; so bleibt die abgespielte Aktion im Takt der Sim.
        on_reset: Wird in ``reset()`` mit dem aufgezeichneten Startzustand (28,) gerufen.
            Die Eval setzt damit den Roboter auf Frame 0 — ohne das spränge der erste
            Befehl aus der Home-Pose auf die Demo-Pose.
    """

    def __init__(self, execution_horizon: int,
                 on_reset: Callable[[np.ndarray], None] | None = None) -> None:
        if execution_horizon < 1:
            raise ValueError(f"execution_horizon muss >= 1 sein: {execution_horizon}")
        self.execution_horizon = int(execution_horizon)
        self.on_reset = on_reset
        self.episode: int | None = None
        self.actions: np.ndarray | None = None
        self.states: np.ndarray | None = None
        self.cursor = 0
        self._err: list[np.ndarray] = []

    # -- Episode ---------------------------------------------------------------

    def load(self, episode: int, actions: np.ndarray, states: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        states = np.asarray(states, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 28 or states.shape != actions.shape:
            raise ValueError(f"Episode {episode}: erwartet action/state (T, 28), erhalten "
                             f"{actions.shape}/{states.shape}")
        if len(actions) == 0:
            raise ValueError(f"Episode {episode}: leer")
        self.episode = int(episode)
        self.actions, self.states = actions, states
        self.cursor = 0
        self._err = []

    @property
    def demo_steps(self) -> int:
        return 0 if self.actions is None else len(self.actions)

    # -- PolicyClient-Schnittstelle ----------------------------------------------

    def ping(self, retries: int = 1, delay: float = 0.0) -> bool:
        return True

    def reset(self) -> None:
        if self.actions is None:
            raise RuntimeError("OracleReplayClient.reset() vor load()")
        self.cursor = 0
        self._err = []
        if self.on_reset is not None:
            self.on_reset(self.states[0].copy())

    def get_action(self, obs: dict) -> np.ndarray:
        if self.actions is None:
            raise RuntimeError("OracleReplayClient.get_action() vor load()")
        if self.cursor < self.demo_steps:
            self._err.append(state_from_obs(obs) - self.states[self.cursor])
        idx = np.minimum(np.arange(self.cursor, self.cursor + CHUNK_LEN), self.demo_steps - 1)
        self.cursor += self.execution_horizon
        return self.actions[idx].copy()

    def close(self) -> None:
        pass

    # -- Auswertung ---------------------------------------------------------------

    def tracking_summary(self) -> dict:
        """Sim-Zustand minus aufgezeichneter Zustand, nur über die Demo-Dauer [rad].

        ``rmse_*``: über alle Stichproben und Gelenke der Gruppe. ``max_abs_*``: größter
        Einzelfehler. ``final_hand_rmse``: letzte Stichprobe — zeigt, ob die Finger am Ende
        dort stehen, wo die reale Hand stand (geklemmt oder vom Würfel blockiert).
        ``per_dim_rmse``: je Dimension, für die Suche nach einzelnen geklemmten Gelenken.
        """
        if not self._err:
            return {"samples": 0}
        e = np.stack(self._err)
        rmse = lambda a: float(np.sqrt(np.mean(a ** 2)))  # noqa: E731
        return {
            "samples": int(len(e)),
            "demo_steps": self.demo_steps,
            "rmse_arm": round(rmse(e[:, ARM_DIMS]), 4),
            "rmse_hand": round(rmse(e[:, HAND_DIMS]), 4),
            "max_abs_arm": round(float(np.abs(e[:, ARM_DIMS]).max()), 4),
            "max_abs_hand": round(float(np.abs(e[:, HAND_DIMS]).max()), 4),
            "final_hand_rmse": round(rmse(e[-1, HAND_DIMS]), 4),
            "per_dim_rmse": [round(float(v), 4) for v in np.sqrt(np.mean(e ** 2, axis=0))],
        }


# ---------------------------------------------------------------------------
# Datensatz lesen
# ---------------------------------------------------------------------------

V21_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


def episode_parquet(root: Path, info: dict, episode: int) -> Path:
    """Parquet-Pfad einer Episode (eine Datei je Episode, LeRobot v2.1).

    Erst die Schablone aus ``meta/info.json``, dann die v2.1-Standardschablone. Grund: auf
    dem IKR-Server meldet die info.json des konvertierten Datensatzes noch die
    v3.0-Schablone (``file-{file_index}``), die Dateien liegen aber je Episode vor.
    """
    chunk = episode // int(info.get("chunks_size", 1000))
    candidates = []
    for template in (info.get("data_path"), V21_TEMPLATE):
        if not template:
            continue
        try:
            candidates.append(root / template.format(episode_chunk=chunk, episode_index=episode))
        except (KeyError, IndexError):
            continue
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1]


def load_episode_arrays(root: str | Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    """(actions, states) einer Episode, je (T, 28) float32. pandas erst hier importiert."""
    import json

    import pandas as pd

    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    path = episode_parquet(root, info, episode)
    if not path.is_file():
        raise FileNotFoundError(f"Episode {episode}: {path} fehlt (Datensatz nicht v2.1?)")
    frame = pd.read_parquet(path, columns=["action", "observation.state"])
    actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
    states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    return actions, states
