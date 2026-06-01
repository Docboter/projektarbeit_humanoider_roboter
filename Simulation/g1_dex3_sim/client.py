"""
Vendored GR00T PolicyClient für den Isaac-Lab-Sim-Container.

Enthält nur die ZMQ-Kommunikation — kein torch, kein gr00t-Import.
Serialisierung ist identisch mit gr00t/policy/server_client.py::MsgSerializer
(Schlüssel "as_npy", nicht "data").

Protokoll:
    Request:  msgpack({ "endpoint": str, "data": dict })
    Response: msgpack(result)
    Obs-Format: Gr00tSimPolicyWrapper flat-key format
"""

from __future__ import annotations

import io
import time

import msgpack
import numpy as np
import zmq


# ---------------------------------------------------------------------------
# Serialisierung — exakte Kopie von gr00t/policy/server_client.py
# ---------------------------------------------------------------------------

class MsgSerializer:
    """Identisch mit server-seitigem MsgSerializer für Protokoll-Kompatibilität."""

    @staticmethod
    def to_bytes(data) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes):
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _decode(obj):
        if not isinstance(obj, dict):
            return obj
        # Beide Varianten abfangen (str-Keys in msgpack 1.0+, bytes-Keys in älteren Versionen)
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(bytes(obj["as_npy"])), allow_pickle=False)
        if b"__ndarray_class__" in obj:
            return np.load(io.BytesIO(bytes(obj[b"as_npy"])), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        raise TypeError(f"Nicht serialisierbarer Typ: {type(obj)}")


# ---------------------------------------------------------------------------
# Policy-Client
# ---------------------------------------------------------------------------

class PolicyClient:
    """
    Schlanker ZMQ-REQ-Client, der mit dem GR00T-Policy-Server kommuniziert.
    Spiegelt gr00t/policy/server_client.py::PolicyClient.

    Observation-Format: Gr00tSimPolicyWrapper flat-key format
        "video.*":   (1, 1, H, W, 3) uint8
        "state.*":   (1, 1, D) float32 — pro Modalitätsgruppe separat
        "annotation.human.task_description": ("task_str",) tuple

    Action-Rückgabe: (CHUNK_SIZE, ACTION_DIM) = (16, 28) float32
    """

    ACTION_KEYS = ["left_arm", "right_arm", "left_dex3", "right_dex3"]
    ACTION_DIM = 28
    CHUNK_SIZE = 16

    def __init__(self, server_url: str = "tcp://localhost:5555", timeout_ms: int = 15_000):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(server_url)
        print(f"[PolicyClient] Verbunden mit {server_url}")

    def _send_recv(self, request: dict):
        self._sock.send(MsgSerializer.to_bytes(request))
        raw = self._sock.recv()
        resp = MsgSerializer.from_bytes(raw)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server-Fehler: {resp['error']}")
        return resp

    def ping(self, retries: int = 5, delay: float = 2.0) -> bool:
        """Wartet auf Server-Bereitschaft. Gibt True zurück wenn ok."""
        for attempt in range(1, retries + 1):
            try:
                resp = self._send_recv({"endpoint": "ping"})
                print(f"[PolicyClient] ping ok (Versuch {attempt}): {resp}")
                return True
            except zmq.error.Again:
                print(f"[PolicyClient] Server nicht bereit (Versuch {attempt}/{retries}), warte {delay}s …")
                time.sleep(delay)
        return False

    def reset(self) -> None:
        """Setzt den Server-internen Policy-State zurück (neue Episode)."""
        self._send_recv({"endpoint": "reset", "data": {}})

    def get_action(self, obs: dict) -> np.ndarray:
        """
        Sendet eine Observation (im Gr00tSimPolicyWrapper-Format) und empfängt Action-Chunk.

        Args:
            obs: Dict gebaut von build_obs() — flat keys, korrekte Shapes.

        Returns:
            (CHUNK_SIZE, ACTION_DIM) = (16, 28) float32
        """
        resp = self._send_recv({
            "endpoint": "get_action",
            "data": {"observation": obs, "options": None},
        })

        # Server-seitig liefert BasePolicy.get_action ein Tuple (action_dict, info_dict).
        # msgpack überträgt Tuples als Liste → resp == [action_dict, info_dict].
        # Das eigentliche Action-Dict ist das erste Element.
        action_dict = resp[0] if isinstance(resp, (list, tuple)) else resp

        # Gr00tSimPolicyWrapper gibt flat action keys zurück:
        # {"action.left_arm": (1,16,7), "action.right_arm": (1,16,7), ...}
        chunk = np.concatenate([
            np.asarray(action_dict[f"action.{k}"], dtype=np.float32)[0]  # (1,16,7) → (16,7)
            for k in self.ACTION_KEYS
        ], axis=-1)  # (16, 28)

        if chunk.shape != (self.CHUNK_SIZE, self.ACTION_DIM):
            raise ValueError(
                f"Unerwartete Action-Shape: {chunk.shape}, "
                f"erwartet ({self.CHUNK_SIZE}, {self.ACTION_DIM})"
            )
        return chunk

    def close(self) -> None:
        self._sock.close()
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Observation-Builder
# ---------------------------------------------------------------------------

def build_obs(
    cam_left_high: np.ndarray,
    cam_right_high: np.ndarray,
    cam_left_wrist: np.ndarray,
    cam_right_wrist: np.ndarray,
    joint_pos: np.ndarray,
    task_description: str = "stack the blocks",
) -> dict:
    """
    Baut das Observation-Dict im Gr00tSimPolicyWrapper flat-key Format auf.

    Args:
        cam_*:         RGB-Arrays (H, W, 3) uint8
        joint_pos:     (28,) float32: [left_arm(7), right_arm(7), left_dex3(7), right_dex3(7)]
        task_description: Task-Prompt für das Language-Modell

    Returns:
        Dict mit:
            "video.*":  (1, 1, H, W, 3) uint8   — Batch=1, Time=1
            "state.*":  (1, 1, D) float32         — pro Joint-Gruppe separat
            "annotation.human.task_description": ("task_str",) tuple
    """
    def vid(arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.uint8)[np.newaxis, np.newaxis]   # → (1,1,H,W,3)

    def sta(arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float32)[np.newaxis, np.newaxis]  # → (1,1,D)

    jp = np.asarray(joint_pos, dtype=np.float32)
    return {
        "video.cam_left_high":  vid(cam_left_high),
        "video.cam_right_high": vid(cam_right_high),
        "video.cam_left_wrist": vid(cam_left_wrist),
        "video.cam_right_wrist": vid(cam_right_wrist),
        "state.left_arm":   sta(jp[0:7]),
        "state.right_arm":  sta(jp[7:14]),
        "state.left_dex3":  sta(jp[14:21]),
        "state.right_dex3": sta(jp[21:28]),
        # Language: tuple of strings (Batch-Dim), NICHT ndarray
        "annotation.human.task_description": (task_description,),
    }
