"""
Vendorter GR00T-Policy-Client für den Isaac-Lab-Sim-Container.

Dieser Client enthält NUR die Teile aus gr00t/policy/server_client.py,
die für die ZMQ-Kommunikation benötigt werden — kein torch, kein gr00t-Import.
So wird der Dependency-Clash mit Isaacs gebündeltem Python vermieden.

Protokoll (REQ/REP über ZMQ):
    Request:  msgpack({ "endpoint": str, "data": dict })
    Response: msgpack(result)  — numpy-Arrays als np.save()-Bytes serialisiert
    Port:     5555 (konfigurierbar)
"""

from __future__ import annotations

import io
import time

import msgpack
import numpy as np
import zmq


# ---------------------------------------------------------------------------
# Serialisierung (identisch mit gr00t/policy/server_client.py)
# ---------------------------------------------------------------------------

class MsgSerializer:
    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj)
            return {"__ndarray_class__": True, "data": buf.getvalue()}
        raise TypeError(f"Nicht serialisierbarer Typ: {type(obj)}")

    @staticmethod
    def _decode(obj: dict):
        if "__ndarray_class__" in obj:
            buf = io.BytesIO(bytes(obj["data"]))
            return np.load(buf, allow_pickle=False)
        return obj

    def serialize(self, data) -> bytes:
        return msgpack.packb(data, default=self._encode, use_bin_type=True)

    def deserialize(self, raw: bytes):
        return msgpack.unpackb(raw, object_hook=self._decode, raw=False)


# ---------------------------------------------------------------------------
# Policy-Client
# ---------------------------------------------------------------------------

class PolicyClient:
    """
    Schlanker ZMQ-REQ-Client, der mit dem GR00T-Policy-Server kommuniziert.

    Args:
        server_url:  ZMQ-Endpunkt des Servers, z.B. "tcp://localhost:5555"
        timeout_ms:  Empfangs-/Sende-Timeout in Millisekunden
    """

    ACTION_DIM = 28
    CHUNK_SIZE = 16

    def __init__(self, server_url: str = "tcp://localhost:5555", timeout_ms: int = 15_000):
        self._serializer = MsgSerializer()
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(server_url)
        print(f"[PolicyClient] Verbunden mit {server_url}")

    # ------------------------------------------------------------------
    # Interne Kommunikation
    # ------------------------------------------------------------------

    def _call(self, endpoint: str, data: dict | None = None):
        payload = {"endpoint": endpoint, "data": data or {}}
        self._sock.send(self._serializer.serialize(payload))
        raw = self._sock.recv()
        resp = self._serializer.deserialize(raw)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server-Fehler [{endpoint}]: {resp['error']}")
        return resp

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def ping(self, retries: int = 5, delay: float = 2.0) -> bool:
        """Wartet auf Server-Bereitschaft. Gibt True zurück wenn ok."""
        for attempt in range(1, retries + 1):
            try:
                resp = self._call("ping")
                print(f"[PolicyClient] ping ok (Versuch {attempt}): {resp}")
                return True
            except zmq.error.Again:
                print(f"[PolicyClient] Server noch nicht bereit (Versuch {attempt}/{retries}), warte {delay}s …")
                time.sleep(delay)
        return False

    def reset(self) -> None:
        """Setzt den Server-internen Policy-State zurück (neue Episode)."""
        self._call("reset")

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """
        Sendet eine Observation und empfängt einen Action-Chunk.

        Args:
            obs: Dict mit Schlüsseln:
                "video.cam_left_high"   — (H, W, 3) uint8
                "video.cam_right_high"  — (H, W, 3) uint8
                "video.cam_left_wrist"  — (H, W, 3) uint8
                "video.cam_right_wrist" — (H, W, 3) uint8
                "state.joint_pos"       — (28,) float32
                "annotation.human.task_description" — str (als bytes-Numpy-Array)

        Returns:
            action_chunk: np.ndarray shape (CHUNK_SIZE, ACTION_DIM) = (16, 28)
        """
        resp = self._call("get_action", {"obs": obs})

        # Server antwortet je nach Version direkt mit dem Array oder als Dict
        if isinstance(resp, np.ndarray):
            chunk = resp
        elif isinstance(resp, dict):
            # Mögliche Schlüssel: "action", "actions", "action_chunk"
            for key in ("action_chunk", "actions", "action"):
                if key in resp:
                    chunk = np.asarray(resp[key], dtype=np.float32)
                    break
            else:
                raise ValueError(f"Unbekanntes Response-Format: {list(resp.keys())}")
        else:
            chunk = np.asarray(resp, dtype=np.float32)

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
# Hilfsfunktionen für Observation-Aufbau
# ---------------------------------------------------------------------------

def build_obs(
    cam_left_high: np.ndarray,
    cam_right_high: np.ndarray,
    cam_left_wrist: np.ndarray,
    cam_right_wrist: np.ndarray,
    joint_pos: np.ndarray,
    task_description: str = "stack the blocks",
) -> dict[str, np.ndarray]:
    """
    Baut das Observation-Dict im GR00T-Eingabeformat auf.

    Kamera-Arrays müssen uint8 RGB (H, W, 3) sein.
    joint_pos muss float32 (28,) sein: [left_arm(7), right_arm(7), left_dex3(7), right_dex3(7)]
    """
    # GR00T erwartet Bilder mit Batch-Dim: (1, H, W, 3) oder (H, W, 3)?
    # Aus run_gr00t_server.py: direkte (H, W, 3)-Arrays, keine Batch-Dim.
    return {
        "video.cam_left_high": cam_left_high.astype(np.uint8),
        "video.cam_right_high": cam_right_high.astype(np.uint8),
        "video.cam_left_wrist": cam_left_wrist.astype(np.uint8),
        "video.cam_right_wrist": cam_right_wrist.astype(np.uint8),
        "state.joint_pos": joint_pos.astype(np.float32),
        # Task-Description als UTF-8-Bytes im ndarray
        "annotation.human.task_description": np.frombuffer(
            task_description.encode("utf-8"), dtype=np.uint8
        ),
    }
