"""
GR00T PolicyClient + Observation-Builder für die **UNITREE_G1**-Baseline
(stock G1 + Dex1-Greifer, un-finetuntes GR00T-N1.6-3B).

Spiegelt ``g1_dex3_sim/client.py``, ist aber an das ``unitree_g1``-Embodiment
angepasst (andere State-/Action-Keys, eine ``ego_view``-Kamera) und vollständig
eigenständig, damit der DEX3-Pfad unberührt bleibt.

State-Keys (7 Gruppen): left_leg, right_leg, waist, left_arm, right_arm,
                        left_hand, right_hand
Action-Keys (7): left_arm, right_arm, left_hand, right_hand, waist,
                 base_height_command, navigate_command
→ In der Sim werden nur left_arm, right_arm, left_hand, right_hand genutzt;
  die Loco-Manip-Dims (waist, base_height_command, navigate_command) werden verworfen.

Die exakten Pro-Gruppe-Dimensionen stammen aus den Normalisierungs-Statistiken
des Checkpoints und werden zur Laufzeit aus einer JSON-Datei gelesen
(erzeugt von scripts/dump_unitree_g1_dims.py). Nichts wird hartkodiert.
"""

from __future__ import annotations

import io
import json
import time

import msgpack
import numpy as np
import zmq

# State-Gruppen, die in der Sim real beobachtet werden (Rest wird zero-gefüllt).
SIM_OBSERVED_STATE = ("left_arm", "right_arm", "left_hand", "right_hand")
# Action-Gruppen, die in der Sim ausgeführt werden (Rest wird verworfen).
SIM_USED_ACTION = ("left_arm", "right_arm", "left_hand", "right_hand")


# ---------------------------------------------------------------------------
# Serialisierung — exakte Kopie von g1_dex3_sim/client.py
# ---------------------------------------------------------------------------

class MsgSerializer:
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
# Dimensions-Metadaten
# ---------------------------------------------------------------------------

def load_dims(dims_file: str) -> dict:
    """Lädt die Pro-Gruppe-Dimensionen (state/action) aus der Dump-JSON.

    Erwartetes Format (scripts/dump_unitree_g1_dims.py):
        {
          "state":  {"left_leg": 6, "right_leg": 6, "waist": 3,
                     "left_arm": 7, "right_arm": 7, "left_hand": 1, "right_hand": 1},
          "action": {"left_arm": 7, "right_arm": 7, "left_hand": 1, "right_hand": 1,
                     "waist": 3, "base_height_command": 1, "navigate_command": 3},
          "state_keys": [...], "action_keys": [...]
        }
    """
    with open(dims_file, "r") as f:
        dims = json.load(f)
    assert "state" in dims and "action" in dims, f"Ungültige Dims-JSON: {dims_file}"
    return dims


# ---------------------------------------------------------------------------
# Policy-Client
# ---------------------------------------------------------------------------

class PolicyClient:
    """ZMQ-REQ-Client für den GR00T-Server im UNITREE_G1-Modus.

    get_action liefert einen (CHUNK_SIZE, 16)-Array:
        [left_arm(7), right_arm(7), left_hand(1), right_hand(1)]
    """

    def __init__(self, dims: dict, server_url: str = "tcp://localhost:5555",
                 timeout_ms: int = 20_000):
        self._dims = dims
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

    def ping(self, retries: int = 10, delay: float = 5.0) -> bool:
        for attempt in range(1, retries + 1):
            try:
                resp = self._send_recv({"endpoint": "ping"})
                print(f"[PolicyClient] ping ok (Versuch {attempt}): {resp}")
                return True
            except zmq.error.Again:
                print(f"[PolicyClient] Server nicht bereit ({attempt}/{retries}), warte {delay}s …")
                time.sleep(delay)
        return False

    def reset(self) -> None:
        self._send_recv({"endpoint": "reset", "data": {}})

    def get_action(self, obs: dict) -> np.ndarray:
        """Sendet eine Observation und gibt einen (T, 16)-Action-Chunk zurück.

        Nur left_arm, right_arm, left_hand, right_hand werden übernommen; die
        Loco-Manip-Dims (waist, base_height_command, navigate_command) verworfen.
        Von mehrdimensionalen Hand-Actions wird die erste Spalte als binäres
        Greifer-Signal genutzt.
        """
        resp = self._send_recv({
            "endpoint": "get_action",
            "data": {"observation": obs, "options": None},
        })
        action_dict = resp[0] if isinstance(resp, (list, tuple)) else resp

        parts = []
        for key in SIM_USED_ACTION:
            arr = np.asarray(action_dict[f"action.{key}"], dtype=np.float32)[0]  # (T, D)
            if key in ("left_hand", "right_hand") and arr.shape[-1] > 1:
                arr = arr[:, :1]  # binäres Greifer-Signal = erste Spalte
            parts.append(arr)
        chunk = np.concatenate(parts, axis=-1)  # (T, 16)

        if chunk.shape[-1] != 16:
            raise ValueError(f"Unerwartete Action-Dim: {chunk.shape} (erwartet (T, 16))")
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
    ego_view: np.ndarray,
    arm_pos: np.ndarray,
    gripper: np.ndarray,
    dims: dict,
    task_description: str = "stack the blocks",
) -> dict:
    """Baut das UNITREE_G1-Observation-Dict im Gr00tSimPolicyWrapper-Flat-Format.

    Args:
        ego_view: RGB (H, W, 3) uint8 — die einzige Policy-Kamera.
        arm_pos:  (14,) float32 — [left_arm(7), right_arm(7)] aktuelle Gelenkwinkel.
        gripper:  (2,) float32 — [left, right] Greifer-Öffnungsskalar.
        dims:     Pro-Gruppe-Dimensionen (siehe load_dims).
        task_description: Language-Prompt.

    Nicht simulierte State-Gruppen (left_leg, right_leg, waist) werden mit Nullen
    der korrekten Dimension gefüllt (off-distribution — bewusst dokumentiert).

    Returns:
        Dict mit "video.ego_view" (1,1,H,W,3) uint8, "state.<gruppe>" (1,1,D) float32
        und "annotation.human.task_description" (tuple).
    """
    state_dims: dict = dims["state"]

    def sta(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float32)[np.newaxis, np.newaxis]  # (1,1,D)

    def zeros(group: str) -> np.ndarray:
        return sta(np.zeros(state_dims[group], dtype=np.float32))

    def hand(value: float, group: str) -> np.ndarray:
        v = np.zeros(state_dims[group], dtype=np.float32)
        v[0] = value
        return sta(v)

    obs = {
        "video.ego_view": np.asarray(ego_view, dtype=np.uint8)[np.newaxis, np.newaxis],
        "annotation.human.task_description": (task_description,),
    }

    arm_pos = np.asarray(arm_pos, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)

    # Alle im Embodiment definierten State-Gruppen befüllen.
    for group in state_dims:
        if group == "left_arm":
            obs["state.left_arm"] = sta(arm_pos[0:7])
        elif group == "right_arm":
            obs["state.right_arm"] = sta(arm_pos[7:14])
        elif group == "left_hand":
            obs["state.left_hand"] = hand(float(gripper[0]), group)
        elif group == "right_hand":
            obs["state.right_hand"] = hand(float(gripper[1]), group)
        else:
            # left_leg, right_leg, waist — nicht simuliert → Null.
            obs[f"state.{group}"] = zeros(group)

    return obs
