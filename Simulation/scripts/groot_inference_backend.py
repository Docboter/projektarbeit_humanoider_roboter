#!/usr/bin/env python3
# TL;DR: Gemeinsame, importierte Bibliothek fuer ONNX/TensorRT-Backends der GR00T-DiT-Inferenz.
"""Gemeinsame ONNX/TensorRT-Helfer fuer die GR00T-N1.6-Inferenz.

Nur der DiT-Forward des Action-Heads wird ersetzt. Vision-/Sprach-Backbone,
Processor, Flow-Schleife und Action-Decoding bleiben Teil der originalen
``Gr00tPolicy``. Dadurch bleibt auch das ZMQ-Protokoll unveraendert.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def checkpoint_fingerprint(model_path: str | Path) -> str:
    """Schneller, stabiler Fingerprint ohne mehrgigabyte-grosse Gewichte ganz zu lesen."""
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint-Verzeichnis fehlt: {root}")
    digest = hashlib.sha256()
    candidates = sorted(
        p for p in root.rglob("*")
        if p.is_file() and (
            p.suffix in {".json", ".yaml", ".yml", ".safetensors"}
            or p.name in {"config", "model"}
        )
    )
    for path in candidates:
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        digest.update(f"{rel}\0{stat.st_size}\0".encode())
        # Konfigurationen komplett, grosse Weight-Shards am Anfang und Ende lesen.
        with path.open("rb") as handle:
            if stat.st_size <= 2 * 1024 * 1024 or path.suffix != ".safetensors":
                digest.update(handle.read())
            else:
                digest.update(handle.read(1024 * 1024))
                handle.seek(-1024 * 1024, 2)
                digest.update(handle.read(1024 * 1024))
    if not candidates:
        raise FileNotFoundError(f"Keine Checkpoint-Dateien unter {root}")
    return digest.hexdigest()


def gpu_fingerprint(torch_module, device: int = 0) -> dict[str, Any]:
    props = torch_module.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "compute_capability": [props.major, props.minor],
        "total_memory": int(props.total_memory),
        "torch": torch_module.__version__,
        "torch_cuda": torch_module.version.cuda,
    }


def metadata_path_for_engine(engine_path: str | Path) -> Path:
    return Path(engine_path).with_name("export_metadata.json")


def load_metadata(engine_path: str | Path) -> dict[str, Any]:
    path = metadata_path_for_engine(engine_path)
    if not path.is_file():
        raise FileNotFoundError(f"TensorRT-Metadaten fehlen: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_engine_metadata(
    engine_path: str | Path, model_path: str | Path, torch_module, trt_module
) -> dict[str, Any]:
    engine = Path(engine_path)
    if not engine.is_file():
        raise FileNotFoundError(f"TensorRT-Engine fehlt: {engine}")
    meta = load_metadata(engine)
    expected_checkpoint = checkpoint_fingerprint(model_path)
    if meta.get("checkpoint_fingerprint") != expected_checkpoint:
        raise RuntimeError(
            "TensorRT-Engine gehoert zu einem anderen Checkpoint: "
            f"{meta.get('checkpoint_fingerprint')} != {expected_checkpoint}"
        )
    current_gpu = gpu_fingerprint(torch_module)
    built_gpu = meta.get("gpu", {})
    for key in ("name", "compute_capability"):
        if built_gpu.get(key) != current_gpu.get(key):
            raise RuntimeError(
                f"TensorRT-Engine ist GPU-spezifisch: {key} "
                f"{built_gpu.get(key)!r} != {current_gpu.get(key)!r}"
            )
    if meta.get("tensorrt") != trt_module.__version__:
        raise RuntimeError(
            "TensorRT-Version der Engine passt nicht zur Runtime: "
            f"{meta.get('tensorrt')} != {trt_module.__version__}"
        )
    return meta


class TensorRTDiT:
    """PyTorch-kompatibler Callable um eine TensorRT-DiT-Engine."""

    def __init__(self, engine_path: str | Path, torch_module, device: int = 0):
        import tensorrt as trt

        self.torch = torch_module
        self.trt = trt
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with Path(engine_path).open("rb") as handle:
            self.engine = self.runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"TensorRT-Engine kann nicht geladen werden: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT ExecutionContext konnte nicht erzeugt werden")

    def _torch_dtype(self, trt_dtype):
        name = str(trt_dtype).lower()
        if "bfloat16" in name or "bf16" in name:
            return self.torch.bfloat16
        if "float16" in name or "half" in name:
            return self.torch.float16
        if "float32" in name or name.endswith("float"):
            return self.torch.float32
        if "int64" in name:
            return self.torch.int64
        if "int32" in name:
            return self.torch.int32
        if "bool" in name:
            return self.torch.bool
        raise TypeError(f"Nicht unterstuetzter TensorRT-Datentyp: {trt_dtype}")

    def __call__(
        self,
        sa_embs,
        vl_embs,
        timestep,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        values = {
            "sa_embs": sa_embs,
            "vl_embs": vl_embs,
            "timestep": timestep,
            "image_mask": image_mask,
            "backbone_attention_mask": backbone_attention_mask,
        }
        device = f"cuda:{self.device}"
        keepalive = []
        output = None
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                value = values.get(name)
                if value is None:
                    raise RuntimeError(f"TensorRT-Engine erwartet fehlenden Input '{name}'")
                value = value.to(device).contiguous()
                values[name] = value
                self.context.set_input_shape(name, tuple(value.shape))
                self.context.set_tensor_address(name, value.data_ptr())
                keepalive.append(value)

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) != self.trt.TensorIOMode.OUTPUT:
                continue
            if output is not None:
                raise RuntimeError("DiT-Engine hat mehr als einen Output")
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._torch_dtype(self.engine.get_tensor_dtype(name))
            output = self.torch.empty(shape, dtype=dtype, device=device)
            self.context.set_tensor_address(name, output.data_ptr())

        if output is None:
            raise RuntimeError("DiT-Engine hat keinen Output")
        stream = self.torch.cuda.current_stream(self.device).cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT-DiT-Inferenz fehlgeschlagen")
        return output


def install_backend(
    policy,
    backend: str,
    engine_path: str | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Installiert das gewaehlte Backend in eine bereits geladene Gr00tPolicy."""
    import torch

    backend = backend.strip().lower()
    dit = policy.model.action_head.model
    if backend == "eager":
        return {"backend": "eager"}
    if backend == "compile":
        dit.forward = torch.compile(dit.forward, mode="max-autotune")
        return {"backend": "compile", "compile_mode": "max-autotune"}
    if backend != "tensorrt":
        raise ValueError(f"Unbekanntes Inferenz-Backend: {backend}")
    if not engine_path:
        raise ValueError("GROOT_TRT_ENGINE_PATH ist fuer backend=tensorrt erforderlich")

    import tensorrt as trt

    if model_path is None:
        raise ValueError("model_path ist fuer backend=tensorrt erforderlich")
    meta = validate_engine_metadata(engine_path, model_path, torch, trt)
    runner = TensorRTDiT(engine_path, torch)

    def trt_forward(
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        if return_all_hidden_states:
            raise RuntimeError("TensorRT-DiT liefert keine Zwischenzustaende")
        return runner(
            hidden_states,
            encoder_hidden_states,
            timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )

    dit.forward = trt_forward
    # Runner am Modell halten; sonst koennte Python die Runtime vorzeitig freigeben.
    dit._groot_trt_runner = runner
    return {"backend": "tensorrt", "engine": str(engine_path), "metadata": meta}
