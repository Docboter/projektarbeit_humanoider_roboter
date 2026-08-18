#!/usr/bin/env python3
"""Exportiert und prueft den GR00T-N1.6-DiT fuer TensorRT.

Der Export benutzt eine synthetische Observation im exakten vier-Kamera-Simformat.
Ein Download des rund 18 GB grossen Trainingsdatensatzes ist daher nicht noetig.
Die TensorRT-Engine wird absichtlich auf der Ziel-GPU gebaut und per Metadaten an
Checkpoint, GPU-Modell, Compute Capability und TensorRT-Version gebunden.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import torch

from groot_inference_backend import (
    TensorRTDiT,
    checkpoint_fingerprint,
    gpu_fingerprint,
    install_backend,
    validate_engine_metadata,
)

CAMERAS = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist")
ACTION_KEYS = ("left_arm", "right_arm", "left_dex3", "right_dex3")


def synthetic_observation(width: int, height: int, task: str) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    obs: dict[str, Any] = {}
    for camera in CAMERAS:
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        obs[f"video.{camera}"] = frame[None, None]
    for name in ACTION_KEYS:
        obs[f"state.{name}"] = rng.standard_normal((1, 1, 7), dtype=np.float32) * 0.05
    obs["annotation.human.task_description"] = (task,)
    return obs


def load_policy(model_path: str, embodiment_tag: str):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag), model_path=model_path, device="cuda"
    )
    return policy, Gr00tSimPolicyWrapper(policy)


class InputCapture:
    def __init__(self):
        self.values: dict[str, torch.Tensor | None] | None = None

    def hook(self, _module, args, kwargs):
        if self.values is not None:
            return

        def value(name: str, pos: int | None):
            item = kwargs.get(name)
            if item is None and pos is not None and len(args) > pos:
                item = args[pos]
            return item.detach().clone() if item is not None else None

        self.values = {
            "sa_embs": value("hidden_states", 0),
            "vl_embs": value("encoder_hidden_states", 1),
            "timestep": value("timestep", 2),
            # Diese beiden sind im AlternateVLDiT nach optionalen Basisargumenten und
            # werden von GR00T als Keywords uebergeben; Positions-Fallback waere falsch.
            "image_mask": value("image_mask", None),
            "backbone_attention_mask": value("backbone_attention_mask", None),
        }


def capture_inputs(policy, wrapped, obs) -> dict[str, torch.Tensor | None]:
    capture = InputCapture()
    handle = policy.model.action_head.model.register_forward_pre_hook(
        capture.hook, with_kwargs=True
    )
    try:
        with torch.inference_mode():
            wrapped.get_action(obs)
    finally:
        handle.remove()
    if capture.values is None:
        raise RuntimeError("DiT-Inputs wurden beim GR00T-Forward nicht erfasst")
    required = ("sa_embs", "vl_embs", "timestep")
    if any(capture.values[name] is None for name in required):
        raise RuntimeError(f"Unvollstaendige DiT-Inputs: {capture.values}")
    return capture.values


def _export_wrapper(dit, has_image_mask: bool, has_backbone_mask: bool):
    if has_image_mask and has_backbone_mask:
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dit = dit

            def forward(self, sa_embs, vl_embs, timestep, image_mask, backbone_attention_mask):
                return self.dit(
                    sa_embs,
                    vl_embs,
                    timestep,
                    image_mask=image_mask,
                    backbone_attention_mask=backbone_attention_mask,
                )
        return Wrapper()
    if has_image_mask:
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dit = dit

            def forward(self, sa_embs, vl_embs, timestep, image_mask):
                return self.dit(sa_embs, vl_embs, timestep, image_mask=image_mask)
        return Wrapper()

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = dit

        def forward(self, sa_embs, vl_embs, timestep):
            return self.dit(sa_embs, vl_embs, timestep)
    return Wrapper()


def export_onnx(policy, captured, onnx_path: Path) -> dict[str, Any]:
    dit = policy.model.action_head.model.eval()
    names = ["sa_embs", "vl_embs", "timestep"]
    if captured["image_mask"] is not None:
        names.append("image_mask")
    if captured["backbone_attention_mask"] is not None:
        names.append("backbone_attention_mask")
    inputs = tuple(captured[name] for name in names)
    wrapper = _export_wrapper(
        dit,
        captured["image_mask"] is not None,
        captured["backbone_attention_mask"] is not None,
    ).eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_path),
            input_names=names,
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )
    import onnx

    onnx.checker.check_model(str(onnx_path))
    return {
        name: {"shape": list(captured[name].shape), "dtype": str(captured[name].dtype)}
        for name in names
    }


def build_engine(onnx_path: Path, engine_path: Path, workspace_mb: int) -> str:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT kann ONNX nicht parsen:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024**2)
    config.set_flag(trt.BuilderFlag.BF16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT-Engine-Build lieferte kein Ergebnis")
    engine_path.write_bytes(serialized)
    return trt.__version__


def seed_everything(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_action(action: dict[str, np.ndarray]) -> np.ndarray:
    arrays = []
    for key in ACTION_KEYS:
        full = f"action.{key}"
        if full not in action:
            raise KeyError(f"Action-Gruppe fehlt: {full}; vorhanden: {sorted(action)}")
        arrays.append(np.asarray(action[full], dtype=np.float32))
    result = np.concatenate(arrays, axis=-1)
    if result.shape != (1, 16, 28):
        raise RuntimeError(f"Unerwartete Action-Shape: {result.shape}, erwartet (1, 16, 28)")
    return result


def timed_actions(wrapped, obs, iterations: int, warmup: int) -> list[float]:
    for index in range(warmup):
        seed_everything(10_000 + index)
        wrapped.get_action(obs)
    torch.cuda.synchronize()
    samples = []
    for index in range(iterations):
        seed_everything(20_000 + index)
        start = time.perf_counter()
        wrapped.get_action(obs)
        torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - start))
    return samples


def stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_ms": round(statistics.fmean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "max_ms": round(max(samples), 3),
    }


def parity(reference: np.ndarray, candidate: np.ndarray, mean_tol: float, max_tol: float):
    delta = np.abs(reference - candidate)
    mean_abs = float(delta.mean())
    max_abs = float(delta.max())
    return {
        "passed": mean_abs <= mean_tol and max_abs <= max_tol,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "mean_tolerance": mean_tol,
        "max_tolerance": max_tol,
    }


def call_dit(forward, captured):
    kwargs = {}
    if captured["image_mask"] is not None:
        kwargs["image_mask"] = captured["image_mask"]
    if captured["backbone_attention_mask"] is not None:
        kwargs["backbone_attention_mask"] = captured["backbone_attention_mask"]
    return forward(
        captured["sa_embs"], captured["vl_embs"], captured["timestep"], **kwargs
    )


def timed_dit(forward, captured, iterations: int, warmup: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            call_dit(forward, captured)
        torch.cuda.synchronize()
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            call_dit(forward, captured)
            torch.cuda.synchronize()
            samples.append(1000 * (time.perf_counter() - start))
    return samples


@contextmanager
def temporary_forward(dit, forward):
    original = dit.forward
    dit.forward = forward
    try:
        yield
    finally:
        dit.forward = original


def validate_and_benchmark(
    policy,
    wrapped,
    obs,
    model_path: str,
    engine_path: Path,
    iterations: int,
    warmup: int,
    mean_tol: float,
    max_tol: float,
) -> dict[str, Any]:
    import tensorrt as trt

    metadata = validate_engine_metadata(engine_path, model_path, torch, trt)
    dit = policy.model.action_head.model
    original_forward = dit.forward
    captured = capture_inputs(policy, wrapped, obs)

    seed_everything()
    eager_action = flatten_action(wrapped.get_action(obs)[0])
    eager_times = timed_actions(wrapped, obs, iterations, warmup)
    eager_dit_times = timed_dit(original_forward, captured, iterations, warmup)

    compile_times = None
    compile_parity = None
    compile_error = None
    try:
        compiled = torch.compile(original_forward, mode="max-autotune")
        with temporary_forward(dit, compiled):
            seed_everything()
            compile_action = flatten_action(wrapped.get_action(obs)[0])
            compile_parity = parity(eager_action, compile_action, mean_tol, max_tol)
            compile_times = timed_actions(wrapped, obs, iterations, warmup)
    except Exception as exc:  # TensorRT bleibt nutzbar, auch wenn Triton/ptxas fehlt.
        compile_error = f"{type(exc).__name__}: {exc}"

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

    with temporary_forward(dit, trt_forward):
        seed_everything()
        trt_action = flatten_action(wrapped.get_action(obs)[0])
        trt_times = timed_actions(wrapped, obs, iterations, warmup)
        trt_dit_times = timed_dit(trt_forward, captured, iterations, warmup)

    trt_parity = parity(eager_action, trt_action, mean_tol, max_tol)
    denoising_steps = int(getattr(policy.model.action_head, "num_inference_timesteps", 1))
    eager_mean = statistics.fmean(eager_times)
    eager_dit_mean = statistics.fmean(eager_dit_times)
    trt_mean = statistics.fmean(trt_times)
    trt_dit_mean = statistics.fmean(trt_dit_times)
    report = {
        "metadata": metadata,
        "policy_action_shape": list(trt_action.shape),
        "zmq_action_shape": list(trt_action.shape[1:]),
        "parity": {
            "eager_vs_compile": compile_parity or {"passed": False, "error": compile_error},
            "eager_vs_tensorrt": trt_parity,
        },
        "policy_latency": {
            "eager": stats(eager_times),
            "compile": stats(compile_times) if compile_times else {"error": compile_error},
            "tensorrt": stats(trt_times),
        },
        "component_latency": {
            "denoising_steps": denoising_steps,
            "eager_dit_one_step_ms": round(eager_dit_mean, 3),
            "tensorrt_dit_one_step_ms": round(trt_dit_mean, 3),
            "eager_action_head_estimated_ms": round(eager_dit_mean * denoising_steps, 3),
            "tensorrt_action_head_estimated_ms": round(trt_dit_mean * denoising_steps, 3),
            "eager_fixed_backbone_processing_estimated_ms": round(
                max(0.0, eager_mean - eager_dit_mean * denoising_steps), 3
            ),
            "tensorrt_fixed_backbone_processing_estimated_ms": round(
                max(0.0, trt_mean - trt_dit_mean * denoising_steps), 3
            ),
        },
    }
    report["validation"] = {
        "engine_loaded": True,
        "checkpoint_gpu_runtime_fingerprint": True,
        "four_policy_cameras_640x480": (
            metadata.get("cameras") == list(CAMERAS)
            and metadata.get("camera_shape") == [480, 640, 3]
        ),
        "zmq_contract_16x28": report["zmq_action_shape"] == [16, 28],
        "compile_parity": bool(compile_parity and compile_parity["passed"]),
        "tensorrt_parity": bool(trt_parity["passed"]),
    }
    report["validation"]["passed"] = all(report["validation"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("export", "build", "validate", "benchmark", "all"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", default="/data/optimized")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--task", default="stack the blocks")
    parser.add_argument("--workspace-mb", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mean-tolerance", type=float, default=0.005)
    parser.add_argument("--max-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ONNX/TensorRT-Optimierung braucht eine CUDA-GPU")
    fingerprint = checkpoint_fingerprint(args.model_path)
    output_dir = Path(args.output_root) / fingerprint[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / "latest.json").write_text(
        json.dumps({"fingerprint": fingerprint, "artifact_dir": str(output_dir)}, indent=2),
        encoding="utf-8",
    )
    onnx_path = output_dir / "dit_model.onnx"
    engine_path = output_dir / "dit_model_bf16.trt"
    metadata_path = output_dir / "export_metadata.json"
    report_path = output_dir / "benchmark_report.json"

    policy = wrapped = obs = captured = None
    if args.phase in {"export", "all", "validate", "benchmark"}:
        policy, wrapped = load_policy(args.model_path, args.embodiment_tag)
        obs = synthetic_observation(args.width, args.height, args.task)

    if args.phase in {"export", "all"}:
        captured = capture_inputs(policy, wrapped, obs)
        inputs = export_onnx(policy, captured, onnx_path)
        base_meta = {
            "format_version": 1,
            "checkpoint_path": str(Path(args.model_path).resolve()),
            "checkpoint_fingerprint": fingerprint,
            "gpu": gpu_fingerprint(torch),
            "onnx_opset": 19,
            "precision": "bf16",
            "camera_shape": [args.height, args.width, 3],
            "cameras": list(CAMERAS),
            "task": args.task,
            "inputs": inputs,
            "tensorrt": None,
        }
        metadata_path.write_text(json.dumps(base_meta, indent=2), encoding="utf-8")
        print(f"[optimize] ONNX: {onnx_path}", flush=True)

    if args.phase in {"build", "all"}:
        if not onnx_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Zuerst Phase 'export' ausfuehren")
        trt_version = build_engine(onnx_path, engine_path, args.workspace_mb)
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        meta["gpu"] = gpu_fingerprint(torch)
        meta["tensorrt"] = trt_version
        meta["engine_size_bytes"] = engine_path.stat().st_size
        metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[optimize] TensorRT: {engine_path}", flush=True)

    if args.phase in {"validate", "benchmark", "all"}:
        import onnx

        onnx.checker.check_model(str(onnx_path))
        if policy is None:
            policy, wrapped = load_policy(args.model_path, args.embodiment_tag)
            obs = synthetic_observation(args.width, args.height, args.task)
        report = validate_and_benchmark(
            policy,
            wrapped,
            obs,
            args.model_path,
            engine_path,
            args.iterations,
            args.warmup,
            args.mean_tolerance,
            args.max_tolerance,
        )
        report["artifact_dir"] = str(output_dir)
        report["validation"]["onnx_checker"] = True
        report["validation"]["passed"] = all(
            value for key, value in report["validation"].items() if key != "passed"
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print(f"[optimize] Report: {report_path}", flush=True)
        if not report["validation"]["passed"]:
            failed = [
                key for key, value in report["validation"].items()
                if key != "passed" and not value
            ]
            raise RuntimeError(
                "Optimierungsvalidierung fehlgeschlagen; Bericht wurde geschrieben: "
                + ", ".join(failed)
            )

    print("[optimize] fertig", flush=True)


if __name__ == "__main__":
    main()
