#!/usr/bin/env python3
# TL;DR: GR00T-ZMQ-Server im Container mit optionalem torch.compile-/TensorRT-Backend fuer den DiT.
"""GR00T-ZMQ-Server mit opt-in torch.compile-/TensorRT-DiT-Backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from groot_inference_backend import checkpoint_fingerprint, install_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--backend", choices=("eager", "compile", "tensorrt"), default="eager")
    parser.add_argument("--engine-path", default="")
    parser.add_argument("--optimized-root", default="/data/optimized")
    args = parser.parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    model_path = str(Path(args.model_path).resolve())
    engine_path = args.engine_path
    if args.backend == "tensorrt" and not engine_path:
        fingerprint = checkpoint_fingerprint(model_path)[:16]
        engine_path = str(Path(args.optimized_root) / fingerprint / "dit_model_bf16.trt")

    print("Starting optimized GR00T inference server...", flush=True)
    print(f"  Model:   {model_path}", flush=True)
    print(f"  Backend: {args.backend}", flush=True)
    print(f"  Engine:  {engine_path or '(keine)'}", flush=True)

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        model_path=model_path,
        device=args.device,
        strict=True,
    )
    details = install_backend(policy, args.backend, engine_path or None, model_path=model_path)
    print("  Backend-Details: " + json.dumps(details, default=str), flush=True)
    wrapped = Gr00tSimPolicyWrapper(policy)
    PolicyServer(policy=wrapped, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
