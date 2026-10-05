#!/usr/bin/env python3
"""Print or execute the pinned vLLM command for one matrix entry."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "experiments/model_matrix.yaml"


def build_command(args: argparse.Namespace) -> list[str]:
    matrix = yaml.safe_load(MATRIX_PATH.read_text())
    try:
        spec = matrix["models"][args.model_key]
    except KeyError as exc:
        choices = ", ".join(sorted(matrix.get("models") or {}))
        raise SystemExit(f"unknown model key {args.model_key!r}; choose one of: {choices}") from exc
    reference = matrix["serving_reference"]
    dtype = reference["dtype"] if args.dtype is None else args.dtype
    max_model_len = (
        spec.get("max_model_len", reference["max_model_len"])
        if args.max_model_len is None else args.max_model_len
    )
    max_num_seqs = (
        reference["max_num_seqs"]
        if args.max_num_seqs is None else args.max_num_seqs
    )
    engine_seed = (
        reference["engine_seed"]
        if args.engine_seed is None else args.engine_seed
    )
    served_model_name = (
        args.served_model_name
        or f"{spec['hf_id']}@{spec['revision']}"
    )
    command = [
        "vllm", "serve", spec["hf_id"],
        "--revision", spec["revision"],
        "--tokenizer-revision", spec["revision"],
        "--served-model-name", served_model_name,
        "--tensor-parallel-size", str(
            args.tensor_parallel_size
            if args.tensor_parallel_size is not None
            else spec["tensor_parallel_size"]
        ),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--generation-config", "vllm",
        "--seed", str(engine_seed),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if spec.get("reasoning_parser"):
        command.extend(("--reasoning-parser", spec["reasoning_parser"]))
    if spec.get("mode") == "hybrid_thinking_enabled":
        command.extend((
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": True}, separators=(",", ":")),
        ))
    if spec.get("vllm_loader") == "mistral":
        command.extend((
            "--tokenizer-mode", "mistral",
            "--config-format", "mistral",
            "--load-format", "mistral",
        ))
    return command


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("model_key")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--served-model-name")
    result.add_argument("--tensor-parallel-size", type=int)
    result.add_argument("--dtype")
    result.add_argument("--max-model-len", type=int)
    result.add_argument("--max-num-seqs", type=int)
    result.add_argument("--engine-seed", type=int)
    result.add_argument(
        "--batch-invariant",
        action="store_true",
        help=(
            "Export VLLM_BATCH_INVARIANT=1. This is an optional deployment "
            "change, not a guarantee of identical output across GPU hosts "
            "and not enabled by the paper's default command."
        ),
    )
    result.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8000)
    return result


def main() -> None:
    args = parser().parse_args()
    for name in ("tensor_parallel_size", "max_model_len", "max_num_seqs"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    command = build_command(args)
    # The serving ENVIRONMENT is part of the frozen serving reference, so print
    # it next to the command: a reader of command.txt must be able to see that
    # batch invariance was on. The previous revision of the cookbook required
    # this env var in prose while nothing set it, so runs claimed a determinism
    # property they did not have.
    prefix = "VLLM_BATCH_INVARIANT=1 " if args.batch_invariant else ""
    print(prefix + shlex.join(command), flush=True)
    if args.execute:
        if args.batch_invariant:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
