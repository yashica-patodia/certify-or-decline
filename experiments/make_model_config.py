#!/usr/bin/env python3
"""Render one explicit all-role provider config from the frozen model matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import shlex
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "experiments/model_matrix.yaml"
# Slowest single-stream decode rate we are willing to plan around, in output
# tokens/second. Measured on the smallest GPU this study can serve from: vLLM
# 0.25.1, Qwen3-8B bf16, 1x L4 24GB sustained 16.3-16.6 tok/s across the whole
# 2026-07-30 smoke (`runs/aws-smoke-20260731/serve-qwen3-8b.out`, n=350 report
# lines). Bigger GPUs are strictly faster, so this floor is conservative for
# every other row in the matrix.
SLOWEST_DECODE_TOKENS_PER_SECOND = 16.0

# How much of the context window must stay available for the prompt. The agent
# loop replays the whole conversation every turn, so the prompt grows while
# `max_tokens` stays fixed: the binding constraint is prompt + max_tokens <=
# context_length at the LAST turn, not the first. Measured 2026-07-31 on 1x L40S
# serving Qwen3-14B at max_model_len 32768: max_tokens 8192 (a quarter of the
# window) ran 61 turns clean, while max_tokens 16384 (half) died mid-run with
# `HTTP 400 ... you requested 16384 output tokens and your prompt contains at
# least 16385 input tokens`. Reserving three quarters for history is therefore
# the validated floor, not a guess.
MIN_CONTEXT_TO_MAX_TOKENS_RATIO = 4
# Every role in configs/defaults.yaml. A role omitted here is NOT a harmless
# gap: load_config dict-merges role entries onto defaults.yaml, so a generated
# config that lacks a role silently inherits the default entry -- i.e. runs that
# role on Opus/Claude while the rest of the study runs on the model under test.
# `test_roles_match_defaults_yaml` pins this against defaults.yaml.
ROLES = (
    "solver",
    "interpreter",
    "formalizer",
    "citation",
    "problem_given",
    "computation",
    # Deciding judge under `_verifier_mode: step_wise_untyped` only, but emitted
    # for every model so the untyped arm runs on the same model as the typed
    # arms it is compared against.
    "step_untyped",
    "pedantry",
    "convention_lift",
    "initial_state",
)


def max_tokens_ceiling(context_length: int) -> int:
    """Largest `max_tokens` that still leaves room for a growing transcript."""
    return context_length // MIN_CONTEXT_TO_MAX_TOKENS_RATIO


def _assert_max_tokens_leaves_room_for_history(
    max_tokens: int, context_length: int
) -> None:
    """Reject a `max_tokens` that a multi-turn run cannot sustain.

    The failure this prevents is not a truncated answer -- it is the provider
    rejecting the request outright, mid-run, once accumulated history pushes
    prompt + max_tokens past the context window. That surfaces as an HTTP 400
    and kills the problem, so it is worth failing at config-generation time
    instead.
    """
    ceiling = max_tokens_ceiling(context_length)
    if max_tokens > ceiling:
        raise SystemExit(
            f"max-tokens {max_tokens} is too large for context length "
            f"{context_length}: the agent loop replays the transcript every "
            f"turn, so leave at least "
            f"{MIN_CONTEXT_TO_MAX_TOKENS_RATIO - 1}/"
            f"{MIN_CONTEXT_TO_MAX_TOKENS_RATIO} of the window for history "
            f"(max {ceiling}). Raise --context-length or lower --max-tokens."
        )


def build_config(args: argparse.Namespace) -> dict:
    matrix_bytes = MATRIX_PATH.read_bytes()
    matrix = yaml.safe_load(matrix_bytes)
    try:
        spec = matrix["models"][args.model_key]
    except KeyError as exc:
        choices = ", ".join(sorted(matrix.get("models") or {}))
        raise SystemExit(f"unknown model key {args.model_key!r}; choose one of: {choices}") from exc

    reference = matrix["serving_reference"]
    dtype = reference["dtype"] if args.dtype is None else args.dtype
    context_length = (
        reference["max_model_len"]
        if args.context_length is None else args.context_length
    )
    max_num_seqs = (
        reference["max_num_seqs"]
        if args.max_num_seqs is None else args.max_num_seqs
    )
    engine_seed = (
        reference["engine_seed"]
        if args.engine_seed is None else args.engine_seed
    )
    # Read quantization from the matrix (the source of truth) rather than
    # inferring it from the model-key string; default to "none" if unset.
    quantization = (
        args.quantization if args.quantization is not None
        else spec.get("quantization", "none")
    )
    system_role = args.system_role or spec.get("system_role", "system")
    served_model_name = (
        args.served_model_name
        or f"{spec['hf_id']}@{spec['revision']}"
    )
    tensor_parallel_size = (
        spec["tensor_parallel_size"]
        if args.tensor_parallel_size is None else args.tensor_parallel_size
    )
    common = {
        "backend": "reasoning_agent",
        "model": served_model_name,
        "model_source": "huggingface",
        "model_revision": spec["revision"],
        "dtype": dtype,
        "quantization": quantization,
        "server_hardware": args.server_hardware,
        "endpoint": args.endpoint,
        "context_length": context_length,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "system_role": system_role,
        "allow_shell": True,
        "allow_host_tools": False,
        # Emitted explicitly: agent_loop and llm both default this to 180s, which
        # is far below the time a `max_tokens`-length reasoning turn needs on a
        # small GPU. Leaving it unset made 4 of 5 problems in the 2026-07-30
        # smoke die as `chat completion timed out` after 4 x 180s, before the
        # harness saw a single token. See `_assert_request_timeout_fits`.
        "request_timeout": args.request_timeout,
        "tool_timeout": args.tool_timeout,
        "tool_output_chars": args.tool_output_chars,
    }
    if args.top_k is not None:
        common["top_k"] = args.top_k
    if args.min_p is not None:
        common["min_p"] = args.min_p
    if args.reasoning_effort:
        common["reasoning_effort"] = args.reasoning_effort
    # A self-hosted vLLM endpoint needs no credential; an authenticated endpoint
    # does. agent_loop reads the NAME of an env var (never the secret itself), so
    # the frozen config stays safe to commit and release.
    if args.api_key_env:
        common["api_key_env"] = args.api_key_env
    if spec.get("mode") == "hybrid_thinking_enabled":
        # serve_vllm passes --default-chat-template-kwargs server-side, so this
        # request-side copy is belt-and-braces for the self-hosted path. Several
        # hosted OpenAI-compatible providers reject the field outright with HTTP
        # 400 ("Extra inputs are not permitted"), which would fail every call, so
        # it is omitted when the endpoint is authenticated/remote.
        if not args.api_key_env:
            common["chat_template_kwargs"] = {"enable_thinking": True}

    return {
        "_experiment_model": {
            "matrix_key": args.model_key,
            "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
            "track": spec["track"],
            "hf_id": spec["hf_id"],
            "revision": spec["revision"],
            "license": spec["license"],
            "quantization": quantization,
            "reasoning_parser": spec.get("reasoning_parser"),
            "chat_format": spec.get("chat_format"),
            "mode": spec.get("mode"),
            "serving_backend": "vllm",
            "serving_backend_version": reference["version"],
            "server": {
                "served_model_name": served_model_name,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_num_seqs": max_num_seqs,
                "max_model_len": context_length,
                "engine_seed": engine_seed,
                "endpoint": args.endpoint,
            },
        },
        **{role: copy.deepcopy(common) for role in ROLES},
    }


def _assert_server_command_matches(config: dict, command_path: Path) -> None:
    """Fail if the generated config disagrees with a recorded serve_vllm command,
    so the serve/config reproducibility invariant is a check, not a prose rule."""
    tokens = shlex.split(command_path.read_text())
    flags: dict[str, str] = {}
    for name, value in zip(tokens, tokens[1:]):
        if name.startswith("--") and not value.startswith("--"):
            flags[name] = value
    server = config["_experiment_model"]["server"]
    expected = {
        "--served-model-name": str(server["served_model_name"]),
        "--tensor-parallel-size": str(server["tensor_parallel_size"]),
        "--dtype": str(config["solver"]["dtype"]),
        "--max-model-len": str(server["max_model_len"]),
        "--max-num-seqs": str(server["max_num_seqs"]),
        "--gpu-memory-utilization": str(server["gpu_memory_utilization"]),
        "--seed": str(server["engine_seed"]),
    }
    mismatched = {
        flag: {"config": value, "serve_command": flags.get(flag)}
        for flag, value in expected.items()
        if flags.get(flag) != value
    }
    if mismatched:
        raise SystemExit(
            f"generated config disagrees with {command_path}: {mismatched}"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("model_key")
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--served-model-name")
    result.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    result.add_argument(
        "--api-key-env",
        help=(
            "Name of the environment variable holding the endpoint's API key "
            "(never the key itself). Set this only for an authenticated/remote "
            "endpoint -- such a row is a separately-frozen appendix condition, "
            "not a matrix row, because weight revision, quantization, and "
            "serving stack are not attestable through a hosted API."
        ),
    )
    result.add_argument("--server-hardware", required=True)
    result.add_argument(
        "--server-command-file", type=Path,
        help="assert the generated config matches this recorded serve_vllm command",
    )
    result.add_argument("--dtype")
    result.add_argument("--quantization")
    result.add_argument("--tensor-parallel-size", type=int)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    result.add_argument("--max-num-seqs", type=int)
    result.add_argument("--engine-seed", type=int)
    result.add_argument("--context-length", type=int)
    result.add_argument("--max-turns", type=int, default=16)
    result.add_argument("--max-tokens", type=int, default=8192)
    result.add_argument("--temperature", type=float, required=True)
    result.add_argument("--top-p", type=float, required=True)
    result.add_argument("--top-k", type=int)
    result.add_argument("--min-p", type=float)
    result.add_argument("--seed", type=int, default=20260718)
    result.add_argument("--system-role", choices=("system", "user"))
    result.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    result.add_argument("--request-timeout", type=float, default=900)
    result.add_argument("--tool-timeout", type=float, default=120)
    result.add_argument("--tool-output-chars", type=int, default=12000)
    return result


def required_request_timeout(max_tokens: int) -> float:
    """Seconds a `max_tokens`-length turn needs on the slowest supported GPU."""
    return max_tokens / SLOWEST_DECODE_TOKENS_PER_SECOND


def _assert_request_timeout_fits(request_timeout: float, max_tokens: int) -> None:
    """Reject a timeout that cannot cover one full-length generation.

    A model is allowed to emit `max_tokens` tokens, so a timeout below the time
    that takes is not a safety margin -- it is a guaranteed failure for exactly
    the longest (most reasoning-heavy) turns, and it fails them *without usage
    accounting*, so the run reports no tokens and no tool calls rather than a
    result. That is what made the 2026-07-30 smoke look like a protocol failure.
    """
    required = required_request_timeout(max_tokens)
    if request_timeout < required:
        raise SystemExit(
            f"request-timeout {request_timeout:g}s cannot cover max-tokens "
            f"{max_tokens} at {SLOWEST_DECODE_TOKENS_PER_SECOND:g} tokens/s "
            f"(needs >= {required:.0f}s). Raise --request-timeout or lower "
            "--max-tokens."
        )


def main() -> None:
    args = parser().parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    config = build_config(args)
    if args.server_command_file is not None:
        _assert_server_command_matches(config, args.server_command_file)
    server = config["_experiment_model"]["server"]
    solver = config["solver"]
    if (
        solver["context_length"] < 1
        or args.max_turns < 1
        or args.max_tokens < 1
        or args.request_timeout <= 0
        or server["tensor_parallel_size"] < 1
        or server["max_num_seqs"] < 1
    ):
        raise SystemExit(
            "context length, max turns, max tokens, request timeout, tensor "
            "parallel size, and max sequences must be positive"
        )
    _assert_request_timeout_fits(args.request_timeout, args.max_tokens)
    _assert_max_tokens_leaves_room_for_history(
        args.max_tokens, solver["context_length"]
    )
    if not 0 <= args.temperature or not 0 < args.top_p <= 1:
        raise SystemExit("temperature must be nonnegative and top-p must be in (0, 1]")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("gpu-memory-utilization must be in (0, 1]")
    if (args.top_k is not None and args.top_k < 1) or (
        args.min_p is not None and not 0 <= args.min_p <= 1
    ):
        raise SystemExit("top-k must be >= 1 and min-p must be in [0, 1]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.out)


if __name__ == "__main__":
    main()
