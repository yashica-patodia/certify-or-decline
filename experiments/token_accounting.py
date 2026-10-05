#!/usr/bin/env python3
"""Compare recorded verifier token usage with the cost of solver draws.

Output-only and input-plus-output matching yield different sampling budgets.
These are token accounting proxies. Heterogeneous models, serving hardware,
and unmeasured prefix-cache savings prevent a FLOP or latency match.

python -m experiments.token_accounting runs/strong-k3/benchmark_strong-t*.json
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys


def collect(paths: list[str]) -> dict:
    per_problem_out: list[int] = []
    per_problem_in: list[int] = []
    solver_out = solver_in = solver_calls = 0
    cache_read = cache_creation = 0

    for path in paths:
        for row in json.load(open(path)):
            metrics = row.get("metrics") or {}
            if not metrics.get("total_output_tokens"):
                continue
            per_problem_out.append(metrics["total_output_tokens"])
            per_problem_in.append(metrics.get("total_input_tokens") or 0)
            cache_read += metrics.get("total_cache_read_input_tokens") or 0
            cache_creation += metrics.get("total_cache_creation_input_tokens") or 0
            calls_by_role = metrics.get("calls_by_role") or {}
            for role, usage in (metrics.get("usage_by_role") or {}).items():
                if "solver" not in role.lower() or not isinstance(usage, dict):
                    continue
                solver_out += usage.get("output_tokens") or 0
                solver_in += usage.get("input_tokens") or 0
                solver_calls += calls_by_role.get(role) or 0

    if not per_problem_out or not solver_calls:
        raise SystemExit("no usage telemetry found in these runs")

    verifier_out = st.mean(per_problem_out)
    verifier_in = st.mean(per_problem_in)
    draw_out = solver_out / solver_calls
    draw_in = solver_in / solver_calls
    return {
        "n_problems": len(per_problem_out),
        "verifier_output_per_problem": round(verifier_out),
        "verifier_input_per_problem": round(verifier_in),
        "verifier_total_per_problem": round(verifier_out + verifier_in),
        "solver_draw_output": round(draw_out),
        "solver_draw_input": round(draw_in),
        "solver_draw_total": round(draw_out + draw_in),
        "k_matched_on_output": round(verifier_out / draw_out, 2),
        "k_matched_on_total": round((verifier_out + verifier_in) / (draw_out + draw_in), 2),
        "cache_tokens_recorded": cache_read + cache_creation,
        "note": 'Output-only and input-plus-output matching are token accounting proxies, not lower and upper bounds on matched compute. Model sizes, serving hardware, and prefix-cache savings are not controlled by these counts; the recorded cache counters do not measure vLLM cache savings.',
    }


def main() -> None:
    paths: list[str] = []
    for arg in sys.argv[1:]:
        paths.extend(sorted(glob.glob(arg)))
    if not paths:
        raise SystemExit("give one or more run files")
    acc = collect(paths)

    print(f"problems with usage telemetry: {acc['n_problems']}\n")
    print(f"{'':22s} {'output':>10} {'input':>10} {'total':>10}")
    print(f"{'Verifier, per problem':22s} {acc['verifier_output_per_problem']:>10,} "
          f"{acc['verifier_input_per_problem']:>10,} {acc['verifier_total_per_problem']:>10,}")
    print(f"{'one solver draw':22s} {acc['solver_draw_output']:>10,} "
          f"{acc['solver_draw_input']:>10,} {acc['solver_draw_total']:>10,}")
    print(f"\nk matched on output tokens : {acc['k_matched_on_output']:.2f}"
          f"  -> k = {round(acc['k_matched_on_output'])}   (output-token proxy)")
    print(f"k matched on total  tokens : {acc['k_matched_on_total']:.2f}"
          f"  -> k = {round(acc['k_matched_on_total'])}   (total-token proxy)")
    if not acc["cache_tokens_recorded"]:
        print("\ncache counters are zero: prefix-cache savings are not measurable "
              "from these records.")
    print(f"\n{acc['note']}")


if __name__ == "__main__":
    main()
