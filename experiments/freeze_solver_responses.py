#!/usr/bin/env python3
"""Freeze one run's initial solver responses into a replay map for the arms.

The process-vs-outcome ablation compares verification decisions across arms.
That comparison is only paired if every arm verified the SAME candidate answer,
but each arm re-solves in its own subprocess at temperature > 0, so in general
they do not. `experiments/pairing.py` detects the divergence after the fact;
this closes it in advance.

Workflow
--------
1. Run the solve ONCE (any arm, or a dedicated `--no-repair` run).
2. Freeze its initial responses:

       python -m experiments.freeze_solver_responses runs/solve-once.json \\
           --out runs/frozen/solver-responses.json

3. Point every arm at the frozen map, e.g. in each arm's config:

       _solver_replay:
         path: runs/frozen/solver-responses.json

4. `pipeline.solver_replay` then supplies the initial answer verbatim, so
   `solver_initial_sha256` is identical across arms BY CONSTRUCTION and
   `experiments.cross_arm` passes its `assert_paired` guard for a reason rather
   than by luck.

Why not greedy decoding
-----------------------
The backlog's alternative -- decode at temperature 0 and assert cross-arm byte
equality -- does not work on this stack. bf16 with tensor parallelism is not
bitwise reproducible even at temperature 0 (`matrix_spec.example.yaml` says so
in its own comment), so that route produces intermittent assertion failures
instead of a guarantee. Replay removes the dependency on serving determinism
entirely.

What this does NOT freeze
-------------------------
Only the FIRST solve. Re-solves inside the repair loop stay live and are
expected to diverge across arms -- that divergence is part of what the ablation
measures, and `--target solver_initial` (the cross-arm comparator) grades the
first response alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def extract(run_path: str | Path) -> dict[str, str]:
    """{problem_id: first solver response} from one sealed run."""
    with Path(run_path).open() as handle:
        rows = json.load(handle)
    if isinstance(rows, dict):
        rows = [rows]

    table: dict[str, str] = {}
    skipped: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{run_path}: entry {index} is not an object")
        problem_id = row.get("id")
        if problem_id is None:
            raise ValueError(f"{run_path}: entry {index} has no 'id'")
        problem_id = str(problem_id)
        if problem_id in table:
            raise ValueError(f"{run_path}: duplicate problem id {problem_id!r}")
        solutions = row.get("solver_solutions")
        if not isinstance(solutions, list) or not solutions:
            # An errored problem never produced a first response. Recording an
            # empty string would hand every arm a degenerate input that looks
            # like a real answer, so omit it -- and make the omission loud,
            # because the arms will then refuse to run it rather than silently
            # solving it live and unpairing it.
            skipped.append(problem_id)
            continue
        first = solutions[0]
        if not isinstance(first, str) or not first.strip():
            skipped.append(problem_id)
            continue
        table[problem_id] = first

    if skipped:
        print(
            f"warning: {len(skipped)} problem(s) had no usable initial solver "
            f"response and are absent from the map: {', '.join(sorted(skipped)[:5])}"
            + (" ..." if len(skipped) > 5 else "")
        )
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="sealed run JSON to freeze")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--expect", type=int,
        help="fail unless exactly this many responses are frozen (cohort size)",
    )
    args = parser.parse_args(argv)

    if args.out.exists():
        # A replay map is an experimental input every downstream arm hashes.
        # Silently rewriting it would invalidate already-completed arms.
        raise SystemExit(f"refusing to overwrite existing replay map: {args.out}")

    table = extract(args.run)
    if args.expect is not None and len(table) != args.expect:
        raise SystemExit(
            f"expected {args.expect} frozen responses, got {len(table)}"
        )

    payload = json.dumps(table, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"{len(table)} responses -> {args.out}")
    print(f"sha256  {digest}")
    print("Record this hash with the condition inputs; every arm must cite it.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
