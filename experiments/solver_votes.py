#!/usr/bin/env python3
"""Redact a raw k-draw solver sample into the published votes schema.

`public_runs/solver_k10_votes.json` was committed as data in 2456823 with no
generator, so this reconstructs one and is validated against it: run with
--k 10 it must reproduce that file byte-for-byte. That check is the reason to
trust the k=30 file it also produces.

What crosses the redaction line and what does not: an answer stored against a
problem id IS the answer key, so answers never appear. Each distinct answer
becomes an anonymous class id in first-seen order, which is enough to ask
whether draws agree with each other and whether the majority is right, and not
enough to recover what any of them said.

    python -m experiments.solver_votes \\
        --draws runs/k30/solver_k10_base.json --k 10 \\
        --out experiments/public_runs/solver_k10_votes.json
"""
from __future__ import annotations

import argparse
import json

from experiments.gpqa_common import options_from_question
from experiments.self_consistency_curve import extract_answer


def answer_of(draw: dict, question: str | None) -> str | None:
    """The draw's final answer, or None if it produced none.

    Delegates to self_consistency_curve.extract_answer rather than parsing
    envelopes here. That function is what produced the committed k=30 curve and
    therefore every self-consistency figure in the write-up, and it is more
    careful than an obvious reimplementation: it takes the LAST well-formed
    envelope (earlier ones are superseded), falls back to a loose match for
    envelopes that are not valid JSON, and reads only the head of the response
    so a run-on generation cannot smuggle in a later letter. A second parser
    here would put two definitions of "answered" in one bundle, which is the
    defect this file exists to avoid.
    """
    if draw.get("error"):
        return None
    try:
        options = options_from_question(question)
    except Exception:
        options = None
    answer = extract_answer(draw.get("text") or "", options)
    return answer.upper() if answer else None


def class_map(draws: dict, question: str | None) -> dict[str, int]:
    """Answer -> anonymous class id, in first-seen order over the draws.

    Exposed so that anything joining a run's answer to these classes uses this
    numbering rather than rebuilding it. Two implementations of "which class is
    this answer" is the same defect as two implementations of "did this draw
    answer", which is what 3e256a4 removed.
    """
    classes: dict[str, int] = {}
    for slot in sorted(draws, key=int):
        answer = answer_of(draws[slot], question)
        if answer is not None:
            classes.setdefault(answer, len(classes))
    return classes


def votes_for(draws: dict, question: str | None, key: str | None) -> dict:
    """One problem's draws as anonymous classes plus per-draw correctness."""
    classes: dict[str, int] = {}
    ids: list[int | None] = []
    correct: list[bool | None] = []
    for slot in sorted(draws, key=int):
        ans = answer_of(draws[slot], question)
        if ans is None:
            ids.append(None)
            correct.append(None)
            continue
        # First-seen order, so the ids carry no information about the answers
        # themselves -- only about which draws agree.
        ids.append(classes.setdefault(ans, len(classes)))
        correct.append(bool(key) and ans == key.strip().upper())
    return {"draw_classes": ids, "draw_correct": correct,
            "k": len(ids), "n_answered": sum(1 for i in ids if i is not None)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", required=True)
    ap.add_argument("--benchmark", default="runs/benchmarks/gpqa-fixed-100.json")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.k < 1:
        ap.error("--k must be positive")

    loaded = json.load(open(args.benchmark))
    # The committed benchmark is {metadata, problems}; a bare list is accepted
    # so a caller can pass a slice without reshaping it.
    rows = loaded["problems"] if isinstance(loaded, dict) else loaded
    bench = {str(p.get("id")): p for p in rows}
    raw = json.load(open(args.draws))
    if set(raw) != set(bench):
        ap.error("draw IDs differ from the benchmark cohort")

    out = {}
    for pid, draws in raw.items():
        if set(draws) != {str(i) for i in range(args.k)}:
            raise SystemExit(
                f"{pid} has {len(draws)} draws, not the --k {args.k} declared. "
                f"A votes file whose k does not match its draws would report a "
                f"denominator the data cannot support.")
        p = bench.get(pid) or {}
        out[pid] = votes_for(draws, p.get("problem") or p.get("question"),
                             str(p.get("expected") or p.get("answer") or ""))

    answered = sum(v["n_answered"] for v in out.values())
    print(f"problems {len(out)}  draws {len(out) * args.k}  answered {answered}"
          f"  unparsed {len(out) * args.k - answered}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
