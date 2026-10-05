#!/usr/bin/env python3
"""Recover the answer-slot violation counts by re-applying the live validators.

WHY THIS EXISTS

Section 2 reported that 238 of 248 formalizer-invalid attempts were answer-slot
violations, split 118 / 73 / 47 across three categories. That split could not be
checked: `reject_reason` is populated only for `formalizer_reject` attempts,
never for `formalizer_invalid`, and the pipeline artifacts for that run are
gone.

But the formalizer's own responses survive in the run file, and the validators
that judge them are still committed in `pipeline.py`. So the classification can
be re-derived by running the real rules over the real outputs. This script does
that, importing nothing from the analysis it is checking.

WHAT REPRODUCES AND WHAT DOES NOT

The dominant category does, to within one: 117 F#-labelled answer slots against
the 118 reported. That is the substantive finding -- the formalizer writes a
fact label where the answer belongs, often enough to dominate every other
failure.

The other two do not. `pipeline.py` has no "option-related" validator at all, so
that category cannot be reconstructed from the deployed rules; and the
denominators differ (proof responses are not 1:1 with invalid attempts, because
a retry produces another response). Those two counts are therefore withdrawn
rather than restated.

Usage:
    python -m experiments.answer_slot_violations
"""
from __future__ import annotations

import argparse
import collections
import json
import re

# Both patterns are copied from pipeline.py's formalizer validator. They are
# duplicated rather than imported so this script checks the rule as written,
# and would not silently follow a later edit to it.
UNRESOLVED_SLOT = re.compile(
    r"[<\[({]*\s*(?:final\s+)?(?:answer|result)\s*[>\])}]*"
    r"\s*(?:(?:=|:)\s*(?:\?|tbd|todo|pending|unknown|[_-]*)?)?\s*",
    re.IGNORECASE,
)
F_LABEL = re.compile(r"^F\d+\s*:", re.IGNORECASE)


def final_answer_slot(payload: dict) -> tuple[str | None, str]:
    """(final state[0], initial goal). The goal is needed: pipeline.py rejects a
    slot that merely echoes it, and omitting that condition is what made an
    earlier version of this script under-count by 51."""
    proof = payload.get("proof") or payload
    steps = proof.get("steps") or []
    if not steps:
        return None, ""
    state = steps[-1].get("state") or []
    initial = proof.get("initial_state")
    if not isinstance(initial, list) or not initial:
        initial = steps[0].get("state") or []
    goal = str(initial[0]).strip() if initial else ""
    return (str(state[0]).strip() if state else None), goal


def classify(run_path: str) -> dict:
    counts: collections.Counter = collections.Counter()
    attempts = collections.Counter()
    examined = 0

    for row in json.load(open(run_path)):
        for attempt in (row.get("attempts") or []):
            attempts[str(attempt.get("phase"))] += 1
        for call in (row.get("calls") or []):
            if call.get("role") != "formalizer" or not call.get("response"):
                continue
            try:
                payload = json.loads(str(call["response"]))
            except Exception:
                counts["unparseable response"] += 1
                continue
            if payload.get("action") != "proof":
                continue
            examined += 1
            slot, goal = final_answer_slot(payload)
            if not slot:
                counts["no answer in state[0]"] += 1
            elif F_LABEL.match(slot):
                counts["F# fact label in the answer slot"] += 1
            elif (slot in {"?", "_", "-"}
                  or UNRESOLVED_SLOT.fullmatch(slot)
                  or (goal and slot.casefold() == goal.casefold())):
                counts["answer slot left unresolved"] += 1
            else:
                counts["passes all three committed validators"] += 1

    return {
        "run_file": run_path.split("/")[-1],
        "formalizer_invalid_attempts": attempts.get("formalizer_invalid", 0),
        "formalizer_reject_attempts": attempts.get("formalizer_reject", 0),
        "proof_responses_examined": examined,
        "by_violation": dict(counts),
        "note": ("Re-applies pipeline.py's committed answer-slot validators to "
                 "the formalizer responses stored in the run. The F#-label "
                 "count reproduces the reported 118 to within one. "
                 "The earlier 'unresolved placeholder' (73) and "
                 "'option-related' (47) counts are withdrawn: there is no "
                 "option-related validator in the pipeline, and proof responses "
                 "are not one-to-one with invalid attempts."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        default="runs/benchmark_gate-gpqa100-t1_20260803_184710_025287.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = classify(args.run)

    print(f"run: {rep['run_file']}\n")
    print(f"  formalizer_invalid attempts : {rep['formalizer_invalid_attempts']}")
    print(f"  formalizer_reject attempts  : {rep['formalizer_reject_attempts']}")
    print(f"  proof responses examined    : {rep['proof_responses_examined']}\n")
    for kind, n in sorted(rep["by_violation"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {kind}")
    print(f"\n{rep['note']}")

    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
