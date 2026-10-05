#!/usr/bin/env python3
"""Join each seed's certified answer to the vote class it falls into.

The votes files say what the sampled draws agreed on; the run records say what
the verifier certified. Neither alone answers whether a wrong certification is
the *same* wrong answer the vote chose. This emits the join, keyed by problem,
one entry per seed and budget.

    python -m experiments.certified_answer_classes \\
        --out experiments/public_runs/certified_answer_classes.json

Class ids are meaningful only against the votes file named in the field:
`k10_class` indexes solver_k10_votes.json, `k30_class` indexes
solver_k30_votes.json, and the two are numbered independently, so the same id
in each does not imply the same answer. The numbering comes from
solver_votes.class_map rather than being rebuilt here.

Field contract, so that a null is never ambiguous:

    status        certified | declined | execution_error
    answer_form   option_letter | option_text | free_form
                  (null unless status=certified)
    k10_class     int   the vote class this answer falls into
                  -1    an option letter no draw at that budget produced
                  null  no class assignable: either status is not certified,
                        or the answer is free_form

The free_form case is not a defect and is not rare: the baseline votes are
option letters, while the verifier answers in the problem's own terms -- a
quantity or an expression rather than a letter. 45 of the 105 certified
observations are of that form
and cannot be placed against a letter class at all. The join is therefore
partial by construction, and the counts block records exactly how partial.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

from experiments.gpqa_common import options_from_question
from experiments.solver_votes import class_map
from experiments.summarize_gpqa import extract_choice

RUNS = "runs/strong-k3/benchmark_strong-t*.json"
BUDGETS = {"k10_class": "runs/k30/solver_k10_base.json",
           "k30_class": "runs/k30/solver_k30_merged.json"}


def seed_of(path: str) -> str:
    """t1 / t2 / t3 from the run filename."""
    name = os.path.basename(path)
    marker = name.index("-t") + 2
    return "t" + name[marker]


def normalise(text: str) -> str:
    """Conservative form for comparing an answer against an option's text.

    Only differences that cannot change which option is named are removed:
    surrounding whitespace, thousands separators, currency and LaTeX wrappers,
    case, and a trailing full stop. Nothing is reordered and no token is
    dropped, so two different options cannot normalise together.
    """
    text = re.sub(r"\\[a-zA-Z]+|[{}$,\s]", "", str(text))
    return text.strip().lower().rstrip(".")


def option_named(answer: str, options: dict | None) -> str | None:
    """The single option whose text the answer reproduces, if exactly one does."""
    body = re.sub(r"^\s*answer\s*[:=]\s*", "", answer, flags=re.I)
    target = normalise(body)
    if not target:
        return None
    hit = [letter for letter, text in (options or {}).items()
           if normalise(text) and normalise(text) == target]
    return hit[0] if len(hit) == 1 else None


def classify(row: dict, options: dict | None, maps: dict) -> dict:
    if row.get("error"):
        return {"status": "execution_error", "answer_form": None,
                **{f: None for f in BUDGETS}}
    if not row.get("verified"):
        return {"status": "declined", "answer_form": None,
                **{f: None for f in BUDGETS}}
    # The run's answer is already the extracted final answer, so it needs the
    # same last normalisation step a draw gets, not the envelope parsing.
    answer = str(row.get("answer") or "")
    normalised = extract_choice(answer, options)
    form = "option_letter"
    if not normalised:
        normalised = option_named(answer, options)
        form = "option_text"
    if not normalised:
        return {"status": "certified", "answer_form": "free_form",
                **{f: None for f in BUDGETS}}
    normalised = normalised.upper()
    out = {"status": "certified", "answer_form": form}
    for field, classes in maps.items():
        out[field] = classes.get(normalised, -1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--benchmark", default="runs/benchmarks/gpqa-fixed-100.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.runs))
    if not paths:
        raise SystemExit(
            f"no run files matched {args.runs!r}, so there is nothing to join. "
            f"Refusing to write an empty artifact. This needs the raw run "
            f"corpus, which is not redistributable; the published result is "
            f"already committed at "
            f"experiments/public_runs/certified_answer_classes.json.")

    loaded = json.load(open(args.benchmark))
    bench = {str(p["id"]): p for p in
             (loaded["problems"] if isinstance(loaded, dict) else loaded)}
    draws = {f: json.load(open(p)) for f, p in BUDGETS.items()}

    out: dict[str, dict] = {}
    counts = {"certified": 0, "declined": 0, "execution_error": 0,
              "by_letter": 0, "by_option_text": 0, "free_form": 0,
              "classified": 0, "novel_answer": 0}
    for path in paths:
        seed = seed_of(path)
        for row in json.load(open(path)):
            pid = str(row.get("id"))
            try:
                options = options_from_question(bench.get(pid, {}).get("question"))
            except Exception:
                options = None
            maps = {f: class_map(draws[f].get(pid, {}),
                                 bench.get(pid, {}).get("question"))
                    for f in BUDGETS}
            rec = classify(row, options, maps)
            out.setdefault(pid, {})[seed] = rec
            counts[rec["status"]] += 1
            if rec["answer_form"] == "free_form":
                counts["free_form"] += 1
            elif rec["answer_form"] in ("option_letter", "option_text"):
                counts["by_letter" if rec["answer_form"] == "option_letter"
                       else "by_option_text"] += 1
                counts["classified"] += 1
                if any(rec[f] == -1 for f in BUDGETS):
                    counts["novel_answer"] += 1

    report = {"counts": counts, "seeds": sorted({seed_of(p) for p in paths}),
              "note": ("Class ids index the votes file named in the field and "
                       "are numbered independently per budget. -1 is an option "
                       "letter no draw produced; null is no assignable class, "
                       "which for status=certified means answer_form=free_form. "
                       "The baseline votes are option letters and the verifier "
                       "answers in the problem's own terms; an answer that "
                       "repeats exactly one option's text names that option and "
                       "is resolved as option_text. What stays free_form is text "
                       "no option text equals."),
              "by_problem": out}

    print(" ".join(f"{k}={v}" for k, v in counts.items()))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
