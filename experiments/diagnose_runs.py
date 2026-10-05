"""Measure construction failures, judge rejections and optional GPQA scoring agreement.

Inputs are full locally generated run lists. The output contains counts only.
The construction and rejection summaries require no inference; --gpqa-scoring
requires the separately generated final-answer grade files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.answer_slot_violations import classify
from experiments.build_public_runs import grader_map
from experiments.gpqa_common import options_from_question
from experiments.summarize_gpqa import extract_choice

JUDGE_ROLES = {"initial_state", "citation", "problem_given", "computation", "step_untyped"}


def diagnose(paths: list[Path], *, gpqa_scoring: bool = False) -> dict:
    construction = Counter()
    terminal_phases = Counter()
    rejection_roles = Counter()
    issue_classes = Counter()
    scoring = Counter()
    seen = set()
    for path in paths:
        if path.resolve() in seen:
            raise ValueError(f"run supplied twice: {path.name}")
        seen.add(path.resolve())
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"not a raw run list: {path.name}")
        grades = grader_map(str(path)) if gpqa_scoring else {}
        slots = classify(str(path))
        construction["proof_responses"] += slots["proof_responses_examined"]
        construction["labelled_answer_slots"] += slots["by_violation"].get("F# fact label in the answer slot", 0)
        for row in rows:
            construction["attempts"] += 1
            completed = not bool(row.get("error"))
            construction["completed"] += completed
            construction["execution_errors"] += not completed
            attempts = row.get("attempts") or []
            phase = attempts[-1].get("phase") if attempts else None
            judged = any(call.get("role") in JUDGE_ROLES for call in row.get("calls") or [])
            if completed:
                terminal_phases[str(phase)] += 1
                construction["completed_never_judged"] += not judged
                invalid = phase == "formalizer_invalid"
                construction["terminal_invalid"] += invalid
                construction["terminal_invalid_never_judged"] += invalid and not judged
            for verdict in row.get("verdicts") or []:
                if verdict.get("accepted") is False:
                    rejection_roles[str(verdict.get("role"))] += 1
                    for issue in verdict.get("issues") or []:
                        if isinstance(issue, dict):
                            issue_classes[str(issue.get("error_class"))] += 1
            if gpqa_scoring and completed and row.get("verified"):
                pid = str(row["id"])
                if not isinstance(grades.get(pid), bool):
                    raise ValueError(f"missing Boolean final-answer grade: {path.name}/{pid}")
                options = options_from_question(row.get("problem") or "")
                if not isinstance(options, dict) or set(options) != set("ABCD"):
                    raise ValueError(f"not a GPQA-style option question: {path.name}/{pid}")
                choice = extract_choice(row.get("answer"), options)
                scoring["certified_answers"] += 1
                if choice is None:
                    scoring["not_parsed"] += 1
                else:
                    scoring["parsed"] += 1
                    matches = choice == str(row.get("expected") or "").strip().upper()
                    scoring["agreement"] += matches == grades[pid]
                    scoring["disagreement"] += matches != grades[pid]
    return {"run_files": len(paths), "construction": dict(construction),
            "completed_terminal_phases": dict(terminal_phases),
            "rejections": sum(rejection_roles.values()), "rejections_by_role": dict(rejection_roles),
            "classified_issues": sum(issue_classes.values()), "issues_by_class": dict(issue_classes),
            "gpqa_parser_grader_agreement": dict(scoring) if gpqa_scoring else None,
            "note": "Judge presence uses call roles. Rejections use the run's recorded verdict list; parser/LLM agreement is not independently validated correctness."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--gpqa-scoring", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    report = diagnose(args.runs, gpqa_scoring=args.gpqa_scoring)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}: {report['construction']['attempts']} attempts")


if __name__ == "__main__":
    main()
