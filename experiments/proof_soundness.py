#!/usr/bin/env python3
"""Proof-soundness precision from the audit artifact, under explicit rules.

WHY THIS EXISTS

Section 8e reports that certified answers are 87.6% correct by key-match but
only **63.8%** sound by proof audit. That 24-point gap is the paper's most
damaging finding about its own system, and the 63.8% could not be re-derived
from `experiments/audit_proofs_strong_k3.json` under any rule:

    codex VALID / 105               71/105 = 67.6%
    neither grader rejects / 105    64/105 = 61.0%
    claude-else-codex VALID / 105   68/105 = 64.8%
    claude VALID / 57 graded        36/ 57 = 63.2%
    both graders VALID / 57 graded  32/ 57 = 56.1%

None is 63.8%. The number was computed once, by hand or by a script that was
not committed, and could not be checked afterwards.

THE COVERAGE PROBLEM THAT CAUSED IT

The two graders did not audit the same rows. Codex returned a verdict on all
105 certified proofs; claude returned one on **57**. Every headline count in
that section -- "20 with an unsupported step, 14 with a wrong one", "25 of 105
correct on a rejected proof" -- is codex's alone. The claim that "this is not
one grader's opinion" rests on the 57-row overlap, where the two agree at
kappa 0.57-0.70, and that is worth stating rather than implying full coverage.

WHAT THIS REPORTS

All four rules, so the reader sees the range rather than one number pulled from
it, plus the grader coverage that makes the range necessary. The qualitative
finding is robust to the choice: under every rule proof-soundness is far below
the 87.6% key-match precision, by 20 to 31 points.

Usage:
    python -m experiments.proof_soundness --out experiments/public_runs/proof_soundness.json
"""
from __future__ import annotations

import argparse
import collections
import json

REJECT = {"ERROR", "GAP", "NONSEQUITUR"}


def verdict(entry) -> str | None:
    return entry.get("verdict") if isinstance(entry, dict) else None


def analyse(path: str) -> dict:
    items = json.load(open(path))["items"]
    rows = list(items.values()) if isinstance(items, dict) else items

    graded_both = [r for r in rows
                   if verdict(r.get("claude")) and verdict(r.get("codex"))]
    coverage = {
        "n_certified_proofs": len(rows),
        "codex_returned_a_verdict": sum(1 for r in rows if verdict(r.get("codex"))),
        "claude_returned_a_verdict": sum(1 for r in rows if verdict(r.get("claude"))),
        "audited_by_both": len(graded_both),
    }

    def rule(name: str, numerator: int, denominator: int, note: str) -> dict:
        return {"rule": name, "sound": numerator, "of": denominator,
                "precision_pct": round(100 * numerator / denominator, 1),
                "note": note}

    codex_valid = sum(1 for r in rows if verdict(r.get("codex")) == "VALID")
    neither = sum(1 for r in rows
                  if verdict(r.get("codex")) not in REJECT
                  and verdict(r.get("claude")) not in REJECT)
    coalesce = sum(1 for r in rows
                   if (verdict(r.get("claude")) or verdict(r.get("codex"))) == "VALID")
    both_valid = sum(1 for r in graded_both
                     if verdict(r["claude"]) == "VALID"
                     and verdict(r["codex"]) == "VALID")

    rules = [
        rule("codex only, full coverage", codex_valid, len(rows),
             "the only grader that audited every proof; most permissive"),
        rule("claude-else-codex", coalesce, len(rows),
             "prefer the stricter grader where it ran"),
        rule("neither grader rejects", neither, len(rows),
             "a proof stands unless someone rejected it"),
        rule("both graders VALID", both_valid, len(graded_both),
             "strictest; only over the rows both audited"),
    ]

    # The row-level finding: a correct answer resting on a rejected proof.
    correct_on_rejected = {
        "codex_rejects": sum(1 for r in rows if r.get("key_match")
                             and verdict(r.get("codex")) in REJECT),
        "either_grader_rejects": sum(1 for r in rows if r.get("key_match")
                                     and (verdict(r.get("codex")) in REJECT
                                          or verdict(r.get("claude")) in REJECT)),
    }
    codex_breakdown = collections.Counter(
        verdict(r.get("codex")) for r in rows if verdict(r.get("codex")) in REJECT)

    return {
        "grader_coverage": coverage,
        "soundness_by_rule": rules,
        "range_pct": [min(r["precision_pct"] for r in rules),
                      max(r["precision_pct"] for r in rules)],
        "correct_answer_on_rejected_proof": correct_on_rejected,
        "codex_rejection_counts": dict(codex_breakdown),
        "note": ("The published 63.8% matched none of these rules and had no "
                 "committed derivation. The range is reported instead. The "
                 "qualitative finding is robust: every rule puts proof "
                 "soundness far below the 87.6% key-match precision."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="experiments/audit_proofs_strong_k3.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = analyse(args.audit)

    cov = report["grader_coverage"]
    print(f"certified proofs: {cov['n_certified_proofs']}")
    print(f"  codex verdicts : {cov['codex_returned_a_verdict']}")
    print(f"  claude verdicts: {cov['claude_returned_a_verdict']}")
    print(f"  audited by both: {cov['audited_by_both']}\n")
    for r in report["soundness_by_rule"]:
        print(f"  {r['rule']:26s} {r['sound']:3d}/{r['of']:3d} = "
              f"{r['precision_pct']:5.1f}%   {r['note']}")
    lo, hi = report["range_pct"]
    print(f"\nproof-soundness range: {lo}% - {hi}%   "
          f"against 87.6% key-match precision")
    c = report["correct_answer_on_rejected_proof"]
    print(f"correct answer on a rejected proof: {c['codex_rejects']} (codex), "
          f"{c['either_grader_rejects']} (either grader)")
    print(f"codex rejections: {report['codex_rejection_counts']}")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
