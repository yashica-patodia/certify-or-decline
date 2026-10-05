#!/usr/bin/env python3
"""What the judges actually reject, computed over the current corpus.

WHY THIS EXISTS

Section 4 reported "233 rejections, 548 classified issues, 16 run files" and a
composition of "60.4% prose-invisible, other+cosmetic 0.9%". Those numbers
describe a corpus snapshot from early August that is not identified anywhere:
the document names neither the 16 files nor the rule that mapped issue classes
onto "prose-invisible". The corpus has since grown to 66 run files with
rejections, and the composition moved with it -- `other` plus `cosmetic` is now
4.0%, not 0.9%.

The grouping rule was never recorded and is reconstructed here (see
PROSE_INVISIBLE). The underlying data is fully derivable: each verdict carries
`accepted`, and
each issue carries `error_class` and `severity`. So rather than leave a stale
figure that cannot be checked, this computes the composition from whatever runs
are present, reporting the raw class distribution instead of a derived grouping
whose definition was lost.

Usage:
    python -m experiments.rejection_composition --out experiments/public_runs/rejection_composition.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import glob
import json

SKIP = ("/grade", "quarantine", "_FAILED", "artifacts", "_proprietary",
        "_duplicate", "recovered")

# "Prose-invisible" was the grouping the original section 4 reported at 60.4%,
# and the rule behind it was never written down. It is RECONSTRUCTED here, on
# semantics first: these three defects cannot be caught by reading the proof
# prose. A hidden premise is something absent; a fabricated citation is a
# reference you would have to go and check; circular reasoning reads as valid
# because every sentence follows from the last. The remaining classes are
# visible to a reader who does the work -- arithmetic can be recomputed, a
# misapplied theorem can be looked up, and cosmetic defects are surface by
# definition.
#
# On the current corpus this grouping gives 60.1%, against the 60.4% reported
# on a corpus a quarter the size. That agreement is corroboration, not the
# basis for the choice.
PROSE_INVISIBLE = ("hidden_premise", "fabricated_citation", "circular_reasoning")


def collect(pattern: str) -> dict:
    # Deduplicate by content. Eight run files exist byte-identically in two
    # non-skipped directories each (a run written to runs/ and also copied into
    # its campaign directory), and counting both inflated every figure in this
    # table: 2,046 rejections against a true 1,878.
    seen: set[str] = set()
    files = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        if any(s in p for s in SKIP):
            continue
        try:
            digest = hashlib.sha256(open(p, "rb").read()).hexdigest()
        except Exception:
            continue
        if digest in seen:
            continue
        seen.add(digest)
        files.append(p)
    rejections = 0
    issues = 0
    by_class: collections.Counter = collections.Counter()
    by_severity: collections.Counter = collections.Counter()
    by_role: collections.Counter = collections.Counter()
    files_with_rejections = 0

    for path in files:
        try:
            rows = json.load(open(path))
        except Exception:
            continue
        seen_here = 0
        for row in rows:
            for verdict in (row.get("verdicts") or []):
                if verdict.get("accepted") is not False:
                    continue
                rejections += 1
                seen_here += 1
                by_role[str(verdict.get("role"))] += 1
                for issue in (verdict.get("issues") or []):
                    if not isinstance(issue, dict):
                        continue
                    issues += 1
                    by_class[str(issue.get("error_class"))] += 1
                    by_severity[str(issue.get("severity"))] += 1
        if seen_here:
            files_with_rejections += 1

    def pct(counter: collections.Counter, total: int) -> dict:
        return {k: {"n": v, "pct": round(100 * v / total, 1)}
                for k, v in counter.most_common()} if total else {}

    invisible = sum(by_class[c] for c in PROSE_INVISIBLE)
    return {
        "prose_invisible_classes": list(PROSE_INVISIBLE),
        "prose_invisible_issues": invisible,
        "prose_invisible_pct": round(100 * invisible / issues, 1) if issues else None,
        "run_files_scanned": len(files),
        "run_files_with_rejections": files_with_rejections,
        "rejections": rejections,
        "classified_issues": issues,
        "by_error_class": pct(by_class, issues),
        "by_severity": pct(by_severity, issues),
        "rejections_by_judge_role": pct(by_role, rejections),
        "note": ("Computed over every run present, not the 16-file snapshot of "
                 "unknown membership that the original section 4 figures came "
                 "from, so these counts move as the corpus grows. The raw "
                 "error_class distribution is the primary report. The "
                 "'prose-invisible' grouping the original used was never "
                 "recorded; PROSE_INVISIBLE here was rebuilt on semantics and "
                 "only then compared against the figure it replaces."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/**/benchmark_*.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = collect(args.runs)

    # Scanning nothing is not a result. The pattern defaults to the raw corpus,
    # which a reader's checkout does not have, so running the command in this
    # module's docstring there produced an all-zero report, exit 0, and -- with
    # --out at its documented value -- overwrote the published artifact with it.
    # Same failure cluster_intervals.py closed at parse time; checked here
    # before anything is written.
    if rep["run_files_scanned"] == 0:
        raise SystemExit(
            f"no run files matched {args.runs!r}, so there is nothing to "
            f"compose. Refusing to write an all-zero artifact over a real one. "
            f"This analysis needs the raw run corpus, which is not "
            f"redistributable; the published result is already committed at "
            f"experiments/public_runs/rejection_composition.json.")

    print(f"run files scanned          : {rep['run_files_scanned']}")
    print(f"  with at least one reject : {rep['run_files_with_rejections']}")
    print(f"rejections                 : {rep['rejections']}")
    print(f"classified issues          : {rep['classified_issues']}\n")
    print(f"prose-invisible             : {rep['prose_invisible_pct']}%  "
          f"({rep['prose_invisible_issues']} issues; "
          f"{', '.join(rep['prose_invisible_classes'])})\n")
    print("by error class:")
    for k, v in rep["by_error_class"].items():
        print(f"   {k:22s} {v['n']:5d}  {v['pct']:5.1f}%")
    print("\nby severity:")
    for k, v in rep["by_severity"].items():
        print(f"   {k:22s} {v['n']:5d}  {v['pct']:5.1f}%")
    print("\nrejections by judge role:")
    for k, v in rep["rejections_by_judge_role"].items():
        print(f"   {k:22s} {v['n']:5d}  {v['pct']:5.1f}%")

    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
