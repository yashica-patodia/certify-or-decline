#!/usr/bin/env python3
"""Risk-coverage curve for the self-consistency baseline, from raw solver draws.

WHY THIS EXISTS

The k=30 curve published in section 8l was produced by an inline script that was
never committed, so the single most load-bearing baseline number in the study
could not be re-derived by a reader -- or audited by its author. Rebuilding it
here surfaced two defects in it, both of which weakened the baseline:

1. ANSWER EXTRACTION. The solver emits its answer inside an action envelope,
   `{"action":"final","response":"C"}`. It also, on a minority of draws, emits a
   MALFORMED envelope -- `{"action":"final","response":"A") Skyrmion"}` -- which
   `json.loads` rejects. A strict parser silently discards those draws even
   though the intended answer is unambiguous. 33 real votes were being thrown
   away.

2. THE AGREEMENT DENOMINATOR. 424 of 3000 draws yield no answer at all (358 are
   empty completions; the rest truncate before the envelope). Whether those
   count against a problem's agreement score is a methodological choice with a
   large effect, and it had never been stated:

     denominator = parsed votes  (standard self-consistency; Wang et al. 2023
                                  marginalises over the sampled paths that
                                  produced an answer)
     denominator = k             (an abstention counts as dissent)

   The published curve used `k` without saying so. That is the choice that makes
   the baseline look worse, and it is also asymmetric with how Verifier is
   scored: a Verifier execution error is treated as a DECLINE and removed from
   coverage, not as evidence against its own confidence.

Both conventions are computed here and both are reported, because the choice
changes the comparison's conclusion and the reader is entitled to see that.

Scoring is exact letter match. The majority vote is a letter and the key is a
letter, so no LLM grader is involved and no grader-mismatch can arise -- unlike
the free-form arms, where `grader_map` is mandatory.

Usage:
    python -m experiments.self_consistency_curve \
        --draws runs/k30/solver_k30_merged.json \
        --benchmark runs/benchmarks/gpqa-fixed-100.json \
        --out experiments/public_runs/self_consistency_k30_curve.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re

from experiments.gpqa_common import options_from_question
from experiments.summarize_gpqa import extract_choice

ENVELOPE = re.compile(r'\{"action"\s*:\s*"final".*?\}', re.S)
LOOSE_RESPONSE = re.compile(
    r'"action"\s*:\s*"final".{0,40}?"response"\s*:\s*"(.*)', re.S
)


def extract_answer(text: str, options: dict | None) -> str | None:
    """The draw's final answer, tolerating the envelopes the solver really emits.

    Order matters: a well-formed envelope is authoritative, so it is tried
    first and only its LAST occurrence counts (earlier ones are superseded).
    The loose path is a fallback for envelopes that are not valid JSON, and it
    reads only the head of the response field so that a run-on generation
    cannot smuggle in a later letter.
    """
    if not text:
        return None
    candidates = ENVELOPE.findall(text)
    for cand in reversed(candidates):
        try:
            payload = json.loads(cand)
        except Exception:
            continue
        choice = extract_choice(str(payload.get("response", "")), options)
        if choice:
            return choice
    loose = LOOSE_RESPONSE.search(text)
    if loose:
        choice = extract_choice(loose.group(1)[:200], options)
        if choice:
            return choice
    return extract_choice(text, options)


def load_votes(draws_path: str, benchmark: str,
               max_draws: int | None = None) -> tuple[dict, dict, dict]:
    bench = json.load(open(benchmark))
    problems = bench["problems"] if isinstance(bench, dict) else bench
    key = {str(p["id"]): str(p.get("answer") or p.get("expected") or "")
           .strip().upper() for p in problems}
    questions = {str(p["id"]): (p.get("question") or p.get("problem") or "")
                 for p in problems}

    draws = json.load(open(draws_path))
    if set(draws) != set(key):
        raise SystemExit("draw IDs differ from the benchmark cohort")
    votes: dict[str, list[str]] = {}
    stats = collections.Counter()
    for pid, per_draw in draws.items():
        try:
            options = options_from_question(questions.get(pid, ""))
        except Exception:
            options = None
        got = []
        # Restricting to the first `max_draws` draw INDICES is what makes the
        # compute-matched (k=8) baseline derivable from the same corpus: a
        # draw's seed comes from its index, so indices 0..7 are exactly the
        # draws a k=8 run would have produced.
        items = sorted(per_draw.items(), key=lambda kv: int(kv[0]))
        if max_draws is not None:
            if not {str(i) for i in range(max_draws)} <= set(per_draw):
                raise SystemExit(f"{pid} is missing draws in the first {max_draws} indices")
            items = [kv for kv in items if int(kv[0]) < max_draws]
        for _, rec in items:
            stats["draws"] += 1
            if rec.get("error"):
                stats["call_errors"] += 1
                stats["unparsed"] += 1
                continue
            choice = extract_answer(rec.get("text") or "", options)
            if choice:
                got.append(choice.upper())
                stats["parsed"] += 1
            else:
                stats["unparsed"] += 1
                if not (rec.get("text") or "").strip():
                    stats["unparsed_empty_completion"] += 1
        votes[pid] = got
    return votes, key, dict(stats)


def curve(votes: dict, key: dict, k: int, denominator: str) -> list[dict]:
    """Achievable operating points, thresholding on majority agreement.

    Each distinct agreement value is one reachable threshold; sweeping between
    them would invent operating points no selector can occupy.
    """
    rows = []
    for pid, vs in votes.items():
        if not vs:
            # nothing parsed: the selector has no answer to offer. It is
            # never selected at any threshold above 0.
            rows.append((0.0, False, pid))
            continue
        counts = collections.Counter(vs)
        top = max(counts.values())
        winner = sorted(k_ for k_, v in counts.items() if v == top)[0]
        denom = len(vs) if denominator == "parsed" else k
        rows.append((top / denom, winner == key.get(pid), pid))

    points = []
    for thresh in sorted({r[0] for r in rows if r[0] > 0}, reverse=True):
        sel = [r for r in rows if r[0] >= thresh]
        n_ok = sum(1 for r in sel if r[1])
        points.append({
            "coverage": round(100 * len(sel) / len(rows), 1),
            "precision": round(100 * n_ok / len(sel), 1),
            "n_answered": len(sel),
            "n_correct": n_ok,
            "agreement_threshold": round(thresh, 4),
        })
    return points


def aurc(points: list[dict]) -> float:
    """Mean selective risk over the covered range (trapezoidal)."""
    if not points:
        return float("nan")
    pts = [(p["coverage"] / 100, 1 - p["precision"] / 100) for p in points]
    area, prev_c, prev_r = 0.0, 0.0, pts[0][1]
    for c, r in pts:
        area += (c - prev_c) * (r + prev_r) / 2
        prev_c, prev_r = c, r
    return area / prev_c if prev_c else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", required=True)
    ap.add_argument("--benchmark", default="runs/benchmarks/gpqa-fixed-100.json")
    ap.add_argument("--k", type=int, default=30,
                    help="votes per problem; also caps the draw indices used")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.k < 1:
        ap.error("--k must be positive")

    votes, key, stats = load_votes(args.draws, args.benchmark,
                               max_draws=args.k)
    print(f"problems: {len(votes)}   draws: {stats.get('draws')}")
    print(f"parsed: {stats.get('parsed')}   unparsed: {stats.get('unparsed')} "
          f"(of which {stats.get('unparsed_empty_completion', 0)} empty completions)")

    out = {
        "selector": f"self-consistency majority vote, k={args.k}",
        "benchmark": "gpqa-fixed-100",
        "n_problems": len(votes),
        "total_draws": stats.get("draws"),
        "draws_parsed": stats.get("parsed"),
        "draws_unparsed": stats.get("unparsed"),
        "draws_unparsed_empty_completion": stats.get("unparsed_empty_completion", 0),
        "scoring": ("exact letter match; the vote and the key are both letters, "
                    "so no LLM grader is involved"),
        "note": (
            "Two agreement denominators are reported because the choice is "
            "methodological, materially changes the comparison, and was "
            "undocumented in the first published version of this curve. "
            "'parsed' is standard self-consistency (majority over the draws that "
            "produced an answer) and is symmetric with the verifier's scoring, where "
            "an execution error is a decline rather than evidence against "
            "confidence. 'all_k' counts a non-answering draw as dissent. "
            "AURC is a mean risk over a coverage range and is NOT comparable "
            "across selectors whose ranges are unequal; compare at matched coverage."
        ),
        "by_denominator": {},
    }
    for denom in ("parsed", "all_k"):
        pts = curve(votes, key, args.k, denom)
        # A selector whose finest reachable point is above 35% coverage has no
        # curve inside the verifier's range, so the restricted AURC is undefined --
        # emit null, not NaN, which is not valid JSON and which every non-Python
        # reader rejects.
        under35 = [p for p in pts if p["coverage"] <= 35.1]
        restricted = round(aurc(under35), 4) if under35 else None
        out["by_denominator"][denom] = {
            "points": pts,
            "aurc_full_range": round(aurc(pts), 4) if pts else None,
            "aurc_restricted_to_35pct": restricted,
            "aurc_restricted_note": (
                None if under35 else
                "undefined: no achievable point lies inside the verifier's 0-35% range"),
            "finest_reachable_coverage": pts[0]["coverage"] if pts else None,
        }
        print(f"\ndenominator={denom}: {len(pts)} achievable points; "
              f"AURC(<=35%) {out['by_denominator'][denom]['aurc_restricted_to_35pct']}")

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
