#!/usr/bin/env python3
"""Strip benchmark content out of run files so results can be published.

The raw run JSONs embed the full GPQA problem statement and its answer key, and
the per-call logs quote the problem back inside every prompt. GPQA
(`Idavidrein/gpqa`) is gated and carries a contamination canary; `DATA.md` states
this repository never redistributes benchmark content. Publishing the raw files
would both breach that and put question/answer pairs into a public artifact that
future models may train on -- the exact contamination channel the study measures.

This follows `paper/build_public_db.py`: keep the IDs and every number, drop the
benchmark text. Anyone with GPQA access can rejoin on `id`.

KEPT   : id, verified, verified_unconditionally, error CLASS, metrics,
         repair_metrics counters + answer hashes,
         verdict accept/reject + error_class (no reasoning text), proof
         STRUCTURE (step count, justification types) and, derived here,
         key_match against the grader -- so every published number is checkable.
DROPPED: problem, expected, answer, solution, solver_solutions, calls, attempts,
         proof step text, verdict reason text.

Usage:
    python -m experiments.build_public_runs OUT_DIR RUN.json [RUN.json ...]
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys


# repair_metrics is mostly counters, but several keys (`final_answer`,
# `first_attempt_answer`, `answer_before_repair`) hold answer text verbatim --
# for a correct certification that IS the answer key. Allowlist rather than
# denylist, so a field added later cannot silently start leaking: keep scalars
# and hashes, drop every free string. The hashes still prove determinism across
# seeds without disclosing content.
def _safe_metrics(metrics: object) -> dict | None:
    if not isinstance(metrics, dict):
        return None
    return {
        k: v for k, v in metrics.items()
        if isinstance(v, (bool, int, float)) or v is None or k.endswith("_sha256")
    } or None

# Error strings quote the model's rejected reply, which quotes the option text.
# Map to a class + the structural counters instead.
_ERROR_CLASSES = [
    (r"same response \d+ times in a row and without", "stall:repeated_response"),
    (r"same response \d+ times in a row and every a", "stall:repeated_schema_reject"),
    (r"exceeded max_turns", "stall:max_turns"),
    (r"projects to >= \d+ input", "context_exhausted"),
    (r"HTTP \d+", "transport_error"),
]


def error_class(error: object) -> str | None:
    """Classify without echoing the message -- it embeds benchmark text."""
    if not error:
        return None
    text = str(error)
    for pattern, name in _ERROR_CLASSES:
        if re.search(pattern, text):
            turns = re.search(r"stopped at turn (\d+)/(\d+)", text)
            if turns:
                return f"{name}@{turns.group(1)}/{turns.group(2)}"
            return name
    return "other"


def strip_row(row: dict, key_match: bool | None) -> dict:
    # An empty answer is wrong by definition, and is never sent to a grader for
    # adjudication. Grader validation on MATH-500 found exactly one false
    # positive in 114 cases and it was this: an empty answer judged correct
    # against `\frac{3\sqrt{3}}{4}`. No certified row in any run has an empty
    # answer (0 of 1,128), so nothing reported changes -- this makes the failure
    # mode impossible rather than merely unexposed.
    answer = str(row.get("answer") or "").strip()
    if not answer or answer.lower() == "none":
        key_match = False
    proof = row.get("proof") or {}
    steps = proof.get("steps") or []
    verdicts = []
    for v in (row.get("verdicts") or []):
        if not isinstance(v, dict):
            continue
        verdicts.append({
            "step_number": v.get("step_number"),
            "role": v.get("role"),
            "accepted": v.get("accepted"),
            # error_class only -- the reasoning text quotes the problem
            "error_classes": [
                (i or {}).get("error_class") for i in (v.get("issues") or [])
                if isinstance(i, dict)
            ],
        })
    return {
        "id": row.get("id"),
        "verified": row.get("verified"),
        "verified_unconditionally": row.get("verified_unconditionally"),
        "error": error_class(row.get("error")),
        "key_match": key_match,
        "n_proof_steps": len(steps),
        "justification_types": [s.get("justification_type") for s in steps],
        "verdicts": verdicts,
        "metrics": _safe_metrics(row.get("metrics")),
        "repair_metrics": _safe_metrics(row.get("repair_metrics")),
        "config_hash": row.get("config_hash"),
        "dataset_revision": row.get("dataset_revision"),
        "benchmark_file_sha256": row.get("benchmark_file_sha256"),
        "verifier_mode": row.get("verifier_mode"),
    }


def grader_map(run_path: str, *, grader: str | None = None,
               target: str = "final", allow_ungraded: bool = False) -> dict:
    """key_match per problem id, from the grading pass.

    Searches every `runs/grade-*/` directory rather than a hardcoded list: an
    arm graded into a new directory (e.g. `runs/grade-stronglow/`) used to fall
    through and return {}, which reads downstream as "no answer was correct".

    Raises when two grade files for the same run disagree, and when `grader` is
    given but no file from that grader exists. Both are failure modes that
    otherwise surface as a quietly wrong precision number: arms compared across
    DIFFERENT grader models look different for reasons that have nothing to do
    with the arms. `strong-k3` was graded by the GPT-5.4 answer grader and
    `stronglow` by claude:opus:max, and the two were compared directly before
    this was noticed.
    """
    base = os.path.basename(run_path)
    found: list[tuple[str, str, dict]] = []
    for path in sorted(glob.glob(os.path.join("runs", "grade-*", base))):
        if "quarantine" in path:
            continue
        data = json.load(open(path))
        # A grade file records WHICH answer it graded. `final` is the verifier's
        # certified answer; `solver_initial` is the raw first draw, graded to
        # measure the no-verification baseline. They grade different objects and
        # legitimately disagree, so mixing them would either trip the
        # disagreement check or, worse, silently substitute one for the other.
        if isinstance(data, dict) and data.get("grading_target", "final") != target:
            continue
        model = data.get("grader_model", "?") if isinstance(data, dict) else "?"
        # A grade file whose grader calls FAILED is not a grade file. grade.py
        # records a failed call as key_match=False and still writes a clean
        # summary, so a run where every call errored reports "0 matched the
        # key" -- indistinguishable from a model that got everything wrong.
        # One such file claimed 0/100 on a run whose real precision is 88.6%.
        m = (data.get("metrics") or {}) if isinstance(data, dict) else {}
        failed = int(m.get("failed_calls") or 0)
        calls = int(m.get("num_calls") or 0)
        # Systemic failure -- a dead endpoint or a wrong config -- makes every
        # call fail and turns the file into "0 matched the key". That has
        # happened twice: 75/75 and 45/45. Reject anything at that scale.
        #
        # An ISOLATED failure is different in kind. On one MATH-500 row the
        # grader model could not emit a conforming response and looped until it
        # ran out of turns; grade.py scores that row incorrect, which understates
        # precision by at most one problem and can never inflate it. Allowing a
        # small, conservative-direction failure rate is preferable to discarding
        # 99 sound judgments, but the threshold is explicit so it cannot quietly
        # widen.
        if failed and (calls == 0 or failed / calls > 0.05):
            raise SystemExit(
                f"{path} has {failed}/{calls} FAILED grader calls. grade.py "
                "scores a failed call as incorrect, so this file understates "
                "precision by an unknown amount. Re-grade it, or move it aside; "
                "do not report anything derived from it."
            )
        rows = data if isinstance(data, list) else (
            data.get("grades") or data.get("results") or []
        )
        out = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            km = r.get("key_match")
            if km is None:
                km = (r.get("final") or {}).get("key_match")
            out[str(r.get("id"))] = km
        found.append((path, model, out))

    if grader is not None:
        found = [f for f in found if f[1] == grader]
        if not found:
            raise SystemExit(
                f"no grades from grader {grader!r} for {base}. Comparing arms "
                "graded by different models is not a valid comparison; grade "
                "this run with that model first."
            )

    if not found:
        # An ungraded run must NOT return an empty map by default. Every
        # analysis caller does `bool(grades.get(pid))`, so an empty map scores
        # every problem WRONG and reports 0% precision without a word -- the
        # same failure that once had a run at 88.6% reported as 0/100. Callers
        # that genuinely tolerate ungraded runs (the bundle builder, which
        # publishes key_match=None) must say so explicitly.
        if allow_ungraded:
            return {}
        raise SystemExit(
            f"no grade file found for {base}. Refusing to return an empty "
            "grade map: callers treat a missing grade as an incorrect answer, "
            "which silently reports 0% precision. Grade the run, or pass "
            "allow_ungraded=True if a missing grade is genuinely acceptable."
        )

    # A grade file must cover every problem in the run. Runs are written
    # incrementally, so copying one off a worker mid-write yields a truncated
    # file that grades cleanly and silently shrinks the denominator: three files
    # were graded this way, one with 11 of 100 problems and two missing the last
    # problem. Coverage is certified/n, so a short run inflates it.
    try:
        run_ids = {str(r.get("id")) for r in json.load(open(run_path))
                   if isinstance(r, dict)}
    except Exception:
        run_ids = set()
    if run_ids:
        for path, model, other in found:
            missing = run_ids - set(other)
            if missing:
                raise SystemExit(
                    f"{path} grades {len(other)} problems but the run has "
                    f"{len(run_ids)}; missing {sorted(missing)[:5]}"
                    f"{'...' if len(missing) > 5 else ''}. Re-grade against the "
                    "complete run -- a partially graded run understates "
                    "coverage and precision."
                )

    first_path, first_model, first = found[0]
    for path, model, other in found[1:]:
        shared = set(first) & set(other)
        clash = [i for i in shared if bool(first[i]) != bool(other[i])]
        if clash:
            raise SystemExit(
                f"grade files disagree for {base} on {len(clash)} problems:\n"
                f"  {first_path} ({first_model})\n  {path} ({model})\n"
                "Resolve before reporting any number derived from them."
            )
    return first


def main() -> None:
    out_dir, paths = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    for path in paths:
        # The bundle builder publishes key_match=None for an ungraded run,
        # which is visible rather than silently wrong, so it is the one
        # caller allowed to proceed without grades.
        grades = grader_map(path, allow_ungraded=True)
        rows = json.load(open(path))
        stripped = [strip_row(r, grades.get(str(r.get("id")))) for r in rows]
        dest = os.path.join(out_dir, os.path.basename(path))
        json.dump(stripped, open(dest, "w"), indent=1)
        raw = os.path.getsize(path) / 1e6
        new = os.path.getsize(dest) / 1e6
        print(f"  {os.path.basename(path)[:46]:48s} {raw:7.1f} MB -> {new:5.2f} MB")


if __name__ == "__main__":
    main()
