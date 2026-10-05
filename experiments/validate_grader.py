#!/usr/bin/env python3
"""Measure the automatic grader's accuracy against known ground truth.

Every precision number in this study is produced by an LLM grader. Until now the
grader had been checked for *consistency* -- no failed calls, agreement with a
second vendor, stability across re-runs -- but never for *accuracy*, because
real runs have no ground truth: whether a free-form answer matches the key is
precisely the judgment under test.

This constructs cases whose correct verdict is certain, from the benchmark
itself. For each problem the keyed option is known, so:

    key letter       ("C")                     -> MUST be judged correct
    wrong letter     ("A" when the key is C)   -> MUST be judged incorrect
    key option text  (verbatim)                -> MUST be judged correct
    wrong option text(verbatim, a distractor)  -> MUST be judged incorrect
    empty answer                               -> MUST be judged incorrect

The last is the case that matters most operationally: `grade.py` scores a failed
grader call as a wrong answer, so a grader that says "correct" on nothing at all
would silently inflate precision.

The grader sees exactly what it sees in a real run -- same prompt file, same
schema, same role config, same call path -- so this measures the deployed
grader, not a reconstruction.

Reports overall accuracy plus the two error rates that matter separately:
FALSE POSITIVES (wrong answer judged correct) inflate precision and are the
dangerous direction; FALSE NEGATIVES deflate it.

Usage:
    python -m experiments.validate_grader --config experiments/configs/grader_oss_gptoss20b.yaml \
        --benchmark runs/benchmarks/gpqa-fixed-100.json --n 40 --out grader_validation.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys

from experiments.gpqa_common import options_from_question
from grade import GRADE_SCHEMA, format_input, load_prompt
from llm import llm as _llm_call
from pipeline import load_config


def build_freeform_cases(benchmark: str, n: int, seed: int) -> list[dict]:
    """Labelled cases for a FREE-FORM benchmark (MATH-500).

    Multiple-choice validation does not transfer here. On GPQA the grader only
    has to match a letter or an option's text; on MATH it must decide
    mathematical equivalence between notations -- `\\frac{14}{3}` against
    `14/3`, `\\left( 3, \\frac{\\pi}{2} \\right)` against `(3, pi/2)`. That is a
    different and much harder judgment, so it needs its own ground truth before
    any MATH run is graded.

    Only variants whose equivalence (or inequivalence) is CERTAIN are generated:

      exact            the reference answer verbatim            -> correct
      latex_stripped   \\frac{a}{b} -> a/b, \\left(..\\right) -> (..)  -> correct
      whitespace       spacing removed                          -> correct
      other_answer     a different problem's answer             -> incorrect
      perturbed        integer answer + 1                       -> incorrect
      empty            ""                                       -> incorrect

    A case is emitted only when the transformation actually changed the string,
    and `other_answer` only when the two answers differ.
    """
    import re as _re
    data = json.load(open(benchmark))
    problems = data["problems"] if isinstance(data, dict) else data
    rng = random.Random(seed)
    pool = [p for p in problems if str(p.get("answer") or "").strip()]
    rng.shuffle(pool)
    # The equivalence judgment is the hard one and is only exercised by answers
    # that carry LaTeX. Sorting those first stops a random draw from filling the
    # sample with bare integers, where the grader has nothing to decide.
    pool.sort(key=lambda p: 0 if any(t in str(p["answer"])
                                     for t in ("\\frac", "\\left", "\\pi",
                                               "\\text", "\\sqrt", "^")) else 1)

    def strip_latex(a: str) -> str:
        out = _re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", a)
        out = out.replace("\\left", "").replace("\\right", "")
        out = out.replace("\\pi", "pi").replace("\\cdot", "*")
        out = _re.sub(r"\\text\{([^{}]*)\}", r"\1", out)
        return out.strip()

    cases: list[dict] = []
    for i, p in enumerate(pool):
        if len({c["id"].split(":")[0] for c in cases}) >= n:
            break
        ans = str(p["answer"]).strip()
        pid = str(p.get("id"))
        base = {"problem": p.get("question") or "", "expected": ans}
        cases.append({**base, "id": f"{pid}:exact", "answer": ans,
                      "truth": True, "kind": "exact answer"})
        stripped = strip_latex(ans)
        if stripped and stripped != ans:
            cases.append({**base, "id": f"{pid}:latex_stripped",
                          "answer": stripped, "truth": True,
                          "kind": "equivalent notation"})
        nospace = _re.sub(r"\s+", "", ans)
        if nospace != ans:
            cases.append({**base, "id": f"{pid}:whitespace", "answer": nospace,
                          "truth": True, "kind": "whitespace variant"})
        other = str(pool[(i + 1) % len(pool)]["answer"]).strip()
        if other and other != ans:
            cases.append({**base, "id": f"{pid}:other", "answer": other,
                          "truth": False, "kind": "different answer"})
        if _re.fullmatch(r"-?\d+", ans):
            cases.append({**base, "id": f"{pid}:perturbed",
                          "answer": str(int(ans) + 1), "truth": False,
                          "kind": "off-by-one number"})
            # AIME answers are integers 0-999 and are conventionally written
            # zero-padded to three digits, so a model may answer 070 where the
            # key says 70. That is the same number and must be graded correct.
            if 0 <= int(ans) < 1000 and len(ans) < 3:
                cases.append({**base, "id": f"{pid}:zeropad",
                              "answer": ans.zfill(3), "truth": True,
                              "kind": "zero-padded integer"})
            cases.append({**base, "id": f"{pid}:trailing",
                          "answer": ans + ".", "truth": True,
                          "kind": "trailing punctuation"})
        cases.append({**base, "id": f"{pid}:empty", "answer": "",
                      "truth": False, "kind": "empty answer"})
    return cases


def build_cases(benchmark: str, n: int, seed: int) -> list[dict]:
    data = json.load(open(benchmark))
    problems = data["problems"] if isinstance(data, dict) else data
    rng = random.Random(seed)
    rng.shuffle(problems)

    cases: list[dict] = []
    for p in problems:
        if len(cases) >= n * 5:
            break
        question = p.get("problem") or p.get("question") or ""
        expected = str(p.get("expected") or p.get("answer") or "").strip().upper()
        try:
            options = options_from_question(question)
        except Exception:
            options = None
        if not isinstance(options, dict) or expected not in {
            k.upper() for k in options
        }:
            continue
        by_letter = {k.upper(): v for k, v in options.items()}
        key_text = str(by_letter[expected])
        wrong_letter = next(
            (l for l in sorted(by_letter) if l != expected), None
        )
        if wrong_letter is None:
            continue
        wrong_text = str(by_letter[wrong_letter])
        pid = str(p.get("id"))
        base = {"problem": question, "expected": expected}
        cases += [
            {**base, "id": f"{pid}:key_letter", "answer": expected,
             "truth": True, "kind": "key letter"},
            {**base, "id": f"{pid}:wrong_letter", "answer": wrong_letter,
             "truth": False, "kind": "wrong letter"},
            {**base, "id": f"{pid}:key_text", "answer": key_text,
             "truth": True, "kind": "key option text"},
            {**base, "id": f"{pid}:wrong_text", "answer": wrong_text,
             "truth": False, "kind": "wrong option text"},
            {**base, "id": f"{pid}:empty", "answer": "",
             "truth": False, "kind": "empty answer"},
        ]
    return cases[: n * 5]


async def grade_case(case: dict, config: dict, prompt_text: str) -> bool | None:
    result = {
        "problem": case["problem"],
        "expected": case["expected"],
        "answer": case["answer"],
        "grading_target": "final",
    }
    try:
        response, _ = await _llm_call(
            format_input(result), role="audit_grader", schema=GRADE_SCHEMA,
            system=prompt_text, config=config, watch=False,
        )
    except Exception:
        return None
    if not isinstance(response, dict):
        return None
    final = response.get("final") or {}
    return bool(final.get("key_match"))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--benchmark", default="runs/benchmarks/gpqa-fixed-100.json")
    ap.add_argument("--n", type=int, default=20, help="problems (x5 cases each)")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="grader_validation.json")
    args = ap.parse_args()

    config = load_config([args.config])
    prompt_text, prompt_sha = load_prompt("final")
    meta = json.load(open(args.benchmark))
    fmt = (meta.get("metadata") or {}).get("answer_format", "") if isinstance(meta, dict) else ""
    builder = build_freeform_cases if "free_form" in fmt else build_cases
    cases = builder(args.benchmark, args.n, args.seed)
    print(f"answer_format={fmt!r} -> {builder.__name__}", file=sys.stderr)
    print(f"{len(cases)} labelled cases from {len(cases)//5} problems; "
          f"grader prompt {prompt_sha[:12]}", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)

    async def one(case):
        async with sem:
            return case, await grade_case(case, config, prompt_text)

    results = await asyncio.gather(*(one(c) for c in cases))

    by_kind: dict[str, list[tuple[bool, bool | None]]] = {}
    fp = fn = failed = 0
    for case, verdict in results:
        by_kind.setdefault(case["kind"], []).append((case["truth"], verdict))
        if verdict is None:
            failed += 1
        elif verdict and not case["truth"]:
            fp += 1
        elif not verdict and case["truth"]:
            fn += 1

    usable = [(t, v) for _, vs in by_kind.items() for t, v in vs if v is not None]
    n = len(usable)
    correct = sum(1 for t, v in usable if t == v)
    print(f"\n{'case type':22s} {'n':>4} {'correct':>8} {'accuracy':>9}")
    for kind, vs in by_kind.items():
        ok = [(t, v) for t, v in vs if v is not None]
        c = sum(1 for t, v in ok if t == v)
        print(f"{kind:22s} {len(ok):4d} {c:8d} {100*c/len(ok) if ok else 0:8.1f}%")
    print(f"\noverall accuracy      : {correct}/{n} = {100*correct/n if n else 0:.1f}%")
    print(f"FALSE POSITIVES       : {fp}  (wrong answer judged correct -- inflates precision)")
    print(f"false negatives       : {fn}  (correct answer judged wrong)")
    print(f"grader call failures  : {failed}")

    json.dump({
        "grader_config": args.config, "grader_prompt_sha": prompt_sha,
        "n_cases": n, "accuracy": correct / n if n else None,
        "false_positives": fp, "false_negatives": fn, "call_failures": failed,
        "by_kind": {k: {"n": len([1 for _, v in vs if v is not None]),
                        "correct": sum(1 for t, v in vs if v is not None and t == v)}
                    for k, vs in by_kind.items()},
        "cases": [{"id": c["id"], "kind": c["kind"], "truth": c["truth"],
                   "verdict": v} for c, v in results],
    }, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
