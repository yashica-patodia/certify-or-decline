#!/usr/bin/env python3
"""Do certified proofs actually establish their answers?

Every precision number in RESULTS_OPEN_WEIGHT.md is `extract_choice` against the
answer key. That measures whether the *answer* is right. It says nothing about
whether the *proof* is right -- and this system's entire claim is verification,
not accuracy. A certified answer that matches the key on an invalid proof is a
false certification that key-match scoring records as a success.

This audits that gap. It samples certified problems, renders the proof as the
judge chain accepted it, and asks an independent grader one question: does this
argument, as written, establish its own stated conclusion?

The audit follows two rules:

* **The grader never sees the answer key**, and is told not to work the problem
  out independently. Otherwise it grades answer correctness -- the thing we
  already measure -- instead of proof validity, and agreement with key-match
  becomes an artifact of the prompt.
* Compare auditors from different model families (default gpt-oss + Qwen).
  Their agreement is descriptive; neither auditor establishes ground truth.

The headline this produces is the cross-tabulation of proof validity against
key match, whose interesting cell is `correct answer, invalid proof` -- answers
the current metric scores as wins that the system should have declined.

Usage:
    python -m experiments.proof_validity_audit RUN.json [RUN.json ...] \
        --sample 50 --graders claude,codex --out audit_proofs.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import sys

from experiments.build_public_runs import grader_map
from llm import llm as _llm_call
from pipeline import load_config

# OpenAI structured outputs (which codex --output-schema uses) rejects a schema
# where a declared property is absent from `required` -- every key must be
# required, and optionality is expressed with a nullable type. Claude tolerates
# the looser form, so an optional `critical_step` here fails on codex only.
VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "critical_step", "reason"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["VALID", "GAP", "ERROR", "NONSEQUITUR"],
        },
        "critical_step": {
            "type": ["integer", "null"],
            "description": "1-indexed step where the defect first appears, "
                           "or null when the verdict is VALID.",
        },
        "reason": {"type": "string"},
    },
}

SYSTEM = """\
You are auditing a chain of reasoning that an automated verifier has already \
ACCEPTED. Your job is to decide whether that acceptance was justified.

You are given a problem, a numbered chain of steps, and the conclusion the \
chain reached. Judge ONLY this question: does the chain, as written, establish \
its stated conclusion?

Rules:
- Do NOT solve the problem yourself and compare answers. A chain can reach a \
correct answer by invalid reasoning; that is exactly what you must catch.
- Do NOT reward plausibility, fluency, or confident phrasing.
- A step asserting a standard fact, law, or constant is acceptable without \
derivation. A step asserting the crux of the problem is not.

Return one verdict:
  VALID       - the conclusion follows; every inferential step is supported.
  GAP         - a step required for the conclusion is asserted without support.
  ERROR       - a step is factually or mathematically wrong.
  NONSEQUITUR - the final state does not yield the stated conclusion.

Give `critical_step` for the first defective step (null when VALID), and one \
or two sentences in `reason`."""


def render(row: dict) -> str:
    proof = row.get("proof") or {}
    lines = [f"PROBLEM:\n{row.get('problem', '').strip()}\n"]
    initial = proof.get("initial_state")
    if initial:
        lines.append(f"INITIAL STATE: {json.dumps(initial)}\n")
    lines.append("REASONING CHAIN:")
    for i, step in enumerate(proof.get("steps") or [], 1):
        state = step.get("state") or []
        newest = state[-1] if state else ""
        lines.append(
            f"  {i}. [{step.get('justification_type')}] {newest}\n"
            f"     justification: {step.get('justification')}"
        )
    lines.append(f"\nCONCLUSION REACHED: {row.get('answer')}")
    assumptions = row.get("verified_under_assumptions") or []
    if assumptions:
        lines.append(
            "\nThe verifier accepted these steps only under stated conventions:\n"
            + "\n".join(
                f"  - step {a.get('step_number')}: {a.get('convention')}"
                for a in assumptions
            )
        )
    return "\n".join(lines)


# Mirrors configs/audit_grader.yaml for claude. For codex, note that
# Proprietary aliases describe the recorded audit and may not remain available.
# --grader-config NAME=YAML permits an explicitly recorded replacement.
GRADER_SETTINGS = {
    # OSS graders, served on AWS and reached over SSH tunnels. These are the
    # only ones whose verdicts are reproducible from the open-weight stack --
    # claude and codex are proprietary subscription services, so a result that
    # depends on them cannot be checked by a reader who has only the models.
    # The pair is deliberately CROSS-FAMILY (gpt-oss vs Qwen): two graders from
    # one family agreeing tells you about the family, not about the proofs.
    "oss-gptoss": {
        "backend": "reasoning_agent",
        "model": "openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee",
        "endpoint": os.getenv("OWRE_GRADER_ENDPOINT", "http://127.0.0.1:18000/v1"),
        # see grader_oss_gptoss20b.yaml: at the template default this model
        # emits no final channel on 11.9% of calls, and a failed grade is
        # silently scored as a defect in the proof.
        "reasoning_effort": "low",
        "temperature": 0, "seed": 20260718,
        "max_tokens": 4096, "context_length": 32768, "max_turns": 8,
        "allow_shell": False, "allow_host_tools": False, "search": False,
        "request_timeout": 900, "preamble": False,
    },
    "oss-qwen": {
        "backend": "reasoning_agent",
        "model": "Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18",
        "endpoint": os.getenv("OWRE_QWEN_AUDIT_ENDPOINT",
                              os.getenv("OWRE_MODEL_ENDPOINT", "http://127.0.0.1:18001/v1")),
        "temperature": 0, "seed": 20260718,
        "max_tokens": 4096, "context_length": 32768, "max_turns": 8,
        "allow_shell": False, "allow_host_tools": False, "search": False,
        "request_timeout": 900, "preamble": False,
    },
    "oss-qwen32b": {
        "backend": "reasoning_agent",
        "model": "Qwen/Qwen3-32B-FP8@aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df",
        "endpoint": os.getenv("OWRE_QWEN32B_AUDIT_ENDPOINT", "http://127.0.0.1:8000/v1"),
        "temperature": 0, "seed": 20260718,
        "max_tokens": 4096, "context_length": 32768, "max_turns": 8,
        "allow_shell": False, "allow_host_tools": False, "search": False,
        "request_timeout": 900, "preamble": False,
    },
    # Proprietary. These were used in the paper; an open-weight GPU host alone
    # cannot repeat these calls. Their recorded labels are released separately.
    "claude": {"backend": "claude", "model": "opus", "effort": "max",
               "preamble": False},
    "codex": {"backend": "codex", "model": "gpt-5.6-terra", "effort": "xhigh",
              "search": True, "preamble": False},
}


GRADER_CONCURRENCY = {"claude": 1, "codex": 2, "oss-gptoss": 4,
                      "oss-qwen": 4, "oss-qwen32b": 4}


async def grade(row: dict, backend: str) -> dict:
    settings = GRADER_SETTINGS.get(backend)
    if settings is None:
        raise SystemExit(f"unknown grader {backend!r}; "
                         f"known: {sorted(GRADER_SETTINGS)}")
    config = {"audit_grader": dict(settings)}
    response, _ = await _llm_call(
        render(row), role="audit_grader", schema=VERDICT_SCHEMA,
        system=SYSTEM, config=config, watch=False,
    )
    if not isinstance(response, dict):
        raise RuntimeError(f"{backend} returned {type(response).__name__}")
    return response


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa on the binary VALID / not-VALID collapse."""
    a = [x == "VALID" for x in a]
    b = [x == "VALID" for x in b]
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--graders", default="oss-gptoss,oss-qwen")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--out", default="audit_proofs.json")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="override the default concurrency for every grader")
    parser.add_argument("--endpoint", action="append", default=[], metavar="NAME=URL",
                        help="override one auditor's OpenAI-compatible endpoint")
    parser.add_argument("--grader-config", action="append", default=[], metavar="NAME=YAML",
                        help="override auditor settings using a YAML audit_grader role")
    parser.add_argument("--limit", type=int, default=None,
                        help="grade at most N still-ungraded items this pass "
                             "(for chunking the tty-bound claude grader)")
    args = parser.parse_args()
    graders = [g.strip() for g in args.graders.split(",")]
    if not graders or len(set(graders)) != len(graders) or any(g not in GRADER_SETTINGS for g in graders):
        parser.error(f"--graders must name distinct entries from {sorted(GRADER_SETTINGS)}")
    if args.sample < 1 or (args.concurrency is not None and args.concurrency < 1):
        parser.error("--sample and --concurrency must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for value in args.grader_config:
        name, separator, path = value.partition("=")
        if not separator or name not in graders:
            parser.error("--grader-config must be SELECTED_GRADER=YAML")
        config = load_config([path])
        if not isinstance(config.get("audit_grader"), dict):
            parser.error(f"no audit_grader role in {path}")
        GRADER_SETTINGS[name] = dict(GRADER_SETTINGS[name], **config["audit_grader"])
    for value in args.endpoint:
        name, separator, endpoint = value.partition("=")
        if not separator or name not in graders or not endpoint.startswith(("http://", "https://")):
            parser.error("--endpoint must be SELECTED_GRADER=http(s)://HOST:PORT/v1")
        GRADER_SETTINGS[name]["endpoint"] = endpoint
    settings_sha256 = hashlib.sha256(json.dumps(
        {name: GRADER_SETTINGS[name] for name in graders}, sort_keys=True,
    ).encode()).hexdigest()

    # Resume: the claude CLI fails (is_error, duration_api_ms=0) when run
    # without a tty, so its half has to be done in foreground chunks. Reload any
    # verdicts already on disk and skip that (item, grader) pair.
    prior: dict = {}
    if os.path.exists(args.out):
        previous = json.load(open(args.out))
        prior = previous.get("items", {})
        if not isinstance(prior, dict):
            parser.error("--out contains released labels, not resumable raw audits; use a fresh path")
        if previous.get("grader_settings_sha256") != settings_sha256:
            parser.error("auditor settings differ or were not recorded; use a fresh --out")

    certified = []
    keys: set[str] = set()
    for path in args.runs:
        run_name = Path(path).stem
        # `row["correct"]` is the IN-RUN score from summarize_gpqa.extract_choice,
        # which cannot read free-form answers and records them as wrong. On
        # strong-k3 it claims 46/105 where the grading pass says 92/105 -- they
        # disagree on 46 of 105 rows. That artifact already forced one published
        # retraction; the audit reads the grade files instead.
        grades = grader_map(path)
        rows = json.load(open(path))
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            parser.error(f"{path} must be a raw run list, not a grade report")
        for row in rows:
            if row.get("verified") and not row.get("error"):
                if not row.get("problem") or not row.get("proof") or not row.get("answer"):
                    parser.error("audit requires full locally generated proofs; released rows are redacted")
                key = f"{run_name}:{row.get('id')}"
                if key in keys:
                    parser.error(f"duplicate certified observation {key}; supply distinct runs")
                keys.add(key)
                if not isinstance(grades.get(str(row.get("id"))), bool):
                    parser.error(f"no Boolean answer grade for {key}")
                row = dict(row, _key_match=grades.get(str(row.get("id"))))
                # Key by (seed, problem): the same problem certifies in several
                # seeds, and collapsing on `id` alone silently drops all but one
                # -- 105 certified rows became 47. The precision figure being
                # audited is over rows, so the audit must be too.
                certified.append(dict(row, _key=key))
    random.Random(args.seed).shuffle(certified)
    sample = certified[: args.sample]
    if set(prior) - {row["_key"] for row in sample}:
        parser.error("existing audit keys differ from this sample; use a fresh --out")
    print(f"{len(certified)} certified rows; auditing {len(sample)} "
          f"with {graders}", file=sys.stderr)

    sems = {g: asyncio.Semaphore(args.concurrency or GRADER_CONCURRENCY.get(g, 4))
            for g in graders}

    async def one(row: dict, backend: str) -> tuple[str, str, dict]:
        done = (prior.get(row["_key"]) or {}).get(backend)
        if done and done.get("verdict") != "GRADER_FAILED":
            return row["_key"], backend, done
        async with sems[backend]:
            last = ""
            # Transient CLI failures (rate limits, skill-install races) are
            # common enough that a single attempt loses a third of the sample.
            for attempt in range(3):
                try:
                    return row["_key"], backend, await grade(row, backend)
                except Exception as exc:
                    last = str(exc)[:200]
                    await asyncio.sleep(5 * (attempt + 1))
            return row["_key"], backend, {"verdict": "GRADER_FAILED",
                                          "reason": last}

    pending = [
        (r, g) for r in sample for g in graders
        if not ((prior.get(r["_key"]) or {}).get(g)
                and (prior[r["_key"]][g].get("verdict") != "GRADER_FAILED"))
    ]
    if args.limit is not None:
        keep = {r["_key"] for r, _ in pending[: args.limit]}
        pending = [(r, g) for r, g in pending if r["_key"] in keep]
    print(f"{len(pending)} (item, grader) pairs to grade this pass",
          file=sys.stderr)
    graded = await asyncio.gather(*(one(r, g) for r, g in pending))
    results = list(graded)
    for r in sample:
        for g in graders:
            done = (prior.get(r["_key"]) or {}).get(g)
            if done and done.get("verdict") != "GRADER_FAILED":
                results.append((r["_key"], g, done))

    by_id: dict = {}
    for pid, backend, verdict in results:
        by_id.setdefault(pid, {})[backend] = verdict
    for row in sample:
        by_id.setdefault(row["_key"], {})["key_match"] = row.get("_key_match")
        by_id[row["_key"]]["id"] = row.get("id")

    usable = [
        (pid, rec) for pid, rec in by_id.items()
        if all(rec.get(g, {}).get("verdict", "GRADER_FAILED") != "GRADER_FAILED"
               for g in graders)
    ]
    print(f"\n{len(usable)}/{len(by_id)} items graded by all graders")

    if len(graders) == 2 and usable:
        a = [rec[graders[0]]["verdict"] for _, rec in usable]
        b = [rec[graders[1]]["verdict"] for _, rec in usable]
        agree = sum(x == y for x, y in zip(a, b)) / len(a)
        print(f"raw agreement {100*agree:.0f}%   "
              f"Cohen's kappa (VALID vs not) {kappa(a, b):.3f}")

    print("\nverdict distribution:")
    for g in graders:
        counts: dict = {}
        for _, rec in usable:
            counts[rec[g]["verdict"]] = counts.get(rec[g]["verdict"], 0) + 1
        print(f"  {g:8s} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("\nPROOF VALIDITY x KEY MATCH (both graders VALID = valid):")
    cells = {(True, True): 0, (True, False): 0,
             (False, True): 0, (False, False): 0}
    for _, rec in usable:
        valid = all(rec[g]["verdict"] == "VALID" for g in graders)
        cells[(valid, rec["key_match"])] += 1
    total = sum(cells.values()) or 1
    print(f"  valid proof   + correct answer : {cells[(True, True)]:3d}")
    print(f"  valid proof   + WRONG answer   : {cells[(True, False)]:3d}")
    print(f"  INVALID proof + correct answer : {cells[(False, True)]:3d}"
          f"   <- scored as success today")
    print(f"  INVALID proof + WRONG answer   : {cells[(False, False)]:3d}")
    sound = cells[(True, True)] / total
    print(f"\n  key-match precision      {100*(cells[(True, True)] + cells[(False, True)])/total:.1f}%")
    print(f"  proof-sound precision    {100*sound:.1f}%")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps({"items": by_id, "graders": graders,
                                    "grader_settings_sha256": settings_sha256,
                                    "sample": len(sample)}, indent=1) + "\n")
    os.replace(temporary, output)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
