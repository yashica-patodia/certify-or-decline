"""Export training-grade datasets from Verifier runs.

Walks ``runs/artifacts/<run_id>/`` (per-call meta.json artifacts and
per-problem result.json checkpoints), ``runs/<run_id>.json`` result
files, ``runs/partial/`` crash-safe states, and ``runs/grades/*.json``
grading outputs, and emits under ``exports/<run_id>/``:

1. Flat tables (parquet when pyarrow is available, else JSONL) in
   ``tables/``: role_calls, witnesses, steps, mutations, premise_edges,
   judge_calls, issues, filter_decisions, convention_lifts,
   repair_attempts, verdicts, rewards, dedup.
2. Ready-to-train JSONL at the export root: sft_witness.jsonl,
   sft_repair.jsonl, verifier_steps.jsonl, decline.jsonl.

Every row carries run_id, problem_id, attempt (where meaningful), and
config_hash. Old artifacts that predate `uses`/`issues` export with
nulls in those columns; the deterministic differ still runs on their
states. See docs/TRAINING_DATA.md for the schema of each output.

Usage:
    python export_training.py                     # every run in runs/artifacts
    python export_training.py <run_id> [...]      # specific runs
    python export_training.py --runs-dir runs --out exports
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import diffing

try:  # parquet is optional; JSONL is the fallback
    import pyarrow as _pa  # noqa: F401
    import pyarrow.parquet as _pq  # noqa: F401
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False


TABLE_NAMES = (
    "role_calls", "witnesses", "steps", "mutations", "premise_edges",
    "judge_calls", "issues", "filter_decisions", "convention_lifts",
    "repair_attempts", "verdicts", "rewards", "dedup",
)
TRAINING_FILES = (
    "sft_witness.jsonl", "sft_repair.jsonl",
    "verifier_steps.jsonl", "decline.jsonl",
)


# ── Loading ──────────────────────────────────────────────────────

def _load_json(path: str | Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def discover_runs(runs_dir: str) -> list[str]:
    root = os.path.join(runs_dir, "artifacts")
    if not os.path.isdir(root):
        return []
    return sorted(
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
        and not name.startswith("grade_")
    )


def _load_grades(runs_dir: str) -> dict[str, dict[str, dict]]:
    """Map run_id -> problem_id -> grade row (with final.key_match)."""
    grades: dict[str, dict[str, dict]] = {}
    for path in glob.glob(os.path.join(runs_dir, "grades", "*.json")):
        summary = _load_json(path)
        if not isinstance(summary, dict):
            continue
        run_file = str(summary.get("run_file") or "")
        run_id = os.path.basename(run_file)
        if run_id.endswith(".json"):
            run_id = run_id[:-5]
        if not run_id:
            continue
        per_problem = grades.setdefault(run_id, {})
        for row in summary.get("results") or []:
            pid = str(row.get("id"))
            final = row.get("final") or {}
            per_problem[pid] = {
                "gold_answer": row.get("expected"),
                "graded_answer": row.get("answer"),
                "key_match": (
                    final.get("key_match")
                    if isinstance(final.get("key_match"), bool) else None
                ),
                "dispute_category": final.get("dispute_category"),
                "grade_source": row.get("grade_source"),
            }
    return grades


def _partial_state_candidates(
    runs_dir: str, run_id: str, problem_id: str,
) -> list[str]:
    """Candidate crash-safe partial paths, newest layout first.

    The harness writes nested hash-slugged paths
    (``runs/partial/<run_slug>/<problem_slug>.json`` via
    ``harness._partial_state_path``); very old runs used a flat
    ``runs/partial/<pid>.json``. Both are checked.
    """
    candidates = []
    try:
        import harness
        nested = harness._partial_state_path(problem_id, run_id)
        # _partial_state_path builds a relative "runs/partial/..." path;
        # re-anchor it under the requested runs_dir.
        candidates.append(
            os.path.join(runs_dir, os.path.relpath(nested, "runs"))
        )
    except Exception:
        pass
    candidates.append(os.path.join(runs_dir, "partial", f"{problem_id}.json"))
    return candidates


def _load_problem_results(runs_dir: str, run_id: str) -> dict[str, dict]:
    """Best-available per-problem result records for one run.

    Priority: per-problem result.json checkpoints > the run's results
    file > runs/partial fallbacks (nested current layout, then the old
    flat layout). Partial states lack metrics/calls but still carry
    problem, solution, and attempts.
    """
    artifact_root = os.path.join(runs_dir, "artifacts", run_id)
    records: dict[str, dict] = {}

    run_file = _load_json(os.path.join(runs_dir, f"{run_id}.json"))
    if isinstance(run_file, list):
        for result in run_file:
            if isinstance(result, dict) and result.get("id") is not None:
                records[str(result["id"])] = result

    if os.path.isdir(artifact_root):
        for name in sorted(os.listdir(artifact_root)):
            problem_dir = os.path.join(artifact_root, name)
            if not os.path.isdir(problem_dir):
                continue
            result = _load_json(os.path.join(problem_dir, "result.json"))
            if isinstance(result, dict):
                records[str(result.get("id", name))] = result
            elif name not in records and _has_call_dirs(problem_dir):
                partial = None
                for candidate in _partial_state_candidates(
                    runs_dir, run_id, name,
                ):
                    loaded = _load_json(candidate)
                    if isinstance(loaded, dict):
                        partial = loaded
                        break
                if isinstance(partial, dict):
                    records[name] = {
                        "id": name,
                        "problem": partial.get("problem"),
                        "solution": partial.get("solution"),
                        "solver_solutions": partial.get("solver_solutions"),
                        "attempts": partial.get("attempts") or [],
                        "verified": _partial_verified(partial),
                        "answer": _partial_answer(partial),
                        "from_partial_state": True,
                    }
    return records


def _has_call_dirs(problem_dir: str) -> bool:
    try:
        return any(
            re.match(r"call_\d+_", name)
            for name in os.listdir(problem_dir)
        )
    except OSError:
        return False


def _partial_verified(partial: dict) -> bool | None:
    attempts = [
        a for a in (partial.get("attempts") or [])
        if a.get("phase") == "verify"
    ]
    if not attempts:
        return None
    return bool(attempts[-1].get("all_ok"))


def _partial_answer(partial: dict) -> str | None:
    for attempt in reversed(partial.get("attempts") or []):
        proof = attempt.get("proof") or {}
        steps = proof.get("steps") or []
        if steps and (steps[-1].get("state") or []):
            return str(steps[-1]["state"][0])
    return None


def _load_role_calls(runs_dir: str, run_id: str) -> list[dict]:
    artifact_root = os.path.join(runs_dir, "artifacts", run_id)
    calls = []
    for meta_path in sorted(glob.glob(
        os.path.join(artifact_root, "*", "call_*", "meta.json")
    )):
        meta = _load_json(meta_path)
        if isinstance(meta, dict):
            meta["_meta_path"] = meta_path
            calls.append(meta)
    return calls


# ── Derivations ──────────────────────────────────────────────────

def _proof_uses_declared(proof: dict) -> bool:
    return any("uses" in (s or {}) for s in (proof.get("steps") or []))


def _witness_states(proof: dict) -> list[list[str]]:
    states = [list(proof.get("initial_state") or [])]
    for step in proof.get("steps") or []:
        states.append([str(e) for e in (step.get("state") or [])])
    return states


def canonical_witness_hash(proof: dict) -> str:
    """Content hash over the full state trajectory, whitespace- and
    id-spacing-normalized, for deduplicating equivalent witnesses."""
    trajectory = [
        [diffing.content_hash(element) for element in state]
        for state in _witness_states(proof)
    ]
    payload = json.dumps(trajectory, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_match(answer, gold) -> bool | None:
    if answer is None or gold in (None, ""):
        return None
    normalize = lambda v: " ".join(str(v).casefold().split())  # noqa: E731
    return normalize(answer) == normalize(gold)


def _step_issue_refs(attempt: dict) -> list[dict]:
    """Failed step numbers with their issue indices — what triggered the
    following repair round."""
    refs = []
    state0 = attempt.get("state0_verdict")
    verdict_rows = [state0] if state0 else []
    verdict_rows += [v for v in (attempt.get("verdicts") or []) if v]
    for verdict in verdict_rows:
        if verdict.get("accepted"):
            continue
        issues = verdict.get("issues")
        refs.append({
            "step_number": verdict.get("step_number"),
            "role": verdict.get("role"),
            "reason": verdict.get("reason"),
            "issue_indices": (
                list(range(1, len(issues) + 1))
                if isinstance(issues, list) else None
            ),
        })
    return refs


def _reward_row(base: dict, attempt: dict, gold: dict | None) -> dict:
    proof = attempt.get("proof") or {}
    steps = proof.get("steps") or []
    uses_declared = _proof_uses_declared(proof)
    states = _witness_states(proof)
    final_answer = (
        str(states[-1][0]) if states and states[-1] else None
    )
    format_valid = bool(
        steps and final_answer and final_answer.strip()
        and final_answer.strip().upper() != "ANSWER"
    )
    state0 = attempt.get("state0_verdict") or {}
    s0_pass = state0.get("accepted") if state0 else None

    licensed_steps = 0
    unlicensed_mutations = 0
    for index, step in enumerate(steps):
        prev_state = states[index]
        mutations = diffing.diff_states(prev_state, states[index + 1])
        if not uses_declared:
            continue
        validation = diffing.validate_step_references(
            step, prev_state, step_number=index + 1,
        )
        if "uses" in step and validation.ok:
            licensed_steps += 1
        else:
            unlicensed_mutations += len(mutations)

    # key_match grades the run's final answer, but this row is per
    # verify attempt — only trust it when this attempt's answer is the
    # one that was graded, otherwise failed intermediate attempts on
    # eventually-solved problems inherit a false +1.
    ternary = None
    if gold is not None:
        graded = gold.get("graded_answer")
        matches_graded = _strict_match(final_answer, graded) is True
        if gold.get("key_match") is True and matches_graded:
            ternary = 1
        elif gold.get("key_match") is False and matches_graded:
            ternary = -1
    if ternary is None and gold is not None and final_answer is not None:
        strict = _strict_match(final_answer, gold.get("gold_answer"))
        if strict is not None:
            ternary = 1 if strict else -1
    if ternary is None and not format_valid:
        ternary = 0

    return {
        **base,
        "format_valid": format_valid,
        "s0_pass": s0_pass,
        "all_ok": attempt.get("all_ok"),
        "licensed_step_fraction": (
            licensed_steps / len(steps) if uses_declared and steps else None
        ),
        "unlicensed_mutation_count": (
            unlicensed_mutations if uses_declared else None
        ),
        "orphan_count": (
            len(diffing.find_orphans(proof)) if uses_declared else None
        ),
        "terminal_reward": ternary,
        "gold_answer": (gold or {}).get("gold_answer"),
    }


# ── Export core ──────────────────────────────────────────────────

def export_run(
    runs_dir: str,
    run_id: str,
    out_dir: str,
    *,
    grades: dict[str, dict[str, dict]] | None = None,
) -> dict[str, int]:
    """Export one run. Returns row counts per output for reporting."""
    artifact_root = os.path.join(runs_dir, "artifacts", run_id)
    run_meta = _load_json(os.path.join(artifact_root, "meta.json")) or {}
    config_hash = run_meta.get("config_hash")
    run_grades = (grades or {}).get(run_id, {})

    results = _load_problem_results(runs_dir, run_id)
    role_call_metas = _load_role_calls(runs_dir, run_id)

    tables: dict[str, list[dict]] = {name: [] for name in TABLE_NAMES}
    training: dict[str, list[dict]] = {name: [] for name in TRAINING_FILES}

    # role_calls straight from per-call meta.json artifacts.
    for meta in role_call_metas:
        usage = meta.get("usage") or {}
        cost = meta.get("cost") or {}
        tables["role_calls"].append({
            "run_id": run_id,
            "problem_id": meta.get("problem_id"),
            "config_hash": meta.get("config_hash") or config_hash,
            "call_id": meta.get("call_id"),
            "role": meta.get("role"),
            "backend": meta.get("backend"),
            "model": meta.get("model"),
            "model_alias": meta.get("model_alias") or meta.get("model_config"),
            "started_at": meta.get("started_at"),
            "duration_ms": meta.get("duration_ms"),
            "retry_count": meta.get("retry_count"),
            "failed": meta.get("failed"),
            "resumed": meta.get("resumed"),
            "has_schema": meta.get("has_schema"),
            "usage_reported": meta.get("usage_reported"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get(
                "cache_creation_input_tokens"
            ),
            "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost_usd": cost.get("amount_usd"),
            "cost_source": cost.get("source"),
            "rollout_group": meta.get("rollout_group"),
            "sample_index": meta.get("sample_index"),
            "artifact_dir": meta.get("artifact_dir") or os.path.dirname(
                meta.get("_meta_path", "")
            ),
        })

    for problem_id, result in sorted(results.items()):
        problem_text = result.get("problem")
        gold = run_grades.get(problem_id)
        if gold is None and str(result.get("expected") or "").strip():
            gold = {"gold_answer": result.get("expected"), "key_match": None}
        rollout_group = result.get("rollout_group")
        sample_index = result.get("sample_index")

        def base_row(attempt_number=None) -> dict:
            row = {
                "run_id": run_id,
                "problem_id": problem_id,
                "attempt": attempt_number,
                "config_hash": result.get("config_hash") or config_hash,
            }
            if rollout_group is not None:
                row["rollout_group"] = rollout_group
            if sample_index is not None:
                row["sample_index"] = sample_index
            return row

        attempts = result.get("attempts") or []
        verify_attempts = [
            a for a in attempts if a.get("phase") == "verify"
        ]
        last_verify = verify_attempts[-1] if verify_attempts else None

        for attempt in attempts:
            attempt_number = attempt.get("attempt")
            phase = attempt.get("phase")
            if phase != "verify":
                continue
            # One-pass verifier arms (answer_score / resolve_agreement) seal
            # proof:null with the real answer in result["answer"] and have NO proof
            # witness. Skip the proof-centric tables for them (witnesses/dedup here;
            # the steps/mutations/premise_edges loop below is naturally empty; rewards
            # is guarded at its append) -- a degenerate empty proof would DROP the
            # answer, collapse every one-pass witness_hash to one constant (dedup
            # collision), and mis-score reward. The judge verdict IS still captured.
            proof_dict = attempt.get("proof")
            one_pass = not proof_dict
            proof = proof_dict or {}
            steps = proof.get("steps") or []
            states = _witness_states(proof)
            uses_declared = _proof_uses_declared(proof)
            if one_pass:
                final_answer = result.get("answer")
            else:
                witness_hash = canonical_witness_hash(proof)
                final_answer = (
                    str(states[-1][0]) if states and states[-1] else None
                )
                certified = bool(attempt.get("all_ok"))
                tables["witnesses"].append({
                    **base_row(attempt_number),
                    "initial_state": json.dumps(proof.get("initial_state") or []),
                    "num_steps": len(steps),
                    "final_answer": final_answer,
                    "all_ok": attempt.get("all_ok"),
                    "certified": certified,
                    "uses_declared": uses_declared,
                    "witness_hash": witness_hash,
                })
                tables["dedup"].append({
                    **base_row(attempt_number),
                    "witness_hash": witness_hash,
                    "certified": certified,
                })

            for index, step in enumerate(steps):
                step_number = index + 1
                prev_state = states[index]
                curr_state = states[index + 1]
                tables["steps"].append({
                    **base_row(attempt_number),
                    "step_number": step_number,
                    "justification_type": step.get("justification_type"),
                    "justification": step.get("justification"),
                    "uses": (
                        json.dumps(step["uses"]) if "uses" in step else None
                    ),
                    "prev_state": json.dumps(prev_state),
                    "state": json.dumps(curr_state),
                })
                for mutation in diffing.diff_states(prev_state, curr_state):
                    tables["mutations"].append({
                        **base_row(attempt_number),
                        "step_number": step_number,
                        "kind": mutation.kind,
                        "element_id": mutation.element_id,
                        "before": mutation.before,
                        "after": mutation.after,
                    })
                for claim_id, used_id in diffing.premise_edges(
                    step, prev_state,
                ):
                    tables["premise_edges"].append({
                        **base_row(attempt_number),
                        "step_number": step_number,
                        "claim_id": claim_id,
                        "used_id": used_id,
                    })

            # Judge verdicts (state 0 + per step).
            verdict_rows = []
            if attempt.get("state0_verdict"):
                verdict_rows.append(attempt["state0_verdict"])
            verdict_rows += [v for v in (attempt.get("verdicts") or []) if v]
            for verdict in verdict_rows:
                issues = verdict.get("issues")
                step_number = verdict.get("step_number")
                tables["judge_calls"].append({
                    **base_row(attempt_number),
                    "step_number": step_number,
                    "role": verdict.get("role"),
                    "accepted": verdict.get("accepted"),
                    "reason": verdict.get("reason"),
                    "num_issues": len(issues) if isinstance(issues, list) else None,
                    "user_msg": verdict.get("user_msg"),
                })
                tables["verdicts"].append({
                    **base_row(attempt_number),
                    "step_number": step_number,
                    "role": verdict.get("role"),
                    "accepted": verdict.get("accepted"),
                    "reason": verdict.get("reason"),
                    "gold_answer": (gold or {}).get("gold_answer"),
                    "gold_key_match": (gold or {}).get("key_match"),
                    "strict_match": _strict_match(
                        final_answer, (gold or {}).get("gold_answer"),
                    ),
                })
                for issue_index, issue in enumerate(issues or [], start=1):
                    if not isinstance(issue, dict):
                        continue
                    tables["issues"].append({
                        **base_row(attempt_number),
                        "step_number": step_number,
                        "issue_index": issue_index,
                        "error_class": issue.get("error_class"),
                        "severity": issue.get("severity"),
                        "description": issue.get("description"),
                    })

                # verifier_steps: state0 has no step payload, skip it there.
                if step_number and step_number >= 1 and step_number <= len(steps):
                    step = steps[step_number - 1]
                    training["verifier_steps.jsonl"].append({
                        **base_row(attempt_number),
                        "problem": problem_text,
                        "prev_state": states[step_number - 1],
                        "state": states[step_number],
                        "justification_type": step.get("justification_type"),
                        "justification": step.get("justification"),
                        "uses": step.get("uses"),
                        "accepted": verdict.get("accepted"),
                        "issues": issues,
                        "reason": verdict.get("reason"),
                    })

            for record in attempt.get("pedantry") or []:
                tables["filter_decisions"].append({
                    **base_row(attempt_number),
                    "step_number": record.get("step_number"),
                    "is_pedantic": record.get("is_pedantic"),
                    "original_reason": record.get("original_reason"),
                    "pedantry_reason": record.get("pedantry_reason"),
                    "pedantic_issue_indices": json.dumps(
                        record["pedantic_issue_indices"]
                    ) if record.get("pedantic_issue_indices") is not None
                    else None,
                    "substantive_issue_indices": json.dumps(
                        record["substantive_issue_indices"]
                    ) if record.get("substantive_issue_indices") is not None
                    else None,
                })
            for record in attempt.get("conventions") or []:
                tables["convention_lifts"].append({
                    **base_row(attempt_number),
                    "step_number": record.get("step_number"),
                    "can_lift": record.get("can_lift"),
                    "convention": record.get("convention"),
                    "source": record.get("source"),
                    "reasoning": record.get("reasoning"),
                })

            if not one_pass:  # no proof -> no proof-reward row (see the witness guard)
                tables["rewards"].append(
                    _reward_row(base_row(attempt_number), attempt, gold)
                )

        # repair_attempts + sft_repair: consecutive attempt transitions.
        for index in range(len(attempts) - 1):
            current, following = attempts[index], attempts[index + 1]
            triggering = (
                _step_issue_refs(current)
                if current.get("phase") == "verify" else None
            )
            tables["repair_attempts"].append({
                **base_row(current.get("attempt")),
                "to_attempt": following.get("attempt"),
                "from_phase": current.get("phase"),
                "to_phase": following.get("phase"),
                "from_all_ok": current.get("all_ok"),
                "to_all_ok": following.get("all_ok"),
                "triggering_issues": (
                    json.dumps(triggering) if triggering is not None else None
                ),
                "reject_reason": current.get("reject_reason"),
                "invalid_reason": (
                    current.get("reason")
                    if current.get("phase") == "formalizer_invalid" else None
                ),
            })
            if (
                current.get("phase") == "verify"
                and current.get("all_ok") is False
                and following.get("phase") == "verify"
            ):
                training["sft_repair.jsonl"].append({
                    **base_row(current.get("attempt")),
                    "to_attempt": following.get("attempt"),
                    "problem": problem_text,
                    "failed_witness": current.get("proof"),
                    "verdicts": [
                        {
                            "step_number": v.get("step_number"),
                            "role": v.get("role"),
                            "accepted": v.get("accepted"),
                            "reason": v.get("reason"),
                            "issues": v.get("issues"),
                        }
                        for v in (
                            ([current["state0_verdict"]]
                             if current.get("state0_verdict") else [])
                            + [x for x in (current.get("verdicts") or []) if x]
                        )
                    ],
                    "corrected_witness": following.get("proof"),
                    "corrected_certified": bool(following.get("all_ok")),
                })

        # sft_witness: problem -> solver CoT + final witness.
        if last_verify is not None and last_verify.get("proof"):
            training["sft_witness.jsonl"].append({
                **base_row(last_verify.get("attempt")),
                "problem": problem_text,
                "solver_cot": result.get("solution"),
                "witness": last_verify.get("proof"),
                "final_answer": result.get("answer") or _partial_answer(
                    {"attempts": attempts},
                ),
                "certified": bool(last_verify.get("all_ok")),
                "certified_unconditionally": result.get(
                    "verified_unconditionally"
                ),
                "assumptions": result.get("verified_under_assumptions"),
                "gold_answer": (gold or {}).get("gold_answer"),
                "gold_key_match": (gold or {}).get("key_match"),
            })

        # decline: runs that ended without a certified witness, with the
        # failure localization as the decline rationale.
        if not result.get("verified"):
            rationale_parts = []
            for attempt in attempts:
                if attempt.get("phase") == "formalizer_reject":
                    rationale_parts.append(
                        f"Formalizer rejected the solution: "
                        f"{attempt.get('reject_reason', '')}"
                    )
                elif attempt.get("phase") == "verify" and not attempt.get("all_ok"):
                    for ref in _step_issue_refs(attempt):
                        rationale_parts.append(
                            f"Step {ref['step_number']} rejected by "
                            f"{ref['role']}: {ref['reason']}"
                        )
            if rationale_parts:
                training["decline.jsonl"].append({
                    **base_row(
                        attempts[-1].get("attempt") if attempts else None
                    ),
                    "problem": problem_text,
                    "final_answer_candidate": result.get("answer"),
                    "decline_rationale": "\n\n".join(rationale_parts),
                    "gold_answer": (gold or {}).get("gold_answer"),
                })

    counts = _write_outputs(out_dir, run_id, tables, training)
    return counts


def _write_outputs(
    out_dir: str,
    run_id: str,
    tables: dict[str, list[dict]],
    training: dict[str, list[dict]],
) -> dict[str, int]:
    export_root = Path(out_dir) / run_id
    tables_dir = export_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, rows in tables.items():
        counts[name] = len(rows)
        if HAVE_PYARROW:
            _write_parquet(tables_dir / f"{name}.parquet", rows)
        else:
            _write_jsonl(tables_dir / f"{name}.jsonl", rows)
    for name, rows in training.items():
        counts[name] = len(rows)
        _write_jsonl(export_root / name, rows)
    return counts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not rows:
        # Preserve the file so downstream globs stay uniform.
        pq.write_table(pa.table({}), path)
        return
    columns = sorted({key for row in rows for key in row})
    data = {key: [row.get(key) for row in rows] for key in columns}
    pq.write_table(pa.table(data), path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Export training-grade datasets from Verifier runs.",
    )
    parser.add_argument(
        "run_ids", nargs="*",
        help="Run ids to export (default: every run under runs/artifacts).",
    )
    parser.add_argument("--runs-dir", default="runs", help="Runs directory.")
    parser.add_argument("--out", default="exports", help="Output directory.")
    args = parser.parse_args(argv)

    run_ids = args.run_ids or discover_runs(args.runs_dir)
    if not run_ids:
        parser.exit(1, "no runs found to export\n")
    grades = _load_grades(args.runs_dir)
    fmt = "parquet" if HAVE_PYARROW else "jsonl"
    for run_id in run_ids:
        counts = export_run(args.runs_dir, run_id, args.out, grades=grades)
        nonzero = {k: v for k, v in counts.items() if v}
        print(f"[{run_id}] tables as {fmt} -> {args.out}/{run_id}/")
        for name in TABLE_NAMES + TRAINING_FILES:
            marker = "*" if counts.get(name) else " "
            print(f"  {marker} {name}: {counts.get(name, 0)} rows")
        if not nonzero:
            print("  (no exportable data found)")


if __name__ == "__main__":
    main()
