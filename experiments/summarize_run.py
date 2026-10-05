#!/usr/bin/env python3
"""Build a reproducible paper-metric summary from one sealed Verifier run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import harness
from grade import (
    FINAL_PROMPT_PATH,
    FIRST_ATTEMPT_PROMPT_PATH,
    SOLVER_PROMPT_PATH,
)


BOOTSTRAP_SEED = 20260718
BOOTSTRAP_RESAMPLES = 10_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return [center - half, center + half]


def _rate(k: int, n: int) -> dict:
    return {
        "numerator": k,
        "denominator": n,
        "value": k / n if n else None,
        "wilson_95": wilson_interval(k, n),
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _paired_bootstrap(
    records: list[tuple[bool, bool, bool, bool]],
    statistic,
    *,
    seed_offset: int = 0,
) -> dict:
    if not records:
        return {
            "interval_95": None,
            "attempted_resamples": 0,
            "valid_resamples": 0,
            "undefined_resamples": 0,
        }
    generator = random.Random(BOOTSTRAP_SEED + seed_offset)
    estimates = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [records[generator.randrange(len(records))] for _ in records]
        value = statistic(sample)
        if value is not None:
            estimates.append(value)
    low = _percentile(estimates, 0.025)
    high = _percentile(estimates, 0.975)
    valid = len(estimates)
    return {
        "interval_95": (
            [low, high] if low is not None and high is not None else None
        ),
        "attempted_resamples": BOOTSTRAP_RESAMPLES,
        "valid_resamples": valid,
        "undefined_resamples": BOOTSTRAP_RESAMPLES - valid,
    }


def _paired_bootstrap_interval(
    records: list[tuple[bool, bool, bool, bool]],
    statistic,
    *,
    seed_offset: int = 0,
) -> list[float] | None:
    """Compatibility wrapper for analysis code outside this PR's ownership."""
    return _paired_bootstrap(
        records, statistic, seed_offset=seed_offset,
    )["interval_95"]


def _mcnemar_exact(before: list[bool], after: list[bool]) -> dict:
    if len(before) != len(after):
        raise ValueError("paired outcomes must have equal lengths")
    lost = sum(left and not right for left, right in zip(before, after))
    gained = sum(not left and right for left, right in zip(before, after))
    discordant = lost + gained
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(lost, gained) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "lost": lost,
        "gained": gained,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def _selected_candidate_key_match(grade_result: dict) -> bool:
    return bool((grade_result.get("final") or {}).get("key_match"))


def _favorable_correct(grade_result: dict) -> bool:
    final = grade_result.get("final") or {}
    return _selected_candidate_key_match(
        grade_result
    ) or final.get("dispute_category") in {
        "convention", "interpretation", "tighter_bound", "edge_case",
        "extraction", "pipeline_drift", "other",
    }


def _load_sealed_run(path: Path) -> tuple[list[dict], dict, str]:
    path = path.resolve(strict=True)
    results = json.loads(path.read_text())
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list) or not all(isinstance(row, dict) for row in results):
        raise RuntimeError("run JSON must contain an object or list of objects")
    root = path.parent / "artifacts" / path.stem
    meta = json.loads((root / "meta.json").read_text())
    if meta.get("status") not in {"completed", "completed_with_errors"}:
        raise RuntimeError(f"source run is not complete: {meta.get('status')!r}")
    manifest = harness.verify_artifact_manifest(str(root))
    source_sha = _sha256(path)
    matching_entries = [
        entry for entry in manifest.get("entries") or []
        if entry.get("external")
        and Path(harness.resolve_artifact_manifest_path(
            str(root), entry,
        )) == path
    ]
    if len(matching_entries) != 1:
        raise RuntimeError(f"source run file is not uniquely sealed: {path}")
    if matching_entries[0].get("sha256") != source_sha:
        raise RuntimeError(f"source run hash does not match its seal: {path}")
    ids = [str(row.get("id")) for row in results]
    problem_set = meta.get("problem_set") or {}
    if problem_set.get("ids") != ids:
        raise RuntimeError("source result order does not match its problem manifest")
    if (meta.get("research_audit") or {}).get(
        "model_runtime_identity_complete"
    ) is not True:
        raise RuntimeError("source model runtime identity is incomplete")
    return results, meta, source_sha


def _resolve_grade_artifact_root(path: Path, summary: dict) -> Path:
    raw_root = summary.get("artifact_root")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise RuntimeError(f"grade summary has no artifact root: {path}")
    reference = Path(raw_root)
    path_base = summary.get("path_base")
    if path_base == "grade_summary_parent":
        if reference.is_absolute():
            raise RuntimeError(
                f"grade summary path is absolute under a relative path base: {path}"
            )
        return (path.parent / reference).resolve(strict=True)
    if path_base is not None:
        raise RuntimeError(f"unsupported grade summary path base: {path_base!r}")

    # Legacy summaries recorded either an absolute path or a cwd-relative
    # `runs/artifacts/...` path. Prefer the original absolute path when it still
    # exists, then recover a unique location inside a relocated bundle.
    if reference.is_absolute() and reference.is_dir():
        return reference.resolve(strict=True)
    candidates = []
    legacy_reference = (
        Path("runs/artifacts") / reference.name
        if reference.is_absolute() else reference
    )
    for ancestor in path.parents:
        candidate = (ancestor / legacy_reference).resolve()
        if candidate.is_dir() and candidate not in candidates:
            candidates.append(candidate)
    if len(candidates) != 1:
        raise RuntimeError(
            "grade artifact root cannot be resolved uniquely from the grade "
            f"summary: {path}"
        )
    return candidates[0]


def _assert_no_truncated_grades(rows: list[dict], path) -> None:
    """Fail closed if any accepted grade's response was cut at the token cap
    (grader_finish_reason == "length"); the analysis must not trust a
    budget-starved verdict. No-op for grade files predating this field."""
    truncated = [
        str(row.get("id")) for row in rows
        if row.get("grader_finish_reason") == "length"
    ]
    if truncated:
        raise RuntimeError(
            "grade run has truncated grader responses (raise the grader "
            f"token budget): {path} ids={truncated}"
        )


def _load_grades(
    paths: list[Path], source_sha256: str, target: str, expected_ids: set[str],
) -> dict[str, dict[str, dict]]:
    graders: dict[str, dict[str, dict]] = {}
    for path in paths:
        path = path.resolve(strict=True)
        summary = json.loads(path.read_text())
        actual_target = summary.get("grading_target") or "final"
        if actual_target != target:
            raise RuntimeError(
                f"grade target mismatch for {path}: {actual_target!r} != {target!r}"
            )
        if summary.get("run_file_sha256") != source_sha256:
            raise RuntimeError(f"grade source hash mismatch: {path}")
        artifact_root = _resolve_grade_artifact_root(path, summary)
        manifest = harness.verify_artifact_manifest(str(artifact_root))
        meta = json.loads((artifact_root / "meta.json").read_text())
        if meta.get("status") not in {"completed", "completed_with_errors"}:
            raise RuntimeError(f"grade run is not complete: {path}")
        if (meta.get("research_audit") or {}).get(
            "model_runtime_identity_complete"
        ) is not True:
            raise RuntimeError(f"grader model runtime identity is incomplete: {path}")
        matching_entries = [
            entry for entry in manifest.get("entries") or []
            if entry.get("external")
            and Path(harness.resolve_artifact_manifest_path(
                str(artifact_root), entry,
            )) == path
        ]
        if len(matching_entries) != 1:
            raise RuntimeError(f"grade file is not uniquely sealed: {path}")
        if matching_entries[0].get("sha256") != _sha256(path):
            raise RuntimeError(f"grade file hash does not match its seal: {path}")
        expected_prompt_path = {
            "final": FINAL_PROMPT_PATH,
            "first_attempt": FIRST_ATTEMPT_PROMPT_PATH,
            "solver_initial": SOLVER_PROMPT_PATH,
        }[target]
        expected_prompt_sha = _sha256(expected_prompt_path)
        if summary.get("grader_prompt_sha") != expected_prompt_sha:
            raise RuntimeError(f"unexpected grader prompt for {target}: {path}")
        if meta.get("grader_prompt_sha256") != expected_prompt_sha:
            raise RuntimeError(f"grade metadata prompt mismatch: {path}")
        rationale = summary.get("rationale_dataset") or {}
        if expected_ids and (
            rationale.get("available") is not True
            or int(rationale.get("records", 0) or 0) != len(expected_ids)
        ):
            raise RuntimeError(f"grade rationale coverage is incomplete: {path}")
        grader = str(summary.get("grader_model") or path.stem)
        if grader in graders:
            raise RuntimeError(f"duplicate grader for {target}: {grader}")
        rows = summary.get("results") or []
        if any(row.get("error") for row in rows):
            raise RuntimeError(f"grade run contains failed problems: {path}")
        _assert_no_truncated_grades(rows, path)
        by_id = {str(row.get("id")): row for row in rows}
        if set(by_id) != expected_ids:
            raise RuntimeError(f"grade ids do not match source run: {path}")
        graders[grader] = by_id
    return graders


def _grader_budget_stats(*grader_sets) -> dict:
    """Grader token headroom + truncation provenance across all grade rows, so
    the paper can report that no verdict was budget-starved."""
    tokens: list[int] = []
    truncated = 0
    for graders in grader_sets:
        for rows in graders.values():
            for row in rows.values():
                value = row.get("grader_output_tokens")
                if isinstance(value, int):
                    tokens.append(value)
                if row.get("grader_truncated"):
                    truncated += 1
    return {
        "max_output_tokens": max(tokens) if tokens else None,
        "grades_with_a_truncated_turn": truncated,
    }


def _graded_metric(
    graders: dict[str, dict[str, dict]], ids: list[str], eligible: set[str],
    *,
    solver_only: bool = False,
) -> dict:
    output: dict[str, dict] = {}
    selected = [pid for pid in ids if pid in eligible]
    for grader, rows in graders.items():
        selected_matches = sum(
            _selected_candidate_key_match(rows[pid]) for pid in selected
        )
        favorable = selected_matches if solver_only else sum(
            _favorable_correct(rows[pid]) for pid in selected
        )
        output[grader] = {
            "selected_candidate_key_match": _rate(
                selected_matches, len(selected),
            ),
            "favorable_unadjudicated": _rate(favorable, len(selected)),
            "selected_candidate_errors": len(selected) - selected_matches,
        }
    if graders:
        consensus_selected = sum(
            all(
                _selected_candidate_key_match(rows[pid])
                for rows in graders.values()
            )
            for pid in selected
        )
        # Deliberate asymmetry under "all_graders_consensus": the headline
        # selected_candidate_key_match is a conjunction (all graders agree),
        # while favorable_unadjudicated is an any-grader union (a lenient upper
        # bound). Both live under the "consensus" key but mean different things.
        favorable_any = consensus_selected if solver_only else sum(
            any(_favorable_correct(rows[pid]) for rows in graders.values())
            for pid in selected
        )
        output["all_graders_consensus"] = {
            "selected_candidate_key_match": _rate(
                consensus_selected, len(selected),
            ),
            "favorable_unadjudicated": _rate(favorable_any, len(selected)),
            "selected_candidate_errors": len(selected) - consensus_selected,
        }
    return output


def _wrong_rates(graded_metric: dict[str, dict]) -> dict[str, dict]:
    output = {}
    for grader, values in graded_metric.items():
        correct = values["selected_candidate_key_match"]
        output[grader] = _rate(
            correct["denominator"] - correct["numerator"],
            correct["denominator"],
        )
    return output


def _wrong_rate_asymmetry(
    certified_wrong: dict[str, dict], declined_wrong: dict[str, dict],
) -> dict[str, float | None]:
    output = {}
    for grader in sorted(set(certified_wrong) & set(declined_wrong)):
        certified = certified_wrong[grader]["value"]
        declined = declined_wrong[grader]["value"]
        output[grader] = (
            declined / certified
            if certified not in {None, 0} and declined is not None
            else None
        )
    return output


def _paired_correctness_transitions(
    first_graders: dict[str, dict[str, dict]],
    final_graders: dict[str, dict[str, dict]],
    ids: list[str],
) -> dict[str, dict]:
    output = {}
    grader_names = sorted(set(first_graders) & set(final_graders))
    views = grader_names + (["all_graders_consensus"] if grader_names else [])
    for grader in views:
        counts = Counter()
        for problem_id in ids:
            first_rows = (
                [first_graders[name][problem_id] for name in grader_names]
                if grader == "all_graders_consensus"
                else [first_graders[grader][problem_id]]
            )
            final_rows = (
                [final_graders[name][problem_id] for name in grader_names]
                if grader == "all_graders_consensus"
                else [final_graders[grader][problem_id]]
            )
            observed = all(
                row.get("grade_source") != "deterministic_missing_target"
                and row.get("answer") is not None
                and bool(str(row.get("answer")).strip())
                for row in first_rows
            )
            before = all(_selected_candidate_key_match(row) for row in first_rows)
            after = all(_selected_candidate_key_match(row) for row in final_rows)
            before_label = (
                "missing" if not observed
                else "correct" if before
                else "incorrect"
            )
            counts[
                before_label
                + "_to_"
                + ("correct" if after else "incorrect")
            ] += 1
        output[grader] = {
            "problems": len(ids),
            "transition_counts": dict(sorted(counts.items())),
            "incorrect_to_correct_rate": _rate(
                counts["incorrect_to_correct"],
                counts["incorrect_to_correct"] + counts["incorrect_to_incorrect"],
            ),
            "correct_to_incorrect_damage_rate": _rate(
                counts["correct_to_incorrect"],
                counts["correct_to_incorrect"] + counts["correct_to_correct"],
            ),
        }
    return output


# Paired-difference statistics over (r0, r1, c0, c1) records, shared verbatim by
# the HLE (this module) and GPQA (summarize_gpqa) reports so the two reports'
# statistic definitions stay provably identical.
def _coverage_difference(sample):
    return sum(r1 - r0 for r0, r1, _c0, _c1 in sample) / len(sample)


def _correct_certified_rate_difference(sample):
    return sum(c1 - c0 for _r0, _r1, c0, c1 in sample) / len(sample)


def _precision_difference(sample):
    r0_total = sum(r0 for r0, _r1, _c0, _c1 in sample)
    r1_total = sum(r1 for _r0, r1, _c0, _c1 in sample)
    if not r0_total or not r1_total:
        return None
    return (
        sum(c1 for _r0, _r1, _c0, c1 in sample) / r1_total
        - sum(c0 for _r0, _r1, c0, _c1 in sample) / r0_total
    )


def _paired_operating_point_changes(
    ids: list[str],
    r0_ids: set[str],
    r1_ids: set[str],
    first_graders: dict[str, dict[str, dict]],
    final_graders: dict[str, dict[str, dict]],
) -> dict:
    coverage_records = [
        (problem_id in r0_ids, problem_id in r1_ids, False, False)
        for problem_id in ids
    ]

    output = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "coverage_r1_minus_r0": {
            "estimate": (len(r1_ids) - len(r0_ids)) / len(ids) if ids else None,
            "paired_bootstrap": _paired_bootstrap(
                coverage_records, _coverage_difference,
            ),
            "mcnemar": _mcnemar_exact(
                [record[0] for record in coverage_records],
                [record[1] for record in coverage_records],
            ),
        },
        "by_grader": {},
    }
    grader_names = sorted(set(first_graders) & set(final_graders))
    views = grader_names + (["all_graders_consensus"] if grader_names else [])
    for offset, grader in enumerate(views, 1):
        records = []
        for problem_id in ids:
            r0 = problem_id in r0_ids
            r1 = problem_id in r1_ids
            if grader == "all_graders_consensus":
                first_correct = all(
                    _selected_candidate_key_match(
                        first_graders[name][problem_id]
                    )
                    for name in grader_names
                )
                final_correct = all(
                    _selected_candidate_key_match(
                        final_graders[name][problem_id]
                    )
                    for name in grader_names
                )
            else:
                first_correct = _selected_candidate_key_match(
                    first_graders[grader][problem_id]
                )
                final_correct = _selected_candidate_key_match(
                    final_graders[grader][problem_id]
                )
            records.append((
                r0,
                r1,
                r0 and first_correct,
                r1 and final_correct,
            ))

        r0_total = len(r0_ids)
        r1_total = len(r1_ids)
        output["by_grader"][grader] = {
            "correct_and_certified_rate_r1_minus_r0": {
                "estimate": (
                    sum(record[3] - record[2] for record in records) / len(records)
                    if records else None
                ),
                "paired_bootstrap": _paired_bootstrap(
                    records, _correct_certified_rate_difference, seed_offset=offset,
                ),
                "mcnemar": _mcnemar_exact(
                    [record[2] for record in records],
                    [record[3] for record in records],
                ),
            },
            "certified_precision_r1_minus_r0": {
                "estimate": (
                    sum(record[3] for record in records) / r1_total
                    - sum(record[2] for record in records) / r0_total
                    if r0_total and r1_total else None
                ),
                "paired_bootstrap": _paired_bootstrap(
                    records, _precision_difference, seed_offset=1000 + offset,
                ),
            },
        }
    return output


def summarize(
    run_path: Path,
    final_grade_paths: list[Path],
    solver_grade_paths: list[Path],
    first_grade_paths: list[Path] | None = None,
) -> dict:
    results, meta, source_sha = _load_sealed_run(run_path)
    ids = [str(row.get("id")) for row in results]
    if len(ids) != len(set(ids)):
        raise RuntimeError("source run has duplicate problem ids")
    id_set = set(ids)
    final_grades = _load_grades(
        final_grade_paths, source_sha, "final", id_set,
    ) if final_grade_paths else {}
    solver_grades = _load_grades(
        solver_grade_paths, source_sha, "solver_initial", id_set,
    ) if solver_grade_paths else {}
    first_grades = _load_grades(
        first_grade_paths or [], source_sha, "first_attempt", id_set,
    ) if first_grade_paths else {}
    nonempty_grader_sets = [
        set(graders) for graders in (final_grades, first_grades, solver_grades)
        if graders
    ]
    if nonempty_grader_sets and any(
        graders != nonempty_grader_sets[0] for graders in nonempty_grader_sets[1:]
    ):
        raise RuntimeError("grader identities differ across grading targets")

    repair_rows = [
        row["repair_metrics"] for row in results
        if isinstance(row.get("repair_metrics"), dict)
    ]
    missing_repair_error_rows = [
        row for row in results
        if not isinstance(row.get("repair_metrics"), dict) and row.get("error")
    ]
    if len(repair_rows) + len(missing_repair_error_rows) != len(results):
        raise RuntimeError("source run is missing repair metrics")
    r0_ids = {
        str(row.get("id")) for row in results
        if not row.get("error")
        and isinstance(row.get("repair_metrics"), dict)
        and row["repair_metrics"].get("first_attempt_verified")
    }
    r1_ids = {
        str(row.get("id")) for row in results
        if not row.get("error") and row.get("verified")
    }
    declined_ids = id_set - r1_ids
    execution_errors = sum(bool(row.get("error")) for row in results)
    if any(not isinstance(row.get("metrics"), dict) for row in results):
        raise RuntimeError("source run is missing per-problem telemetry")
    repair_attempted = sum(bool(row.get("repair_attempted")) for row in repair_rows)
    retry_certified_ids = set()
    semantic_retry_certified_ids = set()
    nonsemantic_retry_certified_ids = set()
    for row in results:
        repair = row.get("repair_metrics")
        if not isinstance(repair, dict):
            continue
        problem_id = str(row.get("id"))
        certified_after_any_retry = bool(
            repair.get("certified_after_any_retry")
        )
        expected_after_any_retry = bool(
            not row.get("error")
            and row.get("verified")
            and repair.get("repair_attempted")
            and not repair.get("first_attempt_verified")
        )
        if certified_after_any_retry != expected_after_any_retry:
            raise RuntimeError(
                f"{problem_id}: certified_after_any_retry is inconsistent"
            )
        semantic = bool(
            certified_after_any_retry and repair.get("semantic_repair_attempted")
        )
        nonsemantic = bool(repair.get("certified_after_nonsemantic_retry"))
        if nonsemantic != bool(certified_after_any_retry and not semantic):
            raise RuntimeError(
                f"{problem_id}: retry certification decomposition is inconsistent"
            )
        if "certified_after_semantic_repair" in repair and bool(
            repair["certified_after_semantic_repair"]
        ) != semantic:
            raise RuntimeError(
                f"{problem_id}: certified_after_semantic_repair is inconsistent"
            )
        if certified_after_any_retry:
            retry_certified_ids.add(problem_id)
        if semantic:
            semantic_retry_certified_ids.add(problem_id)
        if nonsemantic:
            nonsemantic_retry_certified_ids.add(problem_id)
    # Repair is monotone (R0 certified => R1 certified) for step_wise and the one-pass
    # arms, so R0 is a subset of R1 and R1-R0 == retry-certified. holistic_proof,
    # however, certifies on confidence>=threshold while its repair loop stops on the
    # judge's SEPARATE accept flag, so a later attempt can lower confidence and
    # un-certify a problem that was R0-certified -- R0 is legitimately not a subset of
    # R1 there. Skip the monotonicity assertions for that mode (the R0/R1 operating
    # points below remain well-defined); keep them as a hard invariant otherwise.
    modes = {row.get("verifier_mode") for row in results if not row.get("error")}
    monotone_repair = "holistic_proof" not in modes
    if monotone_repair and not r0_ids <= r1_ids:
        raise RuntimeError("invalid operating points: R0 certification is not in R1")
    if monotone_repair and r1_ids - r0_ids != retry_certified_ids:
        raise RuntimeError(
            "certified_after_any_retry does not match the R1 minus R0 transition"
        )
    research_audit = meta.get("research_audit") or {}
    run_args = research_audit.get("run_args") or {}

    r0_precision = _graded_metric(first_grades, ids, r0_ids)
    r1_precision = _graded_metric(final_grades, ids, r1_ids)
    declined_accuracy = _graded_metric(final_grades, ids, declined_ids)
    certified_wrong = _wrong_rates(r1_precision)
    declined_wrong = _wrong_rates(declined_accuracy)
    repair_attempted_ids = [
        str(row.get("id")) for row in results
        if not row.get("error")
        and isinstance(row.get("repair_metrics"), dict)
        and row["repair_metrics"].get("repair_attempted")
    ]

    domains = {}
    domain_names = sorted({str(row.get("category") or "unknown") for row in results})
    for domain in domain_names:
        domain_ids = {
            str(row.get("id")) for row in results
            if str(row.get("category") or "unknown") == domain
        }
        domains[domain] = {
            "problems": len(domain_ids),
            "solver_accuracy": _graded_metric(
                solver_grades, ids, domain_ids, solver_only=True,
            ),
            "r0_coverage": _rate(len(r0_ids & domain_ids), len(domain_ids)),
            "r0_certified_precision": _graded_metric(
                first_grades, ids, r0_ids & domain_ids,
            ),
            "r1_coverage": _rate(len(r1_ids & domain_ids), len(domain_ids)),
            "r1_certified_precision": _graded_metric(
                final_grades, ids, r1_ids & domain_ids,
            ),
        }

    tools = Counter()
    tool_status_counts = Counter()
    evidence_by_role: dict[str, Counter] = {}
    web_search_providers: set[str] = set()
    web_search_totals = Counter()
    usage_complete_calls = 0
    calls = 0
    required_tool_calls = 0
    compliant_required_tool_calls = 0
    required_tool_obligations = 0
    satisfied_required_tool_obligations = 0
    protocol_reprompt_count = 0
    json_action_reprompts = 0
    role_schema_reprompts = 0
    unknown_action_reprompts = 0
    budget_truncated_turns = 0
    budget_truncated_reprompts = 0
    required_tool_policy_reprompts = 0
    telemetry_totals = Counter()
    calls_by_role = Counter()
    calls_by_model = Counter()
    partial_cost_usd = 0.0
    partial_cost_rows = 0
    for row in results:
        metrics = row.get("metrics") or {}
        tools.update(metrics.get("tool_calls_by_name") or {})
        tool_status_counts.update(metrics.get("tool_status_counts") or {})
        web_search_providers.update(metrics.get("web_search_providers") or [])
        for field in (
            "web_search_requests", "web_search_attempts",
            "web_search_provider_requests", "web_search_successes",
            "web_search_failures", "web_search_client_failures",
            "web_search_provider_failures", "web_search_result_count",
            "web_search_total_latency_ms",
        ):
            web_search_totals[field] += int(metrics.get(field, 0) or 0)
        calls += int(metrics.get("num_calls", 0) or 0)
        usage_complete_calls += int(metrics.get("usage_complete_calls", 0) or 0)
        required_tool_calls += int(metrics.get("required_tool_calls", 0) or 0)
        compliant_required_tool_calls += int(
            metrics.get("compliant_required_tool_calls", 0) or 0
        )
        required_tool_obligations += int(
            metrics.get("required_tool_obligations", 0) or 0
        )
        satisfied_required_tool_obligations += int(
            metrics.get("satisfied_required_tool_obligations", 0) or 0
        )
        protocol_reprompt_count += int(
            metrics.get("protocol_reprompt_count", 0) or 0
        )
        json_action_reprompts += int(
            metrics.get("json_action_reprompts", 0) or 0
        )
        role_schema_reprompts += int(
            metrics.get("role_schema_reprompts", 0) or 0
        )
        unknown_action_reprompts += int(
            metrics.get("unknown_action_reprompts", 0) or 0
        )
        budget_truncated_turns += int(
            metrics.get("budget_truncated_turns", 0) or 0
        )
        budget_truncated_reprompts += int(
            metrics.get("budget_truncated_reprompts", 0) or 0
        )
        required_tool_policy_reprompts += int(
            metrics.get("required_tool_policy_reprompts", 0) or 0
        )
        for field in (
            "successful_calls", "failed_calls", "total_call_invocations",
            "failed_call_invocations", "resume_retry_count", "total_retries",
            "problem_duration_ms", "total_llm_duration_ms",
            "total_input_tokens", "total_output_tokens",
            "total_cache_read_input_tokens", "total_tokens",
            "usage_available_calls", "priced_calls",
        ):
            telemetry_totals[field] += int(metrics.get(field, 0) or 0)
        calls_by_role.update(metrics.get("calls_by_role") or {})
        calls_by_model.update(metrics.get("calls_by_model") or {})
        if metrics.get("partial_cost_usd") is not None:
            partial_cost_usd += float(metrics["partial_cost_usd"])
            partial_cost_rows += 1
        for role, counts in (metrics.get("mechanistic_evidence_by_role") or {}).items():
            evidence_by_role.setdefault(role, Counter()).update(counts or {})

    certification_stages: dict[str, set[str]] = {}
    for row in results:
        if row.get("error") or not row.get("verified"):
            continue
        metrics = row.get("repair_metrics") or {}
        if metrics.get("first_attempt_verified"):
            stage = "r0_first_pipeline_attempt"
        elif int(metrics.get("verify_attempts", 0) or 0) <= 1:
            stage = "repair_before_first_verification"
        else:
            stage = f"verification_attempt_{int(metrics['verify_attempts'])}"
        certification_stages.setdefault(stage, set()).add(str(row.get("id")))
    certification_stage_report = {
        stage: {
            "new_certifications": len(stage_ids),
            "marginal_coverage": _rate(len(stage_ids), len(results)),
            "certified_precision": _graded_metric(
                final_grades, ids, stage_ids,
            ),
        }
        for stage, stage_ids in sorted(certification_stages.items())
    }

    report = {
        "schema_version": 1,
        "source_run_name": run_path.name,
        "source_run_sha256": source_sha,
        "run_id": meta.get("run_id"),
        "experiment": {
            "id": run_args.get("experiment_id"),
            "phase": run_args.get("experiment_phase"),
            "trial": run_args.get("trial"),
        },
        "model_provenance": {
            "roles": research_audit.get("role_model_manifest"),
            "runtime_identity": research_audit.get("model_runtime_identity"),
            "runtime_identity_complete": research_audit.get(
                "model_runtime_identity_complete"
            ),
        },
        "sandbox": meta.get("sandbox"),
        "grading_provenance": _grader_budget_stats(
            final_grades, first_grades, solver_grades,
        ),
        "cohort": {
            "problems": len(results),
            "ordered_ids_sha256": harness._sha256_json(ids),
            "dataset_revisions": (meta.get("problem_set") or {}).get(
                "dataset_revisions"
            ),
            "dataset_fingerprints": (meta.get("problem_set") or {}).get(
                "dataset_fingerprints"
            ),
            "execution_errors": execution_errors,
            "intent_to_treat_denominator": len(results),
            "repair_metrics_complete_problems": len(repair_rows),
            "repair_metrics_missing_execution_errors": len(
                missing_repair_error_rows
            ),
        },
        "solver_only": {
            "accuracy": _graded_metric(
                solver_grades, ids, id_set, solver_only=True,
            ),
        },
        "operating_points": {
            "r0_first_pipeline_attempt": {
                "coverage": _rate(len(r0_ids), len(results)),
                "certified_precision": r0_precision,
            },
            "r1_final": {
                "coverage": _rate(len(r1_ids), len(results)),
                "certified_precision": r1_precision,
            },
        },
        "paired_changes": _paired_operating_point_changes(
            ids, r0_ids, r1_ids, first_grades, final_grades,
        ),
        "selection": {
            "certified_wrong_rate": certified_wrong,
            "declined_candidate_accuracy": declined_accuracy,
            "declined_wrong_rate": declined_wrong,
            "declined_to_certified_wrong_rate_ratio": _wrong_rate_asymmetry(
                certified_wrong, declined_wrong,
            ),
        },
        "repair": {
            "metric_scope": {
                "complete_repair_metric_rows": len(repair_rows),
                "missing_execution_error_rows": len(missing_repair_error_rows),
                "note": (
                    "Process counts use complete repair-metric rows and are lower "
                    "bounds. Complete-case yield can be optimistic; the conservative "
                    "yield treats every missing execution-error row as an attempted, "
                    "uncertified repair. Coverage always uses the full ITT cohort."
                ),
            },
            "attempted": repair_attempted,
            "certified_after_any_retry": len(retry_certified_ids),
            "certified_after_semantic_repair": len(
                semantic_retry_certified_ids
            ),
            "semantic_repair_attempted": sum(
                bool(row.get("semantic_repair_attempted"))
                for row in repair_rows
            ),
            "post_judge_repair_attempted": sum(
                bool(row.get("post_judge_repair_attempted"))
                for row in repair_rows
            ),
            "formalizer_invalid_retry_attempted": sum(
                bool(row.get("formalizer_invalid_retry_attempted"))
                for row in repair_rows
            ),
            "certified_after_nonsemantic_retry": len(
                nonsemantic_retry_certified_ids
            ),
            "yield": _rate(len(retry_certified_ids), repair_attempted),
            "yield_missing_errors_as_attempted": _rate(
                len(retry_certified_ids),
                repair_attempted + len(missing_repair_error_rows),
            ),
            "certified_after_any_retry_precision": _graded_metric(
                final_grades, ids, retry_certified_ids,
            ),
            "certified_after_semantic_repair_precision": _graded_metric(
                final_grades, ids, semantic_retry_certified_ids,
            ),
            "certified_after_nonsemantic_retry_precision": _graded_metric(
                final_grades, ids, nonsemantic_retry_certified_ids,
            ),
            "correctness_transitions": _paired_correctness_transitions(
                first_grades, final_grades, repair_attempted_ids,
            ),
            "solver_retries": sum(
                int(row.get("solver_retries", 0) or 0) for row in repair_rows
            ),
            "judge_repair_rounds": sum(
                int(row.get("judge_repair_rounds", 0) or 0)
                for row in repair_rows
            ),
            "formalizer_reject_count": sum(
                int(row.get("formalizer_reject_count", 0) or 0)
                for row in repair_rows
            ),
            "formalizer_invalid_count": sum(
                int(row.get("formalizer_invalid_count", 0) or 0)
                for row in repair_rows
            ),
            "answer_changed": sum(
                value is True
                for value in (
                    row.get("answer_changed_during_repair") for row in repair_rows
                )
            ),
            "answer_change_unobservable": sum(
                row.get("repair_attempted")
                and not row.get("answer_change_observable")
                for row in repair_rows
            ),
            "normalized_answer_changed": sum(
                row.get("normalized_answer_changed_during_repair") is True
                for row in repair_rows
            ),
            "certified_answer_flips": sum(
                row.get("certified_after_any_retry")
                and row.get("answer_changed_during_repair") is True
                for row in repair_rows
            ),
            "certified_normalized_answer_flips": sum(
                row.get("certified_after_any_retry")
                and row.get("normalized_answer_changed_during_repair") is True
                for row in repair_rows
            ),
            "judge_repair_round_distribution": dict(sorted(Counter(
                int(row.get("judge_repair_rounds", 0) or 0)
                for row in repair_rows
            ).items())),
            "certification_stage": certification_stage_report,
        },
        "per_domain": domains,
        "mechanistic_evidence": {
            "tool_calls_by_name": dict(sorted(tools.items())),
            "tool_status_counts": dict(sorted(tool_status_counts.items())),
            "by_role": {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(evidence_by_role.items())
            },
            "required_tool_call_compliance": _rate(
                compliant_required_tool_calls, required_tool_calls,
            ),
            "required_tool_obligation_compliance": _rate(
                satisfied_required_tool_obligations, required_tool_obligations,
            ),
            "web_search": {
                **dict(sorted(web_search_totals.items())),
                "providers": sorted(web_search_providers),
                "mean_provider_latency_ms": (
                    web_search_totals["web_search_total_latency_ms"]
                    / web_search_totals["web_search_provider_requests"]
                    if web_search_totals["web_search_provider_requests"]
                    else None
                ),
            },
            "fetched_source_urls_available": False,
            "note": (
                "Tool status is transport-level evidence. Search results are "
                "snippet leads; this loop does not retain fetched source bodies."
            ),
        },
        "telemetry": {
            "calls": calls,
            "successful_calls": telemetry_totals["successful_calls"],
            "failed_calls": telemetry_totals["failed_calls"],
            "total_call_invocations": telemetry_totals[
                "total_call_invocations"
            ],
            "failed_call_invocations": telemetry_totals[
                "failed_call_invocations"
            ],
            "resume_retry_count": telemetry_totals["resume_retry_count"],
            "total_retries": telemetry_totals["total_retries"],
            "calls_by_role": dict(sorted(calls_by_role.items())),
            "calls_by_model": dict(sorted(calls_by_model.items())),
            "problem_duration_ms_sum": telemetry_totals[
                "problem_duration_ms"
            ],
            "total_llm_duration_ms": telemetry_totals[
                "total_llm_duration_ms"
            ],
            "total_input_tokens": telemetry_totals["total_input_tokens"],
            "total_output_tokens": telemetry_totals["total_output_tokens"],
            "total_cache_read_input_tokens": telemetry_totals[
                "total_cache_read_input_tokens"
            ],
            "total_tokens": telemetry_totals["total_tokens"],
            "usage_available_calls": telemetry_totals[
                "usage_available_calls"
            ],
            "usage_complete_calls": usage_complete_calls,
            "usage_complete_fraction": (
                usage_complete_calls / calls if calls else None
            ),
            "protocol_reprompt_count": protocol_reprompt_count,
            "json_action_reprompts": json_action_reprompts,
            "role_schema_reprompts": role_schema_reprompts,
            "unknown_action_reprompts": unknown_action_reprompts,
            # Budget exhaustion, aggregated to the RUN level. harness.run_one
            # already rolls these up per problem; without this a reader of the
            # run summary sees protocol_reprompt_count with no denominator for
            # the failures deliberately kept out of it, which is how the old
            # `protocol_reprompt_count == 0` gate read as clean while turns were
            # dying at the token cap. Never fold these into that counter.
            "budget_truncated_turns": budget_truncated_turns,
            "budget_truncated_reprompts": budget_truncated_reprompts,
            "required_tool_policy_reprompts": required_tool_policy_reprompts,
            "priced_calls": telemetry_totals["priced_calls"],
            "cost_coverage_fraction": (
                telemetry_totals["priced_calls"] / calls if calls else None
            ),
            "partial_cost_usd": (
                partial_cost_usd if partial_cost_rows else None
            ),
            "total_cost_usd": (
                partial_cost_usd
                if calls and telemetry_totals["priced_calls"] == calls
                else None
            ),
        },
        "graders": {
            "final": sorted(final_grades),
            "first_attempt": sorted(first_grades),
            "solver_initial": sorted(solver_grades),
            "automatic_grades_are_authoritative": False,
        },
    }
    return report


def _format_rate(rate: dict) -> str:
    if rate["value"] is None:
        return "n/a"
    low, high = rate["wilson_95"]
    return (
        f"{rate['numerator']}/{rate['denominator']} "
        f"({100 * rate['value']:.1f}%, 95% CI {100 * low:.1f}-{100 * high:.1f}%)"
    )


def print_report(report: dict) -> None:
    print(f"run: {report['run_id']}")
    print(f"problems: {report['cohort']['problems']}")
    for name, point in report["operating_points"].items():
        print(f"{name} coverage: {_format_rate(point['coverage'])}")
        for grader, values in point["certified_precision"].items():
            print(
                f"{name} strict certified precision [{grader}]: "
                f"{_format_rate(values['selected_candidate_key_match'])}"
            )
    for grader, values in report["solver_only"]["accuracy"].items():
        print(
            f"solver accuracy [{grader}]: "
            f"{_format_rate(values['selected_candidate_key_match'])}"
        )
    print(
        "repair yield: "
        f"{_format_rate(report['repair']['yield'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--final-grade", type=Path, action="append", default=[])
    parser.add_argument("--first-grade", type=Path, action="append", default=[])
    parser.add_argument("--solver-grade", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out and args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    report = summarize(
        args.run, args.final_grade, args.solver_grade, args.first_grade,
    )
    print_report(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
