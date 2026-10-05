"""LLM-based answer grader.

The pipeline's built-in `correct` flag is a naive substring match — fine for
a quick scan. This grader is a stronger (LLM-judge) check: it shows an LLM
grader the problem, expected answer, canonical rationale, and one selected
candidate, and asks whether they conceptually match (handling equivalent
notations, algebraic forms, unordered sets, etc.). Internal traces and the
candidate's repair condition are hidden from the grader.

It reads a sealed run JSON directly rather than the historical audit database.
The condition-neutral prompt is intentionally stricter than the historical
trace-aware grader so the same evaluator can compare R0 and R1 candidates.

Output per problem:
    final.key_match         — did the selected answer match the key?
    final.dispute_category  — if not a plain match, why (convention,
                              interpretation, tighter_bound, edge_case,
                              other, none)
    attempts               — always empty for condition-blind grading
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import harness
from llm import (
    artifact_dir,
    call_log,
    effective_settings,
    llm as _llm_call,
    telemetry_context,
)
from loaders import HLE_DATASET_NAME, HLE_DATASET_REVISION, HLE_DATASET_SPLIT
from pipeline import load_config
from telemetry import aggregate_calls, load_pricing


# ── Structured output schemas ──

ATTEMPT_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "attempt_index": {"type": "integer"},
        "key_match": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["attempt_index", "key_match", "reasoning"],
}

FINAL_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "key_match": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "dispute_category": {
            "type": "string",
            "enum": [
                "none", "convention", "interpretation", "tighter_bound",
                "edge_case", "other",
            ],
        },
    },
    "required": ["key_match", "reasoning", "dispute_category"],
}

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "final": FINAL_GRADE_SCHEMA,
        "attempts": {
            "type": "array",
            "maxItems": 0,
            "items": ATTEMPT_GRADE_SCHEMA,
        },
    },
    "required": ["final", "attempts"],
}

SOLVER_FINAL_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "key_match": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "dispute_category": {
            "type": "string",
            "enum": [
                "none", "convention", "interpretation", "tighter_bound",
                "edge_case", "other",
            ],
        },
    },
    "required": ["key_match", "reasoning", "dispute_category"],
}

SOLVER_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "final": SOLVER_FINAL_GRADE_SCHEMA,
        "attempts": {
            "type": "array",
            "maxItems": 0,
            "items": ATTEMPT_GRADE_SCHEMA,
        },
    },
    "required": ["final", "attempts"],
}

FINAL_PROMPT_PATH = Path(__file__).parent / "grader_prompt.md"
# R0 and R1 must use exactly the same condition-blind candidate rubric.
FIRST_ATTEMPT_PROMPT_PATH = FINAL_PROMPT_PATH
SOLVER_PROMPT_PATH = Path(__file__).parent / "grader_solver_prompt.md"
DEFAULT_GRADER_CONFIG = str(Path(__file__).parent / "configs" / "audit_grader.yaml")


def load_prompt(target: str = "final") -> tuple[str, str]:
    """Return the target-specific model prompt and its SHA-256."""
    if target in {"final", "first_attempt"}:
        path = FINAL_PROMPT_PATH
    elif target == "solver_initial":
        path = SOLVER_PROMPT_PATH
    else:
        raise ValueError(f"unsupported grading target: {target}")
    text = path.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    return text, sha


def schema_for_target(target: str) -> dict:
    if target in {"final", "first_attempt"}:
        return GRADE_SCHEMA
    if target == "solver_initial":
        return SOLVER_GRADE_SCHEMA
    raise ValueError(f"unsupported grading target: {target}")


# ── Turn one run-JSON result into the grader's text input ─────────

def _attempt_summary(attempt: dict) -> tuple[str, str]:
    """Reduce one attempt to (terminal_state, justification) for the grader.
    Shows the full terminal state (see format_input for how this differs from
    the internal audit grader)."""
    if attempt.get("phase") == "formalizer_reject":
        return "(formalizer rejected the solution)", attempt.get("reject_reason", "")
    proof = attempt.get("proof") or {}
    steps = proof.get("steps") or []
    if steps:
        last = steps[-1]
        return str(last.get("state")), str(last.get("justification", ""))
    return "(no steps)", ""


def fetch_rationales(
    ids: set[str],
    revision: str = HLE_DATASET_REVISION,
    *,
    include_provenance: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], dict]:
    """Pull the canonical HLE rationale (and HLE's reviewer validity flags)
    for the given problem ids from the `skylenage/HLE-Verified` dataset, so
    the grader sees the same context the internal audit grader had.

    Returns {id: {"rationale", "is_valid", "error_type"}}. Degrades to {} if
    `datasets` isn't installed or the fetch fails (e.g. a custom-only run, or
    offline) — the grader then just sees no rationale, which is correct for
    questions that have none.
    """
    if not ids:
        empty_provenance = {
            "name": HLE_DATASET_NAME,
            "split": HLE_DATASET_SPLIT,
            "revision": revision,
            "fingerprint": None,
            "available": False,
            "records": 0,
        }
        return ({}, empty_provenance) if include_provenance else {}
    try:
        from datasets import load_dataset
        ds = load_dataset(
            HLE_DATASET_NAME,
            split=HLE_DATASET_SPLIT,
            revision=revision,
        )
    except Exception:
        return ({}, {
            "name": HLE_DATASET_NAME,
            "split": HLE_DATASET_SPLIT,
            "revision": revision,
            "fingerprint": None,
            "available": False,
        }) if include_provenance else {}
    out: dict[str, dict] = {}
    for ex in ds:
        if ex["id"] not in ids:
            continue
        try:
            meta = json.loads(ex["json"])
        except Exception:
            meta = {}
        out[ex["id"]] = {
            "rationale": meta.get("rationale") or "",
            "is_valid": ex.get("rationale_is_valid"),
            "error_type": ex.get("rationale_error_type"),
        }
    provenance = {
        "name": HLE_DATASET_NAME,
        "split": HLE_DATASET_SPLIT,
        "revision": revision,
        "fingerprint": getattr(ds, "_fingerprint", None),
        "available": True,
        "records": len(out),
    }
    return (out, provenance) if include_provenance else out


def _result_for_target(result: dict, target: str) -> dict:
    """Return the auditable view of a source result selected for grading."""
    if target == "final":
        view = dict(result)
        view.update({"attempts": [], "grading_target": target})
        return view
    if target == "solver_initial":
        solutions = result.get("solver_solutions") or []
        if not solutions or not str(solutions[0]).strip():
            raise RuntimeError("source result is missing an initial solver response")
        selected = solutions[0]
        view = dict(result)
        view.update({
            "answer": selected,
            "attempts": [],
            "verified": None,
            "grading_target": target,
        })
        return view
    if target == "first_attempt":
        repair = result.get("repair_metrics")
        answer = repair.get("first_attempt_answer") if isinstance(repair, dict) else None
        attempts = result.get("attempts") or []
        selected_attempt = attempts[0] if attempts else None
        if (
            answer is None
            or not str(answer).strip()
            or not selected_attempt
            or selected_attempt.get("phase") != "verify"
        ):
            raise RuntimeError("source result is missing a first formalized candidate")
        view = dict(result)
        view.update({
            "answer": answer,
            "attempts": [],
            "verified": bool(repair.get("first_attempt_verified")),
            "grading_target": target,
        })
        return view
    raise ValueError(f"unsupported grading target: {target}")


def _target_is_missing(result: dict, target: str) -> bool:
    if target == "solver_initial":
        solutions = result.get("solver_solutions") or []
        return not solutions or not str(solutions[0]).strip()
    if target == "first_attempt":
        try:
            _result_for_target(result, target)
        except RuntimeError:
            return True
        return False
    value = result.get("answer")
    return value is None or not str(value).strip()


def _validate_grade_alignment(verdict: dict, target_result: dict) -> None:
    attempts = verdict.get("attempts")
    if not isinstance(attempts, list):
        raise RuntimeError("grader verdict is missing an attempts array")
    expected = list(range(len(target_result.get("attempts") or [])))
    actual = [attempt.get("attempt_index") for attempt in attempts]
    if actual != expected:
        raise RuntimeError(
            "grader attempt indices do not align with the selected source view: "
            f"actual={actual}, expected={expected}"
        )


def format_input(result: dict, rationale: dict | None = None) -> str:
    """Format the per-problem context that goes to the grader.

    The final and first-attempt views deliberately expose the same fields and
    use the same labels. This keeps the external evaluator blind to the repair
    condition. The source traces remain sealed for later human adjudication.
    """
    rat_text = (rationale or {}).get("rationale") or "(no rationale available)"
    riv = (rationale or {}).get("is_valid")
    ret = (rationale or {}).get("error_type")
    rationale_flags = ""
    if riv is not None or ret is not None:
        rationale_flags = (
            f"(HLE's own reviewers flagged: "
            f"rationale_is_valid={riv!r} rationale_error_type={ret!r})\n"
        )

    def display(value: object, missing: str) -> str:
        return missing if value is None or not str(value).strip() else str(value)

    target = result.get("grading_target") or "final"
    target_label = (
        "INITIAL SOLVER RESPONSE:"
        if target == "solver_initial" else "SELECTED CANDIDATE:"
    )
    parts = [
        "PROBLEM TEXT:",
        result.get("problem") or "(missing)",
        "",
        "EXPECTED ANSWER:",
        display(result.get("expected"), "(missing)"),
        "",
        "HLE RATIONALE:",
        rationale_flags + rat_text,
        "",
        target_label,
        display(result.get("answer"), "(no selected answer)"),
    ]
    return "\n".join(parts)


# ── Grading ───────────────────────────────────────────────────────

async def grade_one(
    result: dict, config: dict, prompt_text: str, rationale: dict | None = None,
    *,
    response_schema: dict = GRADE_SCHEMA,
    watch: bool = True,
) -> dict:
    """Grade one problem result. Returns the parsed structured verdict."""
    user_prompt = format_input(result, rationale)
    response, _session = await _llm_call(
        user_prompt,
        role="audit_grader",
        schema=response_schema,
        system=prompt_text,
        config=config,
        watch=watch,
    )
    if not isinstance(response, dict):
        raise RuntimeError(
            f"grader returned {type(response).__name__}, expected dict"
        )
    return response


def _grade_call_metrics(calls: list[dict], duration_ms: int) -> dict:
    calls = [call for call in calls if call is not None]
    input_tokens = sum(int(call.get("input_tokens", 0) or 0) for call in calls)
    output_tokens = sum(int(call.get("output_tokens", 0) or 0) for call in calls)
    failed = sum(bool(call.get("failed")) for call in calls)
    usage_observed = sum(
        bool(call.get("usage_observed"))
        if "usage_observed" in call
        else (call.get("usage") or {}).get("source") != "unavailable"
        for call in calls
    )
    usage_complete = sum(
        bool(call.get("usage_complete"))
        if "usage_complete" in call
        else bool((call.get("usage") or {}).get("complete"))
        for call in calls
    )
    return {
        "duration_ms": duration_ms,
        "num_calls": len(calls),
        "successful_calls": len(calls) - failed,
        "failed_calls": failed,
        "total_call_invocations": sum(
            int(call.get("invocation_count", 1) or 1) for call in calls
        ),
        "failed_call_invocations": sum(
            int(call.get(
                "failed_invocations", int(bool(call.get("failed"))),
            ) or 0)
            for call in calls
        ),
        "resume_retry_count": sum(
            int(call.get("resume_retry_count", 0) or 0) for call in calls
        ),
        "total_retries": sum(
            int(call.get("retry_count", 0) or 0) for call in calls
        ),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "usage_available_calls": usage_observed,
        "usage_coverage_fraction": (
            usage_observed / len(calls) if calls else None
        ),
        "usage_complete_calls": usage_complete,
        "usage_complete_fraction": (
            usage_complete / len(calls) if calls else None
        ),
        "calls_by_model": dict(Counter(
            f"{call.get('backend', '?')}:{call.get('model') or '?'}"
            for call in calls
        )),
    }


def _read_stable_snapshot(path: Path) -> tuple[bytes, str]:
    """Read one immutable view of a file and return (bytes, sha256)."""
    with path.open("rb") as f:
        before = os.fstat(f.fileno())
        data = f.read()
        after = os.fstat(f.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(data) != after.st_size:
        raise RuntimeError(f"source changed while it was being read: {path}")
    return data, hashlib.sha256(data).hexdigest()


def _resolve_source_run_meta(source_path: Path, source_sha256: str) -> tuple[
    Path | None, bytes | None, str | None
]:
    """Find and verify metadata belonging to the supplied run, if present."""
    meta_path = source_path.parent / "artifacts" / source_path.stem / "meta.json"
    if not meta_path.exists():
        return None, None, None
    meta_bytes, meta_sha256 = _read_stable_snapshot(meta_path)
    try:
        source_meta = json.loads(meta_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"source run metadata is invalid JSON: {meta_path}") from exc
    if source_meta.get("run_id") != source_path.stem:
        raise RuntimeError(
            "source metadata run_id does not match the supplied run: "
            f"{source_meta.get('run_id')!r} != {source_path.stem!r}"
        )
    recorded_save_path = source_meta.get("save_path")
    if recorded_save_path:
        recorded_name = Path(recorded_save_path).name
        if recorded_name != source_path.name:
            raise RuntimeError(
                "source metadata names a different run file: "
                f"{recorded_name!r} != {source_path.name!r}"
            )
    if source_meta.get("status") not in {"completed", "completed_with_errors"}:
        raise RuntimeError(
            "refusing to grade a source run that is not complete: "
            f"status={source_meta.get('status')!r}"
        )
    manifest = harness.verify_artifact_manifest(str(meta_path.parent))
    matching_entries = [
        entry for entry in manifest["entries"]
        if entry.get("external")
        and Path(harness.resolve_artifact_manifest_path(
            str(meta_path.parent), entry,
        )) == source_path
    ]
    if len(matching_entries) != 1:
        raise RuntimeError(
            "source run manifest does not contain exactly one entry for "
            f"{source_path}"
        )
    if matching_entries[0].get("sha256") != source_sha256:
        raise RuntimeError(
            "source run bytes do not match the sealed run manifest: "
            f"{source_path}"
        )
    return meta_path, meta_bytes, meta_sha256


def _problem_artifact_name(index: int, problem_id: object) -> str:
    text = str(problem_id)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "problem"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"problem_{index:04d}_{slug[:48]}_{digest}"


def _runs_root_for_output(output_path: Path) -> Path:
    """Keep grade evidence in the same movable runs bundle as its summary."""
    for parent in output_path.parents:
        if parent.name == "runs":
            return parent
    fallback = (Path.cwd() / "runs").resolve()
    print(
        f"warning: --out {output_path} is outside any runs/ tree; grade evidence "
        f"will be written under {fallback}, so the summary and its artifacts will "
        "not move as a single bundle.",
        file=sys.stderr,
    )
    return fallback


async def grade_run(
    run_file: str,
    config_paths: list[str] | None = None,
    out_path: str | None = None,
    *,
    target: str = "final",
    missing_as_incorrect: bool = False,
    watch: bool = True,
    pricing_path: str | None = None,
) -> dict:
    """Grade a stable snapshot of every problem in a completed run."""
    if target not in {"final", "first_attempt", "solver_initial"}:
        raise ValueError(f"unsupported grading target: {target}")
    source_path = Path(run_file).expanduser().resolve(strict=True)
    source_bytes, source_run_hash = _read_stable_snapshot(source_path)
    try:
        results = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"source run is invalid JSON: {source_path}") from exc
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise RuntimeError("source run must contain one result object or a list of objects")
    source_meta_path, source_meta_bytes, source_meta_sha256 = (
        _resolve_source_run_meta(source_path, source_run_hash)
    )

    resolved_config_paths = list(config_paths or [DEFAULT_GRADER_CONFIG])
    config = load_config(resolved_config_paths)
    resolved_pricing_path = pricing_path or os.getenv("OWRE_PRICING_FILE")
    if resolved_pricing_path:
        load_pricing(resolved_pricing_path)
        config.setdefault("_telemetry", {})["pricing_file"] = resolved_pricing_path
    if "audit_grader" not in config:
        raise SystemExit(
            "No 'audit_grader' role in the loaded config. Pass "
            "--config configs/audit_grader.yaml."
        )
    prompt_text, prompt_sha = load_prompt(target)
    response_schema = schema_for_target(target)
    settings = effective_settings(
        config["audit_grader"], web_search_config=config.get("_web_search"),
    )
    grader_model = (
        f"{settings.get('backend', 'claude')}:"
        f"{settings.get('model')}:{settings.get('effort')}"
    )

    started_at = datetime.now(timezone.utc).isoformat()
    started_perf = time.perf_counter()
    grade_prefix = "grade_" if target == "final" else f"grade_{target}_"
    grade_run_id = (
        f"{grade_prefix}{source_path.stem}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    if out_path is None:
        out_path = str(Path("runs/grades") / f"{grade_run_id}.json")
    output_path = Path(out_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or Path(str(output_path) + ".tmp").exists():
        raise FileExistsError(
            f"grading output already exists; choose a new --out path: {output_path}"
        )

    grade_artifact_root = (
        _runs_root_for_output(output_path) / "artifacts" / grade_run_id
    )
    grade_artifact_root.mkdir(parents=True, exist_ok=False)
    (grade_artifact_root / "source_run.json").write_bytes(source_bytes)
    if source_meta_bytes is not None:
        (grade_artifact_root / "source_run_meta.json").write_bytes(source_meta_bytes)
    (grade_artifact_root / "grader_prompt.txt").write_text(prompt_text)
    (grade_artifact_root / "grader_schema.json").write_text(
        json.dumps(response_schema, indent=2)
    )

    audit = harness._research_audit_metadata(config, {
        "config_paths": resolved_config_paths,
        "backend": settings.get("backend"),
        "watch": watch,
        "docker": False,
        "image": None,
    })
    evaluation_policy = {
        "automatic_grader": True,
        "manual_adjudication_included": False,
        "automatic_grade_is_authoritative": False,
        "missing_target_as_incorrect": missing_as_incorrect,
    }
    meta = {
        "audit_schema_version": harness.AUDIT_SCHEMA_VERSION,
        "run_id": grade_run_id,
        "kind": "audit_grading",
        "grading_target": target,
        "status": "running",
        "started_at": started_at,
        "path_base": "artifact_root",
        "source_run_file": os.path.relpath(source_path, grade_artifact_root),
        "source_run_snapshot_path": "source_run.json",
        "source_run_sha256": source_run_hash,
        "source_run_meta_path": (
            os.path.relpath(source_meta_path, grade_artifact_root)
            if source_meta_path else None
        ),
        "source_run_meta_snapshot_path": (
            "source_run_meta.json" if source_meta_bytes is not None else None
        ),
        "source_run_meta_sha256": source_meta_sha256,
        "output_path": os.path.relpath(output_path, grade_artifact_root),
        "grader_model": grader_model,
        "grader_prompt_sha256": prompt_sha,
        "grader_schema_sha256": harness._sha256_json(response_schema),
        "evaluation_policy": evaluation_policy,
        "config": harness._redact_config(config),
        "config_sha256": audit["config_sha256"],
        "pricing": harness._pricing_provenance(config),
        "research_audit": audit,
        "git": harness._safe_git_state(str(grade_artifact_root), "grading"),
        "host_runtime": harness._host_runtime_metadata(),
        "host_environment": harness._capture_host_environment(
            str(grade_artifact_root), "grading",
        ),
    }
    harness._write_meta(str(grade_artifact_root), meta)

    graded: list[dict] = []
    all_calls: list[dict] = []
    n_match = 0
    completed = 0
    rationale_provenance: dict = {}
    run_metrics: dict = {}
    summary: dict | None = None
    try:
        dataset_revisions = {
            result.get("dataset_revision") for result in results
            if result.get("dataset_revision")
        }
        if len(dataset_revisions) > 1:
            raise RuntimeError(
                "source run mixes HLE dataset revisions; grade each revision "
                "separately so rationale provenance remains unambiguous"
            )
        rationale_revision = (
            next(iter(dataset_revisions))
            if dataset_revisions else HLE_DATASET_REVISION
        )
        fetched = fetch_rationales(
            {result.get("id") for result in results if result.get("id")},
            revision=rationale_revision,
            include_provenance=True,
        )
        if isinstance(fetched, tuple):
            rationales, rationale_provenance = fetched
        else:
            rationales, rationale_provenance = fetched, {}
        hle_ids = {
            str(result.get("id")) for result in results
            if result.get("dataset_name") == HLE_DATASET_NAME
        }
        if hle_ids:
            missing_rationales = sorted(hle_ids - set(rationales))
            if (
                rationale_provenance.get("available") is not True
                or missing_rationales
            ):
                raise RuntimeError(
                    "HLE grading requires canonical rationale coverage for "
                    "every source problem; "
                    f"available={rationale_provenance.get('available')!r}, "
                    f"missing_ids={missing_rationales}"
                )
        (grade_artifact_root / "rationales.json").write_text(
            json.dumps(rationales, indent=2, default=str)
        )
        rationale_provenance["records_sha256"] = harness._sha256_json(rationales)
        harness.update_artifact_root_meta(
            str(grade_artifact_root),
            {"rationale_dataset": rationale_provenance},
            strict=True,
        )
        if rationales:
            print(f"Loaded {len(rationales)} HLE rationale(s) for grader context.")

        for i, result in enumerate(results, start=1):
            pid = result.get("id", f"#{i}")
            print(f"[{i}/{len(results)}] grading {pid}...", flush=True)
            artifact_name = _problem_artifact_name(i, pid)
            problem_artifact_dir = grade_artifact_root / artifact_name
            problem_artifact_dir.mkdir(parents=True, exist_ok=False)
            calls: list[dict] = []
            call_token = call_log.set(calls)
            artifact_token = artifact_dir.set(str(problem_artifact_dir))
            trace_token = telemetry_context.set({
                "run_id": grade_run_id,
                "problem_id": str(pid),
            })
            problem_started_at = datetime.now(timezone.utc).isoformat()
            problem_started = time.perf_counter()
            verdict = None
            problem_error: Exception | None = None
            target_result: dict | None = None
            grade_source = "llm"
            try:
                missing_target = _target_is_missing(result, target)
                if missing_as_incorrect and missing_target:
                    target_result = dict(result)
                    target_result.update({
                        "answer": None,
                        "attempts": [],
                        "verified": None,
                        "grading_target": target,
                    })
                    verdict = {
                        "final": {
                            "key_match": False,
                            "reasoning": f"No {target} response was retained.",
                            "dispute_category": "none",
                        },
                        "attempts": [],
                    }
                    grade_source = "deterministic_missing_target"
                    (problem_artifact_dir / "deterministic_verdict.json").write_text(
                        json.dumps(verdict, indent=2)
                    )
                else:
                    target_result = _result_for_target(result, target)
                    verdict = await grade_one(
                        target_result, config, prompt_text, rationales.get(pid),
                        response_schema=response_schema, watch=watch,
                    )
                final = verdict.get("final")
                if not isinstance(final, dict) or not isinstance(
                    final.get("key_match"), bool
                ):
                    raise RuntimeError("grader verdict is missing a valid final grade")
                _validate_grade_alignment(verdict, target_result)
            except Exception as exc:
                problem_error = exc
                (problem_artifact_dir / "traceback.txt").write_text(
                    traceback.format_exc()
                )
            finally:
                call_log.reset(call_token)
                artifact_dir.reset(artifact_token)
                telemetry_context.reset(trace_token)

            calls = [call for call in calls if call is not None]
            duration_ms = int(round(
                (time.perf_counter() - problem_started) * 1000
            ))
            metrics = aggregate_calls(
                calls,
                problem_started_at=problem_started_at,
                problem_ended_at=datetime.now(timezone.utc).isoformat(),
                problem_duration_ms=duration_ms,
            )
            metrics.update(_grade_call_metrics(calls, duration_ms))
            all_calls.extend(calls)
            completed += 1
            if problem_error is not None:
                print(f"  ! failed: {problem_error}")
                graded.append({
                    "id": pid,
                    "artifact_subdir": artifact_name,
                    "error_type": type(problem_error).__name__,
                    "error": str(problem_error),
                    "calls": calls,
                    "metrics": metrics,
                })
                continue
            assert verdict is not None
            assert target_result is not None
            final = verdict["final"]
            n_match += int(final["key_match"])
            print(
                f"  key_match={final['key_match']} "
                f"dispute={final.get('dispute_category')!r}"
            )
            grader_finish_reasons = [
                resp.get("finish_reason")
                for call in calls
                for resp in (call.get("provider_responses") or [])
            ]
            graded.append({
                "id": pid,
                "artifact_subdir": artifact_name,
                "expected": result.get("expected"),
                "answer": target_result.get("answer"),
                "verified": target_result.get("verified"),
                "target": target,
                "grade_source": grade_source,
                "grader_model": grader_model,
                "grader_prompt_sha": prompt_sha,
                # Truncation provenance: the accepted verdict's finish_reason
                # (always "stop" for a schema-valid grade) and whether any turn
                # hit the token cap; analysis asserts no accepted grade was cut.
                "grader_finish_reason": (
                    grader_finish_reasons[-1] if grader_finish_reasons else None
                ),
                "grader_truncated": any(
                    reason == "length" for reason in grader_finish_reasons
                ),
                "grader_output_tokens": metrics.get("total_output_tokens"),
                "calls": calls,
                "metrics": metrics,
                **verdict,
            })

        finished_at = datetime.now(timezone.utc).isoformat()
        run_duration_ms = int(round(
            (time.perf_counter() - started_perf) * 1000
        ))
        run_metrics = aggregate_calls(
            all_calls,
            problem_started_at=started_at,
            problem_ended_at=finished_at,
            problem_duration_ms=run_duration_ms,
            scope="grading_run",
        )
        run_metrics.update(_grade_call_metrics(all_calls, run_duration_ms))
        successful_grades = sum(1 for item in graded if "error" not in item)
        run_metrics.update({
            "scope": "grading_run",
            "requested_grades": len(results),
            "completed_grades": completed,
            "successful_grades": successful_grades,
        })
        telemetry_path = grade_artifact_root / "telemetry.json"
        telemetry_path.write_text(json.dumps(run_metrics, indent=2, default=str))
        summary = {
            "path_base": "grade_summary_parent",
            "run_file": os.path.relpath(source_path, output_path.parent),
            "run_file_sha256": source_run_hash,
            "source_snapshot": os.path.relpath(
                grade_artifact_root / "source_run.json", output_path.parent,
            ),
            "graded": len(results),
            "key_match": n_match,
            "grader_model": grader_model,
            "grading_target": target,
            "grader_prompt_sha": prompt_sha,
            "artifact_root": os.path.relpath(
                grade_artifact_root, output_path.parent,
            ),
            "rationale_dataset": rationale_provenance,
            "evaluation_policy": evaluation_policy,
            "metrics": run_metrics,
            "results": graded,
        }
        tmp_out = Path(str(output_path) + ".tmp")
        tmp_out.write_text(json.dumps(summary, indent=2))
        os.replace(tmp_out, output_path)

        status = "completed" if successful_grades == len(results) else "completed_with_errors"
        final_meta = harness._read_meta(str(grade_artifact_root))
        final_meta.update({
            "status": status,
            "finished_at": finished_at,
            "metrics": run_metrics,
            "telemetry_path": "telemetry.json",
            "completed_problem_ids": [str(item.get("id")) for item in graded],
            "output_sha256": harness._file_sha256(str(output_path)),
        })
        harness._write_meta(str(grade_artifact_root), final_meta)
        harness.write_artifact_manifest(
            str(grade_artifact_root), external_paths=[str(output_path)],
        )
    except BaseException as exc:
        status = (
            "interrupted"
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
            else "failed"
        )
        try:
            failed_meta = harness._read_meta(str(grade_artifact_root))
            failed_meta.update({
                "status": status,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metrics": run_metrics,
                "completed_problem_ids": [str(item.get("id")) for item in graded],
                "run_error": {"type": type(exc).__name__, "message": str(exc)},
            })
            harness._write_meta(str(grade_artifact_root), failed_meta)
            harness.write_artifact_manifest(
                str(grade_artifact_root),
                external_paths=[str(output_path)] if output_path.exists() else [],
            )
        except BaseException:
            pass
        raise

    assert summary is not None
    print(
        f"\nGraded {len(results)} problems: {n_match} matched the key. "
        f"Wrote {output_path}"
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grade a run JSON.")
    parser.add_argument("run_file", help="Path to a run JSON")
    parser.add_argument("--config", action="append", default=None,
                        help="Grader config YAML (repeatable). "
                             f"Default: {DEFAULT_GRADER_CONFIG}")
    parser.add_argument("--out", default=None, help="Output grades JSON path")
    parser.add_argument(
        "--target", choices=("final", "first_attempt", "solver_initial"),
        default="final",
    )
    parser.add_argument("--missing-as-incorrect", action="store_true")
    parser.add_argument(
        "--watch", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--pricing", default=None)
    args = parser.parse_args()
    asyncio.run(grade_run(
        args.run_file, args.config, args.out,
        target=args.target, missing_as_incorrect=args.missing_as_incorrect,
        watch=args.watch, pricing_path=args.pricing,
    ))
