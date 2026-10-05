"""Research-grade LLM usage and cost accounting.

Provider-reported usage is always retained as the source of truth. Cost is
only populated when the provider reports it or a caller supplies a versioned
pricing catalog; unknown cost remains ``None`` rather than being presented as
zero.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "1.0"
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_output_tokens",
)


def config_for_hash(config: dict[str, Any]) -> dict[str, Any]:
    """Return config suitable for stable cross-machine SHA-256 hashing."""
    return {key: value for key, value in config.items() if key != "_telemetry"}


def normalize_usage(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a stable usage shape while retaining zero-valued dimensions."""
    usage = {}
    for field in TOKEN_FIELDS:
        raw = metadata.get(field, 0)
        if field == "cache_read_input_tokens" and not raw:
            raw = metadata.get("cached_input_tokens", 0)
        usage[field] = max(0, int(raw or 0))
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    reported = (
        bool(metadata.get("provider_usage"))
        or (
            not metadata.get("failed")
            and (
                any(field in metadata for field in TOKEN_FIELDS)
                or "cached_input_tokens" in metadata
            )
        )
    )
    explicit_complete = metadata.get("usage_complete")
    if isinstance(explicit_complete, bool):
        complete = bool(reported and explicit_complete)
    else:
        complete = bool(
            reported and int(metadata.get("retry_count", 0) or 0) == 0
        )
    usage["source"] = (
        "provider_reported_partial" if reported and not complete
        else "provider_reported" if reported
        else "unavailable"
    )
    usage["complete"] = complete
    return usage


@functools.lru_cache(maxsize=8)
def load_pricing(path: str) -> dict[str, Any]:
    """Load and minimally validate a versioned JSON/YAML pricing catalog."""
    catalog_path = Path(path).expanduser().resolve()
    with catalog_path.open() as f:
        if catalog_path.suffix.lower() == ".json":
            catalog = json.load(f)
        else:
            catalog = yaml.safe_load(f)
    if not isinstance(catalog, dict) or not catalog.get("pricing_id"):
        raise ValueError("pricing catalog must contain a non-empty pricing_id")
    if not isinstance(catalog.get("models"), dict):
        raise ValueError("pricing catalog must contain a models mapping")
    valid_semantics = {
        "excludes_cache", "includes_cache_read", "includes_all_cache",
    }
    required_rates = (
        "input_per_million_usd",
        "output_per_million_usd",
        "cache_read_per_million_usd",
        "cache_creation_per_million_usd",
    )
    for model_key, rates in catalog["models"].items():
        if not isinstance(rates, dict):
            raise ValueError(f"pricing entry {model_key!r} must be a mapping")
        if rates.get("input_token_semantics") not in valid_semantics:
            raise ValueError(
                f"pricing entry {model_key!r} has invalid "
                "input_token_semantics"
            )
        for rate_name in required_rates:
            try:
                rate = float(rates[rate_name])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"pricing entry {model_key!r} needs numeric {rate_name}"
                ) from None
            if rate < 0:
                raise ValueError(
                    f"pricing entry {model_key!r} has negative {rate_name}"
                )
    return catalog


def cost_record(
    *,
    backend: str,
    model: str | None,
    usage: dict[str, Any],
    provider_cost_usd: float | None,
    pricing_path: str | None,
    usage_complete: bool,
) -> dict[str, Any]:
    """Build an explicit cost record with provenance and token coverage."""
    if provider_cost_usd is not None:
        return {
            "amount_usd": float(provider_cost_usd),
            "source": "provider_reported",
            "pricing_id": None,
            "currency": "USD",
            "complete": True,
            **({"reason": "retry usage may be unreported"} if not usage_complete else {}),
        }

    if not pricing_path or not usage_complete:
        return {
            "amount_usd": None,
            "source": "unavailable",
            "pricing_id": None,
            "currency": "USD",
            "complete": False,
            **({"reason": "usage unavailable"} if not usage_complete else {}),
        }

    catalog = load_pricing(pricing_path)
    key = f"{backend}:{model}"
    rates = catalog["models"].get(key)
    if not isinstance(rates, dict):
        return {
            "amount_usd": None,
            "source": "unavailable",
            "pricing_id": catalog["pricing_id"],
            "currency": "USD",
            "complete": False,
            "reason": f"no pricing entry for {key}",
        }
    input_semantics = rates.get("input_token_semantics")
    if input_semantics not in {
        "excludes_cache", "includes_cache_read", "includes_all_cache",
    }:
        return {
            "amount_usd": None,
            "source": "unavailable",
            "pricing_id": catalog["pricing_id"],
            "currency": "USD",
            "complete": False,
            "reason": (
                "input_token_semantics must be excludes_cache, "
                "includes_cache_read, or includes_all_cache"
            ),
        }

    rate_fields = {
        "input_tokens": "input_per_million_usd",
        "output_tokens": "output_per_million_usd",
        "cache_read_input_tokens": "cache_read_per_million_usd",
        "cache_creation_input_tokens": "cache_creation_per_million_usd",
    }
    amount = 0.0
    missing_rates: list[str] = []
    for token_field, rate_field in rate_fields.items():
        tokens = usage[token_field]
        if token_field == "input_tokens":
            if input_semantics == "includes_cache_read":
                tokens -= usage["cache_read_input_tokens"]
            elif input_semantics == "includes_all_cache":
                tokens -= (
                    usage["cache_read_input_tokens"]
                    + usage["cache_creation_input_tokens"]
                )
            tokens = max(0, tokens)
        if not tokens:
            continue
        rate = rates.get(rate_field)
        if rate is None:
            missing_rates.append(rate_field)
            continue
        amount += tokens * float(rate) / 1_000_000

    if missing_rates:
        return {
            "amount_usd": None,
            "source": "unavailable",
            "pricing_id": catalog["pricing_id"],
            "currency": "USD",
            "complete": False,
            "reason": f"missing rates: {', '.join(sorted(missing_rates))}",
        }
    return {
        "amount_usd": round(amount, 12),
        "source": "model_pricing_estimate",
        "pricing_id": catalog["pricing_id"],
        "currency": "USD",
        "complete": True,
    }


def enrich_call(
    call: dict[str, Any],
    *,
    pricing_path: str | None = None,
) -> dict[str, Any]:
    """Attach the stable telemetry schema to a call metadata record."""
    usage = normalize_usage(call)
    cost = cost_record(
        backend=call.get("backend", "unknown"),
        model=call.get("model"),
        usage=usage,
        provider_cost_usd=call.get("total_cost_usd"),
        pricing_path=pricing_path,
        usage_complete=bool(usage["complete"]),
    )
    call["telemetry_schema_version"] = SCHEMA_VERSION
    call["usage"] = usage
    call["cost"] = cost
    return call


def _timing_fields(
    scope: str | None,
    *,
    started_at: str,
    ended_at: str,
    duration_ms: int,
) -> dict[str, Any]:
    if scope in {"run", "grading_run"}:
        return {
            "run_started_at": started_at,
            "run_ended_at": ended_at,
            "run_duration_ms": duration_ms,
        }
    return {
        "problem_started_at": started_at,
        "problem_ended_at": ended_at,
        "problem_duration_ms": duration_ms,
    }


def aggregate_calls(
    calls: list[dict[str, Any]],
    *,
    problem_started_at: str,
    problem_ended_at: str,
    problem_duration_ms: int,
    scope: str | None = None,
) -> dict[str, Any]:
    """Aggregate calls without hiding missing usage or cost coverage."""
    totals = {field: 0 for field in TOKEN_FIELDS}
    calls_by_role: dict[str, int] = {}
    calls_by_model: dict[str, int] = {}
    usage_by_role: dict[str, dict[str, int]] = {}
    duration_by_role: dict[str, int] = {}
    tool_calls_by_name: dict[str, int] = {}
    total_llm_duration_ms = 0
    total_retries = 0
    failed_calls = 0
    cost_total = 0.0
    priced_calls = 0
    complete_cost_calls = 0
    cost_sources: dict[str, int] = {}
    usage_complete_calls = 0
    web_search_providers: set[str] = set()
    web_search_attempts = 0
    web_search_provider_requests = 0
    web_search_successes = 0
    web_search_failures = 0
    web_search_client_failures = 0
    web_search_provider_failures = 0
    web_search_result_count = 0
    web_search_total_latency_ms = 0
    web_search_error_categories: dict[str, int] = {}

    for call in calls:
        role = str(call.get("role", "unknown"))
        model_key = f"{call.get('backend', 'unknown')}:{call.get('model') or 'unknown'}"
        usage = call.get("usage") or normalize_usage(call)
        usage_complete_calls += int(bool(usage.get("complete")))
        calls_by_role[role] = calls_by_role.get(role, 0) + 1
        calls_by_model[model_key] = calls_by_model.get(model_key, 0) + 1
        role_usage = usage_by_role.setdefault(
            role, {field: 0 for field in (*TOKEN_FIELDS, "total_tokens")}
        )
        for field in TOKEN_FIELDS:
            value = int(usage.get(field, 0) or 0)
            totals[field] += value
            role_usage[field] += value
        role_usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

        duration = int(call.get("duration_ms", 0) or 0)
        total_llm_duration_ms += duration
        duration_by_role[role] = duration_by_role.get(role, 0) + duration
        total_retries += int(call.get("retry_count", 0) or 0)
        failed_calls += int(bool(call.get("failed")))
        for tool in call.get("tool_calls") or []:
            name = str(tool.get("tool_name") or "unknown")
            tool_calls_by_name[name] = tool_calls_by_name.get(name, 0) + 1

        cost = call.get("cost") or {}
        source = str(cost.get("source") or "unavailable")
        cost_sources[source] = cost_sources.get(source, 0) + 1
        if cost.get("amount_usd") is not None:
            cost_total += float(cost["amount_usd"])
            priced_calls += 1
        complete_cost_calls += int(bool(cost.get("complete")))

        call_attempts = int(
            call.get("web_search_attempts", call.get("web_search_requests", 0))
            or 0
        )
        call_successes = int(call.get("web_search_successes", 0) or 0)
        call_provider_failures = int(
            call.get("web_search_provider_failures", 0) or 0
        )
        call_provider_requests = call.get("web_search_provider_requests")
        if call_provider_requests is None:
            call_provider_requests = call_successes + call_provider_failures
        web_search_attempts += call_attempts
        web_search_provider_requests += int(call_provider_requests or 0)
        web_search_successes += call_successes
        web_search_failures += int(call.get("web_search_failures", 0) or 0)
        web_search_client_failures += int(
            call.get("web_search_client_failures", 0) or 0
        )
        web_search_provider_failures += call_provider_failures
        web_search_result_count += int(
            call.get("web_search_result_count", 0) or 0
        )
        web_search_total_latency_ms += int(
            call.get(
                "web_search_total_latency_ms",
                call.get("web_search_latency_ms", 0),
            )
            or 0
        )
        provider = call.get("web_search_provider")
        if provider and call_attempts:
            web_search_providers.add(str(provider))
        for category, count in (
            call.get("web_search_error_categories") or {}
        ).items():
            web_search_error_categories[str(category)] = (
                web_search_error_categories.get(str(category), 0)
                + int(count or 0)
            )

    total_tokens = totals["input_tokens"] + totals["output_tokens"]
    tokens_by_role = {
        role: usage["input_tokens"] + usage["output_tokens"]
        for role, usage in usage_by_role.items()
    }
    return {
        "telemetry_schema_version": SCHEMA_VERSION,
        **_timing_fields(
            scope,
            started_at=problem_started_at,
            ended_at=problem_ended_at,
            duration_ms=problem_duration_ms,
        ),
        "total_llm_duration_ms": total_llm_duration_ms,
        "num_calls": len(calls),
        "successful_calls": len(calls) - failed_calls,
        "failed_calls": failed_calls,
        "usage_complete_calls": usage_complete_calls,
        "usage_coverage_fraction": (
            usage_complete_calls / len(calls) if calls else None
        ),
        "total_retries": total_retries,
        "calls_by_role": calls_by_role,
        "calls_by_model": calls_by_model,
        "tokens_by_role": tokens_by_role,
        "usage_by_role": usage_by_role,
        "duration_by_role": duration_by_role,
        "tool_calls_by_name": tool_calls_by_name,
        "total_input_tokens": totals["input_tokens"],
        "total_output_tokens": totals["output_tokens"],
        "total_cache_read_input_tokens": totals["cache_read_input_tokens"],
        "total_cache_creation_input_tokens": totals["cache_creation_input_tokens"],
        "total_reasoning_output_tokens": totals["reasoning_output_tokens"],
        "total_tokens": total_tokens,
        "web_search_providers": sorted(web_search_providers),
        # Backward-compatible alias: both fields count model-issued attempts,
        # including client-side misuse and unavailable-tool calls.
        "web_search_requests": web_search_attempts,
        "web_search_attempts": web_search_attempts,
        "web_search_provider_requests": web_search_provider_requests,
        "web_search_successes": web_search_successes,
        "web_search_failures": web_search_failures,
        "web_search_client_failures": web_search_client_failures,
        "web_search_provider_failures": web_search_provider_failures,
        "web_search_result_count": web_search_result_count,
        "web_search_latency_ms": web_search_total_latency_ms,
        "web_search_total_latency_ms": web_search_total_latency_ms,
        "web_search_mean_latency_ms": (
            web_search_total_latency_ms / web_search_provider_requests
            if web_search_provider_requests else None
        ),
        "web_search_error_categories": web_search_error_categories,
        "cost": {
            "amount_usd": round(cost_total, 12) if priced_calls else None,
            "currency": "USD",
            "complete": bool(calls) and complete_cost_calls == len(calls),
            "priced_calls": priced_calls,
            "complete_cost_calls": complete_cost_calls,
            "total_calls": len(calls),
            "coverage_fraction": priced_calls / len(calls) if calls else None,
            "sources": cost_sources,
        },
        # Backward-compatible field. It is intentionally null unless every
        # call is priced, preventing partial cost from looking complete.
        "total_cost_usd": (
            round(cost_total, 12)
            if calls and complete_cost_calls == len(calls)
            else None
        ),
    }
