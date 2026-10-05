"""Shared run infrastructure: config application, parallel execution with
incremental save, result wrapping, and summary printing.

The CLI (cli.py) loads problems (via loaders.py) and drives this module.
Each problem dict must have 'id', 'question', and 'answer' (the expected
answer, used only for the naive correctness check — empty for custom
questions); any extra fields are carried through to the saved result.
apply_args() applies config, run_problems() executes with bounded
parallelism + incremental save, and print_summary() reports.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pipeline
import sandbox as sbx
from llm import (
    artifact_dir,
    call_log,
    effective_settings,
    sandbox_container,
    sandbox_image_id,
    telemetry_context,
)
from observability import attach_events_file, detach_events_file, event, get_logger
from pipeline import run, CONFIG, load_config
from telemetry import (
    SCHEMA_VERSION as TELEMETRY_SCHEMA_VERSION,
    aggregate_calls,
    config_for_hash,
    load_pricing,
)


logger = get_logger("harness")


# ── Run-level metadata capture ───────────────────────────────────

_REDACTED = "<redacted>"
AUDIT_SCHEMA_VERSION = "1.3"
PROBLEM_CHECKPOINT_FILENAME = "_problem_checkpoint.json"
INTERRUPTED_ATTEMPTS_DIR = "_interrupted_attempts"
RETRY_CHECKPOINTS_DIR = "_retry_checkpoints"


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pricing_provenance(config: dict | None = None) -> dict | None:
    resolved_config = CONFIG if config is None else config
    path = (resolved_config.get("_telemetry") or {}).get("pricing_file")
    if not path:
        path = os.getenv("OWRE_PRICING_FILE")
    if not path:
        return None
    expanded = os.path.expanduser(path)
    try:
        with open(expanded, "rb") as f:
            raw = f.read()
    except OSError:
        return {"path": path, "sha256": None, "available": False}
    return {
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "available": True,
    }


def _sha256_json(value) -> str:
    return _sha256_text(_canonical_json(value))


def _file_sha256(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _redact_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc:
        return value
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        port = ""
    return urllib.parse.urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))


def _is_secret_key(key: str) -> bool:
    key = key.lower().replace("-", "_")
    if key.endswith("_env"):
        return False
    secret_names = {
        "api_key",
        "key",
        "access_token",
        "refresh_token",
        "token",
        "secret",
        "password",
        "credential",
        "credentials",
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "client_secret",
        "private_key",
        "connection_string",
        "dsn",
    }
    return key in secret_names or any(key.endswith(f"_{name}") for name in secret_names)


def _redact_config_value(key: str | None, value):
    if key and _is_secret_key(key):
        return _REDACTED if value else value
    if key in {"endpoint", "base_url"} and isinstance(value, str):
        return _redact_url(value)
    if isinstance(value, dict):
        return {
            k: _redact_config_value(str(k), v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_redact_config_value(None, item) for item in value]
    return value


def _redact_config(config: dict) -> dict:
    return _redact_config_value(None, config)


def _config_identity_value(key: str | None, value):
    """Return a secret-safe value whose hash still detects behavior drift."""
    if key and _is_secret_key(key):
        return "<secret-present>" if value else value
    if key in {"endpoint", "base_url"} and isinstance(value, str):
        return {
            "redacted": _redact_url(value),
            "value_sha256": _sha256_text(value),
        }
    if isinstance(value, dict):
        return {
            k: _config_identity_value(str(k), v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_config_identity_value(None, item) for item in value]
    return value


def _effective_role_settings(config: dict, settings: dict) -> dict:
    return effective_settings(
        settings,
        web_search_config=config.get("_web_search"),
    )


def _role_items(config: dict) -> list[tuple[str, dict]]:
    return [
        (role, settings)
        for role, settings in sorted(config.items())
        if not role.startswith("_") and isinstance(settings, dict)
    ]


def _role_model_manifest(config: dict) -> dict:
    manifest = {}
    for role, settings in _role_items(config):
        resolved = _effective_role_settings(config, settings)
        redacted = _redact_config_value(role, resolved)
        endpoint = resolved.get("endpoint") or resolved.get("base_url")
        manifest[role] = {
            "backend": resolved.get("backend", "claude"),
            "model": resolved.get("model"),
            "model_revision": resolved.get("model_revision"),
            "model_source": resolved.get("model_source"),
            "deployment_version": resolved.get("deployment_version"),
            "quantization": resolved.get("quantization"),
            "dtype": resolved.get("dtype"),
            "endpoint": _redact_url(str(endpoint)) if endpoint else None,
            "endpoint_sha256": _sha256_text(str(endpoint)) if endpoint else None,
            "effort": resolved.get("effort"),
            "temperature": resolved.get("temperature"),
            "top_p": resolved.get("top_p"),
            "top_k": resolved.get("top_k"),
            "min_p": resolved.get("min_p"),
            "seed": resolved.get("seed"),
            "reasoning_effort": resolved.get("reasoning_effort"),
            "system_role": resolved.get("system_role", "system"),
            "chat_template_kwargs": resolved.get("chat_template_kwargs"),
            "max_tokens": resolved.get("max_tokens"),
            "max_completion_tokens": resolved.get("max_completion_tokens"),
            "context_length": resolved.get("context_length"),
            "max_turns": resolved.get("max_turns"),
            "search": resolved.get("search"),
            "allow_shell": resolved.get("allow_shell"),
            "allow_host_tools": resolved.get("allow_host_tools"),
            "settings_sha256": _sha256_json(
                _config_identity_value(role, resolved)
            ),
            "settings": redacted,
        }
    return manifest


def _probe_ollama_models(config: dict) -> list[dict]:
    """Resolve local Ollama aliases to immutable digests when available."""
    plans: dict[tuple[str, str], set[str]] = {}
    for role, settings in _role_items(config):
        resolved = _effective_role_settings(config, settings)
        if resolved.get("backend") != "reasoning_agent":
            continue
        endpoint = str(resolved.get("endpoint") or resolved.get("base_url") or "")
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            continue
        if parsed.port not in {None, 11434}:
            continue
        origin = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        plans.setdefault((origin, str(resolved.get("model"))), set()).add(role)

    probes = []
    tags_by_origin: dict[str, dict | Exception] = {}
    for (origin, model), roles in sorted(plans.items()):
        if origin not in tags_by_origin:
            try:
                with urllib.request.urlopen(f"{origin}/api/tags", timeout=2) as response:
                    tags_by_origin[origin] = json.loads(response.read())
            except Exception as exc:
                tags_by_origin[origin] = exc
        payload = tags_by_origin[origin]
        probe = {
            "backend": "ollama",
            "endpoint": _redact_url(origin),
            "model": model,
            "roles": sorted(roles),
        }
        if isinstance(payload, Exception):
            probe.update({
                "resolved": False,
                "error_type": type(payload).__name__,
                "error": str(payload)[:500],
            })
        else:
            candidates = payload.get("models") or []
            match = next((
                item for item in candidates
                if item.get("name") == model or item.get("model") == model
            ), None)
            if match is None and ":" not in model:
                match = next((
                    item for item in candidates
                    if item.get("name") == f"{model}:latest"
                ), None)
            if match is None:
                probe.update({"resolved": False, "error": "model not found"})
            else:
                probe.update({
                    "resolved": True,
                    "resolved_name": match.get("name") or match.get("model"),
                    "digest": match.get("digest"),
                    "size_bytes": match.get("size"),
                    "modified_at": match.get("modified_at"),
                    "details": match.get("details") or {},
                })
        probes.append(probe)
    return probes


def _probe_openai_compatible_models(config: dict) -> list[dict]:
    """Capture model descriptors from local vLLM/OpenAI-compatible servers."""
    probes = []
    seen: set[tuple[str, str]] = set()
    resolved_roles = [
        (role, _effective_role_settings(config, settings))
        for role, settings in _role_items(config)
    ]
    for role, resolved in resolved_roles:
        if resolved.get("backend") != "reasoning_agent":
            continue
        endpoint = str(resolved.get("endpoint") or resolved.get("base_url") or "")
        try:
            parsed = urllib.parse.urlsplit(endpoint)
            port = parsed.port
        except ValueError:
            continue
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            continue
        if port in {None, 11434}:
            continue
        model = str(resolved.get("model"))
        key = (endpoint, model)
        if key in seen:
            continue
        seen.add(key)
        probe = {
            "backend": "openai-compatible",
            "endpoint": _redact_url(endpoint),
            "model": model,
            "roles": sorted(
                candidate_role
                for candidate_role, candidate in resolved_roles
                if (
                    candidate.get("backend") == "reasoning_agent"
                    and str(candidate.get("endpoint") or candidate.get("base_url") or "")
                    == endpoint
                    and str(candidate.get("model")) == model
                )
            ),
        }
        try:
            request = urllib.request.Request(endpoint.rstrip("/") + "/models")
            api_key_env = resolved.get("api_key_env")
            api_key = os.environ.get(api_key_env) if api_key_env else resolved.get("api_key")
            if api_key:
                request.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(request, timeout=2) as response:
                body = json.loads(response.read())
            models = body.get("data") or []
            match = next((item for item in models if item.get("id") == model), None)
            probe.update({
                "resolved": match is not None,
                "descriptor": match,
                "response_sha256": _sha256_json(body),
            })
            if match is None:
                probe["available_model_ids"] = [
                    item.get("id") for item in models[:50]
                ]
        except Exception as exc:
            probe.update({
                "resolved": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
        probes.append(probe)
    return probes


_OFFICIAL_AZURE_ENDPOINT_SUFFIXES = (
    "cognitiveservices.azure.com",
    "openai.azure.com",
    "services.ai.azure.com",
)


def _official_azure_endpoint_account(endpoint: str) -> str | None:
    """Return the account label for an official Azure inference hostname."""
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme.casefold() != "https":
            return None
        hostname = (parsed.hostname or "").rstrip(".")
    except ValueError:
        return None
    hostname = hostname.casefold()
    for suffix in _OFFICIAL_AZURE_ENDPOINT_SUFFIXES:
        marker = "." + suffix
        if hostname == suffix:
            return ""
        if hostname.endswith(marker):
            return hostname[:-len(marker)]
    return None


def _probe_remote_models(config: dict) -> list[dict]:
    """Verify declared remote model identity, including Azure control plane."""
    # Roles that share an endpoint+deployment must also agree on the identity
    # fields recorded as provenance; a silent disagreement is a config bug.
    identity_fields = (
        "model_source", "model_revision", "deployment_version",
        "azure_subscription_id", "azure_resource_group", "azure_account",
    )
    grouped: dict[tuple[str, str], dict] = {}
    for role, settings in _role_items(config):
        resolved = _effective_role_settings(config, settings)
        if resolved.get("backend") != "reasoning_agent":
            continue
        endpoint = str(resolved.get("endpoint") or resolved.get("base_url") or "")
        try:
            hostname = urllib.parse.urlsplit(endpoint).hostname
        except ValueError:
            hostname = None
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            continue
        key = (endpoint, str(resolved.get("model")))
        entry = grouped.get(key)
        if entry is None:
            grouped[key] = {"settings": resolved, "roles": [role]}
            continue
        mismatched = [
            field for field in identity_fields
            if entry["settings"].get(field) != resolved.get(field)
        ]
        if mismatched:
            raise RuntimeError(
                f"roles {entry['roles'] + [role]} share endpoint+model {key} "
                f"but declare different {mismatched}"
            )
        entry["roles"].append(role)

    probes = []
    for (endpoint, deployment), entry in sorted(grouped.items()):
        settings = entry["settings"]
        source = settings.get("model_source")
        probe = {
            "backend": "azure-foundry" if source == "azure-foundry" else "remote-declared",
            "endpoint": _redact_url(endpoint),
            "model": deployment,
            "roles": sorted(entry["roles"]),
            "declared_model_source": source,
            "declared_model_revision": settings.get("model_revision"),
            "declared_deployment_version": settings.get("deployment_version"),
        }
        if source != "azure-foundry":
            probe.update({
                "resolved": False,
                "error_type": "UnverifiedRemoteModel",
                "error": "remote model has no supported live control-plane probe",
            })
            probes.append(probe)
            continue

        resource_group = settings.get("azure_resource_group")
        account = settings.get("azure_account")
        subscription = settings.get("azure_subscription_id")
        expected_upgrade = settings.get("deployment_upgrade_policy")
        if not all((resource_group, account, subscription, expected_upgrade)):
            probe.update({
                "resolved": False,
                "error_type": "IncompleteAzureIdentity",
                "error": (
                    "Azure identity requires azure_resource_group, azure_account, "
                    "azure_subscription_id, and deployment_upgrade_policy"
                ),
            })
            probes.append(probe)
            continue

        endpoint_account = _official_azure_endpoint_account(endpoint)
        if endpoint_account is None:
            probe.update({
                "resolved": False,
                "error_type": "UnverifiableAzureEndpoint",
                "error": (
                    "Azure Foundry identity requires an HTTPS endpoint on an "
                    "official Azure inference hostname"
                ),
                "account": str(account),
            })
            probes.append(probe)
            continue
        if (
            endpoint_account != str(account).casefold()
        ):
            probe.update({
                "resolved": False,
                "error_type": "AzureEndpointAccountMismatch",
                "error": (
                    "Azure endpoint account does not match azure_account: "
                    f"endpoint={endpoint_account!r}, declared={account!r}"
                ),
                "endpoint_account": endpoint_account,
                "account": str(account),
            })
            probes.append(probe)
            continue

        command = [
            "az", "cognitiveservices", "account", "deployment", "show",
            "--subscription", str(subscription),
            "--resource-group", str(resource_group),
            "--name", str(account),
            "--deployment-name", deployment,
            "--output", "json",
        ]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
            )
        except Exception as exc:
            probe.update({
                "resolved": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
            probes.append(probe)
            continue
        if completed.returncode != 0:
            probe.update({
                "resolved": False,
                "error_type": "AzureCLIError",
                "error": completed.stderr.strip()[:500],
            })
            probes.append(probe)
            continue
        try:
            body = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            probe.update({
                "resolved": False,
                "error_type": type(exc).__name__,
                "error": "Azure CLI returned invalid JSON",
            })
            probes.append(probe)
            continue

        properties = body.get("properties") or {}
        model = properties.get("model") or {}
        live_etag = str(body.get("etag") or "").strip('"')
        live_revision = f"{model.get('name')}:{model.get('version')}"
        declared_etag = str(settings.get("deployment_version") or "")
        if declared_etag.startswith("etag-"):
            declared_etag = declared_etag[5:]
        mismatches = []
        comparisons = {
            "model_revision": (
                str(settings.get("model_revision") or ""), live_revision,
            ),
            "deployment_version": (declared_etag, live_etag),
            "deployment_upgrade_policy": (
                str(expected_upgrade),
                str(properties.get("versionUpgradeOption")),
            ),
        }
        for name, (declared, live) in comparisons.items():
            if declared != live:
                mismatches.append({
                    "field": name, "declared": declared, "live": live,
                })
        probe.update({
            "resolved": (
                properties.get("provisioningState") == "Succeeded"
                and properties.get("deploymentState") == "Running"
                and not mismatches
            ),
            "subscription_id": str(subscription),
            "resource_group": str(resource_group),
            "account": str(account),
            "deployment_id": body.get("id"),
            "deployment_etag": live_etag,
            "catalog_model": {
                "format": model.get("format"),
                "name": model.get("name"),
                "version": model.get("version"),
            },
            "version_upgrade_option": properties.get("versionUpgradeOption"),
            "provisioning_state": properties.get("provisioningState"),
            "deployment_state": properties.get("deploymentState"),
            "sku": body.get("sku"),
            "mismatches": mismatches,
            "control_plane_response_sha256": _sha256_json(body),
        })
        if mismatches:
            probe["error_type"] = "AzureIdentityMismatch"
        probes.append(probe)
    return probes


def _prompt_fingerprints(config: dict) -> dict:
    """Fingerprint the prompt each role is actually sent.

    This delegates to `pipeline.agent_prompt`, the single rendering choke point,
    rather than re-deriving the text from raw config. That matters for the verifier
    ablations: `step_wise` and `step_wise_no_completeness` share one `defaults.yaml`,
    so a fingerprint of the *raw* template would be byte-identical across the two arms
    and the recorded provenance could neither show that the completeness clauses were
    stripped nor reveal a strip that silently failed to fire.
    """
    fingerprints = {}
    for role, _ in _role_items(config):
        prompt = pipeline.agent_prompt(role, config)
        fingerprints[role] = {
            "prompt_sha256": _sha256_text(prompt),
            "prompt_chars": len(prompt),
            "preamble_enabled": bool(config.get(role, {}).get("preamble")),
        }
    return fingerprints


def _config_source_fingerprints(paths: list[str]) -> list[dict]:
    return [
        {
            "path": path,
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]


def _stable_model_descriptor(value):
    """Remove server-lifecycle fields that do not identify model weights."""
    volatile_keys = {
        "created",
        "created_at",
        "updated_at",
        "permission",
    }
    if isinstance(value, dict):
        return {
            key: _stable_model_descriptor(item)
            for key, item in sorted(value.items())
            if key not in volatile_keys
        }
    if isinstance(value, list):
        return [_stable_model_descriptor(item) for item in value]
    return value


def _model_runtime_identity(probes: list[dict]) -> list[dict]:
    """Return stable, secret-safe identities from runtime model probes."""
    identities = []
    for probe in probes:
        identity = {
            "backend": probe.get("backend"),
            "endpoint": probe.get("endpoint"),
            "model": probe.get("model"),
            "roles": sorted(probe.get("roles") or []),
            "resolved": bool(probe.get("resolved")),
        }
        if probe.get("backend") == "ollama":
            identity.update({
                "resolved_name": probe.get("resolved_name"),
                "digest": probe.get("digest"),
                "size_bytes": probe.get("size_bytes"),
                "details": _stable_model_descriptor(
                    probe.get("details") or {}
                ),
            })
        elif probe.get("backend") == "openai-compatible":
            identity["descriptor"] = _stable_model_descriptor(
                probe.get("descriptor")
            )
        elif probe.get("backend") in {"azure-foundry", "remote-declared"}:
            identity.update({
                "declared_model_source": probe.get("declared_model_source"),
                "declared_model_revision": probe.get("declared_model_revision"),
                "declared_deployment_version": probe.get(
                    "declared_deployment_version"
                ),
                "subscription_id": probe.get("subscription_id"),
                "resource_group": probe.get("resource_group"),
                "account": probe.get("account"),
                "deployment_id": probe.get("deployment_id"),
                "deployment_etag": probe.get("deployment_etag"),
                "catalog_model": probe.get("catalog_model"),
                "version_upgrade_option": probe.get("version_upgrade_option"),
                "sku": _stable_model_descriptor(probe.get("sku")),
                "control_plane_response_sha256": probe.get(
                    "control_plane_response_sha256"
                ),
            })
        if not identity["resolved"]:
            identity["error_type"] = probe.get("error_type")
        identities.append(identity)
    return sorted(identities, key=_canonical_json)


def _run_args_manifest(args_ref: dict) -> dict:
    cli_args = {
        key: value
        for key, value in (args_ref.get("cli_args") or {}).items()
        if not callable(value)
    }
    return {
        "backend": args_ref.get("backend"),
        "codex_model": args_ref.get("codex_model"),
        "watch": args_ref.get("watch"),
        "docker": args_ref.get("docker"),
        "image": args_ref.get("image"),
        "parallel": args_ref.get("parallel"),
        "resume": args_ref.get("resume"),
        "tag": args_ref.get("tag"),
        "experiment_id": args_ref.get("experiment_id"),
        "experiment_phase": args_ref.get("experiment_phase"),
        "trial": args_ref.get("trial"),
        "samples": args_ref.get("samples"),
        "pricing": args_ref.get("pricing"),
        "config_paths": list(args_ref.get("config_paths") or []),
        "cli_args": _redact_config_value(None, cli_args),
    }


def _run_args_identity(run_args: dict) -> dict:
    """Bind every condition argument except the act of resuming itself."""
    identity = json.loads(json.dumps(run_args, default=str))
    identity.pop("resume", None)
    if isinstance(identity.get("cli_args"), dict):
        identity["cli_args"].pop("resume", None)
        identity["cli_args"].pop("run_path_file", None)
    return identity


def _sandbox_request_identity(args_ref: dict) -> dict:
    enabled = bool(args_ref.get("docker", False))
    if not enabled:
        return {"enabled": False, "image": None, "image_digest": None}
    image = args_ref.get("image") or sbx.DEFAULT_IMAGE
    return {
        "enabled": True,
        "image": image,
        "image_digest": sbx.image_digest(image),
    }


def _research_audit_metadata(config: dict, args_ref: dict) -> dict:
    redacted_config = _redact_config(config)
    config_identity = _config_identity_value(None, config_for_hash(config))
    config_sources = [str(pipeline.DEFAULTS_PATH)] + [
        str(p) for p in args_ref.get("config_paths", [])
    ]
    probe_started = time.perf_counter()
    # These are independent network probes. Run them concurrently so an
    # unavailable local server costs one timeout window rather than two.
    with ThreadPoolExecutor(max_workers=3) as executor:
        ollama_future = executor.submit(_probe_ollama_models, config)
        compatible_future = executor.submit(
            _probe_openai_compatible_models, config,
        )
        remote_future = executor.submit(_probe_remote_models, config)
        model_runtime_probes = (
            ollama_future.result()
            + compatible_future.result()
            + remote_future.result()
        )
    model_runtime_probe_duration_ms = int(round(
        (time.perf_counter() - probe_started) * 1000
    ))
    model_runtime_identity = _model_runtime_identity(model_runtime_probes)
    run_args = _run_args_manifest(args_ref)
    run_args_identity = _run_args_identity(run_args)
    sandbox_identity = _sandbox_request_identity(args_ref)
    resolved_limits = pipeline.resolved_limits(config)
    concurrency = {
        "problem_parallelism": args_ref.get("parallel"),
        "provider_concurrency": (
            resolved_limits.get("max_provider_concurrency")
        ),
    }
    return {
        "config": redacted_config,
        "config_sha256": _sha256_json(config_identity),
        "redacted_config_sha256": _sha256_json(redacted_config),
        "config_sources": config_sources,
        "config_source_fingerprints": _config_source_fingerprints(config_sources),
        "role_model_manifest": _role_model_manifest(config),
        "model_runtime_probes": model_runtime_probes,
        "model_runtime_probe_duration_ms": model_runtime_probe_duration_ms,
        "model_runtime_identity": model_runtime_identity,
        "model_runtime_identity_sha256": _sha256_json(model_runtime_identity),
        "model_runtime_identity_complete": all(
            identity.get("resolved")
            and (
                identity.get("backend") != "ollama"
                or bool(identity.get("digest"))
            )
            for identity in model_runtime_identity
        ),
        "prompt_fingerprints": _prompt_fingerprints(config),
        "limits": resolved_limits,
        "run_args": run_args,
        "run_args_identity": run_args_identity,
        "run_args_identity_sha256": _sha256_json(run_args_identity),
        "sandbox_request_identity": sandbox_identity,
        "sandbox_request_identity_sha256": _sha256_json(sandbox_identity),
        "concurrency": concurrency,
        "concurrency_sha256": _sha256_json(concurrency),
    }


def compute_config_hash(config: dict | None = None) -> str:
    """Reproducibility stamp: SHA-256 over every role's prompt template,
    the resolved (post-default) model configs, and the schema versions in
    play. Computed once at run start and stamped into `run.started`,
    every call's meta.json, and every training-export row, so trained-on
    data can always be traced back to the exact pipeline recipe.
    """
    resolved_config = CONFIG if config is None else config
    role_settings = {}
    for role, settings in _role_items(resolved_config):
        resolved = effective_settings(
            settings, web_search_config=resolved_config.get("_web_search"),
        )
        role_settings[role] = {
            key: value for key, value in sorted(resolved.items())
            if key != "prompt"
        }
    identity = {
        "prompts": _prompt_fingerprints(resolved_config),
        "role_settings": _redact_config_value(None, role_settings),
        "schema_versions": {
            "audit": AUDIT_SCHEMA_VERSION,
            "telemetry": TELEMETRY_SCHEMA_VERSION,
            # Mode-aware: identical to PROOF_SCHEMA for every non-ablation mode, so
            # historical stamps are unchanged, but distinct under
            # step_wise_no_completeness (which neutralizes the `uses` description).
            "proof_schema_sha256": _sha256_json(
                pipeline._proof_schema(resolved_config)
            ),
            "verdict_schema_sha256": _sha256_json(pipeline.VERDICT_SCHEMA),
            "pedantry_schema_sha256": _sha256_json(pipeline.PEDANTRY_SCHEMA),
            "convention_lift_schema_sha256": _sha256_json(
                pipeline.CONVENTION_LIFT_SCHEMA
            ),
            "formalizer_decision_schema_sha256": _sha256_json(
                pipeline._formalizer_decision_schema(resolved_config)
            ),
        },
    }
    # Behavior-changing control knobs live in top-level underscore keys that
    # `_role_items` drops, so they never reach `role_settings` above -- yet they
    # change which answers are CERTIFIED vs ABSTAINED (`_verifier_mode` picks the
    # verifier; `_score_threshold` sets the certified bit; `_limits` sets the repair
    # budget; `_sandbox` toggles network/tool isolation). Fold them in whenever ANY
    # is present, so e.g. the shipped configs/no_repair.yaml (`_limits {1,1,1}`, no
    # `_verifier_mode`) and a `_sandbox.allow_network` flip get DISTINCT recipe
    # stamps -- while a plain config with none of these keeps its historical hash
    # unchanged. (The run-level `config_sha256` already captures these; this closes
    # the per-call / training-row stamp the docstring calls the exact recipe.)
    # `_filters` turns the pedantry / convention-lift passes off. Both are
    # monotonically permissive, so flipping them changes which answers are
    # CERTIFIED -- it must therefore reach the recipe stamp, or a bare run and a
    # full step_wise run would be indistinguishable after the fact.
    #
    # Stamped RESOLVED, and omitted entirely when it equals the default. Two
    # reasons, both provenance:
    #   * canonicalization -- `{convention_lift: false}` and
    #     `{pedantry: true, convention_lift: false}` are the same recipe, and
    #     stamping the raw block gave them different hashes, splitting one cell
    #     into two in any aggregation keyed on config_hash;
    #   * continuity -- stamping the raw block put `"filters": None` into the
    #     block for EVERY config that sets any of the other four keys, silently
    #     changing the config_hash of recipes already run (all existing
    #     minus_completeness runs, no_repair, every sandbox-isolated arm). With
    #     the key omitted at the default, those hashes are byte-identical to
    #     their historical values and only genuinely-filtered runs get a new one.
    # Resolving here also VALIDATES: a malformed `_filters` fails at stamp time
    # rather than mid-run, on every mode rather than only the step-wise paths.
    filters = pipeline.resolved_filters(resolved_config)
    filters_changed = filters != pipeline._DEFAULT_FILTERS

    if filters_changed or any(
        resolved_config.get(k) is not None
        for k in ("_verifier_mode", "_score_threshold", "_limits", "_sandbox")
    ):
        identity["experiment_verifier"] = {
            "mode": resolved_config.get("_verifier_mode"),
            "score_threshold": resolved_config.get("_score_threshold"),
            "limits": resolved_config.get("_limits"),
            "sandbox": resolved_config.get("_sandbox"),
            "answer_score_schema_sha256": _sha256_json(pipeline.ANSWER_SCORE_SCHEMA),
            "answer_judge_schema_sha256": _sha256_json(pipeline.ANSWER_JUDGE_SCHEMA),
            "holistic_proof_schema_sha256": _sha256_json(
                pipeline.HOLISTIC_PROOF_SCHEMA
            ),
        }
        if filters_changed:
            identity["experiment_verifier"]["filters"] = filters
    return _sha256_json(identity)


def expand_problem_samples(problems: list[dict], samples: int) -> list[dict]:
    """Duplicate each problem into `samples` independent rollouts.

    Each copy gets a unique id (`<pid>__s<k>`), and every copy carries
    `rollout_group` (the original id) and `sample_index` (1-based) so
    results and exports can group the K samples. With samples <= 1 the
    input is returned unchanged — ids, fields, and behavior identical.
    """
    if samples <= 1:
        return problems
    expanded = []
    for problem in problems:
        original_id = str(problem.get("id", "?"))
        for k in range(1, samples + 1):
            copy = dict(problem)
            copy["id"] = f"{original_id}__s{k:02d}"
            copy["rollout_group"] = original_id
            copy["sample_index"] = k
            expanded.append(copy)
    return expanded


def _calls_from_artifact_tree(
    artifact_root: str, *, skip_problem_ids: set[str] = frozenset(),
) -> list[dict]:
    """Recover per-call metadata from disk for problems whose in-memory
    results are unavailable (crash, interrupt, resumed process).

    Run-level telemetry used to aggregate only the calls carried in the
    in-memory results list, so an interrupted or partially-recovered run
    wrote a telemetry.json full of zeros even though every call's
    meta.json sat on disk. Reading the artifact tree closes that gap.
    """
    calls: list[dict] = []
    try:
        problem_dirs = sorted(os.listdir(artifact_root))
    except OSError:
        return calls
    for name in problem_dirs:
        if name in skip_problem_ids:
            continue
        problem_dir = os.path.join(artifact_root, name)
        if not os.path.isdir(problem_dir):
            continue
        try:
            call_dirs = sorted(os.listdir(problem_dir))
        except OSError:
            continue
        for call_name in call_dirs:
            if not re.match(r"call_\d+_", call_name):
                continue
            meta_path = os.path.join(problem_dir, call_name, "meta.json")
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(meta, dict):
                calls.append(meta)
    return calls


def _problem_set_manifest(problems: list[dict]) -> dict:
    ids = [str(p.get("id", "")) for p in problems]
    metadata_fields = sorted({
        key
        for problem in problems
        for key in problem
        if key not in {"question", "answer"}
    })
    return {
        "count": len(problems),
        "ids": ids,
        "ids_sha256": _sha256_json(ids),
        "records_sha256": _sha256_json(problems),
        "duplicate_ids": sorted(
            problem_id for problem_id, count in Counter(ids).items() if count > 1
        ),
        "metadata_fields": metadata_fields,
        "datasets": sorted({
            str(p.get("dataset"))
            for p in problems
            if p.get("dataset")
        }),
        "dataset_sources": sorted({
            str(p.get("dataset_source"))
            for p in problems
            if p.get("dataset_source")
        }),
        "dataset_splits": sorted({
            str(p.get("dataset_split"))
            for p in problems
            if p.get("dataset_split")
        }),
        "dataset_revisions": sorted({
            str(p.get("dataset_revision"))
            for p in problems
            if p.get("dataset_revision")
        }),
        "dataset_fingerprints": sorted({
            str(p.get("dataset_fingerprint"))
            for p in problems
            if p.get("dataset_fingerprint")
        }),
        "dataset_configs": sorted({
            str(p.get("dataset_config"))
            for p in problems
            if p.get("dataset_config")
        }),
        "dataset_subsets": sorted({
            str(p.get("dataset_subset"))
            for p in problems
            if p.get("dataset_subset")
        }),
        "categories": dict(Counter(
            str(p.get("category", "?")) for p in problems if p.get("category")
        )),
        "verified_classes": dict(Counter(
            str(p.get("verified_class")) for p in problems
            if p.get("verified_class")
        )),
    }


def _safe_version(binary: str) -> str | None:
    """Run `<binary> --version` and return the stripped output.
    Returns None if the binary is missing or errors out."""
    try:
        r = subprocess.run(
            [binary, "--version"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return (r.stdout or r.stderr or "").strip()


def _safe_git_state(artifact_root: str | None = None, label: str = "run") -> dict:
    """Return current git HEAD sha, branch, and dirty flag. Tolerates
    missing git or non-repo directories. When dirty, retain the tracked diff
    and hashes for untracked files so an experimental run is reconstructable.
    """
    out: dict = {
        "sha": None,
        "branch": None,
        "dirty": None,
        "diff_sha256": None,
        "diff_bytes": 0,
        "diff_path": None,
        "untracked_files": [],
        "untracked_sha256": None,
    }
    for key, argv in [
        ("sha", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    ]:
        try:
            r = subprocess.run(
                argv, capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                out[key] = r.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            out["dirty"] = bool(r.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    if not out["dirty"]:
        return out

    try:
        r = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            capture_output=True, timeout=10,
        )
        if r.returncode == 0:
            patch = r.stdout
            out["diff_sha256"] = hashlib.sha256(patch).hexdigest()
            out["diff_bytes"] = len(patch)
            if artifact_root and patch:
                path = os.path.join(artifact_root, f"git_{label}.patch")
                with open(path, "wb") as f:
                    f.write(patch)
                out["diff_path"] = path
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    try:
        r = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True, timeout=10,
        )
        if r.returncode == 0:
            for raw_path in r.stdout.split(b"\0"):
                if not raw_path:
                    continue
                path = raw_path.decode(errors="replace")
                out["untracked_files"].append({
                    "path": path,
                    "sha256": _file_sha256(path),
                })
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    out["untracked_sha256"] = _sha256_json(out["untracked_files"])
    return out


def _host_runtime_metadata() -> dict:
    memory_bytes = None
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    try:
        installed_packages = sorted({
            f"{distribution.metadata.get('Name') or 'unknown'}=="
            f"{distribution.version}"
            for distribution in importlib.metadata.distributions()
        }, key=str.lower)
        installed_packages_sha256 = _sha256_text("\n".join(installed_packages))
    except Exception:
        installed_packages_sha256 = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": os.path.realpath(sys.executable),
        "installed_packages_sha256": installed_packages_sha256,
    }


def _capture_host_environment(
    root: str,
    label: str,
    *,
    prefer_pip_freeze: bool = True,
) -> dict:
    path = os.path.join(root, f"host_packages_{label}.txt")

    def persist(payload: bytes, source: str) -> dict:
        with open(path, "wb") as f:
            f.write(payload)
        return {
            "available": True,
            "source": source,
            "path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    if prefer_pip_freeze:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                return persist(result.stdout, "pip-freeze")
        except (OSError, subprocess.TimeoutExpired) as exc:
            pip_error = f"{type(exc).__name__}: {exc}"
        else:
            pip_error = result.stderr.decode(errors="replace")[:500]
    else:
        pip_error = "pip freeze skipped outside evaluation/ablation runs"

    try:
        packages = sorted({
            f"{distribution.metadata.get('Name') or 'unknown'}=="
            f"{distribution.version}"
            for distribution in importlib.metadata.distributions()
        }, key=str.lower)
        payload = ("\n".join(packages) + "\n").encode()
        record = persist(payload, "importlib.metadata")
        record["pip_error"] = pip_error
        return record
    except Exception as exc:
        return {
            "available": False,
            "pip_error": pip_error,
            "fallback_error_type": type(exc).__name__,
            "fallback_error": str(exc),
        }


def _read_meta(root: str) -> dict:
    path = os.path.join(root, "meta.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_meta(root: str, meta: dict) -> None:
    path = os.path.join(root, "meta.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    os.replace(tmp, path)


def resolve_artifact_manifest_path(root: str, entry: dict) -> str:
    """Resolve a v1 absolute or v2 artifact-root-relative manifest entry."""
    root = os.path.abspath(root)
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise RuntimeError(f"invalid artifact manifest entry: {entry!r}")
    recorded_path = entry["path"]
    if entry.get("external"):
        if os.path.isabs(recorded_path):
            return os.path.abspath(recorded_path)
        return os.path.abspath(os.path.join(root, os.path.normpath(recorded_path)))
    if os.path.isabs(recorded_path):
        raise RuntimeError(
            f"internal manifest path is absolute: {recorded_path!r}"
        )
    normalized = os.path.normpath(recorded_path)
    path = os.path.abspath(os.path.join(root, normalized))
    try:
        inside_root = os.path.commonpath([root, path]) == root
    except ValueError:
        inside_root = False
    if not inside_root or normalized in {".", ".."}:
        raise RuntimeError(
            f"internal manifest path escapes artifact root: {recorded_path!r}"
        )
    return path


def verify_artifact_manifest(root: str) -> dict:
    """Verify a sealed artifact tree before any resume-time mutation.

    Both listed hashes and the exact internal file set are checked. A resume
    must fail closed if files were changed, removed, or added after sealing.
    External entries (normally the run JSON) are checked as well. Schema 1.0
    absolute external paths and portable schema 2.0 relative paths are accepted.
    """
    root = os.path.abspath(root)
    manifest_path = os.path.join(root, "artifact_manifest.json")
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "cannot resume because the artifact manifest is missing or "
            f"unreadable: {manifest_path}"
        ) from exc
    schema_version = manifest.get("schema_version")
    if schema_version not in {"1.0", "2.0"} or not isinstance(
        manifest.get("entries"), list
    ):
        raise RuntimeError(f"invalid artifact manifest structure: {manifest_path}")

    expected_internal: set[str] = set()
    seen: set[tuple[bool, str]] = set()
    for entry in manifest["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError(f"invalid artifact manifest entry: {entry!r}")
        external = bool(entry.get("external"))
        recorded_path = entry["path"]
        if external:
            if schema_version == "1.0" and not os.path.isabs(recorded_path):
                raise RuntimeError(
                    f"external manifest path is not absolute: {recorded_path!r}"
                )
            if schema_version == "2.0" and os.path.isabs(recorded_path):
                raise RuntimeError(
                    "external schema 2.0 manifest path is not artifact-root-"
                    f"relative: {recorded_path!r}"
                )
        else:
            if os.path.isabs(recorded_path):
                raise RuntimeError(
                    f"internal manifest path is absolute: {recorded_path!r}"
                )
        path = resolve_artifact_manifest_path(root, entry)
        identity = (external, path)
        if not external:
            expected_internal.add(os.path.relpath(path, root))
        if identity in seen:
            raise RuntimeError(f"duplicate artifact manifest path: {recorded_path!r}")
        seen.add(identity)
        if os.path.islink(path):
            raise RuntimeError(f"sealed artifact path is a symlink: {path}")
        if not os.path.isfile(path):
            raise RuntimeError(f"sealed artifact is missing: {path}")
        if os.path.getsize(path) != entry.get("size_bytes"):
            raise RuntimeError(f"sealed artifact size mismatch: {path}")
        digest = _file_sha256(path)
        if digest != entry.get("sha256"):
            raise RuntimeError(f"sealed artifact hash mismatch: {path}")

    actual_internal: set[str] = set()
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.abspath(os.path.join(directory, filename))
            if path == manifest_path:
                continue
            actual_internal.add(os.path.relpath(path, root))
    if actual_internal != expected_internal:
        missing = sorted(expected_internal - actual_internal)
        unexpected = sorted(actual_internal - expected_internal)
        raise RuntimeError(
            "sealed artifact file set mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    return manifest


def _problem_artifact_directory(root: str, problem_id: object) -> str:
    root = os.path.abspath(root)
    path = os.path.abspath(os.path.join(root, str(problem_id)))
    try:
        inside_root = os.path.commonpath([root, path]) == root
    except ValueError:
        inside_root = False
    if not inside_root or path == root:
        raise ValueError(
            f"problem ID escapes the artifact root: {problem_id!r}"
        )
    return path


def _safe_hashed_name(value: object, fallback: str) -> str:
    text = str(value)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or fallback
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:48]}_{digest}"


def _partial_state_path(problem_id: object, run_id: object | None = None) -> str:
    run_name = _safe_hashed_name(run_id or "adhoc", "run")
    problem_name = _safe_hashed_name(problem_id, "problem")
    return os.path.join("runs", "partial", run_name, f"{problem_name}.json")


def _write_problem_checkpoint(
    root: str,
    problem: dict,
    result: dict,
) -> str:
    """Atomically seal one completed problem for crash-safe resume.

    A global run manifest cannot stay current while other problems are still
    writing concurrently. Per-problem checkpoints are immutable once written,
    so a later process can reuse only complete, hash-verified results and
    quarantine every partial problem directory left by an unclean shutdown.
    """
    problem_id = str(problem.get("id", "?"))
    problem_dir = _problem_artifact_directory(root, problem_id)
    os.makedirs(problem_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        problem_dir, PROBLEM_CHECKPOINT_FILENAME,
    )
    checkpoint_tmp = checkpoint_path + ".tmp"
    result_path = os.path.join(problem_dir, "result.json")
    result_tmp = result_path + ".tmp"

    with open(result_tmp, "w") as f:
        json.dump(result, f, indent=2, default=str)
    os.replace(result_tmp, result_path)
    try:
        os.unlink(checkpoint_tmp)
    except FileNotFoundError:
        pass

    entries = []
    for directory, _, filenames in os.walk(problem_dir):
        for filename in sorted(filenames):
            path = os.path.abspath(os.path.join(directory, filename))
            if path in {checkpoint_path, checkpoint_tmp}:
                continue
            if os.path.islink(path):
                raise OSError(
                    f"refusing to checkpoint symlinked artifact: {path}"
                )
            digest = _file_sha256(path)
            if digest is None:
                raise OSError(f"could not hash problem artifact: {path}")
            entries.append({
                "path": os.path.relpath(path, problem_dir),
                "size_bytes": os.path.getsize(path),
                "sha256": digest,
            })
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "problem_id": problem_id,
        "problem_record_sha256": _sha256_json(problem),
        "result_path": "result.json",
        "entries": sorted(entries, key=lambda entry: entry["path"]),
    }
    with open(checkpoint_tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(checkpoint_tmp, checkpoint_path)
    return checkpoint_path


def _verify_problem_checkpoint(
    root: str,
    problem: dict,
) -> tuple[dict, dict]:
    """Verify one immutable problem checkpoint and return its result."""
    problem_id = str(problem.get("id", "?"))
    problem_dir = _problem_artifact_directory(root, problem_id)
    checkpoint_path = os.path.join(
        problem_dir, PROBLEM_CHECKPOINT_FILENAME,
    )
    try:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"problem checkpoint is missing or unreadable: {checkpoint_path}"
        ) from exc
    if (
        checkpoint.get("schema_version") != "1.0"
        or checkpoint.get("problem_id") != problem_id
        or checkpoint.get("problem_record_sha256") != _sha256_json(problem)
        or not isinstance(checkpoint.get("entries"), list)
    ):
        raise RuntimeError(
            f"problem checkpoint identity mismatch: {checkpoint_path}"
        )

    expected_files: set[str] = set()
    for entry in checkpoint["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError(
                f"invalid problem checkpoint entry: {entry!r}"
            )
        recorded_path = entry["path"]
        if os.path.isabs(recorded_path):
            raise RuntimeError(
                f"problem checkpoint path is absolute: {recorded_path!r}"
            )
        normalized = os.path.normpath(recorded_path)
        path = os.path.abspath(os.path.join(problem_dir, normalized))
        try:
            inside_problem = os.path.commonpath([problem_dir, path]) == problem_dir
        except ValueError:
            inside_problem = False
        if not inside_problem or normalized in {".", ".."}:
            raise RuntimeError(
                f"problem checkpoint path escapes its directory: {recorded_path!r}"
            )
        if normalized in expected_files:
            raise RuntimeError(
                f"duplicate problem checkpoint path: {recorded_path!r}"
            )
        expected_files.add(normalized)
        if os.path.islink(path):
            raise RuntimeError(f"checkpointed artifact is a symlink: {path}")
        if not os.path.isfile(path):
            raise RuntimeError(f"checkpointed artifact is missing: {path}")
        if os.path.getsize(path) != entry.get("size_bytes"):
            raise RuntimeError(
                f"checkpointed artifact size mismatch: {path}"
            )
        if _file_sha256(path) != entry.get("sha256"):
            raise RuntimeError(
                f"checkpointed artifact hash mismatch: {path}"
            )

    actual_files: set[str] = set()
    for directory, _, filenames in os.walk(problem_dir):
        for filename in filenames:
            path = os.path.abspath(os.path.join(directory, filename))
            if path == checkpoint_path:
                continue
            actual_files.add(os.path.relpath(path, problem_dir))
    if actual_files != expected_files:
        raise RuntimeError(
            "checkpointed problem file set mismatch: "
            f"missing={sorted(expected_files - actual_files)} "
            f"unexpected={sorted(actual_files - expected_files)}"
        )

    result_relpath = checkpoint.get("result_path")
    if result_relpath not in expected_files:
        raise RuntimeError(
            f"problem checkpoint has no sealed result: {checkpoint_path}"
        )
    result_path = os.path.join(problem_dir, result_relpath)
    try:
        with open(result_path) as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"checkpointed result is unreadable: {result_path}"
        ) from exc
    if not isinstance(result, dict) or str(result.get("id")) != problem_id:
        raise RuntimeError(
            f"checkpointed result identity mismatch: {result_path}"
        )
    return result, checkpoint


def _recover_problem_checkpoints(
    root: str,
    problems: list[dict],
    *,
    integrity_mode: str | None,
) -> list[dict]:
    """Load sealed results and isolate unsealed crash residue."""
    recovered = []
    quarantined = []
    retryable = []
    recovery_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for index, problem in enumerate(problems):
        problem_id = str(problem.get("id", "?"))
        problem_dir = _problem_artifact_directory(root, problem_id)
        checkpoint_path = os.path.join(
            problem_dir, PROBLEM_CHECKPOINT_FILENAME,
        )
        if os.path.isfile(checkpoint_path):
            result, _ = _verify_problem_checkpoint(root, problem)
            if result.get("error"):
                archive_root = os.path.join(
                    root, RETRY_CHECKPOINTS_DIR, recovery_label,
                )
                os.makedirs(archive_root, exist_ok=True)
                archive_name = (
                    f"problem_{index:05d}_{_sha256_text(problem_id)[:12]}.json"
                )
                archive_path = os.path.join(archive_root, archive_name)
                os.replace(checkpoint_path, archive_path)
                retryable.append({
                    "problem_id": problem_id,
                    "prior_checkpoint": os.path.relpath(archive_path, root),
                    "reason": "prior completed result contained an error",
                })
                continue
            recovered.append(result)
            continue
        if not os.path.isdir(problem_dir) or integrity_mode != "problem_checkpoints":
            continue

        archive_root = os.path.join(
            root, INTERRUPTED_ATTEMPTS_DIR, recovery_label,
        )
        os.makedirs(archive_root, exist_ok=True)
        archive_name = (
            f"problem_{index:05d}_{_sha256_text(problem_id)[:12]}"
        )
        archive_path = os.path.join(archive_root, archive_name)
        os.replace(problem_dir, archive_path)
        quarantined.append({
            "problem_id": problem_id,
            "path": os.path.relpath(archive_path, root),
            "reason": "no valid completed-problem checkpoint",
        })

    meta = _read_meta(root)
    resumes = meta.get("resumes") or []
    if resumes:
        resumes[-1]["verified_problem_checkpoint_ids"] = [
            str(result.get("id")) for result in recovered
        ]
        resumes[-1]["quarantined_incomplete_artifacts"] = quarantined
        resumes[-1]["retryable_error_checkpoints"] = retryable
        _write_meta(root, meta)
    return recovered


def _resume_identity(meta: dict) -> dict:
    audit = meta.get("research_audit") or {}
    git = meta.get("git") or {}
    return {
        "audit_schema_version": meta.get("audit_schema_version"),
        "config_sha256": meta.get("config_sha256") or audit.get("config_sha256"),
        "pricing_sha256": (meta.get("pricing") or {}).get("sha256"),
        "run_args_identity": audit.get("run_args_identity"),
        "sandbox_request_identity": audit.get("sandbox_request_identity"),
        "concurrency": audit.get("concurrency"),
        "model_runtime_identity": audit.get("model_runtime_identity"),
        "git_sha": git.get("sha"),
        "git_diff_sha256": git.get("diff_sha256"),
        "git_untracked_sha256": git.get("untracked_sha256"),
        "python_version": meta.get("python_version"),
        "python_executable": meta.get("python_executable"),
        "host_runtime": meta.get("host_runtime"),
        "cli_versions": meta.get("cli_versions"),
    }


def make_artifact_root(
    save_path: str,
    *,
    problem_manifest: dict | None = None,
) -> str:
    """Create runs/artifacts/<run_id>/ and write meta.json.

    `run_id` is the save_path basename without extension. The meta file
    captures everything needed to reproduce the run: argv, cwd, merged
    config, CLI versions, git state, python version, hostname,
    started_at. Artifacts for individual calls land under
    <run_id>/<problem_id>/call_NNN_<role>/.

    Returns the absolute artifact root.
    """
    run_id = os.path.basename(save_path)
    if run_id.endswith(".json"):
        run_id = run_id[:-5]
    root = os.path.abspath(os.path.join("runs", "artifacts", run_id))
    os.makedirs(root, exist_ok=True)

    meta_path = os.path.join(root, "meta.json")
    is_resume = os.path.exists(meta_path)
    existing: dict | None = None
    verified_manifest: dict | None = None
    prior_manifest_sha256: str | None = None
    manifest_error: str | None = None
    integrity_mode: str | None = None
    if is_resume:
        try:
            with open(meta_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                "cannot resume because the existing run metadata is "
                f"unreadable: {meta_path}. Start a new run instead."
            ) from exc
        if not isinstance(existing, dict):
            raise RuntimeError(
                "cannot resume because the existing run metadata is not an "
                f"object: {meta_path}. Start a new run instead."
            )
        try:
            verified_manifest = verify_artifact_manifest(root)
        except RuntimeError as exc:
            if existing.get("status") != "running":
                raise
            # A hard-killed run has readable identity metadata but may have no
            # final run manifest, or a stale one from an earlier resume. Only
            # independently sealed completed-problem checkpoints can be
            # reused; all other problem artifacts are quarantined below.
            manifest_error = str(exc)
            integrity_mode = "problem_checkpoints"
        else:
            prior_manifest_sha256 = _file_sha256(
                os.path.join(root, "artifact_manifest.json")
            )
            integrity_mode = "sealed_manifest"
    invocation_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    research_audit = _research_audit_metadata(CONFIG, _args_ref)
    new_meta = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "hostname": __import__("socket").gethostname(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "host_runtime": _host_runtime_metadata(),
        "host_environment": (
            None if is_resume
            else _capture_host_environment(
                root,
                invocation_label,
                prefer_pip_freeze=(
                    _args_ref.get("experiment_phase")
                    in {"evaluation", "ablation"}
                ),
            )
        ),
        "save_path": save_path,
        "config": _redact_config(CONFIG),
        "config_sha256": research_audit["config_sha256"],
        "pricing": _pricing_provenance(),
        "cli_versions": {
            "claude": _safe_version("claude"),
            "codex": _safe_version("codex"),
            "docker": _safe_version("docker"),
            "ollama": _safe_version("ollama"),
            "nvidia_smi": _safe_version("nvidia-smi"),
        },
        "git": _safe_git_state(
            None if is_resume else root,
            invocation_label,
        ),
        "research_audit": research_audit,
    }
    if problem_manifest is not None:
        new_meta["problem_set"] = problem_manifest
    new_meta["resume_identity"] = _resume_identity(new_meta)
    # If we're resuming an existing run, preserve the original metadata
    # and append resume info as a list of resume events. This keeps a
    # full audit trail of every (re-)invocation against this run dir.
    if existing is not None:
        try:
            existing_audit = existing.get("research_audit") or {}
            if existing_audit.get("model_runtime_identity_complete") is False:
                raise RuntimeError(
                    "resume model provenance is incomplete in the original "
                    "run. Start a new run after resolving every local model."
                )
            if research_audit.get("model_runtime_identity_complete") is False:
                raise RuntimeError(
                    "resume model provenance is incomplete for the current "
                    "invocation. Resolve every local model before resuming."
                )
            prior_identity = _resume_identity(existing)
            new_identity = _resume_identity(new_meta)
            if prior_identity != new_identity:
                raise RuntimeError(
                    "resume provenance mismatch: the effective config or code "
                    f"changed since this run began; original={prior_identity} "
                    f"current={new_identity}. Start a new run instead."
                )
            existing_problem = existing.get("problem_set")
            if problem_manifest and not existing_problem:
                raise RuntimeError(
                    "resume problem-set provenance is missing from the "
                    "original run. Start a new run instead."
                )
            if existing_problem and problem_manifest:
                existing_problem_identity = {
                    "ids_sha256": existing_problem.get("ids_sha256"),
                    "records_sha256": existing_problem.get("records_sha256"),
                }
                problem_identity = {
                    "ids_sha256": problem_manifest.get("ids_sha256"),
                    "records_sha256": problem_manifest.get("records_sha256"),
                }
                if existing_problem_identity != problem_identity:
                    raise RuntimeError(
                        "resume problem-set mismatch: the selected cohort "
                        f"changed; original={existing_problem_identity} "
                        f"current={problem_identity}. Start a new run instead."
                    )
            resumes = existing.get("resumes", [])
            archive_path = None
            if verified_manifest is not None:
                archive_dir = os.path.join(root, "resume_manifests")
                os.makedirs(archive_dir, exist_ok=True)
                archive_path = os.path.join(
                    archive_dir,
                    f"artifact_manifest_{invocation_label}.json",
                )
                with open(archive_path, "w") as f:
                    json.dump(verified_manifest, f, indent=2)
            original_environment = existing.get("host_environment") or {}
            resume_event = {
                "resumed_at": new_meta["started_at"],
                "argv": new_meta["argv"],
                "cli_versions": new_meta["cli_versions"],
                "git": new_meta["git"],
                "config_sha256": new_meta["config_sha256"],
                "research_audit": new_meta["research_audit"],
                "resume_identity": new_identity,
                "host_runtime": new_meta["host_runtime"],
                "host_environment": {
                    "captured": False,
                    "reason": "resume identity matched the original host runtime",
                    "original_sha256": original_environment.get("sha256"),
                    "original_source": original_environment.get("source"),
                },
                "integrity_mode": integrity_mode,
                "global_manifest_error": manifest_error,
                "verified_prior_manifest_sha256": prior_manifest_sha256,
                "prior_manifest_archive": (
                    os.path.relpath(archive_path, root)
                    if archive_path else None
                ),
                "problem_set_identity": (
                    {
                        "ids_sha256": problem_manifest.get("ids_sha256"),
                        "records_sha256": problem_manifest.get("records_sha256"),
                    }
                    if problem_manifest else None
                ),
            }
            resumes.append(resume_event)
            existing["resumes"] = resumes
            existing["status"] = "running"
            _write_meta(root, existing)
            return root
        except RuntimeError:
            raise
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                "cannot resume because the existing run metadata is "
                f"unreadable: {meta_path}. Start a new run instead."
            ) from exc
    _write_meta(root, new_meta)
    return root


def finalize_artifact_root(
    root: str,
    *,
    status: str = "completed",
    requested: int | None = None,
    completed: int | None = None,
    completed_ids: list[str] | None = None,
    error: BaseException | None = None,
) -> None:
    """Finalize run status and counts in the audit record."""
    updates = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    if requested is not None:
        updates["requested_problems"] = requested
    if completed is not None:
        updates["completed_problems"] = completed
    if completed_ids is not None:
        updates["completed_problem_ids"] = completed_ids
    if error is not None:
        updates["run_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    update_artifact_root_meta(root, updates, strict=True)


def update_artifact_root_meta(
    root: str,
    updates: dict,
    *,
    strict: bool = False,
) -> bool:
    """Merge `updates` into the run's meta.json. Silently tolerates
    missing or corrupt files — the meta file is a convenience, not a
    crash-critical artifact."""
    try:
        meta = _read_meta(root)
    except (OSError, json.JSONDecodeError):
        meta = {}
    meta.update(updates)
    try:
        _write_meta(root, meta)
        return True
    except OSError:
        if strict:
            raise
        return False


def record_problem_set_manifest(root: str, manifest: dict) -> None:
    """Persist an immutable cohort manifest and reject resume drift."""
    meta = _read_meta(root)
    existing = meta.get("problem_set")
    identity = {
        "ids_sha256": manifest.get("ids_sha256"),
        "records_sha256": manifest.get("records_sha256"),
    }
    if existing:
        existing_identity = {
            "ids_sha256": existing.get("ids_sha256"),
            "records_sha256": existing.get("records_sha256"),
        }
        if existing_identity != identity:
            raise RuntimeError(
                "resume problem-set mismatch: the selected cohort changed; "
                f"original={existing_identity} current={identity}. Start a "
                "new run instead."
            )
    else:
        meta["problem_set"] = manifest
    resumes = meta.get("resumes") or []
    if resumes:
        resumes[-1]["problem_set_identity"] = identity
        meta["last_resumed_at"] = resumes[-1]["resumed_at"]
    meta["status"] = "running"
    _write_meta(root, meta)


def write_artifact_manifest(
    root: str,
    *,
    external_paths: list[str] | None = None,
) -> str:
    """Write a portable schema 2.0 seal for every retained run artifact."""
    root = os.path.abspath(root)
    manifest_path = os.path.join(root, "artifact_manifest.json")
    manifest_tmp = manifest_path + ".tmp"
    try:
        os.unlink(manifest_tmp)
    except FileNotFoundError:
        pass
    entries = []
    for directory, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            path = os.path.join(directory, filename)
            if (
                os.path.abspath(path) in {manifest_path, manifest_tmp}
                or not os.path.isfile(path)
            ):
                continue
            if os.path.islink(path):
                raise OSError(f"refusing to seal symlinked artifact: {path}")
            digest = _file_sha256(path)
            if digest is None:
                raise OSError(f"could not hash retained artifact: {path}")
            entries.append({
                "path": os.path.relpath(path, root),
                "size_bytes": os.path.getsize(path),
                "sha256": digest,
            })
    for path in external_paths or []:
        if os.path.isfile(path):
            if os.path.islink(path):
                raise OSError(f"refusing to seal symlinked external artifact: {path}")
            digest = _file_sha256(path)
            if digest is None:
                raise OSError(f"could not hash external artifact: {path}")
            absolute_path = os.path.abspath(path)
            entries.append({
                "path": os.path.relpath(absolute_path, root),
                "external": True,
                "size_bytes": os.path.getsize(absolute_path),
                "sha256": digest,
            })
    payload = {
        "schema_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": sorted(entries, key=lambda entry: entry["path"]),
    }
    with open(manifest_tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(manifest_tmp, manifest_path)
    return manifest_path


# ── Argparse / config ────────────────────────────────────────────
#
# The CLI (cli.py) builds its own argument parser and constructs the
# namespace these functions read (config, backend, codex_model, watch,
# docker, image, parallel, tag, resume). apply_args + make_save_path +
# run_problems are the integration surface.

def apply_args(args) -> None:
    """Load config and apply the backend + --codex-model overrides from the
    args namespace.

    The audited Codex preset keeps the formalizer on Claude. Non-Codex
    backends are experimental and should cover all roles so experiments use a
    single agentic loop unless a config explicitly says otherwise.

    --codex-model is applied after the backend, so it overrides the model
    for any role currently on codex (regardless of how it got there).
    """
    loaded = load_config(args.config)
    # Validate the behaviour-changing knobs NOW, at load, rather than at first
    # use. `resolved_filters` was previously first called mid-run -- after the
    # solver, formalizer and every per-step judge had already spent budget --
    # and only on the step-wise paths, so a malformed `_filters` under
    # holistic_proof or answer_score was never validated at all. A typo in a
    # stacked overlay should cost a second, not a run. This mirrors how the
    # completeness markup hard-errors at render time.
    #
    # BEFORE the global CONFIG is touched: a config that fails validation must
    # leave the process unconfigured rather than half-configured.
    pipeline.resolved_filters(loaded)
    CONFIG.update(loaded)
    # Role entries are dicts; top-level keys prefixed with "_" hold
    # shared values (e.g. `_preamble`) and must not be treated as roles.
    if args.backend:
        pin_formalizer_to_claude = args.backend == "codex"
        for role, settings in CONFIG.items():
            if role.startswith("_") or not isinstance(settings, dict):
                continue
            if role == "formalizer" and pin_formalizer_to_claude:
                continue
            settings["backend"] = args.backend
    if args.codex_model:
        for role, settings in CONFIG.items():
            if role.startswith("_") or not isinstance(settings, dict):
                continue
            if settings.get("backend") == "codex":
                settings["model"] = args.codex_model
    pricing_path = (
        getattr(args, "pricing", None) or os.getenv("OWRE_PRICING_FILE")
    )
    if pricing_path:
        load_pricing(pricing_path)
        CONFIG.setdefault("_telemetry", {})["pricing_file"] = pricing_path
    pipeline.WATCH = args.watch
    # Pulled in run_problems; expose here so runners don't need to thread it.
    _args_ref["docker"] = bool(getattr(args, "docker", False))
    _args_ref["image"] = getattr(args, "image", None) or sbx.DEFAULT_IMAGE
    _args_ref["config_paths"] = list(getattr(args, "config", None) or [])
    _args_ref["backend"] = getattr(args, "backend", None)
    _args_ref["codex_model"] = getattr(args, "codex_model", None)
    _args_ref["watch"] = bool(getattr(args, "watch", False))
    _args_ref["parallel"] = getattr(args, "parallel", None)
    _args_ref["resume"] = getattr(args, "resume", None)
    _args_ref["tag"] = getattr(args, "tag", None)
    _args_ref["experiment_id"] = getattr(args, "experiment_id", None)
    _args_ref["experiment_phase"] = getattr(args, "experiment_phase", None)
    _args_ref["trial"] = getattr(args, "trial", None)
    _args_ref["samples"] = getattr(args, "samples", None)
    _args_ref["pricing"] = pricing_path
    _args_ref["cli_args"] = {
        key: value
        for key, value in vars(args).items()
        if not callable(value)
    }


def config_uses_backend(backend: str) -> bool:
    """Return whether any configured role resolves to the given backend."""
    for role, settings in CONFIG.items():
        if role.startswith("_") or not isinstance(settings, dict):
            continue
        if settings.get("backend", "claude") == backend:
            return True
    return False


def sandbox_allow_network() -> bool:
    """Run-level `_sandbox.allow_network` (default True). When False, the
    per-problem container runs with --network=none, closing the shell-egress
    channel that otherwise bypasses the web_search blocklist. Only valid when the
    model runs host-side (reasoning_agent) -- _prepare_sandbox_run guards it."""
    return bool((CONFIG.get("_sandbox") or {}).get("allow_network", True))


def _prepare_sandbox_run(
    artifact_root: str,
    *,
    use_docker: bool,
    sandbox_image: str,
) -> tuple[str | None, str | None, str | None]:
    if not use_docker:
        if not sandbox_allow_network():
            # Network isolation is enforced by `docker run --network=none`, which
            # only exists in docker mode. Without --docker there is no container
            # to isolate, so silently returning here would ship an unsound
            # "contamination-isolated" label. Fail fast instead of ignoring it.
            raise RuntimeError(
                "_sandbox.allow_network: false requires Docker mode (--docker): "
                "network isolation is enforced by the per-problem container's "
                "--network=none, which does not exist without a container. Re-run "
                "with --docker, or drop _sandbox.allow_network for a host run."
            )
        update_artifact_root_meta(
            artifact_root, {"sandbox": {"enabled": False}}, strict=True,
        )
        return None, None, None

    if not sandbox_allow_network():
        # --network=none closes the per-problem container's shell-egress channel.
        # This is SOUND ONLY when the model runs host-side (reasoning_agent) and no
        # role opens a host-side tool channel. Enforce both as an ALLOWLIST
        # (fail-safe defaults: reject anything not known-safe) rather than
        # blocklisting the currently-known in-container backends -- a future
        # backend must opt in explicitly, and the invariant then holds by
        # construction rather than by an incidental call-order coincidence.
        for role, settings in _role_items(CONFIG):
            resolved = _effective_role_settings(CONFIG, settings)
            backend = resolved.get("backend", "claude")
            if backend != "reasoning_agent":
                raise RuntimeError(
                    "_sandbox.allow_network: false requires every role use the "
                    "reasoning_agent backend (its model runs host-side, so only the "
                    f"containerized shell needs egress). Role {role!r} uses "
                    f"backend {backend!r}, whose CLI runs inside the sandbox and "
                    "needs network egress (its web search is also server-side / "
                    "unfilterable). Use configs/tool_off.yaml for a closed-book "
                    "run on that backend instead."
                )
            if resolved.get("allow_host_tools"):
                raise RuntimeError(
                    "_sandbox.allow_network: false is incompatible with "
                    f"allow_host_tools: true (role {role!r}). Host tools run a "
                    "shell on the HOST with the host's unrestricted network, which "
                    "would sidestep the container's --network=none. Set "
                    "allow_host_tools: false for every role under network "
                    "isolation."
                )

    claude_creds_path: str | None = None
    claude_config_path: str | None = None
    try:
        needs_claude = config_uses_backend("claude")
        if needs_claude:
            claude_creds_path = sbx.refresh_claude_credentials()
            claude_config_path = sbx.prepare_claude_config()
        image_digest = sbx.image_digest(sandbox_image)
        requested_sandbox = (
            (_read_meta(artifact_root).get("research_audit") or {}).get(
                "sandbox_request_identity"
            )
            or {}
        )
        requested_digest = requested_sandbox.get("image_digest")
        if requested_digest and image_digest != requested_digest:
            raise RuntimeError(
                "sandbox image identity changed between run initialization "
                f"and container setup: expected={requested_digest} "
                f"current={image_digest}. Start a new run instead."
            )
        print(f"Docker mode: image={sandbox_image} "
              f"image_digest={image_digest} "
              f"creds={'present' if claude_creds_path else 'MISSING'} "
              f"config_snapshot={'present' if claude_config_path else 'MISSING'}")
        if needs_claude and claude_creds_path is None:
            raise RuntimeError(
                "Docker mode requested but could not read Claude credentials "
                "from the macOS Keychain (not logged in, or not on macOS). "
                "The formalizer needs them. Run `claude` and sign in, then "
                "retry (`reasoning-eval doctor` checks this)."
            )
        if image_digest is None:
            raise RuntimeError(
                f"Docker mode requested but image {sandbox_image} is not "
                "available locally. Build it first with the appropriate "
                "Dockerfile (see sandbox/ or sandbox-sage/)."
            )
        if needs_claude and claude_config_path is None:
            raise RuntimeError(
                "Docker mode requested but could not snapshot ~/.claude.json "
                "(file missing or repeatedly corrupted mid-snapshot)."
            )
        container_versions = sbx.container_tool_versions(sandbox_image)
        print(f"  container tools: {container_versions}")
        update_artifact_root_meta(artifact_root, {
            "sandbox": {
                "enabled": True,
                "image": sandbox_image,
                "image_digest": image_digest,
                "container_tool_versions": container_versions,
                "claude_creds_source": (
                    "macOS Keychain to plaintext temp mount"
                    if claude_creds_path else "none"
                ),
            },
        }, strict=True)
        return claude_creds_path, claude_config_path, image_digest
    except BaseException:
        if claude_creds_path:
            sbx.cleanup_credentials(claude_creds_path)
        if claude_config_path:
            sbx.cleanup_claude_config(claude_config_path)
        raise


# Parsed-args-holder so run_problems() can read --docker without
# taking it as a parameter. Runners call apply_args() first.
_args_ref: dict = {"docker": False}


def make_save_path(prefix: str, tag: str | None = None,
                    resume: str | None = None) -> str:
    """runs/<prefix>_<tag>_<timestamp>.json or runs/<prefix>_<timestamp>.json.

    When resume is set, the save_path matches the existing run id so
    the artifact root resolves to the existing directory (the prior
    one). The save file itself gets overwritten on completion — that's
    fine, the per-call artifacts are the source of truth.
    """
    os.makedirs("runs", exist_ok=True)
    if resume:
        return f"runs/{resume}.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{prefix}_{tag}_{timestamp}" if tag else f"{prefix}_{timestamp}"
    return f"runs/{name}.json"


# ── Per-problem run + parallel save ──────────────────────────────

async def run_one(
    problem: dict,
    *,
    artifact_root: str | None = None,
    run_id: str | None = None,
    claude_creds_path: str | None = None,
    claude_config_path: str | None = None,
    use_docker: bool = False,
    sandbox_image: str = sbx.DEFAULT_IMAGE,
    sandbox_image_digest: str | None = None,
    config_hash: str | None = None,
) -> dict:
    """Run pipeline.run() on a problem dict and attach metadata.

    Carries over all fields from the problem dict (except 'question',
    which becomes 'problem' in the result via pipeline.run).

    Also writes a per-problem partial state file under `runs/partial/`
    that gets updated after every meaningful pipeline event. On crash
    mid-problem, that file contains everything that completed.

    When `artifact_root` is given, sets the `artifact_dir` contextvar to
    `<artifact_root>/<pid>/` so every LLM call saves its inputs/outputs
    under that per-problem directory.

    When `use_docker` is True, spins up one sandbox container per
    problem and sets the `sandbox_container` contextvar so llm() wraps
    every call in `docker exec`. The container is destroyed at problem
    end; /workspace contents are first copied to the per-problem
    artifact dir for post-mortem.
    """
    pid = problem.get("id", "?")
    event(
        logger,
        logging.INFO,
        "problem.started",
        "Problem run started",
        run_id=run_id,
        problem_id=pid,
        sandboxed=use_docker,
    )
    print(f"\n{'#'*60}")
    print(f"# {pid}")
    print(f"{'#'*60}")

    partial_path = _partial_state_path(pid, run_id)
    os.makedirs(os.path.dirname(partial_path), exist_ok=True)
    try:
        os.remove(partial_path)
    except FileNotFoundError:
        pass

    # Per-problem LLM call log. llm() appends one dict per completed call.
    # Shared across child asyncio tasks via contextvars — parallel judges
    # all append to this same list.
    calls: list[dict] = []
    token = call_log.set(calls)
    trace_fields = {
        "run_id": run_id,
        "problem_id": str(pid),
    }
    if config_hash:
        trace_fields["config_hash"] = config_hash
    if problem.get("rollout_group") is not None:
        trace_fields["rollout_group"] = str(problem["rollout_group"])
    if problem.get("sample_index") is not None:
        trace_fields["sample_index"] = problem["sample_index"]
    trace_token = telemetry_context.set(trace_fields)

    # Per-problem artifact directory. Every LLM call under this problem
    # writes its raw inputs/outputs into call_NNN_<role>/ here.
    art_token = None
    prob_art_dir: str | None = None
    if artifact_root is not None:
        prob_art_dir = _problem_artifact_directory(artifact_root, pid)
        os.makedirs(prob_art_dir, exist_ok=True)
        art_token = artifact_dir.set(prob_art_dir)

    # Per-problem sandbox container. Held for the whole problem so all
    # roles (solver, formalizer, judges, pedantry) share one environment
    # and session resume works across calls. Also prep a fresh codex
    # state tempdir per problem so codex can write trusted-project
    # state without polluting the host.
    container_id: str | None = None
    codex_state_dir: str | None = None
    sandbox_token = None
    image_token = None
    provider_state_dir = (
        os.path.join(prob_art_dir, "provider_state") if prob_art_dir else None
    )
    codex_resume_state = (
        os.path.join(provider_state_dir, "codex") if provider_state_dir else None
    )
    claude_resume_state = (
        os.path.join(provider_state_dir, "claude_projects")
        if provider_state_dir else None
    )
    workspace_resume_state = (
        os.path.join(prob_art_dir, "workspace") if prob_art_dir else None
    )
    restored_state = {"codex": False, "claude": False, "workspace": False}

    # Fallback timing — only used if sandbox startup fails before the run.
    started_iso = datetime.now().astimezone().isoformat()
    started_perf = time.perf_counter()
    # Sandbox startup is inside the try so a container failure is recorded
    # as a per-problem error (and cleaned up in `finally`) instead of
    # escaping run_one and aborting the whole batch.
    try:
        if use_docker:
            has_codex_resume = bool(
                codex_resume_state and os.path.isdir(codex_resume_state)
            )
            codex_state_dir = sbx.prepare_codex_state_dir(
                codex_resume_state if has_codex_resume else None
            )
            if has_codex_resume and codex_state_dir is None:
                raise RuntimeError("could not restore persisted Codex session state")
            restored_state["codex"] = has_codex_resume
            container_id = sbx.start_sandbox(
                pid=pid,
                run_id=run_id or "run",
                image=sandbox_image,
                claude_creds_path=claude_creds_path,
                claude_config_path=claude_config_path,
                codex_state_dir=codex_state_dir,
                allow_network=sandbox_allow_network(),
            )
            print(f"[{pid}] sandbox container: {container_id[:12]}")
            if workspace_resume_state and os.path.isdir(workspace_resume_state):
                if not sbx.copy_to_container(
                    container_id, workspace_resume_state, "/workspace",
                ):
                    raise RuntimeError("could not restore persisted sandbox workspace")
                restored_state["workspace"] = True
            if claude_resume_state and os.path.isdir(claude_resume_state):
                if not sbx.copy_to_container(
                    container_id,
                    claude_resume_state,
                    "/home/node/.claude/projects",
                ):
                    raise RuntimeError("could not restore persisted Claude session state")
                restored_state["claude"] = True
            sandbox_token = sandbox_container.set(container_id)
            if sandbox_image_digest:
                image_token = sandbox_image_id.set(sandbox_image_digest)
        # Capture timing AFTER sandbox startup so problem_duration_ms measures
        # the reasoning run, not container boot (matches steel's behavior).
        started_iso = datetime.now().astimezone().isoformat()
        started_perf = time.perf_counter()
        # Per-problem wall-clock ceiling (`_limits.max_problem_seconds`; None =
        # unbounded, the historical default). A timeout raises and is caught
        # below exactly like any other failure, so it lands in the same
        # execution-error path -- which run_records now excludes from the
        # selective-prediction stats rather than miscounting as an abstention.
        #
        # THE BUDGET BOUNDS ACCOUNTING, NOT RESOURCE USE. agent_loop awaits
        # `asyncio.to_thread(_chat_completion, ...)`, and cancelling a coroutine
        # cannot interrupt a thread already inside a blocking HTTP call: the
        # worker keeps running until the request completes or hits
        # `request_timeout` (180s by default; frozen experiment configs pin it
        # explicitly, see experiments/make_model_config.py). So a timed-out
        # problem is recorded promptly and correctly, but holds a provider slot
        # / GPU
        # for up to request_timeout longer. Two consequences worth knowing:
        # per-problem wall-clock in the artifacts is exact, while observed
        # throughput under a tight budget may lag it; and if a provider
        # concurrency semaphore is released on cancellation while the thread
        # runs on, in-flight concurrency can transiently exceed the cap. No
        # metric is affected. The real fix is a cancellation event plumbed into
        # `_chat_completion`; until then, keep max_problem_seconds comfortably
        # above request_timeout.
        budget = pipeline.max_problem_seconds()
        coro = run(problem["question"], pid=pid, partial_save_path=partial_path)
        if budget is None:
            result = await coro
        else:
            try:
                result = await asyncio.wait_for(coro, timeout=budget)
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise RuntimeError(
                    f"problem exceeded max_problem_seconds={budget:g}"
                ) from exc
    except Exception as e:
        print(f"[{pid}]   ERROR: {e}")
        traceback_path = None
        partial_state: dict = {}
        try:
            with open(partial_path) as f:
                loaded_partial = json.load(f)
            if isinstance(loaded_partial, dict):
                partial_state = loaded_partial
        except (OSError, json.JSONDecodeError):
            partial_state = {}
        partial_snapshot_path = None
        if prob_art_dir:
            traceback_path = os.path.join(prob_art_dir, "traceback.txt")
            try:
                with open(traceback_path, "w") as f:
                    f.write(traceback.format_exc())
            except OSError:
                traceback_path = None
            if partial_state:
                partial_snapshot_path = os.path.join(
                    prob_art_dir, "partial_state_on_error.json",
                )
                try:
                    with open(partial_snapshot_path, "w") as f:
                        json.dump(partial_state, f, indent=2, default=str)
                except OSError:
                    partial_snapshot_path = None
        event(
            logger,
            logging.ERROR,
            "problem.failed",
            "Problem run failed",
            run_id=run_id,
            problem_id=pid,
            error_type=type(e).__name__,
        )
        result = {
            "answer": None,
            "verified": False,
            "error": str(e),
            "traceback_path": traceback_path,
            "partial_state_path": partial_snapshot_path,
            "solution": partial_state.get("solution"),
            "solver_solutions": partial_state.get("solver_solutions") or [],
            "attempts": partial_state.get("attempts") or [],
            "in_progress_attempt": partial_state.get("in_progress_attempt"),
            "partial_last_event": partial_state.get("last_event"),
        }
    finally:
        call_log.reset(token)
        telemetry_context.reset(trace_token)
        if art_token is not None:
            artifact_dir.reset(art_token)
        if sandbox_token is not None:
            sandbox_container.reset(sandbox_token)
        if image_token is not None:
            sandbox_image_id.reset(image_token)

        # Snapshot container state into artifacts BEFORE stopping.
        # - /workspace captures every scratch file any agent wrote
        # - /home/node/.local captures any runtime `pip install`s
        # - pip_freeze.txt is the definitive post-run Python env
        # - dpkg_list.txt is the system-package equivalent
        # - docker inspect captures authoritative container exit state
        #   (OOMKilled flag, actual limits applied, restart count) —
        #   invaluable when diagnosing mysterious call failures
        container_inspect: dict | None = None
        if container_id and prob_art_dir:
            sbx.snapshot_container_directory(
                container_id, "/workspace",
                os.path.join(prob_art_dir, "workspace"),
            )
            sbx.snapshot_container_directory(
                container_id, "/home/node/.local",
                os.path.join(prob_art_dir, "home_local"),
            )
            if provider_state_dir:
                os.makedirs(provider_state_dir, exist_ok=True)
                sbx.snapshot_container_directory(
                    container_id,
                    "/home/node/.claude/projects",
                    os.path.join(provider_state_dir, "claude_projects"),
                )
                if codex_state_dir:
                    sbx.snapshot_codex_resume_state(
                        codex_state_dir,
                        os.path.join(provider_state_dir, "codex"),
                    )
                try:
                    with open(
                        os.path.join(provider_state_dir, "restore_status.json"),
                        "w",
                    ) as f:
                        json.dump(restored_state, f, indent=2)
                except OSError:
                    pass
            freeze = sbx.pip_freeze_in_container(container_id)
            if freeze is not None:
                try:
                    with open(
                        os.path.join(prob_art_dir, "pip_freeze.txt"), "w",
                    ) as f:
                        f.write(freeze)
                except OSError:
                    pass
            dpkg = sbx.dpkg_list_in_container(container_id)
            if dpkg is not None:
                try:
                    with open(
                        os.path.join(prob_art_dir, "dpkg_list.txt"), "w",
                    ) as f:
                        f.write(dpkg)
                except OSError:
                    pass
            container_inspect = sbx.inspect_container(container_id)
            if container_inspect is not None:
                try:
                    with open(
                        os.path.join(prob_art_dir, "docker_inspect.json"), "w",
                    ) as f:
                        json.dump(container_inspect, f, indent=2, default=str)
                except OSError:
                    pass
        if container_id:
            sbx.stop_sandbox(container_id)
        if codex_state_dir:
            sbx.cleanup_codex_state_dir(codex_state_dir)

    problem_duration_ms = int(round((time.perf_counter() - started_perf) * 1000))
    finished_iso = datetime.now().astimezone().isoformat()

    # Aggregate per-role metrics from the call log. A remaining None means the
    # process was interrupted before llm() could finalize even a failure entry.
    unlogged_call_slots = sum(call is None for call in calls)
    calls = [c for c in calls if c is not None]
    calls_by_role: dict[str, int] = {}
    calls_by_model: dict[str, int] = {}
    tokens_by_role: dict[str, int] = {}
    duration_by_role: dict[str, int] = {}
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cost = 0.0
    priced_calls = 0
    usage_available_calls = 0
    usage_complete_calls = 0
    total_llm_duration_ms = 0
    total_retries = 0
    failed_calls = 0
    total_call_invocations = 0
    failed_call_invocations = 0
    resume_retry_count = 0
    cached_calls = 0
    resumed_session_calls = 0
    protocol_reprompt_count = 0
    json_action_reprompts = 0
    json_action_repairs = 0
    budget_truncated_turns = 0
    budget_truncated_reprompts = 0
    role_schema_reprompts = 0
    unknown_action_reprompts = 0
    required_tool_policy_reprompts = 0
    total_tool_calls = 0
    tool_calls_by_name: dict[str, int] = {}
    tool_calls_by_role: dict[str, dict[str, int]] = {}
    tool_status_counts = {"success": 0, "failure": 0, "unknown": 0}
    tool_status_by_role: dict[str, dict[str, int]] = {}
    required_tool_calls = 0
    compliant_required_tool_calls = 0
    required_tool_obligations = 0
    satisfied_required_tool_obligations = 0
    required_tool_obligations_by_role: dict[str, dict[str, int]] = {}
    satisfied_required_tools_by_role: dict[str, dict[str, int]] = {}
    mechanistic_evidence: dict[str, dict] = {}
    web_search_providers: set[str] = set()
    web_search_requests = 0
    web_search_attempts = 0
    web_search_provider_requests = 0
    web_search_successes = 0
    web_search_failures = 0
    web_search_client_failures = 0
    web_search_provider_failures = 0
    web_search_result_count = 0
    # Contamination controls (configs/searxng_search.yaml): layer 1 drops a
    # result by host, layer 2 by content signature. These are the audit trail
    # for the tool-ON condition, so they have to survive aggregation.
    web_search_blocked_count = 0
    web_search_content_blocked_count = 0
    web_search_latency_ms = 0
    web_search_error_categories: dict[str, int] = {}
    for c in calls:
        r = c.get("role", "?")
        calls_by_role[r] = calls_by_role.get(r, 0) + 1
        model_key = f"{c.get('backend', '?')}:{c.get('model') or '?'}"
        calls_by_model[model_key] = calls_by_model.get(model_key, 0) + 1
        failed_calls += int(bool(c.get("failed")))
        total_call_invocations += int(c.get("invocation_count", 1) or 1)
        failed_call_invocations += int(
            c.get("failed_invocations", int(bool(c.get("failed")))) or 0
        )
        resume_retry_count += int(c.get("resume_retry_count", 0) or 0)
        cached_calls += int(bool(c.get("resumed_from_cache")))
        resumed_session_calls += int(bool(c.get("resumed")))
        protocol_reprompt_count += int(c.get("protocol_reprompt_count", 0) or 0)
        json_action_reprompts += int(c.get("json_action_reprompts", 0) or 0)
        json_action_repairs += int(c.get("json_action_repairs", 0) or 0)
        budget_truncated_turns += int(c.get("budget_truncated_turns", 0) or 0)
        budget_truncated_reprompts += int(
            c.get("budget_truncated_reprompts", 0) or 0
        )
        role_schema_reprompts += int(c.get("role_schema_reprompts", 0) or 0)
        unknown_action_reprompts += int(c.get("unknown_action_reprompts", 0) or 0)
        required_tool_policy_reprompts += int(
            c.get("required_tool_policy_reprompts", 0) or 0
        )
        total_retries += int(c.get("retry_count", 0) or 0)
        in_tokens = c.get("input_tokens", 0) or 0
        out_tokens = c.get("output_tokens", 0) or 0
        usage_observed = c.get("usage_observed")
        if usage_observed is None:
            usage = c.get("usage") or {}
            usage_observed = (
                usage.get("source") != "unavailable"
                if usage else "input_tokens" in c or "output_tokens" in c
            )
        if usage_observed:
            usage_available_calls += 1
        usage_complete = c.get("usage_complete")
        if usage_complete is None:
            usage_complete = bool((c.get("usage") or {}).get("complete"))
        usage_complete_calls += int(bool(usage_complete))
        tokens_by_role[r] = tokens_by_role.get(r, 0) + in_tokens + out_tokens
        dur = c.get("duration_ms", 0) or 0
        duration_by_role[r] = duration_by_role.get(r, 0) + dur
        total_input += in_tokens
        total_output += out_tokens
        total_cache_read += c.get("cache_read_input_tokens", 0) or 0
        total_llm_duration_ms += dur
        cost = (c.get("cost") or {}).get("amount_usd")
        if cost is None:
            cost = c.get("total_cost_usd")
        if cost is not None:
            total_cost += float(cost)
            priced_calls += 1
        evidence = mechanistic_evidence.setdefault(r, {
            "calls": 0,
            "tool_calls": 0,
            "successful_tool_calls": 0,
            "failed_tool_calls": 0,
            "unknown_status_tool_calls": 0,
            "shell_calls": 0,
            "successful_shell_calls": 0,
            "web_search_calls": 0,
            "successful_web_search_calls": 0,
        })
        evidence["calls"] += 1
        required = {
            str(name) for name in (c.get("required_tools") or [])
        }
        satisfied = {
            str(name) for name in (c.get("successful_required_tools") or [])
        }
        if required:
            required_tool_calls += 1
            compliant_required_tool_calls += int(required <= satisfied)
            required_tool_obligations += len(required)
            satisfied_required_tool_obligations += len(required & satisfied)
            role_required = required_tool_obligations_by_role.setdefault(r, {})
            role_satisfied = satisfied_required_tools_by_role.setdefault(r, {})
            for name in required:
                role_required[name] = role_required.get(name, 0) + 1
            for name in required & satisfied:
                role_satisfied[name] = role_satisfied.get(name, 0) + 1
        for tool_call in c.get("all_tool_calls") or c.get("tool_calls") or []:
            name = tool_call.get("tool_name") or "unknown"
            total_tool_calls += 1
            tool_calls_by_name[name] = tool_calls_by_name.get(name, 0) + 1
            role_tools = tool_calls_by_role.setdefault(r, {})
            role_tools[name] = role_tools.get(name, 0) + 1
            evidence["tool_calls"] += 1
            metadata = tool_call.get("metadata") or {}
            ok = metadata.get("ok")
            status = "success" if ok is True else "failure" if ok is False else "unknown"
            tool_status_counts[status] += 1
            role_status = tool_status_by_role.setdefault(
                r, {"success": 0, "failure": 0, "unknown": 0},
            )
            role_status[status] += 1
            evidence_key = {
                "success": "successful_tool_calls",
                "failure": "failed_tool_calls",
                "unknown": "unknown_status_tool_calls",
            }[status]
            evidence[evidence_key] += 1
            if name in {"shell", "exec_command"}:
                evidence["shell_calls"] += 1
                if ok is True:
                    evidence["successful_shell_calls"] += 1
            if name in {"web_search", "search"}:
                evidence["web_search_calls"] += 1
                if ok is True:
                    evidence["successful_web_search_calls"] += 1

        call_search_requests = int(c.get("web_search_requests", 0) or 0)
        call_search_attempts = int(
            c.get("web_search_attempts", call_search_requests) or 0
        )
        call_provider_requests = int(
            c.get("web_search_provider_requests", call_search_requests) or 0
        )
        if c.get("web_search_provider") and call_search_requests:
            web_search_providers.add(str(c["web_search_provider"]))
        web_search_requests += call_search_requests
        web_search_attempts += call_search_attempts
        web_search_provider_requests += call_provider_requests
        web_search_successes += int(c.get("web_search_successes", 0) or 0)
        call_search_failures = int(c.get("web_search_failures", 0) or 0)
        web_search_failures += call_search_failures
        web_search_client_failures += int(
            c.get("web_search_client_failures", 0) or 0
        )
        web_search_provider_failures += int(
            c.get("web_search_provider_failures", call_search_failures) or 0
        )
        web_search_result_count += int(c.get("web_search_result_count", 0) or 0)
        web_search_blocked_count += int(c.get("web_search_blocked_count", 0) or 0)
        web_search_content_blocked_count += int(
            c.get("web_search_content_blocked_count", 0) or 0
        )
        web_search_latency_ms += int(
            c.get(
                "web_search_total_latency_ms",
                c.get("web_search_latency_ms", 0),
            ) or 0
        )
        for category, count in (c.get("web_search_error_categories") or {}).items():
            web_search_error_categories[category] = (
                web_search_error_categories.get(category, 0) + int(count or 0)
            )

    result["calls"] = calls
    normalized_metrics = aggregate_calls(
        calls,
        problem_started_at=started_iso,
        problem_ended_at=finished_iso,
        problem_duration_ms=problem_duration_ms,
    )
    # Retain the richer audit fields alongside the normalized telemetry schema.
    # Complete cost remains null unless every call has auditable pricing.
    normalized_metrics.update({
        "problem_started_at": started_iso,
        "problem_ended_at": finished_iso,
        "problem_duration_ms": problem_duration_ms,
        "total_llm_duration_ms": total_llm_duration_ms,
        "num_calls": len(calls),
        "successful_calls": len(calls) - failed_calls,
        "failed_calls": failed_calls,
        "total_call_invocations": total_call_invocations,
        "failed_call_invocations": failed_call_invocations,
        "resume_retry_count": resume_retry_count,
        "unlogged_call_slots": unlogged_call_slots,
        "cached_calls": cached_calls,
        "resumed_session_calls": resumed_session_calls,
        "protocol_reprompt_count": protocol_reprompt_count,
        "json_action_reprompts": json_action_reprompts,
        # Explain the protocol number rather than only reporting it: repairs are
        # replies the harness rescued, truncations are turns the token cap ended.
        # Without these a reader cannot tell a compliant model from a
        # badly-budgeted one.
        "json_action_repairs": json_action_repairs,
        "budget_truncated_turns": budget_truncated_turns,
        "budget_truncated_reprompts": budget_truncated_reprompts,
        "role_schema_reprompts": role_schema_reprompts,
        "unknown_action_reprompts": unknown_action_reprompts,
        "required_tool_policy_reprompts": required_tool_policy_reprompts,
        "total_retries": total_retries,
        "calls_by_role": calls_by_role,
        "calls_by_model": calls_by_model,
        "tokens_by_role": tokens_by_role,
        "duration_by_role": duration_by_role,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_read_input_tokens": total_cache_read,
        "total_tokens": total_input + total_output,
        "usage_available_calls": usage_available_calls,
        "usage_coverage_fraction": (
            usage_available_calls / len(calls) if calls else None
        ),
        "usage_complete_calls": usage_complete_calls,
        "usage_complete_fraction": (
            usage_complete_calls / len(calls) if calls else None
        ),
        "total_cost_usd": (
            total_cost if calls and priced_calls == len(calls) else None
        ),
        "partial_cost_usd": total_cost if priced_calls else None,
        "priced_calls": priced_calls,
        "cost_coverage_fraction": priced_calls / len(calls) if calls else None,
        "total_tool_calls": total_tool_calls,
        "tool_calls_by_name": tool_calls_by_name,
        "tool_calls_by_role": tool_calls_by_role,
        "tool_status_counts": tool_status_counts,
        "tool_status_by_role": tool_status_by_role,
        "required_tool_calls": required_tool_calls,
        "compliant_required_tool_calls": compliant_required_tool_calls,
        "required_tool_call_compliance_fraction": (
            compliant_required_tool_calls / required_tool_calls
            if required_tool_calls else None
        ),
        "required_tool_obligations": required_tool_obligations,
        "satisfied_required_tool_obligations": satisfied_required_tool_obligations,
        "required_tool_obligation_compliance_fraction": (
            satisfied_required_tool_obligations / required_tool_obligations
            if required_tool_obligations else None
        ),
        "required_tool_obligations_by_role": required_tool_obligations_by_role,
        "satisfied_required_tools_by_role": satisfied_required_tools_by_role,
        "mechanistic_evidence_by_role": mechanistic_evidence,
        "web_search_providers": sorted(web_search_providers),
        "web_search_requests": web_search_requests,
        "web_search_attempts": web_search_attempts,
        "web_search_provider_requests": web_search_provider_requests,
        "web_search_successes": web_search_successes,
        "web_search_failures": web_search_failures,
        "web_search_client_failures": web_search_client_failures,
        "web_search_provider_failures": web_search_provider_failures,
        "web_search_result_count": web_search_result_count,
        "web_search_blocked_count": web_search_blocked_count,
        "web_search_content_blocked_count": web_search_content_blocked_count,
        "web_search_latency_ms": web_search_latency_ms,
        "web_search_total_latency_ms": web_search_latency_ms,
        "web_search_mean_latency_ms": (
            web_search_latency_ms / web_search_provider_requests
            if web_search_provider_requests else None
        ),
        "web_search_error_categories": web_search_error_categories,
    })
    result["metrics"] = normalized_metrics
    if config_hash:
        result["config_hash"] = config_hash

    # Carry over all problem metadata (id, category, expected, etc.)
    for k, v in problem.items():
        if k == "question":
            continue
        result.setdefault(k, v)

    # Naive correctness check — for analysis only, NOT authoritative.
    # Case-insensitive token-boundary match. The expected answer must
    # appear in the given answer bounded on both sides by either string
    # ends or non-alphanumeric characters. This catches "Canals on Mars"
    # inside "ANSWER = canals on Mars (Martian canals)" while rejecting
    # false positives like "1" matching inside "21".
    expected = problem.get("answer", "")
    given = (result.get("answer") or "").strip()
    result["expected"] = expected
    correct = False
    exp_norm = expected.strip().lower()
    if exp_norm:
        given_norm = given.lower()
        pattern = (
            r"(?:^|(?<=[^a-z0-9]))"
            + re.escape(exp_norm)
            + r"(?:$|(?=[^a-z0-9]))"
        )
        correct = bool(re.search(pattern, given_norm))
    result["correct"] = correct
    event(
        logger,
        logging.INFO,
        "problem.completed",
        "Problem run completed",
        run_id=run_id,
        problem_id=pid,
        verified=result.get("verified"),
        duration_ms=problem_duration_ms,
        num_calls=result["metrics"]["num_calls"],
        failed_calls=result["metrics"]["failed_calls"],
    )
    return result


def _aggregate_run_metrics(
    results: list[dict],
    *,
    requested: int,
    duration_ms: int,
) -> dict:
    metric_rows = [result.get("metrics") or {} for result in results]
    verified = sum(bool(result.get("verified")) for result in results)
    errors = sum(bool(result.get("error")) for result in results)
    have_expected = [
        result for result in results if str(result.get("expected") or "").strip()
    ]
    correct = sum(bool(result.get("correct")) for result in have_expected)
    repair_rows = [
        result.get("repair_metrics") for result in results
        if result.get("repair_metrics")
    ]

    def sum_field(field: str) -> int:
        return sum(int(row.get(field, 0) or 0) for row in metric_rows)

    def merge_counts(field: str) -> dict[str, int]:
        merged: dict[str, int] = {}
        for row in metric_rows:
            for key, value in (row.get(field) or {}).items():
                merged[key] = merged.get(key, 0) + int(value or 0)
        return merged

    def merge_nested_counts(field: str) -> dict[str, dict[str, int]]:
        merged: dict[str, dict[str, int]] = {}
        for row in metric_rows:
            for group, counts in (row.get(field) or {}).items():
                target = merged.setdefault(group, {})
                for key, value in (counts or {}).items():
                    target[key] = target.get(key, 0) + int(value or 0)
        return merged

    priced_calls = sum_field("priced_calls")
    num_calls = sum_field("num_calls")
    usage_available_calls = sum_field("usage_available_calls")
    usage_complete_calls = sum_field("usage_complete_calls")
    web_search_attempts = sum(
        int(row.get("web_search_attempts", row.get("web_search_requests", 0)) or 0)
        for row in metric_rows
    )
    web_search_provider_requests = sum(
        int(
            row.get(
                "web_search_provider_requests",
                row.get("web_search_requests", 0),
            ) or 0
        )
        for row in metric_rows
    )
    web_search_total_latency_ms = sum(
        int(
            row.get(
                "web_search_total_latency_ms",
                row.get("web_search_latency_ms", 0),
            ) or 0
        )
        for row in metric_rows
    )
    web_search_client_failures = sum(
        int(row.get("web_search_client_failures", 0) or 0)
        for row in metric_rows
    )
    web_search_provider_failures = sum(
        int(
            row.get(
                "web_search_provider_failures",
                row.get("web_search_failures", 0),
            ) or 0
        )
        for row in metric_rows
    )
    partial_costs = [
        row.get("partial_cost_usd") for row in metric_rows
        if row.get("partial_cost_usd") is not None
    ]
    required_calls = sum_field("required_tool_calls")
    compliant_required_calls = sum_field("compliant_required_tool_calls")
    required_obligations = sum_field("required_tool_obligations")
    satisfied_required_obligations = sum_field(
        "satisfied_required_tool_obligations"
    )
    return {
        "duration_ms": duration_ms,
        "requested_problems": requested,
        "completed_problems": len(results),
        "failed_problems": errors,
        "judge_passed": verified,
        "coverage": verified / len(results) if results else None,
        "graded_by_naive_key": len(have_expected),
        "naive_key_matches": correct,
        "naive_accuracy": correct / len(have_expected) if have_expected else None,
        "num_calls": num_calls,
        "successful_calls": sum_field("successful_calls"),
        "failed_calls": sum_field("failed_calls"),
        "total_call_invocations": sum_field("total_call_invocations"),
        "failed_call_invocations": sum_field("failed_call_invocations"),
        "resume_retry_count": sum_field("resume_retry_count"),
        "unlogged_call_slots": sum_field("unlogged_call_slots"),
        "cached_calls": sum_field("cached_calls"),
        "resumed_session_calls": sum_field("resumed_session_calls"),
        "total_retries": sum_field("total_retries"),
        "protocol_reprompt_count": sum_field("protocol_reprompt_count"),
        "json_action_reprompts": sum_field("json_action_reprompts"),
        "json_action_repairs": sum_field("json_action_repairs"),
        "budget_truncated_turns": sum_field("budget_truncated_turns"),
        "budget_truncated_reprompts": sum_field("budget_truncated_reprompts"),
        "role_schema_reprompts": sum_field("role_schema_reprompts"),
        "unknown_action_reprompts": sum_field("unknown_action_reprompts"),
        "required_tool_policy_reprompts": sum_field(
            "required_tool_policy_reprompts"
        ),
        "calls_by_role": merge_counts("calls_by_role"),
        "calls_by_model": merge_counts("calls_by_model"),
        "tokens_by_role": merge_counts("tokens_by_role"),
        "duration_by_role": merge_counts("duration_by_role"),
        "total_llm_duration_ms": sum_field("total_llm_duration_ms"),
        "total_input_tokens": sum_field("total_input_tokens"),
        "total_output_tokens": sum_field("total_output_tokens"),
        "total_cache_read_input_tokens": sum_field(
            "total_cache_read_input_tokens"
        ),
        "total_tokens": sum_field("total_tokens"),
        "usage_available_calls": usage_available_calls,
        "usage_coverage_fraction": (
            usage_available_calls / num_calls if num_calls else None
        ),
        "usage_complete_calls": usage_complete_calls,
        "usage_complete_fraction": (
            usage_complete_calls / num_calls if num_calls else None
        ),
        "total_tool_calls": sum_field("total_tool_calls"),
        "tool_calls_by_name": merge_counts("tool_calls_by_name"),
        "tool_calls_by_role": merge_nested_counts("tool_calls_by_role"),
        "tool_status_counts": merge_counts("tool_status_counts"),
        "tool_status_by_role": merge_nested_counts("tool_status_by_role"),
        "required_tool_calls": required_calls,
        "compliant_required_tool_calls": compliant_required_calls,
        "required_tool_call_compliance_fraction": (
            compliant_required_calls / required_calls
            if required_calls else None
        ),
        "required_tool_obligations": required_obligations,
        "satisfied_required_tool_obligations": satisfied_required_obligations,
        "required_tool_obligation_compliance_fraction": (
            satisfied_required_obligations / required_obligations
            if required_obligations else None
        ),
        "required_tool_obligations_by_role": merge_nested_counts(
            "required_tool_obligations_by_role"
        ),
        "satisfied_required_tools_by_role": merge_nested_counts(
            "satisfied_required_tools_by_role"
        ),
        "mechanistic_evidence_by_role": merge_nested_counts(
            "mechanistic_evidence_by_role"
        ),
        "web_search_providers": sorted({
            str(provider)
            for row in metric_rows
            for provider in (row.get("web_search_providers") or [])
        }),
        "web_search_requests": sum_field("web_search_requests"),
        "web_search_attempts": web_search_attempts,
        "web_search_provider_requests": web_search_provider_requests,
        "web_search_successes": sum_field("web_search_successes"),
        "web_search_failures": sum_field("web_search_failures"),
        "web_search_client_failures": web_search_client_failures,
        "web_search_provider_failures": web_search_provider_failures,
        "web_search_result_count": sum_field("web_search_result_count"),
        "web_search_blocked_count": sum_field("web_search_blocked_count"),
        "web_search_content_blocked_count": sum_field(
            "web_search_content_blocked_count"
        ),
        "web_search_latency_ms": sum_field("web_search_latency_ms"),
        "web_search_total_latency_ms": web_search_total_latency_ms,
        "web_search_mean_latency_ms": (
            web_search_total_latency_ms / web_search_provider_requests
            if web_search_provider_requests else None
        ),
        "web_search_error_categories": merge_counts(
            "web_search_error_categories"
        ),
        "priced_calls": priced_calls,
        "cost_coverage_fraction": priced_calls / num_calls if num_calls else None,
        "partial_cost_usd": sum(partial_costs) if partial_costs else None,
        "total_cost_usd": (
            sum(partial_costs)
            if num_calls and priced_calls == num_calls else None
        ),
        "repair_metrics_available": len(repair_rows),
        "repair_attempted": sum(
            bool(row.get("repair_attempted")) for row in repair_rows
        ),
        "certified_after_any_retry": sum(
            bool(row.get("certified_after_any_retry")) for row in repair_rows
        ),
        "certified_after_semantic_repair": sum(
            bool(row.get("certified_after_semantic_repair"))
            for row in repair_rows
        ),
        "certified_after_nonsemantic_retry": sum(
            bool(row.get("certified_after_nonsemantic_retry"))
            for row in repair_rows
        ),
        "answer_changed_during_repair": sum(
            bool(row.get("answer_changed_during_repair")) for row in repair_rows
        ),
    }


async def run_problems(
    problems: list[dict],
    parallel: int,
    save_path: str,
) -> list[dict]:
    """Run problems with bounded parallelism and incremental save.

    Uses a semaphore so a new problem starts as soon as any in-flight
    one finishes, instead of waiting for the slowest in a chunk. Each
    completed problem is atomically saved to save_path immediately, so
    ctrl+C never loses results.

    Creates runs/artifacts/<run_id>/ once at start (with meta.json
    capturing argv / config / CLI versions / git sha) and passes it
    through to each run_one so every LLM call writes full artifacts.

    If apply_args() recorded --docker, also:
      - extracts claude subscription creds from the macOS Keychain
        into a tempfile so the sandbox containers can mount it;
      - looks up the local sandbox image digest for call metadata;
      - tells each run_one to spin up its own per-problem container.
    """
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_started_perf = time.perf_counter()
    samples = int(_args_ref.get("samples") or 1)
    distinct_problems = len(problems)
    problems = expand_problem_samples(problems, samples)
    problem_manifest = _problem_set_manifest(problems)
    artifact_root = make_artifact_root(
        save_path,
        problem_manifest=problem_manifest,
    )
    run_id = os.path.basename(save_path)
    if run_id.endswith(".json"):
        run_id = run_id[:-5]
    config_hash = compute_config_hash()
    events_path = attach_events_file(run_id)
    dataset_provenance = {
        (
            problem.get("dataset_name") or problem.get("dataset_source"),
            problem.get("dataset_split"),
            problem.get("dataset_revision"),
            problem.get("dataset_fingerprint"),
        )
        for problem in problems
        if problem.get("dataset_name") or problem.get("dataset_source")
    }
    if dataset_provenance:
        update_artifact_root_meta(artifact_root, {
            "datasets": [
                {
                    "name": name,
                    "split": split,
                    "requested_revision": revision,
                    "fingerprint": fingerprint,
                }
                for name, split, revision, fingerprint in sorted(
                    dataset_provenance, key=lambda item: str(item),
                )
            ],
        }, strict=True)
    update_artifact_root_meta(artifact_root, {
        "config_hash": config_hash,
        "events_path": events_path,
        "samples_per_problem": samples,
    }, strict=True)
    event(
        logger,
        logging.INFO,
        "run.started",
        "Run started",
        run_id=run_id,
        save_path=save_path,
        requested_problems=len(problems),
        distinct_problems=distinct_problems,
        parallel=parallel,
        config_hash=config_hash,
        samples_per_problem=samples,
        events_path=events_path,
    )
    print(f"Artifact root: {artifact_root}")

    use_docker = _args_ref.get("docker", False)
    sandbox_image = _args_ref.get("image") or sbx.DEFAULT_IMAGE
    claude_creds_path: str | None = None
    claude_config_path: str | None = None
    image_digest: str | None = None

    results: list[dict] = []
    result_order = {
        str(problem.get("id", "?")): index
        for index, problem in enumerate(problems)
    }
    completed_ids: set[str] = set()
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(parallel)

    async def run_and_save(problem):
        async with sem:
            result = await run_one(
                problem,
                artifact_root=artifact_root,
                run_id=run_id,
                claude_creds_path=claude_creds_path,
                claude_config_path=claude_config_path,
                use_docker=use_docker,
                sandbox_image=sandbox_image,
                sandbox_image_digest=image_digest,
                config_hash=config_hash,
            )
        async with lock:
            _write_problem_checkpoint(artifact_root, problem, result)
            results.append(result)
            completed_ids.add(str(problem.get("id", "?")))
            results.sort(
                key=lambda row: result_order.get(str(row.get("id")), len(problems))
            )
            tmp_path = save_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(results, f, indent=2)
            os.replace(tmp_path, save_path)
            update_artifact_root_meta(artifact_root, {
                "checkpointed_problem_ids": sorted(
                    completed_ids, key=lambda pid: result_order.get(pid, len(problems)),
                ),
            }, strict=True)
            print(f"\n  [{len(results)}/{len(problems)} saved to {save_path}]")
        return result

    run_error: BaseException | None = None
    run_error_traceback = None
    try:
        record_problem_set_manifest(artifact_root, problem_manifest)
        duplicate_ids = problem_manifest.get("duplicate_ids") or []
        if duplicate_ids:
            raise ValueError(
                "problem IDs must be unique because IDs key result and "
                f"artifact records; duplicates={duplicate_ids}"
            )
        meta = _read_meta(artifact_root)
        resumes = meta.get("resumes") or []
        integrity_mode = (
            resumes[-1].get("integrity_mode") if resumes else None
        )
        recovered = _recover_problem_checkpoints(
            artifact_root,
            problems,
            integrity_mode=integrity_mode,
        )
        if recovered:
            results.extend(recovered)
            completed_ids.update(str(result.get("id")) for result in recovered)
            results.sort(
                key=lambda row: result_order.get(str(row.get("id")), len(problems))
            )
        if resumes:
            # Rebuild the mutable external results file exclusively from
            # verified checkpoints. It may have been updated after the last
            # durable checkpoint when the previous process was killed.
            tmp_path = save_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(results, f, indent=2)
            os.replace(tmp_path, save_path)
        (
            claude_creds_path,
            claude_config_path,
            image_digest,
        ) = _prepare_sandbox_run(
            artifact_root,
            use_docker=use_docker,
            sandbox_image=sandbox_image,
        )
        pending_problems = [
            problem for problem in problems
            if str(problem.get("id", "?")) not in completed_ids
        ]
        await asyncio.gather(*[run_and_save(p) for p in pending_problems])
    except BaseException as exc:
        run_error = exc
        run_error_traceback = exc.__traceback__

    finalization_error: BaseException | None = None
    try:
        if run_error is None and len(results) == len(problems):
            status = (
                "completed_with_errors"
                if any(result.get("error") for result in results)
                else "completed"
            )
        elif isinstance(run_error, (asyncio.CancelledError, KeyboardInterrupt)):
            status = "interrupted"
        else:
            status = "failed"
        run_duration_ms = int(round(
            (time.perf_counter() - run_started_perf) * 1000
        ))
        run_ended_at = datetime.now(timezone.utc).isoformat()
        all_calls = [
            call
            for result in results
            for call in (result.get("calls") or [])
            if call is not None
        ]
        # Problems that never produced an in-memory result (crash /
        # interrupt) still wrote per-call meta.json artifacts. Fold those
        # in so run-level telemetry reflects every call actually made.
        # Partial in-memory call lists undercount by design: a resumed
        # problem whose `calls` only carries post-resume entries is
        # skipped here entirely (pre-resume calls exist only on disk).
        # Always reading disk instead would double-count for normal
        # problems — do not "fix" this into unconditional disk reads.
        covered_ids = {
            os.path.basename(_problem_artifact_directory(
                artifact_root, result.get("id", "?"),
            ))
            for result in results
            if result.get("calls")
        }
        all_calls.extend(_calls_from_artifact_tree(
            artifact_root, skip_problem_ids=covered_ids,
        ))
        run_metrics = aggregate_calls(
            all_calls,
            problem_started_at=run_started_at,
            problem_ended_at=run_ended_at,
            problem_duration_ms=run_duration_ms,
            scope="run",
        )
        run_metrics.update(_aggregate_run_metrics(
            results,
            requested=len(problems),
            duration_ms=run_duration_ms,
        ))
        run_metrics["scope"] = "run"
        telemetry_path = os.path.join(artifact_root, "telemetry.json")
        telemetry_tmp = telemetry_path + ".tmp"
        with open(telemetry_tmp, "w") as f:
            json.dump(run_metrics, f, indent=2, default=str)
        os.replace(telemetry_tmp, telemetry_path)
        update_artifact_root_meta(
            artifact_root,
            {
                "metrics": run_metrics,
                "telemetry_path": telemetry_path,
            },
            strict=True,
        )
        finalize_artifact_root(
            artifact_root,
            status=status,
            requested=len(problems),
            completed=len(results),
            completed_ids=[str(result.get("id", "?")) for result in results],
            error=run_error,
        )
        write_artifact_manifest(
            artifact_root,
            external_paths=[save_path],
        )
        event(
            logger,
            logging.INFO,
            "run.completed",
            "Run completed",
            run_id=run_id,
            status=status,
            completed_problems=len(results),
            requested_problems=len(problems),
            duration_ms=run_duration_ms,
            failed_calls=run_metrics["failed_calls"],
        )
    except BaseException as exc:
        finalization_error = exc
        event(
            logger,
            logging.ERROR,
            "run.finalization_failed",
            "Run finalization failed",
            run_id=run_id,
            error_type=type(exc).__name__,
        )
        try:
            finalize_artifact_root(
                artifact_root,
                status="failed",
                requested=len(problems),
                completed=len(results),
                completed_ids=[
                    str(result.get("id", "?")) for result in results
                ],
                error=exc,
            )
            update_artifact_root_meta(artifact_root, {
                "finalization_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }, strict=True)
            write_artifact_manifest(
                artifact_root,
                external_paths=[save_path],
            )
        except BaseException:
            pass
    finally:
        detach_events_file()
        if claude_creds_path:
            sbx.cleanup_credentials(claude_creds_path)
        if claude_config_path:
            sbx.cleanup_claude_config(claude_config_path)

    if run_error is not None:
        raise run_error.with_traceback(run_error_traceback)
    if finalization_error is not None:
        raise finalization_error
    return results


# ── Summary printing ─────────────────────────────────────────────

def print_summary(results: list[dict], by_category: bool = False) -> None:
    """Print a results summary.

    Two independent signals are reported, and they should not be confused:

      - "judge-passed": did every step of the proof pass the judges? This
        is the thing Verifier actually decides, and it's reported for every
        run (the internal result field is still `verified`).
      - "correct": did the answer match a known expected answer? This is a
        naive substring check, only meaningful for benchmark runs that HAVE
        expected answers. For custom questions there is no expected answer,
        so this is omitted entirely rather than shown as a misleading ✗.
    """
    print(f"\n\n{'='*60}")
    print(f"SUMMARY — {len(results)} problems")
    print(f"{'='*60}\n")

    have_key = any((r.get("expected") or "").strip() for r in results)

    if by_category:
        from collections import Counter

        def category_label(result: dict) -> str:
            # Prepared AIME/MATH records can carry a list of categories. Keep
            # that source metadata in the run, but use a hashable display label.
            value = result.get("category")
            if isinstance(value, (list, tuple)):
                return ", ".join(str(item) for item in value) or "?"
            return str(value) if value is not None else "?"

        cats = Counter(category_label(r) for r in results)
        passed_by_cat = Counter(
            category_label(r) for r in results if r.get("verified")
        )
        correct_by_cat = Counter(
            category_label(r) for r in results if r.get("correct")
        )
        for cat in sorted(cats):
            t = cats[cat]
            line = f"  {cat:28s}  {passed_by_cat.get(cat, 0)}/{t} judge-passed"
            if have_key:
                line += f" · {correct_by_cat.get(cat, 0)}/{t} correct"
            print(line)
    else:
        for r in results:
            status = "JUDGE-PASSED" if r.get("verified") else "REJECTED"
            pid = (r.get("id") or "?")[:20]
            answer = r.get("answer") or r.get("error", "ERROR")
            key = (r.get("expected") or "").strip()
            # The ✓/✗ marks correctness vs a known answer. With no expected
            # answer there's nothing to check, so use a neutral marker.
            mark = (" " if not key else ("✓" if r.get("correct") else "✗"))
            print(f"  {mark} {pid:20s}  {status:13s}  answer: {answer}")
            if key and not r.get("correct"):
                print(f"    expected: {r.get('expected', '')}")

    total = len(results)
    passed = sum(1 for r in results if r.get("verified"))
    pct_p = (100 * passed / total) if total else 0
    print(f"\n  Judge-passed: {passed}/{total} ({pct_p:.1f}%)")
    repair_rows = [r["repair_metrics"] for r in results if r.get("repair_metrics")]
    if repair_rows:
        repair_attempted = sum(1 for r in repair_rows if r.get("repair_attempted"))
        certified_after_any_retry = sum(
            1 for r in repair_rows if r.get("certified_after_any_retry")
        )
        certified_after_semantic_repair = sum(
            1 for r in repair_rows if r.get("certified_after_semantic_repair")
        )
        certified_after_nonsemantic_retry = sum(
            1 for r in repair_rows if r.get("certified_after_nonsemantic_retry")
        )
        answer_changed = sum(
            1 for r in repair_rows if r.get("answer_changed_during_repair")
        )
        answer_change_unknown = sum(
            1 for r in repair_rows
            if r.get("repair_attempted")
            and not r.get("answer_change_observable", False)
        )
        solver_solution_changed = sum(
            1 for r in repair_rows
            if r.get("solver_solution_text_changed_during_repair", False)
        )
        print(
            "  Repair: "
            f"{repair_attempted}/{len(repair_rows)} attempted · "
            f"{certified_after_any_retry} certified after any retry "
            f"({certified_after_semantic_repair} semantic, "
            f"{certified_after_nonsemantic_retry} nonsemantic) · "
            f"{answer_changed} answer changed · "
            f"{answer_change_unknown} answer change unknown · "
            f"{solver_solution_changed} solver solution text changed"
        )
    if have_key:
        correct = sum(1 for r in results if r.get("correct"))
        pct_c = (100 * correct / total) if total else 0
        print(f"  Correct (vs key): {correct}/{total} ({pct_c:.1f}%)")
    else:
        print("  (custom run — no reference answer to check against)")
