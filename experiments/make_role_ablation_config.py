#!/usr/bin/env python3
"""Compose a weak-solver/strong-verifier condition from frozen configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import yaml


# Every defaults.yaml role except `solver` -- i.e. everything the verifier
# config supplies in a weak-solver/strong-verifier condition. Pinned against
# defaults.yaml by test_verifier_roles_match_defaults_yaml.
VERIFIER_ROLES = (
    "interpreter",
    "formalizer",
    "citation",
    "problem_given",
    "computation",
    # Deciding judge under `_verifier_mode: step_wise_untyped`; belongs on the
    # verifier side like every other judge.
    "step_untyped",
    "pedantry",
    "convention_lift",
    "initial_state",
)


def _read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError(f"config is not an object: {path}")
    return config, hashlib.sha256(raw).hexdigest()


def compose(solver_path: Path, verifier_path: Path) -> dict:
    solver, solver_sha = _read(solver_path)
    verifier, verifier_sha = _read(verifier_path)
    if not isinstance(solver.get("solver"), dict):
        raise ValueError("solver config has no solver role")
    missing = [role for role in VERIFIER_ROLES if not isinstance(verifier.get(role), dict)]
    if missing:
        raise ValueError(f"verifier config is missing roles: {missing}")

    # Allowlist the verifier's top-level keys (fail closed) so a stray
    # _experiment_diagnostic / _prompt_profile can't silently ride into the
    # composed condition.
    # `_`-prefixed keys are exempt: only names in VERIFIER_ROLES are copied into
    # the composed output, so a `_model: &model` anchor block cannot ride in.
    allowed = set(VERIFIER_ROLES) | {"solver"}
    unexpected = [
        key for key in verifier if not key.startswith("_") and key not in allowed
    ]
    if unexpected:
        raise ValueError(
            f"verifier config has unexpected top-level keys: {unexpected}"
        )
    output = {role: copy.deepcopy(verifier[role]) for role in VERIFIER_ROLES}
    output["solver"] = copy.deepcopy(solver["solver"])
    output["_experiment_role_ablation"] = {
        "schema_version": 1,
        "solver_config": str(solver_path),
        "solver_config_sha256": solver_sha,
        "verifier_config": str(verifier_path),
        "verifier_config_sha256": verifier_sha,
        "solver_model": copy.deepcopy(solver.get("_experiment_model")),
        "verifier_model": copy.deepcopy(verifier.get("_experiment_model")),
        "role_assignment": {
            "solver": "solver_config",
            **{role: "verifier_config" for role in VERIFIER_ROLES},
        },
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-config", type=Path, required=True)
    parser.add_argument("--verifier-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    if not args.solver_config.is_file() or not args.verifier_config.is_file():
        raise SystemExit("solver and verifier configs must both exist")
    config = compose(args.solver_config, args.verifier_config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.out)


if __name__ == "__main__":
    main()
