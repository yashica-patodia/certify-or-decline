#!/usr/bin/env python3
"""Compose a cross-family judge condition from two frozen per-model configs.

Circularity is the deepest threat to the verifier's "certified" label: when the
solver and the judges are the same model family, "certified" can degrade into
"the family approves of its own reasoning" -- the self-recognition / self-
preference bias documented by Panickssery et al. (NeurIPS 2024, arXiv:2404.13076),
strongest exactly when judge and solver share weights.

This tool builds the control: the model under test still solves, interprets and
formalizes, but the roles that DECIDE certification run on an independent model
family. Run the study in two conditions and compare:

  * intra-family  -- every role on model X (the plain matrix run).
  * cross-family  -- solver/interpreter/formalizer on X; the judge roles on
                     a different-family model Y (this config).

The intra-vs-cross delta in certified precision / false-certification isolates
the judge-family effect, because everything up to the certification decision
(solve -> interpret -> formalize) is held on X in *both* conditions.

Inputs are two configs from experiments/make_model_config.py, generated with
DIFFERENT --endpoint values (two vLLM servers, one per model). See
experiments/CROSS_FAMILY_JUDGES.md for the full protocol (grader family, serving,
analysis).

Style mirrors experiments/make_role_ablation_config.py (fail-closed compose +
sealed provenance).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path

import yaml

# The roles that ACCEPT/REJECT proof steps or adjudicate rejections -- i.e. that
# decide certification. Kept on the independent family in the cross-family
# condition. solver / interpreter / formalizer deliberately stay on the model
# under test so the intra-vs-cross comparison changes only the judge family.
DEFAULT_JUDGE_ROLES = (
    "citation",
    "problem_given",
    "computation",
    # Certification-deciding under `_verifier_mode: step_wise_untyped`; leaving it
    # solver-side would put the untyped arm's judge back on the solver family and
    # defeat the cross-family design.
    "step_untyped",
    "initial_state",
    "pedantry",
    "convention_lift",
)

# The full role set a make_model_config output contains (mirrors
# make_model_config.ROLES). Used as a fail-closed allowlist so a stray top-level
# section can't silently ride into the composed condition as a "role".
KNOWN_ROLES = (
    "solver", "interpreter", "formalizer", "citation", "problem_given",
    "computation", "step_untyped", "pedantry", "convention_lift", "initial_state",
)


def _read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError(f"config is not an object: {path}")
    return config, hashlib.sha256(raw).hexdigest()


def _role_names(config: dict) -> list[str]:
    return [
        key for key, value in config.items()
        if not key.startswith("_") and isinstance(value, dict)
    ]


def _family(config: dict) -> str | None:
    """Best-effort model family from the config's _experiment_model.hf_id
    (e.g. 'Qwen/Qwen3-32B' -> 'qwen', 'meta-llama/...' -> 'meta-llama').

    BEST-EFFORT ONLY: this is the HF org prefix, which is false-negative for a
    same-family model republished under a different org (e.g. 'unsloth/Qwen3-...',
    quantized / GGUF re-uploads, distillations). The guardrail below fails closed
    (requires *confirmed-different* families), but a paper claim of "different
    families" must be verified independently (base-model lineage / fingerprint);
    the full hf_id of both models is sealed in the provenance for that.
    """
    hf_id = ((config.get("_experiment_model") or {}).get("hf_id") or "").strip()
    if not hf_id:
        return None
    return (hf_id.split("/", 1)[0] if "/" in hf_id else hf_id).lower()


def _endpoint(config: dict) -> str | None:
    """The model's serving endpoint (from _experiment_model.server.endpoint)."""
    server = (config.get("_experiment_model") or {}).get("server") or {}
    return (server.get("endpoint") or "").strip() or None


def compose(
    solver_path: Path,
    judge_path: Path,
    *,
    judge_roles: tuple[str, ...] = DEFAULT_JUDGE_ROLES,
    allow_same_family: bool = False,
) -> dict:
    solver, solver_sha = _read(solver_path)
    judge, judge_sha = _read(judge_path)
    judge_roles = tuple(dict.fromkeys(judge_roles))  # dedup, order-preserving

    # Fail-closed allowlist (mirrors make_role_ablation_config.py): reject any
    # unexpected role section on EITHER input so a stray key can't silently ride
    # into the composed condition as a "role".
    #
    # `_`-prefixed keys are exempt: `_role_names` excludes them and the composed
    # output is built only from role names, so a private key (a `_model: &model`
    # anchor block, `_experiment_model`) is structurally incapable of becoming a
    # role. Rejecting them was a false positive that made every generated
    # experiments/configs/*.yaml -- all of which carry a `_model` anchor --
    # uncomposable.
    for label, cfg in (("solver", solver), ("judge", judge)):
        unexpected = sorted(
            k for k in cfg if not k.startswith("_") and k not in KNOWN_ROLES
        )
        if unexpected:
            raise ValueError(
                f"{label} config has unexpected top-level keys {unexpected}; "
                f"allowed: the {len(KNOWN_ROLES)} known roles + `_`-prefixed keys"
            )
    unknown_judge_roles = [r for r in judge_roles if r not in KNOWN_ROLES]
    if unknown_judge_roles:
        raise ValueError(f"unknown judge roles: {unknown_judge_roles}")

    if not isinstance(solver.get("solver"), dict):
        raise ValueError("solver config has no solver role")
    missing_judge = [r for r in judge_roles if not isinstance(judge.get(r), dict)]
    if missing_judge:
        raise ValueError(f"judge config is missing judge roles: {missing_judge}")
    solver_side = [r for r in _role_names(solver) if r not in judge_roles]
    if "solver" not in solver_side:
        raise ValueError("'solver' must be a solver-side role")

    solver_family, judge_family = _family(solver), _family(judge)
    confirmed_different = bool(
        solver_family and judge_family and solver_family != judge_family
    )
    same_family = bool(
        solver_family and judge_family and solver_family == judge_family
    )
    # Fail-closed on the experiment's core validity condition: a genuine
    # cross-family control REQUIRES confirmed-different families. Block when they
    # are the same OR cannot be determined (a None/unparseable hf_id would
    # otherwise pass silently as if cross-family -- fail-open). The escape hatch
    # generates the intra-family / unverified baseline through this same path.
    if not allow_same_family and not confirmed_different:
        raise ValueError(
            f"cannot confirm a cross-family control (solver family="
            f"{solver_family!r}, judge family={judge_family!r}: same or "
            "undetermined). Pass allow_same_family=True (--allow-same-family) to "
            "generate the intra-family/unverified baseline. NB family detection "
            "is the best-effort HF org prefix -- verify different families "
            "independently (base-model lineage) for a paper claim."
        )

    output: dict = {}
    for role in solver_side:
        output[role] = copy.deepcopy(solver[role])
    for role in judge_roles:
        output[role] = copy.deepcopy(judge[role])

    solver_endpoint, judge_endpoint = _endpoint(solver), _endpoint(judge)
    output["_experiment_cross_family"] = {
        "schema_version": 1,
        "solver_config": str(solver_path),
        "solver_config_sha256": solver_sha,
        "judge_config": str(judge_path),
        "judge_config_sha256": judge_sha,
        "solver_model": copy.deepcopy(solver.get("_experiment_model")),
        "judge_model": copy.deepcopy(judge.get("_experiment_model")),
        "judge_roles": list(judge_roles),
        "solver_side_roles": solver_side,
        "families": {"solver": solver_family, "judge": judge_family},
        "family_detection": "hf_org_prefix (best-effort; verify lineage for a claim)",
        "confirmed_different_families": confirmed_different,
        # Same family (or undetermined) => this file is really the intra-family /
        # unverified baseline, not a cross-family control.
        "same_family": same_family,
        "endpoints": {"solver": solver_endpoint, "judge": judge_endpoint},
        # Two servers are required; a shared endpoint is a strong smell (though a
        # gateway CAN multiplex distinct served models) -- recorded + warned, not
        # blocked.
        "same_endpoint": bool(
            solver_endpoint and judge_endpoint and solver_endpoint == judge_endpoint
        ),
        "role_assignment": {
            **{role: "solver_config" for role in solver_side},
            **{role: "judge_config" for role in judge_roles},
        },
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,  # keep the bullets
    )
    parser.add_argument("--solver-config", type=Path, required=True,
                        help="per-model config for the model under test")
    parser.add_argument("--judge-config", type=Path, required=True,
                        help="per-model config for the independent judge family")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--judge-roles",
        help="comma-separated override of the roles run on the judge model "
             f"(default: {','.join(DEFAULT_JUDGE_ROLES)})",
    )
    parser.add_argument(
        "--allow-same-family", action="store_true",
        help="generate the file even when the families are the same or cannot be "
             "confirmed different -- produces the intra-family/unverified baseline "
             "through this same code path (default: fail closed).",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    if not args.solver_config.is_file() or not args.judge_config.is_file():
        raise SystemExit("solver and judge configs must both exist")
    judge_roles = (
        tuple(r.strip() for r in args.judge_roles.split(",") if r.strip())
        if args.judge_roles else DEFAULT_JUDGE_ROLES
    )
    try:
        config = compose(
            args.solver_config, args.judge_config, judge_roles=judge_roles,
            allow_same_family=args.allow_same_family,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    prov = config["_experiment_cross_family"]
    if not prov["confirmed_different_families"]:
        print(
            "WARNING: solver and judge families are the same or undetermined "
            f"({prov['families']}); this is the INTRA-family/unverified baseline, "
            "NOT a cross-family control.",
            file=sys.stderr,
        )
    if prov["same_endpoint"]:
        print(
            f"WARNING: solver and judge share endpoint "
            f"{prov['endpoints']['solver']!r}; the protocol expects two servers -- "
            "confirm they serve distinct models.",
            file=sys.stderr,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.out)


if __name__ == "__main__":
    main()
