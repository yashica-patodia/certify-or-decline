#!/usr/bin/env python3
"""Compose an outcome-verifier ablation overlay from frozen configs.

reasoning-eval's certified label comes from STEP-WISE (process) verification. The
canonical ablation a step-wise claim must beat is the OUTCOME verifier (ORM):
the same judge model reading the FULL candidate solution (reasoning included) and
scoring final-answer correctness -- no formalized proof, no step decomposition
(Cobbe 2021 arXiv:2110.14168; Lightman 2023 arXiv:2305.20050; generative
verifiers, GenRM, arXiv:2408.15240). An ORM conditions on the whole CoT; the
answer is NOT stripped from its reasoning.

Three ablation modes (pick with --mode; grade every arm on the SAME committed
solver response via `grade.py --target solver_initial`, per cookbook §13):

  * `answer_score` (DEFAULT, the preregistered headline ORM): one pass, no
    formalization; a SCORED generative verifier reasons then emits a calibrated
    confidence in [0,1] that the candidate's answer is correct. Persisted raw so
    the analysis sweeps a full risk-coverage curve (a boolean point cannot; §13).
  * `holistic_proof` (the process-vs-outcome MIDDLE arm): SAME formalizer + SAME
    repair loop as step-wise, but one global verdict (+ confidence) on the whole
    proof instead of per-step judges; certifies on confidence >= threshold like
    answer_score (its operating point is on the swept curve). step_wise -
    holistic_proof contrasts the per-step APPARATUS as a bundle (decomposition +
    typed judges + pedantry/convention), NOT step decomposition alone;
    holistic_proof - answer_score is a BUNDLED contrast (formalization +
    representation + repair budget), not single-factor. Keeps the base config's
    repair limits (NOT one-pass) so judge granularity is the main step-wise
    difference. See PROCESS_VS_OUTCOME_ABLATION.md.
  * `resolve_agreement`: one pass; a commit-first re-solver (the judge solves
    independently and certifies on answer agreement = verify-by-resolving). An
    honestly-labeled SEPARATE baseline, NOT the ORM -- keep it distinct.

Stack the overlay LAST via --config, over a make_model_config output. The judge
prompts live HERE (not in configs/defaults.yaml) so the ablation roles stay
self-contained and don't enlarge every other config's role set. Use a
different-family --judge-config for the headline (self-preference control); an
intra-model run is a self-preference sensitivity, not the headline.

Style mirrors experiments/make_role_ablation_config.py (fail-closed + sealed
provenance). See experiments/PROCESS_VS_OUTCOME_ABLATION.md for the full protocol.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import yaml

# Roles a make_model_config output contains (allowlist -> fail closed on strays).
KNOWN_ROLES = (
    "solver", "interpreter", "formalizer", "citation", "problem_given",
    "computation", "step_untyped", "pedantry", "convention_lift", "initial_state",
)
# Keys NOT copied from the source role into the judge role: it gets the mode's
# prompt below (not the source solver prompt) and preamble: false.
_DROP_KEYS = ("prompt", "preamble")

# answer_score (OUTCOME/ORM headline): SCORE the candidate; do NOT re-solve.
ANSWER_SCORER_PROMPT = """\
You are an OUTCOME verifier. Given a problem and a candidate solution, judge
whether the candidate's FINAL ANSWER is correct, seeing only the problem and the
candidate's solution text -- NOT a step-by-step proof.

Reason about whether the candidate's final answer is correct (mentally
sanity-check the key steps -- do NOT re-derive the whole solution from scratch),
then output a CALIBRATED confidence in [0,1] that the candidate's final answer is
correct: 1.0 = certainly correct, 0.0 = certainly wrong, 0.5 = a coin flip. Be
calibrated, not extreme -- reserve high confidence for cases you have genuinely
checked, and use low confidence when unsure. Judge CORRECTNESS, not how polished
or plausible the solution looks (a fluent solution can still be wrong).

Return: reasoning, candidate_answer (the extracted final answer, normalized),
correct (your best boolean guess), and confidence (a number in [0,1]).
"""

# resolve_agreement (commit-first re-solver -- a separate baseline, NOT the ORM).
ANSWER_JUDGE_PROMPT = """\
You are a commit-first verifier. Decide whether to CERTIFY a candidate solution's
FINAL ANSWER as correct, seeing only the problem and the candidate's solution
text. High-precision gate: certify only when confident; when in doubt, DECLINE.

Work in this order, and do NOT let the candidate anchor you:
1. SOLVE THE PROBLEM YOURSELF FIRST, from scratch. Write your own final answer as
   `own_answer`. An LLM candidate is often confidently wrong.
2. Extract the candidate's final answer as `candidate_answer` (normalized).
3. CERTIFY (accepted=true) ONLY IF the candidate's final answer matches your own
   AND you are highly confident yours is correct. Otherwise DECLINE.

Return own_answer, candidate_answer, accepted (bool), and a reason.
"""

# holistic_proof (MIDDLE arm): ONE global verdict on the WHOLE formalized proof.
HOLISTIC_JUDGE_PROMPT = """\
You are a holistic proof verifier. You are given a problem and a candidate's FULL
formalized proof: an initial state (state 0, which states the goal) followed by a
sequence of steps, each with a new state, a justification type, and a
justification. Judge the proof AS A WHOLE -- do NOT decompose it into independent
per-step checks.

Reason about whether every step is valid and whether the steps together correctly
establish the goal in state 0. Then output: accepted (true ONLY if the entire
proof is correct) and a calibrated confidence in [0,1] that the proof is fully
correct (1.0 = certainly correct, 0.0 = certainly wrong). Be calibrated, not
extreme. Judge CORRECTNESS, not how detailed or polished the proof looks.
"""

_MODE_ROLE = {
    "answer_score": ("answer_scorer", ANSWER_SCORER_PROMPT),
    "holistic_proof": ("holistic_judge", HOLISTIC_JUDGE_PROMPT),
    "resolve_agreement": ("answer_judge", ANSWER_JUDGE_PROMPT),
}
# One-pass modes skip formalization + the repair loop; holistic_proof keeps the
# base config's repair loop (same machinery as step-wise) so it is NOT here.
_ONE_PASS_MODES = ("answer_score", "resolve_agreement")


def _read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError(f"config is not an object: {path}")
    return config, hashlib.sha256(raw).hexdigest()


def _check_allowed(config: dict, label: str) -> None:
    # `_`-prefixed keys are exempt: they are never treated as roles, so a
    # `_model: &model` anchor block cannot ride in as one. See the same
    # exemption in make_cross_family_config.compose.
    unexpected = sorted(
        k for k in config if not k.startswith("_") and k not in KNOWN_ROLES
    )
    if unexpected:
        raise ValueError(
            f"{label} config has unexpected top-level keys {unexpected}; "
            f"allowed: the {len(KNOWN_ROLES)} known roles + `_`-prefixed keys"
        )


def _judge_model_settings(judge: dict) -> dict:
    """The judge model's serving/sampling settings for the ablation judge role,
    from a representative role (make_model_config sets the same model on every
    role), minus prompt/preamble so this composer's mode prompt wins."""
    source = judge.get("solver")
    if not isinstance(source, dict):
        raise ValueError("judge config has no solver role to source the model from")
    return {k: copy.deepcopy(v) for k, v in source.items() if k not in _DROP_KEYS}


def _model_identity(exp_model: dict | None) -> str | None:
    """A stable model-WEIGHTS identity for the self-preference control -- the hf_id
    (+ revision/quantization when present), deliberately NOT the endpoint/seed, so
    two configs serving the same weights on different endpoints compare equal.
    None when the config omits `_experiment_model` (identity unknown)."""
    if not isinstance(exp_model, dict):
        return None
    ident = exp_model.get("hf_id") or exp_model.get("model") or exp_model.get("id")
    if not ident:
        return None
    extra = [str(exp_model[k]) for k in ("revision", "quantization")
             if exp_model.get(k)]
    return "|".join([str(ident), *extra])


def _same_model(a: dict | None, b: dict | None) -> bool | None:
    ia, ib = _model_identity(a), _model_identity(b)
    if ia is None or ib is None:
        return None
    return ia == ib


def compose(
    model_config_path: Path,
    judge_config_path: Path,
    *,
    mode: str = "answer_score",
    score_threshold: float = 0.5,
    judge_shell: bool = False,
) -> dict:
    if mode not in _MODE_ROLE:
        raise ValueError(f"mode must be one of {sorted(_MODE_ROLE)}; got {mode!r}")
    if not 0.0 <= float(score_threshold) <= 1.0:
        raise ValueError(f"score_threshold must be in [0, 1]; got {score_threshold!r}")
    model, model_sha = _read(model_config_path)
    judge, judge_sha = _read(judge_config_path)
    _check_allowed(model, "model")
    _check_allowed(judge, "judge")
    if not isinstance(model.get("solver"), dict):
        raise ValueError("model config has no solver role")

    role_name, role_prompt = _MODE_ROLE[mode]
    judge_role = {
        **_judge_model_settings(judge),
        "prompt": role_prompt,
        "preamble": False,
    }
    if not judge_shell:
        # DEFAULT = the pure literature ORM: a text-in / score-out judge that CANNOT
        # shell out to recompute the answer (Cobbe/Lightman/GenRM). A shell-using
        # judge that recomputes is a re-solver -- it blurs answer_score into
        # resolve_agreement AND is the "why not just spend more test-time compute?"
        # confound -- so tool access is OFF unless explicitly requested. Pass
        # --judge-shell for the tool-parity co-headline (judge tools matched to the
        # step-wise judges), reported ALONGSIDE the pure ORM, never as the sole
        # headline. NB: gating is honored only on the reasoning_agent backend (llm.py
        # setdefault respects an explicit False) -- the intended open-weight path;
        # claude/codex ignore it.
        judge_role["allow_shell"] = False
    output: dict = {
        "_verifier_mode": mode,
        role_name: judge_role,
    }
    if mode in _ONE_PASS_MODES:
        # One pass: no repair loop (the code path skips it; this also guards if run
        # through a step-wise-shaped driver). holistic_proof deliberately keeps the
        # base config's repair limits so it matches step-wise's machinery.
        output["_limits"] = {
            "max_verify_attempts": 1,
            "max_solver_answers": 1,
            "max_formalizer_invalid_attempts": 1,
        }
    if mode in ("answer_score", "holistic_proof"):
        # Dev-chosen threshold for the sealed verified bit in BOTH scored arms
        # (answer_score + holistic_proof each certify on confidence >= threshold);
        # the analysis sweeps the RAW confidence for the risk-coverage curve anyway.
        output["_score_threshold"] = float(score_threshold)
    output["_experiment_ablation_verifier"] = {
        "schema_version": 1,
        "verifier_mode": mode,
        "model_config": str(model_config_path),
        "model_config_sha256": model_sha,
        "judge_config": str(judge_config_path),
        "judge_config_sha256": judge_sha,
        "solver_model": copy.deepcopy(model.get("_experiment_model")),
        "judge_model": copy.deepcopy(judge.get("_experiment_model")),
        # same_config = byte-identical config FILES (strict provenance). same_model
        # = identical model WEIGHTS (hf_id), which is what the self-preference
        # control actually turns on: two different files can serve the SAME model
        # (different endpoint/seed), so key intra-model vs cross-family off
        # same_model, not byte-equality. Grade both arms via --target solver_initial.
        "same_config": model_sha == judge_sha,
        "same_model": _same_model(model.get("_experiment_model"),
                                  judge.get("_experiment_model")),
        # Seal whether the judge can shell out: the outcome/holistic judges run a
        # tool-using agent loop under reasoning_agent, so this must be auditable +
        # reported per arm (a tool-using judge can recompute -> a budget/validity
        # confound). False by default (pure ORM); True only with --judge-shell.
        "judge_allow_shell": bool(judge_role.get("allow_shell", True)),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-config", type=Path, required=True,
                        help="make_model_config output for the model under test")
    parser.add_argument("--judge-config", type=Path,
                        help="make_model_config output for the judge model "
                             "(default: --model-config = intra-model sensitivity; "
                             "use a different family for the cross-family headline)")
    parser.add_argument("--mode", choices=sorted(_MODE_ROLE), default="answer_score",
                        help="answer_score = scored outcome verifier (headline); "
                             "holistic_proof = whole-proof middle arm; "
                             "resolve_agreement = commit-first re-solver baseline")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="dev-chosen confidence threshold for the scored arms' "
                             "(answer_score / holistic_proof) sealed verified bit "
                             "(analysis sweeps the raw score regardless); in [0,1]")
    parser.add_argument("--judge-shell", action="store_true",
                        help="give the judge shell/tool access (tool-parity "
                             "co-headline, matching the step-wise judges' tools); "
                             "DEFAULT is tools-off -- the pure text-in/score-out ORM")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    judge_config = args.judge_config or args.model_config
    for path in {args.model_config, judge_config}:
        if not path.is_file():
            raise SystemExit(f"config not found: {path}")
    config = compose(args.model_config, judge_config, mode=args.mode,
                     score_threshold=args.score_threshold,
                     judge_shell=args.judge_shell)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.out)


if __name__ == "__main__":
    main()
