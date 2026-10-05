"""Generate typed reasoning steps and check them with LLM judges.

1. Solve: LLM answers the question (session kept open).
2. Formalize: LLM writes the proof (session kept open).
3. Judge: three LLMs verify each justification in parallel.
4. Repair loop on failure: continue formalizer session with the failed
   verdicts; formalizer either fixes the proof or escalates to the
   solver (continuing solver session) for a new answer.

Usage:
    python pipeline.py "What is 2 + 2?"
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml
from jsonschema import ValidationError, validate

import diffing

# ── Models ──────────────────────────────────────────────────────

@dataclass
class Step:
    state: list[str]
    justification_type: str  # citation | problem_given | computation
    justification: str
    # Prior-state element ids this step's justification consumes, e.g.
    # ["F1", "F3"], ["problem"] for problem-given imports, [] for pure
    # additions from evidence. None on proofs from old artifacts that
    # predate the field.
    uses: list[str] | None = None

@dataclass
class Proof:
    initial_state: list[str]
    steps: list[Step]

@dataclass
class Verdict:
    accepted: bool
    reason: str
    # Classified issues from the judge (see ISSUE_SCHEMA). None on
    # verdicts restored from old artifacts that predate the field.
    issues: list[dict] | None = None

# ── JSON Schemas ────────────────────────────────────────────────

# The base proof schema used by general (math/science) formalization.
# Each step has state + justification_type + justification + uses (the
# prior-state element ids the justification consumes — the premise DAG).
PROOF_SCHEMA = {
    "type": "object",
    "properties": {
        "initial_state": {"type": "array", "items": {"type": "string"}},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "state": {"type": "array", "items": {"type": "string"}},
                    "justification_type": {"type": "string", "enum": ["citation", "problem_given", "computation"]},
                    "justification": {"type": "string"},
                    "uses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Ids of the previous-state elements this step's "
                            "justification consumes (e.g. [\"F1\",\"F3\"]); "
                            "[\"problem\"] for problem-given imports; [] for "
                            "pure additions from external evidence."
                        ),
                    },
                },
                # `uses` is required by the formalizer feedback path below
                # (clearer retry prompt than a schema ValidationError) rather
                # than listed here, so backends that omit it once do not burn
                # a max_verify attempt on a cryptic schema failure.
                "required": ["state", "justification_type", "justification"],
            },
        },
    },
    "required": ["initial_state", "steps"],
}

# Structured-output schemas are part of the prompt: their `description` strings are
# sent to the model. PROOF_SCHEMA's `uses` description tells the formalizer to declare
# the premises each step *consumes*, which is the completeness-of-change invariant in
# structural form -- so the `step_wise_no_completeness` ablation has to neutralize it
# here too, or completeness survives the ablation through the schema. Non-ablation
# modes get the module-level object back unchanged, so every historical
# `proof_schema_sha256` / `formalizer_decision_schema_sha256` stamp is preserved.
_ABLATED_USES_DESCRIPTION = (
    "Optional. Ids of previous-state elements this step's justification refers "
    "to (e.g. [\"F1\",\"F3\"]); [\"problem\"] for problem-given imports; [] for "
    "pure additions from external evidence. Any id listed must exist in the "
    "previous state."
)


def _proof_schema(config: dict | None = None) -> dict:
    """PROOF_SCHEMA as sent to the formalizer for the active verifier mode."""
    if verifier_mode(config) != "step_wise_no_completeness":
        return PROOF_SCHEMA
    schema = copy.deepcopy(PROOF_SCHEMA)
    step = schema["properties"]["steps"]["items"]
    step["properties"]["uses"]["description"] = _ABLATED_USES_DESCRIPTION
    return schema


# One classified issue inside a judge verdict. The prose `reason` stays
# the primary record; `issues` adds a machine-readable classification of
# each enumerated point for training-data extraction.
ISSUE_SCHEMA = {
    "type": "object",
    "properties": {
        "error_class": {
            "type": "string",
            "enum": [
                "hidden_premise",
                "fabricated_citation",
                "arithmetic_error",
                "circular_reasoning",
                "misapplied_theorem",
                "cosmetic",
                "other",
            ],
        },
        "severity": {
            "type": "string",
            "enum": ["substantive", "minor", "cosmetic"],
        },
        "description": {"type": "string"},
    },
    "required": ["error_class", "severity", "description"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "accepted": {"type": "boolean"},
        "reason": {"type": "string"},
        "issues": {
            "type": "array",
            "items": ISSUE_SCHEMA,
            "description": (
                "One entry per enumerated problem found with the step, in "
                "the same order as the numbered points in `reason`. Empty "
                "or omitted when the step is accepted."
            ),
        },
    },
    "required": ["accepted", "reason"],
}

# Appended to every judge role prompt so the classification contract
# lives in one place rather than being duplicated per role in YAML.
JUDGE_ISSUE_INSTRUCTIONS = (
    "In addition to the prose reason, fill the `issues` array with one "
    "entry per numbered issue, in the same order: error_class is one of "
    "hidden_premise, fabricated_citation, arithmetic_error, "
    "circular_reasoning, misapplied_theorem, cosmetic, other; severity is "
    "substantive (would change the answer or invalidate the step), minor "
    "(real but repairable in place), or cosmetic (presentation only); "
    "description restates that single issue in one or two sentences. When "
    "you accept the step, return an empty issues array."
)

# Schema for the resolve_agreement baseline (commit-first re-solve-and-compare) --
# NOT the ORM; the scored ORM headline is answer_score below (see
# PROCESS_VS_OUTCOME_ABLATION.md). `own_answer` is the judge's INDEPENDENT solve
# (commit-first, so it doesn't just rubber-stamp a plausible-looking candidate);
# `candidate_answer` is the extracted final answer that becomes the certified `answer`.
ANSWER_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "own_answer": {"type": "string"},
        "candidate_answer": {"type": "string"},
        "accepted": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["own_answer", "candidate_answer", "accepted", "reason"],
}

# Schema for the answer_score OUTCOME verifier (the ORM headline baseline): a
# SCORED generative judgment of the candidate's final answer. `confidence` in
# [0,1] is the judge's SELF-REPORTED P(candidate answer correct) -- NOT assumed
# calibrated (calibration/ECE is validated in analysis); it is the sweepable
# selective signal that yields a risk-coverage curve (a boolean cannot; cookbook
# §13). It SCORES the candidate; it does NOT re-solve-and-match (that is
# resolve_agreement).
ANSWER_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "candidate_answer": {"type": "string"},
        "correct": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["reasoning", "candidate_answer", "correct", "confidence"],
}

# Schema for the holistic_proof verifier (the process-vs-outcome MIDDLE arm): ONE
# global judgment of the WHOLE formalized proof (state 0 + every step), instead of
# the per-step typed judges. `confidence` in [0,1] is the sweepable selective
# signal (as in answer_score). step_wise - holistic_proof isolates the value of
# STEP DECOMPOSITION (same formalizer + same repair loop; only judge granularity
# differs); holistic_proof - answer_score isolates the value of seeing the
# reasoning at all. See PROCESS_VS_OUTCOME_ABLATION.md.
HOLISTIC_PROOF_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "accepted": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["reasoning", "accepted", "confidence"],
}


def _formalizer_decision_schema(config: dict | None = None) -> dict:
    """Wrap the proof schema in the formalizer's action/reject_reason
    envelope: the formalizer either returns a proof or rejects the
    solution with a reason for the solver.

    The schema only requires the discriminator. Requiring both proof and
    reject_reason for every action is stricter than the pipeline needs and
    causes smaller OSS models to reject otherwise valid proof responses.
    """
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["proof", "reject"]},
            "proof": _proof_schema(config),
            "reject_reason": {"type": "string"},
        },
        "required": ["action"],
    }


def _formalizer_decision_feedback(
    decision: dict, *, enforce_completeness: bool = True,
) -> str | None:
    """Return a retry prompt when a schema-valid formalizer decision is
    semantically incomplete.

    The schema deliberately requires only the discriminator so smaller models
    can return {"action": "proof"} without also fabricating a reject_reason.
    That means action-specific fields must be checked in pipeline code.

    `enforce_completeness=False` (the step_wise_no_completeness ablation) skips the
    structural completeness gate -- the requirement that every step DECLARE the
    `uses` premises that license its change -- while keeping every other check
    (dataflow-reference integrity, F# id stability, proof well-formedness).
    """
    action = decision.get("action")
    if action not in {"proof", "reject"}:
        return (
            "Your previous response did not choose a valid action. Reply again "
            "with exactly one JSON decision whose action is either \"proof\" "
            "or \"reject\"."
        )
    if action == "proof" and not isinstance(decision.get("proof"), dict):
        return (
            "Your previous response chose action='proof' but omitted the "
            "required 'proof' object. Reply again with exactly one JSON "
            "decision: either {\"action\":\"proof\",\"proof\": ...} with a "
            "complete proof object, or {\"action\":\"reject\","
            "\"reject_reason\": ...}."
        )
    if action == "proof":
        try:
            validate(decision["proof"], PROOF_SCHEMA)
        except ValidationError as exc:
            return (
                "Your previous response chose action='proof' but the 'proof' "
                f"object did not match the required schema: {exc.message}. "
                "Reply again with exactly one JSON decision containing either "
                "a complete proof object or a reject_reason."
            )
        proof = decision["proof"]
        initial_state = proof.get("initial_state") or []
        steps = proof.get("steps") or []
        if not steps:
            return (
                "Your proof contains no steps, so it does not resolve the "
                "goal. Return a corrected proof with at least one justified "
                "step, or reject the solver answer."
            )
        final_state = steps[-1].get("state") or []
        if not final_state or not str(final_state[0]).strip():
            return (
                "Your proof's final state has no answer in state[0]. Return a "
                "corrected proof whose final state[0] is the resolved answer."
            )
        initial_goal = str(initial_state[0]).strip() if initial_state else ""
        final_answer = str(final_state[0]).strip()
        unresolved_slot = re.fullmatch(
            r"[<\[({]*\s*(?:final\s+)?(?:answer|result)\s*[>\])}]*"
            r"\s*(?:(?:=|:)\s*(?:\?|tbd|todo|pending|unknown|[_-]*)?)?\s*",
            final_answer,
            flags=re.IGNORECASE,
        )
        if (
            final_answer in {"?", "_", "-"}
            or unresolved_slot is not None
            or (initial_goal and final_answer.casefold() == initial_goal.casefold())
        ):
            return (
                "Your proof leaves the answer slot unresolved: final "
                f"state[0] is {final_answer!r}. The final state[0] must contain "
                "the actual resolved answer, not the original goal placeholder."
            )
        # A fact label leaking into the answer slot. The formalizer prompt
        # reserves the "F1: ..." id prefix for "every entry EXCEPT index 0", so
        # an F#-prefixed state[0] means a fact was written where the answer
        # belongs. The answer is then extracted verbatim -- observed live on two
        # models, which CERTIFIED while shipping 'F1: 2 + 2 = 4' and
        # 'F3: Sum = 720' instead of '4' and '720'.
        #
        # This is exactly the paper's "extraction" dispute category: the right
        # value was computed, but the shipped string is not the answer. Caught
        # here rather than at grading time, it becomes a repair-loop retry --
        # the formalizer is told to fix the slot and usually can -- instead of a
        # certified-but-mis-extracted result that has to be credited later.
        # Case-insensitive: the small models this targets are exactly the ones
        # likeliest to lowercase the label, and 'f3: Sum = 720' corrupts the
        # answer slot just as thoroughly as 'F3: ...'. Still requires digits and
        # a colon, so ordinary answers like 'f = ma' or 'f(x) = 2x' are
        # unaffected.
        if re.match(r"^F\d+\s*:", final_answer, re.IGNORECASE):
            return (
                f"Your proof's final state[0] is {final_answer!r}, which carries "
                "an F# fact label. state[0] is the ANSWER slot, not a fact: F# "
                "ids belong only on entries at index 1 and beyond. Return a "
                "corrected proof whose final state[0] is the bare resolved "
                "answer, with the fact kept as its own later entry."
            )
        # Premise-reference invariants (the DAG layer): every step must
        # declare `uses`, every referenced id must exist in the previous
        # state, and F# ids must stay stable. Enforced here (not via
        # PROOF_SCHEMA.required) so the retry prompt can name the field
        # and the missing steps explicitly.
        # The `uses`-declared check IS the structural half of completeness-of-change
        # ("every change must name the premises that license it"), so the
        # step_wise_no_completeness ablation skips it; the dataflow-reference and
        # id-stability checks below are typing/diff integrity and always run.
        if enforce_completeness:
            missing_uses = [
                index for index, step in enumerate(steps, start=1)
                if "uses" not in (step or {})
            ]
            if missing_uses:
                listed = ", ".join(str(i) for i in missing_uses)
                return (
                    "Your proof's steps are missing the required 'uses' field "
                    f"(steps {listed}). Every step must list the previous-state "
                    "element ids its justification consumes (use \"problem\" for "
                    "problem-given imports and [] for pure additions from "
                    "evidence). Reply again with exactly one JSON decision "
                    "containing a corrected proof."
                )
        reference_errors = diffing.validate_proof_references(proof)
        if reference_errors:
            details = "\n".join(f"- {e}" for e in reference_errors)
            # This retry text is itself a prompt, so it restates the rules the
            # formalizer must satisfy -- including, in the enforcing modes, the
            # completeness one ("lists exactly ... consumes"). Under the ablation that
            # sentence would re-inject completeness mid-run through the repair loop,
            # so the ablated wording asks only for referential integrity. The F# id
            # rules are diff/typing integrity and are stated in both.
            if enforce_completeness:
                uses_rule = (
                    "every step's 'uses' lists exactly the previous-"
                    "state element ids its justification consumes (use "
                    "\"problem\" for problem-given imports and [] for pure "
                    "additions from evidence); "
                )
            else:
                uses_rule = (
                    "any id a step's 'uses' lists must exist in the previous "
                    "state (use \"problem\" for problem-given imports); "
                )
            return (
                "Your proof violates the premise-reference rules:\n"
                f"{details}\n\n"
                f"Rules: {uses_rule}F# ids are permanent — never "
                "renumber an existing fact, and give each new fact the "
                "next unused F# id. Reply again with exactly one JSON "
                "decision containing a corrected proof."
            )
    if action == "reject":
        reason = decision.get("reject_reason")
        if not isinstance(reason, str) or not reason.strip():
            return (
                "Your previous response chose action='reject' but omitted the "
                "required non-empty 'reject_reason'. Reply again with exactly "
                "one JSON decision: either {\"action\":\"reject\","
                "\"reject_reason\": ...}, or {\"action\":\"proof\","
                "\"proof\": ...} with a complete proof object."
            )
    return None

# The pedantry filter's decision on a single failed verdict. The index
# arrays refer to the judge's classified issues by 1-based position (the
# same numbering shown in the pedantry prompt); they are optional so the
# filter still works on verdicts that carry no classified issues.
PEDANTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "is_pedantic": {"type": "boolean"},
        "reason": {"type": "string"},
        "pedantic_issue_indices": {
            "type": "array", "items": {"type": "integer"},
        },
        "substantive_issue_indices": {
            "type": "array", "items": {"type": "integer"},
        },
    },
    "required": ["is_pedantic", "reason"],
}

# The convention-lift judge's decision on a legitimate rejection.
# Runs after pedantry; if it lifts, the rejected step becomes accepted
# with the added convention recorded as an explicit assumption.
CONVENTION_LIFT_SCHEMA = {
    "type": "object",
    "properties": {
        "can_lift": {"type": "boolean"},
        "convention": {"type": "string"},
        "source": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["can_lift", "convention", "source", "reasoning"],
}


# Completeness-of-change text in the role prompts (configs/defaults.yaml) is wrapped
# in one of two complementary marker pairs, resolved here at the single prompt choke
# point (no prompt duplication anywhere, so the arms can never drift apart):
#
#   [[COMPLETENESS]] ... [[/COMPLETENESS]]
#       kept verbatim when completeness is ON (every normal mode -- only the markers
#       are removed, so those prompts are byte-identical to the pre-marker text);
#       dropped entirely for `step_wise_no_completeness`.
#   [[WITHOUT_COMPLETENESS]] ... [[/WITHOUT_COMPLETENESS]]
#       the mirror image: dropped when completeness is ON, kept for the ablation.
#
# The second pair exists because a few sites cannot simply LOSE their completeness
# clause without the surrounding prompt becoming incoherent (e.g. "Two checks:"
# followed by one check) or without silently retaining completeness through an
# unmarked restatement (e.g. the formalizer's `uses` spec, which demands the step
# declare *exactly* the premises it consumes -- that IS completeness-of-change in
# structural form). At those sites the ablation substitutes a neutral, same-role
# rewrite instead of deleting text. Deleting there would confound the experiment in
# BOTH directions: a degraded/incoherent prompt is an alternative explanation for any
# measured drop, and a surviving unmarked restatement shrinks the true effect.
#
# Each family needs two passes, because a marked span is either a whole paragraph or
# a fragment inside a sentence, and the two want different whitespace handling:
#   BLOCK  -- the span occupies entire lines. Drop the lines and one following blank
#             line, so the surrounding paragraph break stays a single blank line.
#   INLINE -- the span sits inside a line ("...correctly applied[[C]], AND ...[[/C]]").
#             Drop exactly the span; touching the newline would reflow the paragraph.
# A single regex with an optional trailing "\n?" cannot do both: it silently eats the
# newline of an inline span that happens to end a line, which reflows the prompt in
# whichever arm did the stripping -- a whitespace-only difference between arms that is
# still a difference the model sees.
def _span_patterns(tag: str) -> tuple[re.Pattern, re.Pattern]:
    open_tag, close_tag = rf"\[\[{tag}\]\]", rf"\[\[/{tag}\]\]"
    # The body is "anything up to, but not across, this family's closing tag". A plain
    # non-greedy `.*?` is not enough for the BLOCK pattern: when the nearest closing
    # tag is not followed by a newline (an inline span that happens to open a line,
    # e.g. "[[C]]Two checks:[[/C]][[W]]One check:[[/W]]"), `.*?` would keep expanding
    # to the next closing tag that IS at end-of-line and delete everything in between.
    body = rf"(?:(?!{close_tag}).)*"
    return (
        re.compile(
            rf"(?m)^[ \t]*{open_tag}{body}{close_tag}[ \t]*\n(?:[ \t]*\n)?", re.DOTALL
        ),
        re.compile(rf"{open_tag}{body}{close_tag}", re.DOTALL),
    )


_COMPLETENESS_SPANS = _span_patterns("COMPLETENESS")
_WITHOUT_COMPLETENESS_SPANS = _span_patterns("WITHOUT_COMPLETENESS")
_COMPLETENESS_MARKERS = re.compile(r"\[\[/?(?:WITHOUT_)?COMPLETENESS\]\]")
# Ordered marker scan used to reject malformed markup. Without this check a stray or
# mis-ordered marker would make the span regex fail to match, and the marker-stripping
# pass would then silently delete the marker alone -- leaving the completeness clause
# fully intact in a run labelled "no completeness". That is a silent confound, so it
# must be a hard error rather than a quiet no-op.
_ANY_MARKER = re.compile(r"\[\[(/?)(WITHOUT_)?COMPLETENESS\]\]")


def _validate_completeness_markers(text: str, role: str) -> None:
    """Raise when completeness markup is malformed.

    Well-formed means: markers appear in strictly alternating open/close pairs of the
    same family, never nested, never interleaved across families, and every open tag
    is closed. Anything else silently changes what the ablation removes.
    """
    open_family: str | None = None
    for match in _ANY_MARKER.finditer(text):
        closing, family = match.group(1) == "/", match.group(2) or ""
        if closing:
            if open_family is None:
                raise ValueError(
                    f"{role} prompt: closing [[/{family}COMPLETENESS]] with no "
                    "matching opening marker"
                )
            if open_family != family:
                raise ValueError(
                    f"{role} prompt: [[{open_family}COMPLETENESS]] closed by "
                    f"[[/{family}COMPLETENESS]] -- marker families must not interleave"
                )
            open_family = None
        else:
            if open_family is not None:
                raise ValueError(
                    f"{role} prompt: [[{family}COMPLETENESS]] opened inside an "
                    f"unclosed [[{open_family}COMPLETENESS]] span -- no nesting"
                )
            open_family = family
    if open_family is not None:
        raise ValueError(
            f"{role} prompt: unclosed [[{open_family}COMPLETENESS]] span"
        )


def agent_prompt(role: str, config: dict | None = None) -> str:
    """Return the system prompt for a role, optionally prefixed with
    the shared `_preamble` block when the role sets `preamble: true`.
    Used to factor out environment descriptions or other content that
    should appear at the top of several roles' prompts without
    duplicating the text in each role.

    This is the single place role prompts are rendered, so it is also where the
    completeness markup is resolved (see above) -- meaning the prompt this returns is
    exactly what the model is sent. `config` defaults to the global CONFIG; pass one
    explicitly to render a role's effective prompt for an arbitrary config (the
    reproducibility stamp in harness.compute_config_hash does this).
    """
    cfg = CONFIG if config is None else config
    settings = cfg.get(role, {})
    p = settings.get("prompt", "")
    _validate_completeness_markers(p, role)
    if verifier_mode(cfg) == "step_wise_no_completeness":
        drop_block, drop_inline = _COMPLETENESS_SPANS
    else:
        drop_block, drop_inline = _WITHOUT_COMPLETENESS_SPANS
    p = drop_inline.sub("", drop_block.sub("", p))
    p = _COMPLETENESS_MARKERS.sub("", p)
    if settings.get("preamble"):
        pre = cfg.get("_preamble", "")
        if pre:
            p = f"{pre}\n\n{p}"
    return p

# ── Config ─────────────────────────────────────────────────────

DEFAULTS_PATH = Path(__file__).parent / "configs" / "defaults.yaml"

# Fields whose ${ENV} references are expanded at load time, so secrets (Azure
# subscription/resource-group/account and endpoints) stay in the environment
# rather than committed configs. Restricted to connection/identity fields so
# prompt text containing a literal `$` is never rewritten.
_ENV_EXPANDED_FIELDS = (
    "endpoint",
    "base_url",
    "azure_subscription_id",
    "azure_resource_group",
    "azure_account",
)


def load_config(override_paths=None) -> dict:
    """Load defaults.yaml and stack zero or more overrides on top.

    `override_paths` accepts None, a single path (str/Path), or a list
    of paths. When a list, overrides are applied in order — later
    entries win on conflicts. Role entries are dict-merged into
    defaults; top-level scalars like `_preamble` are replaced wholesale
    (can't .update() a string).
    """
    with open(DEFAULTS_PATH) as f:
        config = yaml.safe_load(f)

    if override_paths is None:
        paths = []
    elif isinstance(override_paths, (str, Path)):
        paths = [override_paths]
    else:
        paths = list(override_paths)

    for path in paths:
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        for key, value in overrides.items():
            existing = config.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                existing.update(value)
            else:
                config[key] = value
    for value in config.values():
        if isinstance(value, dict):
            for field in _ENV_EXPANDED_FIELDS:
                if isinstance(value.get(field), str):
                    value[field] = os.path.expandvars(value[field])
    return config

CONFIG = {}  # set in main

def agent_settings(role: str) -> dict:
    return CONFIG.get(role, {})

# ── LLM ────────────────────────────────────────────────────────

from llm import llm as _llm_call

WATCH = False  # set in main

async def llm(
    prompt: str,
    *,
    role: str = "solver",
    schema: dict | None = None,
    system: str | None = None,
    resume: str | None = None,
) -> tuple[str | dict, str | None]:
    return await _llm_call(
        prompt, role=role, schema=schema, system=system,
        config=CONFIG, watch=WATCH, resume=resume,
    )

# ── Judges ──────────────────────────────────────────────────────

def _format_proof(proof: Proof) -> str:
    lines = [f"State 0: {proof.initial_state}"]
    for i, step in enumerate(proof.steps):
        uses = f" (uses: {step.uses})" if step.uses is not None else ""
        lines.append(
            f"State {i+1}: {step.state}  "
            f"[{step.justification_type}]{uses} {step.justification}"
        )
    return "\n".join(lines)


def _normalized_issues(data: dict) -> list[dict] | None:
    """Read a verdict's optional `issues` array defensively; None when
    the model omitted it (distinguishable from an explicit empty list)."""
    raw = data.get("issues")
    if not isinstance(raw, list):
        return None
    issues = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        issues.append({
            "error_class": str(item.get("error_class") or "other"),
            "severity": str(item.get("severity") or "substantive"),
            "description": str(item.get("description") or ""),
        })
    return issues


def _format_issue_list(issues: list[dict] | None) -> str:
    """Render classified issues as a 1-based numbered list for prompts
    that need to reference issue indices."""
    if not issues:
        return "(no classified issues were provided)"
    return "\n".join(
        f"{i + 1}. [{issue.get('error_class', 'other')} / "
        f"{issue.get('severity', 'substantive')}] "
        f"{issue.get('description', '')}"
        for i, issue in enumerate(issues)
    )


async def judge(
    step: Step, prev_state: list[str], problem: str, proof: Proof, step_number: int,
    on_complete=None,
) -> tuple[Verdict, dict]:
    # Normally the step's own justification_type selects a type-specialized
    # judge. The `step_wise_untyped` ablation routes every step to one generic
    # judge instead, isolating typed-judge specialization; the type is still
    # rendered into `user_msg` and `_format_proof` below, so only the prompt
    # changes. See VERIFIER_MODES and PROCESS_VS_OUTCOME_ABLATION.md.
    role = (
        UNTYPED_JUDGE_ROLE
        if verifier_mode() == "step_wise_untyped"
        else step.justification_type  # citation | problem_given | computation
    )
    prompt_template = agent_prompt(role)

    # Variables available to all judge prompts
    variables = {
        "step_number": str(step_number),
        "prev_state": str(prev_state),
        "new_state": str(step.state),
        "justification": step.justification,
        "justification_type": step.justification_type,
        "problem": problem,
        "proof": _format_proof(proof),
    }

    # Format the prompt template with variables. If a template references an
    # undefined variable we crash loudly — silently falling back to the raw
    # template would send the judge a broken prompt with literal `{step_number}`
    # placeholders and produce garbage verdicts. Fail fast on this class of bug.
    try:
        system = prompt_template.format(**variables)
    except KeyError as e:
        raise RuntimeError(
            f"Prompt template for role {role!r} references undefined variable {e}. "
            f"Available variables: {sorted(variables.keys())}. "
            f"Check configs/defaults.yaml for a stray {{...}} placeholder."
        )

    # User message is always the full context
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Full proof:\n{_format_proof(proof)}\n\n"
        f"Step {step_number} being judged:\n"
        f"  Previous state: {prev_state}\n"
        f"  New state: {step.state}\n"
        f"  Justification type: {step.justification_type}\n"
        f"  Justification: {step.justification}"
    )

    system = f"{system}\n\n{JUDGE_ISSUE_INSTRUCTIONS}"
    data, _ = await llm(user_msg, role=role, schema=VERDICT_SCHEMA, system=system)
    # llm() enforces VERDICT_SCHEMA (accepted: bool, reason: str, both required),
    # so read strictly and only tolerate extra keys the model may add.
    verdict = Verdict(
        accepted=data["accepted"],
        reason=data.get("reason", ""),
        issues=_normalized_issues(data),
    )
    inputs = {"role": role, "user_msg": user_msg, "system": system}
    if on_complete is not None:
        await on_complete(step_number, verdict, inputs)
    return verdict, inputs


async def judge_initial_state(
    proof: Proof, problem: str, on_complete=None,
) -> tuple[Verdict, dict]:
    """Audit state 0 — the proof's initial state, which never gets a
    per-step justification. Catches two failure modes: (1) the
    formalizer drifting off the goal-in-state[0] convention and
    filling state 0 with definitions instead of a goal (p115 pattern),
    and (2) content smuggled into state 0 that isn't supported by the
    problem text and would otherwise flow through the proof unchecked.

    Runs in parallel with the per-step judges from `_judge_proof`.
    """
    prompt_template = agent_prompt("initial_state")
    variables = {
        "initial_state": str(proof.initial_state),
        "problem": problem,
    }
    try:
        system = prompt_template.format(**variables)
    except KeyError as e:
        raise RuntimeError(
            f"Prompt template for role 'initial_state' references undefined "
            f"variable {e}. Available: {sorted(variables.keys())}."
        )
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Initial state (state 0): {proof.initial_state}\n\n"
        "Audit only this initial state against the problem text. Later proof "
        "states are intentionally omitted because they are outside this "
        "judge's scope."
    )
    system = f"{system}\n\n{JUDGE_ISSUE_INSTRUCTIONS}"
    data, _ = await llm(
        user_msg, role="initial_state", schema=VERDICT_SCHEMA, system=system,
    )
    # llm() enforces VERDICT_SCHEMA (accepted: bool, reason: str, both required),
    # so read strictly and only tolerate extra keys the model may add.
    verdict = Verdict(
        accepted=data["accepted"],
        reason=data.get("reason", ""),
        issues=_normalized_issues(data),
    )
    inputs = {"role": "initial_state", "user_msg": user_msg, "system": system}
    if on_complete is not None:
        await on_complete(0, verdict, inputs)
    return verdict, inputs


async def judge_answer(problem: str, solution: str, on_complete=None):
    """resolve_agreement baseline -- commit-first re-solve-and-compare (NOT the ORM;
    the scored ORM headline is judge_answer_score).

    A single COMMIT-FIRST judgment of the candidate's final answer: the judge solves
    the problem independently first, then certifies only if the candidate's answer
    matches its own and it is confident (else declines). Commit-first is what stops a
    gold-free judge from rubber-stamping a fluent but wrong solution (plausibility
    reward-hacking). Mirrors judge_initial_state but sees the solver's answer instead
    of a decomposed proof.

    Returns (verdict, candidate_answer, inputs). `candidate_answer` is the judge's
    extraction of the candidate's final answer, which becomes the certified answer.
    """
    system = agent_prompt("answer_judge")
    user_msg = (
        f"Problem:\n{problem}\n\n"
        f"Candidate solution (from the solver under test):\n{solution}\n\n"
        "Follow your instructions in order: solve it yourself FIRST, then extract "
        "the candidate's final answer, then decide certify/decline."
    )
    data, _ = await llm(
        user_msg, role="answer_judge", schema=ANSWER_JUDGE_SCHEMA, system=system,
    )
    verdict = Verdict(accepted=bool(data.get("accepted")), reason=data.get("reason", ""))
    candidate_answer = data.get("candidate_answer", "")
    inputs = {
        "role": "answer_judge", "user_msg": user_msg, "system": system,
        "own_answer": data.get("own_answer", ""),
        "candidate_answer": candidate_answer,
    }
    if on_complete is not None:
        await on_complete(0, verdict, inputs)
    return verdict, candidate_answer, inputs


async def judge_answer_score(problem: str, solution: str, on_complete=None):
    """Outcome (answer_score) verifier -- the ORM headline baseline.

    A SCORED generative judgment: the judge reasons about whether the candidate's
    final answer is correct and emits a self-reported `confidence` in [0,1] (its own
    P(answer correct) -- NOT assumed calibrated; calibration/ECE is validated in
    analysis). It SCORES the candidate (it does NOT re-solve-and-match -- that is
    resolve_agreement). The raw confidence is the selective signal, thresholded
    post-hoc into a risk-coverage curve (cookbook §13). Returns (confidence,
    believes_correct, candidate_answer, inputs).
    """
    system = agent_prompt("answer_scorer")
    user_msg = (
        f"Problem:\n{problem}\n\n"
        f"Candidate solution (from the solver under test):\n{solution}\n\n"
        "Reason about whether the candidate's FINAL ANSWER is correct, then output "
        "a calibrated confidence in [0,1] that it is correct."
    )
    data, _ = await llm(
        user_msg, role="answer_scorer", schema=ANSWER_SCORE_SCHEMA, system=system,
    )
    try:
        confidence_raw = float(data.get("confidence"))
    except (TypeError, ValueError):
        confidence_raw = None
    # Clamp to [0,1] for the certified decision, but ALSO keep the pre-clamp value:
    # an out-of-range score is a miscalibration SIGNAL the ECE/reliability analysis
    # needs, so sealing only the clamped value would silently repair it away.
    confidence = 0.0 if confidence_raw is None else max(0.0, min(1.0, confidence_raw))
    candidate_answer = data.get("candidate_answer", "")
    # `correct` is the judge's own boolean guess -- an AUXILIARY diagnostic (fed to
    # the calibration / verdict-vs-confidence agreement analysis). It does NOT gate
    # certification: the raw `confidence` is the sole sweepable selective signal
    # (verified = confidence >= threshold), so the sealed bit stays reconstructible
    # from the swept risk-coverage curve. A `correct=False, confidence=0.95` row is a
    # measurable judge-incoherence signal, not a certification bug.
    believes_correct = bool(data.get("correct"))
    inputs = {
        "role": "answer_scorer", "user_msg": user_msg, "system": system,
        "reasoning": data.get("reasoning", ""), "confidence": confidence,
        "confidence_raw": confidence_raw,
        "candidate_answer": candidate_answer,
    }
    if on_complete is not None:
        await on_complete(
            0, Verdict(accepted=believes_correct, reason=data.get("reasoning", "")),
            inputs,
        )
    return confidence, believes_correct, candidate_answer, inputs


async def judge_proof_holistic(proof: Proof, problem: str, on_complete=None):
    """Holistic (whole-proof) verifier -- the process-vs-outcome MIDDLE arm.

    Sees the SAME formalized proof as step-wise (state 0 + every step) but renders
    ONE global accept/reject verdict plus a self-reported `confidence` in [0,1],
    instead of the per-step typed judges + pedantry + convention passes. Shares
    step-wise's formalizer and repair loop, so step_wise - holistic_proof isolates
    the per-step APPARATUS (decomposition + typed judges + pedantry/convention
    passes) as a bundle -- not step decomposition alone (see
    PROCESS_VS_OUTCOME_ABLATION.md). Certification mirrors answer_score:
    verified = confidence >= threshold, and the raw confidence is sealed + swept
    into a risk-coverage curve (cookbook §13). Returns (confidence, verdict, inputs).
    """
    system = agent_prompt("holistic_judge")
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Full candidate proof (state 0 + each step's new state, justification "
        f"type, and justification):\n{_format_proof(proof)}\n\n"
        "Judge the proof AS A WHOLE: is every step valid and does it correctly "
        "establish the goal in state 0? Output accepted (bool) and a calibrated "
        "confidence in [0,1] that the proof is fully correct."
    )
    data, _ = await llm(
        user_msg, role="holistic_judge", schema=HOLISTIC_PROOF_SCHEMA, system=system,
    )
    verdict = Verdict(accepted=bool(data.get("accepted")), reason=data.get("reasoning", ""))
    try:
        confidence_raw = float(data.get("confidence"))
    except (TypeError, ValueError):
        confidence_raw = None
    # Keep the pre-clamp value too (miscalibration signal for the ECE analysis).
    confidence = 0.0 if confidence_raw is None else max(0.0, min(1.0, confidence_raw))
    inputs = {"role": "holistic_judge", "user_msg": user_msg, "system": system,
              "confidence": confidence, "confidence_raw": confidence_raw}
    if on_complete is not None:
        await on_complete(0, verdict, inputs)
    return confidence, verdict, inputs


# ── Pipeline ────────────────────────────────────────────────────

def _proof_from_dict(data: dict) -> Proof:
    # Tolerate extra keys the model may add on a step (the claude backend leaves the
    # step schema's additionalProperties unset, so a stray key like "confidence"
    # passes validation) -- mirror the deliberately extra-key-tolerant Verdict read
    # rather than letting Step(**s) raise TypeError and drop the whole result.
    step_fields = set(Step.__dataclass_fields__)
    return Proof(
        initial_state=data["initial_state"],
        steps=[Step(**{k: v for k, v in s.items() if k in step_fields})
               for s in data["steps"]],
    )


def _proof_to_dict(proof: Proof) -> dict:
    """Serialize a Proof back to JSON shape. `uses` is emitted only when
    present so proofs restored from old artifacts round-trip unchanged."""
    return {
        "initial_state": proof.initial_state,
        "steps": [
            {
                "state": s.state,
                "justification_type": s.justification_type,
                "justification": s.justification,
                **({"uses": s.uses} if s.uses is not None else {}),
            }
            for s in proof.steps
        ],
    }


def _format_failed_verdicts(
    proof: Proof,
    verdicts: list[Verdict],
    state0_verdict: Verdict | None = None,
) -> str:
    lines = []
    if state0_verdict is not None and not state0_verdict.accepted:
        lines.append(
            f"State 0 (initial_state) FAILED: {proof.initial_state}\n"
            f"  Reason: {state0_verdict.reason}"
        )
    for i, (step, v) in enumerate(zip(proof.steps, verdicts)):
        if not v.accepted:
            lines.append(
                f"Step {i+1} [{step.justification_type}] FAILED: {step.justification}\n"
                f"  Reason: {v.reason}"
            )
    return "\n\n".join(lines)


async def _judge_proof(
    proof: Proof, problem: str, on_each=None,
) -> tuple[list[Verdict], list[dict]]:
    prev_states = [proof.initial_state] + [s.state for s in proof.steps[:-1]]
    judge_results = await asyncio.gather(*[
        judge(step, prev, problem, proof, i + 1, on_complete=on_each)
        for i, (step, prev) in enumerate(zip(proof.steps, prev_states))
    ])
    verdicts = [jr[0] for jr in judge_results]
    inputs = [jr[1] for jr in judge_results]
    return verdicts, inputs


_PEDANTRY_ISSUE_INSTRUCTIONS = (
    "Also classify each numbered issue individually: put the 1-based "
    "numbers of issues that are pedantic into pedantic_issue_indices and "
    "the numbers of issues that point at real errors into "
    "substantive_issue_indices. Every listed issue number must appear in "
    "exactly one of the two arrays."
)


def _pedantry_issue_split(data: dict, issues: list[dict] | None) -> dict:
    """Extract the pedantry filter's per-issue ruling, keeping only
    indices that actually reference a classified issue."""
    valid = range(1, len(issues or []) + 1)

    def clean(key: str) -> list[int]:
        raw = data.get(key)
        if not isinstance(raw, list):
            return []
        return sorted({
            int(i) for i in raw
            if isinstance(i, (int, float))
            and not isinstance(i, bool)
            and int(i) in valid
        })

    return {
        "pedantic_issue_indices": clean("pedantic_issue_indices"),
        "substantive_issue_indices": clean("substantive_issue_indices"),
    }


async def _pedantry_check(
    verdict, step, prev_state, step_number, problem, proof,
    original_reason=None, on_complete=None,
):
    """Ask the pedantry filter whether this rejection is legitimate or pedantic."""
    issue_block = ""
    if verdict.issues:
        issue_block = (
            "\n\nThe judge classified these issues:\n"
            f"{_format_issue_list(verdict.issues)}\n\n"
            f"{_PEDANTRY_ISSUE_INSTRUCTIONS}"
        )
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Full proof:\n{_format_proof(proof)}\n\n"
        f"Step {step_number} being evaluated:\n"
        f"  Previous state: {prev_state}\n"
        f"  New state: {step.state}\n"
        f"  Justification type: {step.justification_type}\n"
        f"  Justification: {step.justification}\n\n"
        f"A judge rejected this step with the following reason:\n{verdict.reason}\n\n"
        f"Is this rejection legitimate (the proof is actually wrong) or "
        f"pedantic (the proof is correct, the judge is being too strict)?"
        f"{issue_block}"
    )
    data, _ = await llm(
        user_msg, role="pedantry", schema=PEDANTRY_SCHEMA,
        system=agent_prompt("pedantry"),
    )
    is_pedantic = data["is_pedantic"]
    reason = data["reason"]
    issue_split = _pedantry_issue_split(data, verdict.issues)
    if on_complete is not None:
        await on_complete(
            step_number, original_reason, is_pedantic, reason, issue_split,
        )
    return is_pedantic, reason, issue_split


async def _convention_lift_step(
    verdict, step, prev_state, step_number, problem, proof,
    pedantry_reason, on_complete=None,
):
    """Ask the convention_lift judge whether a legitimately-rejected
    step can be accepted by invoking a standard, citable, domain-level
    convention. If yes, the step becomes accepted-under-assumption.

    Runs only on per-step verdicts that pedantry already confirmed are
    legitimate. Same parallel-gather shape as _pedantry_check.
    """
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Full proof:\n{_format_proof(proof)}\n\n"
        f"Step {step_number} being evaluated:\n"
        f"  Previous state: {prev_state}\n"
        f"  New state: {step.state}\n"
        f"  Justification type: {step.justification_type}\n"
        f"  Justification: {step.justification}\n\n"
        f"Judge's rejection reason:\n{verdict.reason}\n\n"
        f"Pedantry already confirmed this rejection is legitimate:\n"
        f"{pedantry_reason}\n\n"
        f"Is there a single standard, citable, domain-wide convention "
        f"that — if added as an explicit premise — would fully justify "
        f"this step and resolve the judge's objection?"
    )
    data, _ = await llm(
        user_msg, role="convention_lift", schema=CONVENTION_LIFT_SCHEMA,
        system=agent_prompt("convention_lift"),
    )
    if on_complete is not None:
        await on_complete(step_number, data)
    return data


async def _convention_lift_state0(
    verdict, initial_state, problem, pedantry_reason, on_complete=None,
):
    """Convention_lift applied to state 0 (the initial state)."""
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Initial state (state 0): {initial_state}\n\n"
        f"The initial_state judge rejected this state with the following "
        f"reason:\n{verdict.reason}\n\n"
        f"Pedantry already confirmed this rejection is legitimate:\n"
        f"{pedantry_reason}\n\n"
        f"Is there a single standard, citable, domain-wide convention "
        f"that — if added as an explicit premise — would fully justify "
        f"this initial state and resolve the judge's objection?"
    )
    data, _ = await llm(
        user_msg, role="convention_lift", schema=CONVENTION_LIFT_SCHEMA,
        system=agent_prompt("convention_lift"),
    )
    if on_complete is not None:
        await on_complete(0, data)
    return data


async def _pedantry_check_state0(
    verdict, initial_state, problem, on_complete=None,
):
    """Pedantry filter applied to a failed state 0 verdict. Same
    mechanism as _pedantry_check but adapted for state 0 (which has
    no Step/prev_state structure)."""
    issue_block = ""
    if verdict.issues:
        issue_block = (
            "\n\nThe judge classified these issues:\n"
            f"{_format_issue_list(verdict.issues)}\n\n"
            f"{_PEDANTRY_ISSUE_INSTRUCTIONS}"
        )
    user_msg = (
        f"Problem: {problem}\n\n"
        f"Initial state (state 0) being evaluated: {initial_state}\n\n"
        f"The initial_state judge rejected this state with the following "
        f"reason:\n{verdict.reason}\n\n"
        f"Is this rejection legitimate (state 0 actually contains smuggled "
        f"or made-up content not supported by the problem text) or pedantic "
        f"(state 0 is fine and the judge is being too strict)?"
        f"{issue_block}"
    )
    data, _ = await llm(
        user_msg, role="pedantry", schema=PEDANTRY_SCHEMA,
        system=agent_prompt("pedantry"),
    )
    is_pedantic = data["is_pedantic"]
    reason = data["reason"]
    issue_split = _pedantry_issue_split(data, verdict.issues)
    if on_complete is not None:
        await on_complete(0, verdict.reason, is_pedantic, reason, issue_split)
    return is_pedantic, reason, issue_split


async def _filter_pedantic(
    proof: Proof, verdicts: list[Verdict], problem: str,
    state0_verdict: Verdict | None = None,
    on_each=None, on_state0=None,
) -> tuple[list[Verdict], Verdict | None, list[dict]]:
    """Run pedantry checks on every failed verdict in parallel.

    Returns (updated_verdicts, updated_state0_verdict, pedantry_records).
    Updated verdicts have accepted=True (with a [PEDANTRY OVERRIDE] tag)
    for any verdict marked pedantic. Pedantry_records is a list of
    {step_number, original_reason, is_pedantic, pedantry_reason}; state
    0's record (if any) uses step_number=0.
    """
    failed_indices = [i for i, v in enumerate(verdicts) if not v.accepted]
    state0_failed = (
        state0_verdict is not None and not state0_verdict.accepted
    )
    if not failed_indices and not state0_failed:
        return list(verdicts), state0_verdict, []

    prev_states = [proof.initial_state] + [s.state for s in proof.steps[:-1]]

    tasks = []
    if state0_failed:
        tasks.append(_pedantry_check_state0(
            state0_verdict, proof.initial_state, problem,
            on_complete=on_state0,
        ))
    tasks.extend(
        _pedantry_check(
            verdicts[i],
            proof.steps[i],
            prev_states[i],
            i + 1,
            problem,
            proof,
            original_reason=verdicts[i].reason,
            on_complete=on_each,
        )
        for i in failed_indices
    )
    pedantry_results = await asyncio.gather(*tasks)

    # Unpack: state 0 first (if it ran), then step verdicts.
    s0_result: tuple | None = None
    if state0_failed:
        s0_result = pedantry_results[0]
        pedantry_results = pedantry_results[1:]

    updated = list(verdicts)
    records = []
    for idx, (is_pedantic, reason, issue_split) in zip(
        failed_indices, pedantry_results,
    ):
        original_reason = verdicts[idx].reason
        records.append({
            "step_number": idx + 1,
            "original_reason": original_reason,
            "is_pedantic": is_pedantic,
            "pedantry_reason": reason,
            **issue_split,
        })
        if is_pedantic:
            updated[idx] = Verdict(
                accepted=True,
                reason=f"[PEDANTRY OVERRIDE] {reason}",
                issues=verdicts[idx].issues,
            )

    # Apply state 0 pedantry override too, if applicable.
    updated_state0 = state0_verdict
    if s0_result is not None:
        is_pedantic, reason, issue_split = s0_result
        records.append({
            "step_number": 0,
            "original_reason": state0_verdict.reason,
            "is_pedantic": is_pedantic,
            "pedantry_reason": reason,
            **issue_split,
        })
        if is_pedantic:
            updated_state0 = Verdict(
                accepted=True,
                reason=f"[PEDANTRY OVERRIDE] {reason}",
                issues=state0_verdict.issues,
            )

    return updated, updated_state0, records


def _print_proof(proof: Proof):
    print(f"    {len(proof.steps)} steps")
    print(f"    State 0: {proof.initial_state}")
    for i, step in enumerate(proof.steps):
        print(f"    State {i+1}: {step.state}  [{step.justification_type}]")


def _print_verdicts(
    proof: Proof, verdicts: list[Verdict],
    state0_verdict: Verdict | None = None,
):
    if state0_verdict is not None:
        icon = "PASS" if state0_verdict.accepted else "FAIL"
        print(f"    0. [{icon}] initial_state: {proof.initial_state}")
        if not state0_verdict.accepted:
            print(f"       {state0_verdict.reason}")
    for i, (step, v) in enumerate(zip(proof.steps, verdicts)):
        icon = "PASS" if v.accepted else "FAIL"
        print(f"    {i+1}. [{icon}] {step.justification_type}: {step.justification}")
        if not v.accepted:
            print(f"       {v.reason}")


# Per-problem cycle bounds. Defaults shown below; override via a
# `_limits:` block in a stacked --config YAML.
_DEFAULT_MAX_VERIFY_ATTEMPTS = 3   # formalize+judge cycles per problem
_DEFAULT_MAX_SOLVER_ANSWERS = 3    # distinct solver answers per problem
_DEFAULT_MAX_FORMALIZER_INVALID_ATTEMPTS = 3


# Per-problem wall-clock ceiling. None disables it, which is the historical
# behaviour. Without one, the only bound on a single problem is
# max_turns * request_timeout -- with the frozen 16 x 900s that is 4 hours, and
# even the 180s default gives 14 x 180s ~= 42 minutes -- and a problem whose
# formalizer never converges burns all of it before failing. Observed live: one
# problem consumed ~13 minutes of GPU and recorded attempts=0.
_DEFAULT_MAX_PROBLEM_SECONDS = None


def resolved_limits(config: dict | None = None) -> dict:
    raw = dict((config if config is not None else CONFIG).get("_limits") or {})
    raw.setdefault("max_verify_attempts", _DEFAULT_MAX_VERIFY_ATTEMPTS)
    raw.setdefault("max_solver_answers", _DEFAULT_MAX_SOLVER_ANSWERS)
    raw.setdefault(
        "max_formalizer_invalid_attempts",
        _DEFAULT_MAX_FORMALIZER_INVALID_ATTEMPTS,
    )
    # NOT setdefault-ed: like `max_provider_concurrency`, this is an OPTIONAL
    # limit, and the convention here is that optional limits stay absent unless
    # configured. That keeps `audit["limits"]` byte-identical for every run that
    # does not use a ceiling, so historical audit records stay comparable.
    value = raw.get("max_problem_seconds")
    if value is not None:
        value = float(value)
        if value <= 0:
            raise ValueError(
                f"_limits.max_problem_seconds must be positive or null; got {value!r}"
            )
        raw["max_problem_seconds"] = value
    return raw


def max_problem_seconds(config: dict | None = None) -> float | None:
    """Wall-clock ceiling for one problem, or None for unbounded."""
    return resolved_limits(config).get("max_problem_seconds")


def max_verify_attempts() -> int:
    return int(resolved_limits()["max_verify_attempts"])


def max_solver_answers() -> int:
    return int(resolved_limits()["max_solver_answers"])


def max_formalizer_invalid_attempts() -> int:
    """Protocol/schema corrections, separate from semantic repair rounds."""
    return int(resolved_limits()["max_formalizer_invalid_attempts"])


# Post-judge filter stack. Both passes are ON by default (the audited
# step_wise behavior); turn them off via a `_filters:` block in a stacked
# --config YAML (configs/no_filters.yaml). Orthogonal to `_verifier_mode`
# on purpose: unlike a new mode, this composes with BOTH `step_wise` and
# `step_wise_untyped`, which is what makes the decomposition/typing/filters
# chain in PROCESS_VS_OUTCOME_ABLATION.md expressible without a mode
# explosion.
#
# NOTE both passes are monotonically permissive -- they can only turn a
# REJECT into an ACCEPT, never the reverse -- so for a FIXED sequence of judge
# verdicts, disabling them can only lower certification coverage. That is a
# statement about the filter stack, not a prediction about a re-run: with the
# filters off, a rejection that would have been overridden instead feeds the
# repair loop, which resamples the solver and may certify a different answer.
_DEFAULT_FILTERS = {"pedantry": True, "convention_lift": True}


def resolved_filters(config: dict | None = None) -> dict:
    """Resolve the `_filters` block against its defaults.

    Raises when convention_lift is enabled without pedantry. That combination
    is not merely unusual, it is unimplementable as written: the convention
    judge is structurally DOWNSTREAM of pedantry -- `_convention_lift_step` /
    `_convention_lift_state0` take a `pedantry_reason` argument and their user
    message asserts "Pedantry already confirmed this rejection is legitimate",
    and the candidate set is built from `pedantry_records`. With pedantry off
    there are no records, so the pass would silently never run. A hard error
    beats a silent no-op that would be reported as a convention-lift arm.
    """
    raw = dict((config if config is not None else CONFIG).get("_filters") or {})
    unknown = set(raw) - set(_DEFAULT_FILTERS)
    if unknown:
        raise ValueError(
            f"_filters has unknown key(s) {sorted(unknown)}; "
            f"valid keys are {sorted(_DEFAULT_FILTERS)}"
        )
    for key, default in _DEFAULT_FILTERS.items():
        raw.setdefault(key, default)
        if not isinstance(raw[key], bool):
            raise ValueError(
                f"_filters.{key} must be a boolean; got {raw[key]!r}"
            )
    if raw["convention_lift"] and not raw["pedantry"]:
        raise ValueError(
            "_filters: convention_lift requires pedantry. The convention judge "
            "consumes a pedantry_reason and its candidate set is built from "
            "pedantry_records, so with pedantry off it can never run. Set "
            "convention_lift: false too, or leave pedantry on."
        )
    return raw


def pedantry_enabled(config: dict | None = None) -> bool:
    return bool(resolved_filters(config)["pedantry"])


def convention_lift_enabled(config: dict | None = None) -> bool:
    return bool(resolved_filters(config)["convention_lift"])


# Verification mode. Default `step_wise` is the full process verifier (formalize
# into steps, judge each step). The ablation modes trade away parts of that
# machinery so a step-wise claim can be attributed to the right cause; grade every
# arm on the SAME committed solver response via `grade.py --target solver_initial`
# (cookbook §13). See PROCESS_VS_OUTCOME_ABLATION.md.
#   * `holistic_proof`   -- MIDDLE arm: same formalizer + same repair loop as
#                           step_wise, but ONE global verdict (+ confidence) on the
#                           whole proof instead of per-step typed judges. Certifies
#                           on confidence >= threshold (like answer_score), so its
#                           operating point lies on the swept curve. step_wise -
#                           holistic_proof contrasts the per-step APPARATUS as a
#                           bundle (decomposition + typed judges + pedantry/
#                           convention) -- NOT step decomposition alone.
#   * `answer_score`     -- OUTCOME verifier (ORM): one pass, no formalization; a
#                           scored generative judge emits a self-reported confidence
#                           in [0,1] that the final answer is correct -> a
#                           risk-coverage curve (the preregistered headline baseline).
#                           holistic_proof - answer_score is a BUNDLED contrast
#                           (formalization + representation + repair budget), not
#                           single-factor -- see PROCESS_VS_OUTCOME_ABLATION.md.
#   * `resolve_agreement` -- one pass; commit-first re-solver: the judge solves
#                           independently and certifies on answer agreement (a
#                           separate, honestly-labeled baseline, NOT the ORM).
#   * `step_wise_no_completeness` -- the novelty-isolating ablation (the
#                           "typed-CoT regime" of the paper's Table 1): the FULL
#                           step_wise path (same formalizer, typed states +
#                           diffs, per-step typed judges, pedantry,
#                           convention-lift, repair loop) with ONLY the
#                           completeness-of-change requirement removed. Completeness
#                           is enforced in FOUR places and all four are removed here:
#                           (1) the marked clauses in the judge role prompts, (2) the
#                           formalizer's `uses` spec ("lists exactly ... consumes" is
#                           completeness in structural form), (3) the `uses`
#                           description in PROOF_SCHEMA -- schema descriptions are
#                           sent to the model, so they are prompt -- and (4) the
#                           deterministic missing-`uses` gate plus the repair loop's
#                           retry text. Removing only (4) would leave the formalizer
#                           doing full premise bookkeeping anyway. Diff/typing
#                           integrity (F# id stability, orphan detection, referential
#                           integrity of any `uses` actually emitted) is deliberately
#                           RETAINED. step_wise - step_wise_no_completeness therefore
#                           isolates the completeness invariant itself (not typing/
#                           diffing/decomposition) -- but is not length-controlled, so
#                           treat it as an upper bound until the placebo arm runs. See
#                           PROCESS_VS_OUTCOME_ABLATION.md.
#   * `step_wise_untyped` -- the typing-isolating ablation: the FULL step_wise
#                           path (same formalizer, typed states + diffs, `uses`
#                           bookkeeping, completeness enforcement, pedantry,
#                           convention-lift, repair loop) except that every step
#                           is routed to ONE generic judge role (`step_untyped`)
#                           instead of the citation/problem_given/computation
#                           trio. The step's justification_type is STILL rendered
#                           into the judge's user message and into _format_proof,
#                           so the judges' input representation is held fixed and
#                           only the prompt moves. step_wise - step_wise_untyped
#                           therefore isolates typed-judge SPECIALIZATION -- it
#                           does NOT test whether the type LABEL carries
#                           information (that would need a type-blind arm that
#                           also strips the label, which moves representation as
#                           a second factor). The untyped prompt is the union of
#                           the three typed prompts' guidance, so the arm is
#                           conservative on prompt length rather than starved.
#                           This is the decomposed-but-unspecialized middle arm
#                           that Block Verification (arXiv:2605.20531) occupies.
#                           Run it with `--config configs/untyped_judges.yaml`
#                           stacked last. See PROCESS_VS_OUTCOME_ABLATION.md.
VERIFIER_MODES = (
    "step_wise", "holistic_proof", "answer_score", "resolve_agreement",
    "step_wise_no_completeness", "step_wise_untyped",
)

# Role every step is judged by under `step_wise_untyped`. Defined in
# configs/defaults.yaml alongside the typed judge roles so backend/model
# overrides in harness.apply_args reach it the same way they reach the others.
UNTYPED_JUDGE_ROLE = "step_untyped"


def verifier_mode(config: dict | None = None) -> str:
    cfg = CONFIG if config is None else config
    mode = cfg.get("_verifier_mode") or "step_wise"
    if mode not in VERIFIER_MODES:
        raise ValueError(
            f"_verifier_mode must be one of {VERIFIER_MODES}; got {mode!r}"
        )
    return mode


_SOLVER_REPLAY_CACHE: dict[str, tuple[str, dict[str, str]]] = {}


def solver_replay(config: dict | None = None) -> tuple[str, str, dict[str, str]] | None:
    """Frozen initial solver responses, keyed by problem id. None when unset.

    Returns ``(path, sha256, {problem_id: solution})``.

    **Why this exists.** Every ablation arm re-solves the problem in its own
    subprocess at temperature > 0, so a paired McNemar or bootstrap across arms
    compares verification decisions taken over *different* candidate solutions
    unless the initial solve is identical. `experiments/pairing.py` detects that;
    this makes it not happen. The backlog's alternative -- greedy decoding plus
    an assertion of cross-arm byte equality -- is unsound on this stack: bf16 +
    tensor-parallel serving is not bitwise reproducible even at temperature 0
    (see `experiments/matrix_spec.example.yaml`), so that route yields
    intermittent, unpredictable assertion failures rather than a guarantee.

    Replay pins the FIRST solve only. Later re-solves inside the repair loop are
    arm-specific behaviour and are *supposed* to diverge -- that divergence is
    part of what the ablation measures -- and `solver_initial_sha256`, the seal
    the cross-arm comparator is graded on, covers the first solve alone.
    """
    cfg = CONFIG if config is None else config
    spec = cfg.get("_solver_replay")
    if not spec:
        return None
    if isinstance(spec, str):
        spec = {"path": spec}
    if not isinstance(spec, dict) or not spec.get("path"):
        raise ValueError("_solver_replay must be a path or a mapping with `path`")
    path = str(spec["path"])

    if path not in _SOLVER_REPLAY_CACHE:
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        table = json.loads(raw)
        if not isinstance(table, dict):
            raise ValueError(f"{path}: solver replay map must be a JSON object")
        for key, value in table.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{path}: replay entry {key!r} is not a non-empty string"
                )
        _SOLVER_REPLAY_CACHE[path] = (digest, {str(k): v for k, v in table.items()})

    digest, table = _SOLVER_REPLAY_CACHE[path]
    return path, digest, table


def _answer_from_proof_dict(proof_dict: dict | None) -> str | None:
    if not proof_dict:
        return None
    steps = proof_dict.get("steps") or []
    if not steps:
        return None
    state = steps[-1].get("state") or []
    if not state:
        return None
    return str(state[0])


def _normalize_answer_for_fingerprint(value: str) -> str:
    normalized = " ".join(
        unicodedata.normalize("NFKC", value).casefold().split()
    ).strip(" .")
    normalized = re.sub(
        r"^(?:final\s+)?answer\s*(?:(?:is\s+)|[=:]\s*)",
        "",
        normalized,
    ).strip(" .")
    choice = re.fullmatch(r"[\[(]?([a-d])[\])\].:]?", normalized)
    return choice.group(1) if choice else normalized


def _answer_fingerprints(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    exact = hashlib.sha256(value.encode("utf-8")).hexdigest()
    normalized = _normalize_answer_for_fingerprint(value)
    return exact, hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _build_repair_metrics(
    attempts: list[dict],
    *,
    max_verify: int,
    max_solver: int,
    solver_answers: int,
    solver_solutions: list[str] | None,
    final_answer: object | None,
    verified: bool,
    max_formalizer_invalid: int = _DEFAULT_MAX_FORMALIZER_INVALID_ATTEMPTS,
) -> dict:
    verify_attempt_records = [
        attempt for attempt in attempts if attempt.get("phase") == "verify"
    ]
    # R0 permits one formalizer output. A schema-valid but incomplete proof is
    # therefore a first-attempt decline; any guided correction belongs to R1.
    # Lower-level JSON/action reprompts inside one provider-neutral role call are
    # counted separately in call telemetry because the pipeline never sees them.
    first_attempt = attempts[0] if attempts else None
    first_verify = (
        first_attempt
        if first_attempt and first_attempt.get("phase") == "verify"
        else None
    )
    first_attempt_answer = _answer_from_proof_dict(
        first_verify.get("proof") if first_verify else None
    )
    normalized_final_answer = (
        None if final_answer is None else str(final_answer)
    )
    first_attempt_verified = bool(first_verify and first_verify.get("all_ok"))
    verify_attempts = len(verify_attempt_records)
    solver_retries = max(0, solver_answers - 1)
    judge_repair_rounds = max(0, verify_attempts - 1)
    post_first_attempt_activity = len(attempts) > 1
    post_judge_repair_attempted = any(
        attempt.get("phase") == "verify"
        and attempt.get("all_ok") is False
        and index + 1 < len(attempts)
        for index, attempt in enumerate(attempts)
    )
    semantic_repair_attempted = bool(
        solver_retries or post_judge_repair_attempted
    )
    formalizer_invalid_count = sum(
        1 for attempt in attempts
        if attempt.get("phase") == "formalizer_invalid"
    )
    formalizer_invalid_retry_attempted = bool(
        formalizer_invalid_count and post_first_attempt_activity
    )
    repair_attempted = bool(
        semantic_repair_attempted or post_first_attempt_activity
    )
    answer_change_observable = bool(
        not repair_attempted
        or (
            first_attempt_answer is not None
            and normalized_final_answer is not None
        )
    )
    answer_changed = None
    if not repair_attempted:
        answer_changed = False
    elif answer_change_observable:
        answer_changed = (
            first_attempt_answer.strip() != normalized_final_answer.strip()
        )
    solver_solutions = list(solver_solutions or [])
    solver_solution_text_changed = bool(
        repair_attempted
        and len(solver_solutions) > 1
        and solver_solutions[0].strip() != solver_solutions[-1].strip()
    )
    # Seal the COMMITTED candidate -- the solver's first response, the one every
    # arm is graded on via `grade.py --target solver_initial`. Ablation arms
    # re-solve in separate processes, so a paired McNemar/bootstrap across arms
    # is only meaningful if this hash matches; `experiments/pairing.py` checks
    # that. EXACT bytes, deliberately: any textual difference means the
    # verifiers were handed different inputs, so normalizing here would hide
    # precisely the divergence the seal exists to catch.
    solver_initial_sha256 = (
        hashlib.sha256(solver_solutions[0].encode("utf-8")).hexdigest()
        if solver_solutions else None
    )
    first_exact_hash, first_normalized_hash = _answer_fingerprints(
        first_attempt_answer
    )
    final_exact_hash, final_normalized_hash = _answer_fingerprints(
        normalized_final_answer
    )
    normalized_answer_changed = None
    if not repair_attempted:
        normalized_answer_changed = False
    elif answer_change_observable:
        normalized_answer_changed = first_normalized_hash != final_normalized_hash

    return {
        "repair_enabled": max_verify > 1 or max_solver > 1,
        "repair_attempted": repair_attempted,
        "max_verify_attempts": max_verify,
        "max_solver_answers": max_solver,
        "max_formalizer_invalid_attempts": max_formalizer_invalid,
        "verify_attempts": verify_attempts,
        "solver_answers": solver_answers,
        "solver_retries": solver_retries,
        "formalizer_reject_count": sum(
            1 for attempt in attempts
            if attempt.get("phase") == "formalizer_reject"
        ),
        "formalizer_invalid_count": formalizer_invalid_count,
        "judge_repair_rounds": judge_repair_rounds,
        "post_judge_repair_attempted": post_judge_repair_attempted,
        "semantic_repair_attempted": semantic_repair_attempted,
        "formalizer_invalid_retry_attempted": (
            formalizer_invalid_retry_attempted
        ),
        "first_attempt_verified": first_attempt_verified,
        "final_verified": bool(verified),
        "certified_after_any_retry": bool(
            verified and repair_attempted and not first_attempt_verified
        ),
        "certified_after_semantic_repair": bool(
            verified
            and semantic_repair_attempted
            and not first_attempt_verified
        ),
        "certified_after_nonsemantic_retry": bool(
            verified
            and repair_attempted
            and not first_attempt_verified
            and not semantic_repair_attempted
        ),
        "first_attempt_answer": first_attempt_answer,
        "answer_before_repair": first_attempt_answer if repair_attempted else None,
        "final_answer": normalized_final_answer,
        "answer_before_repair_sha256": first_exact_hash,
        "answer_before_repair_normalized_sha256": first_normalized_hash,
        "final_answer_sha256": final_exact_hash,
        "final_answer_normalized_sha256": final_normalized_hash,
        "answer_change_observable": answer_change_observable,
        "answer_changed_during_repair": answer_changed,
        "normalized_answer_changed_during_repair": normalized_answer_changed,
        "solver_solution_text_changed_during_repair": solver_solution_text_changed,
        "solver_initial_sha256": solver_initial_sha256,
        "solver_solution_count": len(solver_solutions),
    }


async def _solver_call(
    problem, solver_session, retry_reason=None, prior_solution=None,
):
    """First solve or retry. Returns (solution, session_id).

    `prior_solution` covers the replay case (see `solver_replay`): the initial
    answer came from a frozen file, so there is no session to resume. Without
    it, a retry would fall into the first-solve branch and silently re-solve
    from scratch with the rejection reason DISCARDED -- a quietly degraded
    repair loop. Restate the context explicitly instead.
    """
    if solver_session is None and prior_solution is None:
        return await llm(problem, role="solver", system=agent_prompt("solver"))
    if solver_session is None:
        msg = (
            f"{problem}\n\n"
            "You previously answered:\n\n"
            f"{prior_solution}\n\n"
            "That answer was rejected. Reason:\n\n"
            f"{retry_reason}\n\n"
            "Please provide a new answer."
        )
        return await llm(msg, role="solver", system=agent_prompt("solver"))
    msg = (
        "Your previous answer was rejected. Reason:\n\n"
        f"{retry_reason}\n\n"
        "Please provide a new answer."
    )
    return await llm(msg, role="solver", resume=solver_session)


async def _formalizer_call(
    problem,
    solution,
    formalizer_session,
    failed_verdicts_text=None,
    prior_proof=None,
    feedback_text=None,
):
    """Ask the formalizer for a decision (proof or reject).

    Two modes:
    - claude backend: uses session resume. Each call only sends new info;
      the formalizer remembers prior proofs and reasoning from the session.
    - codex backend: stateless. Each call passes the full context (prior
      proof + failed verdicts) because codex exec resume doesn't support
      --output-schema, so we can't get structured output on resumed sessions.

    Returns (decision_dict, session_id).
    """
    backend = agent_settings("formalizer").get("backend", "claude")

    if backend == "claude":
        # Session-resume mode
        if formalizer_session is None:
            user_msg = (
                f"Problem: {problem}\n\nSolution: {solution}\n\n"
                "Formalize this into a proof, or reject if it has errors."
            )
            if feedback_text:
                user_msg = f"{user_msg}\n\n{feedback_text}"
            return await llm(
                user_msg, role="formalizer", schema=_formalizer_decision_schema(),
                system=agent_prompt("formalizer"),
            )
        elif feedback_text:
            return await llm(
                feedback_text, role="formalizer",
                schema=_formalizer_decision_schema(),
                resume=formalizer_session,
            )
        elif failed_verdicts_text:
            user_msg = (
                "Your proof failed verification.\n\n"
                f"Failed steps:\n{failed_verdicts_text}\n\n"
                "Either produce a corrected proof (action='proof') or reject "
                "the underlying solution (action='reject') with a reason for "
                "the solver."
            )
            return await llm(
                user_msg, role="formalizer", schema=_formalizer_decision_schema(),
                resume=formalizer_session,
            )
        else:
            user_msg = (
                f"The solver provided a new answer:\n\n{solution}\n\n"
                "Formalize this into a proof, or reject again if it still has errors."
            )
            return await llm(
                user_msg, role="formalizer", schema=_formalizer_decision_schema(),
                resume=formalizer_session,
            )

    # Stateless mode (codex or any backend that doesn't support resume+schema)
    parts = [f"Problem: {problem}", f"Solution: {solution}"]
    if prior_proof is not None:
        parts.append(
            "Your previous proof attempt:\n" + _format_proof(prior_proof)
        )
    if failed_verdicts_text:
        parts.append(
            "Failed verification steps:\n" + failed_verdicts_text
        )
        parts.append(
            "Either produce a corrected proof (action='proof') or reject "
            "the underlying solution (action='reject') with a reason for "
            "the solver."
        )
    if feedback_text:
        parts.append(feedback_text)
    elif not failed_verdicts_text:
        parts.append("Formalize this into a proof, or reject if it has errors.")
    user_msg = "\n\n".join(parts)
    return await llm(
        user_msg, role="formalizer", schema=_formalizer_decision_schema(),
        system=agent_prompt("formalizer"),
    )


async def run(problem: str, *, pid: str | None = None, partial_save_path: str | None = None):
    """Run the verified-reasoning pipeline on one problem.

    Optional `pid` is the problem id; when set, every print is prefixed
    with `[pid]` so concurrent runs are distinguishable in shared logs.

    Optional `partial_save_path` enables crash-safe per-step persistence.
    The current running state is atomically written to that path after
    every meaningful boundary (solver returns, formalizer returns, each
    individual judge call completes, each individual pedantry call
    completes, and after each attempt is appended). On crash mid-attempt,
    the partial file contains everything that completed before the crash.
    """
    attempts = []  # one entry per formalize+judge cycle

    # ── per-problem logging helper ───────────────────────────────────
    def log(msg: str) -> None:
        prefix = f"[{pid}] " if pid else ""
        print(f"{prefix}{msg}")

    # ── per-step partial save helpers ────────────────────────────────
    state = {
        "pid": pid,
        "problem": problem,
        "solution": None,
        "attempts": attempts,
        "in_progress_attempt": None,
        "last_event": None,
    }
    save_lock = asyncio.Lock()

    async def save_partial(event: str):
        state["last_event"] = event
        if not partial_save_path:
            return
        async with save_lock:
            tmp = partial_save_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, partial_save_path)

    # ── initial solve ────────────────────────────────────────────────
    replay = solver_replay()
    replay_provenance = None
    if replay is None:
        log("\n[solve] Solving...")
        solution, solver_session = await _solver_call(problem, None)
    else:
        replay_path, replay_sha, replay_table = replay
        if pid is None:
            raise ValueError(
                "_solver_replay needs a problem id to look up a frozen "
                "response, but this run passed pid=None"
            )
        if str(pid) not in replay_table:
            # Deliberately fatal. Falling back to a live solve would reintroduce
            # cross-arm divergence for exactly the problems the map failed to
            # cover -- an unpaired subset hiding inside a run that claims to be
            # paired, which is worse than no replay at all.
            raise KeyError(
                f"_solver_replay map {replay_path} has no entry for problem "
                f"{pid!r}; refusing to solve live, which would silently unpair "
                f"this problem across arms"
            )
        solution = replay_table[str(pid)]
        solver_session = None
        replay_provenance = {
            "replayed": True,
            "source": replay_path,
            "source_sha256": replay_sha,
        }
        log("\n[solve] Replaying frozen solver response (no solver call).")
    log(f"    {solution[:200]}")
    state["solution"] = solution
    solver_solutions = [solution]
    state["solver_solutions"] = solver_solutions
    await save_partial("solver_returned")

    formalizer_session = None
    failed_verdicts_text = None  # set after a verification failure
    last_proof = None             # latest proof — used by stateless backends
    formalizer_feedback = None    # set after an incomplete decision shape
    malformed_formalizer_decisions = 0
    verify_attempts = 0
    solver_answers = 1  # initial solve counts as the first answer

    max_verify = max_verify_attempts()
    max_solver = max_solver_answers()
    max_formalizer_invalid = max_formalizer_invalid_attempts()

    mode = verifier_mode()
    if mode in ("answer_score", "resolve_agreement"):
        # One-pass ablation modes (cookbook §13): skip formalization + per-step
        # judging. Correctness for the ablation is graded on the SAME committed
        # solver response in EVERY arm via `grade.py --target solver_initial`, so
        # only the verify DECISION differs. See PROCESS_VS_OUTCOME_ABLATION.md.
        verifier_extra: dict = {}
        if mode == "answer_score":
            # OUTCOME verifier (ORM headline): a scored confidence on the answer.
            log("\n[answer-score] scoring the solver's answer (outcome verifier)...")
            confidence, believes_correct, candidate_answer, as_inputs = (
                await judge_answer_score(problem, solution)
            )
            threshold = float(CONFIG.get("_score_threshold", 0.5))
            verified = confidence >= threshold
            verdicts_dict = [{
                "step_number": 0, "role": "answer_scorer", "accepted": verified,
                "confidence": confidence,
                "reason": (f"confidence {confidence:.3f} "
                           f"{'>=' if verified else '<'} threshold {threshold:.3f}"),
            }]
            # Persist the RAW confidence -- both the clamped selective signal (swept
            # post-hoc into a risk-coverage curve) and the pre-clamp value (so the
            # calibration/ECE analysis can see out-of-range, miscalibrated scores).
            verifier_extra = {
                "verifier_confidence": confidence,
                "verifier_confidence_raw": as_inputs.get("confidence_raw"),
                "verifier_believes_correct": believes_correct,
                "score_threshold": threshold,
            }
        else:  # resolve_agreement: commit-first re-solve-and-compare (NOT self-consistency@k, NOT the ORM)
            log("\n[resolve-agreement] commit-first re-solve + answer agreement...")
            verdict, candidate_answer, _ = await judge_answer(problem, solution)
            verified = bool(verdict.accepted)
            verdicts_dict = [{
                "step_number": 0, "role": "answer_judge", "accepted": verified,
                "reason": verdict.reason,
            }]

        # Display answer = the judge's extraction; the ABLATION grades the shared
        # solver response (--target solver_initial). An empty candidate has nothing
        # to certify -> a permanent decline at ANY threshold.
        answer = (candidate_answer or "").strip() or None
        if answer is None:
            # Applies regardless of the (sub- or supra-threshold) confidence: with no
            # extractable answer the row can never be certified, so null the sealed
            # score. Leaving ANY live confidence here (even a low one) would let a
            # downstream threshold sweep reconstruct this answerless row as certified
            # at cutoffs <= that score -> coverage inflation, contradicting
            # verified=False. None == never certifiable, consistent at every cutoff.
            verified = False
            verdicts_dict[0]["accepted"] = False
            verdicts_dict[0]["reason"] = "no extractable candidate answer; declined"
            if "verifier_confidence" in verifier_extra:
                verifier_extra["verifier_confidence"] = None
                verifier_extra["verifier_confidence_raw"] = None
                verifier_extra["verifier_believes_correct"] = None
                verdicts_dict[0]["confidence"] = None
        attempts.append({
            "attempt": len(attempts) + 1,
            "phase": "verify", "verifier_mode": mode, "all_ok": verified,
            "proof": None, "verdicts": verdicts_dict,
        })
        repair_metrics = _build_repair_metrics(
            attempts, max_verify=max_verify, max_solver=max_solver,
            solver_answers=solver_answers, solver_solutions=solver_solutions,
            final_answer=answer, verified=verified,
            max_formalizer_invalid=max_formalizer_invalid,
        )
        # One pass => R0 == final. Populate the first-attempt (R0) fields so
        # grade.py --target first_attempt and summarize_run's R0 operating point
        # work (the proof-based builder leaves first_attempt_answer None when
        # proof is None, which would break R0 grading of every one-pass run).
        repair_metrics["first_attempt_answer"] = answer
        repair_metrics["first_attempt_verified"] = verified
        log(f"\n{'='*40}")
        log(f"{'JUDGE-PASSED' if verified else 'REJECTED'} ({mode}) — answer: {answer}")
        await save_partial(f"{mode}_done")
        return {
            "problem": problem,
            "answer": answer,
            "verified": verified,
            "verified_unconditionally": verified,
            "verified_under_assumptions": [],
            "solution": solution,
            "solver_solutions": solver_solutions,
            "proof": None,
            "verdicts": verdicts_dict,
            "attempts": attempts,
            "repair_metrics": repair_metrics,
            "verifier_mode": mode,
            **verifier_extra,
        }

    while True:
        # Ask formalizer for a decision
        log(
            f"\n[formalize] verify={verify_attempts}/{max_verify} "
            f"solver_answers={solver_answers}/{max_solver}..."
        )
        decision, new_session = await _formalizer_call(
            problem, solution, formalizer_session, failed_verdicts_text,
            prior_proof=last_proof,
            feedback_text=formalizer_feedback,
        )
        if formalizer_session is None:
            formalizer_session = new_session
        action = decision.get("action")
        log(f"    Formalizer chose: {action}")
        await save_partial(f"formalizer_returned:{action}")

        formalizer_feedback = _formalizer_decision_feedback(
            decision, enforce_completeness=(mode != "step_wise_no_completeness"),
        )
        if formalizer_feedback:
            malformed_formalizer_decisions += 1
            log(f"    Incomplete formalizer decision: {formalizer_feedback[:200]}")
            attempts.append({
                "attempt": len(attempts) + 1,
                "phase": "formalizer_invalid",
                "action": action,
                "reason": formalizer_feedback,
                "decision": decision,
            })
            await save_partial("attempt_appended:formalizer_invalid")
            if malformed_formalizer_decisions >= max_formalizer_invalid:
                log(
                    "\n[!] Max incomplete formalizer decisions "
                    f"({max_formalizer_invalid}) "
                    "reached, giving up"
                )
                break
            continue
        malformed_formalizer_decisions = 0

        if action == "reject":
            formalizer_feedback = None
            reason = decision.get("reject_reason", "")
            log(f"    Rejected: {reason[:200]}")
            attempts.append({
                "attempt": len(attempts) + 1,
                "phase": "formalizer_reject",
                "reject_reason": reason,
            })
            await save_partial("attempt_appended:formalizer_reject")
            if solver_answers >= max_solver:
                log(f"\n[!] Max solver answers ({max_solver}) reached, giving up")
                break
            # Solver retry
            log("\n[solve] Retrying with formalizer's reason...")
            # `solver_solutions[-1]` is what was actually rejected. It matters
            # only on the replay path (no session to resume); the session branch
            # ignores it because the model already has the exchange in context.
            solution, _ = await _solver_call(
                problem,
                solver_session,
                retry_reason=reason,
                prior_solution=solver_solutions[-1] if replay_provenance else None,
            )
            log(f"    {solution[:200]}")
            solver_answers += 1
            solver_solutions.append(solution)
            failed_verdicts_text = None  # reset, this is a fresh formalization
            last_proof = None             # reset, no prior proof for new answer
            state["solution"] = solution
            await save_partial("solver_returned")
            continue

        # action == "proof": judge it
        formalizer_feedback = None
        verify_attempts += 1
        proof = _proof_from_dict(decision["proof"])
        last_proof = proof
        _print_proof(proof)

        if mode == "holistic_proof":
            # MIDDLE arm: ONE global verdict on the whole proof instead of the
            # per-step typed judges + pedantry + convention passes. Same formalizer
            # and same repair loop as step_wise, so step_wise - holistic_proof
            # contrasts the per-step apparatus as a bundle (NOT decomposition alone).
            # The attempt's `all_ok` is the CERTIFIED bit (confidence >= threshold),
            # symmetric with answer_score -- so first_attempt_verified, the repair-
            # lift metrics, and the final `verified` all use the SAME rule and the
            # operating point lies on the swept curve. The repair loop, however,
            # stops on the judge's `accepted` (a FIXED rule, independent of the swept
            # threshold), so the number of repair rounds -- and thus the sealed proof
            # + its confidence -- never depends on the dev cutoff. Handles its own
            # attempt + repair here, then continues (the step-wise block is skipped).
            log("\n[judge] Holistic whole-proof verification...")
            h_conf, h_verdict, h_inputs = await judge_proof_holistic(proof, problem)
            threshold = float(CONFIG.get("_score_threshold", 0.5))
            all_ok = h_conf >= threshold          # certified bit (not the accept)
            in_progress = {
                "attempt": len(attempts) + 1,
                "phase": "verify",
                "proof": _proof_to_dict(proof),
                "state0_verdict": None,
                "verdicts": [{
                    "step_number": 0,
                    "role": "holistic_judge",
                    "user_msg": h_inputs["user_msg"],
                    "system": h_inputs["system"],
                    "accepted": h_verdict.accepted,
                    "reason": h_verdict.reason,
                    "confidence": h_conf,
                }],
                "pedantry": [],
                "conventions": [],
                "all_ok": all_ok,
                "verifier_confidence": h_conf,
                "verifier_confidence_raw": h_inputs.get("confidence_raw"),
                "verifier_believes_correct": h_verdict.accepted,
            }
            attempts.append(in_progress)
            state["in_progress_attempt"] = None
            await save_partial("attempt_appended:verify")
            if h_verdict.accepted:                # repair stops on ACCEPT (fixed rule)
                break
            if verify_attempts >= max_verify:
                log(f"\n[!] Max verify attempts ({max_verify}) reached, giving up")
                break
            # Repair: hand the holistic rejection reason back to the formalizer.
            failed_verdicts_text = h_verdict.reason or "The proof was rejected."
            continue

        # Set up the in-progress attempt with verdict placeholders that
        # get filled in as individual judges complete. state0_verdict
        # is a separate slot — state 0 is audited in parallel with the
        # per-step judges but isn't part of the step-indexed verdicts
        # list (it has no Step associated with it).
        in_progress = {
            "attempt": len(attempts) + 1,
            "phase": "verify",
            "proof": _proof_to_dict(proof),
            "state0_verdict": None,
            "verdicts": [None] * len(proof.steps),
            "pedantry": [],
            "conventions": [],
            "all_ok": None,
        }
        state["in_progress_attempt"] = in_progress
        await save_partial("attempt_started")

        log("\n[judge] Verifying...")

        async def on_judge_complete(step_number, verdict, inputs):
            in_progress["verdicts"][step_number - 1] = {
                "step_number": step_number,
                "role": inputs["role"],
                "user_msg": inputs["user_msg"],
                "system": inputs["system"],
                "accepted": verdict.accepted,
                "reason": verdict.reason,
                "issues": verdict.issues,
            }
            await save_partial(f"judge_completed:{step_number}")

        async def on_state0_complete(_step_number, verdict, inputs):
            in_progress["state0_verdict"] = {
                "step_number": 0,
                "role": inputs["role"],
                "user_msg": inputs["user_msg"],
                "system": inputs["system"],
                "accepted": verdict.accepted,
                "reason": verdict.reason,
                "issues": verdict.issues,
            }
            await save_partial("state0_judge_completed")

        # Run state 0 judge and the per-step judges concurrently.
        # Wasted work on a state 0 failure is trivial; keeping them in
        # one gather means the repair loop gets richer feedback.
        state0_task = judge_initial_state(
            proof, problem, on_complete=on_state0_complete,
        )
        step_judges_task = _judge_proof(
            proof, problem, on_each=on_judge_complete,
        )
        (state0_verdict, state0_input), (verdicts, judge_inputs) = \
            await asyncio.gather(state0_task, step_judges_task)
        _print_verdicts(proof, verdicts, state0_verdict=state0_verdict)

        # Pedantry filter — for each failed verdict (including state 0),
        # decide if the rejection is legitimate or just nitpicking.
        # Pedantic ones get marked accepted with a [PEDANTRY OVERRIDE] tag.
        n_failed_steps = sum(1 for v in verdicts if not v.accepted)
        state0_failed = not state0_verdict.accepted
        n_failed_before = n_failed_steps + (1 if state0_failed else 0)
        # `_filters.pedantry: false` (configs/no_filters.yaml) skips the pass
        # entirely, leaving every judge rejection standing. Because the pass is
        # monotonically permissive, disabling it can only lower coverage.
        run_pedantry = pedantry_enabled()
        if n_failed_before > 0 and run_pedantry:
            log(f"\n[pedantry] Filtering {n_failed_before} failures"
                f"{' (incl. state 0)' if state0_failed else ''}...")

            async def on_pedantry_complete(
                step_number, original_reason, is_pedantic, reason, issue_split,
            ):
                in_progress["pedantry"].append({
                    "step_number": step_number,
                    "original_reason": original_reason,
                    "is_pedantic": is_pedantic,
                    "pedantry_reason": reason,
                    **issue_split,
                })
                await save_partial(f"pedantry_completed:{step_number}")

            verdicts, state0_verdict, pedantry_records = await _filter_pedantic(
                proof, verdicts, problem,
                state0_verdict=state0_verdict,
                on_each=on_pedantry_complete,
                on_state0=on_pedantry_complete,
            )
            n_pedantic = sum(1 for r in pedantry_records if r["is_pedantic"])
            n_legit = n_failed_before - n_pedantic
            log(f"    {n_pedantic} marked pedantic, {n_legit} legitimate")
        else:
            if n_failed_before > 0 and not run_pedantry:
                log(f"\n[pedantry] SKIPPED (_filters.pedantry: false) — "
                    f"{n_failed_before} rejection(s) stand")
            pedantry_records = []

        # Convention-lift pass: for each verdict that survived pedantry
        # as a legitimate rejection, ask whether a standard domain
        # convention would fully justify it. If so, override to
        # accepted-under-assumption and record the convention.
        conventions_added: list[dict] = []
        legit_rejections = [
            (r, v, s) for r, v, s in (
                [
                    (r, state0_verdict, "state0") for r in pedantry_records
                    if r["step_number"] == 0 and not r["is_pedantic"]
                ] + [
                    (r, verdicts[r["step_number"] - 1], "step") for r in pedantry_records
                    if r["step_number"] >= 1 and not r["is_pedantic"]
                ]
            )
            if not v.accepted
        ]
        # `_filters.convention_lift: false` skips the pass. With pedantry off,
        # `pedantry_records` is empty so `legit_rejections` is empty anyway --
        # resolved_filters() rejects convention_lift-without-pedantry precisely
        # so that implicit skip can never be mistaken for a real lift pass.
        run_convention_lift = convention_lift_enabled()
        if legit_rejections and not run_convention_lift:
            log(f"\n[convention_lift] SKIPPED (_filters.convention_lift: false) "
                f"— {len(legit_rejections)} legitimate rejection(s) stand")
        if legit_rejections and run_convention_lift:
            log(f"\n[convention_lift] Evaluating {len(legit_rejections)} "
                f"legitimate rejections for standard-convention lift...")

            async def on_convention_complete(step_number, data):
                conventions_added.append({
                    "step_number": step_number,
                    "can_lift": data.get("can_lift"),
                    "convention": data.get("convention"),
                    "source": data.get("source"),
                    "reasoning": data.get("reasoning"),
                })
                await save_partial(f"convention_lift_completed:{step_number}")

            prev_states = [proof.initial_state] + [s.state for s in proof.steps[:-1]]
            tasks = []
            for pr, v, kind in legit_rejections:
                sn = pr["step_number"]
                ped_reason = pr["pedantry_reason"]
                if kind == "state0":
                    tasks.append(_convention_lift_state0(
                        v, proof.initial_state, problem, ped_reason,
                        on_complete=on_convention_complete,
                    ))
                else:
                    tasks.append(_convention_lift_step(
                        v, proof.steps[sn - 1], prev_states[sn - 1],
                        sn, problem, proof, ped_reason,
                        on_complete=on_convention_complete,
                    ))
            results = await asyncio.gather(*tasks)

            # Apply accepted-under-assumption overrides.
            n_lifted = 0
            for (pr, v, kind), data in zip(legit_rejections, results):
                if not data.get("can_lift"):
                    continue
                n_lifted += 1
                tag = (
                    f"[ASSUMPTION: {data.get('convention','')}] "
                    f"(source: {data.get('source','')})"
                )
                new_verdict = Verdict(accepted=True, reason=tag)
                if kind == "state0":
                    state0_verdict = new_verdict
                else:
                    verdicts[pr["step_number"] - 1] = new_verdict
            n_unlifted = len(legit_rejections) - n_lifted
            log(f"    {n_lifted} lifted under convention, {n_unlifted} not")

        all_ok = state0_verdict.accepted and all(v.accepted for v in verdicts)
        # Replace any pedantry-overridden verdicts in the in-progress record.
        in_progress["state0_verdict"] = {
            "step_number": 0,
            "role": state0_input["role"],
            "user_msg": state0_input["user_msg"],
            "system": state0_input["system"],
            "accepted": state0_verdict.accepted,
            "reason": state0_verdict.reason,
            "issues": state0_verdict.issues,
        }
        for i, v in enumerate(verdicts):
            in_progress["verdicts"][i] = {
                "step_number": i + 1,
                "role": judge_inputs[i]["role"],
                "user_msg": judge_inputs[i]["user_msg"],
                "system": judge_inputs[i]["system"],
                "accepted": v.accepted,
                "reason": v.reason,
                "issues": v.issues,
            }
        in_progress["pedantry"] = pedantry_records
        in_progress["conventions"] = conventions_added
        in_progress["all_ok"] = all_ok
        attempts.append(in_progress)
        state["in_progress_attempt"] = None
        await save_partial("attempt_appended:verify")

        if all_ok:
            break

        if verify_attempts >= max_verify:
            log(f"\n[!] Max verify attempts ({max_verify}) reached, giving up")
            break

        # Set up for next iteration: formalizer will get the failed verdicts
        # including a state 0 failure if one occurred.
        failed_verdicts_text = _format_failed_verdicts(
            proof, verdicts, state0_verdict=state0_verdict,
        )

    # Build final result from the last attempt that produced a proof
    last_proof_attempt = next(
        (a for a in reversed(attempts) if a.get("phase") == "verify"),
        None,
    )
    if last_proof_attempt:
        verified = last_proof_attempt["all_ok"]
        proof_dict = last_proof_attempt["proof"]
        verdicts_dict = last_proof_attempt["verdicts"]
        # Safe extraction (returns None on an empty terminal state) instead of raw
        # indexing -- holistic_proof's single global verdict scrutinizes state
        # structure less than the per-step judges, so an empty state is reachable.
        answer = _answer_from_proof_dict(proof_dict)
        # Collect any conventions the last (winning) attempt invoked.
        # A proof is verified_unconditionally iff it's verified AND no
        # assumptions were added.
        verified_under_assumptions = [
            {
                "step_number": c["step_number"],
                "convention": c["convention"],
                "source": c["source"],
            }
            for c in (last_proof_attempt.get("conventions") or [])
            if c.get("can_lift")
        ]
    else:
        # All attempts ended in rejection — no proof was ever produced
        verified = False
        proof_dict = None
        verdicts_dict = []
        answer = None
        verified_under_assumptions = []

    # holistic_proof: seal the scored-arm extras. `verified` is already the certified
    # bit (last attempt's `all_ok` = confidence >= threshold, set in the holistic
    # branch), so it lies ON the swept risk-coverage curve; here we just seal the raw
    # + clamped confidence, the auxiliary `believes` (the judge's accept, which drove
    # the fixed repair rule), and the threshold -- and force-decline on an empty
    # terminal (mirrors answer_score's empty-candidate guard).
    holistic_extra: dict = {}
    if mode == "holistic_proof":
        h_conf = (last_proof_attempt or {}).get("verifier_confidence")
        h_conf_raw = (last_proof_attempt or {}).get("verifier_confidence_raw")
        believes = ((last_proof_attempt or {}).get("verifier_believes_correct")
                    if last_proof_attempt is not None else None)
        threshold = float(CONFIG.get("_score_threshold", 0.5))
        if answer is None:
            # No terminal answer -> never certifiable at any threshold; null the
            # sealed score (mirrors answer_score's empty-candidate guard) so a
            # downstream threshold sweep can't reconstruct this row as certified,
            # and force-decline + reflect it in the audit verdict.
            h_conf = h_conf_raw = believes = None
            verified = False
            if verdicts_dict:
                verdicts_dict[0]["accepted"] = False
                verdicts_dict[0]["confidence"] = None
                verdicts_dict[0]["reason"] = "empty terminal state; declined"
        else:
            verified = bool(h_conf is not None and h_conf >= threshold)
        holistic_extra = {
            "verifier_confidence": h_conf,
            "verifier_confidence_raw": h_conf_raw,
            "verifier_believes_correct": believes,
            "score_threshold": threshold,
        }

    verified_unconditionally = bool(verified and not verified_under_assumptions)
    repair_metrics = _build_repair_metrics(
        attempts,
        max_verify=max_verify,
        max_solver=max_solver,
        solver_answers=solver_answers,
        solver_solutions=solver_solutions,
        final_answer=answer,
        verified=verified,
        max_formalizer_invalid=max_formalizer_invalid,
    )

    print(f"\n{'='*40}")
    if verified and verified_under_assumptions:
        print(
            f"JUDGE-PASSED (under {len(verified_under_assumptions)} "
            f"assumption{'s' if len(verified_under_assumptions)!=1 else ''}) "
            f"— answer: {answer}"
        )
    else:
        print(f"{'JUDGE-PASSED' if verified else 'REJECTED'} — answer: {answer}")

    # holistic_proof seals the (clamped + raw) whole-proof confidence, the auxiliary
    # believes bit, and the threshold -- all computed in `holistic_extra` above,
    # mirroring answer_score's scored sealing. step_wise has no single scalar
    # confidence (its signal is the all-pass conjunction), so it omits these keys.
    verifier_extra = holistic_extra

    return {
        "problem": problem,
        "answer": answer,
        "verified": verified,
        "verified_unconditionally": verified_unconditionally,
        "verified_under_assumptions": verified_under_assumptions,
        "solution": solution,
        "solver_solutions": solver_solutions,
        "proof": proof_dict,
        "verdicts": verdicts_dict,
        "attempts": attempts,
        "repair_metrics": repair_metrics,
        "verifier_mode": mode,
        # Present ONLY on a replayed run. A reader must be able to tell that the
        # solver did not run here, so this is never emitted as `false` -- an
        # absent key means a live solve, and a present key carries the source
        # and its hash so the frozen map is auditable from the sealed result.
        **({"solver_replay": replay_provenance} if replay_provenance else {}),
        **verifier_extra,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Typed reasoning and LLM step audits")
    parser.add_argument("problem", nargs="?", default="What is 2 + 2?")
    # Debug entry point only — the real CLI is `reasoning-eval` (cli.py). Backend
    # selection lives there; this just runs one problem against the current
    # config so you can poke at the pipeline directly.
    parser.add_argument("--config", action="append", default=None, help="Path to config YAML. Repeat to stack multiple.")
    parser.add_argument("--watch", action="store_true", help="Stream LLM events to stderr in real time")
    args = parser.parse_args()

    CONFIG.update(load_config(args.config))
    WATCH = args.watch
    asyncio.run(run(args.problem))
