"""Deterministic witness-state diffing. No LLM calls.

Witness states are full lists of strings. Index 0 is the answer slot
(the goal being resolved). Other elements are established facts,
conventionally prefixed with a stable id like ``F1:`` or ``F2:``.
These functions turn consecutive states into explicit mutations, join
them with each step's declared ``uses`` premises to form a dependency
DAG, and detect structural hidden premises (orphans).

Everything here is a pure function over proof dicts as stored in
``runs/partial/<pid>.json`` (``proof["initial_state"]``,
``proof["steps"][i]["state"]``, optional ``proof["steps"][i]["uses"]``).
Old artifacts without ``uses`` still diff cleanly; ``uses``-dependent
outputs simply come back empty.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

ANSWER_SLOT_ID = "A0"
PROBLEM_TOKEN = "problem"

_FACT_ID = re.compile(r"^\s*(F\d+)\s*:\s*(.*)$", re.DOTALL)


def parse_element(text: str) -> tuple[str | None, str]:
    """Split a state element into (fact id, content).

    Returns ``(None, text)`` when the element carries no ``F#:`` prefix.
    """
    match = _FACT_ID.match(str(text))
    if match:
        return match.group(1), match.group(2).strip()
    return None, str(text).strip()


def content_hash(text: str) -> str:
    """Stable hash of an element's content with the id prefix and
    insignificant whitespace stripped, so a pure renumbering is
    detectable as the same content under a different id."""
    _, content = parse_element(text)
    normalized = " ".join(content.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class Mutation:
    kind: str  # add | remove | modify | answer_slot
    element_id: str | None
    before: str | None = None
    after: str | None = None

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "element_id": self.element_id,
            "before": self.before,
            "after": self.after,
        }


def _index_state(state: list[str]) -> tuple[dict[str, str], list[str]]:
    """Return (labelled facts by id, unlabelled elements) for the
    non-answer-slot portion of a state. A duplicated id keeps the first
    occurrence; later duplicates are treated as unlabelled so they still
    surface in the diff instead of silently collapsing."""
    labelled: dict[str, str] = {}
    unlabelled: list[str] = []
    for element in state[1:]:
        fact_id, _ = parse_element(element)
        if fact_id is not None and fact_id not in labelled:
            labelled[fact_id] = str(element)
        else:
            unlabelled.append(str(element))
    return labelled, unlabelled


def diff_states(prev: list[str], curr: list[str]) -> list[Mutation]:
    """Diff two consecutive witness states into explicit mutations.

    Kinds:
    - ``answer_slot``: index 0 changed.
    - ``add``: an id (or unlabelled element) present only in ``curr``.
    - ``remove``: an id (or unlabelled element) present only in ``prev``.
    - ``modify``: same id in both states with changed content.
    """
    prev = [str(e) for e in (prev or [])]
    curr = [str(e) for e in (curr or [])]
    mutations: list[Mutation] = []

    prev_answer = prev[0] if prev else None
    curr_answer = curr[0] if curr else None
    if prev_answer != curr_answer:
        mutations.append(Mutation(
            kind="answer_slot",
            element_id=ANSWER_SLOT_ID,
            before=prev_answer,
            after=curr_answer,
        ))

    prev_facts, prev_loose = _index_state(prev)
    curr_facts, curr_loose = _index_state(curr)

    for fact_id in sorted(
        set(prev_facts) | set(curr_facts),
        key=lambda fid: (len(fid), fid),
    ):
        before = prev_facts.get(fact_id)
        after = curr_facts.get(fact_id)
        if before is None:
            mutations.append(Mutation("add", fact_id, None, after))
        elif after is None:
            mutations.append(Mutation("remove", fact_id, before, None))
        elif content_hash(before) != content_hash(after):
            mutations.append(Mutation("modify", fact_id, before, after))

    # Unlabelled elements match by exact content; anything unmatched is
    # an add/remove with element_id=None.
    prev_pool = list(prev_loose)
    for element in curr_loose:
        if element in prev_pool:
            prev_pool.remove(element)
        else:
            mutations.append(Mutation("add", None, None, element))
    for element in prev_pool:
        mutations.append(Mutation("remove", None, element, None))
    return mutations


def find_renumbering(prev: list[str], curr: list[str]) -> list[tuple[str, str]]:
    """Detect facts whose content moved to a different id between two
    states. Returns ``(old_id, new_id)`` pairs. A pure renumbering is
    forbidden: ids must stay stable so premise edges remain valid."""
    prev_facts, _ = _index_state([""] + [e for e in (prev or [])[1:]])
    curr_facts, _ = _index_state([""] + [e for e in (curr or [])[1:]])
    prev_by_hash: dict[str, str] = {}
    for fact_id, element in prev_facts.items():
        prev_by_hash.setdefault(content_hash(element), fact_id)
    moved: list[tuple[str, str]] = []
    for fact_id, element in curr_facts.items():
        # Same id + same content is stable. Skip before the hash map so
        # duplicate content under another id (F1: x, F4: x) is not
        # misread as an F1→F4 renumbering on an unchanged state.
        prev_element = prev_facts.get(fact_id)
        if (
            prev_element is not None
            and content_hash(prev_element) == content_hash(element)
        ):
            continue
        old_id = prev_by_hash.get(content_hash(element))
        # A renumbering means the content MOVED: it was at old_id in prev and old_id
        # no longer carries it in curr. If old_id still holds this same content in
        # curr, this is a DUPLICATE under a new id (e.g. F1: x>0 remains while a step
        # newly derives F4: x>0), not a move -- flagging it would give two logically
        # equivalent proofs opposite verdicts and one-directionally deflate the
        # completeness-of-change metric (licensed_step_fraction / unlicensed_mutation).
        curr_at_old = curr_facts.get(old_id)
        if (
            old_id is not None
            and old_id != fact_id
            and (curr_at_old is None
                 or content_hash(curr_at_old) != content_hash(element))
        ):
            moved.append((old_id, fact_id))
    return sorted(moved)


def _step_uses(step: dict) -> list[str]:
    uses = step.get("uses")
    if not isinstance(uses, list):
        return []
    return [str(u) for u in uses]


def premise_edges(step: dict, prev_state: list[str]) -> list[tuple[str, str]]:
    """Dependency edges ``(claim_id, used_id)`` for one step.

    Claims are the elements this step introduced or changed (from the
    deterministic diff); the used ids come from the step's declared
    ``uses``. Steps without ``uses`` (old artifacts) yield no edges.
    """
    uses = _step_uses(step)
    if not uses:
        return []
    claims: list[str] = []
    for mutation in diff_states(prev_state, step.get("state") or []):
        if mutation.kind == "remove":
            continue
        claims.append(
            mutation.element_id if mutation.element_id is not None
            else content_hash(mutation.after or "")[:12]
        )
    edges = [(claim, used) for claim in claims for used in uses]
    # Deduplicate, preserving order.
    seen: set[tuple[str, str]] = set()
    return [e for e in edges if not (e in seen or seen.add(e))]


def find_orphans(proof: dict) -> list[str]:
    """Ids consumed via ``uses`` that were never introduced by S0, the
    problem, or any prior step — i.e. structural hidden premises.

    ``"problem"`` is always a legal source. The answer slot id is legal
    once any state exists. Introduction is cumulative: a fact dropped in
    a later state was still introduced, so referencing it is not an
    orphan (that stricter check belongs to per-step validation).
    """
    initial_state = proof.get("initial_state") or []
    steps = proof.get("steps") or []
    introduced: set[str] = {PROBLEM_TOKEN, ANSWER_SLOT_ID}
    for element in initial_state[1:]:
        fact_id, _ = parse_element(element)
        if fact_id is not None:
            introduced.add(fact_id)

    orphans: list[str] = []
    for step in steps:
        for used in _step_uses(step):
            if used not in introduced and used not in orphans:
                orphans.append(used)
        for element in (step.get("state") or [])[1:]:
            fact_id, _ = parse_element(element)
            if fact_id is not None:
                introduced.add(fact_id)
    return orphans


def validate_proof_references(proof: dict) -> list[str]:
    """Validate id discipline across a whole proof dict.

    Returns a list of human-readable violations (empty when clean):
    - a step's ``uses`` references an id absent from its previous state;
    - an ``F#`` id is renumbered (same content under a new id) or reused
      for different content after being introduced;
    - plus any structural orphans from :func:`find_orphans`.
    """
    errors: list[str] = []
    initial_state = [str(e) for e in (proof.get("initial_state") or [])]
    steps = proof.get("steps") or []

    # Latest content seen under each id. Changing content under an id
    # is a legal `modify` while the id is live in the previous state;
    # reviving a dropped id with different content is a reuse violation.
    id_content: dict[str, str] = {}

    def record_ids(state: list[str]) -> None:
        for element in state[1:]:
            fact_id, _ = parse_element(element)
            if fact_id is not None:
                id_content[fact_id] = content_hash(element)

    record_ids(initial_state)
    prev_state = initial_state
    live_ids = set(id_content)
    for index, step in enumerate(steps):
        step_number = index + 1
        curr_state = [str(e) for e in (step.get("state") or [])]
        validation = validate_step_references(
            step, prev_state, step_number=step_number,
        )
        errors.extend(validation.errors)
        prev_ids = {
            parse_element(e)[0] for e in prev_state[1:]
            if parse_element(e)[0] is not None
        }
        for element in curr_state[1:]:
            fact_id, _ = parse_element(element)
            if fact_id is None:
                continue
            if (
                fact_id in live_ids
                and fact_id not in prev_ids
                and id_content.get(fact_id) != content_hash(element)
            ):
                errors.append(
                    f"step {step_number}: id {fact_id!r} was previously used "
                    "for different content and may not be reused; new facts "
                    "must take the next unused F# id"
                )
            live_ids.add(fact_id)
        record_ids(curr_state)
        prev_state = curr_state

    for orphan in find_orphans(proof):
        errors.append(
            f"uses references {orphan!r}, which is never introduced by the "
            "initial state, the problem, or any prior step"
        )
    return errors


@dataclass
class StepValidation:
    """Result of validating one step's id discipline against the
    previous state. Used by the formalizer feedback path."""
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_step_references(
    step: dict, prev_state: list[str], *, step_number: int,
) -> StepValidation:
    """Check one step's ``uses`` and id stability against ``prev_state``.

    Violations reported:
    - ``uses`` references an id absent from the previous state (and not
      ``"problem"``/the answer slot);
    - a fact's content was renumbered to a different id.
    """
    validation = StepValidation()
    prev_ids = {PROBLEM_TOKEN, ANSWER_SLOT_ID}
    for element in (prev_state or [])[1:]:
        fact_id, _ = parse_element(element)
        if fact_id is not None:
            prev_ids.add(fact_id)
    for used in _step_uses(step):
        if used not in prev_ids:
            validation.errors.append(
                f"step {step_number}: uses references {used!r}, which is not "
                "in the previous state (legal sources: existing F# ids, "
                f"{PROBLEM_TOKEN!r} for problem-given imports)"
            )
    for old_id, new_id in find_renumbering(prev_state or [], step.get("state") or []):
        validation.errors.append(
            f"step {step_number}: fact {old_id!r} was renumbered to "
            f"{new_id!r}; F# ids must stay stable across states"
        )
    return validation
