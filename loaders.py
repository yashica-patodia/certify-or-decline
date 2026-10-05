"""Problem loaders — turn an HLE problem or an ad-hoc question into the
`problem` dicts the pipeline runs on.

Every loader returns a list of dicts. Each dict must have:
    id        — unique string
    question  — the problem text the solver sees
    answer    — expected answer (used only by the naive `correct` check;
                empty string for open-ended questions)
plus any extra fields you want carried through to the saved result.

This module is pure data-wrangling: no LLM calls, no Docker, no argument
parsing. The CLI (cli.py) wires these into the harness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HLE_DATASET_NAME = "skylenage/HLE-Verified"
HLE_DATASET_REVISION = "0bc83643672d4f68a5f89998617a639d85e7318b"
HLE_DATASET_SPLIT = "train"


# ── Humanity's Last Exam ──────────────────────────────────────────

def load_hle(
    category: str | None = None,
    max_questions: int | None = None,
    skip: int = 0,
    skip_images: bool = True,
    ids: list[str] | None = None,
    subset: str = "Gold subset",
    revision: str = HLE_DATASET_REVISION,
) -> list[dict]:
    """Load HLE-Verified (https://arxiv.org/abs/2602.13964).

    By default, only the Gold subset (668 expert-verified clean items) is
    returned. Pass subset=None to load all 2,500 items, or one of:
      "Gold subset"      — 668 fully-verified items
      "Revision subset"  — 1,143 expert-repaired items
      "Uncertain subset" — 689 items with documented uncertainty

    When `ids` is given, the subset filter is ignored — you always get the
    requested items regardless of which subset they belong to.

    Images are skipped by default (this is a text-only pipeline).
    """
    if ids is not None and len(ids) != len(set(ids)):
        raise ValueError("requested HLE ids contain duplicates")
    from datasets import load_dataset  # imported lazily so other CLI
                                       # commands don't need `datasets`

    ds = load_dataset(
        HLE_DATASET_NAME,
        split=HLE_DATASET_SPLIT,
        revision=revision,
    )
    dataset_fingerprint = getattr(ds, "_fingerprint", None)
    dataset_config = getattr(getattr(ds, "info", None), "config_name", None)
    problems = []
    id_set = set(ids) if ids else None
    skipped = 0

    for ex in ds:
        # Image flag and answer_type live inside the nested `json` blob.
        meta = json.loads(ex["json"])

        if id_set is not None:
            if ex["id"] not in id_set:
                continue
        else:
            if subset and ex["Verified_Classes"] != subset:
                continue

        if skip_images and meta.get("image"):
            continue
        if category and ex["category"] != category:
            continue
        if id_set is None:
            skipped += 1
            if skipped <= skip:
                continue

        problems.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "dataset": "HLE-Verified",
            "dataset_name": HLE_DATASET_NAME,
            "dataset_source": HLE_DATASET_NAME,
            "dataset_split": HLE_DATASET_SPLIT,
            "dataset_subset": subset if subset and id_set is None else "ids",
            "dataset_revision": revision,
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_config": dataset_config,
            "answer_type": meta.get("answer_type", ""),
            "category": ex["category"],
            "verified_class": ex["Verified_Classes"],
        })

        if max_questions and len(problems) >= max_questions:
            break

    if ids is not None:
        by_id = {problem["id"]: problem for problem in problems}
        missing = [problem_id for problem_id in ids if problem_id not in by_id]
        if missing:
            raise ValueError(
                "requested HLE ids are absent from the pinned text-only cohort: "
                f"{missing}"
            )
        problems = [by_id[problem_id] for problem_id in ids]
    return problems


# ── Arbitrary questions ───────────────────────────────────────────

def build_question(question: str, problem_id: str = "question") -> list[dict]:
    """A single ad-hoc question. No expected answer."""
    return [{
        "id": problem_id,
        "question": question,
        "answer": "",
        "dataset": "custom",
        "dataset_name": "custom",
    }]


def load_json_benchmark(
    path: str | Path,
    *,
    ids: list[str] | None = None,
    max_questions: int | None = None,
) -> list[dict]:
    """Load a local benchmark object/list without redistributing its text."""
    if ids is not None and len(ids) != len(set(ids)):
        raise ValueError("requested benchmark ids contain duplicates")
    source = Path(path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    payload = json.loads(raw)
    metadata = (payload.get("metadata") or {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        raise ValueError("benchmark metadata must be an object")
    rows = payload.get("problems") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("benchmark JSON must be a list or contain a problems list")
    requested = set(ids) if ids is not None else None
    validated_rows = []
    source_ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"benchmark row {index} is not an object")
        problem_id = str(row.get("id") or "").strip()
        question = row.get("question")
        if not problem_id or not isinstance(question, str) or not question.strip():
            raise ValueError(f"benchmark row {index} needs nonempty id and question")
        if problem_id in source_ids:
            raise ValueError(f"benchmark contains duplicate id: {problem_id}")
        source_ids.add(problem_id)
        validated_rows.append((row, problem_id, question))

    file_sha256 = hashlib.sha256(raw).hexdigest()
    problems = []
    for row, problem_id, question in validated_rows:
        if requested is not None and problem_id not in requested:
            continue
        answer = row.get("answer")
        problem = dict(row)
        problem.update({
            "id": problem_id,
            "question": question,
            "answer": "" if answer is None else str(answer),
            "dataset": str(metadata.get("name") or row.get("dataset") or source.stem),
            "dataset_name": str(
                metadata.get("dataset_name")
                or row.get("dataset_name")
                or metadata.get("name")
                or source.stem
            ),
            "dataset_source": str(
                metadata.get("source") or row.get("dataset_source") or source.name
            ),
            "dataset_revision": str(
                metadata.get("dataset_revision")
                or row.get("dataset_revision")
                or "local"
            ),
            "benchmark_file_sha256": file_sha256,
        })
        for field in ("dataset_fingerprint", "dataset_config", "dataset_subset"):
            value = metadata.get(field) or row.get(field)
            if value is not None:
                problem[field] = value
        problems.append(problem)
        # Don't let max_questions truncate an explicit id request before all
        # requested ids are collected (which would misreport them as not found).
        if max_questions and requested is None and len(problems) >= max_questions:
            break
    if requested is not None:
        found = {problem["id"] for problem in problems}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"benchmark ids not found: {missing}")
        by_id = {problem["id"]: problem for problem in problems}
        problems = [by_id[problem_id] for problem_id in ids]
    return problems
