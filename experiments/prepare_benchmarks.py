"""Rebuild the paper cohorts from official datasets and text-free manifests.

python -m experiments.prepare_benchmarks --benchmark all
Generated questions and answer keys are written only to the ignored runs/ tree.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("gpqa", "aime2025", "math500")
LETTERS = "ABCD"
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def object_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text_hash(encoded)


def manifest_for(benchmark: str) -> dict:
    manifest = json.loads((ROOT / "cohorts" / f"{benchmark}.json").read_text())
    require(manifest["schema_version"] == 1, "Unsupported cohort schema")
    require(manifest["benchmark"] == benchmark, "Cohort name mismatch")
    entries = manifest["entries"]
    expected_count = {"gpqa": 100, "aime2025": 30, "math500": 100}[benchmark]
    require(manifest["count"] == len(entries) == expected_count, "Cohort count mismatch")
    require(len({e["id"] for e in entries}) == len(entries), "Duplicate cohort IDs")
    require(len({e["source_row_index_0based"] for e in entries}) == len(entries), "Duplicate source rows")
    require(re.fullmatch(r"[0-9a-f]{40}", manifest["dataset"]["revision"]), "Invalid dataset pin")
    require(HASH_PATTERN.fullmatch(manifest["original_benchmark_file_sha256"]), "Invalid original file hash")
    common = {"id", "source_row_index_0based", "question_sha256", "question_answer_sha256"}
    expected_fields = common | ({"source_question_sha256", "option_sha256", "cohort_batch", "category"}
                                if benchmark == "gpqa" else {"source_row_sha256"})
    for entry in entries:
        require(set(entry) == expected_fields, f"Unexpected manifest fields for {entry['id']}")
        require(re.fullmatch(r"[A-Za-z0-9_-]+", entry["id"]), "Invalid cohort ID")
        index = entry["source_row_index_0based"]
        require(type(index) is int and 0 <= index < manifest["dataset"]["source_row_count"], "Invalid source index")
        for field, value in entry.items():
            if field.endswith("_sha256") and field != "option_sha256":
                require(isinstance(value, str) and HASH_PATTERN.fullmatch(value), "Invalid content hash")
        if benchmark == "gpqa":
            require(set(entry["option_sha256"]) == set(LETTERS), "Missing option hashes")
            require(all(HASH_PATTERN.fullmatch(v) for v in entry["option_sha256"].values()), "Invalid option hash")
            require(entry["cohort_batch"] in ("initial_25", "extension_75"), "Invalid cohort batch")
    if benchmark == "gpqa":
        require(Counter(e["cohort_batch"] for e in entries) == {"initial_25": 25, "extension_75": 75}, "GPQA batch sizes differ")
        require(manifest["question_rendering"] == {
            "stem_transform": "identity", "stem_suffix": "\n\nOptions:\n",
            "option_template": "{letter}) {text}", "option_separator": "\n", "option_transform": "identity",
        }, "Unsupported GPQA rendering")
    else:
        require([e["source_row_index_0based"] for e in entries] == list(range(expected_count)), "Dataset order differs")
    return manifest


def check_problem(problem: dict, entry: dict) -> None:
    pid = entry["id"]
    require(problem["id"] == pid, f"Problem ID mismatch: {pid}")
    require(problem["source_row_index_0based"] == entry["source_row_index_0based"], f"Source index mismatch: {pid}")
    question, answer = problem["question"], str(problem["answer"])
    require(text_hash(question) == entry["question_sha256"], f"Question hash mismatch: {pid}")
    require(object_hash({"question": question, "answer": answer}) == entry["question_answer_sha256"],
            f"Question/answer hash mismatch: {pid}")


def reconstruct(manifest: dict, source_rows: list[dict]) -> dict:
    spec = manifest["dataset"]
    require(len(source_rows) == spec["source_row_count"], "Official split row count differs")
    benchmark = manifest["benchmark"]
    problems = []
    for entry in manifest["entries"]:
        pid, index = entry["id"], entry["source_row_index_0based"]
        source = source_rows[index]
        if benchmark == "gpqa":
            stem = source["Question"]
            require(text_hash(stem) == entry["source_question_sha256"], f"GPQA stem mismatch: {pid}")
            candidates = [source["Correct Answer"]] + [source[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
            require(Counter(map(text_hash, candidates)) == Counter(entry["option_sha256"].values()),
                    f"GPQA option set mismatch: {pid}")
            # Content hashes encode displayed order without disclosing the gold letter.
            # Identical distractors are allowed; their positions have identical text.
            by_hash = {text_hash(text): text for text in candidates}
            options = {letter: by_hash[entry["option_sha256"][letter]] for letter in LETTERS}
            question = stem + manifest["question_rendering"]["stem_suffix"] + "\n".join(
                f"{letter}) {options[letter]}" for letter in LETTERS)
            answers = [letter for letter in LETTERS if options[letter] == source["Correct Answer"] and
                       object_hash({"question": question, "answer": letter}) == entry["question_answer_sha256"]]
            require(len(answers) == 1, f"GPQA answer mapping mismatch: {pid}")
            problem = {"id": pid, "question": question, "answer": answers[0], "answer_type": "multiple_choice",
                       "category": entry["category"], "cohort_batch": entry["cohort_batch"]}
        else:
            require(object_hash(source) == entry["source_row_sha256"], f"Official source row mismatch: {pid}")
            problem = {"id": pid, "question": str(source["problem"]), "answer": str(source["answer"])}
            if benchmark == "aime2025":
                problem.update(answer_type="integer", category=source["problem_type"], split=spec["split"])
                require(pid == f"aime2025_{spec['split']}_{int(source['problem_idx']):02d}", f"AIME ID mismatch: {pid}")
            else:
                problem.update(answer_type="free_form", category=source["subject"], level=source["level"],
                               source_unique_id=source["unique_id"])
                require(pid == f"math500_{index:03d}", f"MATH ID mismatch: {pid}")
        problem["source_row_index_0based"] = index
        check_problem(problem, entry)
        problems.append(problem)
    metadata = {"name": Path(manifest["output_file"]).stem, "count": len(problems),
                "dataset_name": spec["name"], "dataset_config": spec["config"], "split": spec["split"],
                "dataset_revision": spec["revision"], "revision_status": spec["revision_status"],
                "source": f"https://huggingface.co/datasets/{spec['name']}", "selection": manifest["selection"],
                "cohort_manifest_sha256": object_hash(manifest), "headline_eligible": True,
                "answer_format": "shuffled_multiple_choice_letter" if benchmark == "gpqa" else "free_form_expression"}
    return {"metadata": metadata, "problems": problems}


def load_source(manifest: dict, offline_path: Path | None) -> list[dict]:
    spec = manifest["dataset"]
    if offline_path:
        payload = json.loads(offline_path.read_text())
        require(isinstance(payload, dict) and {"dataset", "revision", "split", "rows"} <= set(payload),
                "Offline source must contain dataset, revision, split, and rows")
        for field, expected in (("dataset", spec["name"]), ("revision", spec["revision"]), ("split", spec["split"])):
            require(payload[field] == expected, f"Offline source {field} mismatch")
        return payload["rows"]
    from datasets import load_dataset
    dataset = load_dataset(spec["name"], spec["config"], split=spec["split"], revision=spec["revision"])
    return [dict(row) for row in dataset]


def check_existing(manifest: dict, path: Path) -> None:
    payload = json.loads(path.read_text())
    problems = payload["problems"]
    require(len(problems) == manifest["count"], "Local benchmark count mismatch")
    for problem, entry in zip(problems, manifest["entries"]):
        check_problem(problem, entry)
    print(f"{manifest['benchmark']}: checked {len(problems)} ordered IDs, row indices, questions, and answers")


def keyed_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, path = value.partition("=")
        require(separator and key in BENCHMARKS and path, "Use BENCHMARK=PATH")
        require(key not in result, f"Duplicate path for {key}")
        result[key] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=(*BENCHMARKS, "all"), default="all")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/benchmarks")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source-json", action="append", default=[], metavar="BENCHMARK=PATH",
                      help="Use an authorized offline export of the pinned official split")
    mode.add_argument("--check-existing", action="append", default=[], metavar="BENCHMARK=PATH",
                      help="Check local benchmark files against the manifests without downloading")
    args = parser.parse_args()
    benchmarks = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    sources, existing = keyed_paths(args.source_json), keyed_paths(args.check_existing)
    require(set(sources) <= set(benchmarks), "Offline source provided for an unselected benchmark")
    if existing:
        require(set(existing) == set(benchmarks), "Provide one existing file per selected benchmark")
        for benchmark in benchmarks:
            check_existing(manifest_for(benchmark), existing[benchmark])
        return
    # Check all destinations and reconstruct all cohorts before writing any files.
    # Download/authentication or content mismatches cannot leave a partial cohort set.
    manifests = [manifest_for(benchmark) for benchmark in benchmarks]
    paths = [args.out_dir / manifest["output_file"] for manifest in manifests]
    require(all(not path.exists() for path in paths), "Refusing to overwrite an existing benchmark file")
    payloads = [reconstruct(manifest, load_source(manifest, sources.get(manifest["benchmark"]))) for manifest in manifests]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for path, payload in zip(paths, payloads):
            with path.open("x", encoding="utf-8") as handle:
                written.append(path)
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    except BaseException:
        for path in written:
            path.unlink()
        raise
    for path, manifest in zip(paths, manifests):
        print(f"{manifest['benchmark']}: wrote {manifest['count']} problems to {path}; SHA-256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Preparation failed: {exc}") from None
