"""Summarize freshly executed paper arms using their separately graded answers.

This does not compare new inference with frozen historical labels. Run after
experiments.run_paper; source questions and answers are not written to the report.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

from experiments.build_public_runs import grader_map
from experiments.cluster_intervals import cluster_ci
from experiments.prepare_benchmarks import manifest_for
from experiments.run_paper import ARMS, ROOT


def summarize(paths: list[Path], expected_ids: set[str], resamples: int, seed: int) -> dict:
    clusters = defaultdict(list)
    counts = dict(attempts=0, certified=0, correct=0, declined=0, execution_errors=0,
                  initial_correct=0, unconditional=0, unconditional_correct=0,
                  under_conventions=0, under_conventions_correct=0)
    by_run = []
    seen = set()
    for path in paths:
        if path.resolve() in seen:
            raise ValueError(f"run supplied twice: {path.name}")
        seen.add(path.resolve())
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"not a raw run list: {path.name}")
        ids = [str(row["id"]) for row in rows]
        if len(set(ids)) != len(ids) or set(ids) != expected_ids:
            raise ValueError(f"incomplete or duplicated cohort: {path.name}")
        final = grader_map(str(path), target="final")
        initial = grader_map(str(path), target="solver_initial")
        run_cert = run_correct = 0
        for row in rows:
            pid = str(row["id"])
            if not isinstance(final.get(pid), bool) or not isinstance(initial.get(pid), bool):
                raise ValueError(f"missing Boolean grade: {path.name}/{pid}")
            error = bool(row.get("error"))
            certified = bool(row.get("verified")) and not error
            correct = final[pid]
            counts["attempts"] += 1
            counts["certified"] += certified
            counts["correct"] += certified and correct
            counts["execution_errors"] += error
            counts["declined"] += not error and not certified
            counts["initial_correct"] += initial[pid]
            clusters[pid].append((certified, correct))
            run_cert += certified
            run_correct += certified and correct
            if certified:
                group = "unconditional" if row.get("verified_unconditionally") else "under_conventions"
                counts[group] += 1
                counts[group + "_correct"] += correct
        by_run.append({"run": path.name, "certified": run_cert, "correct": run_correct})
    intervals = cluster_ci(dict(clusters), resamples, seed)
    intervals = {key: [v if math.isfinite(v) else None for v in value]
                 if isinstance(value, list) else value for key, value in intervals.items()}
    total, certified = counts["attempts"], counts["certified"]
    return {**counts, "problems": len(expected_ids), "runs": len(paths),
            "coverage": round(100 * certified / total, 1),
            "precision": round(100 * counts["correct"] / certified, 1) if certified else None,
            "initial_accuracy": round(100 * counts["initial_correct"] / total, 1),
            **intervals, "per_run": by_run}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="paper")
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--out", type=Path, help="default: runs/CAMPAIGN/summary.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.campaign) or args.resamples < 1:
        parser.error("invalid --campaign or nonpositive --resamples")
    report = {"new_inference": True, "bootstrap_resamples": args.resamples,
              "bootstrap_seed": args.seed,
              "note": "Fixed sampling seeds do not make repeats independent; auditor labels are not ground truth.",
              "arms": {}}
    for slug, arm in ARMS.items():
        pointers = sorted((ROOT / "runs" / args.campaign / slug).glob("trial-*.run-path.txt"))
        if not pointers:
            continue
        paths = [Path(pointer.read_text().strip()) for pointer in pointers]
        expected_ids = {entry["id"] for entry in manifest_for(arm.benchmark)["entries"]}
        report["arms"][arm.label] = summarize(paths, expected_ids, args.resamples, args.seed)
    if not report["arms"]:
        parser.error(f"no completed run pointers in runs/{args.campaign}")
    output = args.out or ROOT / "runs" / args.campaign / "summary.json"
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output}: {len(report['arms'])} new-inference arms")


if __name__ == "__main__":
    main()
