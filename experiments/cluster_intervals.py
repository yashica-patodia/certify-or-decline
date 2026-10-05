"""Problem-cluster bootstrap intervals for repeated benchmark runs.

The public numerical reproduction entry point is:
    python -m analysis.reproduce --out results.json

This module also retains the original raw-run command below. It requires
local raw runs and their grading files, which are not included in the release.
The GPQA strong certified precision is 87.6%, with a problem-cluster 95%
interval of [77.7, 95.8].

An arm contributes several observations of each problem. Resampling problems
with all their recorded runs attached accounts for this dependence. These
paper runs reuse a fixed sampling seed; they are not independent-seed
replications. The interval conditions on the recorded runs and grading labels.

Percentile intervals are used because precision is a ratio whose selected
set and denominator both vary when problem clusters are resampled.

Raw-run usage:
    python -m experiments.cluster_intervals --out experiments/public_runs/arms_with_ci.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
from collections import defaultdict

from experiments.build_public_runs import grader_map

# label -> globs of the run files that make up the arm.
#
# Nothing in the repository recorded this mapping: the published summaries name
# arms, but which run files each is built from was known only to whoever ran
# the inline scripts. It was recovered by searching run-file groups for the one
# whose (k, n_observations, coverage, precision) reproduces the published arm
# exactly, and is written down here so the next reader does not have to.
#
# The AIME globs deliberately point at runs/aime2025/ and so exclude
# runs/_duplicate_seed_runs/, which holds a second, ungraded seed-3 run of the
# 32B arm.
ARMS: dict[str, list[str]] = {
    "GPQA strong": ["runs/strong-k3/benchmark_strong-t*.json"],
    # the only arm built from two groups, which is why k=4
    "GPQA budget": ["runs/2x2rep/benchmark_2x2rep-budget-t*.json",
                    "runs/2x2rep/benchmark_r2budget-t*.json"],
    "GPQA fswap": ["runs/2x2rep/benchmark_r2fswap-t*.json"],
    "GPQA 32B": ["runs/scale32b/benchmark_scale32b-t*.json"],
    "GPQA Phi-4": ["runs/phi4/benchmark_phi4-budget-t*.json"],
    "GPQA effort=high": ["runs/stronghigh/benchmark_stronghigh-t*.json"],
    # judge-prompt sensitivity: the same pipeline with two paraphrased judge
    # prompts. Arm A has two runs and arm B three, so they are not a matched
    # pair; that asymmetry is why the comparison is reported as a range rather
    # than a difference.
    "GPQA judge-prompt A": ["runs/judgepara/benchmark_judgepara-a-t*.json"],
    "GPQA judge-prompt B": ["runs/judgepara/benchmark_judgepara-b-t*.json"],
    "AIME strong": ["runs/aime2025/benchmark_aime-strong-t*.json"],
    "AIME budget": ["runs/aime2025/benchmark_aime-budget-t*.json"],
    "AIME 32B": ["runs/aime2025/benchmark_aime-32b-t*.json"],
}


def load_arm(patterns: list[str]) -> tuple[dict, list[float], list[str], dict]:
    """(problem id -> [(certified, correct), ...], per-run precision, configs).

    The config hashes are returned for the record, but a difference between
    them is NOT evidence that an arm mixes configurations. compute_config_hash
    folds each role's `endpoint` into the stamp, so serving a role from a
    different box changes the hash while every semantic setting stays
    byte-identical. That is exactly what AIME strong's two hashes are: the
    gpt-oss formalizer used one deployment for runs 1-2 and another for
    run 3. Use experiments/arm_consistency.py, which compares the resolved
    per-role settings, to decide whether an arm is really two arms.
    """
    by_problem: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    paths = sorted(p for pat in patterns for p in glob.glob(pat))
    if not paths:
        raise SystemExit(f"no runs match {patterns}")
    per_seed: list[float] = []
    config_hashes: set[str] = set()
    # A non-answer is either a clean decline or an execution error. Reporting
    # only the total hides which, and the two mean very different things: one
    # is the verifier working, the other is the method failing to run.
    declined = errored = 0
    for path in paths:
        grades = grader_map(path)
        cert = ok = 0
        for row in json.load(open(path)):
            pid = str(row.get("id"))
            certified = bool(row.get("verified")) and not row.get("error")
            correct = bool(grades.get(pid))
            if row.get("error"):
                errored += 1
            elif not row.get("verified"):
                declined += 1
            if row.get("config_hash"):
                config_hashes.add(row["config_hash"])
            by_problem[pid].append((certified, correct))
            if certified:
                cert += 1
                ok += correct
        per_seed.append(round(100 * ok / cert, 1) if cert else float("nan"))
    return (dict(by_problem), per_seed, sorted(config_hashes),
            {"declined": declined, "errored": errored})


def tier_split(pattern: str) -> dict:
    """The unconditional / conditional certification tiers, cluster-bootstrapped.

    Section 8k's central claim is a contrast between these two tiers. Its
    intervals were computed once, outside any committed script, and were far
    narrower than a cluster bootstrap gives -- narrow enough to look separated
    when they are not. They are generated here so the claim is checkable.
    """
    by_problem: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        grades = grader_map(path)
        for row in json.load(open(path)):
            pid = str(row.get("id"))
            certified = bool(row.get("verified")) and not row.get("error")
            by_problem[pid].append((certified,
                                    bool(row.get("verified_unconditionally")),
                                    bool(grades.get(pid))))

    def summarise(keep, label: str, seed: int, resamples: int) -> dict:
        obs = [o for v in by_problem.values() for o in v]
        picked = [o for o in obs if keep(o)]
        rng = random.Random(seed)
        pids = list(by_problem)
        draws = []
        for _ in range(resamples):
            drawn = [by_problem[rng.choice(pids)] for _ in pids]
            sel = [o for v in drawn for o in v if keep(o)]
            if sel:
                draws.append(100 * sum(1 for o in sel if o[2]) / len(sel))
        draws.sort()
        lo = draws[max(0, int(0.025 * len(draws)) - 1)] if draws else float("nan")
        hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))] if draws else float("nan")
        return {
            "tier": label,
            "answered": len(picked),
            "coverage": round(100 * len(picked) / len(obs), 1),
            "precision": round(100 * sum(1 for o in picked if o[2]) / len(picked), 1),
            "precision_ci95_cluster": [round(lo, 1), round(hi, 1)],
        }

    uncond = summarise(lambda o: o[0] and o[1], "unconditional", 20260819, 20000)
    cond = summarise(lambda o: o[0] and not o[1], "under conventions", 20260819, 20000)
    separated = uncond["precision_ci95_cluster"][0] > cond["precision_ci95_cluster"][1]
    return {"tiers": [uncond, cond],
            "intervals_disjoint": separated,
            "note": ("Disjointness is evaluated on the cluster bootstrap, the "
                     "same method used for every arm. An earlier revision "
                     "reported these tiers with much narrower intervals from an "
                     "uncommitted computation, which made them look disjoint "
                     "when they are not.")}


def point(by_problem: dict) -> tuple[float, float, int, int]:
    obs = [o for v in by_problem.values() for o in v]
    cert = [o for o in obs if o[0]]
    coverage = 100 * len(cert) / len(obs) if obs else float("nan")
    precision = 100 * sum(1 for o in cert if o[1]) / len(cert) if cert else float("nan")
    return coverage, precision, len(obs), len(cert)


def cluster_ci(by_problem: dict, resamples: int, seed: int) -> dict:
    rng = random.Random(seed)
    pids = list(by_problem)
    covs: list[float] = []
    precs: list[float] = []
    for _ in range(resamples):
        drawn = [by_problem[rng.choice(pids)] for _ in pids]
        obs = [o for v in drawn for o in v]
        cert = [o for o in obs if o[0]]
        if not obs:
            continue
        covs.append(100 * len(cert) / len(obs))
        # A resample can certify nothing; that yields no precision estimate
        # rather than a zero, which would drag the lower bound down spuriously.
        if cert:
            precs.append(100 * sum(1 for o in cert if o[1]) / len(cert))

    def pct(xs: list[float]) -> list[float]:
        if not xs:
            return [float("nan"), float("nan")]
        xs = sorted(xs)
        lo = xs[max(0, int(0.025 * len(xs)) - 1)]
        hi = xs[min(len(xs) - 1, int(0.975 * len(xs)))]
        return [round(lo, 1), round(hi, 1)]

    return {"coverage_ci95_cluster": pct(covs),
            "precision_ci95_cluster": pct(precs),
            "precision_resamples_usable": len(precs)}


def wilson(k: int, n: int) -> list[float]:
    if not n:
        return [float("nan"), float("nan")]
    z = 1.959963985
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0.0, centre - half), 1),
            round(100 * min(1.0, centre + half), 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    # 20000 rather than 4000: at 4000 the Monte Carlo error on an interval
    # endpoint is a few tenths of a point, which is the same size as the
    # differences the document quotes, and it caused doc/artifact drift.
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # The tier artifact's path is derived from --out by substring replacement,
    # which silently returns the SAME path when the substring is absent and then
    # overwrote the arms file -- headline interval included -- with no error.
    # Checked before anything is written.
    if args.out and "arms_with_ci" not in args.out:
        raise SystemExit(
            f"--out must contain 'arms_with_ci' so the certification-tier "
            f"artifact can be named alongside it; got {args.out!r}")

    out = []
    for label, patterns in ARMS.items():
        by_problem, per_seed, configs, nonanswer = load_arm(patterns)
        coverage, precision, n_obs, n_cert = point(by_problem)
        n_ok = sum(1 for v in by_problem.values() for o in v if o[0] and o[1])
        rec = {
            "label": label,
            "k": len(next(iter(by_problem.values()))),
            "n_problems": len(by_problem),
            "n_observations": n_obs,
            "n_certified": n_cert,
            "n_declined": nonanswer["declined"],
            "n_execution_errors": nonanswer["errored"],
            "coverage": round(coverage, 1),
            "precision": round(precision, 1),
            "config_hashes": [c[:12] for c in configs],
            # NOT a defect flag: the hash encodes the serving endpoint, so
            # this is true whenever an arm's seeds ran on different machines.
            "config_hash_differs": len(configs) > 1,
            "per_seed_precision": per_seed,
            "precision_ci95_wilson_naive": wilson(n_ok, n_cert),
            **cluster_ci(by_problem, args.resamples, args.seed),
            "interval_method": ("cluster bootstrap over problems "
                                f"({args.resamples} resamples); seeds of the "
                                "same problem are correlated, so Wilson on "
                                "pooled observations understates width"),
        }
        out.append(rec)
        if rec["config_hash_differs"]:
            print(f"  note {label}: config hashes differ {rec['config_hashes']} "
                  "-- expected when seeds ran on different machines; see "
                  "arm_consistency.py for the semantic check")
        print(f"{label:14s} n={rec['n_problems']:3d} obs={n_obs:3d} "
              f"cert={n_cert:3d}  coverage {coverage:5.1f}%  "
              f"precision {precision:5.1f}%  cluster "
              f"{rec['precision_ci95_cluster']}  naive "
              f"{rec['precision_ci95_wilson_naive']}")

    tiers = tier_split("runs/strong-k3/benchmark_strong-t*.json")
    print("\nGPQA strong, certification tiers:")
    for t in tiers["tiers"]:
        print(f"  {t['tier']:18s} {t['answered']:3d} answered  coverage "
              f"{t['coverage']:5.1f}%  precision {t['precision']:5.1f}% "
              f"{t['precision_ci95_cluster']}")
    print(f"  intervals separated: {tiers['intervals_disjoint']}")

    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        tier_path = args.out.replace("arms_with_ci", "certification_tiers")
        json.dump(tiers, open(tier_path, "w"), indent=1)
        print(f"wrote {tier_path}")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
