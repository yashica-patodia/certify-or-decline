"""Plan or run the 19 paper arms against separately hosted inference services.

The default is a dry run: print commands without writing files or calling models.
--execute runs evaluation and both answer-grading targets. Trial numbers label
new runs; they retain the recorded seed and do not recreate historical outputs.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_FILES = {
    "gpqa": "gpqa-fixed-100.json",
    "aime2025": "aime2025-all.json",
    "math500": "math500-fixed-100.json",
}
JUDGES = ("initial_state", "citation", "problem_given", "computation",
          "pedantry", "convention_lift", "step_untyped")


@dataclass(frozen=True)
class Arm:
    label: str
    benchmark: str
    stack: str
    change: str = ""


ARMS = {
    "gpqa-strong": Arm("GPQA strong", "gpqa", "strong"),
    "gpqa-budget": Arm("GPQA budget", "gpqa", "budget"),
    "gpqa-fswap": Arm("GPQA fswap", "gpqa", "strong", "default-limits"),
    "gpqa-32b": Arm("GPQA 32B", "gpqa", "qwen32b"),
    "gpqa-phi4": Arm("GPQA Phi-4", "gpqa", "phi4"),
    "gpqa-effort-high": Arm("GPQA effort=high", "gpqa", "strong", "effort-high"),
    "gpqa-prompt-a": Arm("GPQA judge-prompt A", "gpqa", "budget", "prompt-a"),
    "gpqa-prompt-b": Arm("GPQA judge-prompt B", "gpqa", "budget", "prompt-b"),
    "aime-strong": Arm("AIME strong", "aime2025", "strong"),
    "aime-budget": Arm("AIME budget", "aime2025", "budget"),
    "aime-32b": Arm("AIME 32B", "aime2025", "qwen32b"),
    "gpqa-ctl": Arm("GPQA ctl", "gpqa", "budget", "default-limits"),
    "gpqa-fswap-budget": Arm("GPQA fswap+budget", "gpqa", "strong"),
    "gpqa-crossjudge": Arm("GPQA crossjudge", "gpqa", "budget", "crossjudge"),
    "gpqa-minus-completeness": Arm("GPQA minus-completeness", "gpqa", "budget", "minus-completeness"),
    "gpqa-effort-low": Arm("GPQA effort=low", "gpqa", "strong", "effort-low"),
    "math-strong": Arm("MATH strong", "math500", "strong"),
    "math-budget": Arm("MATH budget", "math500", "budget"),
    "math-32b": Arm("MATH 32B", "math500", "qwen32b"),
}


def merge(base: dict, overlay: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        result[key] = (merge(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict)
                       else copy.deepcopy(value))
    return result


def arm_config(arm: Arm) -> dict:
    config = yaml.safe_load((ROOT / "configs/paper" / f"{arm.stack}.yaml").read_text())
    if arm.change == "default-limits":
        config["_limits"].update(max_verify_attempts=3, max_formalizer_invalid_attempts=3)
    elif arm.change.startswith("effort-"):
        config["formalizer"]["reasoning_effort"] = arm.change.removeprefix("effort-")
    elif arm.change.startswith("prompt-"):
        letter = arm.change[-1]
        path = ROOT / "experiments/frozen_configs/session_2026_08" / f"judge_para_{letter}.yaml"
        config = merge(config, yaml.safe_load(path.read_text()))
    elif arm.change == "minus-completeness":
        config = merge(config, yaml.safe_load((ROOT / "configs/minus_completeness.yaml").read_text()))
    elif arm.change == "crossjudge":
        donor = yaml.safe_load((ROOT / "configs/paper/strong.yaml").read_text())["formalizer"]
        for role in JUDGES:
            config[role] = copy.deepcopy(donor)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", choices=["all", *ARMS],
                        help="select arms; repeatable; default: all (dry run)")
    parser.add_argument("--list", action="store_true", help="list settings and repeat counts")
    parser.add_argument("--execute", action="store_true", help="execute the printed model calls")
    parser.add_argument("--trial", type=int, help="run one repeat ordinal instead of all repeats")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max", type=int, help="smoke run on the first N problems; not a full arm")
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "runs/benchmarks")
    parser.add_argument("--campaign", default="paper", help="output namespace under runs/")
    parser.add_argument("--skip-grading", action="store_true", help="run inference only; grade later")
    args = parser.parse_args()
    if args.parallel < 1 or (args.trial is not None and args.trial < 1) or (args.max is not None and args.max < 1):
        parser.error("--parallel, --trial and --max must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.campaign):
        parser.error("--campaign must use letters, digits, underscores or hyphens")
    historical = json.loads((ROOT / "analysis/arms.json").read_text())
    if {arm.label for arm in ARMS.values()} != set(historical):
        parser.error("runner and released arm mapping differ")
    selected = list(ARMS) if not args.arm or "all" in args.arm else list(dict.fromkeys(args.arm))
    if args.list:
        for slug in selected:
            arm = ARMS[slug]
            config = arm_config(arm)
            models = sorted({v["model"] for k, v in config.items() if not k.startswith("_")})
            print(f"{slug}: {arm.label}; repeats={len(historical[arm.label])}; "
                  f"limits={config['_limits']}; models={', '.join(models)}")
        return
    namespace = args.campaign + (f"-smoke{args.max}" if args.max else "")
    output_root = ROOT / "runs" / namespace
    planned = []
    for slug in selected:
        arm = ARMS[slug]
        count = len(historical[arm.label])
        if args.trial is not None and args.trial > count:
            parser.error(f"{slug} has {count} recorded repeats, not {args.trial}")
        trials = [args.trial] if args.trial else range(1, count + 1)
        for trial in trials:
            pointer = output_root / slug / f"trial-{trial}.run-path.txt"
            config_path = output_root / "configs" / f"{slug}.yaml"
            benchmark = args.benchmark_dir.resolve() / BENCHMARK_FILES[arm.benchmark]
            command = [sys.executable, str(ROOT / "cli.py"), "benchmark", str(benchmark),
                       "--backend", "reasoning_agent", "--no-docker", "--config", str(config_path),
                       "--parallel", str(args.parallel), "--tag", f"{namespace}-{slug}-t{trial}",
                       "--experiment-id", f"{namespace}/{slug}",
                       "--experiment-phase", "smoke" if args.max else "evaluation",
                       "--trial", str(trial), "--run-path-file", str(pointer), "--no-watch"]
            if args.max:
                command += ["--max", str(args.max)]
            planned.append((slug, arm, trial, pointer, config_path, benchmark, command))
    # Validate the whole selection before starting any GPU calls.
    if args.execute:
        from experiments.prepare_benchmarks import check_existing, manifest_for
        checked = set()
        for slug, arm, trial, pointer, config_path, benchmark, command in planned:
            if pointer.exists():
                parser.error(f"run pointer already exists: {pointer}; choose a new --campaign")
            if benchmark not in checked:
                check_existing(manifest_for(arm.benchmark), benchmark)
                checked.add(benchmark)
    for slug, arm, trial, pointer, config_path, benchmark, command in planned:
        print(shlex.join(command), flush=True)
        if not args.execute:
            if not args.skip_grading:
                print("  then grade final and solver_initial with configs/paper/grader.yaml")
            continue
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(arm_config(arm), sort_keys=False))
        subprocess.run(command, cwd=ROOT, check=True)
        run_path = Path(pointer.read_text().strip())
        if not args.skip_grading:
            for target in ("final", "solver_initial"):
                output = ROOT / "runs" / f"grade-{namespace}-{target}" / run_path.name
                grade_command = [sys.executable, str(ROOT / "cli.py"), "grade", str(run_path),
                                 "--config", str(ROOT / "configs/paper/grader.yaml"),
                                 "--target", target, "--out", str(output),
                                 "--missing-as-incorrect", "--no-watch"]
                print(shlex.join(grade_command), flush=True)
                subprocess.run(grade_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
