#!/usr/bin/env python3
"""Sample the solver k=10 per problem, to give self-consistency a real curve.

WHY THIS EXISTS

The paper's central comparison comes down to: at the verifier's operating point,
does cheap self-consistency do as well? That has never actually been tested.
With k=3 votes the agreement signal takes only two useful thresholds --
unanimous (63% coverage, 73.0% precision) and >=2/3 (87% coverage, 60.9%) --
and Verifier certifies at 33-37% coverage. Neither baseline point is anywhere
near the verifier's, so "matched coverage" was never matched; the reported
comparison instead intersects the two selected sets, which restricts to
problems both answer and flatters the baseline (92.3% on the intersection
against 73.0% over everything it answers).

k=10 gives ten thresholds, several of them in the 30-40% coverage band, so the
comparison can finally be made where it matters -- in whichever direction it
falls.

FIDELITY

The prompts are not reconstructed. Each problem's solver messages are read back
from the artifacts of a completed run (`call_000_solver/messages.json`), so the
model sees byte-identical input to what the pipeline sent. Decoding parameters
use the recorded sampling recipe in DEFAULTS, including top_k=20. They are not
loaded from the verifier config. Seeds vary by draw index; this does not
establish statistical independence or identical output across GPU hosts.

Usage (on a box serving Qwen3-14B):
    python -m experiments.sample_solver_k10 \
        --artifacts runs/artifacts/benchmark_strong-t1_... \
        --k 10 --out runs/solver_k10.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
from pathlib import Path
import urllib.request

DEFAULTS = {
    "model": "Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18",
    "endpoint": "http://127.0.0.1:8000/v1",
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 8192,
    "base_seed": 20260718,
}


def load_problem_messages(artifacts: str) -> dict[str, list]:
    """The exact system+user messages each solver call received."""
    out: dict[str, list] = {}
    for path in sorted(glob.glob(os.path.join(artifacts, "*", "call_000_solver",
                                              "messages.json"))):
        pid = path.split(os.sep)[-3]
        meta_path = Path(path).with_name("meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            pid = str(meta.get("problem_id") or pid)
        messages = json.load(open(path))
        # Keep only the leading system/user turns: later turns are the agent
        # loop's own protocol traffic, not the problem statement.
        head = []
        for m in messages:
            if m.get("role") in ("system", "user"):
                head.append({"role": m["role"], "content": m.get("content") or ""})
            else:
                break
        if head:
            if pid in out:
                raise SystemExit(f"duplicate solver artifact for problem {pid}")
            out[pid] = head
    return out


def sample_one(args) -> tuple[str, int, str, str]:
    pid, draw, messages, cfg = args
    body = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "top_k": cfg["top_k"],
        "max_tokens": cfg["max_tokens"],
        # A draw's seed depends on its index, not its worker or completion order.
        "seed": cfg["base_seed"] + 1000 * draw,
    }
    req = urllib.request.Request(
        cfg["endpoint"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if cfg.get("api_key_env"):
        key = os.environ.get(cfg["api_key_env"])
        if not key:
            raise SystemExit(f"missing API key environment variable {cfg['api_key_env']}")
        req.add_header("Authorization", f"Bearer {key}")
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        msg = (resp["choices"][0].get("message") or {})
        return pid, draw, (msg.get("content") or ""), ""
    except Exception as exc:
        return pid, draw, "", str(exc)[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True,
                    help="artifacts dir of a completed run (for the prompts)")
    ap.add_argument("--k", type=int, default=10)
    # Draw indices can be split across machines: each box computes a disjoint
    # slice and the slices merge, because a draw's seed is derived from its
    # INDEX (base + 1000*draw), not from its position in the loop.
    ap.add_argument("--draws", default=None,
                    help="inclusive index range, e.g. 10-14; default 0..k-1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--endpoint", default=DEFAULTS["endpoint"])
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--base-seed", type=int, default=DEFAULTS["base_seed"])
    ap.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    ap.add_argument("--top-p", type=float, default=DEFAULTS["top_p"])
    ap.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    ap.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    ap.add_argument("--api-key-env", help="Name of an optional endpoint API-key variable")
    args = ap.parse_args()

    if args.k < 1 or args.parallel < 1 or args.max_tokens < 1:
        ap.error("--k, --parallel and --max-tokens must be positive")
    cfg = {key: getattr(args, key) for key in DEFAULTS}
    cfg["api_key_env"] = args.api_key_env
    if args.api_key_env and not os.environ.get(args.api_key_env):
        ap.error(f"missing API key environment variable {args.api_key_env}")
    problems = load_problem_messages(args.artifacts)
    if not problems:
        raise SystemExit(f"no solver messages under {args.artifacts}")
    print(f"{len(problems)} problems x k={args.k} = "
          f"{len(problems) * args.k} solver draws")

    # resume: never re-pay for draws already on disk
    done: dict = {}
    if os.path.exists(args.out):
        done = json.load(open(args.out))
        if not isinstance(done, dict) or set(done) - set(problems):
            ap.error("existing draws do not match this artifact cohort; use a fresh --out")

    recipe_path = Path(args.out + ".settings.json")
    recipe = {key: value for key, value in cfg.items() if key != "endpoint"}
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        ap.error("sampling settings differ from the existing output; use a fresh --out")
    if done and not recipe_path.exists():
        ap.error("existing draws have no settings record; use a fresh --out")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")

    if args.draws:
        lo, hi = (int(x) for x in args.draws.split("-"))
        if lo < 0 or hi < lo:
            ap.error("--draws must be an increasing nonnegative range")
        draw_ids = list(range(lo, hi + 1))
    else:
        draw_ids = list(range(args.k))
    jobs = [
        (pid, d, msgs, cfg)
        for pid, msgs in problems.items()
        for d in draw_ids
        if str(d) not in (done.get(pid) or {})
    ]
    print(f"{len(jobs)} draws still needed")

    completed = 0
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for pid, draw, text, err in ex.map(sample_one, jobs):
            done.setdefault(pid, {})[str(draw)] = {"text": text, "error": err}
            completed += 1
            if completed % 25 == 0:
                json.dump(done, open(args.out, "w"))
                print(f"  {completed}/{len(jobs)}", flush=True)

    json.dump(done, open(args.out, "w"))
    errors = sum(1 for p in done.values() for d in p.values() if d.get("error"))
    print(f"wrote {args.out}: {len(done)} problems, "
          f"{sum(len(p) for p in done.values())} draws, {errors} errors")


if __name__ == "__main__":
    main()
