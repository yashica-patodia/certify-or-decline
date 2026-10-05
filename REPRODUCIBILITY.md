# Reproducibility

## Recorded statistics

`python3 -m analysis.reproduce --out results.json` recomputes the recorded statistics without inference or benchmark access. Its input cohort mapping is explicit in `analysis/arms.json`, including which run files contribute to each arm. It fails on missing labels, duplicate IDs, incomplete repeated-run cohorts, or disagreement with the published arm intervals and audit counts.

Problem-cluster intervals use 20,000 resamples, seed 20260819, and the upstream percentile convention. The paired precision comparison uses 20,000 resamples, seed 20260906, Python `random.Random`, and sorted problem IDs. Each sampled ID carries all three observed verifier runs and the first eight draws of the fixed 30-draw voting corpus. The eight-draw policy accepts only eight identical parsed answers, counting missing answers as dissent. Each precision is recomputed over that selector's own accepted set; coverage is not retuned inside the resample.

One generated MATH budget certification has a failed answer-grader call. It is scored incorrect, matching the original aggregate; the redacted row explicitly carries `answer_grader_failed: true`. A failure is not treated as a successful judgment.

The accepted manuscript reported a paired interval of [-8.1, 18.3] points, without a committed generator identifying its exact sampling/order implementation. The explicitly specified calculation released here returns [-8.2, 18.5]; it uses the same observations, point estimate, seed, and resample count. The suggested camera-ready copy uses the released calculation. Both intervals leave the comparison unresolved.

These intervals condition on the recorded runs, draws, and grading labels. They do not quantify new-run variation, systematic scoring errors, or independently validated proof correctness. Reused seeds are not independent-seed replications.

The unconditional and convention-dependent groups can contain observations of the same problem in different runs. Their precision contrast is descriptive, not an estimate of the causal effect of adding a convention.

The 72-output GPQA scoring check compares the existing syntactic option parser against the answer grader on actual generated answers. It covers 60 option letters and 12 option-text matches. It does not establish the grading accuracy of the other 33 free-form answers.

`analysis/rejection_events.json` freezes the 77-file corpus behind the rejection analysis, including the 67 files that contain rejections. It retains only judge roles and issue categories. Some corpus runs do not contribute to a table arm; their full records are unnecessary for reproducing this category count.

`analysis/construction_diagnostic.json` contains per-problem termination phases, judge-presence flags, and answer-slot counters from the separate default-14B diagnostic. It distinguishes its 88 completed cases from the full 100-attempt denominator. `experiments/audit_proofs_*.json` retain external verdicts and defective-step numbers, with benchmark-derived explanations removed.

## New model runs

See [GPU_HOST.md](GPU_HOST.md) for the independent GPU-host setup and rerun
commands. `experiments.run_paper` plans or executes all 19 reported arm
definitions; `experiments.summarize_campaign` summarizes freshly generated,
separately graded runs. These do not replace the frozen numerical records.

Install the runtime with `python -m pip install -e .` and consult `python cli.py --help`. The paper settings are in `configs/paper/{strong,budget,qwen32b,phi4}.yaml`; prompt definitions are in `configs/defaults.yaml`. The original retry and paraphrase overlays are retained under `experiments/frozen_configs/session_2026_08/`. Every role in the portable overlays disables shell and search access.

The overlays pin these model revisions:

| Model | Revision |
| --- | --- |
| Qwen3-14B | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| gpt-oss-20b | `6cee5e81ee83917806bbde320786a8fb61efebee` |
| Qwen3-32B-FP8 | `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df` |
| Phi-4 | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` |

Configure your inference service to expose the recorded model name, including the revision suffix, or explicitly adapt that name to your endpoint. `experiments/serve_vllm.py` and `experiments/make_model_config.py` contain the original serving/configuration utilities. With a locally prepared benchmark file and both inference endpoints running, a strong-arm run can be started with:

```bash
python cli.py benchmark runs/benchmarks/gpqa-fixed-100.json --backend reasoning_agent --no-docker \
  --config configs/paper/strong.yaml --parallel 1 --tag strong \
  --experiment-phase evaluation --trial 1 --no-watch
```

Use the explicit backend and `--no-docker` flags with the paper overlays. Use the portable budget, qwen32b, or phi4 overlay to run those stacks. For the historical strong arm, solver repair/verification ceilings are reconstructed from the recorded call settings in the overlay; the separate frozen retry overlay reproduces the budget-ceiling variant. Do not treat distinct trial labels as distinct seeds: these paper overlays retain the original fixed seed.

Grade the final selected answers and the initial solver answers separately. Set `OWRE_GRADER_ENDPOINT` to the gpt-oss-20b service and use `python cli.py grade <sealed-run.json> --config configs/paper/grader.yaml --target final` and the same command with `--target solver_initial`. The answer-grader settings are retained from the original configuration. Calls to `cli.py grade` require the corresponding grader service; they are not part of CPU-only numerical reproduction.

Numeric telemetry is a token accounting measure; heterogeneous model calls do not have equal FLOPs, latency, or price per token.

## Cohort preparation

The public manifests and [preparation instructions](cohorts/README.md) specify the exact cohorts. Install the runtime, obtain access to [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa), log in with `hf auth login`, and run:

```bash
python -m experiments.prepare_benchmarks --benchmark all
```

The script loads pinned official dataset revisions, selects the recorded rows, restores the recorded GPQA option order by matching content hashes, and verifies every question/answer pair. It writes `gpqa-fixed-100.json`, `aime2025-all.json`, and `math500-fixed-100.json` under `runs/benchmarks/`. It refuses to overwrite existing files and checks all selected cohorts before writing. A separate analysis checkout is unnecessary.

AIME uses all 30 AIME I/II 2025 problems; MATH uses the first 100 MATH-500 test rows. Their original metadata omitted a dataset revision. The release therefore records reconstruction pins, verified against all 130 original question/answer pairs and row metadata. GPQA's revision was recorded in its original preparation. Its manifest was recovered from the 100 original local inputs; independent downloading from its official source requires authorized access. The manifest distinguishes these sources of revision information.

The original benchmark-file hashes are retained in the manifests and numerical records. Prepared files have updated metadata and their own hashes; their evaluated IDs, questions, answers, and order must match the content hashes. The public manifests contain no question text, answer keys, or permutations relative to the source's correct-answer column. Keep generated benchmark files and raw runs local and follow each dataset's access terms.

The release includes recorded statistics, prompts, and code, but omits benchmark-derived generated answers, reasoning, original audit renderings, and deployment endpoints. Fresh inference and external auditing are needed to produce new reasoning chains. A fixed sampling seed does not guarantee identical output across GPU deployments.

## Renamed interfaces

The standalone release uses the `reasoning-eval` command, `reasoning_agent` backend, and `OWRE_*` environment variables. Analysis fields use the `verifier_*` prefix. Original numerical observations, model revisions, configuration hashes, and benchmark hashes are retained.

Two model-facing branding strings were renamed: the agent-loop system preamble and one sentence in the initial-answer grader prompt. These edits can affect newly generated outputs; the released statistics continue to describe the original runs. This snapshot does not establish that every historical run used the final source version.
