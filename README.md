# Open-weight reasoning evaluation

This repository contains the reasoning runtime, the experiment and analysis code used for this study, and redacted records for GPQA-100, AIME-2025, and the first 100 MATH-500 problems. Certification means acceptance under configured LLM judge and filter rules. These experiments provide no human- or proof-kernel-established ground truth for the reasoning chains.

## Reproduce the numerical results

Python 3.12 or newer is sufficient. Run from the repository root:

```bash
python3 -m analysis.reproduce --out results.json
python3 -m analysis.verify_release
```

These commands use the Python standard library and make no network or model calls. The first recomputes all 19 reported arms' coverage and precision, the 11 problem-cluster intervals, the two certification types, the paired GPQA voting comparison, token accounting, construction diagnostics, rejection frequencies, and auditor agreement. It checks those results against the committed numerical artifacts. The second checks file hashes, the cohort manifests, and the redaction schema of the numerical run rows. The GPQA budget initial-answer baseline has two graded runs (200 attempts); its verifier result pools four runs (400 attempts). This distinction is retained in the output.

To regenerate the figure, install `.[figures]` and run `python -m analysis.figure --out figures/figure1.pdf`.

The main GPQA result is 92 key-matching answers among 105 certifications, from 300 attempts on 100 problems: 87.6% precision at 35.0% coverage. Of these, unconditional acceptance gives 52/54 key matches; acceptance under recorded conventions gives 40/51. At 35% coverage, eight-draw self-consistency gives 29/35. The paired comparison has a broad interval and establishes neither superiority nor equivalence.

`analysis/paper_results.json` contains a computed report. `experiments/public_runs/` contains the redacted rows and numerical reference artifacts. The `proof_soundness` names inherited from upstream refer to LLM-assessed acceptance of rendered chains, not established mathematical soundness.

## Run new experiments

[GPU_HOST.md](GPU_HOST.md) gives the independent-host workflow: GPU
installation, pinned serving commands, all 19 experiment arms, initial and
final answer grading, 30-draw voting, and open-weight proof audits. Use
`python -m experiments.run_paper --list` to inspect the arms and repeat counts.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python cli.py --help
python -m experiments.serve_vllm --help
```

Prepare the exact problem cohorts with:

```bash
# After obtaining GPQA access and logging in to Hugging Face:
python -m experiments.prepare_benchmarks --benchmark all
```

[cohorts/README.md](cohorts/README.md) documents the 230 ordered problem IDs, source row indices, dataset pins, content hashes, and GPQA option ordering. The script downloads from the official sources, checks the hashes, and writes benchmark files under the ignored `runs/benchmarks/` directory. No companion repository is required. AIME and MATH-500 can also be prepared individually without GPQA access.

The runtime supports OpenAI-compatible inference endpoints. The four overlays in `configs/paper/` pin the recorded model revisions and inference settings. Set the appropriate endpoint variables:

```bash
export OWRE_MODEL_ENDPOINT=http://127.0.0.1:8000/v1
export OWRE_FORMALIZER_ENDPOINT=http://127.0.0.1:8001/v1
python cli.py benchmark --help
```

The final command shows the CLI options. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for dataset preparation and the distinction between reproducing recorded statistics and generating new model outputs. Serving the models requires suitable GPU hardware and model dependencies. The portable overlays were reconstructed from call records, with endpoints replaced by environment variables; they are not byte-identical historical configuration files.

## Worked example

[examples/euclid_primes.json](examples/euclid_primes.json) contains the complete sixteen-step state trace for the infinitude of primes from a prior public worked example. It retains states, justifications, and final judge verdicts. At step 2, a pedantry filter overrides a judge's objection to an assumption-label change. This is an illustration from the prior workflow, not a benchmark observation in the present open-weight study. Example licensing is in [DATA.md](DATA.md).

## License

The released code is licensed under the [Apache License 2.0](LICENSE). See [DATA.md](DATA.md) for data and example licensing.
