# Run the experiments on an independent GPU host

The evaluation code can run on a Linux GPU host or on a separate client that
reaches its OpenAI-compatible inference services. No original cloud account,
private IP address, Docker image, or companion checkout is required for the
open-weight experiments.

These instructions generate **new model outputs**. Use `analysis.reproduce` to
recompute the released historical statistics. The deployment commands and data
flow have been checked without a CUDA GPU; actual model loading, memory use,
and inference on a new GPU host have not been tested for this release.

## 1. Install and prepare the cohorts

Use a fresh Python 3.12 environment on Linux with a compatible NVIDIA driver.
The serving reference records vLLM 0.25.1, which is pinned in the `gpu` extra.
Follow the [vLLM GPU installation instructions](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
for driver and wheel compatibility. FP8 and MXFP4 checkpoints additionally need
hardware supported by those formats' vLLM kernels. Provide enough GPU memory
for weights, the recorded context length, KV cache, and runtime overhead.
This release does not establish a minimum VRAM requirement.

```bash
git clone https://github.com/yashica-patodia/certify-or-decline.git
cd certify-or-decline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[gpu]'
mkdir -p runs
python -m pip freeze > runs/environment.txt
nvidia-smi > runs/gpu.txt

# Obtain access to GPQA on Hugging Face, then authenticate:
hf auth login
python -m experiments.prepare_benchmarks --benchmark all
```

GPQA requires authorized access. AIME and MATH can be prepared separately with
`--benchmark aime2025` and `--benchmark math500`. The preparation script checks
the exact ordered cohorts against the public manifests. See
[cohorts/README.md](cohorts/README.md). Keep benchmark files and raw outputs local.

## 2. Start the pinned model services

`serve_vllm` prints its command by default. **Add `--execute` to launch it.** It
pins both weights and tokenizer and exposes the revision-suffixed model alias
used by the paper configs.

For the strong arm, start Qwen3-14B and gpt-oss-20b in separate terminals on
separate GPU allocations:

```bash
# Terminal A; activate the environment in each terminal.
CUDA_VISIBLE_DEVICES=0 python -m experiments.serve_vllm qwen3-14b --port 8000 --execute

# Terminal B.
CUDA_VISIBLE_DEVICES=1 python -m experiments.serve_vllm gpt-oss-20b --port 8001 --execute
```

These assignments assume two suitable GPUs. The services may instead run on
separate hosts with endpoints reachable from the evaluation client. A single
large GPU can host both only if each service's explicit memory allocation fits;
the default 90% allocation must not be used for two services sharing one GPU.
The budget arm needs only Qwen3-14B for inference. Its answer-grading stage
needs gpt-oss-20b, which can be started later using `--skip-grading` during
evaluation.

For whole-stack scaling arms, replace the service on port 8000 with one of:

```bash
python -m experiments.serve_vllm qwen3-32b-fp8 --port 8000 --execute
python -m experiments.serve_vllm phi4 --port 8000 --execute
```

Start only the service needed for the selected arm. The FP8 checkpoint is the
paper's 32B model; `qwen3-32b` selects a different BF16 checkpoint. Phi-4
automatically uses a 16,384-token context; the other paper models use 32,768.
Use `--tensor-parallel-size N` to allocate multiple GPUs when needed and record
the change. Reducing context length or changing weights changes the experiment.

## 3. Evaluate and grade the paper arms

In the evaluation terminal, with both services available:

```bash
export OWRE_MODEL_ENDPOINT=http://127.0.0.1:8000/v1
export OWRE_FORMALIZER_ENDPOINT=http://127.0.0.1:8001/v1
export OWRE_GRADER_ENDPOINT=http://127.0.0.1:8001/v1

# List all 19 arms and their repeat counts, models, and retry limits.
python -m experiments.run_paper --list

# Print the planned commands; this makes no model calls or file changes.
python -m experiments.run_paper --arm gpqa-strong --trial 1

# First check the deployment on one problem, in a separate smoke namespace.
python -m experiments.run_paper --arm gpqa-strong --trial 1 --max 1 --execute

# Then generate three full strong-arm runs and grade both answer targets.
python -m experiments.run_paper --arm gpqa-strong --execute
```

The runner validates the input cohort before inference, disables Docker and
external tools, and writes configs and completed-run pointers under
`runs/paper/`. It grades final answers and initial solver answers separately,
using the retained open-weight grader settings. A missing answer is counted
incorrect; model-call failures remain visible in the grade file.

Run other arms by the names in `--list`, after serving their required models.
The runner covers all 19 released arm definitions and matches their recorded
repeat counts. `--arm all` plans the complete campaign, but does not start or
swap model services; execute compatible arms together and change the service
before a different whole-stack model. `--campaign NAME` creates a new output
namespace. Existing run pointers are never overwritten.

Trial numbers are repeat ordinals. The recorded seed stays 20260718, including
the fourth ctl/budget repeat; these are not independent-seed replications.
Prompt A has two repeats, effort=high one, and most other arms three. The
fswap+budget and strong groups have the same role assignment and retry limits.

If grading was deferred, read a pointer and grade explicitly:

```bash
RUN_PATH=$(cat runs/paper/gpqa-strong/trial-1.run-path.txt)
RUN_NAME=$(basename "$RUN_PATH")
python cli.py grade "$RUN_PATH" --config configs/paper/grader.yaml --target final \
  --out "runs/grade-paper-final/$RUN_NAME" --missing-as-incorrect --no-watch
python cli.py grade "$RUN_PATH" --config configs/paper/grader.yaml --target solver_initial \
  --out "runs/grade-paper-solver_initial/$RUN_NAME" --missing-as-incorrect --no-watch
```

To summarize completed full-cohort arms, including problem-cluster intervals,
execution failures and unconditional/convention-dependent counts:

```bash
python -m experiments.summarize_campaign --campaign paper
```

The report is `runs/paper/summary.json`. It rejects incomplete cohorts and
missing grades. All newly run baseline repetitions are summarized; the frozen
paper's GPQA budget baseline instead used only two graded initial-answer runs.
New outputs need not reproduce the historical percentages.

## 4. Sample and score self-consistency

The voting sampler uses the initial solver messages saved by a **new full run**.
The public numerical records omit those messages and cannot be used as prompts.
Obtain the artifact directory from that run's filename:

```bash
RUN_PATH=$(cat runs/paper/gpqa-strong/trial-1.run-path.txt)
RUN_NAME=$(basename "$RUN_PATH" .json)
python -m experiments.sample_solver_k10 --artifacts "runs/artifacts/$RUN_NAME" \
  --k 30 --parallel 1 --endpoint "$OWRE_MODEL_ENDPOINT" --out runs/paper/solver_k30.json

python -m experiments.solver_votes --draws runs/paper/solver_k30.json \
  --benchmark runs/benchmarks/gpqa-fixed-100.json --k 30 --out runs/paper/solver_k30_votes.json
python -m experiments.self_consistency_curve --draws runs/paper/solver_k30.json \
  --benchmark runs/benchmarks/gpqa-fixed-100.json --k 8 --out runs/paper/sc_k8.json
python -m experiments.self_consistency_curve --draws runs/paper/solver_k30.json \
  --benchmark runs/benchmarks/gpqa-fixed-100.json --k 19 --out runs/paper/sc_k19.json
python -m experiments.self_consistency_curve --draws runs/paper/solver_k30.json \
  --benchmark runs/benchmarks/gpqa-fixed-100.json --k 30 --out runs/paper/sc_k30.json
```

Despite its historical filename, the sampler supports any positive `--k`. The
recorded recipe is Qwen3-14B, temperature 0.6, top-p 0.95, **top-k 20**, 8,192
output tokens, and seed `20260718 + 1000 * draw_index`. The verifier overlays
do not explicitly send top-k. The sampler's defaults are not inferred from the
verifier config. It saves a settings sidecar and rejects resuming with changed
settings. A fixed seed does not guarantee identical outputs across hosts.

## 5. Audit locally generated proofs

After final-answer grading, audit all certifications from the three new strong
runs with the open-weight auditors:

```bash
R1=$(cat runs/paper/gpqa-strong/trial-1.run-path.txt)
R2=$(cat runs/paper/gpqa-strong/trial-2.run-path.txt)
R3=$(cat runs/paper/gpqa-strong/trial-3.run-path.txt)
python -m experiments.proof_validity_audit "$R1" "$R2" "$R3" \
  --sample 100000 --graders oss-gptoss,oss-qwen --concurrency 1 \
  --endpoint "oss-gptoss=$OWRE_GRADER_ENDPOINT" \
  --endpoint "oss-qwen=$OWRE_MODEL_ENDPOINT" --out runs/paper/audit_open_weight.json
```

Each certification is identified by its complete run filename and problem ID,
so repeated observations are retained. Use `oss-qwen32b` with a served FP8
endpoint to audit the judge-scale arm. `--grader-config NAME=YAML` can override
an auditor's `audit_grader` settings; a replacement is a new experiment.

This reproduces the **shortened rendering protocol**, which includes the last
state entry per step. It cannot resolve whether a rejection reflects the proof,
the rendering, or an auditor error. The proprietary GPT/Claude audit scripts
also remain callable with `--graders codex,claude`, but require the respective
authenticated CLIs and access to the recorded models. A GPU host alone is
insufficient for those calls, and the Claude `opus` alias was not a checkpoint
pin. The published auditor labels can be analyzed without those services.

## 6. Grader and construction diagnostics

Constructed answer-grader checks can be repeated with:

```bash
python -m experiments.validate_grader --config configs/paper/grader.yaml \
  --benchmark runs/benchmarks/gpqa-fixed-100.json --n 40 --concurrency 1 \
  --out runs/paper/grader_validation_gpqa.json
python -m experiments.validate_grader --config configs/paper/grader.yaml \
  --benchmark runs/benchmarks/aime2025-all.json --n 30 --concurrency 1 \
  --out runs/paper/grader_validation_aime.json
python -m experiments.validate_grader --config configs/paper/grader.yaml \
  --benchmark runs/benchmarks/math500-fixed-100.json --n 30 --concurrency 1 \
  --out runs/paper/grader_validation_math.json
```

For new default-limit construction runs, use the `gpqa-ctl` arm. Its raw
`attempts`, `verdicts`, and per-call artifacts retain construction phases and
judge presence. Summarize a new run with:

```bash
RUN_PATH=$(cat runs/paper/gpqa-ctl/trial-1.run-path.txt)
python -m experiments.diagnose_runs "$RUN_PATH" --out runs/paper/construction.json
```

Pass multiple run paths to aggregate their recorded rejection classes. Add
`--gpqa-scoring` after final-answer grading to compare the syntactic option
parser and LLM grader on actual outputs. Unparsed answers are reported
separately. `experiments.answer_slot_violations --run RUN.json --out PATH`
examines formalizer answer-slot failures. The released separate diagnostic
cohort and 72-output scoring check are frozen numerical records, not independent
human validation or a promise to recover those exact counts from new inference.

## What remains outside a GPU rerun

- Access-controlled GPQA content must be obtained by the reader.
- Historical reasoning chains and complete benchmark prompts are not public;
  newly generated chains are needed for new audits.
- Proprietary auditor calls need separately authorized services and model access.
- There is no Lean/kernel or human-adjudication experiment in this release.
- The release renamed two model-facing strings, as described in
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md). It does not claim byte-identical
  historical inference or identical output on different GPUs.
