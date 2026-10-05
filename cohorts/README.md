# Exact paper cohorts

These manifests allow readers to prepare the evaluated inputs from official datasets. No private companion checkout is needed. They contain IDs, source row indices, and hashes; they contain no questions or answer keys.

| Manifest | Official dataset | Split | Selection |
| --- | --- | --- | --- |
| [gpqa.json](gpqa.json) | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa), `gpqa_diamond` | train | 100 fixed rows: initial 25 and all 75 prespecified extension rows |
| [aime2025.json](aime2025.json) | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025) | train | All 30 rows, in dataset order |
| [math500.json](math500.json) | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | test | First 100 of 500 rows, in dataset order |

## Prepare local inputs

From the repository root:

```bash
python -m pip install -e .
# GPQA requires access through its official dataset page.
hf auth login
python -m experiments.prepare_benchmarks --benchmark all
```

For the two ungated cohorts, run individually:

```bash
python -m experiments.prepare_benchmarks --benchmark aime2025
python -m experiments.prepare_benchmarks --benchmark math500
```

The output files are `runs/benchmarks/gpqa-fixed-100.json`, `runs/benchmarks/aime2025-all.json`, and `runs/benchmarks/math500-fixed-100.json`. This directory is ignored by Git. The script refuses to overwrite existing files. Keep benchmark text and keys local.

GPQA options are ordered by matching each source option's exact UTF-8 SHA-256 digest to the digest for its displayed letter. This avoids publishing an ordering relative to the source's correct-answer column, which would expose the gold letter. The answer is then derived from the authorized source and checked using the joint question/answer digest. Duplicate distractor text is allowed.

AIME and MATH retain the exact source question strings and string representations of the answers. No whitespace normalization is applied. GPQA retains the source question followed by `\n\nOptions:\n`, then `A) ...` through `D) ...`, separated by one newline. Source-option whitespace is preserved.

GPQA's revision was recorded in the original preparation. AIME and MATH's original metadata did not record a revision; their release pins were resolved later and checked against every selected original question, answer, and row metadata field. The `revision_status` and `revision_note` fields make this distinction explicit.

The original file hash identifies the historical input artifact. New preparation updates metadata and the GPQA batch-field name, so its file hash differs. Ordered IDs, source row indices, questions, and answers are checked individually against the original cohort.

## Check an existing local cohort

This uses only the standard library, without network or model calls:

```bash
python -m experiments.prepare_benchmarks --benchmark gpqa \
  --check-existing gpqa=runs/benchmarks/gpqa-fixed-100.json
```

Use `--benchmark all` with one `--check-existing BENCHMARK=PATH` argument per cohort to check all three files. `python -m analysis.verify_release` checks manifest structure and agreement between cohort IDs and the public numerical runs without benchmark access.

## Authorized offline source exports

The preparation script also accepts `--source-json BENCHMARK=PATH` for an authorized local export of a pinned official split. The UTF-8 JSON object must contain `dataset`, `revision`, `split`, and `rows`. `rows` is the complete split in original order, including the source's original columns. Export it with the same `datasets.load_dataset` name, configuration, split, and revision recorded in the corresponding manifest.

```bash
python -m experiments.prepare_benchmarks --benchmark gpqa \
  --source-json gpqa=/private/path/gpqa-official-split.json
```

The export and prepared benchmarks are local inputs; they must not be committed. All selected source content hashes are checked before files are written.
