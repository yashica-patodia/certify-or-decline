# Published data

The release contains numerical observations, anonymous vote classes, issue categories, and model audit labels. It omits benchmark questions, expected answers, generated answers, reasoning text, instantiated model prompts and responses, and benchmark-derived auditor explanations. Static prompt templates needed by the runtime are included. IDs and hashes are retained for readers with authorized benchmark access to join against their local records.

The exception is `examples/euclid_primes.json`, a previously public trace on the non-benchmark question about infinitude of primes. It illustrates the format without redistributing GPQA, AIME, MATH, or HLE items. The example is from the project described in [arXiv:2607.01223](https://arxiv.org/abs/2607.01223) and is shared under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Copyright and attribution are included in the example's metadata. Execution logs were omitted and the release description was updated. The Apache-2.0 license applies to the code.

`MANIFEST.sha256` records the published file hashes. Files under `analysis/` provide a CPU-only reproduction path from the retained numerical evidence. Analysis scripts that require raw inputs remain available for inspecting the recorded procedure.

`cohorts/` publishes ordered IDs, row indices, dataset revision pins, and hashes for all 230 evaluated problems. GPQA option-content hashes restore the displayed ordering for readers with authorized source access. Question/answer pairs use joint hashes, rather than exposing short numeric answers through separate answer hashes. The preparation script obtains benchmark text from its official source and keeps generated files local.

Raw runs and locally generated benchmark files belong under `runs/`, which is ignored by Git. Obtain benchmarks from their official sources and follow their access terms when rerunning or sharing any new data.
