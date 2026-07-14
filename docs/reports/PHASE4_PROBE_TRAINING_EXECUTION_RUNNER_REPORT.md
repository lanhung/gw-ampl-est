# Phase 4 probe-training execution-runner implementation report

Date: 2026-07-14  
Status: implementation complete; Stage A access and scientific optimization closed

## Outcome

The preregistered probe model now has a complete fail-closed path from an atomic
Stage A publication to six deterministic 16k/32k fits and the paired learning-curve
decision. This work did not read Stage A staging, construct the scientific 16k
membership, start an optimizer, create a scientific checkpoint or inspect final
evaluation.

The model configuration identity after freezing the executable development
protocol is:

`4930651be569725748a0025311ea8b479a217ecab8137be9855a7ee6d2c0377c`

## Integration findings fixed before publication

1. Stage A atomically publishes one parent containing train and validation. The
   original reader expected a child `dataset_manifest.json`, which the immutable
   active generator correctly does not create. The resolver now validates the
   parent manifest, both child run manifests, exact counts, identities, RC.4 hash,
   generator commit, direct-target unit weights and group-disjoint evidence.
2. The original input-standardizer interface accepted a retained sequence of
   prepared examples. At 16k/32k, retaining strain tensors would not be bounded
   memory. Input and target moments are now fitted in streaming metadata-only
   passes; Zarr is not opened during rung preparation.
3. CRPS and coverage functions existed but were not connected to training. Each
   best checkpoint now evaluates the identical 6,144 validation cases with 1,024
   deterministic draws, retaining per-case NLP, CRPS, marginal intervals, joint
   HPD coverage, EM-cell coverage and validation-internal tail views.
4. The frozen 10,000-replicate physical-system paired bootstrap and all-condition
   stopping rule are executable. Its only outcomes are `lock_train_32k` and
   `continue_to_train_65k`; the latter grants no Stage B authorization.
5. The prior candidate named Python 3.11 although AutoDL supplies Python 3.10.12.
   Because the package supports Python 3.9+, the real 3.10.12/CUDA 12.8 environment
   is now the candidate rather than fabricating an unavailable runtime. Zarr is
   locked to 2.18.3, the Python-3.10-compatible Zarr-v2 release already proven by
   the Stage A writer, together with Numcodecs 0.13.1; the originally proposed
   Zarr 2.18.7 / Numcodecs 0.15.1 pair requires Python 3.11 and was rejected by
   the clean environment installation rather than silently approximated.
6. A global random element permutation would have defeated the lazy reader's
   one-shard cache and caused tens of thousands of Parquet/Zarr opens per epoch.
   The epoch-addressed sampler now randomizes shard order and row order within
   each shard, retaining a complete deterministic permutation while keeping I/O
   local.

## Execution structure

- `run_probe_training.py` remains a non-accessing plan unless `--execute` is
  supplied with a later exact authorization.
- A first authorized `--prepare-rung-only` pass hashes deterministic membership,
  parent/namespace manifests and streaming standardizers without opening strain.
- `launch_probe_rung.py` then runs seeds 0, 1 and 2 concurrently on three distinct
  GPUs. Each fit has an independent output identity and atomic checkpoint path.
- Checkpoint resume binds code, environment, data manifests, final-evaluation
  commitment, rung membership, standardizers, seed and model configuration.
- Only noisy strain is opened and whitened using the declared detector ASD. Clean
  and noise products, density provenance and diagnostic labels are not model
  inputs.
- `compare_probe_learning_curve.py` reads development case tables only. It cannot
  read calibration, SBC, IID/OOD/mismatch or final-evaluation products.

The future authorization schema and commands are recorded in
`docs/PHASE4_PROBE_TRAINING_EXECUTION_RUNBOOK.md`.

## Verification

Completed locally:

- full pytest: 245 passed, 6 optional dependency skips;
- focused training tests: 16 passed, 1 local Zarr skip;
- maintained-scope Ruff: passed;
- mypy over `src` and all Phase 4 scripts: passed (58 script/source files);
- sdist and wheel build: passed;
- fail-closed runner plan: passed with `stage_a_accessed=false` and
  `optimizer_started=false`;
- synthetic paired learning-curve decisions exercised both the 32k lock and 65k
  continuation exits.

The optional Parquet/Zarr reader will be rerun only against the future atomic
publication. The wheel built from pushed implementation commit
`c2edcab57beb048b22bb5f1887d556e268025ac9` has SHA-256
`a24442ba2730980e68f291d4748a3e68293a97c0b7a3b4c489749cefd74d869b`.
Its isolated AutoDL Torch/nflows engineering smoke used full 16,384-sample
strain, one forward/backward step, conditional sampling and sampled log density,
and passed twice with replay
SHA-256 `ae4e68c02b2723698ae68c34d28bc673ff0e545cd6cd70ea86b46f13508b702d`.
It used random in-memory inputs and is not a scientific fit or I/O result.

The clean AutoDL environment itself is installed and GPU-visible with Python
3.10.12, Torch 2.10.0+cu128, four RTX 5000 Ada devices and the exact versions in
the model configuration. Its normalized freeze SHA-256 is
`2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`.

## Concurrent Stage A status

At the read-only `2026-07-14T14:18:07Z` snapshot, Stage A had 89/256 complete train
shards (11,392 accepted train systems), 16 partial train shards, no error result and
297,501,061,120 free bytes. Validation, final cross-split validation and atomic
publication had not started. This is progress evidence only.

## Gates that remain closed

- reading Stage A staging or resolving scientific membership;
- scientific 16k/32k optimization;
- learning-curve execution on scientific results;
- model/architecture selection beyond the frozen probe;
- calibration, SBC and all final evaluation;
- Stage B / 65k generation;
- real noise and GWOSC/GWTC.

The next scientific gate can be reviewed only after Stage A is atomically published
and its parent manifest hash is known. It must bind the final training merge commit,
model hash, immutable environment hash and finalized evaluation commitment.
