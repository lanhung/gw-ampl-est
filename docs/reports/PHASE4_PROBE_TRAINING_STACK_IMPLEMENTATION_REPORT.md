# Phase 4 probe-training stack implementation report

## Outcome

The preregistered probe software is implemented and passes local engineering
checks. Scientific probe training remains closed.

The exact implementation checkpoint is
`19f8dc0621f610825d000f37af333f384a963e55`.

This work ran in parallel with the immutable Stage A generator. It did not
sync code into the active AutoDL checkout, read Stage A staging, train on a
scientific record, save a model checkpoint, select an architecture, fit
calibration or access final evaluation/GWOSC/GWTC.

## Implemented

- manifest-first, bounded-memory indexing of complete published shards;
- refusal of staging and partial-shard roots;
- lazy Parquet record and `noisy.zarr` row reads, with no clean/noise array
  access;
- exact direct-target `q=p`, unit-weight and split checks;
- Bilby-compatible `nfft`, PSD division and time-domain whitening
  normalization without per-event observed-standard-deviation normalization;
- zero-preserving unavailable detector masks;
- image-role astrometry DeepSets inputs, scalar uncertainty/missingness inputs
  and explicit log-absolute-magnification targets;
- training-rung-only, observed-value input scaling that preserves exact zero
  semantics for missing values;
- an explicit analyst-selected SIE/EPL model-hypothesis condition, required by
  the model-conditional estimand without exposing continuous lens truth;
- a shared-weight mask-aware 1D ResNet GW encoder, EM DeepSets/scalar encoder,
  10-transform width-256 conditional rational-quadratic NSF;
- target standardization with the exact density Jacobian;
- model-before-construction seed enforcement, epoch-addressed shuffle order,
  AdamW, clipping and early stopping;
- atomic checkpoint/resume with Python/NumPy/Torch RNG state, immutable code,
  environment, manifest, evaluation-commitment, membership and preprocessing
  identities;
- development NLP, CRPS and coverage metrics;
- a planner that refuses scientific execution without a later authorization,
  atomic Stage A publication and finalized evaluation commitment.

## Important contract findings

The adaptive contract selects the 16,384-system probe by deterministic SHA-256
rank over the complete 32,768-system train rung. The first 16,384 systems
generated are not equivalent, so early scientific training on that prefix is
forbidden.

Phase 3A contains systems with five physical images. The candidate DeepSets
capacity is therefore five rather than four. Censor flags are exposed only for
actually observed astrometry items; raw image IDs and unavailable truth image
counts are not model inputs.

The final-evaluation commitment is still
`unfinalized_design_template`. This is a deliberate current blocker for
scientific training and is enforced by code.

## Verification

- full pytest: 225 passed, 5 optional dependency skips;
- focused Phase 4/training and input-policy tests: 37 passed;
- maintained-scope Ruff: passed;
- mypy: passed for 44 source files;
- sdist and wheel: passed;
- PyTorch `2.10.0+cpu` plus nflows `0.14` in-memory smoke: passed;
- model parameters: 7,796,596;
- smoke tensor length: 16,384 samples;
- optimizer steps: one;
- two independent executions produced replay SHA-256
  `b4a7126c83963e8d05c3014ecc1385bc90226d3c9863ad99aedf7fcc7277523e`;
- smoke evidence SHA-256:
  `dc95569531362d415ed99c2caea478964e4f4286d94b6c4a2db1e12ce17e3989`.

The five skips are the existing optional Lenstronomy/Zarr tests absent from
the lightweight local environment; their authoritative scientific paths were
previously exercised on AutoDL. No AutoDL training-environment claim is made
in this report.

## Current Stage A snapshot

At `2026-07-14T12:52:16Z`, Stage A had 80 complete train shards and 16 partial
train shards: 10,240/32,768 train systems, 31.25% of train and 26.32% of all
304 Stage A shards. The run had no error pattern and retained 309,241,184,256
free bytes. Validation and final publication had not started.

## Next gate

After exact Stage A publication, independently validate 32,768 train plus
6,144 validation systems, resolve the ranked 16k membership, finalize and hash
the deterministic final-evaluation generation commitment, build an immutable
AutoDL GPU training environment, run a published-reader canary, and request a
separate three-seed probe-training authorization. No calibration, SBC or final
evaluation opens with that probe gate.
