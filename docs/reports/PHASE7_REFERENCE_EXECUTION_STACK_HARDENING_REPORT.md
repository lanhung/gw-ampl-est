# Phase 7 reference execution-stack hardening report

## Outcome

The frozen RC.7 non-neural reference now has a reusable metadata-only execution
core. No scientific training record, validation record, final case, checkpoint
or calibration artifact was opened. Reference execution remains unauthorized.

## Corrected integration contract

The earlier report said the published reader retained the exact Parquet
`em_cell` label, but `metadata_example()` still returned `None`. That label is
required because RC.7 selects exactly 256 neighbors from the same adopted lens
family and the same one of eight EM cells. This was report/code drift, found
before scientific reference execution.

One typed `_partition_em_cell()` accessor now serves metadata-only reads,
calibration/SBC reads and full examples. It rejects missing/non-string labels.
The label stays beside `PreparedExample` as nondeployable bookkeeping and is
not part of the model-input collator.

## Execution core

`ReferenceBankIndex` now:

- consumes only metadata examples transformed by the selected training-rung
  `InputStandardizer`;
- rejects any opened GW strain, missing family/cell label or duplicate ID;
- groups the bank by exact `(lens_family, em_cell)`;
- has an order-invariant SHA-256 identity;
- vectorizes squared-Euclidean distances within the one eligible stratum;
- selects exactly 256 neighbors by stable `(distance, physical_system_id)`;
- requires query-bank group disjointness in the execution surface;
- preserves the frozen weighting, Scott bandwidth, covariance floor and
  deterministic 4,096-draw sampler.

The per-case score surface computes KDE log probability, CRPS, 50/80/90/95%
central marginal coverage, descriptive joint central coverage and interval
width. Draws are held only for one case and are not persisted.

## Verification

- full pytest: 335 passed, 7 optional-dependency skips;
- focused reference/reader tests: 13 passed;
- maintained-scope Ruff: passed;
- mypy: passed;
- package build: passed.

Synthetic regressions prove bank-order invariance, exact family/cell filtering,
self/split exclusion, duplicate rejection, sparse-stratum failure, normalized
KDE density, deterministic draws and score replay. A non-optional reader test
proves the metadata path retains the cell without opening Zarr.

## Closed gates

Scientific reference-bank access, validation-reference execution, final-data
materialization/unsealing, final-reference execution, checkpoint access,
calibration refitting, model tuning and GWOSC/GWTC access remain false. A later
exact authorization must bind the locked corrected training rung, selected
preprocessing artifact, query publication, immutable code/environment and
output identities.
