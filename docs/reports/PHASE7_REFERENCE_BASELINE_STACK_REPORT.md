# Phase 7 reference-baseline stack report

## Outcome

The downstream-only reference-baseline addendum is frozen as
`1.1.0-rc.7`, canonical hash
`1df98c89fc418eddfd9ec766cb04311e0f3d9f40836a0d9ba1dd691d6bc1724e`.
It prospectively corrects one non-executable inherited claim and implements a
transparent non-neural simulation reference. No scientific data, checkpoint,
calibration map or final case was accessed.

## Scientific correction

The inherited “likelihood gold joint model” expected DINGO-style importance
weights and a 10% efficiency threshold. The primary NSF defines only
`q(log|mu_1|, log|mu_2| | x)`, whereas the exact simulator likelihood is a
function of the complete source, lens, dynamics and image-selection latent
state. No normalized neural proposal exists on that complete state. The
required full-latent `p/q` is therefore undefined, not merely slow.

RC.7 forbids an exact-likelihood-correction or importance-efficiency claim.
This does not weaken or change the preregistered SBC, IID coverage, tail
coverage, mismatch diagnostics or calibration maps. It removes a test whose
mathematical prerequisites the selected estimator never supplied.

## Implemented reference

`selected_prior_em_timing_knn_kde_v1`:

- uses exactly 256 same-family, same-EM-cell, group-disjoint training-rung
  neighbors;
- uses standardized deployable EM/timing values and masks only;
- excludes GW strain, detector masks, query truth and privileged provenance;
- fixes distance, bandwidth, kernel weighting, Scott KDE bandwidth and a
  `1e-6` covariance floor before validation/final access;
- provides a normalized two-dimensional mixture density and 4,096
  deterministic posterior draws;
- fails closed on insufficient strata, overlap, nonfinite values or singular
  covariance.

The published reader now retains the exact Parquet `em_cell` label beside the
prepared example. It remains nondeployable metadata and is not collated into
the model. This prevents two availability cells with the same modality
signature from being incorrectly pooled by the reference.

## Verification

- focused reference tests: 8 passed;
- combined Phase 7 focused tests: 17 passed;
- full pytest: 303 passed, 6 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed for 71 source files, including the Phase 4 and Phase 7
  execution-disabled scripts;
- sdist and wheel build: passed;
- RC.7 canonical hash and execution-disabled dry-run plan: reproduced.

Synthetic tests prove neighbor-order stability, exact family/cell filtering,
self-exclusion, normalized weights, positive KDE covariance, normalized density
formula, deterministic replay, and invariance to query target/GW changes.

## Closed gates

Reference-bank access, validation execution, final materialization/unsealing,
final baseline execution, checkpoint access, calibration refitting, model
retuning and GWOSC/GWTC access all remain false. RC.7 supplies no scientific
baseline result and does not open final evaluation.
