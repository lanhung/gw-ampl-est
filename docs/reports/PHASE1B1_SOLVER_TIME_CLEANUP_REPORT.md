# Phase 1B.1 solver-time cleanup report

## Outcome

The narrow post-review cleanup passed. It corrected solver API units and
fixture diagnostics without invoking the smoke generator or changing the
frozen dataset.

## Corrections

`PhysicalImage.arrival_time_dimensionless` was removed because it represented
different units in different adapters. The replacement fields are:

- `fermat_potential_dimensionless: Optional[float]`;
- `arrival_time_seconds: Optional[float]`.

A valid physical image must supply at least one finite coordinate. A valid
solution must expose one common coordinate across all images and return them in
increasing order. The SIS solver supplies its analytic dimensionless Fermat
coordinate only. The Lenstronomy adapter retains that raw coordinate and also
supplies a physical delay in seconds normalized to zero at the earliest image.

The fixture summary now computes `selected_pair_is_first_two` from the actual
first two solver-returned IDs. Verified results are:

| Fixture | Selected pair is first two |
| --- | --- |
| SIS double | true |
| SIE double | true |
| SIE quad, image 0 + image 2 | false |
| EPL quad | true |

## Verification

- local pytest: 96 passed, one optional module skipped because local
  lenstronomy is absent;
- AutoDL pytest: 101 passed, including all optional Lenstronomy contracts and
  the four fixture diagnostic assertions;
- Ruff passed for `src`, `tests`, Phase 1B scripts, and schema generation;
- mypy passed for 17 source files;
- sdist and wheel build passed;
- existing v2 metadata round-trip tests passed;
- pre/post SHA-256 values for frozen `manifest.json`, `records.parquet`, and
  `validation.json` were identical.

## Frozen-artifact boundary

The following remain unchanged:

- dataset ID `gwlens-v2-2.0.0-alpha.2-ae86beab1c132682`;
- generator commit `d7287f8cc800406cc1e727a177fd27d7ca02cddf`;
- authorization commit `a607ddb6844cb8a9cd6761011cb545633cd4fdf1`;
- Zarr arrays, Parquet records, manifest, artifact hashes, and validation
  results.

The updated `results/phase1b/solver_fixtures.json` is post-review solver
evidence and is not part of the frozen dataset manifest.

## Deferred work

No Phase 2 work, model training, scientific data generation, or GWOSC/GWTC
download was performed. Production statistical design remains gated on the
Phase 1B PR merge.
