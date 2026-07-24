# Phase 7 ablation IID aggregate runtime report

## Outcome

The implementation-only RC.8 runtime now closes the final descriptive
aggregation gap. A future reviewed IID execution allocates one aggregate
artifact in addition to six view/seed score artifacts and six same-seed paired
comparisons.

## Fail-closed inputs

The finalizer requires:

- both frozen views, `gw_only` and `em_only`;
- model seeds 0, 1 and 2 for each view;
- all six score artifacts and companion run summaries;
- all six paired-comparison JSON artifacts;
- exact authorized paths and matching SHA-256 identities;
- the terminal 131,072-system architecture identity;
- the reviewed immutable inference commit and environment.

Missing, reused, mismatched or nonfinite evidence stops aggregation.

## Output contract

The aggregate retains all six seed-level comparisons and applies the frozen
RC.8 mean and sample-standard-deviation summary separately for each view.
It selects no best seed, has no superiority gate and cannot trigger
retraining, tuning or architecture changes.

The CLI defaults to dry-run. `--execute` requires the future reviewed IID
authorization and writes atomically to its exact allocated output path.

## Safety

Implementation and tests used synthetic completed-job evidence only. They did
not open:

- a scientific checkpoint;
- calibration-fit or IID records;
- primary scientific score artifacts;
- posterior draws;
- GWOSC or GWTC products.

Calibration refitting, SBC, non-IID ablation analysis and manuscript claim
finalization remain closed.

## Verification

- focused ablation release/runtime tests: 17 passed;
- full test suite: 558 passed, 8 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy: 79 source files passed;
- sdist and wheel build: passed.
