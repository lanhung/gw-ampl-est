# Phase 7 legacy SIS point-regression stress-control stack

## Outcome

The inherited PDF-era SIS point regressor now has a read-only, fail-closed
verification stack. This is implementation evidence only. No authorized
legacy execution result was created, no checkpoint was deserialized, and no v2
scientific or final-evaluation record was opened.

## Frozen legacy identity

The verifier binds the already audited immutable assets under
`/root/autodl-tmp/tmp`:

- checkpoint SHA-256
  `c558afbc638b7c04e1889b76b375d777129736c174831ba4477f461489e39bd1`;
- saved validation-prediction SHA-256
  `64f85960c6ada5cd7ad85c3f9e9e017e5e57beba544f27be162805531a072023`;
- exactly 500 unique model-selected validation rows;
- generator SHA-256 `ae3879fb...` and training-entry SHA-256 `10777f88...`.

The future read-only job hashes the opaque checkpoint without loading it,
streams the saved CSV, recomputes MAE, RMSE, MAPE and Pearson correlation, and
checks the SIS signed-magnification identity for truth and predictions. It
also proves size, mtime and inode identity did not change during verification.
Any small result must be written atomically under the new AutoDL project root;
the legacy roots are never writable.

## Claim boundary

This baseline is permanently described as:

- SIS-only point regression;
- one synthetic ET detector with stationary Gaussian design noise;
- a model-selected validation partition, not an independent test;
- a point prediction, not a posterior or calibrated-coverage result;
- an out-of-domain historical stress control, not a matched H1/L1/V1
  competitor.

Applying this checkpoint to v2 final data is forbidden because its input,
detector, lens-family and inferential contracts do not match. Its historical
metrics may contextualize earlier work but cannot support the new method's
scientific performance claims.

## Verification

- focused synthetic tests: 5 passed;
- full local pytest: 353 passed, 7 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy: 83 source files passed;
- sdist and wheel build: passed;
- dry command reports legacy asset reading blocked;
- synthetic fixtures reproduce all metrics and reject hash, duplicate-ID,
  SIS-identity and output-root drift;
- checkpoint bytes are never deserialized or modified.

## Remaining gate

A later read-only execution authorization must bind the merged verifier
commit/wheel, exact legacy hashes, immutable before/after stat identity and one
new-project evidence output. It does not depend on or unblock the active 65k
probe and cannot access final evaluation.
