# Phase 7 RC.7 reference-query release-gate report

## Outcome

The exact release-control path for the frozen non-neural RC.7 reference is
implemented and remains execution-disabled.

Implementation commit:

`beaa41be7827990edc57bc4de5253a7e7ed298ea`

No scientific training record, validation record, final-evaluation record,
checkpoint or reference-bank entry was opened. All tests used synthetic
directory and JSON fixtures.

## Pre-execution defects resolved

The prior prospective runner did not match the repository's atomic
publication layout:

- it required a `dataset_manifest.json` inside a dataset child, although the
  atomic common parent owns the only manifest;
- it represented every query role by one child, although the 4,096-case
  balanced-tail diagnostic is four independent 1,024-case stratum children.

The corrected implementation binds every child by dataset ID, child root,
parent root and parent-manifest SHA-256. The four tail children are validated
separately and exposed through a bounded-memory concatenated metadata-only
reader. Duplicate physical-system IDs fail closed.

## Release and review chain

`scripts/phase7/prepare_reference_release.py` creates a non-authorizing packet
for exactly one role:

- `validation`: one 6,144-case child;
- `iid_test`: one 8,192-case child;
- `balanced_tail_diagnostic`: four 1,024-case children.

The packet binds:

- one permitted terminal 131k size decision;
- the twelve-result development-only architecture lock;
- the selected 131k membership and input standardizer;
- the corrected-65k and terminal publication identities;
- the exact query child/parent catalog;
- an immutable wheel and environment;
- a fresh output root.

`scripts/phase7/authorize_reference_query.py` requires a separate delegated
review whose packet hash and complete scope match. A validation review cannot
unseal final data. IID and balanced-tail reviews must explicitly authorize
their final-data role.

The runtime independently repeats parent-child, count, decision,
standardizer, wheel, environment and closed-boundary validation before record
access. It continues to persist only bounded per-case scores and aggregates,
not posterior draws.

## Scientific interpretation

The reference remains a same-family/same-EM-cell selected-prior kNN/KDE
simulation reference. It is not an exact likelihood, a gold posterior or an
importance-sampling correction. SBC, IID and tail coverage remain the primary
calibration and approximation evidence.

## Verification

- focused reference/final-publication tests: 16 passed;
- full local suite: 524 passed, 8 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed over 74 source files;
- sdist and wheel build: passed;
- package/scripts compiled: passed.

The eight skips reflect unavailable optional local dependencies
(`lenstronomy`, `zarr`, `pyarrow` and `torch`) and are not failures of the
reference release path.

## Closed boundaries

This implementation does not authorize:

- scientific reference-bank access;
- validation or final reference execution;
- final-evaluation materialization or unsealing;
- checkpoint access;
- calibration refitting;
- model retraining or tuning;
- architecture or size selection;
- extension above 131,072;
- likelihood-gold or importance-sampling-efficiency claims;
- GWOSC/GWTC access.

The running terminal 131k three-seed probe remains the scientific critical
path. Reference execution can only be reviewed after the terminal size and
architecture decisions exist.
