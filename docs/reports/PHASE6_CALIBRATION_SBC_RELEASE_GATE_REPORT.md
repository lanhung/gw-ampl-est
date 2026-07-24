# Phase 6 calibration/SBC materialization release gate

## Outcome

The exact release and authorization builder for the frozen calibration-fit and
SBC development pools is implemented without generating data or opening a
checkpoint.

The non-authorizing release packet requires:

- a valid terminal 65k-to-131k decision;
- the frozen twelve-result, three-seed architecture lock;
- immutable Stage A, Stage B, combined-65k, waveform-correction,
  terminal-increment, combined-131k and development-tail references;
- an exact non-editable generator wheel and environment lock;
- exactly 4,096 calibration-fit plus 2,048 SBC systems in 48 shards.

A separate delegated-review hash is required before the builder can emit the
runtime authorization. The generated authorization keeps calibration fitting,
SBC statistics, checkpoint access, final evaluation, model tuning and
GWOSC/GWTC false.

## Safety

This implementation:

- does not generate an official identity or pair;
- does not read any scientific publication or checkpoint;
- does not execute calibration, SBC or final evaluation;
- does not change the direct-target distribution, split seeds, counts or
  numerical-validity rule;
- rejects terminal-reference, decision, wheel, review-scope and downstream-gate
  drift.

Execution remains closed until the terminal probe and architecture selection
finish, the exact wheel passes on AutoDL and a future packet receives delegated
review.

## Verification

- Phase 5/6 focused tests: 31 passed;
- full maintained suite: 501 passed, 7 optional-dependency skips;
- Ruff: passed;
- mypy: passed;
- sdist and wheel: passed.
