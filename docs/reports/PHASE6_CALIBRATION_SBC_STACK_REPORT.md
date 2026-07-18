# Phase 6 calibration/SBC preregistration and software report

## Outcome

Downstream preregistration `1.1.0-rc.5` is frozen with canonical hash
`033b996930c93e7e4a9881fc3de49bb85cf4be96fcbd890bf2543b46368c9d8e`.
Pure split-conformal calibration, independent SBC ranks, exact discrete
histogram expectations and Holm correction are implemented and tested on
synthetic fixtures.

No calibration-fit or SBC system was generated or read. No selected checkpoint
was opened, no calibration map was fitted scientifically and no final-evaluation
case was accessed.

## Verification

- full local suite: 276 passed, 6 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed for 52 source files;
- sdist and wheel: built successfully;
- canonical preregistration hash: reproduced exactly;
- the future runner rejects any SBC artifact that does not contain exactly
  1,024 frozen replicates.

## Implemented contracts

- empirical two-sided marginal PIT scores;
- flow-log-density HPD inclusion scores;
- noninterpolated finite-sample conformal order statistics;
- global and balanced eight-cell maps;
- independent map application without refitting;
- deterministic selection of 1,024 SBC systems;
- ranks for primary, secondary, sum, difference and joint density;
- deterministic tie handling;
- exact 20-bin discrete-uniform expectations;
- Pearson chi-square p-values from a tested integer/half-integer gamma
  recurrence, without an undeclared base-package dependency, and Holm
  familywise correction;
- an identity-bound future statistics runner that refuses existing outputs.

The future runner accepts separate, hashed calibration-fit and SBC score
artifacts. Its code path fits maps only from the calibration artifact and uses
SBC only for rank tests and independent calibrated coverage.

## Claim boundary

The method calibrates credible-region levels. It does not transform posterior
samples or claim a new analytic calibrated density. Tail-specific maps are not
fit. All three selected-architecture seeds will be retained and calibrated
separately after a later authorization.
