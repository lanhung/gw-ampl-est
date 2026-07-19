# Phase 7 RC.6 ablation-training stack

## Outcome

The future GW-only and EM-only training paths are implemented and fail closed.
This is software-release evidence only. No scientific data, checkpoint,
optimizer, calibration/SBC product or final-evaluation case was opened.

## Frozen semantics

The implementation reads the unchanged `1.1.0-rc.6` analysis addendum at
canonical hash
`7e0e252f0a972e0b0ad2fe8f93f74f1f0172639a6fb258fc7a953be5fb7973e1`.
It implements exactly:

- `gw_only`: retain whitened strain, detector masks, observed GW timing and
  lens-family condition; zero all EM scalar values/masks, astrometry and
  modality masks;
- `em_only`: retain EM observations and lens-family condition; zero strain,
  detector masks and the observed GW time-difference value/mask.

The primary 65k input standardizer is applied before either view. Both models
reuse the selected primary architecture, optimizer, effective batch, epoch
budget, early stopping and development set. Targets and group membership never
change. Each view has its own model-configuration hash, and seeds 0, 1 and 2
are all retained.

## Fail-closed execution design

A future execution authorization must bind:

- the terminal `lock_train_65k` result and selected-architecture decision;
- the atomic corrected 65k train and unchanged validation identities;
- the primary rung membership and standardizer artifact;
- the finalized evaluation commitment;
- an immutable training commit, wheel and CUDA environment;
- exactly two views, three seeds and six output identities.

The launcher runs at most three seeds concurrently and processes the two views
sequentially. The result collector requires all six results to share the same
data, environment, membership, standardizers and commitment, while requiring
distinct GW-only and EM-only model identities. It reports three-seed mean
development NLP and never selects a seed or tunes another architecture.

The current gate is
`configs/execution/phase7_ablation_training_stack_authorization.yaml`; every
scientific execution, checkpoint, calibration/SBC, final-evaluation and
GWOSC/GWTC flag is false.

## Verification

- focused ablation tests: 8 passed;
- full local pytest: 343 passed, 7 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy: 79 source files passed;
- sdist and wheel build: passed;
- non-executing runner returns an implementation-ready/optimizer-blocked plan;
- the implementation authorization is rejected by the scientific execution
  gate before any publication path is indexed.

## Remaining gate

No ablation fit may run until the terminal training-size result locks 65k and
the 12-result development-only architecture comparison locks one architecture.
That later authorization cannot open calibration, SBC or final evaluation.
