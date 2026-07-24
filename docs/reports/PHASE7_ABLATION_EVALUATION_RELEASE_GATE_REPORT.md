# Phase 7 ablation calibration/IID release-gate report

## Outcome

The missing execution-control handoff for the frozen RC.8 ablation analysis is
implemented. It creates no scientific authorization and uses no scientific
artifact.

## Ordered release design

The calibration release can be assembled only after all six GW-only/EM-only
fits and the primary calibration/SBC analysis complete. It validates the
selected 131,072-system architecture, hashes every ablation checkpoint and run
summary, binds the atomic 4,096-case calibration-fit publication and allocates
exactly:

- six immutable calibration score artifacts;
- six independent split-conformal maps, one per view and seed.

The reviewed calibration authorization keeps IID unsealing, comparison,
retraining, SBC and every non-IID diagnostic false.

The IID release can be assembled only after the six calibration results
complete. It verifies every view/seed/checkpoint/map identity, discovers
exactly one 8,192-case IID namespace from the primary final scores, requires
the same namespace for seeds 0, 1 and 2, and allocates exactly:

- six ablation IID score artifacts;
- six same-seed paired-comparison artifacts.

The reviewed IID authorization cannot refit calibration, run SBC or evaluate
tail, cross-family, parameter-OOD, waveform-mismatch or PSD-mismatch cases.
Results cannot select a seed, change architecture or trigger retraining.

## Fail-closed properties

- Release packets are non-authorizing.
- Each stage requires a separate SHA-256-bound delegated review.
- Calibration must finish before IID release assembly.
- Primary and ablation model/architecture identities must agree.
- Every input and output path remains under the AutoDL project root.
- Existing output identities are rejected.
- Exact wheel and environment evidence are mandatory.
- Scientific checkpoints, data and scores remain inaccessible under the
  implementation-only gate.

## Verification

- focused tests: 7 passed;
- full suite: 548 passed, 8 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy over 78 source files: passed;
- sdist and wheel build: passed.

Only synthetic fixtures were used. No checkpoint, calibration-fit case, final
IID case, calibration map or primary score artifact was opened.

## Remaining gate

The active 131k three-seed probe and terminal decision remain the critical
path. Size lock, architecture lock, six ablation fits, primary
calibration/SBC, sealed final materialization and primary IID inference must
all complete before these future releases can be assembled.
