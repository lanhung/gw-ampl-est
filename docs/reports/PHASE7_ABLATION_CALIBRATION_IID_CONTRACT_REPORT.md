# Phase 7 ablation calibration and IID contract report

## Outcome

The downstream calibration and final-IID semantics for the two frozen input
ablations are now explicit and machine checked. This is a preregistration and
pure-implementation result only; no scientific checkpoint, calibration case
or final case was opened.

The frozen addendum is:

- version: `1.1.0-rc.8`;
- path:
  `configs/statistics/ablation_calibration_iid_preregistration.yaml`;
- canonical hash:
  `219160f67030bad745b0a4573d78d02f9d0db7536a6490c907196e8570647c9a`.

## Scientific resolution

GW-only and EM-only are separately trained estimators. Each view and each
retained model seed must therefore fit an independent split-conformal
credible-region map on the same 4,096 calibration-fit systems used by the
primary model. The six maps cannot be pooled, and a primary-model map cannot
be applied to an ablation.

The controls may later be evaluated only on the primary model's same 8,192 IID
physical systems. Every comparison is paired by physical-system ID and model
seed. The frozen metrics are NLP per target dimension, both marginal CRPS
values and calibrated marginal interval widths. The paired bootstrap uses
10,000 physical-system resamples and the frozen seed domain
`final_iid_ablation_paired_bootstrap_v1`.

This is a descriptive modality analysis. It has no superiority threshold,
does not select a best seed and cannot trigger retraining, architecture
changes or calibration refitting. Independent ablation SBC, balanced-tail,
cross-family, parameter-OOD, waveform-mismatch and PSD-mismatch analyses are
not declared.

## Implementation

`src/gwlens_mm/training/ablation_evaluation.py` provides:

- a fail-closed RC.8 and authorization validator;
- calibration-fit and IID dataset adapters that apply the view after the
  primary standardizer;
- independent map fitting and exact view/seed/checkpoint identity checks;
- overall, lens-family and EM-cell IID summaries with raw counts and Wilson
  intervals;
- deterministic paired bootstrap comparisons;
- a dry-run plan proving all execution boundaries remain closed.

## Verification

- focused tests: 6 passed;
- full tests: 541 passed, 8 optional-dependency skips;
- maintained Ruff: passed;
- mypy over 77 source files: passed;
- sdist and wheel build: passed.

Only synthetic fixtures were used. The active terminal 131k probe was not
modified, synchronized or inspected for interim scientific metrics.

## Closed boundaries

The implementation authorizes no checkpoint access, calibration-map fitting,
IID unsealing, scientific inference, paired-comparison execution, model
training or tuning, manuscript-claim finalization, real-noise work, or
GWOSC/GWTC access. Exact later releases must bind the completed architecture,
all six ablation checkpoints, the calibration publication, the sealed IID
publication, immutable software and fresh output identities.
