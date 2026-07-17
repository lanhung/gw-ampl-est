# Phase 5 architecture-selection stack implementation report

## Outcome

The execution-disabled software path for the preregistered final architecture
comparison is implemented. It does not claim a locked training size, fitted
candidate or selected architecture. Stage B and the 65k probe remain upstream
gates.

## Frozen grid

The only four candidates are:

The canonical grid configuration SHA-256 is
`abb3ef575e0f37a8f0150169391efb350b1c53893508bf8ba2505f9219075355`.

| Architecture | Transforms | Conditioner width | Fit source |
|---|---:|---:|---|
| `nsf-t06-w128` | 6 | 128 | three future new fits |
| `nsf-t06-w256` | 6 | 256 | three future new fits |
| `nsf-t10-w128` | 10 | 128 | three future new fits |
| `nsf-t10-w256` | 10 | 256 | reuse three locked-rung probe fits |

Every candidate inherits the same encoders, context, spline bins, optimizer,
epoch budget, early stopping, effective batch, direct-target objective,
training membership, standardizers and 6,144-system validation set. Candidate
construction changes only its identity, transform count and conditioner width.

## Fail-closed execution contract

A future execution authorization must bind:

- the terminal `lock_train_65k` decision and hash;
- Stage A, Stage B and combined-publication identities;
- all three completed probe summaries and checkpoints;
- one membership, input standardizer and target standardizer across reused fits;
- all three generated candidate-model hashes;
- the immutable training commit, wheel and CUDA environment;
- exactly three new architectures and seeds 0, 1 and 2.

The launcher runs one architecture at a time and its three seeds concurrently,
so no more than three fits share the machine. It rejects the reused probe
architecture, limiting the new work to nine fits.

## Selection

The selector requires exactly twelve complete development results. It ranks
architectures by mean validation NLP across all three seeds, never selects a
best seed and uses lower trainable parameter count only for an exact mean tie.
It reports all seed values and records that calibration, SBC and final
evaluation were not accessed. The output is atomic, authorization-path-bound
and cannot automatically open the next phase.

## Current safety state

`configs/execution/phase5_architecture_selection_stack_authorization.yaml`
authorizes implementation and synthetic tests only. It denies Stage B staging
access, scientific data access, architecture fitting, architecture selection,
model tuning, calibration, SBC, final evaluation and GWOSC/GWTC access.
