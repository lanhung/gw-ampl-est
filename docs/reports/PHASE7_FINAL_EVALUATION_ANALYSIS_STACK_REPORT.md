# Phase 7 final-evaluation analysis stack report

## Outcome

The downstream-only final-evaluation analysis preregistration is frozen as
`1.1.0-rc.6`, canonical hash
`7e0e252f0a972e0b0ad2fe8f93f74f1f0172639a6fb258fc7a953be5fb7973e1`.
The pure analysis contract and synthetic-fixture tests are implemented. No
final-evaluation case, scientific checkpoint, calibration artifact, or GWOSC/
GWTC product was opened.

## Pre-data contradiction resolved

The immutable final generator configuration described one cross-family cell as
SIE truth analyzed under EPL with fixed slope 2.08. The frozen estimator has a
two-value family condition but no density-slope input, so that analysis could
not be executed as written. This was found before final materialization and
before any final metric existed.

RC.6 leaves the finalized generator commitment byte-identical and maps its
legacy context ID to an executable EPL-family-conditioned posterior that
marginalizes the frozen training slope prior. The two family-marginalized cells
use an exact equal-density SIE/EPL mixture. Truth generation, split counts,
seeds, accepted IDs, and distributions are unchanged.

## Implemented contracts

- exact 20,480-system split arithmetic and cross-split ID disjointness;
- three-seed, 4,096-draw, no-best-seed reporting;
- stable equal-family log-density and exact 2,048+2,048 draw mixture;
- counterfactual family-condition copies that do not modify observations,
  targets, or bookkeeping identities;
- GW-only and EM-only deployable-input views with explicit GW-timing semantics;
- raw-count coverage gates using the frozen floor/three-standard-error rule and
  Wilson 95% intervals;
- mean and sample-standard-deviation aggregation requiring seeds 0, 1, and 2;
- a dry-run CLI with null data/checkpoint/calibration identities and no
  execution entry point.

## Immutable evidence

- final generator commit:
  `bc02054c1f95e7f6cd143fb9dc796ae48f0a15ac`;
- generator configuration hash:
  `11277a2a4c5d233e6f525b3ab5d6ece90c115d818d752849076f3a136e574d66`;
- generation commitment SHA-256:
  `c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`.

Tests reproduce both immutable hashes. Neither source file was modified.

## Verification

- focused Phase 7 tests: 9 passed;
- full pytest: 295 passed, 6 optional dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed for 69 source/script files;
- sdist and wheel build: passed.

## Closed gates

All materialization, unsealing, checkpoint inference, ablation training,
baseline execution, gold-likelihood execution, calibration refitting,
GWOSC/GWTC access, and manuscript-claim finalization flags remain false.

The inherited matched non-neural and gold likelihood baselines still require a
separate executable likelihood specification. Phase 7 scientific execution is
not authorized by this implementation checkpoint.
