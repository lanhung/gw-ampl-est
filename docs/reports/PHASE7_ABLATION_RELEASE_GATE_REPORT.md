# Phase 7 terminal ablation release-gate report

## Outcome

The implementation-only release chain for the six preregistered terminal
input-ablation fits is complete at:

`93ca300366cdc00b8d407fc18badd312d4946844`

It does not authorize a scientific fit.

## Frozen future execution set

- locked training rung: 131,072 physical systems;
- views: `gw_only`, `em_only`;
- seeds: 0, 1, 2;
- exact fit count: 6;
- maximum concurrent fits: 3;
- data-loader workers per fit: 4;
- selected architecture, optimizer, effective batch and budget unchanged;
- final-evaluation commitment remains sealed.

The non-authorizing packet must bind the terminal-size decision, the
twelve-result architecture decision, selected primary and ablation model
hashes, the exact 131k preparation and standardizers, corrected and terminal
publication identities, immutable wheel/environment and six fresh output
paths. A separate SHA-256-bound delegated review is required to create the
runtime YAML.

## Verification

- focused Phase 7 ablation tests: 14 passed;
- full pytest: 518 passed, 8 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy: 73 source files passed;
- sdist and wheel: passed;
- full-repository Ruff: the previously recorded 18 Phase 0 legacy findings
  remain and were not represented as a repository-wide pass.

## Safety boundary

Completed:

- release packet builder;
- delegated-review authorization builder;
- exact six-output identity allocation;
- terminal/runtime authorization-field compatibility;
- drift and closed-boundary tests.

Failed:

- no Phase 7 implementation or synthetic-fixture gate failed;
- no new runtime or scientific failure was observed.

Deferred and still unauthorized:

- completion and decision of the active 131k probe;
- nine new architecture fits and architecture selection;
- six ablation fits;
- calibration/SBC materialization and execution;
- final materialization, unsealing and inference;
- real-noise and GWOSC/GWTC access.

No scientific data, checkpoint, metric, calibration map or final record was
opened while implementing this gate.
