# Phase 6 calibration/SBC score and statistics release gate

## Outcome

The exact post-materialization release chain is implemented and execution
remains closed.

The implementation resolves the remaining handoff from a future locked
architecture and atomic calibration/SBC publication to:

- six immutable score artifacts: two splits × three retained model seeds;
- three independently fitted calibration maps;
- three independent five-statistic SBC analyses.

No best seed is selected and no calibration map is pooled across seeds.

## Frozen inference contract

Configuration:

`configs/inference/phase6_calibration_sbc_scores.yaml`

Canonical hash:

`47df45922b8db62970e5b0a7c8315c14d95b5fc1ac7e97b030a975fe31d4f2d8`

The contract fixes:

- selected training rung: 131,072;
- retained model seeds: 0, 1 and 2;
- calibration-fit posterior draws per case: 4,096;
- SBC posterior draws per replicate: 1,024;
- deterministic SBC subset seed: 2026071601;
- posterior-draw chunk size: 256;
- physical inference batch size: 32;
- six noncolliding split/model RNG namespaces.

Posterior draw arrays are not persisted. Each artifact contains only the
frozen marginal/joint scores, SBC ranks where applicable, case identities and
the exact model/publication identity.

## Two reviewed releases

The score release packet verifies:

- the terminal 131k size and twelve-result architecture lock;
- all three selected run summaries and `best.ckpt` hashes;
- the selected model configuration and shared preprocessing identities;
- the future 4,096 calibration-fit plus 2,048 SBC publication;
- exact split counts, disjointness and closed publication flags;
- one exact inference wheel, environment and six fresh output paths.

Only a separately hash-bound delegated review may create the checkpoint-
inference authorization. That authorization cannot fit calibration or execute
SBC.

After all six scores complete, the statistics packet verifies:

- exact score and summary hashes;
- identical development case membership across model seeds;
- calibration/SBC group separation;
- split, checkpoint, architecture and inference-commit identity;
- all five frozen SBC rank arrays.

A second delegated review may then authorize exactly three seedwise
calibration/SBC jobs. Checkpoint access remains false in that statistics gate.

## Verification

- focused Phase 6 tests: 17 passed;
- full repository tests: 506 passed, 7 optional-dependency skips;
- Ruff: passed;
- mypy: passed;
- sdist and wheel: passed.

## Boundaries

This implementation did not:

- read a scientific checkpoint;
- read a calibration/SBC publication;
- create a score artifact;
- fit a calibration map;
- execute SBC;
- access final evaluation;
- train or tune a model;
- access GWOSC/GWTC.

All execution requires later exact evidence and delegated review.
