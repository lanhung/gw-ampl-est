# Phase 5 terminal downstream binding-stack report

## Outcome

The post-terminal handoff is implemented without opening scientific data or a
checkpoint. A shared typed adapter now distinguishes the prospective 131,072-
system route from historical 32k/65k authorizations and prevents a terminal
checkpoint from entering calibration or final inference through an old gate.

The adapter accepts exactly the two preregistered terminal outcomes:

- `lock_train_131k_saturated`;
- `lock_train_131k_resource_capped_data_limited`.

Both require 131,072 selected training systems, all three retained probe seeds,
the frozen 6,144-case core validation comparison and no automatic extension.
The architecture handoff requires exactly twelve results: three reused terminal
probe fits and nine new fits, selected by three-seed mean validation NLP without
selecting a best seed.

## Checkpoint safety

Historical calibration/final-inference authorizations that do not contain an
explicit locked rung remain restricted to 32,768 or 65,536. A 131,072-system
checkpoint is accepted only when a later exact authorization explicitly binds:

```yaml
selected_architecture:
  locked_training_rung: 131072
```

The future terminal reference descriptor also binds the logical 131,072 train
count, unchanged 6,144 validation systems, separate 512-case development-tail
pool, exact manifest hashes, direct-target equality and the no-extension cap.

## Verification

- terminal/calibration/final focused tests: 25 passed, one optional Torch skip;
- full local suite: 387 passed, seven optional dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed for 65 source files;
- sdist and wheel build: passed.

All tests used configuration files, decision fixtures and synthetic identities.
No active staging path, scientific publication, model checkpoint, calibration
case, SBC case or final case was opened.

## Remaining gate

This is implementation evidence only. Future Phase 6/7 execution must separately
bind the atomic terminal manifests, terminal and architecture decision hashes,
all three retained checkpoints, the terminal preprocessing identity, immutable
wheel/CUDA environment, finalized final-evaluation commitment and exact output
identities. Calibration/SBC materialization, checkpoint inference, final
materialization/unsealing, ablation fits, reference execution, real noise and
GWOSC/GWTC access all remain closed.
