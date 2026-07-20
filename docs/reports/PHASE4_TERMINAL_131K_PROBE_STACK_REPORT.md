# Phase 4 terminal 131k probe-stack implementation report

## Outcome

The future terminal probe software is implementation-complete and remains
execution-closed. No active Stage A/B/terminal array or scientific checkpoint
was opened, and no optimizer or learning-curve decision ran.

The frozen implementation checkpoint is
`77257c3d4871937883eebd330fb8496246a85ff4`; the built wheel SHA-256 is
`dea53afc08609789ea6c1ac066ed411bf1aad135434cf33f86b5e46e3f92e0ad`.

## Implemented contracts

- fail-closed resolution of the atomic corrected-65k, terminal increment,
  logical 131k and four-namespace development-tail publications;
- bounded-memory concatenation of exactly 131,072 unique training systems;
- unchanged 6,144-system core validation reader;
- three-seed, from-scratch 131k probe preparation and launcher;
- evaluation of retained 65k and new 131k checkpoints on the same 512 tail
  cases, exactly 128 in each frozen stratum;
- a 10,000-replicate paired core-validation NLP comparison using the frozen
  seed domain;
- mandatory raw coverage, EM-cell and tail diagnostics that do not override
  the preregistered resource-cap outcome;
- only two terminal decisions, both locking exactly 131,072 and forbidding
  automatic extension.

## Verification

- focused terminal/rung tests: 13 passed;
- full local suite: 376 passed, 7 optional dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed for 66 source files and all new orchestration scripts;
- sdist and wheel build: passed.

All new tests use synthetic CSV/manifests or the existing authorization-denial
paths. The implementation-only authorization keeps terminal publication access,
retained-checkpoint access, scientific training, decision execution,
architecture selection, calibration, SBC, final evaluation and GWOSC/GWTC
false.

## Next exact gate

After the active materialization atomically publishes, a separate release must
bind the train/tail/combined manifest hashes, exact training commit and wheel,
unchanged model configuration, finalized evaluation commitment and CUDA
environment. Only that later release may open the publications, evaluate the
retained checkpoints or start the 131k optimizer.
