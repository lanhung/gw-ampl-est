# Phase 5 terminal 131k architecture-stack report

## Outcome

The existing four-architecture, three-seed grid is now executable in software
at the future terminal 131k lock while every scientific execution flag remains
false. The historical corrected-65k architecture implementation was not
rewritten.

The implementation checkpoint is
`2689119d0526c82f8145c0424741e56a048e96df`; the exact built wheel SHA-256 is
`fc10efa29ba129f19ab3874d88a6cab4c0840045fa1ef8b5b102ca91f8c9231f`.

## Terminal adapter

The adapter requires one of the two preregistered terminal decisions, exact
131,072-member and 6,144-validation publication identities, all three terminal
probe summaries/checkpoints and one common standardizer identity. It reuses the
three `nsf-t10-w256` probe fits and permits only nine new fits:

- `nsf-t06-w128`, seeds 0/1/2;
- `nsf-t06-w256`, seeds 0/1/2;
- `nsf-t10-w128`, seeds 0/1/2.

Selection remains the three-seed mean core-validation NLP with lower trainable
parameter count as the exact tie-break. No best seed is selected. Tail,
calibration, SBC and final evaluation cannot affect architecture choice.

## Verification

- Phase 5 focused tests: 11 passed;
- full local suite: 378 passed, 7 optional dependency skips;
- Ruff and mypy passed for 67 source files and the three terminal scripts;
- sdist and wheel build passed.

All new authorization values are false for data/checkpoint access, fitting,
selection and downstream science. A later exact gate is required after the
terminal learning-curve decision exists.
