# Phase 5 terminal architecture launcher recovery report

## Outcome

The implementation-only nine-fit launcher now survives a controller
interruption without silently retraining completed models or overwriting
failed evidence.

## Completed-fit reuse

A fit is reusable only when:

- the launcher result JSON exists;
- the run-local `run_summary.json` exists;
- `best.ckpt` exists;
- both JSON payloads are identical;
- status, architecture and seed match the frozen job;
- calibration/final access and extension-above-131k remain false.

## Partial-fit resume

A partial fit is resumable only when its run directory and `last.ckpt` exist
and neither final JSON exists. The launcher passes that exact checkpoint to
the already typed fit runner. A partial result, summary-only identity,
checkpoint-free directory or mixed final/partial state stops execution.

The launcher records fresh, resumed and reused fit counts. It refuses to
overwrite an existing failed parent summary. A completed parent summary is
returned only after revalidating all nine fit identities.

## Safety

The change is orchestration-only. It does not alter:

- the 131,072-system membership;
- the four-architecture grid;
- seeds, optimizer or early stopping;
- validation metrics or selection;
- calibration, SBC or final-evaluation gates.

Tests use fake processes and synthetic checkpoint bytes only. No scientific
data, checkpoint or metric was opened.

## Verification

- Phase 5 focused tests: 18 passed;
- full suite: 565 passed, 8 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy over 79 source files: passed;
- sdist and wheel build: passed.
