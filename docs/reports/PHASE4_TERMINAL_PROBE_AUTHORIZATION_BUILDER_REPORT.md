# Phase 4 terminal-probe authorization-builder report

## Outcome

A fail-closed builder now converts two independent post-publication evidence
objects into the future exact terminal-probe authorization:

1. the non-authorizing terminal release packet;
2. a separate delegated-review decision bound to that packet SHA-256.

It has not been run on scientific evidence and has not created an execution
authorization.

## Review contract

The review decision must freeze exactly:

- training rung 131,072;
- seeds 0, 1 and 2;
- publication access, optimizer execution and the terminal comparison;
- the retained corrected-65k checkpoint/output root;
- one distinct new 131k output root;
- model tuning, architecture selection, calibration, SBC, final evaluation,
  extension above 131,072, real noise and GWOSC/GWTC all false.

The scope and closed-boundary key sets must be exact. Unknown keys fail closed.
The release packet must still be review-ready and non-authorizing, and its
independent closeout must carry recomputed tree evidence and the exact
65,536-increment/512-tail/131,072-logical counts.

## Output contract

The builder derives the train-increment, combined-131k and development-tail
publication roots from closeout identities; copies exact wheel, model and
environment evidence from the packet; binds the finalized evaluation
commitment; fixes three concurrent fits and the frozen microbatch geometry;
and keeps every downstream gate closed. Before atomic YAML publication, the
result is passed through the same packet-binding validator called by the
scientific training runtime.

## Verification

Synthetic tests cover a valid two-evidence authorization and fail-closed drift
in packet hash, rung, seeds, downstream flags, output identity, unknown flags
and closeout counts. No AutoDL publication, checkpoint or optimizer was used.

- authorization/release focused suite: 32 passed;
- full local suite: 427 passed, 7 optional dependency skips;
- Ruff: passed;
- mypy over `src` and all Phase 4 scripts: passed;
- sdist and wheel build: passed.

## Safety

The builder does not perform delegated review and cannot run training. A real
review JSON can be created only after atomic materialization, independent
closeout, exact-wheel AutoDL tests and release-packet inspection all pass.
