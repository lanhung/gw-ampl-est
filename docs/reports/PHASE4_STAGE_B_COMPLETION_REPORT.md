# Phase 4 Stage B completion report

## Outcome

Stage B passed and atomically published the exact authorized direct-target
training extension. The frozen runner produced exactly 32,768 accepted
physical systems in 256 complete shards of 128. It generated no validation,
calibration, SBC or final-evaluation system.

The independent post-publisher verifier passed against the execution result,
both atomic parent manifests and all 256 complete shard markers. No optimizer
was authorized or started.

## Frozen identities

- RC.4 preregistration hash:
  `5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98`;
- generator commit: `2be777e727ef9d8e1a85f89c68966df5d37932b0`;
- orchestration commit: `a198b90cc3ebd695a5b6c277e0843e0e19919b18`;
- parent run: `phase4-stage-b-2be777e727ef-6a4f106f9640`;
- Stage B dataset:
  `gwlens-v2-2.0.0-alpha.3-d19597390506b6f8-train-extension`;
- combined reference: `phase4-train-65k-2be777e727ef-6a4f106f9640`.

## Publication evidence

| Evidence | Result |
| --- | ---: |
| Stage B accepted systems | 32,768 |
| Stage B complete shards | 256 |
| Pairs per shard | 128 |
| Cumulative train systems | 65,536 |
| Frozen validation systems | 6,144 |
| Stage B publication bytes | 39,441,798,884 |
| Remaining free bytes | 223,772,577,792 |
| Stage B parent manifest SHA-256 | `b4d7df6300d0919f148b98fd8ce658216bdfa64752026dc9477321874e31f0da` |
| Stage B publication-tree SHA-256 | `373590aa01001a20e0631e672245dba050447cef0e03d45f185645476d735ee1` |
| Combined manifest SHA-256 | `753ace3d2fe475f1279b3bd8560005017f4e75a822fa951d94f9ada60eb3eca4` |

Generation began at `2026-07-17T10:10:29Z`; the execution result was
published at `2026-07-18T18:53:58Z`. The elapsed interval was 32.725 hours,
or about 1,001 accepted systems per elapsed hour. The conservative launch
projection was 33.373 active hours.

## Scientific and grouping validation

- proposal and evaluation distributions are identical;
- every importance weight is exactly one;
- Stage A and Stage B source, lens, physical-system, pair and noise IDs are
  disjoint;
- Stage B and validation physical-system groups are disjoint;
- the combined manifest contains exactly 65,536 unique training systems;
- the combined reference preserves the strict nested train ladder;
- the Stage A validation publication remains the only 6,144-system
  development set;
- no GWOSC/GWTC product was accessed.

The independent closeout JSON has status
`passed_independent_stage_b_closeout`. Its first invocation stopped before
artifact validation because the old remote environment did not expose the new
training package; rerunning the same read-only verifier with the synchronized
repository `src` directory on `PYTHONPATH` passed. This did not modify either
publication.

Local closeout verification passed 313 tests with seven documented optional-
dependency skips, maintained-scope Ruff, mypy over 57 source files, sdist and
wheel builds, all seven evidence hashes and JSON parsing. The repository-wide
Ruff command continues to report only the 18 previously documented immutable
Phase-0 script findings; the CI-maintained scope passes.

## Boundaries and next gate

This closeout does not authorize the 65k optimizer. Architecture selection,
calibration, SBC, final evaluation, extension beyond 65,536, real noise and
GWOSC/GWTC remain closed.

The next gate must bind the Stage A, Stage B and combined manifest hashes, the
completed 32k learning-curve decision, finalized evaluation commitment, exact
training commit and wheel, model configuration and immutable CUDA environment.
Only then may the three frozen seed-0/1/2 65k probes run and produce the
terminal 32k-to-65k decision.
