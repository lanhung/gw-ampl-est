# Phase 3B.2 proposal A/B count consistency report

Status: **design correction complete; awaiting human review; execution disabled**.

## Scope

Phase 3B.2 resolves one machine-readable count contradiction before PR review.
It generated no pair, did not implement or run proposal-v2, trained no model,
fit no calibration, ran no scientific evaluation and accessed no GWOSC/GWTC
product. Phase 3A code and artifacts and the parent RC.5 configuration remain
unchanged.

## RC.3 identity

- version: `1.1.0-rc.3`;
- canonical configuration hash:
  `6082475631539d3069edacc52f41b37fb8fe725ccd7c6bc9980cc3008795a927`;
- final-evaluation commitment template SHA-256:
  `015e4ee50b78f9bd80f17723ccf084fad22768b04c09ba9da9677b62808b6533`;
- commitment status: unfinalized design template; future generator placeholder
  remains unresolved.

All execution flags remain false and status remains `awaiting_human_review`.

## Resolved A/B count

RC.2 described a 512-pair qualification but also required 512 pairs in each of
two arms. RC.3 freezes the only consistent interpretation:

| Arm | Matched blocks | Pairs/block | Accepted pairs |
|---|---:|---:|---:|
| RC.5 control | 16 | 32 | 512 |
| proposal-v2 candidate | 16 | 32 | 512 |
| **Total** | — | — | **1,024** |

The future authorization hard maximum is exactly 1,024 accepted engineering
pairs across both arms. Neither arm is scientific data.

## Artifact identities and safety

One parent A/B run identity deterministically derives two different dataset
identities. The RC.5 control and proposal-v2 candidate have separate manifests
and checksums under one parent comparison manifest. Environment, workers and
telemetry contracts match. Both arms permanently deny scientific, training,
calibration and test use and are forbidden from every scientific split.

Concrete IDs remain future authorization outputs because the authorization
commit does not yet exist. RC.3 freezes their templates and the requirement
that they be resolved exactly once and differ.

## Conservative resource gate

Both arms are budgeted at the measured Phase 3A RC.5 rate rather than assuming
candidate acceleration:

| Quantity | Projection |
|---|---:|
| control active time | 0.7432 h |
| candidate conservative active time | 0.7432 h |
| combined conservative active time | 1.4864 h |
| combined attempts after per-arm ceiling | 363,926 |
| combined publication bytes | 1,112,673,640 |
| temporary/staging reserve | 20,333,802,092 bytes |
| projected peak | 21,446,475,732 bytes |
| minimum prelaunch free space | 121,446,475,732 bytes |

The hypothetical 2× candidate case would take 1.1148 combined active hours,
but it is unmeasured and cannot reduce the prelaunch resource gate.

## Verification

- full local pytest: 157 passed, with three optional Lenstronomy skips;
- Phase 3B focused tests: 18 passed;
- maintained-scope Ruff: passed;
- mypy: passed for 28 source files;
- source distribution and wheel: built successfully;
- canonical RC.3 hash, commitment SHA-256 and A/B resource arithmetic:
  reproduced exactly.

The build retains the known missing-README warning. The previously documented
Phase 0 repository-wide Ruff findings remain outside this maintained scope.

## Completed, failed and deferred

Completed: explicit A/B arithmetic, distinct artifact identity contract,
engineering-only policy, conservative resource projection, RC.3/hash update,
commitment refresh, documentation and safety tests.

Failed execution: none; Phase 3B.2 executed nothing. The RC.2 count wording is
superseded design history.

Deferred: executable proposal distributions, proposal-v2 implementation, the
1,024-pair A/B qualification, all scientific generation, training,
calibration, SBC/IID/OOD/mismatch evaluation, real noise, GWOSC/GWTC and Phase
3C.

Phase 3B.2 stops here for human review and PR-triggered CI has not been run.
