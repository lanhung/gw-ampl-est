# Phase 4 terminal-tail parallel recovery review

## Outcome

Delegated engineering review stopped the original development-tail segment at
2026-07-21T20:02:00Z before its frozen 12-hour worker cap. The already atomic
65,536-system train increment remains valid and read-only. The tail segment had
no complete shard and therefore contributed zero cases to any development or
scientific result.

The stopped tail evidence is retained at:

`/root/autodl-tmp/lensing-4/data_v2/scientific/terminal_131k/interrupted_evidence/tail-resource-projection-stop-20260721T2000Z`

It contains one partial shard, 15 partial cases, 91,839 attempts and 85,707,721
file bytes. It may not be reused, deleted, published, trained on or counted
toward the 512-case development pool.

## Reason for the early fail-close

The first frozen high-absolute-magnification stratum accepted 15 of 91,839
attempts, an observed rate of 0.00016333. At the measured 7.76 attempts per
second, the one-by-128 physical shard was projected to reach only about 55
accepted cases by its 12-hour worker cap. Using the two-sided 95% binomial
upper acceptance-rate bound still projected only about 81 cases. Even at that
upper bound, the probability of reaching 128 before the cap was approximately
1.2e-7.

Waiting for the deterministic cap could not provide useful scientific evidence
and would have consumed roughly nine additional active hours. This is an
execution-partition/resource mismatch, not a failure of the frozen conditional
tail distribution.

## Authorized engineering correction

The recovery preserves:

- preregistration `1.2.0-rc.1` and its canonical hash;
- generator commit `a4e6bac014ccd521d510c97593cb1368e826d5eb`;
- the exact immutable generator wheel;
- direct evaluation-target generation and exact unit weights;
- four development-only strata with exactly 128 accepted cases each;
- the original master root seed and conditional-stratum definitions;
- the terminal 131,072 training-system cap;
- permanent exclusion of the 512 cases from training, architecture selection,
  calibration and final claims.

Only the physical execution partition changes. Each stratum uses 32 atomic
shards of four accepted cases instead of one shard of 128, allowing the
existing 32-worker host to execute the rare conditional rejection streams in
parallel. New tail parent, dataset, ID-prefix, attempt-namespace and combined
reference identities are mandatory. The stopped partial identity remains
immutable.

The recovery runner must pass a fresh release certificate, exact wheel and
environment checks, train-publication hash binding, free-space gates, complete
namespace validation and cross-component grouped-ID validation before atomic
tail and combined-reference publication.

The preflight additionally installs the recovery package from commit
`6c8d717d5c095d8ab881355d01cc10e0ff84db1b` into a separate noneditable target.
Its wheel SHA-256 is
`230ae7c89708aca9f492e59e9e49e3bc0d076aad404afb4a8430c6447baee05c`.
A member-by-member wheel comparison proves that all generator-relevant package
files are byte-identical to the original `a4e6bac...` wheel; only unrelated
downstream modules and the terminal orchestration package are allowed to differ.
The runtime-binding orchestration commit is
`ab8e18934eac23cb73be7f7e9c92ce8cb2a3f598`.

Pre-freeze verification passed 440 tests with seven optional dependency skips;
the eight focused recovery tests passed. Maintained-scope Ruff, mypy over all
67 source files, sdist and wheel builds passed. The exact orchestration commit
is `6c8d717d5c095d8ab881355d01cc10e0ff84db1b`.

## Closed boundaries

The correction authorizes no 64-worker execution, train rewrite, optimizer,
architecture selection, calibration, SBC, final-evaluation access, extension
above 131,072, real noise or GWOSC/GWTC access. Passing recovery only completes
the terminal materialization evidence required for the separately reviewed
131k probe gate.
