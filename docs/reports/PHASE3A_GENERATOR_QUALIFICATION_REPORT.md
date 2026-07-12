# Phase 3A generator qualification report

Status: **blocked before generation**.

## Authorized scope

Human authorization permits exactly 4,096 non-scientific qualification pairs
in 32 shards of 128. Full production, training, calibration, scientific tests,
real noise and GWOSC/GWTC access remain closed.

## Completed checks

- started from clean checkpoint
  `2bb1ea43cb50f0fbaf8eea9f88750a90603b596a` on the required branch;
- verified ancestry from `5f4d75698ccbd131710fa3705b677db5fa765c9c`;
- recomputed preregistration hash
  `a7d475150b1c01d8e539a3fd5eb8d83f2ce696c5d78125f4c435c7519803aef1`;
- verified authorizing commit
  `a7be12c5c0dce7c0570911749f6e431c2033c020` and every denial flag;
- verified no Phase 3A staging or publication existed on AutoDL;
- recorded AutoDL free space of 342,407,393,280 bytes, above the prelaunch
  gate, without creating output directories;
- audited every master-prompt section against code, tests and required
  execution evidence.

## Hard blocker

RC.2 does not define an executable normalized source-plane density. It freezes
the proposal as uniform in a solver bounding region conditioned on multiple
images, but gives no bounding region. It freezes the evaluation source plane as
uniform in the multiply-imaged cross-section, but gives no area measure,
caustic/pseudo-caustic convention, numerical algorithm or tolerance.

Those omissions matter because the execution contract requires exact normalized
proposal and evaluation log probabilities. A selected solver search window or
numerical caustic grid would change the implied distribution and importance
weights. The Phase 3A prompt explicitly forbids such an implementation-time
change and requires a hard stop on contradictions.

## Work not executed

- accepted microbenchmark pairs: 0;
- qualification pairs: 0;
- complete shards: 0;
- published datasets: 0;
- waveform, Galkin, whitening and runtime qualification: not executed;
- throughput and full-production projections: not measured.

No scientific or engineering result is inferred from an unexecuted run.

## Verification at stop

- local pytest: 120 passed, two optional Lenstronomy tests skipped because the
  dependency is intentionally absent on Vultr;
- authoritative AutoDL maintained test set: 126 passed, including optional
  Lenstronomy/Galkin contracts; Ruff, mypy and package build passed;
- repository-wide Ruff reached the known frozen Phase 0 manifest-builder
  formatting findings (18 findings). The authoritative maintained-code scope is
  checked separately; the immutable audit builder was not reformatted;
- package builds completed locally and on AutoDL with the previously recorded
  missing-README warning;
- no generated array, Parquet file, Zarr store, checkpoint or attempt journal
  entered the repository.

## Required human decision

A reviewed, versioned preregistration amendment must define:

1. the proposal bounding region as a function of declared lens parameters;
2. the exact conditional source-plane density after multiple-image selection;
3. the evaluation cross-section measure;
4. treatment of tangential caustics, pseudo-caustics and doubles/quads;
5. the deterministic area/sampling algorithm and numerical tolerances;
6. how normalized log densities and importance weights are computed and tested.

After amendment, Phase 3A must restart from the new frozen configuration hash.
This report does not authorize full production or any model training.

## RC.3 follow-up

The human-approved RC.3 finite source square resolved exact preselection
normalization, but its first extreme boundary probe exposed an invalid support
audit expectation: steep singular EPL (`gamma=2.5`) remained multiply imaged at
all eight tested square corners/edge midpoints. RC.3 is therefore superseded
before generation. RC.4 must retain the finite normalized square while labeling
it a truncated benchmark, with primary/reference classification agreement—not
absence of multiple imaging—as the boundary criterion.

## RC.4 source-boundary gate

RC.4 replaces the impossible full-cross-section claim with a finite truncated
benchmark and primary/reference solver agreement. The committed gate evaluated
eight extreme SIE/EPL cases at 124 unique square-boundary points each. All 992
comparisons passed: maximum position difference was
`2.814339822894673e-11 arcsec` and maximum relative magnification difference was
`4.0990360870580904e-10`. Runtime was 461.36 seconds with 16 process workers.
This is solver-contract evidence only; no waveform pair or shard was generated.
