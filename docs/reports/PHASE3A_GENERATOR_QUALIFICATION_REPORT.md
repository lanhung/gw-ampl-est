# Phase 3A generator qualification report

Status: **blocked before microbenchmark**.

## Authorized scope

Human authorization permits exactly 4,096 non-scientific qualification pairs
in 32 shards of 128. Full production, training, calibration, scientific tests,
real noise and GWOSC/GWTC access remain closed.

## Initial RC.2 checks and blocker

- started from clean checkpoint
  `2bb1ea43cb50f0fbaf8eea9f88750a90603b596a` on the required branch;
- verified ancestry from `5f4d75698ccbd131710fa3705b677db5fa765c9c`;
- recomputed the then-current RC.2 preregistration hash
  `a7d475150b1c01d8e539a3fd5eb8d83f2ce696c5d78125f4c435c7519803aef1`;
- verified the then-current authorizing commit
  `a7be12c5c0dce7c0570911749f6e431c2033c020` and every denial flag;
- verified no Phase 3A staging or publication existed on AutoDL;
- recorded AutoDL free space of 342,407,393,280 bytes, above the prelaunch
  gate, without creating output directories;
- audited every master-prompt section against code, tests and required
  execution evidence.

### RC.2 hard blocker

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

## Final RC.4 execution state

- accepted microbenchmark pairs: 0;
- qualification pairs: 0;
- complete shards: 0;
- published datasets: 0;
- waveform boundary gate: executed and failed all four cases;
- source-plane boundary gate: executed and passed all 992 comparisons;
- Galkin convergence, whitening and runtime qualification: not executed;
- throughput and full-production projections: not measured.

No throughput or qualification-performance result is inferred from the
unexecuted microbenchmark and dataset run.

## RC.2 verification at its stop

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

## RC.2 required human decision (resolved by RC.3/RC.4)

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

## RC.4 waveform-boundary gate

The frozen pre-execution generator commit
`a2b8a02b4631e86c39e1b682e4424ecc2f2c5ca9` failed all four deterministic
waveform boundary cases. The 8-second/32-second aligned-reference differences
were 0.0338--0.0609 versus the frozen `1e-5` maximum. Edge-energy fractions
were `6.19e-5`--`3.04e-4` versus the frozen `1e-6` maximum. All products were
finite, so the failure is containment/reference consistency rather than NaN or
Inf.

This hard failure prevents the 32-accepted-pair microbenchmark. No qualification
shard or dataset publication was created. Criteria were not relaxed after the
result. Phase 3A requires a separately reviewed waveform-window revision before
restarting every pre-execution gate.

Follow-up read-only diagnostics are recorded in
`results/phase3a/waveform_boundary_diagnostic.json`. They exclude a simple
alignment or scalar-amplitude explanation, quantify 32/64/128-second
frequency-grid convergence, and identify an independent factor-2048 clean
strain inverse-transform normalization defect. A proposed replacement contract
is documented for human review in
`docs/reports/PHASE3A_WAVEFORM_WINDOW_CONTRACT_PROPOSAL.md`; it is not an
authorization or a frozen configuration.

## Verification at the RC.4 stop

- local pytest: 132 passed and three optional Lenstronomy tests skipped;
- maintained-scope Ruff: passed;
- mypy: passed for 27 source files;
- package sdist and wheel: built successfully;
- repository-wide Ruff: the same 18 pre-existing Phase 0 manifest-builder
  findings remain; that frozen audit utility was not modified;
- AutoDL production staging and publication roots: empty;
- AutoDL free space at final inspection: 342,403,129,344 bytes;
- no Zarr, Parquet, checkpoint, cache or attempt journal entered Git.
