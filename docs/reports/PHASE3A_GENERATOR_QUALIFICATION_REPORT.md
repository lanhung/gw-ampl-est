# Phase 3A generator qualification report

Status: **qualification data published; Phase 3A stopped for human review**.

## Scope and immutable identities

Phase 3A generated engineering qualification data only. It did not train a
model, fit calibration, run a scientific test, access GWOSC/GWTC, generate real
noise, modify the frozen smoke artifact or authorize later phases.

- frozen preregistration: `1.0.0-rc.5`;
- preregistration hash:
  `4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb`;
- authorizing commit:
  `bba0cdd6a750ff367674a85b8722432e613586d8`;
- generator commit:
  `fbcd0616611d9cdf915ef0af030e6061c1be7f59`;
- configuration hash:
  `f8920a89787ea94f634a080b8d4d70a7205f5f382417bd169077a6d562d2145c`;
- schema: `2.0.0-alpha.3`;
- dataset ID: `gwlens-v2-2.0.0-alpha.3-7081b2e8be3a84e1`.

The microbenchmark, interruption run and resumed run all used the same frozen
generator commit and configuration. The microbenchmark dataset ID is separate
and its 32 pairs are excluded from the published 4,096.

## Pre-execution gates

All frozen scientific and numerical gates passed before the formal run.

- Source-plane boundary: 992/992 comparisons passed in 516.35 seconds with
  16 workers. Maximum position difference was
  `2.814339822894673e-11 arcsec`; maximum relative magnification difference was
  `4.0990360870580904e-10`.
- Mass-sheet transform: source scaling, image invariance, signed and absolute
  magnification scaling, Fermat scaling and physical-delay scaling passed.
  Maximum reported absolute error was `4.440892098500626e-16`.
- Waveform boundary: all four RC.5 fixtures passed the 64/128-second comparison,
  exact guards, containment, energy-retention, transform-normalization and
  stored-clean SNR checks. Maximum conditioned relative difference was
  `0.003952665852776242`, below `0.005`.
- Whitening: H1/L1/V1 finite fractions were 1.0 and standard deviations were
  `0.9976085164`, `0.9992541913` and `1.0018185472`. No per-event observed
  standard-deviation normalization was used. Exact PSD files and hashes passed.
- Galkin convergence: 16 balanced cases passed the 4,000 versus 16,000 sample
  comparison; maximum relative difference was `0.011053280820040276`, below
  `0.02`.
- Deployable-input policy: all 32 exact denied fields and five alias fixtures
  were rejected. Truth, selection statistics and importance weights remain
  non-deployable.

Local pre-execution checks were 139 passed with three optional dependency
skips. The authoritative AutoDL environment passed 146 tests. Phase 3A scope
Ruff, mypy and package builds passed locally and remotely.

## Microbenchmark gate

The deterministic microbenchmark passed every gate:

- accepted pairs: 32;
- attempts: 14,349;
- acceptance rate: `0.0022301205658930936`;
- elapsed time: 544.61 seconds;
- accepted pairs/hour: 211.71 with eight workers;
- both lens families and all eight EM cells exercised;
- doubles, triples and quads exercised;
- projected 4,096-pair time: 9.67 hours, below 24 hours;
- projected output: 3,946,818,816 bytes, below 10 GB;
- projected remaining free space: 334,392,419,072 bytes, above 100 GB;
- peak child RSS: 297,288 KiB;
- projected 16-worker RSS: 4,756,608 KiB.

The formal run completed faster than the conservative microbenchmark
projection because the observed aggregate acceptance rate was higher.

## Interruption and deterministic resume

The first invocation completed exactly shards 0, 1 and 2, then stopped
intentionally with 384 accepted pairs. Their tree hashes were recorded before
resume. The second invocation reused the identical run ID, dataset ID, root
seed, code, authorization and configuration.

The first three pre-resume hashes were:

- shard 0: `d49921120d21cf1971724939b37858f3981ac0a90bb455b5d1afba45fbd20bc1`;
- shard 1: `c163c1de53a82f7fc9101342873582c44d26d47c4f3deb0531c96d34fee80715`;
- shard 2: `afa8c2182dbec4596e251b14cbd64377e51562dcaaf0c3189185c4c9a0eef6f6`.

The post-resume file contains the same three lines. Both files have SHA-256
`9af4da78bd83fdec599bc88357196b2157640ad4c217d696d6b7de41c3087816`.
No completed shard was rewritten and no partial shard was published.

## Published qualification measurements

The runner atomically published the dataset at 2026-07-12 13:22:47 UTC.

- accepted pairs: 4,096 exactly;
- complete shards: 32 exactly;
- accepted pairs per shard: 128 exactly;
- attempts: 1,455,699;
- acceptance rate: `0.002813768505714437`;
- active elapsed time: 21,404.39 seconds (5.95 hours);
- accepted pairs/hour: 688.91;
- attempts/hour: 244,833.76;
- EPL accepted/attempted: 2,029/727,987;
- SIE accepted/attempted: 2,067/727,712;
- multiplicities 2/3/4/5: 1,924/591/1,241/340;
- each of the eight EM cells: 512 accepted;
- fewer-than-two-physical-images rejections: 1,026,090;
- fewer-than-two-selected-images rejections: 425,513;
- published bytes: 4,450,694,559;
- shard bytes: 3,944,326,856;
- remaining free bytes: 330,450,604,032;
- published tree SHA-256:
  `25b0a0d45c030c90da7037ac5f820dced4d602019f0ae2b8ce3a9d4a2d9ac133`.

An independent post-publication reread of the complete tree reproduced both
that digest and the recorded 4,450,694,559-byte content count.

The final validator passed exact pair/shard counts, alpha.3 records, unique
pair/source/lens/physical-system/noise IDs, grouped qualification confinement,
physical-image retention, selected-pair identity, deployable-input policy and
exact float32 `noisy = clean + noise` decomposition. The publication manifest
states that scientific, training, calibration and test use are false, full
production is false, and GWOSC/GWTC were not accessed.

## Storage and resource observations

The official staging directory was absent after publication, the published
directory existed with 32 complete and zero partial shards, and measured size
was 4,451,101,072 filesystem bytes. This is below the 10 GB gate. Free space
remained above 330 GB, well above the 100 GB gate.

The runner persisted the microbenchmark peak RSS but did not persist a
continuous peak-RSS counter for the full run. A recorded process snapshot
summed 5,446,536 KiB across the 16 workers; this is a sampled value, not a claim
of an exact run maximum. Sixteen workers were observed near one CPU each, but a
time-integrated CPU-utilization series was not persisted. These are reporting
limitations, not evidence of an exceeded resource gate.

## Superseded and failed attempts

The execution history is retained rather than rewritten:

- RC.2 stopped before generation because the normalized source-plane measure
  was incomplete.
- RC.3 was superseded after its boundary claim failed for steep EPL systems.
- RC.4 passed the corrected source-boundary gate but failed all four frozen
  waveform-boundary fixtures and stopped before the microbenchmark.
- RC.5 resolved the waveform contract under separate human review. An initial
  `3b71e7c...` code freeze was superseded before data generation after remote
  mypy found two NumPy inference issues.

No pair from a superseded microbenchmark or commit entered the final dataset.

## Final quality checks and limitations

After publication, local checks again passed 139 tests with three optional
skips; AutoDL passed 146 tests. Phase 3A code Ruff, mypy and builds passed in
both environments.

The literal repository-wide command `ruff check src tests scripts` still finds
18 pre-existing formatting findings in
`scripts/remote/build_phase0_manifests.py`. That old inventory builder is not
part of the Phase 3A generator. It was not changed after the microbenchmark,
because changing any code would invalidate the frozen generator-commit rule.
This known maintained-scope distinction is disclosed rather than reported as a
full-repository Ruff pass.

The published qualification data use synthetic Gaussian curve-conditioned
noise only. They provide no scientific performance, calibration, coverage,
real-noise or GWTC evidence.

## Projection and next human gate

Linear extrapolation of the measured RC.5 rate to the historical 118,784-pair
plan is about 172.4 active hours and 129.1 GB of published data. This is a
historical resource projection only, not an authorization or recommendation.

Human review may instead preregister a staged scientific plan, for example a
16,384/32,768/65,536 training ladder with fixed disjoint evaluation sets and
predeclared learning-curve stopping rules. Such a revision must use a new
version/hash and must keep the 4,096 qualification pairs outside every
scientific split. Synthetic OOD and real-noise/GWTC work require separate
authorization boundaries.

Phase 3A stops here. Full or staged scientific production, model training,
post-hoc calibration, SBC, scientific IID/OOD tests, real noise, GWOSC/GWTC,
manuscript work and Phase 3B remain unauthorized pending human review.
