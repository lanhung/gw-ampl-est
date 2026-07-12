# Project state

## Current phase

Human review accepted Phase 3A and PR #4 merged at
`589b6a554d5bf8213c3014b5cb6f3b0e0f4edd5e`. The 4,096-pair qualification
artifact remains permanently non-scientific. Phase 3B.1 has completed its
design-only corrections and is awaiting human review on adaptive
preregistration `1.1.0-rc.2`, canonical hash
`b94e7733d7fbb6f4c9dc4d5842b6a87f29e0515b4047b7b1604bca1438d15805`.

No scientific pair, proposal-v2 qualification pair or model may be generated
or trained. Calibration, SBC, final evaluation, real noise, GWOSC/GWTC and
Phase 3C remain unauthorized. The next action is human review, not execution.

## Completed

- established Vultr as the sole authoritative Git repository;
- configured dedicated key-based AutoDL access and safe sync scripts;
- created isolated AutoDL project directories under `/root/autodl-tmp/lensing-4`;
- reconciled qkzhang, wjx pair-verification and `/tmp` PDF-baseline lineages;
- traced PDF metrics to exact data/code/checkpoint evidence;
- created curated immutable source snapshots and manifests;
- classified legacy datasets and code for retain/reuse/rewrite/exclude decisions;
- documented v2-smoke scope and storage gate.
- defined the authoritative v2 physics and Fourier/Morse conventions;
- implemented and tested the SIS analytic control and general solver protocol;
- implemented fail-closed input policies and grouped split validation;
- implemented the v2 logical metadata schema, provenance, seeds, and manifests;
- accepted ADR-001 for Zarr v2 plus Parquet smoke storage;
- validated an execution-disabled 48-pair smoke specification;
- hardened EM astrometry with explicit physical image IDs;
- aligned detector-noise provenance to every selected image and detector slot;
- replaced the bare timing float with a typed uncertainty-bearing product;
- enforced complete extra-image status and primary-definition semantics;
- added zero-fill/decomposition and complete-manifest validators;
- upgraded the unmaterialized schema to `2.0.0-alpha.2`;
- passed 93 pytest cases, Ruff, mypy, package build, the prior AutoDL SIS
  contract, and the AutoDL alpha.2 metadata contract.
- merged the Phase 1A PR after its GitHub Actions check passed;
- pinned an isolated AutoDL Phase 1B scientific environment;
- numerically validated SIS, SIE+shear, and EPL+shear deterministic fixtures;
- implemented detector-time-aware IMRPhenomXPHM smoke waveform generation;
- generated and atomically published exactly 48 accepted engineering pairs;
- validated Zarr v2 arrays, Parquet records, checksums, policies, six-slot
  noise provenance, grouped IDs, and interruption/resume behavior;
- demonstrated matched-response amplitude and Morse-phase preservation at
  maximum relative errors below `2.0e-16`;
- kept all published waveform arrays on AutoDL; only small manifests and
  validation evidence are tracked by Git.
- separated solver-level dimensionless Fermat potential from physical arrival
  seconds without changing the frozen v2 record schema or dataset;
- corrected first-two fixture diagnostics and re-ran 101 AutoDL tests;
- verified the frozen manifest, Parquet records, and validation-file hashes
  were identical before and after Phase 1B.1.
- merged PR #2 after GitHub Actions passed at merge commit
  `2a8d8de39d332f90339bd4e7d4c49f66697e6c01`;
- tagged the exact generator commit as
  `gwlens-v2-2.0.0-alpha.2-ae86beab1c132682` and made the published AutoDL
  artifact read-only while preserving its hashes;
- opened Phase 2 as a design-only preregistration phase.
- froze broad proposal and balanced benchmark distributions while explicitly
  denying that the benchmark is an astrophysical rate population;
- froze the conditional two-magnification estimand, synthetic selection model,
  eight EM availability cells, estimator grid, baselines and ablations;
- fixed validation/calibration/IID/OOD counts from binomial precision targets;
- pinned detector-specific synthetic curve names and hashes and identified the
  frozen Phase 1B generic PSD label as imprecise provenance;
- created a fail-closed Phase 3 plan with exact storage arithmetic, a 4,096-pair
  qualification gate, 128-pair atomic shards and byte-identical resume rules;
- passed 99 local tests plus one optional skip and 104 AutoDL tests, Ruff,
  mypy, and package builds.
- split post-hoc calibration fitting from independent SBC without changing the
  118,784-pair total, and separated development from final gold diagnostics;
- froze architecture selection across three-seed means and prohibited
  best-seed selection;
- froze an explicit mass-sheet transformation, an environment observation,
  and an alpha.3 scientific schema while preserving alpha.2 smoke loading;
- froze a Lenstronomy Galkin spherical-power-law/Hernquist/Osipkov–Merritt
  kinematics forward model and prohibited an Einstein-radius shortcut;
- replaced ambiguous OOD/mismatch names with exact pre-result strata and
  detector-specific alternate-PSD hashes;
- created a primary-source literature matrix, verified bibliography and
  conservative novelty statement with no unsupported “first” claim;
- expanded storage gating from raw capacity to peak staging/publication use
  plus a 100 GB post-peak reserve.
- created the separate Phase 3A human authorization, bounded-memory shard
  infrastructure, ADR-002, fail-closed attempt journal, PSD hash checks and
  resource preflight;
- verified the Phase 3A branch checkpoint, frozen preregistration hash,
  authorization denials, authorizing commit, remote free space and absence of
  existing Phase 3A staging/publication;
- completed a section-by-section Phase 3A implementation gap audit and stopped
  before generation when the normalized source-plane density could not be
  implemented from the frozen specification.
- defined the exact RC.3 source-plane preselection measure, normalized angular
  log density, selection-conditioning boundary and deterministic Lenstronomy
  numerical/support-audit contract.
- corrected the finite source square's claim boundary in RC.4 after a
  pre-generation steep-EPL probe; boundary validation now requires
  primary/reference solver agreement rather than absence of multiple images.
- froze RC.5 and the exact Phase 3A generator commit, passed source-plane,
  mass-sheet, Galkin, waveform-boundary, whitening and input-policy gates;
- passed the deterministic 32-pair microbenchmark and every resource gate;
- generated exactly 4,096 non-scientific qualification pairs from 1,455,699
  attempts and atomically published 32 shards of 128 pairs on AutoDL;
- verified byte-identical hashes for the first three shards across intentional
  interruption and resume, unique grouped IDs and exact float32 decomposition;
- retained 330,450,604,032 free bytes after a 4,450,694,559-byte publication;
- stopped Phase 3A with full production, staged scientific production,
  training, calibration, scientific testing and GWOSC/GWTC still closed.
- merged accepted Phase 3A PR #4 after CI using merge commit
  `589b6a554d5bf8213c3014b5cb6f3b0e0f4edd5e`;
- opened Phase 3B as design-only adaptive-production preregistration work;
- froze a nested 16k/32k/65k ladder, development-only stopping evidence and a
  separately sealed 20,480-system final evaluation pool;
- designed but did not authorize a support-preserving 512-pair proposal-v2
  engineering qualification and a separate real-noise/catalog boundary.
- superseded Phase 3B RC.1 with design-only RC.2 after human statistical review;
- reclassified 16k as a probe subset and limited final locks to 32k/65k;
- staged future materialization as 38,912 scale-selection systems, a conditional
  32,768-system extension and 26,624 post-lock systems;
- froze importance-weighted target correction for any efficient training
  proposal while requiring direct-target validation, calibration, SBC and IID;
- replaced unknowable pre-materialization accepted IDs with a hashed
  deterministic final-evaluation generation commitment template;
- made a 2× throughput lower confidence bound mandatory for proposal-v2 and
  required executable normalized component densities before its future gate;
- required reuse of three locked-rung probe fits and froze one stored Gaussian
  noise realization per independent physical system;
- kept all generation, training, proposal-v2, calibration, evaluation,
  GWOSC/GWTC and Phase 3C execution closed pending human review.

## Not started by design

- proposal-v2 engineering qualification;
- scientific data materialization;
- model or posterior training;
- calibration, SBC or final scientific evaluation;
- GWOSC/GWTC download;
- real-noise injection or catalog scan;
- manuscript work.

## Next recommended phase

Human review should inspect Phase 3B.1 RC.2's target correction, commitment,
throughput A/B endpoint and staged arithmetic. No PR merge or subsequent gate
follows automatically. If accepted, proposal-v2 still needs an executable
density specification before a separately authorized 512-pair engineering
qualification; scientific Stage A and training remain independent later gates.
