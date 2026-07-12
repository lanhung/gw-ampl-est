# Project state

## Current phase

Phase 1B and Phase 1B.1 are merged and frozen. Phase 2 release candidate 1
received a request-changes review; Phase 2.1 hardening has produced
preregistration `1.0.0-rc.2` on `phase2/preregistration`, and the user-directed
automatic review passed. No training, catalog download, Phase 3A execution, or
scientific data generation is authorized by this state.

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

## Not started by design

- model or posterior training;
- GWOSC/GWTC download;
- manuscript work.

## Next recommended phase

RC.2 may be merged after CI. A later, separate decision may explicitly
authorize only Phase 3A generator qualification; rc.2 itself does not grant
that authorization. Full production remains gated on measured
qualification throughput, solver acceptance, dynamics convergence, peak-disk
amplification, storage and interruption/resume evidence. Do not reuse the
engineering smoke artifact scientifically.
