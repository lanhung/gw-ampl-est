# Project state

## Current phase

Direct-target RC.4 is frozen at hash
`5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98`.
The exact-count Stage A run is active under parent
`phase4-stage-a-2be777e727ef-d3a60034bbd6`, using frozen generator commit
`2be777e727ef9d8e1a85f89c68966df5d37932b0`. At the read-only snapshot
`2026-07-14T10:50:24Z`, 64/256 train shards were atomically complete, 16 train
shards were partial, 8,192/32,768 train systems were complete, no error pattern
was present and 312,123,940,864 bytes remained free. Validation generation and
final publication have not started.

In parallel, an implementation-only branch now contains the candidate probe
training stack. It may run unit/integration and in-memory engineering smoke
tests but cannot read Stage A, start scientific training or inspect final
evaluation. Scientific probe training remains separately gated until Stage A
is published, all 32k train IDs resolve the deterministic 16k subset, and the
final-evaluation generation commitment is finalized and hashed.

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
- designed but did not authorize a support-preserving proposal-v2 engineering
  A/B qualification and a separate real-noise/catalog boundary;
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
- resolved the proposal qualification count as two distinct 512-pair arms,
  exactly 1,024 engineering-only accepted pairs in total;
- froze separate parent/control/candidate manifest and dataset identity
  templates and permanent scientific-use denial for both arms;
- added a conservative double-RC.5 prelaunch projection of 1.4864 active hours,
  1,112,673,640 publication bytes and 121,446,475,732 minimum free bytes;
- upgraded the design to RC.3 and refreshed the final-evaluation commitment
  reference without resolving its future generator placeholder.
- merged Phase 3B PR #5 after its PR-triggered CI passed;
- implemented exact proposal-v2 RC.1 sampling, full latent densities,
  deterministic replay, privileged policy and a dry-run-only A/B skeleton;
- ran exactly 200,000 latent-only draws with zero waveform pairs and recorded
  finite density, support, weight and replay evidence;
- hard-stopped proposal-v2 RC.1 after overall/family relative ESS failed the
  frozen thresholds; no post-result tuning or A/B generation occurred.
- merged Phase 3C-0 PR #6 after CI while retaining its negative evidence;
- implemented exact evaluation-target sampling and target-anchored proposal-v3;
- certified population ESS >=0.55 and empirically measured 0.78532 overall;
- measured RC.5 baseline ESS 0.11776 as diagnostic-only evidence;
- kept A/B, waveform generation, training and external access closed.
- accepted and merged proposal-v3 latent evidence, then opened only the bounded
  Phase 3C-A 512+512 engineering A/B gate;
- froze generator commit `185e68d4346d84edc118a9197ffb8bceeb026ee4`
  after 191 local tests, Ruff, mypy and build, then passed 198 AutoDL tests and
  every inherited physics/numerical preflight;
- atomically completed one 32-pair block per arm and stopped at the first
  matched-block health gate when the new validator used the wrong alpha.3
  distribution-metadata attribute name;
- retained both block hashes and staging evidence, published nothing, computed
  no throughput endpoint and kept Stage A/training/GWOSC/GWTC closed.
- merged the Phase 3C-A failure evidence in PR #8 and created a fresh retry
  identity rather than resuming either failed artifact;
- corrected distribution metadata validation through a typed package helper
  and exercised the real JSON/Parquet/Zarr health path;
- passed 193 local tests, 202 AutoDL tests and every inherited preflight under
  frozen generator commit `324bab47aff5c4ed2b2041099a103735a40463f0`;
- passed the corrected first matched-block health gate without inspecting an
  interim endpoint;
- atomically completed 12 control and 12 proposal-v3 blocks (384 pairs each)
  before the control arm reached its frozen six-hour cap during block 12;
- retained 24 block hashes, one incomplete control block and the fail-closed
  result without publication, bootstrap or post-selection inference;
- closed proposal optimization after its one full retry and selected a future
  direct-target route that still requires a new scientific preregistration and
  Stage A authorization.
- merged the Phase 3C-A.1 report in PR #9 at
  `ce0cf464cf5b56e3df5e1b0c93ffadc12f2e517a`;
- created direct-target preregistration RC.4 without modifying adaptive RC.3;
- froze exact q=p, zero-log-weight and unit-weight scientific training
  semantics while retaining the 32,768/6,144 Stage A counts;
- implemented typed production-generator train/validation contexts, full
  namespace validation, a bounded-memory Stage A runner and atomic parent
  publication;
- implemented an 8+8 disposable canary with distinct non-scientific identities
  and byte-identical interruption/resume validation;
- implemented one fail-closed release-gate command and an exact AutoDL
  dependency lock;
- verified the design-state gate creates no official identities and both
  execution runners refuse absent authorization.
- froze pre-execution implementation checkpoint
  `87325ef0fede15304378cfb846ed3ba88ba8c5af` and synchronized it to AutoDL;
- passed 210 local tests plus 219 AutoDL tests, maintained-scope Ruff, mypy and
  package builds, and reproduced the RC.3, RC.4, environment-lock and
  final-evaluation commitment hashes;
- confirmed that the canary and Stage A output roots remain absent, no pair was
  generated and the release certificate remains blocked with null official
  identities.
- merged the RC.4 pre-execution implementation and a narrow authorization-only
  integration fix, freezing generator commit
  `2be777e727ef9d8e1a85f89c68966df5d37932b0` and wheel hash `14104f8a...`;
- installed the exact wheel non-editably on AutoDL and passed 220 tests plus
  Ruff, mypy, environment-lock and authorization preflight checks;
- generated exactly 8+8 disposable direct-target canary pairs, intentionally
  interrupted after the first namespace and verified its shard byte-identical
  after resume;
- passed q=p/unit-weight, JSON/Parquet/Zarr/schema, array decomposition,
  grouped-ID disjointness, telemetry-field and independent checksum checks;
- retained Stage A as blocked with no official identities and did not inspect
  canary throughput or ESS.
- accepted the passed canary through a separate exact-count materialization
  authorization while keeping training, calibration, evaluation and external
  data access closed;
- passed a hardened single release gate that verified the actual frozen wheel,
  dependency lock, canary, PSD identities, disk gate and authorization, then
  created the official Stage A parent/train/validation identities;
- launched the exact 32,768-train plus 6,144-validation direct-target run at
  `2026-07-14T02:07:18Z` with 16 workers and an atomic resume strategy;
- at `2026-07-14T08:05:43Z`, completed 42 of 304 total shards (5,376 accepted
  pairs), retained 16 partial train shards, observed zero execution errors and
  kept the official parent under staging;
- at `2026-07-14T12:52:16Z`, completed 80 of 304 total shards (10,240 accepted
  train systems), retained 16 partial train shards, observed zero error
  artifacts and kept the parent unpublished with 309,241,184,256 free bytes;
- independently read-validated the first completed 128-pair train shard,
  including complete-marker/artifact integrity, q=p and exact unit weights.
- implemented a bounded-memory published-shard reader that opens noisy strain
  only, exact Bilby-compatible PSD whitening, safe feature/target extraction,
  the mask-aware GW/EM conditional NSF, deterministic checkpoint/resume,
  learning-curve metrics and a fail-closed training planner;
- passed a full-length 16,384-sample in-memory PyTorch/nflows smoke twice with
  byte-identical replay hash `b4a7126c83963e8d05c3014ecc1385bc90226d3c9863ad99aedf7fcc7277523e`;
- passed 225 local tests with five optional skips, maintained-scope Ruff,
  mypy for 44 source files and package build; no scientific data or checkpoint
  was read or written by the model smoke.
- froze the implementation-only probe-training stack at
  `19f8dc0621f610825d000f37af333f384a963e55`; scientific training remains
  blocked on Stage A publication, the final-evaluation commitment and a
  separate execution authorization.
- implemented all 15 deterministic final-evaluation generator contexts for the
  frozen 20,480-system sealed pool, including exact tail priority, cross-family,
  parameter-OOD, waveform and PSD mismatch semantics;
- implemented a fail-closed future runner with streaming record/array/journal
  validation, global grouped-ID checks, resource caps and atomic sealed parent
  publication; no final-evaluation pair or identity was created;
- froze that generator at `bc02054c1f95e7f6cd143fb9dc796ae48f0a15ac`
  and finalized the pre-training deterministic generation commitment with
  SHA-256 `c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`;
- at `2026-07-14T13:15:29Z`, observed 82/256 train shards complete (10,496
  accepted systems), 16 partial train shards and zero execution errors; Stage A
  remained staging-only and validation had not started.

## Not started or not yet complete by design

- further proposal engineering qualification (permanently closed);
- Stage A validation generation and final atomic publication;
- scientific model or posterior training (implementation exists, execution is closed);
- calibration, SBC or final scientific evaluation;
- GWOSC/GWTC download;
- real-noise injection or catalog scan;
- manuscript work.

## Current execution and next gate

Stage A is actively materializing under parent
`phase4-stage-a-2be777e727ef-d3a60034bbd6`. It must finish all 304 shards,
complete full validation and atomically publish before a Phase 4 completion
report can be written. The final-evaluation implementation and commitment are
now frozen without opening materialization. Model training remains a later,
independent gate.
