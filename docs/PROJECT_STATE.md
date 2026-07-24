# Project state

## Current phase

Direct-target RC.4 is frozen at hash
`5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98`.
The exact-count Stage A run passed and atomically published under parent
`phase4-stage-a-2be777e727ef-d3a60034bbd6`, using frozen generator commit
`2be777e727ef9d8e1a85f89c68966df5d37932b0`. The publication contains exactly
32,768 train plus 6,144 validation systems in 304 complete 128-system shards,
with no partial shard. The parent manifest SHA-256 is
`4f3e6b3a7ca1a995d7a7643c48410e479fb812e4a01ff66537232b9d64bf3314`.

In parallel, the probe stack now has a fail-closed execution runner, atomic-parent
publication resolver, streaming rung standardization, deterministic three-GPU seed
launcher, development metrics and the preregistered paired learning-curve decision.
It may run unit/integration and in-memory engineering smoke tests but cannot read
Stage A or start scientific training. The final-evaluation generation commitment
is finalized at SHA-256
`c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`.
The earlier authorized 16k/32k probe workflow completed, but its learning-curve
result is now superseded because the numerical-waveform audit found affected
systems in both rungs. Calibration and final evaluation were not accessed. The
exact-count Stage B extension remains immutable and complete.

Stage B passed and atomically published exactly 32,768 additional train
systems in 256 shards under parent
`phase4-stage-b-2be777e727ef-6a4f106f9640`. Its parent manifest SHA-256 is
`b4d7df6300d0919f148b98fd8ce658216bdfa64752026dc9477321874e31f0da`.
The atomic combined 65k reference manifest SHA-256 is
`753ace3d2fe475f1279b3bd8560005017f4e75a822fa951d94f9ada60eb3eca4`.
Independent closeout validation passed exact counts, unit weights and group
disjointness. No new validation system was generated. The 65k optimizer
remains closed pending a separate identity-bound training authorization.

The first authorized 65k launch subsequently stopped before its first optimizer
step when whitening encountered a finite but catastrophic source-waveform bin.
A read-only exhaustive audit of all 71,680 published records found exactly five
IMRPhenomXPHM isolated-bin pathologies: two Stage A train and three Stage B train,
with none in validation. One entered the deterministic 16k subset and both Stage
A failures entered 32k, so the prior 16k/32k learning-curve result is superseded.
No valid 65k checkpoint or terminal decision exists.

Numerical-correction preregistration `1.1.1-rc.1`, hash
`7fca209de9f06e98da1c5a96ae0f4fc6daec5d2f0c2339a718e1f899bb915b69`, is now
frozen. It adds a source-polarization isolated-bin rejection before lensing and
selection while preserving the target, q=p, physical selection, approximant,
counts, model and stopping rule. The five-system correction has atomically
published and passed independent closeout under parent
`phase4-waveform-correction-499f86b3159a-1db109b08189`; manifest SHA-256 is
`0fcfb117c620d58a2e0ccd8b19c0d3f3a371dd844fb637b50c8b565eee6864f2` and tree
SHA-256 is
`a57aa2691e256b34403392f595e964dceec1325cfc54a38ed4d2a0b714d38c12`.
Corrected counts are 32,768 Stage A train, 32,768 Stage B train, 65,536 combined
train and 6,144 unchanged validation systems. The original publications and
failed training output remain immutable. Training is still closed until a new
release binds the corrected views, recomputed membership, model, wheel and CUDA
environment.

The corrected probe stack is now implemented. It resolves the correction parent
and all immutable base manifests metadata-only, filters exactly the five affected
physical systems, lazily concatenates the five replacements, and derives new
training-view hashes for both 32k and 65k. The existing 16k SHA-256 rank rule is
applied only after the full corrected 32k membership exists. The runner supports
a fresh 16k/32k rerun and a conditional fresh 65k rerun. Its implementation
checkpoint itself opened no corrected data and authorized no optimizer.

That implementation gate subsequently opened under exact identities. All six
fresh corrected 16k/32k fits completed and the frozen comparison measured a
development NLP improvement of 0.211849 nat per target dimension, with 95%
interval [0.200116, 0.223464]. The resulting decision is
`continue_to_train_65k`. The three corrected 65k probe seeds then completed
from scratch under commit `adcb1a79e1534e4d742238aa99869c57da95dd96`.
Their terminal 32k-to-65k comparison measured an NLP improvement of 0.201437
nat per target dimension, with 95% interval [0.191498, 0.211788]. The frozen
decision is `stop_data_limited_and_new_preregistration`; independent replay was
byte-identical and decision SHA-256 is `90c238a0...`. No calibration or final
case was opened.

Human review has now accepted the prospective terminal-scale preregistration
`1.2.0-rc.1`, canonical hash
`77ff5b6b45b886657e90023c50ae002afffb077db594c80665166d537fd2346a`.
It retains the corrected 65k publication as a strict subset, proposes exactly
65,536 additional direct-target systems for a terminal count of 131,072 and
adds an independent 512-system development-tail pool with 128 cases per frozen
stratum. The terminal outcome is labeled either saturated or resource-capped
data limited; both stop at 131k and require a later architecture execution
gate. No new pair or optimizer is authorized by the design freeze.

The terminal materialization software is now frozen at commit
`a4e6bac014ccd521d510c97593cb1368e826d5eb`, with exact wheel SHA-256
`c7bc8ecadb373ed5d7307ee9c96b131cc68cc9ad8ea10ae2100c54aed2a8958f`.
Full local and AutoDL regression passed. The exact-count execution gate permits
only 65,536 new direct-target train systems plus the four-by-128 development-
tail pool. At review, AutoDL had 221.613 GB free versus a conservative 201.597
GB launch threshold. No official identity or new pair exists until the release
gate repeats that measurement and returns ready.

The corrected-probe gates bound the corrected views, implementation commit,
wheel, model, CUDA environment and final-evaluation commitment. The authorized
16k/32k and terminal 65k runs are now complete; all earlier pre-correction
probe checkpoints and metrics remain superseded. Every downstream scientific
phase remains separately gated, and the terminal data-limited decision prevents
the existing post-lock architecture gate from opening.

The post-lock architecture-selection software is implemented without
scientific data access. It freezes the fixed 2x2 grid, reuses the locked-rung
10-transform/width-256 probe and limits future work to nine new fits. It now
requires the same immutable correction overlay used by the terminal probe and
cannot silently reopen the five excluded base systems. No architecture fit or
selection has been executed.

Before calibration data exist, a downstream-only RC.5 addendum now freezes the
previously unspecified post-hoc algorithm as split-conformal marginal and
joint credible-region level maps. Independent SBC uses five frozen rank
statistics with Holm correction. Pure statistical code and synthetic fixtures
are implemented; materialization, checkpoint inference, calibration fitting,
SBC execution and final evaluation remain closed.

The Phase 6 direct-target materialization and selected-checkpoint score stacks
are now also implemented and execution-disabled. They bind future atomic Stage
A/Stage B parents, exact 4,096+2,048 counts and per-seed checkpoint/publication
identities; no calibration/SBC data or checkpoint has been accessed.

Before final data exist, downstream-only RC.6 now freezes executable final-
evaluation analysis semantics. The 20,480-system generation commitment remains
unchanged. A legacy fixed-EPL-slope cross-family label was not executable by the
family-conditioned estimator, so RC.6 prospectively maps it to the frozen EPL
training-prior-marginalized condition and freezes exact equal-family mixture
semantics. Pure metric and ablation-view code is implemented; final
materialization, unsealing, checkpoint inference, ablation training, baselines
and GWOSC/GWTC remain closed.

An additional pre-data audit established that the historical DINGO-style
likelihood-gold gate cannot be applied to a two-target marginal NSF: no neural
density exists on the complete nuisance state required for full likelihood
weights. RC.7 now forbids that claim and freezes an executable non-neural
selected-prior EM/timing kNN/KDE simulation reference. Its deterministic
metadata-only bank index, exact-role execution gate and bounded streaming
score writer are complete. The future runner records raw coverage counts and
Wilson intervals while never persisting posterior draws. It remains
implementation-only and has not opened the scientific reference bank,
validation or final data.

The two RC.6 input-ablation training views now have a complete fail-closed
software stack. GW-only removes EM values, masks and astrometry while retaining
GW timing and family; EM-only removes strain, detector masks and observed GW
timing while retaining EM and family. Both apply after the locked primary
standardizer and reuse the selected architecture, optimizer and budget. The
future launcher is capped at exactly six fits and three concurrent seeds. No
scientific array, checkpoint, optimizer, calibration/SBC or final case has been
opened; a later exact execution gate remains mandatory.

The remaining inherited legacy SIS point-regression control now has a
read-only verifier. It binds the audited PDF-era checkpoint and 500-row saved
validation prediction hashes, recomputes point metrics and the SIS signed-
magnification identity without deserializing the checkpoint, and rejects every
v2/final-data use. Current authorization is implementation-only; the legacy
assets remain immutable and no official reproduction result has been written.

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
- at `2026-07-14T14:18:07Z`, observed 89/256 train shards complete (11,392
  accepted systems), 16 partial train shards and zero execution errors; this is
  34.77% of train and 29.28% of the full 304-shard Stage A contract.
- completed the fail-closed future probe runner, atomic-parent resolver,
  metadata-only streaming rung preparation, deterministic shard-local epoch
  sampling, three-GPU launcher, validation development metrics and paired
  learning-curve decision; all scientific execution flags remain false.
- installed and normalized the isolated AutoDL candidate environment at freeze
  SHA-256 `2e45000a...`; a full-length GPU engineering smoke passed twice with
  replay SHA-256 `ae4e68c0...`, without reading Stage A.
- atomically published and validated exact direct-target Stage A: 32,768 train
  plus 6,144 validation systems, 304 complete shards, exact unit weights,
  group-disjoint splits and publication-tree SHA-256 `1c9d95d0...`;
- completed Stage A in 39.63 elapsed hours with 46,491,822,064 published bytes
  and 264,872,910,848 bytes remaining free.
- published an immutable five-system correction overlay and independently
  validated exact corrected counts, numerical source spectra, decomposition,
  unit weights, hashes and grouped identities;
- implemented and froze typed corrected 32k/65k training views at commit
  `adcb1a79e1534e4d742238aa99869c57da95dd96`;
- recomputed the 16k membership after applying the overlay and completed all
  six fresh 16k/32k seed fits from scratch;
- independently reproduced the 10,000-replicate paired comparison byte-for-byte;
  corrected NLP improvement was 0.211849 with 95% interval
  [0.200116, 0.223464], requiring continuation to the 65k probe.

## Not started or not yet complete by design

- further proposal engineering qualification (permanently closed);
- post-size-lock architecture selection (blocked because 65k did not lock);
- any training rung above 65,536 (requires a new preregistration);
- calibration, SBC or final scientific evaluation;
- GWOSC/GWTC download;
- real-noise injection or catalog scan;
- manuscript work.

## Current execution and next gate

Stage A, Stage B and the five-system correction overlay are atomically
published. Corrected 16k, 32k and 65k probe fits are complete. The terminal
development decision is `stop_data_limited_and_new_preregistration`, exact JSON
SHA-256 `90c238a0d85d941c9e90a68e8a215a8d9025f57ffe7757ff89dd14c267f6d72f`.
The 32k-to-65k improvement remains decisively above the saturation threshold;
development EM-cell and tail conditions also did not all pass.

The required next gate is a new scientific preregistration. RC.4 does not
authorize an extension above 65,536, and the architecture runner cannot execute
because it requires `lock_train_65k`. Architecture selection, calibration/SBC
materialization, final evaluation, real noise and GWOSC/GWTC are closed. The
completed 65k checkpoints and development evidence remain immutable.

That architecture implementation is correction-aware and merged, but cannot be
executed under the terminal data-limited decision. The future Phase 6 and
Phase 7 generator builders were hardened to inherit numerical-validity
preregistration `1.1.1-rc.1`. The Phase 6 supplemental commitment SHA-256 is
`af87affbaf56695fe0a6c7f422a70fed154dd2df2255df819348ad204dd0ccd4`; the
final-evaluation supplemental commitment SHA-256 is
`431c09f2c279e1c745bd118fb1b0c06643de7dc42f605af78a49ca99b5b0019b`.
The original final-evaluation commitment remains exactly `c13412ec...` and all
downstream execution flags remain false.

The future sealed materialization runner and release gate now support the narrow
post-incident generator revision without rewriting that commitment. They bind
the supplemental hash, correction publication and corrected 65k logical
reference, derive 15 collision-free namespace identities only after a ready
gate, and remain fail-closed under the current implementation authorization.

The future calibration/SBC release path is also correction-aware. It requires
the immutable Stage A, Stage B, combined-base and correction publications and
uses the same five-exclusion/five-replacement logical view as the active probe.
No Phase 6 identity or pair has been created; materialization, checkpoint access,
calibration fitting and SBC execution remain closed.

The terminal 131k materialization is active under its exact 32-worker scheduler.
The first complete atomic shard has now appeared while all 32 workers remain
alive; no execution-result or failure artifact exists. At that observation,
about 50.9 GiB memory and 216.7 GB disk remained available. This proves forward
atomic progress but is not publication evidence.

In parallel, the terminal probe execution stack is implemented under a
synthetic-only gate. It supports the logical corrected-65k plus 65,536-system
increment reader, exact 131,072-member identity, the three frozen probe seeds,
evaluation of both retained 65k and new 131k checkpoints on the same four-by-128
tail pool, and the RC.1 terminal decision. Scientific data/checkpoint access and
optimizer execution remain closed until the materialization completes and a new
exact training release is frozen.

The post-lock architecture software now also supports the terminal rung without
changing the historical 65k code path. It validates either legitimate 131k
terminal label, reuses all three terminal probe fits, executes at most nine new
fits and selects from exactly twelve development-only results. The implementation
gate remains synthetic-only; no architecture checkpoint or scientific record was
opened.

The future RC.7 non-neural reference core is now execution-ready at the pure
software layer. A reader drift was corrected so metadata-only examples retain
their exact Parquet EM-cell partition label without opening Zarr. A deterministic
in-memory index groups the standardized corrected training rung by exact lens
family and EM cell, selects 256 neighbors by `(distance, physical_system_id)`,
and emits small per-case CRPS, KDE NLP, central-coverage and interval-width
scores without persisting 4,096-draw arrays. All tests use synthetic fixtures;
no scientific bank, validation query or final case was opened. A fail-closed
execution wrapper now additionally binds the terminal 65k decision, selected
architecture, corrected logical publication, primary-rung standardizer,
immutable wheel/environment and one exact query publication. It streams one
query at a time into atomic JSONL and summary products with raw coverage counts
and Wilson intervals; current authorization still rejects every scientific
execution attempt.

## Terminal 131k materialization concurrency update

The exact terminal release is active under the unchanged scientific generator
and official identities. Its first 16-worker segment was stopped before any
complete shard, and all 16 partial shards (631,856,490 bytes) were sealed as
immutable interruption evidence. No completed or published output was lost.

The scheduler-only implementation at
`8977ca55f13963441afdda831afb190a3872517c` passed 370 local tests, Ruff,
mypy and build. Exactly 32 workers are authorized for the fresh staging
restart; 64 workers are rejected. The release must pass a new exact preflight
and first-complete-shard resource observation before unattended continuation.
All downstream scientific gates remain unchanged and closed.

That worker-32 preflight passed with no blockers and 220,975,267,840 free
bytes. The segment subsequently completed and atomically published the exact
65,536-system train increment in all 512 shards. The original one-by-128
development-tail layout was then stopped after 3.29 active hours at 15 accepted
cases from 91,839 attempts: even the 95% upper acceptance-rate bound implied
only a 1.2e-7 chance of reaching 128 by the 12-hour worker cap. No tail shard
was complete, and the one partial tree is immutable non-result evidence.

This is an engineering execution-partition mismatch rather than a change to
the conditional tail population. A same-phase recovery keeps all four
128-case strata and the original scientific/root-seed contract, but uses 32
atomic four-case shards per stratum so the authorized 32 physical workers can
operate. It binds the read-only train publication, requires fresh tail and
combined identities and leaves training and every downstream execution gate
closed until independent atomic closeout passes. The frozen recovery
package commit is `6c8d717d5c095d8ab881355d01cc10e0ff84db1b`; runtime
binding is frozen at `ab8e18934eac23cb73be7f7e9c92ce8cb2a3f598`.

The shared terminal downstream binding stack is also implemented under a
synthetic-only gate. It rejects 65k labels in the terminal path, accepts only
the two frozen 131k resource-cap outcomes, requires the exact twelve-result
development architecture lock and prevents a 131k checkpoint from entering an
old 32k/65k score authorization. Focused tests passed 25 cases with one optional
Torch skip; the full suite passed 387 tests with seven optional dependency
skips, and Ruff, mypy and package build passed. No staging path, scientific
publication, checkpoint, calibration/SBC record or final case was opened.
The frozen implementation commit is `cfb3e92f6600975c81e7dfdc58237ebf82acce7c`
and its exact wheel SHA-256 is `35909951c13cffbe695fe4af631d282fd58634e4f80156057a8cd107609c2b4a`.

The future calibration/SBC and sealed final materialization runners now have an
implementation-only terminal reference mode. It preserves the historical 65k
parser, but terminal execution requires exact 131k combined/increment/tail and
decision hashes. Calibration/SBC will prove disjointness against exactly
137,728 train+validation+development-tail systems; final generation retains its
unchanged 20,480 sealed cases. Materialization-focused tests passed 27 cases;
the full suite passed 388 with seven optional skips, and Ruff, mypy, script
compilation and build passed. No scientific record or identity was created.
The frozen implementation commit is `45d05287fbd9a8b7f9bc1999b749be5c521d7931`
and its exact wheel SHA-256 is `bc5d3cd2fd6f898b08590be7f348dc4970edb7fe5f23f4422ffc29185336f4cd`.

Terminal adapters for the two RC.6 input ablations and the RC.7 non-neural
reference are implemented. Both support the logical 131k publication while
preserving historical 65k replay. The ablations share the locked membership,
architecture, optimizer and seeds; the reference bank uses all 131,072 train
systems but never the 512 development-tail cases. Seventeen focused and 392
full tests passed with seven optional skips; Ruff and mypy passed. Execution,
checkpoint and query access remain closed.

The frozen terminal analysis-adapter implementation is
`c5cd67d0537dad81797d2a77913a5f3bbd142f00`; its exact wheel SHA-256 is
`0ae3da4bbb96312b1347babe03ed95cfa45950966c12959e921e78abf7981fd7`.

The terminal runner was audited end to end: after the 512 train shards it
automatically executes all four tail namespaces, cross-component validation,
tree checksums, resource gates and atomic combined publication. An independent
closeout command now validates the exact result through a second read-only
path and recomputes both publication trees by default. Seventeen focused and
399 full tests passed with seven optional skips; Ruff, mypy and build passed.
No active staging directory or scientific checkpoint was opened locally.

A terminal probe release-review packet is also implemented. It binds the
future closeout to an exact wheel tested non-editably on AutoDL, the normalized
CUDA environment, three frozen-model GPUs, model configuration and finalized
evaluation commitment. Six focused and 405 full tests passed with seven
optional skips; Ruff, mypy and build passed. The packet cannot authorize or
execute training.

The frozen release-packet implementation is
`099c5762be9c72f7ded420c64f456db885ec37e5`; its candidate exact wheel
SHA-256 is
`93b541c30e5df571bbbc5b07bef423665814e510a13d0d0595e2a3de2d0e83d7`.

The corresponding exact-wheel verifier is implemented locally without
touching the active AutoDL checkout. It probes the requested runtime in a
separate interpreter, binds PEP 610 archive evidence to the wheel SHA-256,
rejects editable or repository-`src` imports, verifies CUDA/GPU identity and
runs focused plus full tests under `/dev/null` pytest configuration with
`--noconftest`. The latter is necessary because the maintained
`tests/conftest.py` explicitly inserts the checkout's `src` path and would
otherwise defeat installed-wheel isolation even when pytest configuration is
disabled. The
repository root is available only for `scripts/` imports; `src/` is never
added to `PYTHONPATH`. The current local suite passes 412 tests with seven
optional dependency skips. The verifier has not yet run on the future final
post-publication wheel and does not create a training authorization.

The future execution gate is now also machine-bound to the resulting release
packet rather than trusting an authorization narrative. It checks the packet
hash, non-authorizing status, atomic publication identities and counts, exact
wheel/model/environment fields, GPU inventory, finalized evaluation
commitment and every closed downstream flag before publication resolution.
The packet SHA propagates into terminal preprocessing, three-seed run evidence
and retained-65k tail summaries. These checks use synthetic fixtures only;
no terminal authorization, data access or optimizer execution occurred.

The release packet now additionally binds the exact three retained corrected-
65k checkpoints and run summaries. Directory identity alone is rejected. Each
artifact SHA-256 and the shared manifest, standardizer, model, environment,
training-commit and final-evaluation-commitment identity are copied into the
authorization and revalidated before a checkpoint is loaded. This hardening
used synthetic fixtures only and did not access a scientific checkpoint.

The post-publication authorization assembly is also implemented as a separate
two-evidence operation. It accepts only a review-ready packet plus an explicit
delegated-review JSON bound to that packet SHA, the 131,072 rung, seeds 0/1/2,
the retained 65k input root, one fresh terminal output root and an exact set of
closed downstream boundaries. It derives publication roots from independent
closeout identities and self-validates the resulting YAML using the runtime
release-binding validator before atomic write. This software has only used
synthetic packet/closeout/review fixtures.

The release-evidence path contract is now portable across the two hosts.
Closeout, packet and delegated-review JSON files must be committed below
`results/phase4/` and are referenced relative to the explicit repository root;
absolute packet paths and `..` traversal are rejected. Focused tests passed 27
cases and the full suite passed 429 with seven optional dependency skips; Ruff
and mypy passed. The candidate wheel built before this correction is
superseded and was never installed or authorized. A new exact wheel must be
built from the merged portability commit after terminal publication.

The release checkout now also supports the unavoidable two-commit handoff:
the immutable training software is frozen first, while independently replayed
closeout evidence is committed only after publication. Packet assembly accepts
a clean descendant only when the complete Git diff is contained in its exact
closeout-evidence allowlist and records the descendant commit in the packet.
Thirty focused and 432 full tests pass with seven optional dependency skips;
Ruff and mypy pass. No scientific data or checkpoint was opened by this
checkout-control change.

## Terminal development-tail microshard recovery

The terminal train increment remains complete and immutable: 65,536 new
direct-target train systems in 512 shards. The original one-shard tail and the
subsequent fixed-four-case recovery both stopped on prospective resource
evidence, not on a physics, schema or storage error.

The fixed-four-case run completed all 128 high-absolute-magnification cases,
then reached only six partial extreme-relative cases in 328,134 attempts. The
published train attempt stream independently fixes the expected rare-stratum
rate and implies a `1.414149e-26` optimistic probability that all fixed quotas
could finish before their caps. The 1,058,865,790-byte failed parent is
immutable and excluded.

A fresh implementation now represents each 128-case stratum as 128 atomic
one-case shards. Thirty-two physical workers dynamically consume those
deterministic shard tasks. The estimated complete pool cost is about 11.2
million attempts and 42 active hours; the fail-closed hard cap is 96 hours and
the post-run disk floor remains 100 GB. Local verification passed 477 tests
with seven optional-dependency skips, Ruff, mypy and package build.

The exact 32-worker microshard execution is authorized. The frozen release is
commit `adb4c0981fd15a809005212c76dd972a59822489`, wheel SHA-256
`a5b08e40ddcff7d542a68b195d5bfc52577e2a67a8a978e374e1d7581f1e4b52`,
and generator-core manifest SHA-256
`ebb900d52719dd570e378b63a6d2178b5b47a4b4ed6326769fa55e486b6ebda5`.
The host has 64 logical CPUs but only 32 physical cores; the official run is
fixed at 32 workers to preserve memory, I/O and operating-system headroom.

Next steps are:

1. reproduce the AutoDL release certificate;
2. generate exactly 512 development-only cases and atomically publish the
   tail plus combined 131k reference;
3. run the independent closeout and only then review the 131k probe gate.

Training, architecture selection, calibration, SBC, final evaluation, real
noise and GWOSC/GWTC remain closed.

## Terminal 131k publication closeout

The dynamic microshard execution completed on 2026-07-23. It published exactly
512 development-only tail cases in 512 atomic one-case shards, with exactly 128
cases in each of the four frozen strata. The tail parent is
`phase4-terminal-tail-micro128-adb4c0981fd1-30fa02d9ec5b`; its manifest
SHA-256 is `58fcafd58cbcd407ecf6b35dfa98c0bd2bd66f37151e19e6bf530ca2601260c7`
and its independently recomputed tree SHA-256 is
`90ca582f3bd768046f9ceabb4d42689d76945be2c963b0290ac432662ff619c0`.

The logical terminal reference
`phase4-train-131k-adb4c0981fd1-30fa02d9ec5b` now binds exactly 131,072
unique direct-target train systems and the unchanged 6,144-system validation
publication. Its manifest SHA-256 is
`ad26d51d4f9475c6710cdfee4e71409526e1d776e0b8ec14734feff02855cee5`.
The separately published development-tail pool is diagnostic-only and remains
excluded from training.

Independent closeout recomputed the 79,373,465,020-byte train-increment tree
and the 8,989,124,609-byte development-tail tree, verified q=p and exact unit
weights, and confirmed that neither failed tail parent was reused. No
GWOSC/GWTC access occurred. The observed free space was 253,429,231,616 bytes,
well above the 100 GB floor.

The materialization critical path is complete. The next step is not direct
training: the exact post-publication wheel verification, release packet and
delegated terminal-probe authorization must first bind the publication hashes,
retained corrected-65k artifacts, model configuration, finalized evaluation
commitment and immutable CUDA environment. Architecture selection,
calibration, SBC, final evaluation, extension above 131,072 and GWOSC/GWTC
remain closed.

## Terminal 131k probe release authorization

The first execution attempt stopped before preprocessing because the real
terminal increment parent has one singular `validation` mapping while the
training reader recognized only the older plural `validations` layout. No
preparation, checkpoint, optimizer step or scientific metric exists from that
attempt.

A narrow reader compatibility correction accepted both unambiguous manifest
layouts and rejects conflicting dual declarations. The exact corrected wheel
indexed all 65,536 unique IDs from the real parent without opening strain. Its
isolated AutoDL runtime imported `gwlens_mm` from the non-editable wheel,
detected four NVIDIA RTX 5000 Ada Generation devices, passed 70 focused tests
and passed 486 full tests with three optional skips.

Release packet SHA-256
`d2e4fde7b918ce363ca67781d7a462d97ffe37dd4fadde186f587b44be7cdf7a`
binds:

- the exact 131,072-system combined publication and 512-case tail pool;
- training commit `d8a3f1153155797921267557672c03d1ea6543a9`;
- wheel SHA-256
  `fd8da0465f9609e31805abf01f1bf41dc07b486b8e470a6c345a64923b63dda8`;
- the immutable CUDA environment and probe-model configuration;
- all three retained corrected-65k summaries and best checkpoints;
- the finalized pre-training final-evaluation commitment.

Delegated review approved only the 131k probe for seeds 0, 1 and 2. The fresh
output identity is
`/root/autodl-tmp/lensing-4/training/phase4/terminal-probe-131k-d8a3f11-d2e4fde`.
The runtime may evaluate retained 65k checkpoints on the development-tail
pool, fit the 131k probe from scratch and produce exactly one terminal
learning-curve decision. Architecture selection, calibration, SBC, final
evaluation, extension above 131,072 and GWOSC/GWTC remain closed.

## Terminal architecture release-control implementation

The post-lock architecture runner already existed, but an exact release and
authorization builder did not. Implementation commit
`4ef6626eef201aeb91a74f5e9d799ec410459c6a` now creates:

- a non-authorizing release packet bound to the terminal decision;
- hashes for the three reused 131k probe summaries and best checkpoints;
- the frozen grid and all three candidate model hashes;
- an exact wheel, CUDA environment and fresh architecture output identity;
- a separate delegated-review document and fail-closed runtime authorization.

The terminal architecture runtime now verifies checkpoint bytes as well as
summary bytes before reuse. Verification passed 489 full tests with seven
optional skips, 26 focused tests, Ruff and mypy.

No terminal probe metric was read to implement this software, and architecture
execution remains closed. After the terminal decision, a new exact wheel must
pass AutoDL verification and the release packet must receive delegated review
before any of the nine fits can start.

## Calibration/SBC materialization release control

The Phase 6 generator and atomic publisher already enforced the frozen
4,096-system calibration-fit and 2,048-system SBC split, but its future exact
authorization previously had to be assembled manually. The release builder now
binds:

- the terminal 131k size decision and twelve-result architecture lock;
- the selected architecture identity without selecting a best seed;
- the exact generator wheel and immutable environment;
- all seven Stage A, Stage B, correction and terminal publication roots and
  manifest hashes;
- the unchanged direct-target distribution, numerical-validity contract and
  48-shard count arithmetic.

The packet is non-authorizing. Only a separately hash-bound delegated review
can create the materialization-only YAML. Checkpoint inference, calibration
fitting, SBC statistics, final evaluation, model tuning and GWOSC/GWTC remain
closed until their own later gates.
