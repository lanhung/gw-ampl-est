# Failures and unresolved evidence

## Phase 0

- The exact generator commit and environment for qkzhang February arrays cannot
  be recovered because the root has no Git history or dataset manifest.
- The 0228 generation seed is unavailable.
- The detector identity of qkzhang two-dimensional strain arrays is unproven.
- wjx generator Git history begins with an imported/cleaned project baseline and
  does not prove the commit that produced every large data directory.
- The PDF baseline directory has no Git history and its local/remote PDF binaries
  are different renderings.
- The PDF checkpoint has validation results only; no untouched test posterior or
  calibration result exists.
- A full checksum scan of approximately 1 TB of waveforms was deliberately not
  run because it would create unnecessary I/O load and is not needed for the
  Phase 0 decision.

These are documented limitations, not reasons to alter or discard legacy files.

## Phase 1A

- `lenstronomy` was absent from the checked AutoDL Python environment. No
  non-SIS numerical reference fixture was executed and no remote package was
  installed; Phase 1B must pin and validate its chosen solver.
- The alpha JSON Schema is a top-level interoperability boundary; nested
  scientific validation currently depends on the typed Python model.
- The first Phase 1 sync revealed that `data/` exclusions also matched
  `configs/data/` and that tool caches were not excluded. Root-anchored data
  exclusions and explicit cache exclusions corrected future sync behavior.
- Waveform projection, preprocessing amplitude preservation, resume behavior,
  and noisy/clean/noise equality are deliberately untested until Phase 1B.
- The managed Vultr sandbox could not perform a default editable pip install:
  build isolation could not reach package indexes and the user site was
  read-only. A no-network wheel build and isolated `/tmp` target import passed,
  so this is an environment limitation rather than a package failure.

No Phase 1A failure was hidden by generating data or weakening a physics
constraint.

## Phase 1A.1

- No external non-SIS solver was installed or executed; no SIE/EPL numerical
  result is claimed.
- Strain zero-fill and decomposition semantics passed only small in-memory
  tests. Zarr shard publication, interruption/resume, and dataset-level checks
  remain Phase 1B work.
- The JSON boundary now covers the changed association objects and rejects old
  fields, but cross-record physics remains authoritative in typed Python.
- Isolated `python -m build` succeeded but warned that the repository lacks a
  conventional package README.
- No PR-triggered GitHub Actions result is claimed before a PR exists.

The Phase 1A positional astrometry, image-only noise provenance, bare timing,
and incomplete extra-image status weaknesses were resolved before data
materialization.

## Phase 1B

- The first five-pair staging command contained a manually mistyped generator
  commit. The hash did not identify a Git object. Publication was never
  attempted; the staging directory is retained under the new AutoDL project
  root with an `ABANDONED_FAILED_STAGING.txt` marker and is excluded from every
  result and count.
- The first resume test with the initial generator commit failed because Bilby
  2.6 PSD-conditioned noise uses `bilby.core.utils.random.Generator`, not NumPy's
  legacy global RNG. The byte mismatch stopped publication automatically. The
  generator was fixed to seed Bilby's RNG explicitly, an independent
  determinism check passed, and a new generator commit/dataset ID was used.
- The published smoke artifact uses synthetic Bilby curve-conditioned Gaussian
  noise, identity float32 preprocessing, and a fixed engineering lens/source
  grid. It is not evidence for real-noise performance, population performance,
  posterior calibration, or any scientific inference claim.
- The SIS smoke delay uses a documented one-day engineering conversion of the
  analytic dimensionless Fermat coordinate. It is a schema/windowing control,
  not an astrophysical time-delay population model.
- Initial Phase 1B branch CI failed only under the Python 3.9 runner because
  newer NumPy typing inferred a division result too broadly for `float()`.
  A dedicated float-returning safe-norm helper resolved the cross-version
  NumPy stub ambiguity; no runtime algorithm or published dataset bytes
  changed.

Both failed staging states remain immutable evidence; neither was published or
silently removed.

## Phase 1B.1 review findings

- `selected_pair_is_first_two` compared against hard-coded Lenstronomy IDs, so
  the SIS double was incorrectly reported as false. The diagnostic now uses the
  actual first two returned image IDs; this was an evidence-file defect only.
- `PhysicalImage.arrival_time_dimensionless` held dimensionless SIS Fermat
  coordinates but physical seconds for Lenstronomy. The overloaded field was
  removed and replaced with explicit optional dimensionless and seconds
  quantities. The frozen dataset had already handled these paths explicitly,
  so no published array or record changed.
- The repository-wide command `ruff check scripts` also scans the frozen Phase
  0 manifest-builder script, which contains pre-existing long audit-table lines.
  Phase 1B.1 therefore ran Ruff on `src`, `tests`, all `scripts/phase1b`, and the
  schema generator. The exclusion is explicit; no new lint failure was hidden.

## Phase 1B.1 closeout operations

- The first Git push used the host's default SSH identity and GitHub rejected
  it. Re-running with the repository's dedicated deploy identity succeeded;
  no commit or remote branch was lost.
- The first closeout pytest command ran from the AutoDL project root rather
  than its `repo/` code directory. Pytest followed the editable installation
  into `repo/tests` but could not import the script namespace from that working
  directory, yielding 100 passed and one path-related failure. Re-running from
  `/root/autodl-tmp/lensing-4/repo` passed all 101 tests. This was an operator
  working-directory error, not a solver or test portability defect.

## Phase 2 design audit findings

- The frozen Phase 1B field `synthetic_gaussian_design_psd` overgeneralizes the
  actual Bilby 2.6.0 defaults. H1 and L1 loaded `aLIGO_O4_high_asd.txt` with
  SHA-256 `eb5ec9b081c3d86d2f4257b9aff6a57566d168b8a95e5e57b7909eebad021780`;
  V1 loaded `AdV_psd.txt` with SHA-256
  `c2532150f63dbaa76d451e2c074390272e259e4e2eedc3e684d2205582aa0764`.
  The immutable engineering artifact is not regenerated. Future manifests must
  store detector-specific curve names and hashes and use the label synthetic
  Gaussian curve-conditioned noise.
- The pinned LALSuite 7.26.1 environment does not recognize `SEOBNRv5PHM`.
  Phase 2 therefore freezes `SEOBNRv4PHM`, which is available, as the waveform
  mismatch case. No unavailable approximant is claimed or silently replaced.
- Package builds still warn that the repository has no conventional top-level
  README. The sdist and wheel complete successfully; this packaging-quality
  warning is deferred and is not hidden as a failed build.
- Because the safe sync intentionally omits `--delete`, the disposable AutoDL
  checkout retained a superseded untracked Phase 2 test after that test was
  consolidated into a tracked module on Vultr. An unrestricted remote pytest
  therefore reported 108 passes by running both copies. The authoritative
  Vultr file set passed 104 tests remotely with the stale path explicitly
  ignored. No AutoDL source file was edited or deleted.

## Phase 2.1 request-changes review

- RC.1 used one calibration split both to fit post-hoc correction and to source
  SBC, so corrected calibration could not be treated as independent evidence.
  RC.2 creates group-disjoint fitting and diagnostic splits at fixed total
  size.
- RC.1 sourced the likelihood gold subset only from IID test and did not forbid
  revision after failure. RC.2 separates development and report-once final
  subsets.
- RC.1 could be interpreted as selecting the best of twelve architecture/seed
  fits. RC.2 selects architectures by their three-seed mean and reports every
  seed.
- External convergence, velocity dispersion and named OOD/mismatch sets were
  under-specified in RC.1. RC.2 connects external convergence through an MST,
  freezes an actual Galkin model and fixes diagnostic boundaries.
- The initial literature list was adequate for identifiability but inadequate
  for a publication novelty claim. RC.2 adds a source-linked comparison matrix
  and explicitly declines an unsupported “first” claim.
- Vultr still lacks Lenstronomy, so the local optional adapter and dynamics
  tests remain skipped. The final declared dynamics fixture passed read-only on
  the pinned AutoDL environment: 4,000/16,000-sample values were 237.720593 and
  237.937421 km/s (relative difference about 0.00091, below 0.02). This is a
  numerical contract check, not a simulated dataset or performance result.

## Phase 3A pre-generation hard stop

- The clean `2bb1ea43cb50f0fbaf8eea9f88750a90603b596a` checkpoint passed branch,
  ancestry, authorization, frozen-hash and remote collision checks.
- The frozen proposal specifies
  `uniform_in_solver_bounding_region_then_condition_on_multiple_images`, but
  neither the bounding region nor its relationship to lens scale is defined.
- The frozen evaluation distribution specifies
  `uniform_in_multiply_imaged_cross_section`, but the cross-section measure,
  caustic/pseudo-caustic treatment, numerical integration method and tolerance
  are undefined.
- The Phase 3A prompt simultaneously requires normalized proposal/evaluation
  log densities and forbids altering or completing frozen distributions merely
  to make generation pass. Repository-wide search found no additional contract.

No microbenchmark pair, qualification pair, staging directory or publication
was created. This is a preregistration execution contradiction, not a runtime
or solver failure. Human review must provide a versioned contract; code must not
silently choose a solver search window or approximate caustic area.

## Phase 2.2 source-plane contract resolution

- RC.2 did not define the solver bounding region or an executable normalized
  multiply-imaged cross-section measure. Phase 3A correctly stopped before any
  microbenchmark or qualification data were created.
- Human-approved RC.3 resolves this by freezing an exact normalized
  preselection source-plane measure and treating lens multiplicity/detection as
  explicit selection events. It does not estimate a caustic area and call that
  estimate exact.
- The source support and solver search window still require the frozen Phase 3A
  boundary comparison. A failed support audit remains a hard stop.

## Phase 3A RC.3 source-boundary audit

- The pre-generation boundary probe found that EPL slope 2.5 systems can remain
  multiply imaged at every tested edge/corner of `[-2.5,2.5)^2`. This is
  consistent with the singular steep power-law deflection and means a claim of
  covering a finite full multiply-imaged cross-section is not executable.
- SIE and shallow-EPL boundary probes returned no multiple images in the same
  test. No pair or shard was generated.
- RC.3 is retained as superseded history. A new version must call the finite
  source square a deliberately truncated benchmark support and test primary
  versus reference solver classification on its boundary rather than require
  absence of multiple images there.

## Phase 3A RC.4 waveform-boundary hard failure

- The frozen generator commit
  `a2b8a02b4631e86c39e1b682e4424ecc2f2c5ca9` evaluated four deterministic
  8-second IMRPhenomXPHM boundary fixtures against aligned 32-second references.
- All arrays were finite, but all four cases failed. Relative differences were
  0.0338--0.0609 against a `1e-5` limit. Leading/trailing 0.25-second energy
  fractions were `6.19e-5`--`3.04e-4` against a `1e-6` limit.
- The 32-accepted-pair microbenchmark and 4,096-pair run were not started. The
  frozen criteria cannot be relaxed after observing this result; a separately
  reviewed waveform-window/configuration revision is required.
- An additional non-publication 1,000-attempt acceptance probe found 2 accepted
  systems and projected roughly 224.8 hours for 4,096 at the measured serial
  rate, independently exceeding the 24-hour gate. This probe is diagnostic,
  not a substitute for the prohibited formal microbenchmark.
- Follow-up read-only diagnostics ruled out a simple alignment error and found
  that the stored clean-strain conversion uses unnormalized `numpy.irfft`
  rather than Bilby's `infft`. At 2048 Hz this would suppress clean strain by a
  factor of 2048 relative to stored noise. This must be corrected under a new
  reviewed waveform contract before any official generation.
- RC.5 corrects this before execution by using Bilby's normalized `infft`, a
  fixed 64-second construction, conditioned 8-second publication and
  stored-clean selection SNR. Precommit AutoDL diagnostics passed all four
  128-second reference comparisons; these remain non-authoritative until rerun
  on the clean generator commit.

## Phase 2.3 finite-support clarification

- RC.3's first extreme probe found steep EPL doubles on every tested finite
  source-square boundary point. No dataset output was created.
- RC.4 corrects the claim boundary: the exact square is a deliberately
  truncated benchmark and does not purport to contain the complete strong-lens
  cross-section. Primary/reference solver agreement replaces the impossible
  no-multiple-images-at-boundary requirement.

## Phase 3A completion limitations

- The final runner persisted microbenchmark peak RSS but not a continuous
  peak-RSS counter for the 4,096-pair run. A sampled 16-worker snapshot totaled
  5,446,536 KiB; this is not represented as an exact maximum. A future
  production runner should persist cgroup or process-tree peak memory and
  time-integrated CPU utilization.
- The literal repository-wide Ruff command continues to report 18 pre-existing
  formatting findings in `scripts/remote/build_phase0_manifests.py`. Phase 3A
  scope Ruff passes. The old script was deliberately not reformatted after the
  microbenchmark because any code change would violate the frozen generator
  commit and force a full rerun.
- These reporting/tooling limitations did not alter the 4,096-pair validation,
  byte-identical resume result, 10 GB output gate or 100 GB free-space gate.

## Phase 3B design uncertainties and deferred evidence

- RC.5 resource projections assume linear scaling of Phase 3A attempts, active
  time and published bytes. They are planning estimates, not measured
  scientific-production performance.
- The proposal-v2 2× scenario is explicitly hypothetical. No caustic-aware
  proposal was implemented or run, and its ESS thresholds have no execution
  result.
- Future scientific IDs and manifest commitments are specified by deterministic
  assignment rules but are not materialized in Phase 3B.
- The proposed 91-event catalog size is not treated as immutable. Release,
  inclusion rules and event-list hash remain future human-review inputs.
- No model exists for the newly frozen learning-curve thresholds; their first
  use requires separate training authorization and cannot inspect final tests.

## Phase 3B RC.1 statistical request changes

- RC.1 listed 49,152 as a possible final total even though its first stopping
  decision required the 32k fit. No execution occurred. RC.2 removes the
  impossible 16k lock and retains 16k only as probe evidence.
- RC.1 recorded proposal/evaluation weights but did not freeze how a model
  trained under a changed proposal would retain the evaluation posterior.
  No model was trained. RC.2 requires the full-latent importance-weighted
  conditional likelihood and direct-target validation/calibration/SBC/IID.
- RC.1 said final accepted IDs would be frozen before materialization, although
  selection history determines them. No final data existed or were viewed.
  RC.2 instead provides a deterministic generation commitment template whose
  placeholder must be finalized and hashed before training.
- RC.1 allowed acceptance or throughput to pass proposal-v2. No proposal was
  implemented or tested. RC.2 makes a 2× throughput lower confidence bound
  mandatory and requires an executable density specification before any gate.

## Phase 3B.1 A/B count ambiguity

- RC.2 declared a 512-pair proposal qualification while its two-arm design
  separately required 512 accepted pairs in each arm. That would make the
  actual total 1,024 and could exceed a literal 512-pair authorization.
- No proposal was implemented, no pair was generated and no resource was
  consumed. RC.3 resolves the design before PR review by freezing 512 pairs per
  arm, 1,024 total, distinct arm identities and a double-RC.5 resource gate.

## Phase 3C-0 proposal-v2 latent ESS hard failure

- Exact proposal-v2 RC.1 produced finite log densities/weights for all 200,000
  latent draws, mean importance weight 1.01044, zero support holes and
  byte-identical replay.
- Overall relative ESS was 0.09202 against 0.50. SIE/EPL ESS was 0.11969/0.07433
  against 0.40. The frozen gate therefore failed before any waveform pair or
  A/B artifact was created.
- The failure is consistent with leaving the broad RC.5 lens-factor proposal
  unchanged while targeting concentrated evaluation distributions. This is a
  diagnostic explanation, not permission to retune the frozen mixture.
- Proposal-v2 RC.1 remains unauthorized and RC.5 remains the fallback. A new
  reviewed proposal version is required for further work.

## Phase 3C-0.2 resolution and retained limitation

- Target-anchored proposal-v3 passed all latent gates; the earlier RC.1 failure
  remains immutable and was not tuned in place.
- RC.5 baseline diagnostic ESS was only 0.11776 overall and 0.09484 for EPL.
  This is not a retrospectively frozen failure gate, but it remains a material
  warning against direct weighted training from broad RC.5 draws.
- No A/B or waveform evidence exists for v3. Latent ESS passage cannot be
  represented as a throughput result or execution authorization.

## Phase 3C-A first matched-block health-validator failure

- Generator commit `185e68d4346d84edc118a9197ffb8bceeb026ee4`
  passed all local and AutoDL pre-execution gates.
- The official run atomically completed one 32-pair RC.5 block and one 32-pair
  proposal-v3 block, then the health validator accessed nonexistent
  `DistributionMetadata.evaluation_log_probability`; the schema field is
  `evaluation_prior_log_probability`.
- The machine outcome is `execution_failed`. No second block or publication
  exists, and no interim/final throughput ratio or post-selection ESS was
  computed.
- Both completed blocks remain immutable engineering-only staging evidence.
  They cannot be reused in a corrected run. A new commit, new identities and
  human review are required before retrying.

## Phase 3C-A.1 active-time-cap stop

- Commit `324bab47aff5c4ed2b2041099a103735a40463f0` corrected the alpha.3
  accessor through a typed helper. The real JSON/Parquet/Zarr health regression,
  193 local tests, 202 AutoDL tests and all inherited preflights passed.
- The corrected first matched-block health gate passed both arms and explicitly
  did not inspect interim throughput.
- The new run completed 12 atomic 32-pair blocks per arm, then control block 12
  exceeded the frozen six-hour active-time cap. Machine state is
  `execution_failed`; one control partial block remains.
- No publication, paired bootstrap, throughput ratio, effective-throughput
  ratio or post-selection ESS exists. The 384+384 completed pairs cannot be
  used scientifically, resumed, published or combined with another attempt.
- This was the single full retry. Proposal optimization is closed and no
  proposal-v4 or third A/B is planned. Direct evaluation-target generation is
  the fallback, pending a new scientific preregistration and Stage A execution
  authorization.

## Phase 4 pre-execution requirements resolved; execution remains incomplete

- The exact wheel, dependency lock, 8+8 canary, PSDs, disk gate and Stage A
  authorization passed the hardened release gate with no blockers.
- Official Stage A identities now exist and the 38,912-system run is active.
- This is not yet completion evidence: validation generation, the final
  dataset manifest, full grouped-ID checks and atomic publication remain
  pending.
- No model-training result exists, and training remains unauthorized.

## Phase 4 training-stack deferred execution boundary

- The implementation-only stack passed unit, type, lint, build and deterministic
  in-memory model smoke checks. This is not a model fit or performance result.
- The isolated AutoDL environment installation correctly rejected Zarr 2.18.7
  and Numcodecs 0.15.1 because those releases require Python 3.11. The executable
  Python-3.10 environment is now frozen to Zarr 2.18.3 and Numcodecs 0.13.1;
  no scientific data or generator environment was changed.
- A disposable in-memory AutoDL GPU smoke then exercised full-length strain,
  forward/backward optimization, conditional sampling and sampled log density
  twice with byte-identical replay. It did not read Stage A or produce a model
  performance result.
- The final-evaluation commitment is now finalized. The scientific training gate
  correctly refuses execution until Stage A publication and a separate training
  authorization also exist.
- The published-shard reader has not read Stage A staging and cannot be treated
  as end-to-end scientific I/O evidence until atomic publication is complete.
- Review also caught a prospective I/O failure: a global random permutation would
  reopen lazy stores almost once per sample. The shard-local deterministic sampler
  fixes that before training without changing the epoch permutation contract.

## Phase 4 final-evaluation implementation boundary

- The final-evaluation commitment remained intentionally unfinalized while its
  six generator paths were absent. The Stage A generator commit was not used as
  a false placeholder.
- Implementation review caught endpoint drift before data existed: the first
  OOD sampler draft used the common `[lower, upper)` convention for all
  intervals, while the frozen parent specifies `(0.15,0.25]` for high shear and
  external convergence and `(2.5,2.7]` for the high-slope branch. Sampler,
  density, validators and tests now use the exact parent semantics.
- No pair or official identity existed during this correction. It is an
  implementation finding, not a scientific result or a preregistration change.
- Final-evaluation execution and unsealing remain unauthorized; the implemented
  runner must fail before generation without a future exact authorization,
  finalized commitment and release certificate.

## Phase 4 probe-runner integration findings resolved before data access

- Review found that the original reader expected `dataset_manifest.json` inside
  each train/validation child. The active immutable Stage A runner correctly writes
  one manifest at the atomic parent, so the old reader would have failed after
  publication. The reader now resolves and hashes child identities from that parent
  contract; Stage A generation code and artifacts were not changed.
- The original `InputStandardizer.fit` retained a sequence of prepared examples.
  At 16k/32k, retaining their strain tensors would violate the bounded-memory
  requirement. Rung preprocessing now streams Parquet metadata and never opens
  strain arrays.
- Development metric functions existed but were not connected to a scientific
  runner. The runner now writes per-case validation NLP, CRPS, marginal coverage,
  joint HPD coverage, interval width, EM-cell and internal-tail diagnostics, then
  applies the frozen paired bootstrap. No scientific result was calculated while
  making these corrections.
- The prior software candidate named Python 3.11, while the qualified AutoDL host
  provides Python 3.10.12 and the package supports Python 3.9 or newer. The isolated
  training environment is therefore pinned to the real 3.10.12 runtime before any
  training authorization, rather than fabricating a 3.11 execution identity.
- A post-merge synthetic CUDA checkpoint canary found that `torch.load` with a
  CUDA map location also moved the saved CPU RNG ByteTensor to CUDA. Torch's RNG
  restore APIs require host ByteTensors. The same-phase patch explicitly moves
  CPU and CUDA RNG-state tensors back to host before restoration and is covered
  by a regression plus byte-identical interruption/resume canary. No scientific
  data or checkpoint was involved.

## Phase 4 Stage A completed without a new hard failure

- The exact 38,912-system direct-target run passed all publication validators
  and atomically published. No partial shard, grouped-ID leakage, nonunit weight,
  external-data access or legacy-root mutation was found.
- Actual attempts were 5.25% above and published bytes 9.96% above their planning
  estimates. These were projection deviations, not gate failures: elapsed time
  was 29.8% below projection, bytes stayed below the projected peak envelope and
  post-publication free space exceeded the hard floor by 164.87 GB.
- No scientific training result exists. Treating the passed data publication as
  model performance evidence remains forbidden.

## Phase 4 probe first-batch physical-memory failure

- The authorized 16k rung membership and metadata-only standardization passed,
  but all three initial seed processes exhausted 32 GB GPU memory on their first
  physical batch of 256 full-length six-stream waveforms. No optimizer step,
  checkpoint or development metric was produced.
- The frozen scientific batch remains 256. The same-phase engineering correction
  splits each ordered effective batch into four physical microbatches of 64 and
  accumulates their exactly scaled gradients before one clip and optimizer step.
  Learning rate, optimizer, sample order, effective batch, model architecture,
  data, target and stopping rules are unchanged.
- The failed output root is immutable and excluded from later results. Corrected
  execution requires a new code/wheel/model-config identity and a new output root;
  it is not a resume of the failed pre-checkpoint processes.

## Phase 4 32k probe is not saturated

- This is a preregistered scientific decision, not an execution failure. Six
  corrected fits completed and the 32k rung materially improved over 16k.
- The paired NLP-improvement 95% interval was [0.223545, 0.248638], far above
  the less-than-0.01 saturation condition. All three CRPS improvements also
  exceeded 22%.
- Uncalibrated development coverage and EM-cell conditions did not support a
  32k lock, and the extreme-relative-magnification internal view contained 40
  rather than the required 128 cases. No final evaluation was opened to resolve
  the decision.
- The frozen response is a separately authorized 65k extension. This result
  does not permit changing the model, inspecting final tests or generating more
  than the single 32,768-system Stage B increment.

## Phase 6 preregistration audit caught an unspecified calibration map

The parent contracts named a post-hoc calibration split and froze coverage/SBC
thresholds, but did not define the map that would be fitted. Leaving that choice
until after calibration data existed would permit outcome-dependent method
selection. No calibration or SBC data had been generated or inspected when the
gap was found.

RC.5 resolves the issue prospectively with split-conformal credible-region
level maps, exact finite-sample order statistics and independent SBC. It does
not modify already generated RC.4 data or claim to recalibrate the analytic
flow density. All Phase 6 execution remains closed pending size and
architecture lock plus separate materialization and inference gates.

## Phase 6 execution-stack audit found no new scientific failure

- The first implementation draft bound published child roots too weakly and
  did not embed enough identity metadata to prevent a calibration score from
  one seed being paired with SBC scores from another. This was found before any
  Phase 6 data, checkpoint or official identity existed.
- The release gate now resolves the exact atomic Stage A parent and validates
  the Stage B parent contract; score artifacts bind split, seed, architecture,
  checkpoint, publication and code identities; the statistics runner rejects
  duplicate, overlapping or mixed artifacts.
- Publication free space is checked before the atomic rename. These are
  fail-closed engineering corrections, not post-result scientific changes.
- No calibration/SBC pair was generated, no checkpoint was accessed, and no
  calibration map or SBC statistic was fitted or evaluated.

## Phase 7 preregistration audit found an impossible fixed-slope analysis

- The immutable final generator context `sie_truth_epl_assumed` carried a
  legacy analysis label `epl_external_shear_fixed_slope_2.08`.
- The selected conditional NSF accepts only a two-value lens-family condition;
  it has no deployable density-slope input. A fixed-slope counterfactual could
  not be evaluated by this estimator.
- The issue was found before any final-evaluation system was materialized or
  unsealed and before any checkpoint, calibration map or final metric was
  accessed.
- RC.6 changes only downstream analysis interpretation. The legacy generator
  namespace and finalized commitment remain byte-identical; EPL conditioning
  marginalizes the frozen training slope prior, and family marginalization is
  an exact equal-density SIE/EPL mixture.
- Final execution remains closed. The finding is a prospective contract
  correction, not a scientific result or post-result threshold change.

## Phase 7 baseline audit found an undefined importance weight

- The inherited likelihood-gold gate was modeled on DINGO-IS, which uses a
  proposal over the complete parameters entering its likelihood.
- The selected NSF has only two target dimensions. It does not provide a
  density for the full lens, source, dynamics, BBH and image-selection nuisance
  state, so a full simulator-likelihood `p/q` cannot be evaluated.
- No importance-sampling run failed; the statistic was undefined by model
  construction. The gap was found before reference-bank or final-data access.
- RC.7 prospectively removes the exact/gold/IS-efficiency claim and freezes a
  clearly labeled non-neural simulation reference. SBC and held-out coverage
  are unchanged.
- Treating the approximate kNN/KDE reference as an exact likelihood is a hard
  stop. Scientific execution remains separately gated.

## Phase 7 execution-stack audit found incomplete downstream identity binding

- The first pure inference draft could validate the sealed final parent and
  compute in-memory metrics, but did not yet implement the complete
  checkpoint-to-score execution path.
- The Phase 6 statistics summary also omitted the hashes of the calibration
  map, independent SBC summary and independent calibrated-coverage files.
  A later final gate could therefore have bound the run summary without
  cryptographically binding every product it consumed.
- This was found before final materialization, checkpoint access, calibration
  fitting or final inference. The runner now binds 45 seed/namespace outputs,
  and the Phase 6 summary records all downstream hashes.
- This is a prospective engineering hardening, not a failed scientific run or
  a change to RC.5/RC.6/RC.7 metrics and thresholds.

## Training pre-execution audit caught authorization-envelope mismatches

- The terminal-rung runner and future architecture-grid runner each replaced
  the shared engine status with a task-specific status and omitted the two
  generic publication/commitment booleans required by the shared training
  engine. Their first future fit would therefore have stopped before its first
  optimizer step despite a valid external authorization.
- The issue was found while Stage B was still materializing. No 65k membership
  was resolved, no optimizer or checkpoint existed and no scientific metric was
  observed.
- All probe rungs and architecture candidates now construct one typed engine-
  level authorization envelope; task-specific evidence is added beside, rather
  than substituted for, that common contract. Regression checks exercise both
  identities through the same engine validator used immediately before
  optimizer construction.
- Model, data, optimizer, batch geometry, sample order and scientific stopping
  rules are unchanged. This is a prospective software-integration correction.

## Learning-curve pre-execution audit found a missing exact-count assertion

- The paired comparator required identical validation IDs between rungs and
  seeds, but did not independently require the frozen count of 6,144 systems.
  A consistently truncated set could therefore have reached the terminal
  decision path.
- The completed 16k/32k evidence contains all 6,144 cases and is unchanged.
  Stage B was still materializing and no 65k optimizer or decision existed
  when the omission was found.
- Both learning-curve comparisons now reject any count other than 6,144, and a
  regression removes one row from an otherwise valid three-seed terminal-rung
  fixture. Thresholds, bootstrap, membership and decision semantics are
  unchanged.

## Stage B independent verifier required the synchronized source package

- The first post-publication verifier invocation used the old generator
  environment without the newly added `gwlens_mm.training` package and exited
  at import time.
- It did not open or modify Stage B or the combined publication.
- The same read-only verifier was rerun against the synchronized authoritative
  repository with `PYTHONPATH=src` and passed all count, identity, manifest,
  shard, unit-weight, free-space and group-disjointness checks.
- Future immutable training execution uses a separately hashed wheel, so this
  verifier-only launch issue does not weaken the training environment gate.

## The first 65k probe exposed five numerical source-waveform pathologies

- Seed 0 and seed 1 stopped during input whitening before their first optimizer
  step; seed 2 was terminated as part of the fail-closed launcher shutdown.
- The immediate failing system contained a single IMRPhenomXPHM polarization
  bin near 37.86 Hz with amplitude of order unity while neighboring bins were of
  order 1e-24. Its recorded network SNR was consequently 4.37e22.
- An exhaustive read-only audit regenerated source polarizations for all 71,680
  published train and validation records. Exactly five train systems had
  peak/positive-in-band-99.9%-quantile ratios above 10; validation had none.
- Four failures had gross SNR outliers. A fifth ratio of 12,983 had network SNR
  195.7, proving that an SNR-only filter is insufficient. Every retained record
  had ratio at most 1.705.
- One failure belongs to the frozen 16k subset and both Stage A failures belong
  to 32k. The previous learning-curve decision is superseded. No 65k checkpoint
  or metric exists.
- Original publications and the failed training root are retained immutable.
  Clipping, row deletion and in-place replacement are forbidden. The versioned
  correction uses deterministic pre-selection rejection plus an exact-count,
  fresh-identity replacement overlay.
- The authorized overlay subsequently completed and passed independent
  closeout. Exactly five replacements restore the frozen counts; their source
  spectral ratios are below 1.096, arrays satisfy exact decomposition, weights
  are one and all grouped identities are disjoint. This repairs the data view,
  not the superseded learning-curve result; no optimizer was authorized.
- The subsequent corrected 16k/32k rerun completed all six fresh fits with zero
  process failures. Its independently replayed decision is
  `continue_to_train_65k`. This resolves the invalid learning-curve evidence but
  does not revive the failed pre-correction 65k run or authorize a 65k optimizer.
- A prospective audit then found that the still-unmaterialized calibration/SBC
  and final-evaluation builders did not yet inject the frozen source-waveform
  rejection field. No downstream pair existed. Typed builders now apply the
  exact rule to every future IMRPhenomXPHM namespace, with hash-bound addenda
  preserving the original configurations and final commitment byte-for-byte.
  The SEOBNRv4PHM mismatch namespace is explicitly excluded from the
  approximant-specific ratio rule rather than silently applying an unreviewed
  threshold.
- A second prospective audit found that the historical final commitment bound
  generator commit `bc02054c...`, while the new rejection helper necessarily
  lives in later code. The old runner required those identities to be equal,
  making compliant materialization impossible. No final identity or pair
  existed. The runner now preserves the old commitment as evidence and requires
  a future authorization/release certificate to bind a narrowly revised
  generator plus the addendum. The corrected train reference is also resolved
  as an overlay, preventing the five excluded base systems from re-entering
  leakage checks.
- The corresponding Phase 6 audit found that its future calibration/SBC
  leakage check still streamed only the original Stage A and Stage B parents.
  This could omit the five replacement identities and describe the superseded
  logical train set. No calibration/SBC identity or pair existed. The release
  path now resolves the four-parent correction contract first and checks exact
  65,536-train plus 6,144-validation corrected membership before execution.

## Phase 7 report and metadata reader drifted on the EM-cell field

- The original RC.7 stack report stated that metadata-only examples retained
  the exact Parquet `em_cell`, but the current reader still returned
  `em_cell=None` from `metadata_example()`.
- No scientific reference bank or validation/final query had been opened, so
  no neighbor, metric or result was affected.
- The typed published reader now uses one partition-label accessor for
  metadata, calibration/SBC and full-strain paths. A non-optional regression
  exercises the metadata path without Zarr, and the reference index rejects a
  missing cell, duplicate bank/query IDs, split overlap and sparse strata.
- The correction changes only offline stratification metadata. It does not
  alter model inputs, the selected standardizer, frozen RC.7 distance, KDE or
  scientific execution gate.

## The corrected 65k probe remained decisively data limited

- This is a preregistered scientific stopping result, not an execution failure:
  all three launchers returned zero and produced valid checkpoints, summaries
  and development metrics.
- The 32k-to-65k NLP improvement was 0.201437 nat per target dimension, with
  paired-bootstrap 95% interval [0.191498, 0.211788]. The saturation upper-bound
  requirement was below 0.01.
- No seed passed every development EM-cell tolerance. The frozen extreme-
  relative-magnification internal view also contained only 40 cases versus the
  required 128.
- The exact decision is `stop_data_limited_and_new_preregistration`; replay was
  byte-identical. No calibration or final-evaluation case was accessed.
- RC.4 forbids automatic extension above 65,536 and the architecture execution
  gate requires `lock_train_65k`. Both paths are closed until a new scientific
  contract receives explicit human review.
- Human review subsequently froze `1.2.0-rc.1` prospectively. It retains this
  failure unchanged, adds one terminal 131k resource-capped rung and fixes the
  independent development-tail count before any new data exist. It does not
  reinterpret 65k as saturated or open execution by itself.

## The original terminal-tail shard layout could not meet its worker cap

- The 32-worker terminal run successfully completed and atomically published
  all 65,536 new train systems before entering the development-only tail pool.
- The first high-absolute-magnification tail namespace used the frozen single
  128-case shard. After 3.29 active hours it had 91,839 attempts and only 15
  partial accepted cases; no shard or tail parent was complete.
- The measured-rate projection was about 55 cases at the 12-hour worker cap.
  Even the 95% upper acceptance-rate bound projected about 81, with only
  approximately 1.2e-7 probability of completing 128 by the cap.
- The process was stopped early rather than spending another nine hours to
  reproduce the inevitable resource failure. Its 85,707,721-byte partial tree
  is immutable, unpublished and excluded from every result.
- Recovery changes only the physical partition to 32 four-case shards per
  stratum. Counts, target, conditional strata, root seed, weights and all
  scientific gates remain unchanged; the train increment is read-only.
