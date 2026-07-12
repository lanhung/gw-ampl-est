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
