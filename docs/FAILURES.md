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

## Phase 2.3 finite-support clarification

- RC.3's first extreme probe found steep EPL doubles on every tested finite
  source-square boundary point. No dataset output was created.
- RC.4 corrects the claim boundary: the exact square is a deliberately
  truncated benchmark and does not purport to contain the complete strong-lens
  cross-section. Primary/reference solver agreement replaces the impossible
  no-multiple-images-at-boundary requirement.
