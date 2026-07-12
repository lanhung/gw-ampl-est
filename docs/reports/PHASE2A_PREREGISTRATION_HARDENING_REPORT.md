# Phase 2.1 preregistration hardening report

## Review decision

The RC.1 review decision was **request changes**. Those changes have been
implemented as preregistration `1.0.0-rc.2`; the user-directed automatic review
finds the specified design blockers resolved and approves the design for CI
and merge. This report does not authorize Phase 3A, full generation,
training, GWOSC/GWTC access, or use of the frozen smoke data for science.

## Resolved blockers

- Calibration fitting uses 6,144 `calibration_fit` systems; SBC uses a
  group-disjoint 2,048-system `sbc_diagnostic` split and 1,024 fixed
  realizations. The combined count remains 8,192.
- A validation gold subset may guide development; a frozen IID final subset is
  report-once and cannot trigger retuning in RC.2.
- Four architectures are selected using mean validation negative log
  probability across exactly three seeds. Median is diagnostic; no best seed
  is selected, and all selected-architecture seeds must be reported.
- External convergence enters the lens solution through an explicit
  mass-sheet transformation and has informative, weak, or unavailable
  deployable environment observations in scientific schema alpha.3. Frozen
  alpha.2 records load without migration and are not regenerated.
- Velocity dispersion uses a pinned Lenstronomy Galkin forward model with
  declared mass/light/anisotropy, aperture, PSF and luminosity weighting.
  Einstein-radius inversion is forbidden.
- All eight EM cells now carry scalar uncertainties, redshift modes,
  environment and kinematics states, timing errors, modality lists and
  covariance assumptions.
- The former family-OOD and combined mismatch labels are replaced by exact
  cross-family misspecification, parameter-region OOD, waveform mismatch and
  PSD mismatch definitions. Four tail strata have pre-result boundaries.
- Conditional mass-ratio support enforces a 10-solar-mass secondary without
  rejection; normalized density and importance-weight formulas are declared.
- Peak disk use includes metadata/chunks, one active temporary shard, retained
  failed shards and run/cache reserve. Prelaunch free space must exceed
  281,292,799,182 bytes, leaving at least 100 GB after projected peak use.
- A source-linked literature matrix, bibliography and novelty statement cover
  the nearest GW-pair, Bayesian lens, NPE, multimessenger and LVK work. No
  unsupported “first” claim is made.

## Immutable configuration and arithmetic

- canonical rc.2 configuration hash:
  `a7d475150b1c01d8e539a3fd5eb8d83f2ce696c5d78125f4c435c7519803aef1`;
- total planned count including qualification: 118,784 pairs;
- raw arrays: 140,123,308,032 bytes;
- projected peak: 181,292,799,182 bytes;
- measured free space recorded at preregistration: 342,407,888,896 bytes;
- projected post-peak free space: 161,115,089,714 bytes;
- Phase 3A raw qualification estimate: 4,831,838,208 bytes;
- Phase 3A projected peak: 5,807,608,883 bytes.

## Verification

Vultr passed 115 tests with two expected Lenstronomy skips, Ruff, mypy over 17
source files, and sdist/wheel build. The generated JSON schema matches its
artifact. The build retains the previously recorded missing-README warning.

After safe no-delete synchronization, the pinned AutoDL environment passed 121
authoritative tests, including Lenstronomy solver fixtures, the mass-sheet
contracts and the new dynamics convergence contract. The dynamics fixture
returned 237.720593 km/s at 4,000 samples and 237.937421 km/s at 16,000 samples,
a relative difference of approximately 0.00091 against a 0.02 limit. AutoDL
also passed Ruff, mypy and package build.

One initial AutoDL invocation of the `pytest` console script reproduced the
known namespace-path anomaly and failed to import `scripts.phase1b`; running
the same authoritative checkout with `python -m pytest` passed all tests. The
safe sync also retains a superseded untracked Phase 2 test, so it was explicitly
ignored. Neither issue changes scientific code or results.

## Completed, failed and deferred

Completed: all RC.2 design, schema, documentation and contract-test items.
Failed scientific runs: none, because no generation or inference was executed.
Deferred: actual solver acceptance, runtime, staging amplification, whitening
statistics, and restart behavior for 4,096 qualification pairs. Those are
Phase 3A engineering measurements and require a new explicit authorization.
Full 118,784-pair production, training and real-noise/catalog studies remain
closed even if Phase 3A is later authorized.

## Automatic-review recommendation

From the publication-quality perspective, RC.2 passes review: the original
statistical leakage and physical-observation blockers are resolved. PR #3 may
merge after its required CI succeeds. Approval/merge remains separate from the
later, narrowly scoped authorization of 4,096 engineering qualification pairs.
