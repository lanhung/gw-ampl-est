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
