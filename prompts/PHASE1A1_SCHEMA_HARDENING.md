# Phase 1A.1 — Harden the v2 schema before smoke generation

Work only on `phase1/physics-schema`. This is a narrow human-review
remediation task.

Do not generate waveform data, execute the smoke configuration, train a model,
download GWTC/GWOSC data, install lenstronomy, modify legacy files, or change
the established magnification, Fourier, Morse, or secondary-over-primary
conventions unless a failing physics test proves an inconsistency.

Read `AGENTS.md`, the Phase 1A report, physics/schema/input/split/provenance
documents, and ADR-001 first. Fix the following before materializing v2 data.

## 1. Identified EM image astrometry

Replace parallel positional/covariance tuples with a typed per-image
`ImageAstrometryObservation` containing image ID, position, and covariance.
Require unique IDs, membership in physical lens images, support selected and
non-selected images, no index-based identity, and null plus modality mask for
missing astrometry. Update serialization, examples, JSON boundary, policies,
docs, and negative tests for unknown/duplicate IDs, invalid covariance, and
implicit position-only input.

## 2. Detector-specific noise provenance

Replace two image-only segment IDs with explicit image-by-detector references
aligned to selected images and H1/L1/V1. Each available slot requires a
nonempty segment ID; each unavailable slot requires none. Detector order and
mask must agree. Every non-null segment enters grouped split validation.
Synthetic Gaussian identifiers must be labeled as synthetic and must remain
capable of later representing GWOSC detector/GPS products.

## 3. Timing observation product

Replace the bare observed time-difference float with `TimingObservation` carrying
value, uncertainty, measurement method, and optional reference. Truth and
observation remain separate. Exact truth-delay aliases stay denylisted.
Uncertainty is finite and strictly positive except for an explicitly declared
deterministic control. Update policy, schema, example, smoke observation model,
docs, and tests.

## 4. Complete non-selected image accounting

Enforce that disjoint unselected and censored image sets exactly equal all
physical image IDs minus the selected pair. Add positive and negative
multi-image tests. No physical image may disappear silently.

## 5. Primary-definition semantics

Validate: earliest primary arrival is not later than secondary; brightest
primary absolute magnification is not lower; minimum-image primary has minimum
Morse class; catalog anchor has no arrival/brightness constraint. Use documented
tolerances and test inconsistent metadata.

## 6. Detector-mask array semantics

Document and test small in-memory arrays only: unavailable slots are zero in
noisy, clean, and noise; available slots satisfy noisy approximately equals
clean plus noise; masks are mandatory model inputs; NaN is not the missing-slot
representation. Do not generate a dataset.

## 7. Manifest completeness

Add planned pair count. A complete dataset requires accepted equals planned,
attempted greater than or equal to accepted, and every published artifact
complete with SHA-256 and byte size. Pending staging artifacts cannot occur in
a published complete manifest. Add negative tests.

## 8. Field-specific EM validation

Require Einstein radius and velocity dispersion greater than zero, redshifts
nonnegative, and all uncertainties finite and positive. Document whether
observed lens redshift below source redshift is a hard constraint, warning, or
quality flag and why.

## 9. Schema version

Increment `2.0.0-alpha.1` to `2.0.0-alpha.2` in constants, examples, generated
schema, smoke configuration, docs, tests, and migration notes. Record that
alpha.1 was metadata-only and never materialized, so no physical migration is
required.

## 10. Acceptance criteria

All old and new tests, Ruff, mypy, and package build must pass. EM astrometry
must be ID-keyed; noise provenance image-detector keyed; timing typed; every
extra image accounted; primary semantics checked; missing detector slots
zero-filled and tested; complete manifests internally complete; schema version
alpha.2; smoke execution remains disabled; no data, fabricated solver result,
legacy edit, or large artifact is permitted.

Create `docs/reports/PHASE1A1_SCHEMA_HARDENING_REPORT.md`, update project state,
decisions, failures, schema/examples/tests, inspect status/diff, commit with
`fix: harden v2 observation and provenance schema`, push, and stop before
Phase 1B.
