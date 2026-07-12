# Phase 1A.1 schema-hardening report

Date: 2026-07-12
Branch: `phase1/physics-schema`
Human-review task checkpoint: `6b62dd4`

## Outcome

The human-review remediation is complete. Schema version
`2.0.0-alpha.2` replaces metadata-only alpha.1 before any v2 arrays or Parquet
records were materialized. No physical migration or data regeneration was
needed because no v2 dataset exists.

No waveform was generated, the execution-disabled smoke configuration was not
run, no model was trained, no GWOSC/GWTC product was downloaded, no external
solver result was claimed, and no legacy file was modified.

## Blocking issues resolved

### Image-ID-keyed EM astrometry

Parallel position/covariance tuples were replaced by
`ImageAstrometryObservation(image_id, position_arcsec, covariance_arcsec2)`.
Validation requires unique IDs drawn from all physical lens images, so optical
astrometry can include selected and non-selected images without relying on
tuple order. Null astrometry plus a false modality mask represents missingness;
truth is not imputed. Old positional fields are rejected by both the typed
parser and generated JSON boundary.

### Image-detector noise provenance

Provenance now contains six ordered `DetectorNoiseReference` objects for
primary/secondary crossed with H1/L1/V1. Each reference explicitly carries
image ID, detector, availability, optional segment ID, and noise source.
Available slots require distinct nonempty IDs; unavailable slots require null
IDs and source `unavailable`; availability must equal the GW detector mask.
Every non-null ID is included by `SplitAssignment.from_v2_record()`.

The example uses `synthetic_gaussian_design_psd` IDs. They are not described as
GWOSC noise segments.

### Typed timing observation

The bare observed delay float was replaced by `TimingObservation` with value,
standard deviation, measurement method, reference, and deterministic-control
flag. Ordinary observations require positive uncertainty. Zero uncertainty is
accepted only when the record explicitly declares a deterministic control.
Truth-delay names and aliases remain fail-closed denylist entries.

### Complete extra-image accounting

The disjoint union of unselected and censored IDs must now equal every physical
image not in the selected GW pair. Missing, invalid, overlapping, or duplicate
statuses fail validation.

## Additional consistency hardening

- `earliest_arriving`, `brightest`, and `minimum_image` primary definitions are
  checked against physical-image truth; catalog anchor has no such constraint.
- Small in-memory float32 validators require unavailable noisy/clean/noise
  slots to be exactly zero, reject NaN, and require
  `noisy = clean + noise` for available slots.
- Complete manifests require accepted count equal planned count and all
  published artifacts complete with SHA-256 and byte size.
- Einstein radius and velocity dispersion must be positive; redshifts must be
  nonnegative; all uncertainties remain finite and positive.
- `z_l < z_s` is an observed point-summary quality flag, not a hard rejection,
  because broad photometric posteriors can overlap.
- The generated JSON boundary now defines astrometry, timing, and noise objects
  and rejects removed alpha.1 field shapes.
- CI checks the generated schema artifact and package build in addition to
  pytest, Ruff, and mypy.

## Verification

Completed local checks:

```text
pytest: 93 passed
Ruff: all checks passed
mypy: success, no issues in 13 source files
python -m build: isolated sdist and wheel built successfully
AutoDL: alpha.2 metadata round trip and detector-noise contract passed
```

Build artifacts were written under `/tmp/gwlens-mm-build-alpha2-final`, not
Git. The build emitted only a non-fatal warning that the repository lacks a
conventional package README; both artifacts completed.

The final lightweight sync excluded build products and data. AutoDL parsed the
alpha.2 example, found five available detector-qualified noise IDs, confirmed
the smoke output remained empty, and found no array/checkpoint artifacts in the
repository copy.

## Remaining limitations

- No `lenstronomy` or other external non-SIS numerical fixture was executed;
  this remains Phase 1B solver-environment work.
- Zero-fill/decomposition checks use small in-memory arrays only. End-to-end
  shard publication and resumability remain untested until Phase 1B.
- The JSON boundary covers the changed association objects and forbids removed
  fields, while the Python typed layer remains authoritative for cross-record
  physics and mask invariants.
- No PR-triggered GitHub Actions result is claimed here. The pushed branch is
  ready for a PR and formal CI review.

## Gate recommendation

Phase 1A plus Phase 1A.1 is ready for human/PR review. Keep
`execution_authorized: false` and do not begin Phase 1B from this branch. After
CI and schema-diff approval, merge to main and create a new Phase 1B branch.
