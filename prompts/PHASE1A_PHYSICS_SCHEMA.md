# Phase 1A — Physics API, v2 schema and leakage safeguards

Execute Phase 1A only.

Read first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/DECISIONS.md`
- `docs/FAILURES.md`
- `docs/DATA_PATH_AUTHORITY.md`
- `docs/LEGACY_DATA_DECISION.md`
- `docs/LEGACY_CODE_REUSE_DECISION.md`
- `docs/audits/PHASE0_REMOTE_AUDIT.md`
- `docs/audits/PDF_TO_CODE_DATA_TRACEABILITY.md`
- `docs/reference/baseline_realobs_quality_mu0_report_zh.pdf`

Do not train a model. Do not generate waveform datasets. Do not download
GWTC or GWOSC data. Do not modify legacy remote files. Do not start manuscript
writing. Do not browse the web unless a primary official package API must be
verified. Do not install a large scientific environment on Vultr.

The objective is to establish the scientifically correct and testable
foundation for v2 before any data generation.

## 1. Repository and environment check

Confirm branch `phase1/physics-schema` and a clean starting point apart from
the gate/prompt changes. Inspect existing packaging and tests without broad
reorganization. If needed, create a minimal `src/gwlens_mm/` package. Core
tests must run without Bilby, LALSuite, GWpy, GPUs, or heavy solvers. Optional
solver contracts may run on AutoDL but must not generate data.

## 2. Physics quantity conventions

Define explicit types, names, and documentation for signed and absolute
magnification, strain-amplitude factor, relative flux magnification, relative
strain amplitude, parity, Morse index, arrival time, signed/absolute delay,
image ordering, apparent/physical luminosity distance, source/image-plane
coordinates, Einstein scale, external convergence/shear, and lens family.

Do not use ambiguous bare `mu0`, `mu1`, `A21`, or `td`. Distinguish
`mu_signed`, `mu_abs = abs(mu_signed)`, and
`amplitude_factor = sqrt(mu_abs)`. Define relative flux as secondary over
primary and relative strain amplitude as its square root. Define the selected
pair's primary role explicitly; earliest, brightest, minimum, and catalog
anchor are not interchangeable.

Represent all physical images separately from a selected observed GW pair,
including extra undetected or censored images. Create
`docs/PHYSICS_CONVENTIONS.md`, typed package models, and consistency tests.

## 3. Fourier and Morse conventions

Document and implement the positive-frequency convention, time-shift sign,
minimum/saddle/maximum phase factors, Morse representation, and negative
frequency handling. Test all three Morse classes, conjugate symmetry,
time-shift direction, and image ordering. Independently test any legacy idea.

## 4. SIS analytic control

For `0 < y < 1`, implement stable relations equivalent to:

```text
mu_plus_signed = 1 + 1/y
mu_minus_signed = 1 - 1/y
mu_plus_abs = abs(mu_plus_signed)
mu_minus_abs = abs(mu_minus_signed)
relative_flux_magnification = mu_minus_abs / mu_plus_abs
relative_strain_amplitude = sqrt(relative_flux_magnification)
mu_plus_abs = 2 / (1 - relative_flux_magnification)
mu_minus_signed = -2 * relative_flux_magnification /
                  (1 - relative_flux_magnification)
```

Test boundary-near and moderate values, invalid domains, finite transformed
coordinates, round trips, and legacy fixture agreement. Never silently clip.
Provide log absolute magnification and logit-relative representations. SIS is
an analytic control, not the general model.

## 5. General lens-solver interface

Define an implementation-independent protocol for SIS, SIE+external shear,
elliptical power-law+external shear, and optional external convergence/model
discrepancy. Solver output contains every physical image: ID, position, signed
magnification, arrival/Fermat ordering, parity, Morse class, and validity.
Pair selection separately records image IDs, reason, detector visibility,
extra images, and censoring. Fully implement SIS and lightweight non-SIS
adapter contracts. Document in `docs/LENS_SOLVER_INTERFACE.md`. Do not generate
a dataset.

## 6. Privileged variables and model input policy

Create `docs/PRIVILEGED_INPUT_POLICY.md`, a machine-readable denylist and
allowlist, and a fail-closed validator. Deny exact source-plane coordinates,
true magnifications/ratios, latent lens/source/noise parameters, exact values
where noisy observations are intended, clean signal, isolated noise, optimal
SNR, simulation-only diagnostics, group IDs, and proposal/prior weights.
Distinguish model input, training target, grouping/provenance metadata, and
privileged diagnostic. Tests must reject forbidden/suspicious aliases and
unknown fields.

## 7. V2 logical data schema

Design a logical schema before physical storage, separating:

- pair index and selected image IDs;
- source truth and waveform metadata;
- lens truth and all physical images;
- image truth;
- GW observations and array references;
- noisy EM observations, covariance, masks, and censoring;
- complete provenance;
- separate noisy, clean, and noise products.

Do not repeat a full time array per event. Create `docs/V2_SCHEMA_SPEC.md`,
versioned models, generated JSON Schema where practical, one metadata-only
example, round-trip tests, and a migration policy.

## 8. Grouped split policy

Create `docs/SPLIT_POLICY.md` and define train, validation/model-selection,
calibration, IID test, balanced-tail diagnostic, lens-family OOD,
parameter-region OOD, real-noise test, and waveform/PSD-mismatch test.
Validators must reject source, lens/system, selected pair, noise segment, or
augmentation-parent overlap. Never use row-wise random splitting. Test
deliberately corrupted examples.

## 9. Proposal and evaluation prior

Keep the simulation proposal, training resampling distribution,
astrophysical evaluation prior, and balanced diagnostic distribution
separate. Record log probabilities, population/importance weight, validity,
and clipping information as provenance/evaluation fields, never deployable
inputs. Define interfaces only; do not choose the final prior.

## 10. Storage architecture decision

Create `docs/adr/ADR-001-v2-storage-format.md`. Compare Zarr+Parquet,
sharded HDF5+Parquet, and any justified alternative for resumability,
chunking, partial reads, checksums, schema evolution, writers, recovery,
AutoDL compatibility, and dependency stability. Select Phase 1B storage.
Estimate 48, 10,000, and 100,000 one-second float32 pairs, with two images,
three detector slots, and separate noisy/clean/noise arrays. Do not generate.

## 11. Seed and manifest design

Implement stable child seeds for source, lens, pair selection, detector noise,
EM measurement noise, missing modalities, and augmentation. Implement config
hashing, dataset ID generation, manifest serialization/validation,
duplicate-ID detection, and checksum interfaces. Document in
`docs/PROVENANCE_AND_SEEDS.md`.

## 12. Smoke specification

Create but do not execute `configs/data/v2_smoke.yaml`, specifying exactly 48
accepted pairs: 24 SIS, 12 SIE+shear, 12 elliptical power-law+shear. Specify
two selected images, H1/L1/V1 slots and masks, one-second float32 arrays at
4096 Hz, separate noisy/clean/noise, noisy EM observations/covariance,
missing modalities, deterministic grouped IDs, and accepted-count control.

## 13. Tests and quality gates

Test magnification conversions, SIS identities/domains, Morse phases,
time-shift convention, image ordering, multi-image selection, input policy,
schema round trip, covariance/masks, grouped split leakage, seeds, config hash,
manifest validation, and storage estimates. Configure pytest, Ruff, typing,
and lightweight GitHub Actions where practical. Optional solver tests must be
marked separately.

## 14. Required outputs

At minimum create:

- `docs/PHYSICS_CONVENTIONS.md`
- `docs/LENS_SOLVER_INTERFACE.md`
- `docs/PRIVILEGED_INPUT_POLICY.md`
- `docs/V2_SCHEMA_SPEC.md`
- `docs/SPLIT_POLICY.md`
- `docs/PROVENANCE_AND_SEEDS.md`
- `docs/adr/ADR-001-v2-storage-format.md`
- `docs/reports/PHASE1A_REPORT.md`
- `configs/data/v2_smoke.yaml`
- physics/schema/policy/provenance modules and machine-readable policies;
- tests and a metadata-only schema example;
- updated project-state, decision, failure, and experiment-registry records.

No waveform arrays may be created in Phase 1A.

## 15. Acceptance criteria

Phase 1A completes only when lightweight tests pass, SIS round trips pass,
forbidden aliases are rejected, schema round trips losslessly, corrupt split
examples fail, seed/hash behavior is deterministic, physical images and
selected pairs are separate, storage has an ADR, smoke configuration validates
without execution, no large data/checkpoint enters Git, no legacy file changes,
and state documents are current.

At completion inspect the diff, validate, commit with
`feat: define v2 lensing physics and dataset schema`, push the Phase 1 branch,
and stop before Phase 1B.
