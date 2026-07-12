# Phase 1B — Implement, generate and validate the v2 engineering smoke dataset

Execute Phase 1B only on `phase1b/v2-smoke` after reading `AGENTS.md`, both
Phase 1A reports, physics/solver/input/schema/split/provenance documents,
ADR-001, and `configs/data/v2_smoke.yaml`.

Do not train a model, download GWTC/GWOSC data, exceed 48 accepted pairs,
modify legacy data, describe synthetic Gaussian noise as real detector noise,
or weaken reviewed physics conventions. This dataset is an engineering
artifact forbidden from scientific training, calibration, IID/OOD testing, or
performance claims.

## 1. Branch and authorization

Confirm main contains merged PR #1, branch and tree are correct, schema is
`2.0.0-alpha.2`, and output is below
`/root/autodl-tmp/lensing-4/data_v2/smoke`. Change
`execution_authorized` to true only on this branch and record the authorizing
Git commit in the manifest. Never write under a legacy root.

## 2. Pinned AutoDL environment

Inspect Python, CUDA, disk, and packages. Create reproducible pins for NumPy,
SciPy, pandas, PyArrow, Zarr v2 (not v3), compatible numcodecs, lenstronomy,
Bilby, LALSuite/lalsimulation as required, Astropy, and this package. Record
exact versions and install commands. Before data generation, run deterministic
SIS, SIE+shear, and EPL+shear numerical fixtures. Do not fabricate results.

## 3. Solver numerical contracts

Return all physical images with unique stable IDs, finite position, finite
nonzero signed magnification, consistent parity/Morse type, finite Fermat/time
ordering, valid selected IDs, complete extra-image status, and solver version.
Test a SIS double, SIE double, SIE quad, EPL quad, and a pair not equal to the
first two solver outputs.

## 4. Generator

Implement a modular resumable configuration-driven generator with accepted
counts exactly 24 SIS, 12 SIE+shear, and 12 EPL+shear. Record every rejected
attempt/reason and derive stable source, lens, pair, waveform, detector-noise,
EM-noise, and missing-modality IDs/seeds. Privileged truth is never deployable.

## 5. GW construction

Use two independently centered one-second windows. Galaxy-scale delay lives in
metadata, never as a multi-day shift inside one tensor. Each image has its own
geocentric arrival time and detector response. For each image, construct the
same-time unlensed source reference, apply `sqrt(abs(mu_signed))`, apply Morse
phase, project at that image time, add synthetic Gaussian design-PSD noise, and
store separate float32 clean/noise/noisy arrays of shape `(2,3,4096)`.
Unavailable slots are zero; available slots satisfy noisy=clean+noise.

## 6. Amplitude preservation

Never interpret a raw cross-image detector-strain ratio as magnification because
antenna response changes with arrival time. For each image/detector, compare the
clean lensed projection against an unlensed reference with identical source,
arrival time, sky, response, PSD, and preprocessing. Report expected scaling
before projection, after matched-response projection, and after preprocessing;
test Morse phase separately.

## 7. EM simulator

Generate noisy image-ID-keyed astrometry, lens center, Einstein radius,
lens/source redshift, velocity dispersion, uncertainties, and deterministic
missing masks. Cover complete EM, missing velocity dispersion, missing source
redshift, selected/non-selected astrometry, and a broad-redshift quality-flag
case. Never replace missing observations with truth.

## 8. Detector/noise provenance

Provide exactly six image-detector references per pair. Available slots require
unique deterministic synthetic IDs; unavailable slots require null ID/source
`unavailable`; all agree with masks and enter grouped leakage checks. Vary masks
while retaining at least one detector per image. Never label these GWOSC.

## 9. Storage and resume

Use Zarr v2 plus Parquet, unique dataset ID and staging, unique shards,
single-writer atomic manifest publication, SHA-256/bytes for every artifact,
rejection log, and no repeated time arrays. Intentionally interrupt and resume
without duplicate IDs or changed bytes. Never publish incomplete shards.
Manifest requires `dataset_purpose: engineering_smoke`,
`scientific_use_authorized: false`, planned/accepted 48, attempted >=48, and
all artifacts complete. Stop if total output exceeds 1 GB.

## 10. Validation

Validate all 48 records for schema round trip, solver/image/primary/extra-image
contracts, parity/Morse/timing, detector masks and qualified noise provenance,
decomposition/zero-fill/finite float32 arrays, astrometry/covariance/missing
masks, input policy, grouped leakage, deterministic hashes/seeds, complete
manifest/checksums, and resume. Any censored image requires a nonempty reason.

## 11. Outputs

Large artifacts remain on AutoDL. Commit only code, pins, tests, config,
manifest copy, rejection/validation summaries, small figures, runtime/storage
report, `docs/reports/PHASE1B_SMOKE_REPORT.md`, and updated state/decisions/
failures. Never commit Zarr/Parquet arrays, caches, checkpoints, or credentials.

## 12. Acceptance

Phase 1B passes only with exactly 48 at 24/12/12, numerical non-SIS fixtures,
physical-system/pair separation, all validators, matched-response amplitude
preservation, no cross-image ratio claim, resume/atomic publication, <1 GB,
engineering-only markings, no training/GWOSC/legacy changes, and no large Git
artifact. Run pytest, Ruff, mypy, build, AutoDL validator, and checksums.

Commit `feat: generate and validate v2 engineering smoke dataset`, push, and
stop.
