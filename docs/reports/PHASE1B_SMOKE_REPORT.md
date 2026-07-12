# Phase 1B engineering-smoke report

## Outcome

Phase 1B passed its engineering acceptance gate. Exactly 48 accepted pairs
were generated and atomically published on AutoDL as:

`gwlens-v2-2.0.0-alpha.2-ae86beab1c132682`

This is an engineering artifact, not a training, calibration, test, or paper
performance dataset. Its manifest sets `scientific_use_authorized` to false.

## Reproducible identity

- schema: `2.0.0-alpha.2`;
- authorization commit: `a607ddb6844cb8a9cd6761011cb545633cd4fdf1`;
- generator commit: `d7287f8cc800406cc1e727a177fd27d7ca02cddf`;
- configuration SHA-256:
  `c90de2d3f3d3faa9b999a7cf9c12657364b18d58af7f5995fec39361e3cbb3da`;
- root seed: `20260712`;
- remote publication root:
  `/root/autodl-tmp/lensing-4/data_v2/smoke/published/gwlens-v2-2.0.0-alpha.2-ae86beab1c132682`.

The isolated environment and exact freeze are copied under
`results/phase1b/`. Core pinned components include NumPy 1.26.4, SciPy 1.13.1,
Zarr 2.18.3, numcodecs 0.13.1, lenstronomy 1.13.6, Bilby 2.6.0, LALSuite
7.26.1, PyArrow 17.0.0, and Astropy 6.1.7.

## Composition and representation

| Lens family | Accepted pairs |
| --- | ---: |
| SIS | 24 |
| SIE + external shear | 12 |
| EPL + external shear | 12 |
| Total | 48 |

Every pair contains two selected GW images, three detector slots (H1, L1,
V1), 4096 float32 samples per one-second window, and separate noisy, clean,
and noise products. Detector availability varies by image. Unavailable slots
are exactly zero and available slots satisfy `noisy = clean + noise`.

All physical lens images are retained. Several four-image systems select a
non-consecutive observed GW pair; every remaining physical image is explicitly
unselected or censored, and every censored image has a reason. EM astrometry is
keyed by physical image ID and includes non-selected images. Missing source
redshift and velocity-dispersion patterns are represented by masks, never truth
imputation.

## Numerical contracts

Before materialization, deterministic fixtures passed for an SIS double, SIE
double, SIE quad, and EPL quad. The fixture suite also selected `image_0` and
`image_2` from a quad to prove that solver order is not pair identity.

Waveforms use IMRPhenomXPHM. Each image receives its own geocentric arrival
time and detector projection. The galaxy-scale delay is stored in the timing
product and never shifted inside a one-second tensor. Lensing amplitude and
Morse phase are checked against an unlensed projection with the same source,
arrival time, and detector response.

Verified maxima over all available image-detector slots:

- matched-response amplitude relative error: `1.850266557252657e-16`;
- Morse-phase relative error: `1.9938245703229228e-16`;
- identity-preprocessing amplitude relative error: `1.850266557252657e-16`.

No raw detector-strain ratio between separated images was interpreted as a
magnification ratio.

## Storage, provenance, and validation

Published arrays occupy about 9.1 MB; the complete smoke area, including
staging and retained failed evidence, occupies about 19 MB, far below the 1 GB
stop condition. Accepted-pair completion spanned approximately 33 seconds on
AutoDL, excluding environment installation and validation.

The published products are Zarr v2 arrays plus a Parquet metadata table. The
manifest records 48 planned, 48 accepted, and 48 attempted pairs; six
image-detector noise references are present per record. Artifact SHA-256 and
byte counts were independently recomputed after publication and matched.
Zarr products independently validated as `(48, 2, 3, 4096)` float32 arrays.

The official AutoDL run passed 98 tests, including optional lenstronomy
contracts. Local lightweight checks passed 95 tests with one optional solver
module skipped because lenstronomy is intentionally absent locally; Ruff,
mypy, and package build passed.

## Interruption and failures

The generator was deliberately stopped after five pairs. Their completion-file
hashes were captured, the run resumed, and those hashes were unchanged.

Two earlier staging attempts failed safely:

1. a manually mistyped, nonexistent generator commit was detected during
   provenance review;
2. a resume byte mismatch exposed that Bilby uses its own RNG rather than
   NumPy's legacy global RNG.

Neither attempt was published. Both remain marked or isolated under the new
AutoDL project root. The RNG defect was corrected by explicitly seeding Bilby's
generator, independently tested, committed, and assigned a new dataset ID.

## Scientific limitations

The artifact uses synthetic design-PSD Gaussian noise, a fixed engineering
source/lens grid, identity preprocessing, and noisily simulated EM products. It
does not validate real detector noise, astrophysical population coverage,
posterior calibration, absolute-magnification inference, lens-family
marginalization, or GWTC behavior. It must not enter later scientific splits.

## Evidence files

- `results/phase1b/smoke_manifest.json`;
- `results/phase1b/smoke_validation.json`;
- `results/phase1b/smoke_attempts.csv`;
- `results/phase1b/solver_fixtures.json`;
- `results/phase1b/interrupted_run.json`;
- `results/phase1b/pre_resume_hashes.sha256`;
- `results/phase1b/post_resume_hashes.sha256`;
- `results/phase1b/environment.json`;
- `results/phase1b/environment.freeze.txt`.

## Reproduction

From the synchronized generator commit on AutoDL:

```bash
/root/autodl-tmp/lensing-4/envs/phase1b-smoke/bin/python \
  scripts/phase1b/generate_smoke.py \
  --config configs/data/v2_smoke.yaml \
  --generator-commit d7287f8cc800406cc1e727a177fd27d7ca02cddf \
  --authorizing-commit a607ddb6844cb8a9cd6761011cb545633cd4fdf1 \
  --policy-root configs/policy
```

Re-execution against the complete publication validates and returns the
existing manifest rather than generating more pairs.
