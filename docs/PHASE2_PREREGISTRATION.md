# Phase 2 statistical preregistration

Status: `1.0.0-rc.3`, source-plane execution contract approved by the human
project owner; this file alone authorizes no execution.
The machine-readable authority is
`configs/statistics/phase2_preregistration.yaml`.

## Estimand and claim boundary

For a candidate galaxy-scale lensed GW pair with declared EM information,
estimate the model-conditional joint posterior
`p(log|mu_primary|, log|mu_secondary| | GW, EM, selection, M)`, where `M`
includes the lens family, cosmology, external-convergence treatment, population
and observation models. This is not a model-independent absolute-
magnification measurement.

Derived targets are the log-magnification difference, relative absolute macro
magnification and relative strain amplitude. An observed EM flux ratio is not
identified with the macro ratio without explicit variability, extinction,
microlensing and measurement models.

## Proposal, evaluation population and source support

The broad proposal covers SIE+shear and EPL+shear equally, `z_l=0.1–1.0`,
`z_s=0.5–3.0` with `z_s-z_l>=0.1`, Einstein radius `0.3–3.0 arcsec`, axis
ratio `0.4–1.0`, shear `0–0.15`, EPL slope `1.6–2.5`, and external convergence
`-0.15–0.15`. The balanced evaluation population is a benchmark, not an
inferred astrophysical rate population.

The BBH joint support is normalized rather than rejection-sampled: draw `m1`
from the declared power law, then `q` uniformly on
`[max(0.25,10/m1),1]`. Thus `m2>=10 Msun` by construction and the conditional
normalization is present in proposal and evaluation log densities.

RC.3 defines source position before lens-multiplicity and detection selection.
With `u=beta/theta_E`, both coordinates are independently uniform on
`[-2.5,2.5)`. Relative to `d beta_x d beta_y`, the exact normalized log density
is `-log(25)-2 log(theta_E)`. Proposal and evaluation use the same source-plane
factor, so it cancels exactly in their importance ratio. Multiple imaging and
detection remain explicit selection events; no numerically estimated caustic
area is inserted into a supposedly exact density.

The solver contract takes the validated union of the EPL/SIE analytical solver
and a deterministic grid search with an `8 theta_E` window and `0.02 theta_E`
grid separation. Candidates must pass a `1e-8 arcsec` lens-equation residual;
strongly demagnified central images are retained. Phase 3A compares boundary
fixtures against an analytical solver at twice the angular sampling and a
`12 theta_E`, `0.005 theta_E` grid. Failure means the declared source support
is not qualified and stops generation.

## External convergence and dynamics

External convergence enters the forward model through
`lambda_mst=1-kappa_ext`. Source coordinates and Fermat/time-delay differences
scale by lambda, image positions stay invariant, and signed magnifications
scale by `lambda^-2`. The alpha.3 schema stores a deployable environment
observation as posterior mean and standard deviation, never latent truth.

Velocity dispersion uses Lenstronomy 1.13.6 Galkin with a circularized
spherical power-law dynamics mass model, Hernquist tracer light,
Osipkov--Merritt anisotropy, a one-arcsecond circular aperture, and 0.7-arcsec
Gaussian seeing. The exact light/anisotropy priors, sample count and Phase 3A
convergence tolerance are frozen. Direct inversion from Einstein radius alone
is forbidden.

## GW, selection and EM observation models

The baseline is eight seconds at 2048 Hz in H1/L1/V1, IMRPhenomXPHM, and the
detector-specific synthetic Gaussian curves named and hashed in the config.
SEOBNRv4PHM supplies waveform mismatch. PSD mismatch separately uses the
verified zero-detuned-high-power H1/L1 curve while holding V1 fixed.

Both selected images must pass the synthetic selection surrogate: network SNR
at least 10, at least two detectors, and per-contributing-detector SNR at least
4. Selection statistics are privileged and never deployable inputs.

Eight EM cells explicitly enumerate modalities, errors, spectroscopic or
photometric redshift mode, timing uncertainty, environment state and
kinematics state. Informative/weak environment observations use external-
convergence posterior standard deviations 0.02/0.06; unavailable is null and
masked. The equal cell mixture is a stress benchmark, not a survey frequency.

## Splits and leakage control

Fixed accepted counts are:

- train 65,536; validation 8,192;
- calibration-fit 6,144; SBC-diagnostic 2,048;
- IID test 16,384;
- balanced-tail 4,096; cross-family misspecification 4,096;
- parameter-region OOD 4,096;
- waveform mismatch 2,048; PSD mismatch 2,048;
- engineering qualification 4,096; real-noise zero.

The total remains 118,784. Calibration-fit and SBC-diagnostic groups are
disjoint in source, lens, physical system, pair, noise segment and augmentation
parent. SBC never fits a calibration map. IID is untouched until architecture
and calibration are frozen.

## Estimator and seed aggregation

The primary estimator is a mask-aware conditional neural-spline flow combining
a shared-weight image/detector 1D ResNet and an image-keyed DeepSets EM encoder.
The 2-by-2 architecture grid and three seeds produce at most 12 fits.

Architecture selection uses mean validation negative log probability across
all three seeds; median is a robustness statistic. No best seed is selected.
All three selected-architecture seeds proceed to IID/OOD and all individual
values, mean and dispersion are reported. Exact ties choose fewer parameters.
GW-only and EM-only ablations receive the same tuning budget.

## Calibration and likelihood gold subsets

Primary coverage levels are 50%, 80%, 90% and 95% for each log magnification
and the joint density-rank region. Wilson intervals and raw counts are always
reported. The frozen marginal, joint, EM-cell and tail tolerances are encoded
in the config; no average metric overrides a failed preregistered stratum.

Post-hoc correction uses only `calibration_fit`. SBC uses 1,024 deterministic
realizations drawn only from `sbc_diagnostic`. A 256-case validation gold subset
may trigger development changes. A separate 256-case IID gold subset is
reported once after freeze and cannot trigger retuning; failure downgrades the
claim or requires a new preregistration version.

## Diagnostic, OOD and mismatch definitions

Balanced-tail cases are priority-assigned to four fixed strata: absolute
magnification at least 20; selected-image min/max ratio at most 0.10; secondary
network SNR in `[10,12)`; or extreme profile/environment values. Cross-family
tests fix two wrong-family and two family-marginalized cells. Parameter OOD
fixes slope outside training, `q=0.25–0.40`, shear `0.15–0.25`, and
`|kappa_ext|=0.15–0.25`. Exact interval endpoints and counts are in YAML.

## Storage and resume boundary

Raw arrays require 140,123,308,032 bytes. Peak disk separately includes 10%
metadata/chunk overhead, 5% retained-failure cap, 20 GB run/checkpoint/cache
reserve, and one active shard: 181,292,799,182 bytes. Publication is an atomic
same-filesystem rename, not a second full copy. Pre-launch free space must
exceed 281,292,799,182 bytes, preserving a 100 GB post-peak floor.

Phase 3A remains separately gated at 4,096 engineering pairs, 4,831,838,208
raw bytes and 5,807,608,883 projected peak bytes. It must measure acceptance,
throughput, waveform boundaries, whitening, dynamics convergence, actual disk
amplification and byte-identical resume before any larger authorization.

## Hard stops

No data generation, training, GWOSC/GWTC download, or Phase 3A authorization
follows from this release candidate or its merge. Privileged-input access,
group leakage, disconnected external convergence, dynamics shortcuts,
undeclared waveform/PSD versions, invalid weights, incomplete manifests,
waveform truncation or resume mismatch are hard failures.
