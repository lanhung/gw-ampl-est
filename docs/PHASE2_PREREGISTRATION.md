# Phase 2 statistical preregistration

Status: `1.0.0-rc.1`, awaiting human review; no execution authorized.

## Scientific question and estimands

For a candidate galaxy-scale lensed GW pair with declared EM information,
estimate the joint posterior of
`(log_abs_mu_primary, log_abs_mu_secondary)`. The posterior is conditional on
the lens family and all declared population, cosmology, external-convergence,
selection, waveform, noise and EM observation models.

Preregistered derived quantities are the log absolute-magnification difference,
relative absolute magnification, and relative strain amplitude. An observed EM
flux ratio is not silently equated to the macro-model magnification ratio;
source variability, extinction, microlensing and its measurement model would
need explicit treatment. Signed magnification, parity and Morse index remain
distinct quantities.

## Frozen benchmark population and proposal

The machine-readable values are authoritative in
`configs/statistics/phase2_preregistration.yaml`.

The broad proposal covers SIE+shear and EPL+shear equally, lens redshift
`0.1–1.0`, source redshift `0.5–3.0` with `z_s-z_l >= 0.1`, Einstein radius
`0.3–3.0 arcsec`, axis ratio `0.4–1.0`, shear amplitude `0–0.15`, EPL slope
`1.6–2.5`, and external convergence `-0.15–0.15`.

The evaluation distribution is a balanced, literature-informed benchmark, not
an inferred astrophysical rate population. In particular, its EPL slope is a
truncated normal with mean 2.08 and standard deviation 0.16, while external
convergence remains an explicit truncated nuisance. The BBH benchmark uses a
broad source-frame mass and spin distribution and derives luminosity distance
from source redshift and the declared cosmology. It is not labeled as the
GWTC population.

Proposal draws must cover every evaluation population and tail. Proposal
metadata and weights are evaluation-only and remain denylisted as deployable
inputs.

## Observation, selection and missingness models

The baseline is an eight-second, 2048 Hz H1/L1/V1 observation using
IMRPhenomXPHM and the detector-specific frozen curves named and hashed in the
configuration. SEOBNRv4PHM is the waveform mismatch case supported by the
pinned LALSuite environment. Every output remains explicitly synthetic
Gaussian curve-conditioned noise.

A pair enters the benchmark population only when both selected images pass
the preregistered synthetic matched-filter surrogate: network SNR at least 10,
at least two detectors per image, and per-contributing-detector SNR at least 4.
These privileged selection statistics are never model inputs. The posterior
and evaluation prior are conditional on this selection.

EM availability is assigned deterministically and equally across eight named
scenarios ranging from full spectroscopic information to sparse
astrometry/timing. The mixture is a benchmark stress design, not a claim about
survey missingness frequencies. Exact latent values are never substituted for
missing measurements.

## Splits, sizes and leakage control

Use the existing grouped split policy. Freeze stable group assignments before
materialization. `validation` selects architecture/hyperparameters;
`calibration` fits any post-hoc correction; `iid_test` is untouched until both
are frozen. Tail and OOD tests are reported separately and never pooled to
conceal failures.

The fixed counts are 65,536 train; 8,192 validation; 8,192 calibration; 16,384
IID test; and 4,096 each for balanced-tail, lens-family OOD, parameter-region
OOD, and waveform/PSD mismatch. A separate 4,096-pair generator qualification
set is engineering-only. The real-noise split has count zero and requires a
separate GWOSC authorization.

At 95% normal approximation, 16,384 IID cases give worst-case binomial
half-width 0.0077; 2,048 cases in each of eight EM cells give 0.0217; 1,024
cases per tail stratum give 0.0306. Report Wilson intervals rather than relying
on the approximation.

## Methods to compare

- joint GW+EM amortized posterior estimator;
- same estimator with GW-only inputs;
- same estimator with EM-only inputs;
- deterministic SIS point-regression legacy baseline, labeled out of domain;
- non-neural lens-posterior baseline under matched priors;
- likelihood-based GW reference on the fixed gold subset;
- oracle diagnostics only in a separate privileged analysis path.

The primary estimator is a mask-aware conditional neural-spline flow over the
two log absolute magnifications. It combines a shared-weight per-image/per-
detector 1D ResNet with an image-keyed DeepSets EM encoder. The fixed model-
selection grid is flow transforms `{6, 10}` by conditioner widths `{128, 256}`
and three training seeds, for at most 12 fits. Validation negative log
probability selects the candidate; exact ties choose fewer parameters. GW-only
and EM-only ablations reuse the same eligible backbone rather than receiving a
different tuning budget.

## Calibration and accuracy endpoints

Primary endpoints on the untouched IID test are marginal empirical coverage at
50%, 80%, 90% and 95% for both log absolute magnifications, plus joint-region
coverage from the flow log-density rank. Report Wilson 95% intervals and raw
counts for every endpoint.

The marginal acceptance tolerance is the larger of 0.01 and three binomial
standard errors. The joint tolerance is the larger of 0.015 and three standard
errors. Each EM cell uses the larger of 0.04 and three standard errors; each
balanced-tail stratum uses 0.06. These rules are fixed before any estimator is
trained.

Secondary endpoints are interval width/sharpness, bias and RMSE of posterior
means, proper scoring rules, posterior correlation, tail conditional coverage,
1,024-case SBC rank diagnostics, and importance sampling on a 256-case gold
subset. At least 95% of gold cases must have importance-sampling efficiency at
least 0.10.

No single average metric can override a failed preregistered coverage stratum.
Post-hoc calibration is fitted only on `calibration`, versioned, and then
applied once to IID/OOD tests.

## Ablations and stress tests

- remove each EM modality and exercise naturally missing combinations;
- withhold velocity dispersion or external-convergence information;
- train/evaluate across SIE versus EPL family mismatch;
- waveform and PSD mismatch;
- detector dropout and SNR/magnification tails;
- perturbed redshift/astrometry uncertainty and selection functions;
- prior sensitivity for density slope, external convergence and population.

## Failure and stopping rules

Invalid solver states, nonfinite weights, proposal-support failures, cross-split
group leakage, privileged-input access, incomplete manifests, undeclared
PSD/waveform versions, waveform truncation, resume mismatch, or mismatch
between declared and executed priors are hard failures. Coverage failure is a
scientific result, not permission to retune on the test set.

## Storage, runtime and resume boundary

At three `(2, 3, 16384)` float32 products, one pair occupies 1,179,648 raw
bytes. All 118,784 planned pairs, including qualification, require
140,123,308,032 raw bytes. A 30% reserve gives 182,160,300,442 bytes; against
342,407,888,896 bytes measured free on AutoDL, the projected remainder is
160,247,588,454 bytes.

Phase 3A must run only the 4,096-pair qualification stage first. Its raw arrays
are 4,831,838,208 bytes, below the 10 GB safety threshold. Shards contain 128
pairs, publish atomically, write append-only attempts, and must pass a
byte-identical interruption/resume test. The qualification report must measure
accepted-pair throughput and use it to estimate wall time before any job that
may exceed one hour. No production run may start from the current document.

## Remaining external gate

The numerical and procedural choices are frozen in release-candidate form.
Phase 3A remains blocked on human review of the Phase 2 report and explicit
authorization. Full production remains additionally blocked on the Phase 3A
throughput, solver acceptance-rate, storage and resume evidence.
