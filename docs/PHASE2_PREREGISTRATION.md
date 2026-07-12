# Phase 2 statistical preregistration

Status: entry draft; no execution authorized.

## Scientific question and estimands

For a candidate galaxy-scale lensed GW pair with declared EM information,
estimate the joint posterior of
`(log_abs_mu_primary, log_abs_mu_secondary)`. The posterior is conditional on
the lens family and all declared population, cosmology, external-convergence,
selection, waveform, noise and EM observation models.

Preregistered derived quantities are the log absolute-magnification difference,
absolute-magnification ratio, flux ratio, and amplitude-factor ratio. Signed
magnification, parity and Morse index remain distinct discrete quantities.

## Population and generative model to freeze before Phase 3

- source population and evaluation prior, separately versioned;
- SIE+external-shear and EPL+external-shear lens populations;
- explicit density-slope, external convergence and model-discrepancy priors;
- H1/L1/V1 detector network, waveform family mixture and calibration model;
- noisy GW selection and pair/image detection model;
- EM astrometry, redshift, Einstein-scale, kinematic and missingness models;
- proposal distribution and exact importance-weight convention.

Proposal draws must cover every evaluation population and tail. Proposal
metadata and weights are evaluation-only and remain denylisted as deployable
inputs.

## Splits and leakage control

Use the existing grouped split policy. Freeze stable group assignments before
materialization. `validation` selects architecture/hyperparameters;
`calibration` fits any post-hoc correction; `iid_test` is untouched until both
are frozen. Tail, lens-family, parameter-region, waveform/PSD and real-noise
tests are reported separately and never pooled to conceal failures.

The final counts are not guessed here. Phase 2 must compute the minimum count
per preregistered stratum needed to estimate each nominal coverage probability
to a stated binomial half-width, then add a documented allowance for invalid
simulations and selection losses.

## Methods to compare

- joint GW+EM amortized posterior estimator;
- same estimator with GW-only inputs;
- same estimator with EM-only inputs;
- deterministic SIS point-regression legacy baseline, labeled out of domain;
- non-neural lens-posterior baseline under matched priors;
- likelihood-based GW reference on a computationally bounded gold subset;
- oracle diagnostics only in a separate privileged analysis path.

## Calibration and accuracy endpoints

Primary endpoints on the untouched IID test are marginal empirical coverage at
50%, 80%, 90% and 95% for both log absolute magnifications, plus joint-region
coverage under a method frozen in Phase 2. Report binomial confidence intervals
and raw counts for every endpoint.

Secondary endpoints are interval width/sharpness, bias and RMSE of posterior
means, proper scoring rules, posterior correlation, tail conditional coverage,
SBC rank diagnostics, and—where an exact likelihood is available—importance
sampling efficiency and discrepancies from the gold posterior.

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
group leakage, privileged-input access, incomplete manifests, or mismatch
between declared and executed priors are hard failures. Coverage failure is a
scientific result, not permission to retune on the test set.

Before any job longer than one hour or any action that may create more than
10 GB, Phase 2 must provide a fixed configuration, byte/runtime estimate, log
path, manifest schema, checkpoint cadence and deterministic resume test.

## Unresolved items blocking Phase 3

- numerical prior bounds and source/lens population versions;
- exact selection function and EM missingness distribution;
- coverage interval construction and target half-width;
- split and stratum sample sizes from precision calculations;
- architecture and likelihood-based gold-subset budgets;
- production waveform/PSD/calibration versions;
- storage/runtime estimate and resume plan;
- human acceptance of the completed Phase 2 report.
