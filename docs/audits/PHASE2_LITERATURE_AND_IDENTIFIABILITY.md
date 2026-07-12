# Phase 2 literature and identifiability audit

## Scope

This audit concerns galaxy-scale, geometrical-optics strong lensing of
temporally separated GW images, with EM information about the lens system. It
does not treat short-delay overlapping images as the primary regime and does
not treat pair identification as the inference target.

## Evidence map

- Strong lensing produces repeated signals with image-dependent time delay,
  amplitude factor and Morse phase. Pair-identification work establishes why
  common intrinsic parameters and lensing phase structure matter, but it does
  not by itself identify absolute magnification: [Haris et al. (2018)](https://arxiv.org/abs/1807.07062),
  [Dai et al. (2020)](https://arxiv.org/abs/2007.12709).
- Non-axisymmetric galaxy-lens reconstruction from only two GW images remains
  strongly degenerate; EM astrometry and lens information are therefore part
  of the intended likelihood, not optional truth-derived shortcuts:
  [Seo, Li & Hendry (2023)](https://arxiv.org/abs/2311.05543).
- Dark-siren lens reconstruction has similarity and mass-sheet degeneracies;
  two GW images do not generally determine a complete SIE model:
  [Poon et al. (2024)](https://arxiv.org/abs/2406.06463).
- Under a mass-sheet transformation, image positions and flux/time-delay
  ratios can remain invariant while absolute magnification changes. Absolute
  magnification must therefore be conditional on external convergence,
  kinematic/environment information and model discrepancy:
  [Birrer et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182856/).
- Approximate posterior algorithms require calibration checks over replicated
  draws from the declared joint model. Rank-based simulation-based calibration
  is an implementation check, while held-out conditional coverage and OOD
  stress tests address estimator behavior:
  [Talts et al. (2018)](https://arxiv.org/abs/1804.06788),
  [Lemos et al. (2023)](https://proceedings.mlr.press/v202/lemos23a.html).
- Neural posterior proposals can be verified and corrected on a tractable
  likelihood subset with importance sampling; effective sample size and
  proposal support are required diagnostics rather than proof by visual
  posterior agreement: [Dax et al. (2023)](https://arxiv.org/abs/2210.05686).

## Identifiability boundary

The deployable observations are the two noisy GW detector networks and the
declared EM/timing products. They inform, but do not uniquely determine, the
source luminosity distance, lens mass profile, source position, external
convergence and two absolute magnifications. The intended output is therefore

```text
p(log|mu_primary|, log|mu_secondary|, nuisance |
  GW_pair, EM_observations, timing, availability,
  lens_family, cosmology, population_and_selection_model).
```

It is not an unconditional measurement of absolute magnification. In
particular:

- GW amplitude alone constrains a magnified distance combination;
- relative image amplitude can inform a magnification ratio but not the common
  absolute scale;
- image positions and ratios do not eliminate the mass-sheet direction;
- velocity dispersion, redshifts, lens environment and model-discrepancy
  assumptions can reduce uncertainty but must be propagated, never inserted as
  exact truth;
- missing EM modalities must broaden or reshape the posterior rather than be
  imputed from latent simulation variables.

## Required identifiability experiments

All experiments use newly authorized future scientific simulations, never the
Phase 1B smoke artifact.

1. Compare GW-only, EM-only and joint inference under identical evaluation
   priors.
2. Use nested EM availability cells: astrometry; plus redshifts; plus Einstein
   scale; plus velocity dispersion; plus environment/external convergence.
3. Compare SIE+shear and EPL+shear both in-family and cross-family.
4. Vary external convergence and density-slope/model-discrepancy priors to
   expose mass-sheet sensitivity.
5. Stratify coverage by image parity, Morse pair, magnification tail, SNR,
   delay, detector availability and EM missingness.
6. Include a likelihood-based gold subset and analytic SIS identity controls;
   neither may stand in for the production non-SIS problem.

## Claims explicitly out of scope

- lensing-pair discovery performance;
- model-independent absolute magnification;
- real-noise robustness before a versioned GWOSC/DQ/PSD protocol is executed;
- microlensing inference for overlapping images;
- population or cosmological conclusions from the engineering smoke set.
