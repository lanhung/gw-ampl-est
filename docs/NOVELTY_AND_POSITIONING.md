# Novelty and positioning

## Claim discipline

This project does **not** preregister a “first” claim. Its defensible
contribution is the evaluated intersection of several requirements that prior
work usually treats separately:

- galaxy-scale, temporally separated GW image pairs rather than overlapping
  wave-optics images;
- a joint posterior for signed/absolute macro magnifications, explicitly
  conditional on lens family, cosmology, external convergence, selection, and
  the observation model;
- deployable GW and optical/environment/kinematics observations, with latent
  truth and proposal weights denied to the estimator;
- proposal/evaluation population separation, group-disjoint model development,
  independent post-hoc calibration and SBC, frozen IID/OOD tests, and
  a prospectively frozen non-neural simulation reference;
- explicit mass-sheet, distance–magnification, lens-family, waveform, and PSD
  misspecification tests.

The publication claim must remain no stronger than the tested model families
and synthetic observation process. Phase 3A cannot support accuracy,
calibration, real-noise, or GWTC claims.

## Relationship to the nearest work

GOLUM and the Lo–Magaña Hernandez framework are direct references for
separated-image GW pair analysis. They primarily use GW consistency and
relative lensing parameters to identify or analyze pairs. Gravelamps provides
general parameterized lens inference for GW signals. LVK searches establish
the catalog-level detection context. Those are baselines or context, not work
to be relabeled as equivalent to an optical-lens-conditioned absolute
magnification posterior.

DINGO establishes that a full-parameter amortized GW posterior can be checked
with importance sampling and calibration diagnostics. The present target-only
NSF has no full nuisance-space proposal and therefore makes no likelihood-
correction or importance-efficiency claim. DINGO-lensing and neural
micro/millilensing papers demonstrate related neural inference in overlapping
or wave-optics regimes. Their scientific regime is not interchangeable with
well-separated galaxy macro images.

Multimessenger lens-reconstruction and degeneracy studies are the closest
physical antecedents. They motivate the optical observation model and also
limit the claim: external convergence and lens dynamics must enter a real
forward model, and absolute magnification remains conditional rather than
directly identified.

## Publication target and decision rule

The design is aimed at a methods-and-astrophysics paper suitable in scope for
PRD or ApJS; journal suitability depends on later validated results, not this
preregistration. A stronger venue would require evidence beyond the current
authorization, plausibly including robust OOD behavior, scientifically useful
ablation conclusions, and eventually a separately preregistered real-data/noise
study.

The eventual abstract and title must be chosen after frozen evaluation. If
coverage, the frozen non-neural reference, or mismatch performance fails, the
paper must report the failure or narrow its claim; the IID test may not be
reused for method development.

## Baseline map

- Pair identification: Haris et al., Goyal et al., Lo–Magaña Hernandez, GOLUM.
- Bayesian lens inference: Gravelamps and conventional lens reconstruction.
- Neural posterior methodology: DINGO and importance sampling.
- Different physical regimes: DINGO-lensing and micro/millilensing NPE.
- Public-search context: LVK O3a/full-O3 searches.
- Identifiability: mass-sheet and lensed-GW cosmography analyses.

The machine-readable comparison is in `docs/literature_matrix.csv`; verified
bibliographic records are in `references/verified_references.bib`.
