# Executable non-neural reference baseline

## Why the inherited likelihood-gold gate is superseded

The inherited Phase 2 text borrowed the DINGO-IS pattern: draw from a neural
posterior proposal and attach exact likelihood/prior weights, then require at
least 10% importance-sampling efficiency on 95% of 256 cases. That correction
is valid only when proposal and target densities are evaluable on the same
complete latent parameter space. DINGO's proposal is a full GW-parameter
posterior ([Dax et al. 2023](https://arxiv.org/abs/2210.05686)).

This project's deployed NSF is deliberately a two-dimensional marginal over
the selected images' log absolute magnifications. The simulator likelihood
also depends on unreported nuisance variables: lens structure and source
position, external convergence and dynamics nuisance, BBH masses and spins,
sky/orientation/phase, and the selected-image mapping. The NSF supplies no
normalized conditional density for those nuisance dimensions. Consequently,
an exact full-latent importance weight cannot be constructed from its samples.

Calling a two-dimensional reweighting “likelihood corrected” would be false.
RC.7 therefore prospectively removes the DINGO-style efficiency claim before
final materialization. It does not change data, model, target, calibration,
SBC, or final split membership. Independent SBC and IID/tail coverage remain
the primary approximation and calibration evidence.

## Frozen non-neural simulation reference

The replacement baseline is
`selected_prior_em_timing_knn_kde_v1`. It is a transparent non-neural
simulation reference, not an exact likelihood and not a gold posterior.

The reference bank is the final locked scientific training rung. Queries are
group-disjoint validation, IID, or balanced-tail systems. Candidate references
must have the exact adopted lens family and exact EM availability cell, and the
query physical-system ID is excluded.

Neighbor selection uses only deployable EM/timing information:

- observed time difference and uncertainty;
- noisy astrometry, roles, covariance and censoring values;
- noisy scalar EM values and uncertainties;
- scalar, astrometry, modality and censoring masks;
- adopted lens-family hypothesis.

GW strain, detector masks, query truth, source/lens/noise IDs, selection SNR,
proposal density and all other privileged provenance are excluded. Inputs use
the selected primary model's training-rung standardizer; no baseline-specific
fit or tuning is allowed.

The flattened standardized values and masks use squared Euclidean distance.
Exactly 256 neighbors are selected by `(distance, physical_system_id)`. If
fewer than 256 exact family/cell candidates exist, the baseline fails closed.
With `d_i^2` and `h^2=max_i d_i^2`, weights are

\[
  a_i=\exp[-d_i^2/(2h^2)],\qquad w_i=a_i/\sum_j a_j.
\]

The empirical two-target posterior is a normalized weighted Gaussian KDE. Its
common covariance is the weighted target covariance multiplied by Scott's
two-dimensional factor `n_eff^(-1/3)` (the squared bandwidth), plus a frozen
`1e-6 I` variance floor. Density evaluation is an exact normalized Gaussian
mixture. Each query has 4,096 deterministic draws from a query-ID seed.

## Metrics and interpretation

The reference reports CRPS, marginal coverage, descriptive central joint
coverage, interval width and KDE NLP. It has no neural-model seed. It may be
compared to the EM-only and joint NSF, but it may not be described as:

- an exact observation-likelihood posterior;
- a correction of the NSF;
- a full multimessenger baseline using GW strain;
- a gold standard.

The finite simulation bank and distance approximation are explicit
limitations. Failure of this reference does not permit tuning on final data.

## Analytic control

The existing SIS identity remains an algebra, sign, parity and ordering
control. It is not a substitute for SIE/EPL accuracy and does not enter the
20,480-system final pool.

## Execution boundary

The current implementation is pure and synthetic-only. It neither opens the
training reference bank nor a validation/final query. A later execution gate
must bind the locked-rung parent hashes, selected preprocessing artifact,
query publication, code/wheel and output identities. Final data remain sealed.
