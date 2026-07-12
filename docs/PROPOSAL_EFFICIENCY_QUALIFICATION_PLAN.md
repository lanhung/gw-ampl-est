# Proposal-efficiency qualification plan

Status: future engineering design only; the 512-pair run is unauthorized.

## Motivation and invariant target

Phase 3A accepted 0.2814% of proposals: 1,026,090 attempts lacked two physical
images and 425,513 lacked two images passing synthetic selection. These are
engineering measurements, not a scientific population.

Proposal v2 may change sampling efficiency but not the frozen RC.5 evaluation
target. It must record exact normalized proposal/evaluation log densities and
importance weights for every attempt.

## Support-preserving mixture

The candidate is an exact mixture with weight 0.8 on a caustic-aware and
distance/magnification-stratified efficient component and weight 0.2 on the
complete RC.5 broad proposal. The normalized density is evaluated with
log-sum-exp. The positive RC.5 safety component supplies support wherever the
frozen benchmark has support; finite samples alone are not used to claim
support completeness.

## Future 512-pair gate

All 512 accepted pairs are engineering-only and use a new dataset ID,
authorization and generator commit. Adoption requires:

- at least 2× RC.5 acceptance rate or accepted-pair throughput under a
  comparable pinned environment;
- finite importance weights;
- relative weight ESS at least 0.50 overall, 0.40 in each lens family and 0.25
  in every EM cell;
- no single normalized weight above 0.05;
- passed lens-family, multiplicity, EM-cell and frozen-tail support checks;
- no density-normalization, duplicate-ID, bounded-memory or resume failure.

An incomparable environment, confidence interval crossing 2×, support failure
or ESS failure is inconclusive/failing and retains RC.5. No threshold may be
relaxed after seeing the 512-pair result.

Phase 3B does not run this gate.
