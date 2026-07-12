# Proposal-v2 exact-mixture executable specification

Status: `proposal-v2-exact-mixture-1.0.0-rc.1` implemented for latent-only
validation and **failed its frozen ESS gate**. It is not authorized for A/B or
scientific generation.

Configuration hash:
`e4e249da3f177202960e8a6f6c0c347a25aa572abb818b6d0e172469a75e45b5`.

## Density and measure

The proposal is defined before lens multiplicity and detector selection on the
complete RC.5 latent measure, including the angular source-plane measure. It is

```text
q_v2 = 0.2 q_rc5 + 0.6 q_wide + 0.1 q_narrow + 0.1 q_low_z.
```

All components inherit RC.5 lens family, lens/source parameters, masses,
spins, sky/orientation and external convergence except for the two declared
changes. No component conditions on acceptance.

`q_rc5` is uniform in dimensionless source coordinates on `[-2.5,2.5)^2`.
`q_wide` and `q_narrow` replace each coordinate with the exactly normalized,
zero-mean truncated normal of sigma 1.5 or 0.8. Angular source-position density
includes `-2 log(theta_E)`.

`q_low_z` uses the wide source density and, for
`z_min=max(0.5,z_lens+0.1)`, samples
`t=(z_source-z_min)/(3-z_min) ~ Beta(1,2)`. Its exact conditional density is
`2(1-t)/(3-z_min)` on the half-open interval. Invalid intervals fail closed.

Every complete component and the inherited evaluation target include the
conditional redshift and mass-ratio widths, log-uniform Einstein-radius
Jacobian, source-plane Jacobian, truncated normal constants, family
probability, spin/orientation densities and applicable EPL slope density.
Mixture evaluation uses stable log-sum-exp.

## Sampling and deterministic identity

Sampling uses only a caller-owned `numpy.random.Generator`. Truncated normals
use exact rejection sampling from their untruncated normals. No NumPy, Bilby or
Lenstronomy global RNG is accessed. Half-open upper boundaries match the
density evaluator. A canonical binary record of every latent draw supports
byte-identical replay hashing.

## Privileged provenance

Component index, component log densities, proposal/evaluation log density,
log/normalized weights and proposal seeds are denylisted model inputs under
policy version `1.3.0`. They remain privileged training provenance only.

## Dry-run A/B contract

The skeleton validates but cannot execute two sequential arms, 16 alternating
blocks per arm, 32 accepted pairs per block, 512 per arm and 1,024 total.
Control and candidate IDs differ. Frozen caps are 1,000,000 attempts and six
active hours per arm. Candidate cap failure retains RC.5; control cap failure
invalidates comparison; partial blocks cannot be omitted.

## Latent-only result

The authorized 200,000-draw preflight had finite fraction 1.0, mean weight
1.01044, zero support holes and byte-identical replay. It failed overall ESS
(0.0920 versus 0.50) and both family ESS gates (SIE 0.1197, EPL 0.0743 versus
0.40). The proposal version is frozen post-result and must not be tuned in
place.
