# Privileged-input policy

## Fail-closed rule

A field can enter a deployable model only if its exact canonical name appears
in `configs/policy/deployable_input_allowlist.json`. Exact denylisted names,
suspicious aliases, duplicates, and unknown names are rejected. Unknown fields
are not automatically approved because they sound observational.

The policy distinguishes four roles:

1. model input;
2. training target;
3. grouping/provenance metadata;
4. privileged diagnostic.

Permission in one role does not imply permission as a model input.

## Forbidden classes

The machine-readable denylist includes:

- exact `y`, `u`, beta/source-plane coordinates;
- true signed/absolute magnification, flux ratio, and amplitude ratio;
- exact source, lens, lens-center, image-position, or noise truth;
- exact external convergence, stellar anisotropy, or latent light profile;
- exact time delay where a typed timing observation is intended;
- clean injected signal and isolated noise realization;
- optimal SNR and simulation-only quality/oracle fields;
- source, lens, system, pair, noise-segment, and augmentation IDs;
- proposal component identity, RNG seed, component/full proposal or evaluation
  log probability, and importance/population weights.

Aliases such as `mu0`, `mu1`, `A21`, `trueMu`, `oracle*`, and
`cleanWaveform` are also rejected. The alias detector is a defense-in-depth
check; policy review must add new aliases when legacy or external schemas are
introduced.

## Allowed observations

The current allowlist contains GW pair strain observations, detector masks and
identity, typed timing, image-ID-keyed astrometry, other noisy EM measurements
with uncertainty, external-convergence posterior mean/standard deviation and
its availability flag, modality masks/censoring, and preprocessing/PSD/
calibration/DQ metadata. Exact `true_time_delay_seconds`, `kappa_ext_true` and
aliases remain denylisted. It does not mean every allowed field must be used by
a future model.

The clean/noise products remain required dataset artifacts for validation but
are not deployable inputs. Group IDs remain required for leakage-safe splitting
but are not neural features. Proposal component indices, component log
densities, proposal/evaluation log probabilities, log/normalized weights and
proposal seeds remain privileged provenance under policy version `1.3.0`.

## Change control

Both JSON policies carry the same version. A policy change requires tests and
review. Fields must never be silently renamed around the denylist. The
validator lives in `gwlens_mm.policy.InputPolicy`.
