# Phase 3C-0 proposal-v2 implementation report

Status: **implementation complete; latent preflight failed; hard stop for human review**.

## Scope and identities

Phase 3C-0 began from Phase 3B merge commit
`80c795a36b902798fe52598262f8b0328755cfac` under the separate design-only
authorization. Adaptive preregistration RC.3 and its canonical hash
`6082475631539d3069edacc52f41b37fb8fe725ccd7c6bc9980cc3008795a927`
were not changed.

- proposal version: `proposal-v2-exact-mixture-1.0.0-rc.1`;
- proposal configuration hash:
  `e4e249da3f177202960e8a6f6c0c347a25aa572abb818b6d0e172469a75e45b5`;
- preflight seed: `2026071204`;
- latent draws: 200,000;
- replay digest:
  `02c1d1dae179703478b7c2250385912cf5c09fdb262b405f75440c0bac46703d`.

No waveform pair was generated, no accepted-pair generator was called, no A/B
qualification ran, no model was trained and no GWOSC/GWTC product was accessed.

## Implementation evidence

The code implements all four normalized components, full latent-state
component/mixture/evaluation densities, stable log-sum-exp, exact conditional
redshift and mass-ratio factors, angular source-plane Jacobian, importance
weights, deterministic replay and a dry-run-only A/B validator. Policy 1.3.0
denies every new proposal diagnostic as a deployable input.

Analytic/numerical tests cover one- and two-dimensional source densities,
low-z conditionals, half-open boundaries, support positivity, mixture
log-sum-exp, deterministic sampling and exact A/B count/cap contracts.

## Frozen latent preflight

| Metric | Result | Gate | Outcome |
|---|---:|---:|---|
| finite log q / log p / weight | 1.0 / 1.0 / 1.0 | 1.0 | pass |
| mean unnormalized weight | 1.01044 | [0.98, 1.02] | pass |
| support holes | 0 | 0 | pass |
| replay byte-identical | true | true | pass |
| overall relative ESS | 0.09202 | >= 0.50 | **fail** |
| SIE relative ESS | 0.11969 | >= 0.40 | **fail** |
| EPL relative ESS | 0.07433 | >= 0.40 | **fail** |

The observed component counts (39,574 / 120,604 / 19,902 / 19,920) are
consistent with the frozen mixture. Maximum normalized weight was 0.000632.

## Interpretation

The sampler and density are internally coherent: mean weight is near one and
support/replay checks pass. The poor ESS is instead a mismatch between the
complete RC.5-broad inherited lens factors and the more concentrated evaluation
target. Centralizing only source position and part of source redshift cannot
meet the full-latent weight-efficiency threshold.

This interpretation is diagnostic, not an authorization to change the frozen
mixture. No parameter was tuned after observing the result.

## Software verification

- full pytest: 178 passed, with three optional Lenstronomy skips;
- proposal/policy focused tests: 39 passed;
- maintained-scope Ruff: passed;
- mypy: passed for 31 source files;
- source distribution and wheel: built successfully;
- proposal configuration hash and recorded replay digest: reproduced exactly.

The package build retains the known missing-README warning. No large latent
sample array was retained or committed.

## Completed, failed and deferred

Completed: exact proposal code/configuration, density and normalization tests,
privileged policy hardening, dry-run A/B skeleton, caps/telemetry contract and
latent evidence.

Failed: the predeclared overall and per-family ESS gates. The proposal candidate
is rejected for A/B under this version.

Deferred: any new proposal version, the 1,024-pair A/B run, scientific
generation, model training, calibration/evaluation, real noise and GWOSC/GWTC.

Phase 3C-0 stops here for human review. Phase 3C-A remains unauthorized and RC.5
remains the fallback proposal.
