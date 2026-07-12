# Phase 3C-0.2 — Target-anchored exact proposal and latent preflight

Execute Phase 3C-0.2 only on `phase3c0/target-anchored-proposal`.

Read first: AGENTS.md, the Phase 3C-0.2 authorization, adaptive RC.3, parent
RC.5, immutable proposal-v2 RC.1 configuration/report/result, proposal
efficiency plan, privileged policy and provenance/seeds documentation.

This phase is latent-only. Do not generate waveform pairs, call accepted-pair
generation, run lens solving/detector/Galkin/selection, run A/B, train, access
GWOSC/GWTC, modify adaptive RC.3, alter RC.1 evidence or authorize Phase 3C-A.

## Immutable RC.1 record

Preserve `proposal-v2-exact-mixture-1.0.0-rc.1`, its configuration/hash,
replay, ESS result and report byte-for-byte. Create a new version/hash.

## Exact proposal-v3

Create `proposal-v3-target-anchored-mixture-1.0.0-rc.1` at
`configs/proposals/proposal_v3_target_anchored_mixture.yaml`:

    q_v3 = 0.20 q_rc5 + 0.55 p_eval + 0.25 q_central

`q_rc5` is the exact existing broad proposal. `p_eval` is the exact inherited
pre-selection evaluation target sampler/density. `q_central` equals p_eval for
all factors except independent source coordinates use the exact sigma=0.8
truncated normal on [-2.5,2.5). No low-z tilt or selection conditioning.

Implement exact stable mixture density and `log w=log p_eval-log q_v3`. Verify
`q_v3 >= 0.55 p_eval`, also conditional on each common-probability lens family.

## ESS certificate

Create a machine-readable certificate that `w<=1/0.55`, hence with normalized
density and E_q[w]=1, `E_q[w^2]<=1/0.55` and population relative ESS >=0.55.
State all normalization, support, pre-selection and common-family assumptions.
This does not replace empirical validation.

## Latent diagnostics

Run exactly 200,000 deterministic RC.5 draws as a diagnostic, reporting finite
fractions, mean weight, overall/SIE/EPL ESS, maximum normalized weight and
quantiles. Do not retrospectively label RC.5 pass/fail.

Report diagnostic marginal factor groups for RC.5 and v3: lens structure,
redshifts, source plane, external convergence, BBH mass, spin/orientation and
complete state. Do not claim an order-dependent decomposition is unique.

Run exactly 200,000 deterministic v3 draws. Require all finite, zero support
holes, mean weight [0.98,1.02], overall ESS >=0.50, SIE/EPL ESS >=0.40,
byte-identical replay, anchor inequality and certificate. Freeze seed/hash/
tolerances first and do not tune after results. Pass or fail stops for review;
passing never authorizes A/B.

## A/B dry-run update

Update only the dry-run candidate identity to proposal-v3 RC.1. Preserve
512+512, 16x32 blocks, distinct identities, telemetry, throughput gate and
caps. Do not execute.

## Outputs and completion

Create the v3 config/spec/ADR/report and results for RC.5 baseline, v3
validation, factorwise diagnostics, ESS certificate and replay SHA256. Do not
commit samples. Add exact sampler/density/anchor/bound/replay/immutability/A-B/
safety tests. Run full pytest, focused tests, Ruff, mypy, build and hashes.
Update AGENTS, decisions, failures, project state and registry.

Commit `feat: implement target-anchored proposal v3`, push and stop. Do not
create a PR, run A/B or authorize Phase 3C-A.
