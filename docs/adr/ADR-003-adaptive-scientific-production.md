# ADR-003: Adaptive scientific-production allocation and sealing

Status: revised by preregistration `1.1.0-rc.2`; awaiting human review and
execution disabled.

## Context

Phase 3A qualified bounded-memory generation but measured only 0.2814%
acceptance. A one-shot allocation would commit substantial time and storage
before learning-curve evidence exists. Conversely, inspecting final tests to
decide scale would leak evaluation evidence into development.

## Decision

- Assign deterministic group and attempt namespaces before materialization;
  do not claim accepted IDs are known before selection executes.
- Use 16k only as a probe subset of 32k; permit final lock only at 32k or 65k.
- Materialize Stage A (32k train plus validation), conditional Stage B (32k
  more train), then post-lock Stage C (calibration, SBC and final evaluation).
- Permit scale and architecture decisions from validation only.
- Materialize final evaluation after size/architecture lock by default.
- Finalize and hash a deterministic generation commitment before training.
- Correct any efficient training proposal to the evaluation target with the
  frozen unclipped importance-weighted conditional likelihood.
- Keep calibration-fit, SBC and final evaluation behind separate gates.
- Store one Gaussian noise realization per physical system and count any
  future augmentation separately from independent systems.
- Reuse exact locked-rung probe fits in architecture selection.
- Use atomic 128-system shards and the Phase 3A manifest/checksum conventions.
- Persist continuous peak RSS, integrated CPU utilization and peak staging
  bytes in any future execution.

## Consequences

Locking at 32k or 65k produces 65,536 or 98,304 total scientific systems.
There is no 16k final total. Evidence that 65k remains data-limited cannot
enlarge the training pool under this ADR; it requires a new preregistration.

The pre-training commitment fixes generation identities, seed/attempt domains,
allocation rules and validators, not unknowable accepted IDs. Later accepted
IDs must be verified as its deterministic output. Early materialization still
requires a reviewed sealed-access amendment.
