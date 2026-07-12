# ADR-003: Adaptive scientific-production allocation and sealing

Status: proposed by preregistration `1.1.0-rc.1`; execution disabled.

## Context

Phase 3A qualified bounded-memory generation but measured only 0.2814%
acceptance. A one-shot allocation would commit substantial time and storage
before learning-curve evidence exists. Conversely, inspecting final tests to
decide scale would leak evaluation evidence into development.

## Decision

- Assign stable group ranks and all split memberships before materialization.
- Use cumulative 16k/32k/65k training cutoffs with fixed development/final
  pools.
- Permit scale and architecture decisions from validation only.
- Materialize final evaluation after size/architecture lock by default.
- Keep calibration-fit, SBC and final evaluation behind separate gates.
- Count physical systems separately from noise or EM augmentations.
- Use atomic 128-system shards and the Phase 3A manifest/checksum conventions.
- Persist continuous peak RSS, integrated CPU utilization and peak staging
  bytes in any future execution.

## Consequences

Stopping at 16k, 32k or 65k produces 49,152, 65,536 or 98,304 total scientific
systems after fixed development and final pools. Evidence that 65k remains
data-limited cannot enlarge the training pool under this ADR; it requires a new
preregistration.

Final data may have IDs/seeds committed before training without arrays being
materialized. If a later design permits early materialization, cryptographic
sealing and access logs require a reviewed ADR amendment.
