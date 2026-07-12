# ADR-005: Target-anchored proposal-v3

Status: latent preflight passed; awaiting human review; A/B closed.

## Decision

Replace rejected proposal-v2 RC.1 with a new immutable candidate consisting of
20% RC.5 broad support, 55% exact evaluation target and 25% evaluation target
with a central source plane. Preserve RC.1 evidence unchanged.

The positive target component gives an analytic weight upper bound and ESS
lower bound. Empirical 200,000-draw validation remains mandatory. RC.5 is also
measured separately without applying a retrospective gate.

## Consequences

V3 passed every latent gate with ESS 0.785. RC.5 diagnostic ESS was 0.118,
confirming that generator qualification alone does not imply efficient
target-weighted training. Lens structural factors dominate RC.5 variance.

The future dry-run identity now names v3, while pair counts, telemetry,
throughput rule and caps remain unchanged. A/B requires new human authorization.
