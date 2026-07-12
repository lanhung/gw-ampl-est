# ADR-004: Exact proposal-v2 mixture and latent hard stop

Status: implemented under Phase 3C-0; candidate rejected by frozen latent ESS
gate; A/B execution unauthorized.

## Context

Phase 3A measured low accepted-pair throughput. RC.3 required any efficient
training proposal to have an executable normalized density, full support,
target correction and a latent ESS gate before A/B generation.

## Decision

- Implement the reviewed 0.2/0.6/0.1/0.1 RC.5/wide/narrow/low-z mixture exactly.
- Retain the 0.2 RC.5 safety component and explicit selection conditioning.
- Evaluate `p_eval/q_v2` on the complete pre-selection latent state.
- Use process-local deterministic RNG and exact half-open support semantics.
- Keep proposal diagnostics privileged and the future A/B runner dry-run-only.
- Apply the predeclared 200,000-draw mean-weight and ESS thresholds without
  post-result changes.

## Result and consequence

Normalization, support, finiteness, mean weight and replay checks passed. The
overall relative ESS was 0.0920; family ESS was 0.1197 for SIE and 0.0743 for
EPL. All are below the frozen 0.50/0.40 gates.

This is a hard stop, not permission to tune RC.1. Proposal-v2 RC.1 cannot enter
the 1,024-pair A/B qualification. RC.5 remains the only qualified proposal.
Any revised candidate needs a new version, reviewed distribution and repeated
latent-only preflight. Zero waveform pairs were generated.
