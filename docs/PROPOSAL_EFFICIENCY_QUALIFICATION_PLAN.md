# Proposal-efficiency A/B qualification plan

Status: future engineering design only under `1.1.0-rc.3`; proposal-v2 and the
A/B qualification are unauthorized.

The future design contains 512 accepted pairs per arm and 1,024 accepted
engineering pairs total:

| Arm | Accepted pairs | Use |
|---|---:|---|
| RC.5 control | 512 | engineering control only |
| proposal-v2 candidate | 512 | engineering candidate only |

Both arms permanently deny scientific, training, calibration and test use and
can never enter a scientific split. The hard authorization maximum is exactly
1,024 accepted pairs across both arms.

## Invariant target and executable prerequisite

Proposal v2 may improve training-system sampling but cannot change the RC.5
evaluation target. Later training under `q_train(theta)` must use the frozen
full-latent importance-weighted conditional NLP. Weights remain privileged and
never deployable inputs. Validation, calibration, SBC and IID use direct target
draws.

The candidate components remain concepts, not executable distributions. Before
the future A/B gate can be authorized, a separate review must freeze each
sampler, exact normalized log density, support, truncation/normalization,
conditionals, seed domains, latent coverage and numerical boundary tests. The
0.2 RC.5 broad-support mixture remains mandatory.

## Separate artifact identities

One parent A/B run identity binds two distinct dataset identities: an RC.5
control dataset and a proposal-v2 candidate dataset. The parent has one
comparison manifest; each arm has its own manifest and checksums. Environment,
worker count and telemetry contracts are identical. The concrete future IDs
are derived once from the future authorization commit and RC.3 hash using the
machine-readable templates; they cannot be shared between arms.

## Matched A/B design

Each arm has 16 matched blocks of 32 accepted pairs, so `16 × 32 = 512` per arm
and `2 × 512 = 1,024` total. Arm order alternates deterministically. A frozen
10,000-replicate matched-block bootstrap produces the log throughput-rate ratio
95% interval. Proposal-v2 can be adopted only if its lower bound is at least
2.0. Acceptance rate and CPU/solver/density timing are secondary; acceptance
alone cannot pass.

## Conservative resource gate

Both arms use the measured RC.5 rate for prelaunch planning:

- control: 0.7432 active hours;
- candidate conservative: 0.7432 active hours;
- combined conservative: 1.4864 active hours;
- combined projected publication: 1,112,673,640 bytes;
- temporary/staging reserve: 20,333,802,092 bytes;
- projected peak: 21,446,475,732 bytes;
- minimum prelaunch free space: 121,446,475,732 bytes, preserving 100 GB after peak.

A candidate running at exactly 2× would reduce the combined estimate to 1.1148
hours, but that unmeasured hypothesis cannot lower the prelaunch gate.

Phase 3B.2 implements or runs none of this future A/B qualification.
