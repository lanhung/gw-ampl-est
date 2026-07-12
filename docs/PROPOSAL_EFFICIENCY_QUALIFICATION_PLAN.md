# Proposal-efficiency qualification plan

Status: future engineering design only; proposal-v2 and its 512-pair A/B run
are unauthorized.

## Invariant target and training correction

Proposal v2 may improve training-system sampling but cannot change the RC.5
evaluation target. Training under proposal `q_train(theta)` must minimize the
importance-weighted conditional NLP with
`log w = log p_eval(theta) - log q_train(theta)` over the complete latent
state. Weights are global mean-one per rung, unclipped, finite, privileged and
never model inputs. Validation, calibration, SBC and IID use direct target
draws; SBC may not use uncorrected proposal draws.

## Executable specification before authorization

The current caustic-aware and distance/magnification-stratified labels are
concepts, not executable probability distributions. Before any 512-pair
authorization, a separate review must freeze for every component its sampling
algorithm, exact normalized log-density evaluator, support, truncation and
normalization constants, mixture-variable semantics, conditionals,
deterministic seed domains, full latent-variable coverage, numerical
normalization tests and boundary tests.

The final mixture must retain weight 0.2 on the RC.5 broad-support safety
component and evaluate its exact normalized density with log-sum-exp.

## Future A/B gate

The mandatory primary endpoint is accepted pairs per active hour. RC.5 control
and proposal-v2 candidate each produce 512 engineering-only accepted pairs on
the same hardware, environment, worker count and code path where possible.
Each arm has sixteen matched blocks of 32 accepted pairs. Arm order alternates
deterministically with even/odd reversal. Active time excludes only declared
operator pauses. A 10,000-replicate bootstrap resamples matched block indices
using seed domain `proposal_v2_throughput_ab_bootstrap_v1`, recomputes the ratio
of aggregate candidate and control accepted-pair rates, and forms a percentile
95% interval in log-rate-ratio space before exponentiating. The lower bound
must be at least 2.0.

Acceptance-rate ratio, attempts per accepted pair, CPU seconds per accepted
pair, lens-solver time and proposal-density time are secondary. An acceptance
gain without the required throughput gain cannot authorize adoption. Endpoint
switching after results is forbidden; an interval crossing 2.0 is
inconclusive/failing and retains RC.5.

The candidate must additionally pass finite-weight, relative-ESS,
maximum-weight, support, bounded-memory, duplicate-ID and resume gates. No
threshold may be relaxed after observation. Phase 3B.1 runs none of these
procedures.
