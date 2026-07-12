# Adaptive scientific-production plan

Status: design-only preregistration `1.1.0-rc.2`, awaiting human review. No
generation, training, calibration, evaluation, proposal qualification or
external-data access is authorized.

The machine-readable authority is
`configs/statistics/adaptive_scientific_production_preregistration.yaml`. Its
canonical hash is recorded after the complete RC.2 configuration is frozen.

## Boundary inherited from Phase 3A

Phase 3A qualified generator commit
`fbcd0616611d9cdf915ef0af030e6061c1be7f59` using 4,096 engineering-only
pairs. Dataset `gwlens-v2-2.0.0-alpha.3-7081b2e8be3a84e1` remains permanently
outside every scientific split. Its throughput and rejection evidence may
inform engineering projections but never scientific training or evaluation.

RC.5 remains the parent evaluation target. RC.2 changes allocation, target
correction and decision rules; it does not alter the parent estimand,
source-plane measure, observation model, selection model or waveform contract.

## Executable training sizes

The cumulative relation is:

```text
train_16k_probe_subset (16,384) ⊂ train_32k (32,768) ⊂ train_65k (65,536)
```

The 16k subset is evidence for the 16k-to-32k learning-curve comparison. It is
not an independently lockable final size: generation continues to 32k
regardless of the 16k result. The only final locks are 32k and 65k. Evidence
that 65k remains data-limited stops execution and requires a new
preregistration rather than an automatic larger rung.

Every reported sample count is an independent accepted physical-system count.
Exactly one synthetic Gaussian noise realization is stored per system. Future
noise augmentation is currently unauthorized, would remain within its parent
split, and would not create an additional independent system.

## Materialization sequence

| Stage | Materialization | Increment | Condition |
|---|---|---:|---|
| A | train 32,768 + validation 6,144 | 38,912 | future separate authorization |
| B | add 32,768 training systems | 32,768 | only if the 32k rule continues to 65k |
| C | calibration 4,096 + SBC 2,048 + final 20,480 | 26,624 | after size and architecture lock |

Stage A's first 16,384 ranked training systems form the probe subset. Stage C
cannot begin early unless a separately reviewed sealed-materialization gate is
created.

The achievable completed scientific totals are therefore:

| Locked training size | Training | Development | Final | Total |
|---|---:|---:|---:|---:|
| 32k | 32,768 | 12,288 | 20,480 | 65,536 |
| 65k | 65,536 | 12,288 | 20,480 | 98,304 |

The development pool is validation 6,144, calibration-fit 4,096 and independent
SBC 2,048. The final pool is IID 8,192, balanced tail 4,096 and four 2,048-case
cross-family/OOD/waveform/PSD splits. All groups remain disjoint.

## Training proposal and posterior target

The scientific target is `p_eval(theta)`. A separately qualified efficient
training proposal `q_train(theta)` may sample training systems, but ordinary
unweighted NPE training under a changed proposal is forbidden. The frozen
objective is

```text
sum_i w_i [-log r_phi(mu_i | x_i)] / sum_i w_i,
log w_i = log p_eval(theta_i) - log q_train(theta_i).
```

Weights use the complete latent proposal/evaluation state, are globally
normalized to mean one within each rung, are never clipped, and remain
privileged provenance rather than deployable inputs. Finite-weight and ESS
summaries are mandatory overall, by lens family and by EM cell.

Validation, calibration-fit, SBC and IID are direct draws from their declared
evaluation generative target. In particular, SBC cannot use uncorrected
proposal-v2 draws. OOD and mismatch splits retain their separately declared
diagnostic distributions.

## Evaluation commitment

Accepted IDs do not exist before selection runs. Before training, the project
instead finalizes and hashes a deterministic generation commitment containing
the future generator commit, RC.2 hash, root and split seed domains,
attempt-stream and accepted-rank rules, counts, distribution identities,
versions, grouping rules and manifest validators. The design template is
`results/phase3b/final_evaluation_commitment.json`.

After authorized materialization, accepted IDs and manifests must reproduce
that commitment exactly. This separates reproducibility and sealing from the
false claim that unknown accepted IDs were frozen in advance.

## Model and proposal decisions

The fixed 10-transform, width-256 probe is trained at each required rung with
seeds 0, 1 and 2 using validation only. At the locked rung, these three fits
must be reused in the final 12-result architecture comparison if their data,
objective, optimizer and budget match. At most nine new fits are then needed.

Proposal-v2 adoption requires the lower bound of a frozen 95% confidence
interval for accepted pairs per active hour relative to RC.5 to be at least
2.0. Acceptance is secondary and cannot authorize adoption alone. Before even
authorizing the 512-pair A/B gate, every proposal component needs a reviewed,
executable sampler and exact normalized density specification. Phase 3B.1 does
not implement or run it.

## Resource projection and authorization

RC.5 baseline projections are 56.48 active hours for Stage A, 47.57 for the
conditional Stage B increment and 38.65 for Stage C. Completed totals project
to 95.13 hours/71.21 GB at 32k lock or 142.70 hours/106.82 GB at 65k lock.

The proposal-v2 scenario is explicitly unmeasured and applies only to training
systems. Assuming exactly 2× training throughput while direct-target nontraining
splits retain RC.5 rate gives 71.35 and 95.13 hours for the two completed
totals. It is not an observed result.

Every later action requires a new human gate. RC.2 authorizes no pair,
training fit, calibration, SBC, final evaluation, real-noise work, GWOSC/GWTC
access or Phase 3C execution.
