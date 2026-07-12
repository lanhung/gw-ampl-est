# Learning-curve stopping rule

Status: frozen design for preregistration `1.1.0-rc.2`; execution disabled and
awaiting human review.

## Probe subset and target-corrected objective

`train_16k_probe_subset` is the first 16,384 ranked members of `train_32k`.
It cannot be locked as a final training size. Stage A generation continues to
32,768 systems regardless of its result.

If training uses a proposal different from the RC.5 evaluation target, every
probe fit uses the same globally mean-one, unclipped importance weights and
weighted conditional NLP objective at 16k, 32k and 65k. Ordinary unweighted
training is a hard failure. Validation consists of direct evaluation-target
draws and never needs proposal correction.

The fixed probe is the mask-aware multimessenger conditional NSF with 10
transforms and width 256. Seeds 0, 1 and 2 train from scratch using identical
AdamW, batch, epoch and early-stopping rules. Scale selection cannot access
calibration-fit, SBC, IID, tail, OOD or mismatch data.

## Paired statistics

All rungs compare the same 6,144 validation systems. Improvements are
smaller-rung metric minus larger-rung metric. A 10,000-replicate paired
bootstrap resamples physical-system IDs and reports 95% intervals and every
seed result.

Primary metric: validation negative log probability in nats per target
dimension. Secondary metrics are median CRPS, marginal and joint coverage
error, EM-cell conditional coverage, validation-internal tail views and
interval width.

## Decision at 32,768

Lock 32k only if every condition holds:

1. the 95% upper bound for NLP improvement is below 0.01 nat per target
   dimension;
2. median CRPS relative improvement is below 1%;
3. maximum marginal-coverage-error improvement is below 0.005;
4. every validation EM cell meets the frozen 0.04 marginal and 0.05 joint
   tolerances;
5. no EM-cell coverage degrades by more than 0.02;
6. all three seeds' NLP and CRPS point improvements are below threshold.

Any meaningful improvement, failed tolerance, insufficient internal-tail
count or confidence interval touching a threshold continues to 65k. There is
no 16k final decision.

## Decision at 65,536

Apply the same paired 32k-to-65k test. Saturation locks 65k. Meaningful
improvement or ambiguity stops as data-limited/inconclusive and requires a new
preregistration; it never authorizes automatic extension.

## Fit reuse

After size lock, the four architectures by three seeds are compared using mean
validation NLP. The three 10-transform/width-256 probe fits at the locked rung
must be reused when all execution identities match, leaving at most nine new
fits and twelve total results. Retraining the identical probe without a
declared failure is forbidden. Calibration, SBC and final evaluation remain
behind separate later gates.
