# Terminal 131k training-scale plan

Status: frozen design for preregistration `1.2.0-rc.1`; every execution flag is
false.

## Why a new contract is required

The corrected 32k-to-65k probe comparison completed successfully but did not
saturate. The paired NLP improvement was `0.201437` nat per target dimension
with 95% interval `[0.191498, 0.211788]`, whereas saturation required the upper
bound to be below `0.01`. The resulting
`stop_data_limited_and_new_preregistration` decision is retained unchanged.

The existing validation publication also contains only 40 cases in the
extreme-relative-magnification view. Adding training systems cannot make that
fixed count reach the inherited minimum of 128. A new prospective contract
must therefore address both training scale and development-tail precision.

## Terminal nested training rung

The existing corrected 65,536-system train publication is an immutable strict
subset. A future separately authorized materialization may add exactly 65,536
new direct evaluation-target systems, giving exactly 131,072 unique training
systems. The increment uses the frozen direct-target distribution, unit
importance weights and waveform numerical-validity rejection. It contains 512
atomic shards of 128.

No rung above 131,072 is authorized. The terminal comparison has two honest
outcomes:

- `lock_train_131k_saturated` when the prospective saturation rules pass;
- `lock_train_131k_resource_capped_data_limited` otherwise.

Both outcomes permit a later architecture-selection review, but the second
requires the permanent reporting label that the terminal scale remains data
limited. Neither outcome authorizes more training data.

## Independent development-tail pool

A separately identified 512-system development-only pool contains exactly 128
priority-assigned systems in each frozen balanced-tail stratum:

1. high absolute magnification;
2. extreme relative magnification;
3. second image near the selection threshold;
4. extreme profile or environment.

It is group-disjoint from training, core validation, calibration, SBC, every
final split and all engineering artifacts. It reuses neither final-evaluation
IDs nor seed domains and can never be used for training, architecture
selection, calibration or a reported final test. Retained 65k checkpoints and
new 131k checkpoints are evaluated on the same pool only for development-tail
diagnostics.

## Prospective terminal decision

The unchanged 6,144-case core validation set owns the paired 10,000-replicate
NLP bootstrap. Saturation requires all of:

- the 95% upper bound of the mean NLP improvement is below `0.01` nat per
  target dimension;
- median CRPS relative improvement is below 1%;
- all three seeds have NLP and CRPS point improvements below those thresholds.

Raw marginal, joint and EM-cell coverage, coverage degradation, interval width
and all four tail views remain mandatory reports. They do not block the
terminal size lock because the already frozen post-lock split-conformal and
independent SBC contracts own calibrated coverage claims. This is a
prospective 65k-to-131k rule; it does not reinterpret the failed 65k lock.

Calibration, SBC, IID/OOD/mismatch data and final evaluation cannot influence
the terminal decision.

## Architecture handoff

After either terminal outcome, a later exact gate may execute the inherited
2-by-2 architecture grid at 131k. The three 10-transform, width-256 probe fits
are reused; at most nine new fits are permitted. All three seeds are retained
and selecting a best seed remains forbidden.

## Resource projection

The measured 32,768-system Stage B publication was 39,441,798,884 bytes and
took 32.725 elapsed hours. Linear projection gives:

- 65,536-system increment: 78,883,597,768 published bytes;
- projected generation interval: 65.45 hours;
- projected train-increment peak: 100,980,232,376 bytes;
- free space after that peak from the 65k closeout baseline: 120,633,487,176
  bytes;
- 512-system tail publication: approximately 616,278,108 bytes.

The storage margin is only about 20 GB above the frozen 100 GB floor. These are
planning values, not execution evidence. A future release gate must remeasure
free space and stop if the exact publication, staging and reserve requirements
cannot be met.

## Closed work

This design authorizes no data generation, optimizer, architecture fit,
calibration, SBC, final evaluation, real noise or GWOSC/GWTC access. Each needs
its own identity-bound execution gate.
