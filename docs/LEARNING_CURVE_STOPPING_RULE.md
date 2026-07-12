# Learning-curve stopping rule

Status: frozen design for preregistration `1.1.0-rc.1`; execution disabled.

## Probe model and data access

Scale decisions use only the same 6,144 validation physical systems. The probe
is the existing mask-aware multimessenger conditional NSF with 10 transforms
and conditioner width 256. Seeds 0, 1 and 2 train independently from scratch at
16,384, 32,768 and, when required, 65,536 systems.

All rungs use AdamW (`lr=3e-4`, betas 0.9/0.999, weight decay `1e-5`), batch
size 256, gradient clipping at 5, at most 200 epochs, and validation-NLP early
stopping with patience 20 and minimum delta 0.001. Post-hoc calibration is
forbidden during scale selection.

Final IID, tail, cross-family, parameter OOD, waveform mismatch and PSD
mismatch cases are inaccessible. Calibration-fit and SBC cases are also
inaccessible. Tail summaries used during stopping are fixed views inside the
validation split and never reuse final-tail IDs.

## Paired statistics

Improvements are smaller-rung metric minus larger-rung metric, so positive is
better. Pairing is by physical-system ID. The bootstrap resamples paired
physical systems 10,000 times using the frozen bootstrap seed domain and
reports 95% intervals. All per-seed values are retained.

Primary metric: negative log probability in nats per target dimension.

Secondary metrics: median CRPS, maximum raw marginal coverage error, joint
coverage error, EM-cell conditional coverage, validation-internal tail views
and interval width.

## Decision at 32,768

Lock 32,768 only when every condition holds:

1. the paired-bootstrap 95% upper bound for NLP improvement is below 0.01 nat
   per target dimension;
2. median CRPS relative improvement is below 1%;
3. maximum marginal-coverage-error improvement is below 0.005;
4. every validation EM cell has maximum absolute marginal coverage error at
   most 0.04 and joint error at most 0.05;
5. no EM-cell coverage degrades by more than 0.02;
6. each of the three seeds has NLP and CRPS point improvement below its
   threshold.

Any meaningful improvement, failed tolerance, insufficient validation-tail
count, or confidence interval touching/straddling a threshold is a gray result
and continues to 65,536.

## Decision at 65,536

Apply the same paired 32k-to-65k rule. If all saturation conditions hold, lock
65,536. If improvement remains meaningful or the result is gray, report a
data-limited or inconclusive regime and stop. A larger training set requires a
new preregistration and human authorization.

No test result may alter either decision. After size lock, the separate four-
architecture by three-seed selection may begin only under a later execution
gate.
