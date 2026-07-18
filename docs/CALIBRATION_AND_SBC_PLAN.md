# Calibration and independent SBC plan

## Scope

Preregistration `1.1.0-rc.5` is a downstream-only clarification. It preserves
the RC.4 estimand, direct-target data, likelihood/selection model, training
objective and split counts. It freezes the calibration algorithm before the
calibration-fit or SBC data are materialized.

## Credible-region level calibration

For each selected-architecture training seed, the raw flow supplies 4,096
posterior draws for every calibration-fit case. For either marginal target,
the empirical midrank PIT is `u` and the central-region inclusion score is:

```text
s = 2 |u - 1/2|.
```

For the joint posterior, the HPD inclusion score is the fraction of posterior
draws whose flow log density is at least the truth log density. A raw region of
mass `q` includes the truth exactly when its score is at most `q`.

For nominal level `a` and `n` calibration scores, the frozen threshold is the
sorted score at one-based rank:

```text
min(n, ceil((n + 1) a)).
```

There is no interpolation, clipping or smoothing after observing results. One
global diagnostic map and one map for each of the eight balanced EM cells are
fitted. The matching EM-cell map is primary for case-level intervals.

These maps calibrate reported marginal central intervals and joint HPD regions.
They do not alter posterior samples or create a new normalized flow density.
CRPS and point summaries therefore remain raw selected-model summaries.

## Independent SBC

The lowest 1,024 SHA-256-ranked IDs in the 2,048-system SBC namespace are used.
Every replicate has 1,024 posterior draws. Exchangeability ranks are computed
for:

- primary log absolute magnification;
- secondary log absolute magnification;
- their sum;
- their difference;
- joint flow log density.

Ranks range from 0 to 1,024. Exact ties are broken deterministically within the
tie block using a statistic- and physical-system-keyed SHA-256 value. The 1,025
possible ranks are assigned to 20 bins, so the chi-square expectation uses the
exact number of discrete ranks in each bin rather than assuming exactly 5%.
The five p-values receive Holm step-down correction at familywise alpha 0.01.

SBC cases never fit or revise the calibration map. They also provide an
independent coverage check after applying the already frozen EM-cell maps.

## Failure boundary

Failure lowers or narrows the scientific claim, or requires a new prospective
preregistration. It cannot reopen training, choose a different calibration
method, access final evaluation or tune on SBC.
