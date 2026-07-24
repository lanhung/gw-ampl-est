# Phase 7 final-score summary release-gate report

## Outcome

The final-score summary implementation and release gate passed at commit
`b4c7f0a07c717e2965ea3244c88c74d070449ce1`.

The software remains non-executable on scientific artifacts. It used synthetic
NPZ score fixtures only and opened no final record, checkpoint or calibration
map.

## Exact input contract

The future runtime requires:

- exactly fifteen frozen final namespaces;
- exactly seeds 0, 1 and 2;
- exactly 45 NPZ score artifacts plus companion summaries;
- exact artifact hashes from the completed final-inference authorization;
- no persisted posterior-draw array;
- identical physical-system IDs, truth, lens-family labels, EM-cell labels,
  split labels and diagnostic contexts across seeds;
- finite NLP, CRPS, region-score, coverage and interval-width arrays;
- an exact common selected architecture.

The IID namespace must contain all eight frozen EM cells in equal 1,024-case
groups. Score semantics additionally verify NLP against truth log density,
nonnegative CRPS/interval widths and region-score support.

## Frozen reporting

For every seed, the runtime reports:

- IID overall;
- IID by lens family;
- IID by each EM cell;
- every balanced-tail stratum;
- every cross-family cell;
- every parameter-OOD stratum;
- waveform mismatch and PSD mismatch;
- lens-family views within each diagnostic namespace.

Coverage outputs retain raw successes, totals and Wilson 95% intervals. IID
overall uses the frozen 0.01 marginal and 0.015 joint floors, IID EM cells use
0.04 and balanced-tail strata use 0.06, each combined with three binomial
standard errors. Diagnostic groups receive no pass/fail threshold.

All scalar metrics are aggregated across seeds as mean and sample standard
deviation. A failed IID or tail gate can only produce
`narrow_claims_and_report_failures`; it cannot select a seed, retune a model or
change a threshold.

## Release boundary

The packet is non-authorizing. After all score jobs complete, it will bind:

- the final-inference authorization;
- all 45 artifact and companion-summary hashes;
- exact implementation wheel and environment;
- one fresh summary output.

A separate delegated review must approve the packet before the runtime YAML
can exist. The runtime authorization permits only derived score reads and one
small summary. Final records, checkpoints, calibration maps, training,
calibration refitting, architecture/size selection, threshold changes,
manuscript-claim finalization and GWOSC/GWTC remain false.

## Verification

- focused tests: 8 passed;
- full tests: 535 passed, 8 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed over 76 source files;
- sdist and wheel build: passed;
- synthetic 45-artifact runtime and tamper regression: passed.
