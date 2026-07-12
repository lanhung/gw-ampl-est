# Phase 3B adaptive production preregistration report

Status: **design release candidate complete; execution disabled; human review required**.

## Authorization boundary

Human review accepted Phase 3A, and PR #4 merged through commit
`589b6a554d5bf8213c3014b5cb6f3b0e0f4edd5e`. Phase 3B was authorized for
design only. No waveform pair was generated, no model was trained or tuned, no
calibration/SBC/IID/OOD/mismatch procedure ran, and no GWOSC/GWTC product was
accessed.

The Phase 3A artifact remains permanently engineering-only:

- generator commit: `fbcd0616611d9cdf915ef0af030e6061c1be7f59`;
- dataset: `gwlens-v2-2.0.0-alpha.3-7081b2e8be3a84e1`;
- accepted pairs: 4,096;
- scientific/training/calibration/test use: false.

## New immutable design candidate

- version: `1.1.0-rc.1`;
- configuration:
  `configs/statistics/adaptive_scientific_production_preregistration.yaml`;
- canonical hash:
  `ba5dae2aa769331b917d3f622bfc967c607700f9908521576301841cb71d804b`;
- parent: RC.5 hash
  `4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb`.

RC.5 estimands, benchmark population, source-plane measure, observation models
and selection model are inherited. The release candidate changes allocation,
stopping and future authorization boundaries rather than silently changing the
evaluation target.

## Adaptive allocation

Training uses deterministic, cumulative physical-system ranks:

```text
16,384 ⊂ 32,768 ⊂ 65,536
```

The fixed development pool contains 6,144 validation, 4,096 calibration-fit
and 2,048 SBC systems. The fixed final pool contains 8,192 IID, 4,096 balanced
tail and four 2,048-system cross-family/OOD/waveform/PSD sets. Scientific totals
are therefore 49,152, 65,536 or 98,304 depending on the locked training rung.
The 4,096 Phase 3A pairs are excluded from every total.

All source, lens, physical-system, pair, noise-segment and augmentation-parent
groups are assigned before materialization and are disjoint across splits.
Noise augmentation never creates a new independent physical system.

## Stopping and model selection

One 10-transform, width-256 probe model with seeds 0/1/2 is trained from scratch
per rung under one optimizer and epoch policy. Only identical validation cases
can drive the paired 10,000-replicate bootstrap decision.

At 32k, stopping requires all frozen NLP, CRPS, coverage, EM-cell degradation
and three-seed conditions. Any gray or meaningfully improving result continues
to 65k. At 65k, meaningful improvement or ambiguity stops as data-limited or
inconclusive and requires a new preregistration; it never authorizes an
automatic larger rung.

Calibration-fit, SBC and all final evaluation splits are forbidden inputs to
scale or architecture selection. Only after size lock does the four-
architecture by three-seed selection occur at one size. Calibration, SBC and
final evaluation still require later individual execution gates.

## Proposal-efficiency future gate

Phase 3A's 0.2814% acceptance motivates, but does not authorize, a 512-pair
engineering qualification. The proposal-v2 design uses an exact mixture with a
positive 0.2 RC.5 broad-support safety component. This supplies the support
guarantee; finite qualification samples do not claim to prove support.

Adoption requires at least 2× acceptance or throughput, finite importance
weights, overall/family/EM-cell relative ESS thresholds, bounded maximum
normalized weight and frozen support checks. Failure or an incomparable/
ambiguous result retains RC.5.

## Resource projections

RC.5 projections linearly scale measured Phase 3A attempts, active time and
published bytes. Peak storage adds 5% failure evidence, a 20 GB reserve and one
active shard.

| Total | Attempts | Hours | Published bytes | Peak bytes | Remaining bytes |
|---:|---:|---:|---:|---:|---:|
| 49,152 | 17,468,388 | 71.35 | 53,408,334,708 | 76,217,835,649 | 254,232,768,383 |
| 65,536 | 23,291,184 | 95.13 | 71,211,112,944 | 94,910,752,797 | 235,539,851,235 |
| 98,304 | 34,936,776 | 142.70 | 106,816,669,416 | 132,296,587,092 | 198,154,016,940 |

The separate 2× proposal-v2 scenario is hypothetical and unmeasured. Storage
does not shrink in that scenario. Future execution must continuously persist
process-tree/cgroup peak RSS, time-integrated CPU use and peak staging bytes.

## Real-noise and catalog boundary

The proposed 91-event analysis is represented as a future versioned selection,
not a fixed fact. A separate gate must freeze release/products, inclusion rules,
event-list hash, detector/DQ rules, PSD/off-source noise, ranking statistic,
background, multiple-testing correction and null-result policy. Synthetic OOD
authorization cannot open GWOSC/GWTC access.

## Machine-readable verification

Twelve new tests prove the frozen hash, closed execution flags, permanent
Phase 3A exclusion, nested ladder, count arithmetic, development/final roles,
validation-only stopping, calibration/SBC separation, 65k hard stop,
proposal-v2 denial, measured projection arithmetic and GWOSC/GWTC denial.

Full local verification passed:

- pytest: 151 passed, three optional Lenstronomy skips;
- maintained-scope Ruff: passed;
- mypy: passed for 28 source files;
- sdist and wheel: built successfully with the known missing-README warning;
- canonical configuration hash: reproduced exactly.

Repository-wide Ruff retains the already documented 18 Phase 0 inventory-
builder findings; immutable audit evidence was not rewritten in this design
change.

## Completed, failed and deferred

Completed: versioned configuration, nested allocation, data-access boundary,
paired stopping rule, proposal-efficiency future gate, resource projections,
real-noise/catalog boundary, safety tests and project documentation.

Failed scientific or engineering runs: none; Phase 3B executed none.

Deferred behind new human gates: 512-pair proposal qualification, scientific
materialization, learning-curve training, architecture selection, calibration,
SBC, final evaluation, real-noise work, GWOSC/GWTC and Phase 3C.

Phase 3B stops here for human review.
