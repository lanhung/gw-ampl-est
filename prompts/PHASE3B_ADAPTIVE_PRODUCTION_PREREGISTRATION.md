# Phase 3B — Adaptive scientific-production preregistration

Execute Phase 3B design work only.

Do not generate any waveform pair.
Do not train or tune any model.
Do not fit calibration.
Do not execute IID, OOD, SBC or mismatch tests.
Do not access GWOSC or GWTC.
Do not modify or scientifically reuse the Phase 3A qualification artifact.
Do not authorize Phase 3C or later phases.

Read first:

- `AGENTS.md`
- `docs/reports/PHASE3A_GENERATOR_QUALIFICATION_REPORT.md`
- `results/phase3a/qualification_validation.json`
- `results/phase3a/throughput.json`
- `configs/statistics/phase2_preregistration.yaml`
- `docs/DECISIONS.md`
- `docs/FAILURES.md`
- `docs/PROJECT_STATE.md`
- `docs/PHASE2_PREREGISTRATION.md`
- `docs/reports/PHASE2A_PREREGISTRATION_HARDENING_REPORT.md`

Work only on:

`phase3b/adaptive-production-preregistration`

======================================================================
1. PHASE 3A ACCEPTANCE
======================================================================

Record Phase 3A as accepted engineering qualification evidence.

Freeze:

- generator commit:
  `fbcd0616611d9cdf915ef0af030e6061c1be7f59`;
- qualification dataset:
  `gwlens-v2-2.0.0-alpha.3-7081b2e8be3a84e1`;
- exactly 4,096 permanently non-scientific pairs;
- observed acceptance and throughput;
- three-shard byte-identical resume;
- publication checksum and storage measurements.

The Phase 3A artifact must remain excluded from every future split.

======================================================================
2. NEW PREREGISTRATION VERSION
======================================================================

Create a new scientific-production preregistration:

`1.1.0-rc.1`

Do not overwrite RC.5.

Create:

- `configs/statistics/adaptive_scientific_production_preregistration.yaml`;
- `docs/ADAPTIVE_SCIENTIFIC_PRODUCTION_PLAN.md`;
- `docs/LEARNING_CURVE_STOPPING_RULE.md`;
- `docs/reports/PHASE3B_ADAPTIVE_PRODUCTION_PREREGISTRATION_REPORT.md`.

Generate a new canonical configuration hash.

All execution flags remain false.

======================================================================
3. FREEZE NESTED TRAINING LADDER
======================================================================

Freeze cumulative nested training systems:

- rung 1: 16,384;
- rung 2: 32,768;
- rung 3: 65,536.

Require:

`train_16k` is a strict subset of `train_32k`,
and `train_32k` is a strict subset of `train_65k`.

Freeze all source, lens, physical-system, noise and pair group assignments
before materialization.

No test or diagnostic case may enter any training rung.

======================================================================
4. FREEZE DEVELOPMENT POOL
======================================================================

Freeze exactly 12,288 development systems:

- validation: 6,144;
- calibration_fit: 4,096;
- sbc_diagnostic: 2,048.

Roles:

- validation may be used for learning-curve stopping and architecture selection;
- calibration_fit may only fit final post-hoc calibration after architecture
  and training size are frozen;
- sbc_diagnostic may only run SBC and may never fit calibration.

The three sets must be group-disjoint.

======================================================================
5. FREEZE FINAL EVALUATION POOL
======================================================================

Freeze exactly 20,480 final evaluation systems:

- iid_test: 8,192;
- balanced_tail_diagnostic: 4,096;
- cross_family_misspecification_test: 2,048;
- parameter_region_ood: 2,048;
- waveform_mismatch_test: 2,048;
- psd_mismatch_test: 2,048.

Freeze IDs, seeds, distributions and assignments before training.

Do not permit learning-curve stopping, architecture selection, calibration
fitting or proposal tuning to inspect these cases.

They may be materialized only after the training size is locked, or may be
materialized earlier but remain cryptographically sealed and unread.

======================================================================
6. LEARNING-CURVE PROBE
======================================================================

Freeze a single probe model for training-size decisions:

- conditional NSF;
- 10 transforms;
- conditioner width 256;
- existing mask-aware GW and EM encoders;
- seeds 0, 1 and 2;
- identical optimizer, epoch budget, early stopping and batch policy at all
  training sizes;
- every rung trained from scratch;
- no post-hoc calibration during scale selection.

Compare rungs on identical validation cases using paired statistics.

Primary metric:

- validation negative log probability per target dimension.

Secondary metrics:

- CRPS;
- raw marginal coverage error;
- joint coverage error;
- EM-cell conditional coverage;
- tail-development summaries;
- interval width.

Do not use IID/OOD metrics.

======================================================================
7. STOPPING RULE
======================================================================

At 32,768 systems, stop growth when all are true:

1. the upper bound of the paired-bootstrap 95% confidence interval for
   16k-to-32k NLP improvement is below 0.01 nat per target dimension;
2. median CRPS improvement is below 1%;
3. maximum marginal-coverage-error improvement is below 0.005;
4. every development EM cell meets its frozen tolerance;
5. no EM cell coverage degrades by more than 0.02;
6. the three seeds agree that the improvement is below the threshold.

If any rule indicates meaningful improvement, or a confidence interval
straddles the threshold, continue to 65,536.

At 65,536:

- apply the same 32k-to-65k comparison;
- if still clearly data limited, stop and require a new preregistration;
- do not automatically generate any larger training set.

Report every seed and paired bootstrap interval.

======================================================================
8. FINAL MODEL-SELECTION RULE
======================================================================

After training size is frozen, run the existing four architecture
combinations and three seeds at that size only.

Select architecture by mean validation NLP across three seeds.

Do not select the best seed.

Report all three seeds.

Only after architecture and size are frozen may calibration_fit, SBC and
final evaluation be opened according to their individual gates.

======================================================================
9. PROPOSAL-EFFICIENCY PLAN
======================================================================

Use the Phase 3A rejection summary only as engineering evidence, never as a
scientific split.

Design a proposal-v2 candidate that preserves the frozen evaluation target
while improving generation efficiency.

The proposal may include:

- caustic-aware source-plane components;
- a broad-support safety component;
- distance/magnification stratification;
- exact normalized proposal log density;
- explicit importance weights.

Create:

`docs/PROPOSAL_EFFICIENCY_QUALIFICATION_PLAN.md`

Freeze a future 512-accepted-pair engineering qualification gate.

Proposal v2 may be adopted only if:

- support covers the complete frozen benchmark;
- all importance weights are finite;
- weight ESS exceeds a preregistered threshold;
- acceptance or accepted-pair throughput improves by at least 2×;
- lens-family, multiplicity, EM-cell and tail support checks pass.

If the gate fails, retain RC.5 proposal behavior.

Do not run the 512-pair qualification in Phase 3B.

======================================================================
10. RESOURCE PROJECTIONS
======================================================================

Using measured Phase 3A throughput and byte rate, calculate projected:

- attempts;
- active wall time;
- published storage;
- peak storage;
- remaining disk;

for total accepted counts:

- 49,152;
- 65,536;
- 98,304.

Report baseline RC.5 projections and separate proposal-v2 hypothetical
projections. Do not present unmeasured proposal-v2 speedups as completed
results.

Add continuous process-tree/cgroup peak RSS and time-integrated CPU
utilization requirements to future production execution.

======================================================================
11. REAL-NOISE AND CATALOG BOUNDARY
======================================================================

Create a design-only future protocol for:

- real-noise injections;
- counterfactual companion injections;
- catalog pair scan.

The proposed 91-event catalog must be represented as a future, versioned
event inclusion list, not assumed to be a permanently correct event count.

Require a separate authorization, release/version freeze, event-list hash,
DQ rules, PSD protocol, ranking statistic, background and multiple-testing
correction.

No GWOSC/GWTC access occurs in Phase 3B.

======================================================================
12. MACHINE-READABLE SAFETY TESTS
======================================================================

Add tests proving:

- every execution flag remains false;
- no Phase 3A ID enters a scientific split;
- nested train membership is correct;
- all development and final splits are group-disjoint;
- total-count arithmetic is correct;
- final evaluation cannot be used for stopping;
- calibration_fit and SBC remain independent;
- 65k extension cannot occur automatically;
- proposal-v2 qualification remains unauthorized;
- GWOSC/GWTC remains unauthorized.

======================================================================
13. REQUIRED UPDATES
======================================================================

Update:

- `AGENTS.md`;
- `docs/DECISIONS.md`;
- `docs/PROJECT_STATE.md`;
- `docs/FAILURES.md`;
- `results/experiment_registry.csv`;
- storage and runtime estimates.

Do not modify the Phase 3A generator commit or published artifact.

The current known Phase 0 Ruff findings may be documented or handled in a
separate maintenance change, but do not rewrite immutable audit evidence as
part of the scientific design.

======================================================================
14. ACCEPTANCE CRITERIA
======================================================================

Phase 3B passes only when:

1. the new preregistration has a new version and hash;
2. the 16k/32k/65k ladder is nested;
3. stopping uses development data only;
4. final IID/OOD data cannot affect scale selection;
5. calibration and SBC remain independent;
6. 65k does not authorize automatic extension;
7. proposal-efficiency qualification has a separate future gate;
8. resource projections use measured Phase 3A values;
9. the Phase 3A artifact remains permanently non-scientific;
10. real-noise/GWTC remains separately gated;
11. no pair is generated;
12. no model is trained;
13. no calibration or scientific test is run.

Run:

- pytest;
- Ruff on maintained scope;
- mypy;
- package build;
- configuration hash and split arithmetic checks.

Commit with:

`docs: preregister adaptive scientific production ladder`

Push the branch and stop for human review.
