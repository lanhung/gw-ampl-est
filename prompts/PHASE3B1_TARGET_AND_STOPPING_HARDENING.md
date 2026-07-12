# Phase 3B.1 — Harden adaptive production target and stopping semantics

Work only on:

`phase3b/adaptive-production-preregistration`

Start from the clean checkpoint:

`022a5f275f183f09be21533b4fc204a1e6c3acfa`

This remains a design-only phase.

Do not generate any pair.
Do not train or tune any model.
Do not fit calibration.
Do not run SBC, IID, OOD or mismatch evaluation.
Do not access GWOSC or GWTC.
Do not modify the Phase 3A artifact or generator commit.
Do not authorize proposal-v2 qualification or scientific production.

Read:

- `AGENTS.md`
- `configs/statistics/adaptive_scientific_production_preregistration.yaml`
- `docs/ADAPTIVE_SCIENTIFIC_PRODUCTION_PLAN.md`
- `docs/LEARNING_CURVE_STOPPING_RULE.md`
- `docs/PROPOSAL_EFFICIENCY_QUALIFICATION_PLAN.md`
- `docs/reports/PHASE3B_ADAPTIVE_PRODUCTION_PREREGISTRATION_REPORT.md`
- `configs/statistics/phase2_preregistration.yaml`
- `docs/reports/PHASE3A_GENERATOR_QUALIFICATION_REPORT.md`

Update preregistration version from:

`1.1.0-rc.1`

to:

`1.1.0-rc.2`

and generate a new canonical hash.

======================================================================
1. REMOVE THE NON-EXECUTABLE 16K FINAL STOP
======================================================================

Keep the nested training subsets:

- 16,384;
- 32,768;
- 65,536.

Rename or explicitly classify 16,384 as:

`train_16k_probe_subset`

It is a learning-curve probe subset of `train_32k`, not an independently
lockable final training size.

The only final size decisions in RC.2 are:

- lock 32,768 after the 16k-to-32k comparison passes every saturation rule;
- otherwise continue to 65,536;
- at 65,536, lock 65,536 if saturated;
- otherwise stop as data-limited/inconclusive and require a new
  preregistration.

Remove 49,152 from the list of achievable final scientific totals.

The achievable final totals are:

- 65,536 when training locks at 32,768;
- 98,304 when training locks at 65,536.

Keep 16k metrics as probe evidence only.

======================================================================
2. FREEZE MATERIALIZATION SEQUENCE
======================================================================

Define future materialization stages:

Stage A — scale-selection data:

- train_32k: 32,768 accepted physical systems;
- validation: 6,144 accepted physical systems;
- total: 38,912.

The first 16,384 training systems form the 16k probe subset.
The 16k probe may begin once available under a future training authorization,
but generation to 32k continues regardless of the 16k result.

Stage B — conditional extension:

- add 32,768 training systems only when the preregistered 32k rule requires
  continuation to 65k.

Stage C — post-lock data:

- calibration_fit: 4,096;
- sbc_diagnostic: 2,048;
- final evaluation: 20,480;
- total: 26,624.

Stage C begins only after training size and architecture are locked, unless
a separate reviewed sealed-materialization authorization says otherwise.

Update runtime and storage projections for every stage.

======================================================================
3. DEFINE TRAINING-PROPOSAL TARGET CORRECTION
======================================================================

The project distinguishes:

- scientific evaluation target distribution `p_eval(theta)`;
- efficient scientific training proposal `q_train(theta)`.

If proposal-v2 is adopted for training data, ordinary unweighted NPE training
is forbidden.

Freeze the training objective as importance-weighted conditional negative log
probability:

    loss =
      sum_i w_i * [-log r_phi(mu_i | x_i)] / sum_i w_i

where:

    log w_i = log p_eval(theta_i) - log q_train(theta_i)

Requirements:

- weights are computed from the full latent proposal/evaluation variables;
- exact log proposal and evaluation densities are stored as privileged
  provenance;
- weights are not deployable model inputs;
- weights are normalized globally within each training rung to mean one;
- weight clipping is forbidden in this preregistration;
- all weights must be finite;
- overall, per-family and per-EM-cell ESS are reported;
- optimizer batches are sampled uniformly and use the frozen per-example
  weights, unless an exactly equivalent reviewed weighted sampler is specified;
- the same weighting semantics are used at 16k, 32k and 65k.

Validation must represent the evaluation target rather than the training
proposal. Choose and freeze one of:

A. direct draws from the evaluation generative distribution — preferred; or
B. a fully specified target-weighted validation estimator.

Calibration-fit, SBC and IID must use direct draws from their declared target
generative distributions. In particular, SBC may not use uncorrected
proposal-v2 draws.

OOD and mismatch splits continue to use their individually declared
distributions.

Add tests proving that changing `q_train` without correction cannot silently
change the declared posterior target.

======================================================================
4. REPLACE IMPOSSIBLE PRE-MATERIALIZATION ID FREEZE
======================================================================

Concrete accepted IDs cannot be known before running the selection pipeline.

Replace the current wording with a deterministic cryptographic generation
commitment.

Create a schema/specification for:

`results/phase3b/final_evaluation_commitment.json`

The commitment must contain:

- future scientific generator commit placeholder;
- preregistration version and hash;
- schema version;
- root seed;
- split-specific seed domains;
- attempt-stream namespaces;
- attempt-ID allocation rules;
- accepted-rank allocation rules;
- split counts;
- proposal/evaluation distribution IDs;
- waveform/PSD/observation versions;
- grouping rules;
- expected manifest-validation rules.

Before training begins, the commitment must be finalized and hashed.

After later materialization, actual accepted IDs and manifests must be
verified as deterministic outputs of this commitment.

Do not claim that unknown accepted IDs themselves were frozen before
materialization.

======================================================================
5. MAKE THROUGHPUT THE PROPOSAL-V2 PRIMARY GATE
======================================================================

The proposal-v2 adoption endpoint is total production efficiency, not raw
acceptance alone.

Replace:

`acceptance OR throughput >= 2x`

with:

- primary mandatory endpoint:
  accepted pairs per active hour ratio versus RC.5;
- required lower bound:
  the frozen 95% confidence-interval lower bound must be at least 2.0;
- secondary reported endpoints:
  acceptance-rate ratio,
  attempts per accepted pair,
  CPU seconds per accepted pair,
  lens-solver time,
  proposal-density evaluation time.

Freeze an A/B design using:

- the same hardware and environment;
- the same worker count;
- the same generator code path where possible;
- an RC.5 control and proposal-v2 candidate;
- a fixed accepted count or attempt budget;
- an exact confidence-interval method;
- no post-result endpoint switching.

An acceptance-rate gain without a throughput gain must not authorize
proposal-v2 adoption.

======================================================================
6. REQUIRE EXECUTABLE PROPOSAL COMPONENTS BEFORE THE 512-PAIR GATE
======================================================================

The current proposal component names are conceptual.

Before any future 512-pair authorization, require a separately reviewed
executable proposal specification defining for every component:

- sample algorithm;
- exact normalized log-density evaluator;
- support;
- truncation and normalization constants;
- mixture-variable semantics;
- all conditionals;
- deterministic seed domains;
- numerical normalization tests;
- boundary tests;
- full latent-variable coverage.

The 20% RC.5 safety mixture remains required.

Phase 3B.1 does not implement or run the proposal.

======================================================================
7. REUSE LOCKED-RUNG PROBE FITS
======================================================================

When the locked training size is 32k or 65k, the existing:

- 10 transforms;
- width 256;
- seeds 0, 1 and 2

probe fits at that exact size must be reused in final architecture selection,
provided their data, objective, optimizer and training budget match exactly.

Freeze:

- maximum architecture results: 12;
- maximum additional fits after size lock: 9;
- retraining the identical probe combination without a declared failure is
  forbidden.

======================================================================
8. DEFINE PHYSICAL-SYSTEM AND NOISE-REALIZATION SEMANTICS
======================================================================

Freeze whether scientific storage contains:

- exactly one stored Gaussian noise realization per accepted physical system;
  or
- a fixed declared number.

Any training-time noise augmentation:

- must remain within the parent physical-system split;
- must not count as another physical system;
- must use identical augmentation policy across rungs;
- must not leak validation/final noise or source identities;
- must be frozen before training authorization.

Update learning-curve sample-size language to refer consistently to independent
physical systems.

======================================================================
9. CLOSE THE PHASE 3B GATE
======================================================================

After revision:

- status becomes `awaiting_human_review`;
- all execution and training flags remain false;
- proposal-v2 remains unauthorized;
- scientific materialization remains unauthorized;
- GWOSC/GWTC remains unauthorized;
- Phase 3C remains unauthorized.

Update:

- `AGENTS.md`
- adaptive preregistration YAML
- adaptive production plan
- learning-curve stopping rule
- proposal-efficiency plan
- Phase 3B report
- decisions
- failures
- project state
- experiment registry
- resource projections

Add tests for:

- no 16k final lock;
- only 32k/65k final totals;
- staged materialization arithmetic;
- weighted training-target correction requirement;
- direct-target SBC requirement;
- deterministic future-evaluation commitment;
- throughput-only proposal adoption;
- exact proposal specification required before authorization;
- probe-fit reuse;
- physical-system/noise semantics;
- all execution flags false.

Run:

- `python -m pytest -q`
- maintained-scope Ruff
- mypy
- package build
- canonical hash check
- count/resource arithmetic checks

Create:

`docs/reports/PHASE3B1_TARGET_AND_STOPPING_HARDENING_REPORT.md`

Commit with:

`docs: harden adaptive target correction and stopping semantics`

Push and stop for human review.
