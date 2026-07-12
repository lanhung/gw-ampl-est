# Phase 3B.1 target and stopping hardening report

Status: **design correction complete; awaiting human review; execution disabled**.

## Scope and unchanged boundaries

Phase 3B.1 revised statistical semantics only. It generated no pair, trained
no model, fit no calibration, ran no SBC/IID/OOD/mismatch procedure, did not
run the proposal-v2 gate and did not access GWOSC/GWTC. It did not change the
Phase 3A generator, publication or parent RC.5 configuration.

The Phase 3A dataset
`gwlens-v2-2.0.0-alpha.3-7081b2e8be3a84e1` remains permanently excluded from
science.

## RC.2 identity

- version: `1.1.0-rc.2`;
- configuration:
  `configs/statistics/adaptive_scientific_production_preregistration.yaml`;
- canonical hash:
  `b94e7733d7fbb6f4c9dc4d5842b6a87f29e0515b4047b7b1604bca1438d15805`;
- parent RC.5 hash:
  `4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb`.

All execution and scientific-use flags remain false. Status is
`awaiting_human_review`.

## Corrected stopping and materialization

The 16,384-system rung is now `train_16k_probe_subset`, a strict subset of
`train_32k`, and cannot be a final lock. Stage A materializes 32,768 training
plus 6,144 validation systems. The first 16,384 may begin an authorized future
probe fit, but data generation continues to 32k.

Only two completed totals are achievable:

| Lock | Total |
|---|---:|
| 32,768 training | 65,536 |
| 65,536 training | 98,304 |

Stage B conditionally adds 32,768 training systems. Stage C, only after size
and architecture lock, adds 4,096 calibration, 2,048 SBC and 20,480 final
systems. A data-limited or inconclusive 65k result requires a new
preregistration.

## Proposal-corrected target

If future training uses proposal-v2, ordinary unweighted NPE is prohibited.
RC.2 freezes the normalized importance-weighted conditional NLP using complete
latent-state `p_eval/q_train` weights. Weights are globally mean-one within
each rung, unclipped, finite, privileged and excluded from deployable inputs.
Overall/family/EM-cell ESS and maximum-weight diagnostics are mandatory.

Validation, calibration-fit, SBC and IID use direct target-generative draws.
SBC cannot use uncorrected proposal-v2 draws. Tests fail closed if a changed
training proposal can silently change the declared posterior target.

## Deterministic final-evaluation commitment

RC.2 no longer claims accepted IDs are known before selection. The design
template `results/phase3b/final_evaluation_commitment.json` freezes the future
generator placeholder, preregistration identity, seed and attempt domains,
accepted-rank allocation, counts, distributions, versions, groups and
validators. Its current template SHA-256 is
`29a1b8487679e7f6a671395e47288c6bace45eb21b141f0ec94940391d14f272`.

The placeholder must be resolved and the commitment re-hashed before any
training. Later materialized accepted IDs must be verified as deterministic
outputs of that finalized commitment.

## Proposal-v2 and architecture efficiency

The mandatory proposal endpoint is now throughput. The frozen 95% interval's
lower bound for accepted pairs per active hour relative to RC.5 must be at
least 2.0. Acceptance and CPU/solver/density measures are secondary; acceptance
alone cannot pass the gate.

Before a future proposal A/B authorization, every candidate component requires a reviewed
executable sampler, exact normalized density, support and normalization
constants, conditionals, seed domains and numerical/boundary tests. The 20%
RC.5 safety mixture remains required.

At the locked rung, matching 10-transform/width-256 probe fits for seeds 0/1/2
must be reused. The architecture comparison has twelve results but at most nine
new fits.

## Physical systems, noise and resources

One stored Gaussian noise realization corresponds to each independent accepted
physical system. Training-time augmentation is currently unauthorized; any
future policy must remain within the parent split, count separately and be
identical across rungs.

Measured RC.5 linear projections are:

| Stage/total | Accepted systems | Attempts | Active hours | Published bytes |
|---|---:|---:|---:|---:|
| Stage A | 38,912 | 13,829,141 | 56.48 | 42,281,598,311 |
| Stage B increment | 32,768 | 11,645,592 | 47.57 | 35,605,556,472 |
| Stage C increment | 26,624 | 9,462,044 | 38.65 | 28,929,514,634 |
| 32k final | 65,536 | 23,291,184 | 95.13 | 71,211,112,944 |
| 65k final | 98,304 | 34,936,776 | 142.70 | 106,816,669,416 |

The unmeasured 2× proposal scenario is now applied only to training systems;
direct-target nontraining splits retain baseline rate. It projects 71.35 or
95.13 active hours and is explicitly not a completed result.

## Verification

- full local pytest: 154 passed, with three optional Lenstronomy skips;
- Phase 3B.1 focused safety tests: 15 passed;
- maintained-scope Ruff: passed;
- mypy: passed for 28 source files;
- source distribution and wheel: built successfully;
- canonical RC.2 hash and commitment SHA-256: reproduced exactly;
- staged count and resource arithmetic: passed.

The build retains the known missing-README warning. Repository-wide Ruff also
retains the previously documented 18 Phase 0 inventory-script findings; neither
is rewritten as a Phase 3B.1 scientific-design result.

## Completed, failed and deferred

Completed: RC.2 stopping semantics, staged arithmetic, weighted target
correction, direct-target evaluation rules, commitment template/hash,
throughput-only proposal gate, executable-density prerequisite, probe reuse,
noise semantics, projections and machine-readable safety tests.

Failed runs: none. No execution occurred. RC.1 is superseded design history.

Deferred: all data materialization, proposal-v2 implementation/qualification,
training, calibration, SBC, IID/OOD/mismatch evaluation, real noise,
GWOSC/GWTC and Phase 3C.

Phase 3B.1 stops here for human review.

Phase 3B.2 later supersedes RC.2 only for the ambiguous A/B accepted-pair
count. RC.2's other scientific and statistical conclusions remain review
history.
