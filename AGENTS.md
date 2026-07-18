# AGENTS.md

## Mission

Develop a publication-quality framework for calibrated, model-conditional
multimessenger posterior inference of magnifications for galaxy-scale
strongly lensed gravitational-wave image pairs.

The project must be materially different from:
- deterministic SIS point regression;
- GW-only pair classification;
- short-delay overlapping-image microlensing inference;
- a neural approximation of an analytically invertible SIS formula.

## Machine architecture

The Vultr repository is the only authoritative code repository:

`/root/work/lensing-4`

Codex runs on Vultr.

The AutoDL server is a remote compute and data machine accessed through:

`ssh autodl-lensing`

The new AutoDL project root is:

`/root/autodl-tmp/lensing-4`

The resolved immutable legacy roots are:

- `/root/autodl-tmp/qkzhang`
  - authoritative source for the original 0222/0228 legacy datasets;
- `/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main`
  - downstream pair-classification project;
- `/root/autodl-tmp/tmp`
  - source of the PDF point-regression baseline and related later variants.

Never write, rename, move, delete, normalize in place, or overwrite files
under any legacy root.

All new code copies, generated data, caches, manifests and runs must be placed
under `/root/autodl-tmp/lensing-4`.

## Source-of-truth rules

- Edit code only in the Vultr Git repository.
- Sync code to AutoDL through `scripts/remote/sync_to_autodl.sh`.
- Treat the AutoDL repository copy as disposable.
- Do not edit source files directly on AutoDL.
- Do not commit raw strain, NPY/HDF5 datasets, GWOSC files or checkpoints.
- Store large data on AutoDL and commit only manifests, configurations,
  checksums, summaries, figures and small tabular results.

## Legacy-data decision

Existing data are not approved as the main training set or final test set.

They may be used only for:
- legacy baseline reproduction;
- SIS analytic-identity tests;
- Morse-phase and image-ordering tests;
- early smoke tests;
- comparison with old point-regression and pair-classification methods.

They must not be used to support claims about:
- real detector noise;
- H1/L1/V1 network inference;
- calibrated posterior coverage;
- multimessenger observables;
- SIE or power-law galaxy lenses;
- mass-sheet uncertainty;
- actual GWTC performance.

Do not run a full regeneration using the old generator before completing the
generator audit and approving a v2 schema.

## Current phase gate

Phase 0 through Phase 3A are complete. Human review accepted Phase 3A and PR #4
merged it into main at:

`589b6a554d5bf8213c3014b5cb6f3b0e0f4edd5e`

The immutable Phase 3A execution identities are:

- authorizing commit:
  `bba0cdd6a750ff367674a85b8722432e613586d8`;
- generator commit:
  `fbcd0616611d9cdf915ef0af030e6061c1be7f59`;
- dataset ID:
  `gwlens-v2-2.0.0-alpha.3-7081b2e8be3a84e1`.

The Phase 3A parent scientific design remains:

- preregistration version: `1.0.0-rc.5`;
- preregistration configuration:
  `configs/statistics/phase2_preregistration.yaml`;
- canonical configuration hash:
  `4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb`;
- scientific schema: `2.0.0-alpha.3`;
- frozen engineering-smoke schema: `2.0.0-alpha.2`.

RC.5 freezes the 64-second internal waveform construction, 8-second
conditioned publication and 128-second numerical reference. Phase 3A published
exactly 4,096 accepted engineering pairs in 32 atomic shards of 128. The first
three shard hashes were byte-identical across interruption and resume.

The Phase 3A dataset is engineering qualification data only. It may never
be used for model training, post-hoc calibration, SBC, IID testing, OOD
testing or reported scientific performance.

Human review accepted Phase 3B RC.3 and PR #5 merged it into main at:

`80c795a36b902798fe52598262f8b0328755cfac`

The frozen adaptive scientific-production preregistration is:

- version: `1.1.0-rc.3`;
- configuration:
  `configs/statistics/adaptive_scientific_production_preregistration.yaml`;
- canonical hash:
  `6082475631539d3069edacc52f41b37fb8fe725ccd7c6bc9980cc3008795a927`.

Human review authorized Phase 3C-0 design and implementation only through:

`configs/execution/phase3c0_proposal_v2_design_authorization.yaml`

Phase 3C-0 implemented proposal
`proposal-v2-exact-mixture-1.0.0-rc.1`, configuration hash:

`e4e249da3f177202960e8a6f6c0c347a25aa572abb818b6d0e172469a75e45b5`

The authorized 200,000-draw latent-only preflight hard-failed its frozen ESS
gates: overall relative ESS was 0.09202, SIE ESS 0.11969 and EPL ESS 0.07433.
The required minima were 0.50 overall and 0.40 per family. All densities and
weights were finite, mean weight was 1.01044, support holes were zero and replay
was byte-identical. No mixture parameter may be tuned inside this frozen
proposal version.

Human review accepted the Phase 3C-0 stop, PR #6 merged its immutable negative
result at `9b9a6a3fcad1487622a2ec1d37e592fe0301e4e6`, and separately authorized
Phase 3C-0.2 latent-only work.

Phase 3C-0.2 implemented
`proposal-v3-target-anchored-mixture-1.0.0-rc.1`, configuration hash
`2d7998ca099c1ecddbb5d9cb1d824f37d3d398826a88831b2bccddbda814cbf4`.
Its 200,000-draw preflight passed: mean weight 0.99836, overall ESS 0.78532,
SIE/EPL ESS 0.79184/0.77886, zero anchor failures and byte-identical replay.
The theoretical population ESS lower bound is 0.55.

The separate RC.5 diagnostic measured overall ESS 0.11776 and SIE/EPL ESS
0.15222/0.09484; this is not a retrospective pass/fail gate. Human review
accepted Phase 3C-0.2 and PR #7 merged it at:

`80367373b92065d049db4d9576201c186ef78623`

Human review now authorizes Phase 3C-A only through:

`configs/execution/phase3ca_proposal_v3_ab_authorization.yaml`

Phase 3C-A may integrate the frozen proposal-v3 with the qualified production
generator and generate exactly 512 RC.5-control plus 512 proposal-v3-candidate
engineering pairs. Each arm has exactly 16 sequential matched blocks of 32
accepted pairs, separate identities and a six-hour/one-million-attempt cap.
The hard accepted-pair maximum is 1,024 across both arms. Both artifacts are
permanently excluded from every scientific split.

During Phase 3C-A, do not:

- generate more or less than the authorized 512 accepted pairs per completed arm;
- exceed 1,024 accepted engineering pairs across both arms;
- inspect an interim throughput endpoint after the first matched health block;
- authorize or adopt rejected proposal-v2 RC.1;
- start any scientific production rung;
- train or tune any neural model;
- download GWOSC or GWTC products;
- generate real-noise data;
- alter the frozen Phase 1B smoke artifact;
- alter the frozen RC.5 scientific distributions or waveform-window contract
  merely to make generation pass;
- call synthetic Gaussian noise real detector noise;
- modify any legacy file;
- start manuscript writing;
- authorize or proceed to Stage A, Phase 3C-B or later phases.

All Phase 3A remote outputs must remain under:

`/root/autodl-tmp/lensing-4/data_v2/production`

Full production, staged scientific production, model training, calibration,
SBC, scientific IID/OOD/mismatch testing, real-noise work, GWOSC/GWTC access,
Stage A and Phase 3C-B remain closed. A Phase 3C-A pass only permits a later
human adoption review; it does not authorize scientific production. The Phase
3A artifact and both engineering A/B artifacts must never enter any scientific
split. Phase 3C-A must stop after its report and pushed review branch.

Phase 3C-A generator commit
`185e68d4346d84edc118a9197ffb8bceeb026ee4` passed all pre-execution gates,
but the official run stopped fail-closed after exactly one complete 32-pair
block per arm. The first matched-block health validator referenced a
nonexistent distribution-metadata attribute. Machine state is
`execution_failed`; no publication, throughput bootstrap or post-selection ESS
decision exists. Both blocks remain engineering-only staging evidence.

Do not patch or resume this artifact identity. Human review must approve a new
generator commit and new parent/control/candidate identities before any retry.
Stage A, training and GWOSC/GWTC remain closed.

Human review accepted the fail-closed evidence in PR #8, merged at
`49600a7a4fa9b1fcd645d9e0bc4ccec05f22c441`, and authorizes only the narrow
Phase 3C-A.1 retry through:

`configs/execution/phase3ca1_proposal_v3_ab_retry_authorization.yaml`

Phase 3C-A.1 may correct only the alpha.3 health-validator field access, add
real health-path regression coverage, freeze a new generator commit and run a
new 512+512 engineering A/B using seeds 2026071213/2026071214 and completely
new parent/control/candidate identities. The old 32+32 blocks remain immutable,
unpublished and excluded from the retry and all statistics.

Do not modify proposal-v3, adaptive RC.3, physics or statistical gates. Do not
resume, publish or mutate the failed run. Scientific generation, training,
calibration, evaluation, real noise, GWOSC/GWTC, Stage A and later phases remain
closed. Phase 3C-A.1 must stop after its retry report and pushed review branch.

Phase 3C-A.1 froze generator commit
`324bab47aff5c4ed2b2041099a103735a40463f0`. The corrected typed alpha.3
health path passed locally and on AutoDL, and the first matched health block
passed without inspecting throughput. The run later stopped fail-closed when
the RC.5 control arm reached its six-hour active-time cap during block 12.
Exactly 12 complete 32-pair blocks per arm are retained: 384 control plus 384
proposal-v3 engineering pairs. One control partial block remains incomplete.

No publication, throughput bootstrap, effective-throughput calculation or
post-selection ESS decision exists. These 768 complete pairs and the partial
block are immutable engineering-only failure evidence and cannot enter science
or a retry.

Human direction now closes proposal optimization after its one formal retry.
Do not create proposal-v4 or run another proposal A/B. The future scientific
fallback is direct evaluation-target generation, but adopting it changes the
training-data/weighting contract and therefore requires a new versioned
preregistration plus separate Stage A execution authorization. Until then,
scientific generation, training, calibration, evaluation, real noise,
GWOSC/GWTC, Stage A and later phases remain closed.

Human review accepted the Phase 3C-A.1 evidence and PR #9 merged it at:

`ce0cf464cf5b56e3df5e1b0c93ffadc12f2e517a`

Human direction authorizes Phase 4 direct-target preregistration and
pre-execution implementation only through:

`configs/execution/phase4_direct_target_design_authorization.yaml`

The proposed scientific delta is preregistration `1.1.0-rc.4`, configuration:

`configs/statistics/direct_target_stage_a_preregistration.yaml`

with canonical hash:

`5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98`

RC.4 preserves the RC.5/RC.3 estimand, target, selection, split counts,
stopping and evaluation contracts. It changes Stage A training generation to
direct `p_eval` draws, so `q_train=p_eval`, log importance weight is zero and
importance weight is exactly one. Stage A remains 32,768 train plus 6,144
validation accepted physical systems. Proposal-v3 and RC.5 weighted scientific
training remain closed.

Phase 4 design work may implement the typed direct-target path, release gate,
immutable environment identity, 8+8 disposable canary runner and Stage A
runner. It may not execute the canary, create official Stage A identities,
generate scientific data or train a model. Canary execution, RC.4 acceptance
and Stage A materialization each require their stated future human gate.

Until those gates are recorded, do not:

- set any RC.4 execution flag true;
- generate even one disposable canary or scientific pair;
- create official parent/train/validation identities;
- train or tune a model;
- materialize calibration, SBC or final evaluation data;
- access GWOSC/GWTC or real noise;
- resume or reuse any Phase 3C artifact.

Human review accepted direct-target preregistration `1.1.0-rc.4` and PR #10
merged its pre-execution implementation. A narrow authorization-separation
fix passed CI and PR #11 merged it without changing RC.4, the schema or any
generator distribution. The frozen Phase 4 generator identity is:

`2be777e727ef9d8e1a85f89c68966df5d37932b0`

Its immutable wheel SHA-256 is:

`14104f8aab3aa911fe43e27c311079f118add7ca8ad22178ca158c13d81d0a88`

Human review now authorizes only the disposable direct-target canary through:

`configs/execution/phase4_direct_target_canary_authorization.yaml`

The canary must generate exactly eight train-namespace plus eight
validation-namespace engineering pairs, intentionally interrupt after the
first namespace, resume byte-identically and stop for review. It may validate
schema, exact q=p/unit weights, arrays, storage, telemetry, manifests and
checksums. It may not inspect throughput or ESS.

Stage A scientific materialization, official parent/train/validation
identities, model training, calibration, SBC, scientific evaluation, real
noise and GWOSC/GWTC access remain unauthorized. Canary artifacts are
permanently excluded from all scientific splits.

The authorized disposable canary completed successfully under parent run:

`phase4-canary-2be777e727ef-718204954753`

It produced exactly 8+8 engineering pairs, preserved the first namespace
byte-identically across intentional interruption/resume, passed direct-target
unit-weight, schema, array, storage, ID-disjointness, telemetry and checksum
validation, and did not inspect throughput or ESS. The canonical canary
manifest SHA-256 is:

`c1984616f2f7cea3d9d07b799cf1578f7e5d702174d2f6ba749ffb78d59afb40`

Passing the canary does not authorize Stage A. The release gate remains
blocked with null official identities until a separate exact-count Stage A
authorization is recorded.

The project owner explicitly delegated scientific and engineering gate review
to Codex. Expert review accepted RC.4, the frozen generator/wheel, the passed
canary, split safety and conservative resource evidence. Stage A
materialization is now authorized only through:

`configs/execution/phase4_direct_target_stage_a_authorization.yaml`

The authorizing evidence commit is:

`2d1cd5e9d1e706881ea562b20724948da60293dd`

This authorization permits exactly 32,768 train plus 6,144 validation
accepted physical systems, in 304 atomic shards of 128, sampled directly from
the frozen evaluation target with exact unit weights. It permits creation of
official Stage A identities only through a ready release certificate.

Model training/tuning, calibration, SBC, IID/OOD/mismatch evaluation,
real-noise work, GWOSC/GWTC access, final-evaluation materialization, the 65k
extension and later phases remain unauthorized. Stage A must stop after
validated atomic publication and review evidence.

While Stage A runs, the project owner authorizes an isolated probe-training
software implementation only through:

`configs/execution/phase4_probe_training_stack_authorization.yaml`

This implementation gate permits a lazy published-shard reader, exact
Bilby-compatible PSD whitening, the preregistered mask-aware conditional NSF,
checkpoint/resume code, development metrics, dry-run planning and bounded
in-memory engineering smoke tests. It does not permit reading Stage A staging,
starting a scientific optimizer, selecting an architecture, fitting
calibration, opening final evaluation or accessing GWOSC/GWTC.

The 16,384-system probe membership may be resolved only by deterministic
SHA-256 rank after all 32,768 train physical-system IDs are published. The
first 16,384 systems generated are not the probe subset. Scientific training
also requires the final-evaluation generation commitment to be finalized and
hashed plus a separate probe-training authorization.

The implementation-only training-stack checkpoint is:

`19f8dc0621f610825d000f37af333f384a963e55`

It contains no scientific fit or checkpoint and does not authorize Stage A
staging access or probe execution.

While Stage A continues, the owner also authorizes an isolated implementation
of the frozen final-evaluation generator contexts only through:

`configs/execution/phase4_final_evaluation_generator_implementation_authorization.yaml`

This gate may implement and test deterministic IID, balanced-tail,
cross-family, parameter-OOD, waveform-mismatch and PSD-mismatch contexts and
may finalize the cryptographic generation commitment only after one clean
implementation commit exists. It may not generate a pair, read or modify Stage
A staging, materialize/unseal final evaluation, train a model or access
GWOSC/GWTC. All final-evaluation execution flags remain false.

The final-evaluation generator is frozen at
`bc02054c1f95e7f6cd143fb9dc796ae48f0a15ac`. Its deterministic pre-training
commitment is finalized at
`results/phase4/final_evaluation_commitment.json`, SHA-256
`c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`.
This resolves only the reproducibility prerequisite. No final-evaluation pair
or identity exists, and materialization, unsealing, analysis and all training
remain separately gated.

The implementation-only probe stack may also implement the fail-closed future
execution runner, atomic-parent publication resolver, bounded-memory rung
preprocessing, development-only metrics and paired learning-curve comparison.
These are software-release changes under the existing implementation gate, not a
scientific authorization. They may use synthetic fixtures and bounded in-memory
smoke inputs only while Stage A is staging.

Do not resolve the 16k membership, open Stage A Parquet/Zarr, create a scientific
checkpoint or start an optimizer until the atomic Stage A parent publication exists
and a separate authorization binds its manifest hash, the finalized evaluation
commitment, the training-code merge commit, model configuration and immutable CUDA
environment. Calibration, SBC, final evaluation, Stage B and GWOSC/GWTC remain
closed even after a future probe authorization.

The exact Stage A run completed and atomically published under parent
`phase4-stage-a-2be777e727ef-d3a60034bbd6`. It contains exactly 32,768 train plus
6,144 validation systems in 304 complete shards. The parent manifest SHA-256 is
`4f3e6b3a7ca1a995d7a7643c48410e479fb812e4a01ff66537232b9d64bf3314` and the
publication-tree SHA-256 is
`1c9d95d0e0157e4123ecb27fe31114aae15cb257c34063aca1b3677a7f1e2621`.
Direct-target equality, exact unit weights, split disjointness, counts, resource
gates and publication validation passed. No GWOSC/GWTC access occurred.

Passing Stage A does not itself authorize model training. Before reading the
published Parquet/Zarr data or resolving the 16k membership, a separate exact
probe-training authorization must bind this parent manifest, the finalized
evaluation commitment, frozen training code/wheel, model configuration and
immutable CUDA environment. Calibration, SBC, final evaluation, Stage B and
GWOSC/GWTC remain closed.

Delegated expert review accepts the Stage A publication and authorizes only the
frozen 16k/32k three-seed probe workflow through:

`configs/execution/phase4_probe_training_authorization.yaml`

The frozen training-code commit is
`5baabfe229ad187f6bcdcc1dea7cf42aa43c41e9`, the exact wheel SHA-256 is
`262b6446cb200f2ae432e0e33ed35d986ad475321d59ea39bcae9cb528b9c393`, the
model configuration hash is
`8d0919c211b6aa057712a402f689f06d9ea916ba3c0c11cc32d0561aeb8d3087`, and
the normalized CUDA environment SHA-256 is
`2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`.

The first 16k execution failed before its first optimizer step because a
physical batch of 256 exceeded GPU memory. It produced no checkpoint or
scientific metric and its output root is immutable. Delegated engineering
review authorizes a fresh run under the new frozen identities above using four
ordered physical microbatches of 64 per effective batch of 256. The optimizer,
learning rate, sample order, effective batch, architecture, data, target and
stopping rules are unchanged. This is not a resume of the failed execution.

This gate permits reading the one atomic Stage A publication, resolving the
16,384-member subset by the frozen SHA-256 rank rule, fitting the 16k and 32k
probe model from scratch for seeds 0, 1 and 2, and applying the frozen paired
learning-curve decision. It does not authorize model tuning, calibration, SBC,
final evaluation, Stage B, real noise or GWOSC/GWTC. A `continue_to_train_65k`
decision is evidence for a later gate, not authorization to generate or train
the 65k rung.

All six authorized 16k/32k probe fits have completed under training commit
`5baabfe229ad187f6bcdcc1dea7cf42aa43c41e9`. The frozen 10,000-replicate paired
bootstrap measured an NLP improvement of 0.236314 nat per target dimension with
95% interval [0.223545, 0.248638]. The preregistered decision is
`continue_to_train_65k`.

This decision does not authorize Stage B. Before generating another system, a
separate gate must bind exactly 32,768 new direct-target training systems,
group-disjoint identities, the unchanged RC.4 target and an atomic extension
publication. Calibration, SBC, final evaluation, training the 65k rung,
GWOSC/GWTC and any extension beyond 65k remain closed.

Delegated expert review authorized that exact Stage B increment through
`configs/execution/phase4_direct_target_stage_b_authorization.yaml`. Its frozen
orchestration release is merged at `a198b90cc3ebd695a5b6c277e0843e0e19919b18`.
The official parent is `phase4-stage-b-2be777e727ef-6a4f106f9640`, and exactly
32,768 direct-target train systems in 256 atomic shards are now materializing.
No new validation system is authorized. The 65k optimizer remains closed until
Stage B and its combined reference publish and receive a separate gate.

While Stage B runs, the implementation-only training-stack gate permits the
fail-closed combined-publication reader, 65k three-seed launcher and terminal
32k-to-65k comparison to be completed using synthetic fixtures only. This does
not permit reading Stage B staging, resolving the 65k membership, starting an
optimizer, selecting an architecture, calibrating, opening final evaluation or
accessing GWOSC/GWTC.

The delegated implementation gate also permits the frozen post-size-lock
architecture grid, nine-new-fit runner and validation-only selector to be
implemented through
`configs/execution/phase5_architecture_selection_stack_authorization.yaml`.
This is software work only: architecture fitting and selection remain closed
until the 65k decision locks the training size and a later authorization binds
all twelve results, including reuse of the three locked-rung probe fits.

Expert audit found that the inherited preregistration fixed calibration data,
coverage criteria and SBC counts but did not machine-freeze the post-hoc map.
Before any calibration data exist, delegated scientific review therefore
freezes the downstream-only addendum `1.1.0-rc.5` at
`configs/statistics/calibration_sbc_preregistration.yaml`, canonical hash
`033b996930c93e7e4a9881fc3de49bb85cf4be96fcbd890bf2543b46368c9d8e`.
It uses split-conformal credible-region level maps fitted only on 4,096
calibration-fit systems and independent five-statistic SBC on 1,024 of 2,048
SBC systems. It does not change RC.4 data, the target or the model density.
Only pure implementation and synthetic tests are authorized through
`configs/execution/phase6_calibration_sbc_stack_authorization.yaml`;
materialization, checkpoint access, calibration fitting and SBC execution
remain closed.

The same implementation-only boundary permits a fail-closed direct-target
materialization runner and selected-checkpoint score-extraction runner through
`configs/execution/phase6_calibration_sbc_materialization_stack_authorization.yaml`.
The future data plan is exactly 4,096 calibration-fit plus 2,048 SBC systems in
48 atomic shards, configuration hash
`c55dd46d1afefe60753e2b112363261015ea914d55e80c4a5108721cb0b6a17e`.
It has no official identities and every execution flag is false. The runners
may not access a checkpoint, generate a pair, fit a calibration map or execute
SBC until training size and architecture are locked and later exact gates bind
the publications, checkpoints, code, environment and output identities.

Before final-evaluation data exist, delegated scientific review also freezes
the downstream-only analysis addendum `1.1.0-rc.6` at
`configs/statistics/final_evaluation_analysis_preregistration.yaml`, canonical
hash `7e0e252f0a972e0b0ad2fe8f93f74f1f0172639a6fb258fc7a953be5fb7973e1`.
The addendum preserves the finalized 20,480-system generator commitment and
resolves only a non-executable cross-family analysis label: the deployed model
has a lens-family condition but no fixed EPL-slope input. The executable
diagnostic therefore uses the frozen EPL training-prior-marginalized family
condition or an exact equal-density SIE/EPL mixture. Truth generation, counts,
seeds, IDs and split distributions are unchanged.

Only pure metric, counterfactual-condition and ablation-view implementation is
authorized through
`configs/execution/phase7_final_evaluation_analysis_stack_authorization.yaml`.
Final materialization/unsealing, checkpoint access, ablation training,
baseline/gold execution, calibration refitting and GWOSC/GWTC access remain
closed. The inherited matched non-neural and gold likelihood baselines require
a separately reviewed executable likelihood specification before final data
may be unsealed.

## Scientific integrity

- Never fabricate results, citations, completed runs or calibration.
- Separate completed results from plans.
- Absolute magnification must always be described as conditional on the
  adopted lens model and electromagnetic information.
- Keep signed magnification, absolute magnification, amplitude factor,
  relative magnification, flux ratio, parity and Morse index distinct.
- Never expose latent truth variables as deployable observables.
- Maintain an explicit privileged-variable denylist.
- Treat distance-magnification and mass-sheet degeneracies as core scientific
  issues.

## Remote command safety

- Use the `autodl-lensing` SSH alias.
- Never include a password in commands, prompts, files or logs.
- Do not use `rsync --delete`.
- Do not use recursive deletion on AutoDL.
- Before a command that may create more than 10 GB, estimate storage.
- Before a job expected to exceed one hour, create a configuration, manifest,
  log path and resume strategy.
- Read large NPY arrays through headers or memmap before loading them.
- Do not calculate full checksums of hundreds of GB without first estimating
  runtime and I/O cost.

## Reproducibility

Continuously maintain:

- `docs/PROJECT_STATE.md`
- `docs/DECISIONS.md`
- `docs/FAILURES.md`
- `docs/audits/`
- `docs/REMOTE_TOPOLOGY.md`
- `manifests/`
- `results/experiment_registry.csv`

All future datasets must record:
- generator Git commit;
- configuration hash;
- seed hierarchy;
- source, lens, image, pair and noise-segment IDs;
- waveform and lens-solver versions;
- detector labels;
- GWOSC product version where applicable;
- PSD and data-quality procedure;
- split assignment.

## Completion protocol

At the end of each phase:

1. run relevant checks;
2. update project state and decisions;
3. list completed, failed and deferred work;
4. inspect `git diff`;
5. commit with a descriptive message.

Do not proceed to the next phase until the current phase report exists.
