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
The official parent `phase4-stage-b-2be777e727ef-6a4f106f9640` has now
atomically published exactly 32,768 direct-target train systems in 256 shards.
The Stage B parent manifest SHA-256 is
`b4d7df6300d0919f148b98fd8ce658216bdfa64752026dc9477321874e31f0da` and
the publication-tree SHA-256 is
`373590aa01001a20e0631e672245dba050447cef0e03d45f185645476d735ee1`.

The atomic combined 65k reference is
`phase4-train-65k-2be777e727ef-6a4f106f9640`, manifest SHA-256
`753ace3d2fe475f1279b3bd8560005017f4e75a822fa951d94f9ada60eb3eca4`.
It binds exactly 65,536 unique train systems and the unchanged 6,144-system
validation publication. Independent closeout validation passed, including
q=p, exact unit weights and cross-component group disjointness. No new
validation system was generated and no GWOSC/GWTC product was accessed.

Passing Stage B does not authorize the 65k optimizer. A separate gate must
bind these manifests, the training release, model configuration, finalized
evaluation commitment and immutable CUDA environment before any Stage B array
or 65k membership may be opened for training.

Delegated expert review accepts the atomic Stage B and combined 65k
publications and authorizes only the frozen 65k three-seed probe workflow
through:

`configs/execution/phase4_probe_65k_training_authorization.yaml`

The frozen training-code commit is
`a514e101e6a09c513f03ce4c4633459498e77457`, the exact wheel SHA-256 is
`cde98326596b8d2a49287a51ff141ce37c0c9c7cf022798b459fa7c53b816bdc`, the
model configuration hash remains
`8d0919c211b6aa057712a402f689f06d9ea916ba3c0c11cc32d0561aeb8d3087`, and
the immutable CUDA environment SHA-256 remains
`2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`.

This gate permits reading only the atomic Stage A, Stage B and combined 65k
publications, fitting the frozen 65k probe from scratch for seeds 0, 1 and 2,
and applying the terminal preregistered 32k-to-65k comparison. It does not
authorize model tuning, architecture selection, calibration, SBC, final
evaluation, any extension above 65,536 systems, real noise or GWOSC/GWTC. A
`lock_train_65k` decision is evidence for a later architecture-selection gate;
a `stop_data_limited_and_new_preregistration` decision requires stopping for a
new scientific contract.

The implementation-only training-stack gate completed the fail-closed
combined-publication reader, 65k three-seed launcher and terminal 32k-to-65k
comparison using synthetic fixtures only. This completion does not permit
resolving the 65k membership, starting an optimizer, selecting an architecture,
calibrating, opening final evaluation or accessing GWOSC/GWTC.

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

That review found the inherited DINGO-style likelihood-gold obligation
mathematically incompatible with the selected estimator: the NSF defines only
a two-magnification marginal and has no normalized proposal density over the
complete source/lens/dynamics nuisance space required by the simulator
likelihood. Before any final data exist, delegated review therefore freezes
the downstream-only RC.7 correction at
`configs/statistics/reference_baseline_preregistration.yaml`, canonical hash
`1df98c89fc418eddfd9ec766cb04311e0f3d9f40836a0d9ba1dd691d6bc1724e`.

RC.7 forbids likelihood-correction and importance-sampling-efficiency claims.
It replaces the non-executable gold label with a frozen, non-neural
same-family/same-EM-cell selected-prior kNN/KDE simulation reference, explicitly
not an exact likelihood or gold posterior. Pure implementation is authorized
only through
`configs/execution/phase7_reference_baseline_stack_authorization.yaml`.
Scientific reference-bank access, validation/final baseline execution, final
unsealing, checkpoint access and GWOSC/GWTC remain closed. SBC, IID and tail
coverage remain the primary approximation/calibration evidence.

The implementation-only boundary also permits a fail-closed final inference
and score-artifact stack through
`configs/execution/phase7_final_inference_stack_authorization.yaml`. It may
resolve synthetic sealed manifests, exercise synthetic checkpoint inference,
apply synthetic calibration maps and test the frozen cross-family semantics.
It may not materialize or unseal final data, access a scientific checkpoint or
calibration map, create a scientific score artifact, refit calibration, train
or tune a model, finalize a manuscript claim or access GWOSC/GWTC. A later
exact gate must bind the sealed parent, all three retained checkpoints,
same-seed calibration/SBC outputs, the selected architecture, immutable
inference environment and every output identity before one final record may be
opened.

The first authorized 65k probe launch under training commit
`a514e101e6a09c513f03ce4c4633459498e77457` stopped before its first optimizer
step. The training whitener exposed a finite but catastrophically large
IMRPhenomXPHM source-polarization bin in the published train data. An exhaustive
read-only regeneration audit of all 71,680 Stage A/Stage B records found exactly
five numerical pathologies: two in Stage A train, three in Stage B train and none
in validation. The largest nonpathological isolated-bin ratio was 1.704718; the
five failures ranged from 12,983 to 2.562e24. No calibration, SBC or final result
was inspected.

Because one affected Stage A system entered the frozen 16k subset and both
entered the 32k rung, the existing 16k/32k learning-curve evidence is superseded.
The failed 65k output has no checkpoint or valid scientific metric and may not be
resumed. The original Stage A, Stage B and combined 65k publications remain
immutable.

Delegated scientific review freezes the numerical correction preregistration
`1.1.1-rc.1` at
`configs/statistics/waveform_numerical_validity_preregistration.yaml`, canonical
hash `7fca209de9f06e98da1c5a96ae0f4fc6daec5d2f0c2339a718e1f899bb915b69`.
It rejects an attempted source waveform before lensing and selection when either
polarization's maximum strictly positive in-band amplitude divided by its
linear 99.9th percentile exceeds 10. Clipping, waveform repair and parameter
substitution are forbidden. The estimand, evaluation target, q=p unit weights,
selection, waveform approximant, PSD, counts, model and learning-curve rule are
unchanged.

Delegated expert review accepts implementation commit
`499f86b3159af82612e38c134cd81003eedcc4e4`, exact wheel SHA-256
`1088b2be49e879cbc44fc834b09c67947b45f2da444e15a3f41856abf60729f2`,
the real-record regression and immutable base hashes. Exact replacement
materialization and corrected-view publication are now authorized through
`configs/execution/phase4_waveform_numerical_correction_authorization.yaml`.
The correction must exclude exactly two Stage A plus three
Stage B systems, publish exactly two plus three fresh direct-target replacements,
retain all original publications read-only, keep validation at 6,144, and restore
exact corrected counts of 32,768/32,768/65,536. Training, architecture selection,
calibration, SBC, final evaluation, extension above 65k, real noise and
GWOSC/GWTC remain closed.

The authorized correction completed and atomically published under parent
`phase4-waveform-correction-499f86b3159a-1db109b08189`. It restored exact
corrected counts of 32,768 Stage A train, 32,768 Stage B train, 65,536 combined
train and 6,144 unchanged validation systems. The parent manifest SHA-256 is
`0fcfb117c620d58a2e0ccd8b19c0d3f3a371dd844fb637b50c8b565eee6864f2`; the
publication-tree SHA-256 is
`a57aa2691e256b34403392f595e964dceec1325cfc54a38ed4d2a0b714d38c12`.
Independent closeout passed source-spectrum, Zarr/Parquet, exact decomposition,
unit-weight, original-hash and grouped-ID validation. The original
publications were not modified.

Passing the correction does not authorize training. The old 16k/32k metrics
remain superseded. A separate gate must bind the correction manifest, a
corrected-view reader, recomputed 16k membership, immutable training wheel and
CUDA environment before any corrected train array or optimizer may be opened.

The corrected-view training implementation is complete on branch
`phase4/corrected-probe-rerun`. It provides a metadata-only overlay resolver,
lazy 32k/65k readers, fresh 16k membership resolution and fail-closed runners
for the 16k/32k rerun and a conditional 65k rerun. This implementation status
does not authorize reading scientific Parquet/Zarr, resolving membership,
fitting standardizers, creating checkpoints or starting an optimizer. A later
authorization must bind the implementation commit, exact wheel, correction
view hashes and immutable CUDA environment.

Delegated expert review now authorizes only the fresh corrected 16k/32k
three-seed probe workflow through:

`configs/execution/phase4_corrected_probe_training_authorization.yaml`

The frozen training implementation is
`adcb1a79e1534e4d742238aa99869c57da95dd96`, wheel SHA-256
`44208c61577b71488872c75eced03dbca3384cf5d03baaecc9f3447bdaeef24a`.
The corrected Stage A training-view SHA-256 is
`b9390a7faad4bb8097abb09041f1f229e13c6419677926f05642ac611dc6ced2`.
This gate permits recomputing the 16k SHA-ranked membership and fitting the
unchanged 16k/32k probe for seeds 0, 1 and 2 from scratch. Superseded
checkpoints and metrics may not be resumed or reused. Architecture selection,
calibration, SBC, final evaluation, corrected 65k training, extension above
65k, real noise and GWOSC/GWTC remain closed.

All six fresh corrected 16k/32k fits completed under training commit
`adcb1a79e1534e4d742238aa99869c57da95dd96`. The independently reproduced
10,000-replicate paired bootstrap measured a development NLP improvement of
0.211849 nat per target dimension with 95% interval [0.200116, 0.223464]. The
frozen decision is `continue_to_train_65k`; its exact JSON SHA-256 is
`fe2890e025f5574a4ea45942b698e0b24db3801650125cd5f128126e435633cf`.

This decision does not authorize the corrected 65k optimizer. A separate gate
must bind the decision hash, corrected combined-view hash, immutable training
wheel and CUDA environment before any corrected 65k array is opened. The old
65k output and all superseded checkpoints remain excluded. Architecture
selection, calibration, SBC, final evaluation, extension above 65,536,
real-noise work and GWOSC/GWTC remain closed.

Delegated expert review accepts that exact entry evidence and authorizes only
the fresh corrected 65k three-seed probe through:

`configs/execution/phase4_corrected_65k_probe_training_authorization.yaml`

The gate binds corrected combined-view SHA-256
`da8aaa8d86afb4d93156191976b420bfc7bbc7dfe0fdc6c6f627515d804a7379`,
decision SHA-256 `fe2890e025f5574a4ea45942b698e0b24db3801650125cd5f128126e435633cf`,
code `adcb1a79e1534e4d742238aa99869c57da95dd96`, exact wheel and the unchanged
CUDA environment. It permits seeds 0, 1 and 2 from scratch and the terminal
32k-to-65k development comparison only. Architecture selection, calibration,
SBC, final evaluation, extension above 65,536, real noise and GWOSC/GWTC remain
closed.

All three corrected 65k fits completed with zero launcher failures. The frozen
10,000-replicate paired comparison measured an NLP improvement of 0.201437 nat
per target dimension with 95% interval [0.191498, 0.211788]. The terminal
decision is:

`stop_data_limited_and_new_preregistration`

Its exact JSON SHA-256 is
`90c238a0d85d941c9e90a68e8a215a8d9025f57ffe7757ff89dd14c267f6d72f`;
an independent replay was byte-identical. Calibration and final evaluation
were not accessed.

This is a successful execution with a major scientific stopping result, not a
training failure. RC.4 forbids automatic extension above 65,536. Architecture
selection cannot start because its gate requires `lock_train_65k`. Do not
reinterpret the completed 65k rung as locked, change the saturation rule after
seeing the data, generate a larger rung, fit an architecture, open calibration,
SBC or final evaluation, or access GWOSC/GWTC. A new versioned scientific
preregistration and explicit human review are required before any of those
actions.

The existing Phase 5 implementation-only gate supports the same immutable
correction overlay. The corrected implementation is frozen at
`d7e87a84ffc69a0e7825eb448c8cfdabe4e7fd4d`; its wheel SHA-256 is
`fcc766a43a61ffdda3e0fca83fbefff4a010c5d35d39ab27b637fb34dbf5490a`.
This checkpoint has not opened a scientific array or fit an architecture. A
future scientific contract may reuse this software, but the present terminal
decision cannot satisfy its `lock_train_65k` execution precondition.

The same implementation-only downstream gates now carry the already frozen
`1.1.1-rc.1` waveform numerical-validity rule prospectively into every future
IMRPhenomXPHM calibration/SBC and final-evaluation generator namespace. The
original Phase 6 configuration and finalized final-evaluation commitment remain
byte-identical. Separate hashed addenda bind the correction without changing
counts, seeds, diagnostic distributions or authorization flags. The deliberate
SEOBNRv4PHM waveform-mismatch namespace does not inherit an IMRPhenomXPHM-
specific ratio threshold; it remains subject to its frozen finite-array and
waveform-boundary validation. No pair, checkpoint, calibration statistic or
final case was opened by this prospective software hardening.

The final-evaluation release path also treats the original planned generator
commit as historical commitment evidence, not executable post-incident code. A
future exact materialization gate must bind a new immutable generator commit,
the unchanged original commitment, the numerical-validity addendum, correction
publication and logical corrected 65k train reference. The release certificate
derives official identities only after those checks pass. It filters the five
excluded base systems and includes their five replacements when proving group
disjointness. This resolves software executability only; final materialization
and unsealing remain closed.

The future calibration/SBC materialization gate likewise resolves the exact
corrected training reference before deriving any official identity. It binds
the Stage A, Stage B, combined-base and correction parents; excludes the two
Stage A plus three Stage B pathological base systems; includes all five
replacement systems; and requires exactly 65,536 logical train plus 6,144
unchanged validation systems. This is implementation-only hardening. No
calibration/SBC identity, pair, checkpoint statistic or execution gate exists.

The Phase 7 non-neural reference implementation is now hardened for future
execution without opening a scientific record. The typed published reader
retains the exact Parquet EM-cell label on metadata-only examples, and the
reference core builds a deterministic vectorized index by exact lens family and
EM cell using the selected training-rung standardizer. It scores CRPS, KDE NLP,
central marginal/joint coverage and interval width one query at a time without
persisting posterior draws. Scientific reference-bank access, validation/final
query access and reference execution remain closed until a later exact gate
binds the locked training rung, selected preprocessing artifact and query
publication.

The implementation-only boundary now also includes a fail-closed runner for
the two RC.6 input ablations through:

`configs/execution/phase7_ablation_training_stack_authorization.yaml`

It applies the GW-only or EM-only view after the primary locked-rung
standardizer, preserves the selected architecture, optimizer, effective batch,
budget and all targets, and limits future execution to exactly two views by
seeds 0, 1 and 2. The future six-fit gate must bind the terminal 65k lock,
selected-architecture decision, corrected publication, primary standardizers,
training wheel, CUDA environment and finalized evaluation commitment. This
implementation checkpoint may use synthetic fixtures only. It does not permit
opening scientific arrays or checkpoints, starting an optimizer, selecting an
architecture, accessing calibration/SBC/final data or using GWOSC/GWTC.

The implementation-only boundary also contains the future RC.7 reference-query
runner through:

`configs/execution/phase7_reference_execution_stack_authorization.yaml`

It binds a future terminal size lock, selected architecture, corrected logical
training publication, selected-rung standardizer, one exact query publication,
immutable software/environment and output identity. It streams metadata-only
queries to deterministic per-case and aggregate score artifacts with raw
coverage counts and Wilson intervals, without persisting posterior draws or
opening GW strain. Current authorization permits synthetic fixtures only;
scientific bank access, validation/final query execution, checkpoint access,
final unsealing and GWOSC/GWTC remain closed.

The inherited legacy SIS point regressor is implemented only as a read-only
descriptive stress-control verifier through:

`configs/execution/phase7_legacy_sis_stress_control_authorization.yaml`

It may hash the opaque legacy checkpoint and recompute the saved 500-row
model-selected validation point metrics under a later exact read-only gate. It
must never deserialize or write the legacy checkpoint, open v2 final data, or
be called a posterior, calibrated result, independent test or matched H1/L1/V1
competitor. Current authorization permits synthetic fixtures only.

Human review accepted the corrected 65k data-limited result and explicitly
approved the prospective terminal-scale preregistration `1.2.0-rc.1` at:

`configs/statistics/terminal_131k_preregistration.yaml`

Its canonical hash is:

`77ff5b6b45b886657e90023c50ae002afffb077db594c80665166d537fd2346a`

The contract retains the corrected 65,536-system publication as an immutable
strict subset, freezes exactly 65,536 additional direct-target training
systems and caps the terminal training rung at 131,072. It also freezes a
separate 512-system development-only tail pool with exactly 128 systems in
each of the four inherited priority strata. That pool must be group-disjoint
from all train, validation, calibration, SBC, final and engineering artifacts
and may never be used for training, architecture selection, calibration or a
final claim.

The future 65k-to-131k comparison uses the unchanged 6,144 core validation
systems for its paired NLP decision. Raw coverage and EM-cell coverage remain
mandatory development reports but are nonblocking for the terminal resource-
cap lock; calibrated claims remain owned by the separately frozen calibration
and SBC gates. The exact terminal outcome must be either
`lock_train_131k_saturated` or
`lock_train_131k_resource_capped_data_limited`. Both stop at 131,072 and may
only support a later separately authorized architecture review. No extension
above 131,072 is authorized.

The current authorization is design-only through:

`configs/execution/phase4_terminal_131k_preregistration_authorization.yaml`

It authorizes no pair generation, tail materialization, optimizer,
architecture fit, calibration, SBC, final evaluation, real noise or
GWOSC/GWTC access. A separate exact release and resource gate is required
before any new identity or output may be created.

The terminal materialization implementation is frozen at
`a4e6bac014ccd521d510c97593cb1368e826d5eb`, exact wheel SHA-256
`c7bc8ecadb373ed5d7307ee9c96b131cc68cc9ad8ea10ae2100c54aed2a8958f`.
Local verification passed 368 tests with seven optional skips; the exact wheel
passed 378 AutoDL tests with one optional Torch skip. The measured AutoDL free
space was 221,613,056,000 bytes, above the conservative 201,596,510,484-byte
prelaunch gate.

Delegated expert review now authorizes exact materialization only through:

`configs/execution/phase4_terminal_131k_execution_authorization.yaml`

This gate permits exactly 65,536 new direct-target train systems in 512 atomic
shards and exactly 512 development-tail systems, 128 in each of four priority
strata. It permits a logical atomic 131,072-system training reference after
group-disjoint validation. It does not authorize the 131k optimizer,
architecture selection, calibration, SBC, final evaluation, extension above
131,072, real noise or GWOSC/GWTC. Official identities may be derived only by
a fresh ready release certificate, and publication must stop for review.

The first official terminal-materialization segment started with the frozen
configuration's 16-worker process pool. At `2026-07-20T01:21:24Z` it was
stopped before any atomic shard completed in order to apply the owner's
engineering-only concurrency request. Exactly 16 partial shards totaling
631,856,490 bytes were moved by same-filesystem rename into immutable
interruption evidence; they may not be reused, deleted or counted.

Delegated engineering review authorizes a scheduler-only restart at exactly 32
workers through:

`configs/execution/phase4_terminal_131k_worker32_authorization.yaml`

The orchestration implementation is frozen at
`8977ca55f13963441afdda831afb190a3872517c`. The scientific generator commit,
wheel, configuration hash, preregistration, root seeds, counts and official
identities remain unchanged. A fresh release certificate must bind the 32-
worker scheduler before execution. Sixty-four workers are explicitly not
authorized. Training, architecture selection, calibration, SBC, final
evaluation, extension above 131,072, real noise and GWOSC/GWTC remain closed.

The worker-32 release certificate passed with no blockers and
220,975,267,840 free bytes. The official segment started at
`2026-07-20T01:32:59.843129+00:00` as AutoDL PID `2515891`, with exactly 32
worker children and the unchanged official identities. This is active staging,
not a completed or published result.

While that immutable materialization runs, delegated engineering review
authorizes only a synthetic-fixture implementation of the terminal probe stack
through:

`configs/execution/phase4_terminal_131k_probe_stack_authorization.yaml`

The software may validate the future atomic 131k publication, build a bounded-
memory corrected-65k-plus-increment reader, evaluate the three retained 65k
checkpoints on the new 512-case development-tail pool, launch the frozen 131k
probe for seeds 0/1/2 and apply the preregistered terminal comparison. It may not
open any active staging or scientific publication, access a checkpoint, start an
optimizer, execute a scientific decision, fit an architecture, calibrate, unseal
final evaluation or access GWOSC/GWTC. A later exact gate must bind the completed
publication manifests, training commit/wheel, model configuration and CUDA
environment before any terminal scientific array may be opened.

The implementation checkpoint is
`77257c3d4871937883eebd330fb8496246a85ff4`; its exact wheel SHA-256 is
`dea53afc08609789ea6c1ac066ed411bf1aad135434cf33f86b5e46e3f92e0ad`.
This identity is implementation evidence only and does not activate the future
scientific gate.

The same synthetic-only boundary permits the terminal post-lock architecture
adapter through:

`configs/execution/phase5_terminal_131k_architecture_stack_authorization.yaml`

It may accept either frozen 131k lock label, validate reuse of the three 131k
probe fits and implement the nine remaining grid fits plus validation-only
selection. It may not open scientific data/checkpoints, start a fit, select an
architecture, calibrate, unseal final evaluation or access GWOSC/GWTC. The
historical 65k architecture path remains unchanged.

The terminal architecture implementation checkpoint is
`2689119d0526c82f8145c0424741e56a048e96df`; its exact wheel SHA-256 is
`fc10efa29ba129f19ab3874d88a6cab4c0840045fa1ef8b5b102ca91f8c9231f`.
It remains implementation evidence, not an execution identity.

The synthetic-only implementation boundary now also includes the shared
terminal downstream binding stack through:

`configs/execution/phase5_terminal_downstream_stack_authorization.yaml`

It accepts only the two frozen 131k terminal labels and the exact twelve-result
development-only architecture lock. Historical score-inference gates without
an explicit locked-rung field remain limited to 32k/65k checkpoints; a 131k
checkpoint requires a later authorization that explicitly binds
`locked_training_rung: 131072`. The stack validates only small decision and
manifest identities. It does not open active staging, scientific data,
checkpoints, calibration/SBC, final evaluation, ablation/reference execution,
real noise or GWOSC/GWTC.

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
