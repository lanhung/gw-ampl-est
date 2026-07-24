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
worker children and the unchanged official identities. That launch record is
retained as historical execution evidence.

The 32-worker segment completed and atomically published all 65,536 new train
systems in 512 shards. Its original development-tail layout then exposed a
pure execution-partition resource mismatch: the first high-magnification
one-by-128 shard accepted 15 of 91,839 attempts in 3.29 active hours. Its
95%-upper-bound projection still had only a 1.2e-7 probability of reaching 128
before the frozen 12-hour worker cap. Delegated engineering review therefore
stopped the segment at `2026-07-21T20:02:00Z`, before the inevitable cap, and
retained the zero-complete/one-partial tail tree as immutable non-result
evidence. The atomic train publication was not modified.

The same scientific tail contract is now authorized for an engineering-only
parallel recovery through:

`configs/execution/phase4_terminal_131k_tail_parallel_recovery_authorization.yaml`

It preserves the generator, preregistration, distributions, four strata,
128 cases per stratum, root seed, q=p unit weights and 131,072 terminal cap.
Only the storage/execution partition changes from one 128-case shard to 32
four-case shards per stratum so the existing 32 physical workers can be used.
The recovery requires new tail and combined identities, excludes the stopped
partial evidence and may reuse only the read-only published train increment.
Sixty-four workers, training and every downstream scientific gate remain
closed.

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

The terminal downstream implementation checkpoint is
`cfb3e92f6600975c81e7dfdc58237ebf82acce7c`; its exact wheel SHA-256 is
`35909951c13cffbe695fe4af631d282fd58634e4f80156057a8cd107609c2b4a`.
These are implementation identities only and do not activate a future exact
scientific gate.

The implementation-only boundary also includes terminal reference adapters for
future calibration/SBC and sealed final materialization through:

`configs/execution/phase6_terminal_materialization_adapter_authorization.yaml`

The adapters preserve historical corrected-65k replay while requiring terminal
mode to bind exact combined/increment/validation/development-tail hashes, a
131,072-system lock and the twelve-result architecture decision. The future
calibration/SBC leakage reference must include terminal train, validation and
the separately nontraining 512-case development-tail pool. The final pool stays
sealed and unchanged at 20,480 cases. This software authorizes no materialization,
unsealing, checkpoint access, statistic, fit, real noise or GWOSC/GWTC access.

The terminal materialization-adapter implementation checkpoint is
`45d05287fbd9a8b7f9bc1999b749be5c521d7931`; its exact wheel SHA-256 is
`bc5d3cd2fd6f898b08590be7f348dc4970edb7fe5f23f4422ffc29185336f4cd`.
It is not an execution identity.

The implementation-only terminal analysis adapters are authorized through:

`configs/execution/phase7_terminal_analysis_adapter_authorization.yaml`

They allow future ablation and RC.7 reference runners to use the logical
131,072-system training view only after an exact gate. The two ablations retain
the selected architecture and seeds 0/1/2; the reference bank retains exact
lens-family/EM-cell grouping and 256-neighbor semantics. The 512 development-
tail cases enter neither training nor the reference bank. No scientific array,
checkpoint, optimizer, query, final record, real noise or GWOSC/GWTC access is
authorized by this implementation gate.

The terminal analysis-adapter implementation checkpoint is
`c5cd67d0537dad81797d2a77913a5f3bbd142f00`; its exact wheel SHA-256 is
`0ae3da4bbb96312b1347babe03ed95cfa45950966c12959e921e78abf7981fd7`.
It is implementation evidence only and does not authorize execution.

The active terminal runner already continues automatically from the 512-shard
train increment through all four development-tail namespaces, cross-component
validation and atomic combined publication. A separate read-only closeout
command is implemented at `scripts/phase4/closeout_terminal_131k.py`. It binds
the exact worker-32 result and, by default, independently recomputes both large
publication-tree hashes and the post-publication free-space gate. Its software
implementation does not authorize the terminal optimizer or any downstream
scientific execution.

The post-publication handoff also has a non-authorizing release-review packet
at `scripts/phase4/prepare_terminal_probe_release.py`. It requires the
independent closeout, exact training commit/wheel and AutoDL wheel-test result,
the normalized CUDA environment, at least three frozen-model GPUs, the probe
model configuration and final-evaluation commitment. Its only successful
status is readiness for delegated authorization review; it may not create the
authorization or open a publication.

The terminal probe release-packet implementation checkpoint is
`099c5762be9c72f7ded420c64f456db885ec37e5`; its exact wheel SHA-256 is
`93b541c30e5df571bbbc5b07bef423665814e510a13d0d0595e2a3de2d0e83d7`.
This is candidate software evidence only until the exact wheel passes the
post-publication AutoDL test contract.

The exact-wheel test contract now has a fail-closed verifier at
`scripts/phase4/verify_terminal_training_wheel.py`. It requires the installed
archive hash to equal the candidate wheel, rejects editable and repository
`src` imports, verifies CUDA and the frozen GPU model, and runs both focused
and full tests with the repository pytest configuration disabled so its
`src` path cannot shadow the installed distribution. The pytest commands also
use `--noconftest`; this is required because `tests/conftest.py` otherwise
inserts the repository `src` path independently of pytest configuration. The
verifier may run
only as post-publication software evidence; it opens no scientific data and
cannot authorize or start an optimizer.

The future terminal-probe execution gate must additionally bind the SHA-256
of that separately reviewed release packet. The typed training gate rejects
packet, publication, wheel, environment, model, GPU, commitment or safety-flag
drift before resolving a scientific dataset. The accepted packet hash is
carried into shared preprocessing, every 131k run identity summary and the
retained-65k development-tail evaluation. No terminal-probe authorization or
optimizer exists yet.

The release packet must also hash-bind all three retained corrected-65k
`best.ckpt` and `run_summary.json` artifacts, plus their shared training,
manifest, standardizer, model, environment and final-evaluation-commitment
identity. The runtime verifies these hashes before loading a retained
checkpoint and requires its embedded identity to equal the bound summary.
Binding only the retained output directory is insufficient.

A separate authorization builder is implemented at
`scripts/phase4/authorize_terminal_probe.py`. It cannot promote a release
packet by itself: it requires a second JSON review decision whose packet hash,
131k rung, seeds, retained-65k input, new output identity and exact closed
boundary set all match. The generated YAML is revalidated through the actual
runtime packet-binding function before atomic write. The builder has not been
executed on scientific evidence and no terminal-probe authorization exists.

Terminal closeout, release-packet and delegated-review evidence use
repository-relative paths below `results/phase4/`. Host-absolute packet paths
and parent traversal fail closed, so the same committed evidence is resolved
under the authoritative Vultr root and disposable AutoDL root without editing
the authorization. This software portability fix authorizes no publication
access, membership resolution or optimizer.

Release-packet assembly may run from a clean descendant of the frozen training
commit only when every intervening path is an exact registered closeout or
project-state evidence file. The packet records that review-checkout commit;
any post-freeze source, model, configuration or unregistered-file change fails
closed. This resolves the necessary publish-then-review ordering without
changing the exact wheel used for training.

The worker-32 fixed-four-case tail recovery stopped fail-closed before its
resource cap. The first stratum completed exactly 128 cases, but the
extreme-relative-magnification stratum produced only six partial cases from
328,134 attempts. A read-only audit of the already published 65,536-system
train increment measured 347 such cases in 24,636,731 proposal attempts. Even
using the one-sided 95% optimistic rate bound, the probability that all 32
fixed shards would independently reach four cases by their worker caps was
`1.4141494186098862e-26`.

The failed parent is immutable at:

`/root/autodl-tmp/lensing-4/data_v2/scientific/terminal_131k/interrupted_evidence/tail-parallel32-resource-stop-20260722T0002Z`

Its tree SHA-256 is
`2866e66739aa26f70e560bc8bacb196baccc2406acbcb64719d5b4a2338a253a`.
It contains 32 complete first-stratum shards and 32 second-stratum partial
shards. None may be resumed, reused, published or interpreted scientifically.

A dynamic microshard recovery is authorized through:

`configs/execution/phase4_terminal_131k_tail_microshard_recovery_authorization.yaml`

It preserves the four direct-target conditional strata, 128 accepted cases
per stratum, root seeds, target, unit weights and terminal 131,072-system cap.
It changes only the physical layout to 128 atomic one-case shards per stratum,
dynamically scheduled over 32 physical workers. The frozen orchestration commit
is `adb4c0981fd15a809005212c76dd972a59822489`, its wheel SHA-256 is
`a5b08e40ddcff7d542a68b195d5bfc52577e2a67a8a978e374e1d7581f1e4b52`,
and its generator-core manifest SHA-256 is
`ebb900d52719dd570e378b63a6d2178b5b47a4b4ed6326769fa55e486b6ebda5`.
The exact wheel is installed non-editably in a distinct AutoDL runtime.

Execution is authorized for exactly 512 development-only tail cases using 32
workers. Sixty-four workers remain closed: the host has 64 logical CPUs but
only 32 physical cores, about 61 GB available memory at authorization, and
only about 140 GB free disk against a 125 GB prelaunch floor. Do not alter the
worker count, counts, identities, seeds or scientific distributions inside the
official run. Stop after atomic tail and combined-131k publication.

The dynamic microshard recovery completed and atomically published exactly 512
development-only cases: four namespaces of 128 one-case shards. The immutable
tail parent is:

`phase4-terminal-tail-micro128-adb4c0981fd1-30fa02d9ec5b`

Its manifest SHA-256 is
`58fcafd58cbcd407ecf6b35dfa98c0bd2bd66f37151e19e6bf530ca2601260c7`
and its independently recomputed publication-tree SHA-256 is
`90ca582f3bd768046f9ceabb4d42689d76945be2c963b0290ac432662ff619c0`.

The atomic combined terminal training reference is:

`phase4-train-131k-adb4c0981fd1-30fa02d9ec5b`

It binds exactly 131,072 unique direct-target train systems, the unchanged
6,144-system validation publication and the separate 512-case development
tail. Its manifest SHA-256 is
`ad26d51d4f9475c6710cdfee4e71409526e1d776e0b8ec14734feff02855cee5`.
Independent closeout passed with q=p, exact unit weights, no reuse of failed
tail evidence, no GWOSC/GWTC access and 253,429,231,616 free bytes remaining.

Passing closeout does not authorize the terminal optimizer. A separate exact
release packet and delegated authorization must bind this closeout, the
immutable training wheel, CUDA environment, model configuration, finalized
evaluation commitment and retained corrected-65k checkpoint hashes before the
131k membership or any scientific checkpoint may be opened.

Training, architecture selection, calibration, SBC, final evaluation,
extension above 131,072, real noise and GWOSC/GWTC remain closed until their
separate machine-readable gates pass.

The first authorized reader stopped before preprocessing or its first optimizer
step because the immutable terminal train parent used its real singular
`validation` manifest field while the reader accepted only the older
`validations` mapping. It created no preparation, checkpoint or scientific
metric. This was a schema-accessor integration defect, not a data, model or
scientific-contract failure.

The narrow compatibility correction is frozen at
`d8a3f1153155797921267557672c03d1ea6543a9`. Its exact terminal training wheel
passed the post-publication AutoDL contract: 70 focused tests passed and 486
full tests passed with three optional skips. A read-only regression indexed all
65,536 terminal increment IDs from the real singular parent manifest. The
installed module came from the non-editable wheel in the isolated CUDA
environment; all four observed GPUs were NVIDIA RTX 5000 Ada Generation
devices. No strain was opened and no optimizer was started.

Delegated scientific and engineering review accepts release packet
`results/phase4/terminal_probe_release_packet.json`, SHA-256
`d2e4fde7b918ce363ca67781d7a462d97ffe37dd4fadde186f587b44be7cdf7a`,
and authorizes only the frozen 131,072-system three-seed terminal probe through:

`configs/execution/phase4_terminal_131k_probe_authorization.yaml`

The frozen training software commit is
`d8a3f1153155797921267557672c03d1ea6543a9`, the exact wheel SHA-256 is
`fd8da0465f9609e31805abf01f1bf41dc07b486b8e470a6c345a64923b63dda8`,
the model configuration hash is
`8d0919c211b6aa057712a402f689f06d9ea916ba3c0c11cc32d0561aeb8d3087`,
and the CUDA environment hash is
`2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`.

This gate permits reading only the exact corrected 65k base, terminal
increment, atomic combined 131k reference, unchanged 6,144-system validation
set and separate 512-case development-tail pool. It permits fitting the frozen
probe from scratch for seeds 0, 1 and 2, evaluating the retained 65k
checkpoints on the new development-tail pool and applying the preregistered
terminal comparison.

Model tuning, architecture selection, calibration, SBC, final evaluation,
extension above 131,072, real noise and GWOSC/GWTC remain closed. Either
permitted terminal decision locks the training resource cap at 131,072 and
requires a later architecture-selection authorization.

While the terminal probe runs, the implementation-only Phase 5 boundary also
permits exact post-lock release-control software. Implementation commit
`4ef6626eef201aeb91a74f5e9d799ec410459c6a` adds a non-authorizing release
packet, separate delegated-review authorization builder and mandatory SHA-256
binding of all three reused probe summaries and `best.ckpt` files. It passed
489 full tests with seven optional-dependency skips, 26 focused tests, Ruff and
mypy.

This implementation does not authorize architecture data or checkpoint access,
the nine new fits or architecture selection. A future release packet must bind
one completed terminal decision, a separately verified exact wheel and fresh
output identity. Calibration, SBC, final evaluation, extension above 131,072,
real noise and GWOSC/GWTC remain closed.

The Phase 6 implementation-only boundary now also contains the exact
post-materialization score and statistics release chain. It freezes six score
artifacts (calibration-fit and SBC for each retained model seed), requires one
hash-bound delegated review before checkpoint inference, and requires a second
review before three seedwise calibration/SBC analyses. Calibration maps are
fitted separately per seed; no best seed or seed pooling is permitted.

This software has used synthetic score fixtures only. It does not authorize
checkpoint access, calibration/SBC publication access, score extraction,
calibration fitting, SBC execution, final evaluation or GWOSC/GWTC access.
Those gates require the terminal architecture decision, exact selected
checkpoint hashes, the atomic 4,096+2,048 publication, an exact inference
wheel/environment and fresh output identities.

The Phase 7 implementation-only boundary now also contains the exact sealed
final-materialization release chain. It represents each Stage A, Stage B,
correction, terminal, calibration and SBC dataset child by its own directory
while hash-binding that child to the atomic common-parent manifest. This
matches the real publication layout; child directories do not fabricate
duplicate `dataset_manifest.json` files.

The future release packet binds the terminal size and architecture decisions,
the exact generator wheel and environment, all eight child-dataset references,
the five immutable correction exclusions and the 20,480-case/160-shard sealed
contract. A separate delegated review is required before the
materialization-only YAML can be created. No official final identity is
derived by the packet or review builder.

This implementation has used synthetic directory fixtures only. Final
materialization, unsealing, checkpoint or calibration-map access, scientific
inference, model training and GWOSC/GWTC remain closed.

The Phase 7 implementation-only boundary now also contains an exact final
inference release chain through:

`configs/execution/phase7_final_inference_release_stack_authorization.yaml`

Implementation commit `7373d2d456a3cb73300392a2a2fa604380d6b77b` binds a
future sealed 20,480-case/15-namespace publication, the terminal 131k
architecture decision, all three selected checkpoint hashes and the three
same-seed calibration/SBC result bundles. It allocates exactly 45 fresh score
artifacts (15 namespaces times seeds 0/1/2) without selecting a best seed or
persisting posterior draws.

The release packet is non-authorizing. A separate hash-bound delegated review
must approve the exact unsealing, checkpoint inference, same-seed calibration
map application and immutable score paths before the runtime authorization may
be created. Model training/tuning, calibration refitting, architecture or size
selection, result-threshold changes, ablation/reference execution and
GWOSC/GWTC remain closed. Only synthetic fixtures were used; no final record,
scientific checkpoint or calibration map was opened.

The implementation-only Phase 7 boundary now also contains the exact terminal
input-ablation release chain through:

`configs/execution/phase7_ablation_release_stack_authorization.yaml`

Implementation commit `93ca300366cdc00b8d407fc18badd312d4946844` creates a
non-authorizing packet that binds the future 131k terminal decision, the
twelve-result architecture lock, the primary 131k membership and
standardizers, immutable training wheel/environment and exactly six fresh
outputs: GW-only and EM-only views for seeds 0, 1 and 2. A separate
hash-bound delegated review is required before the runtime authorization can
open data or start one fit.

This implementation used synthetic fixtures only. Terminal probe completion,
architecture selection, all six ablation fits, calibration/SBC, final
evaluation and GWOSC/GWTC remain separately closed.

The implementation-only Phase 7 boundary now also contains the exact RC.7
reference-query release chain through:

`configs/execution/phase7_reference_execution_stack_authorization.yaml`

Implementation commit `beaa41be7827990edc57bc4de5253a7e7ed298ea`
corrects the future runner to bind query children to their atomic parent
manifest instead of requiring a nonexistent child manifest. Validation and
IID each bind one child; the 4,096-case balanced-tail query binds four
independent 1,024-case children and presents them as one bounded-memory
metadata-only dataset.

The same implementation adds a non-authorizing role-specific release packet
and a separate hash-bound delegated-review authorization builder. A validation
release cannot unseal final data, while IID and balanced-tail releases require
their explicit final-unsealing scope. Scientific reference-bank access,
validation/final query execution, checkpoint access and GWOSC/GWTC remain
closed. Only synthetic directory fixtures were used; no scientific record,
checkpoint or final case was opened.

The implementation-only Phase 7 boundary now also contains an exact read-only
release chain for the frozen legacy SIS descriptive stress control through:

`configs/execution/phase7_legacy_sis_stress_control_authorization.yaml`

Implementation commit `be669fab86ce5f251c56732da6187c3f633e8e8b`
adds a non-authorizing release packet, a separately hash-bound delegated-review
authorization builder and one typed runtime gate. The future execution may
hash, stat and read the frozen prediction CSV, but it may never deserialize the
legacy checkpoint or write below a legacy root. Before/after inode, size and
mtime identity is mandatory, and any evidence output must be fresh and below
the new project root.

This implementation used synthetic fixtures only. No legacy asset,
scientific record, checkpoint, final case or GWOSC/GWTC product was opened.
Actual legacy read-only reproduction remains closed until an exact wheel and
environment pass on AutoDL and the resulting packet receives a separate
delegated review.

The implementation-only Phase 7 boundary now also contains an exact final-score
summary release chain through:

`configs/execution/phase7_final_summary_release_stack_authorization.yaml`

Implementation commit `b4c7f0a07c717e2965ea3244c88c74d070449ce1`
requires exactly 45 immutable score artifacts: fifteen frozen namespaces for
each retained model seed 0, 1 and 2. It validates complete array schemas,
forbids posterior-draw persistence, requires byte-identical case/truth identity
across seeds and reports IID, lens-family, all eight EM-cell, all four
balanced-tail and every OOD/mismatch context without selecting a best seed.

The implementation applies the frozen IID, EM-cell and balanced-tail coverage
rules with raw counts and Wilson intervals. Failed gates can only narrow
claims. A non-authorizing packet and separate hash-bound delegated review are
required before any scientific score artifact may be opened. Final records,
checkpoints, calibration maps, threshold changes, manuscript-claim
finalization and GWOSC/GWTC remain closed. Only synthetic score fixtures were
used.

Before calibration-fit or final IID data exist, delegated scientific review
also freezes the downstream-only ablation addendum `1.1.0-rc.8` at
`configs/statistics/ablation_calibration_iid_preregistration.yaml`, canonical
hash `219160f67030bad745b0a4573d78d02f9d0db7536a6490c907196e8570647c9a`.
Each GW-only and EM-only checkpoint must fit its own split-conformal map for
its own model seed on the same 4,096 calibration-fit cases used by the
primary model. Primary maps, pooled maps and IID refitting are forbidden.

The six ablation results may later be compared descriptively with the
same-seed primary model only on the identical 8,192 IID cases using a frozen
10,000-replicate paired bootstrap. No best seed is selected, and no ablation
SBC, tail, OOD or mismatch execution is declared. Pure adapters and synthetic
tests are authorized through
`configs/execution/phase7_ablation_calibration_iid_stack_authorization.yaml`.
Scientific checkpoint access, calibration fitting, IID unsealing, comparison
execution, retraining, tuning and GWOSC/GWTC remain closed.

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
