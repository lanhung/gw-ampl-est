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
