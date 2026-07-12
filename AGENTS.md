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

Phase 0, Phase 1A, Phase 1A.1, Phase 1B, Phase 1B.1, Phase 2 and
Phase 2.1 are complete and merged into main at:

`8e14d927146ac3b2223343eaca89a323c58d5e13`

Human review explicitly authorizes Phase 3A only.

The frozen scientific design is:

- preregistration version: `1.0.0-rc.3`;
- preregistration configuration:
  `configs/statistics/phase2_preregistration.yaml`;
- canonical configuration hash:
  `16a75327df5aacafa1fb4459e19429cc08d3350cd3056986356ef3c57864c1e8`;
- scientific schema: `2.0.0-alpha.3`;
- frozen engineering-smoke schema: `2.0.0-alpha.2`.

Phase 3A may only:

- implement the bounded-memory production generator and storage pipeline;
- implement and validate the alpha.3 scientific observation schema;
- validate mass-sheet external convergence and stellar-kinematics contracts;
- run small preflight and runtime qualification cases;
- generate exactly 4,096 accepted generator-qualification pairs;
- publish exactly 32 atomic shards of 128 accepted pairs;
- measure acceptance rate, throughput, peak memory, peak disk use,
  whitening behavior and interruption/resume reproducibility;
- commit only code, configurations, manifests, reports and small evidence.

The Phase 3A dataset is engineering qualification data only. It may never
be used for model training, post-hoc calibration, SBC, IID testing, OOD
testing or reported scientific performance.

During Phase 3A, do not:

- generate more than 4,096 accepted pairs;
- start the 118,784-pair production plan;
- train or tune any neural model;
- download GWOSC or GWTC products;
- generate real-noise data;
- alter the frozen Phase 1B smoke artifact;
- alter the frozen RC.3 scientific distributions merely to make generation pass;
- call synthetic Gaussian noise real detector noise;
- modify any legacy file;
- start manuscript writing;
- proceed to Phase 3B or later phases.

All Phase 3A remote outputs must remain under:

`/root/autodl-tmp/lensing-4/data_v2/production`

Any hard failure must stop publication and be recorded. Phase 3A must stop
after the 4,096-pair qualification report and human-review gate.

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
