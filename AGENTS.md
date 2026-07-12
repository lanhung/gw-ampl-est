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

Phase 3C-0 may retain only implementation code, configurations, unit tests,
latent-only evidence, dry-run A/B validation and documentation. It must now
stop for human review.

During and after this hard stop, do not:

- generate any additional pair from the qualification or scientific plan;
- run the 1,024-pair proposal A/B qualification;
- authorize or adopt proposal-v2 RC.1;
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
- authorize or proceed to Phase 3C-A or later phases.

All Phase 3A remote outputs must remain under:

`/root/autodl-tmp/lensing-4/data_v2/production`

Full production, staged scientific production, proposal-v2 qualification,
model training, calibration, SBC, scientific IID/OOD/mismatch testing,
real-noise work, GWOSC/GWTC access and Phase 3C-A remain closed. RC.5 remains
the only qualified proposal. The Phase 3A artifact and all future engineering
A/B artifacts must never enter any scientific split.

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
