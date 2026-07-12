# Phase 2.1 — Harden the preregistration before Phase 3A

Work only on `phase2/preregistration`. This is a design and documentation
correction phase. Do not generate waveform pairs, train a model, download
GWOSC/GWTC products, modify the frozen Phase 1B artifact, authorize Phase 3A,
or merge PR #3. Stop after Phase 2.1 and human/automatic review.

Read `AGENTS.md`, `docs/PHASE2_PREREGISTRATION.md`,
`configs/statistics/phase2_preregistration.yaml`,
`docs/audits/PHASE2_LITERATURE_AND_IDENTIFIABILITY.md`,
`docs/reports/PHASE2_ENTRY_REPORT.md`, `docs/PHYSICS_CONVENTIONS.md`,
`docs/V2_SCHEMA_SPEC.md`, and `docs/PRIVILEGED_INPUT_POLICY.md`.

The objective is to update preregistration `1.0.0-rc.1` to `1.0.0-rc.2`
and remove statistical and physical ambiguity before any 4,096-pair
qualification dataset is generated.

## Required corrections

1. Replace the 8,192-case calibration split with disjoint
   `calibration_fit: 6144` and `sbc_diagnostic: 2048`. Post-hoc correction may
   use only the former; 1,024 deterministic SBC replicates come only from the
   latter. Source, lens, physical-system, and noise groups may not overlap.
2. Define a 256-case validation `gold_development_subset`, which may trigger
   revision, and a frozen 256-case IID `gold_final_subset`, which is reported
   once after freeze and may only downgrade claims or require a new
   preregistration.
3. Run exactly three seeds per architecture. Select architecture by mean
   validation negative log probability across seeds, use the median only as a
   robustness diagnostic, never choose the best seed, retain all seeds, and
   break ties by lower parameter count.
4. Specify an external-convergence mass-sheet forward model and its effects on
   source position, image positions, signed/absolute magnification, Fermat
   potential, and physical delay. Require deterministic Phase 3A contracts.
5. Add deployable informative, weak, and unavailable environment observations
   for external convergence, with an explicit modality mask. Use future schema
   `2.0.0-alpha.3`; do not migrate or regenerate the frozen alpha.2 smoke data.
6. Generate velocity dispersion only through a declared Jeans/equivalent
   dynamics forward model with mass and light models, anisotropy, aperture,
   PSF, luminosity weighting, versions, and uncertainty. Otherwise remove it
   from Phase 3A; never use a direct truth shortcut.
7. Make all eight EM cells executable: explicit modalities, uncertainties,
   covariance, redshift model, timing, environment, and kinematics state.
8. Replace the false `lens_family_ood` label with exact cross-family
   misspecification tests (or justify a genuinely unseen family); freeze exact
   parameter OOD regions; separate waveform and PSD mismatch; verify PSD file
   versions/hashes; and freeze balanced-tail strata before generation.
9. Define joint primary-mass/mass-ratio/secondary-mass support and normalized
   proposal/evaluation log-density formulas for importance weighting.
10. Create `docs/literature_matrix.csv`,
    `references/verified_references.bib`, and
    `docs/NOVELTY_AND_POSITIONING.md`, covering GOLUM, separated-image
    Bayesian identification, Gravelamps, DINGO/DINGO-lensing, reconstruction
    and degeneracies, LVK searches, neural micro/millilensing inference, ML
    pair identification, and multimessenger optical modeling. Verify primary
    sources and do not claim “first” without support.
11. Estimate peak disk use separately for shard staging, atomic publication,
    checksum verification, retained failures, and simultaneous products. Gate
    production on Phase 3A's measured amplification and a post-peak reserve.
12. Update rc.2 hash, reports, project state, decisions, failures, experiment
    registry, and PR #3 body. The body must no longer list numerical priors,
    sample sizes, observation models, or storage plans as unfinished.

## Acceptance

Phase 2.1 passes only if all thirteen design corrections above are complete;
all authorization flags remain false; no data are generated; no model is
trained; and no GWOSC/GWTC product is downloaded. Run pytest, Ruff, mypy,
package build, count/storage/hash consistency checks, and optional AutoDL
scientific contracts. Create
`docs/reports/PHASE2A_PREREGISTRATION_HARDENING_REPORT.md`, commit as
`docs: harden preregistration for independent calibration and physical EM models`,
push, and stop for review.
