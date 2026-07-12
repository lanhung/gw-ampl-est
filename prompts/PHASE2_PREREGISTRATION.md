# Phase 2 — Literature, identifiability, and preregistration

Work only on `phase2/preregistration`.

Phase 2 is documentation and statistical design. Do not generate waveform
pairs, train a model, download GWOSC/GWTC products, mutate legacy roots, or use
the frozen 48-pair smoke artifact for any scientific analysis.

Required deliverables:

1. audit literature relevant to galaxy-scale lensed-GW image physics,
   multimessenger lens reconstruction, mass-sheet/distance degeneracies, and
   posterior calibration;
2. define model-conditional estimands and non-identifiability boundaries;
3. preregister target populations, proposal and evaluation priors,
   observation/selection models, splits, baselines, ablations, calibration and
   OOD tests;
4. preserve the deployable-input allowlist and privileged-variable denylist;
5. define minimum sample sizes by a documented power/precision calculation,
   plus storage, runtime, logging, manifest, and resume plans before execution;
6. write `docs/reports/PHASE2_REPORT.md` and obtain human review before opening
   any Phase 3 execution gate.

The Phase 2 entry commit may establish the audit, preregistration skeleton,
execution-disabled configuration, and report. Stop before data or model work.
