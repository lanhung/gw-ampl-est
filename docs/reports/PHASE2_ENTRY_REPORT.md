# Phase 2 preregistration report

## Outcome

Phase 2 design work is complete as preregistration `1.0.0-rc.1` and awaits
human review. No waveform pairs were generated, no model was trained, no
GWOSC/GWTC product was downloaded, and no legacy or frozen smoke dataset file
was modified.

## Completed

- audited literature and stated the distance-magnification, similarity and
  mass-sheet identifiability boundaries;
- froze a model-conditional joint posterior estimand for the two selected
  images' log absolute magnifications;
- separated a broad support proposal from a balanced benchmark evaluation
  distribution and denied astrophysical-rate claims for both;
- fixed numerical lens/source priors, Planck18 baseline, explicit external
  convergence, synthetic pair selection and eight EM availability cells;
- fixed an H1/L1/V1 eight-second observation, IMRPhenomXPHM baseline,
  SEOBNRv4PHM mismatch and detector-specific curve identities/hashes;
- fixed the mask-aware multimessenger spline-flow model-selection budget,
  matched ablations, baselines and likelihood gold subset;
- fixed train/validation/calibration/IID/diagnostic counts and coverage, SBC,
  stratification and failure rules;
- created an execution-disabled Phase 3 plan with exact storage budget, log and
  staging paths, atomic shard protocol and resume test;
- added tests that fail if authorization opens, counts/storage diverge or
  evaluation support leaves the proposal.

## Verification

- canonical configuration hash:
  `4ae2899a054342fcc1100554f72cd826969afb7030885edbcaacb251efd603aa`;
- local: 99 passed, one optional Lenstronomy module skipped; Ruff, mypy, sdist
  and wheel build passed;
- AutoDL: 104 authoritative tests passed including Lenstronomy fixtures; Ruff,
  mypy, sdist and wheel build passed. A stale duplicate test retained by
  no-delete sync was explicitly ignored and is recorded in `docs/FAILURES.md`;
- planned 118,784 pairs equal 140,123,308,032 raw array bytes; 30% reserve is
  182,160,300,442 bytes; measured free space was 342,407,888,896 bytes;
- the Phase 3A qualification subset is 4,831,838,208 raw bytes, below 10 GB.

## Findings and failed attempts

The Phase 2 audit found that Phase 1B's frozen generic
`synthetic_gaussian_design_psd` label is imprecise. Bilby 2.6.0 used
`aLIGO_O4_high_asd.txt` for H1/L1 and `AdV_psd.txt` for V1. The immutable smoke
artifact was not changed; the limitation and exact hashes are now documented.

The pinned LALSuite rejects `SEOBNRv5PHM`; the available SEOBNRv4PHM is
preregistered instead. Package build succeeds but retains the known missing-
README warning. No failed scientific result exists because Phase 2 performed
no inference or generation.

## Deferred and external gates

Human review is the only remaining Phase 2 gate. Approval must explicitly open
Phase 3A; it does not authorize full production or training. Phase 3A must first
write its concrete manifest, measure solver acceptance and throughput, validate
waveform boundaries and whitening, confirm actual storage, and demonstrate
interruption/resume on the 4,096-pair engineering qualification set. Those
measurements then determine the full runtime estimate. `real_noise_test`
remains empty until a separate GWOSC, data-quality and PSD authorization.
