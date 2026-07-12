# Legacy data decision

## Main decision

No existing dataset is approved for v2 main training or final scientific testing.
All legacy data remain immutable benchmarks.

## Asset classification

| Asset | Classification | Permitted use |
|---|---|---|
| qkzhang SIS 0222/0228 | retain unchanged as legacy | reproduce pair baseline, SIS identities, early smoke |
| qkzhang PM 0222/0228 | legacy comparison only | old pair-verification and OOD toy checks |
| qkzhang unlensed 0222/0228 | legacy background only | reproduce old pair protocol |
| wjx `final_v3`/`ligo_reduced` | smoke-only duplicate lineage | detector-axis and loader regression tests |
| wjx `ligo_full` | legacy scale benchmark | I/O/throughput tests only |
| wjx `ligo_bf_sis` | exclude | incomplete duplicate |
| PDF before-obsfix data | exact legacy baseline | reproduce point-regression checkpoint and diagnostics |
| PDF obsfix data | privileged diagnostics/smoke | inspect observable-fix behavior only |
| legacy CQT/Mel PNG | exclude from absolute-magnification model | old classifier only |

## Why none can be main training

- qkzhang arrays lack a reliable detector identity/axis and use Gaussian design
  noise with incomplete generator provenance.
- wjx H1/L1 regenerations fix the axis but retain simplified SIS/PM priors,
  Gaussian design noise, no V1, no EM likelihood and no lens-model uncertainty.
- the PDF data improve truth/observable separation but are ET-only, SIS-only,
  Gaussian-noise, A21-balanced point-regression data.
- none includes SIE/power-law+shear, external convergence, mass-sheet nuisance,
  missing modalities, covariance-aware EM observations or posterior targets.

## Why none can be final testing

Final tests must be fixed before model selection and cover IID calibration, lens
family mismatch, real detector noise, waveform/PSD mismatch and missing modalities.
Every existing set was built for an earlier task or inspected during baseline
development. Reusing it as final evidence would create task mismatch and selection
bias.

## Privileged-variable denylist for v2 deployable inputs

- exact `y`, `u`, `log_u`, signed/absolute magnifications and A21/R21 truth;
- exact beta/beta_x/beta_y and exact lens center;
- exact lens parameters when only noisy measurements are intended;
- exact time delay or geocentric truth when an observed timing product is intended;
- clean signal, isolated noise realization, optimal SNR and simulation-only
  quality diagnostics;
- source/lens/image truth IDs except as grouping metadata unavailable to the model.

## Required v2 regeneration

1. explicit `(pair, image, detector, sample)` schema and detector masks;
2. SIS analytic control plus SIE+shear and power-law+shear;
3. external convergence/model discrepancy and lens-family mixture;
4. noisy EM observables with covariance and missing-modality masks;
5. proposal distribution separated from astrophysical evaluation prior;
6. group IDs and frozen train/calibration/test/OOD splits;
7. posterior targets and SBC/coverage-ready truth tables;
8. Gaussian IID stage followed by GWOSC real-noise segments with DQ/PSD manifests.

## Minimum v2 smoke dataset

Generate 10–100 pairs covering SIS analytic limits and at least a few SIE and
power-law systems. Include three detectors in the schema even when a detector is
masked, two images, truth/observable tables, covariance, group IDs and manifests.
Store noisy/clean/noise arrays separately and test amplitude preservation.

Expected smoke storage is below 5 GB and should finish in less than one hour on
the available GPUs/CPUs. No full generation is approved until these tests pass.

For 10,000 pairs at one second, float32, two images and three detectors, one
strain tensor is about 0.98 GB. Storing noisy, clean and noise tensors is about
2.95 GB before metadata/overhead. Keeping full 24-second versions multiplies this
by 24; therefore v2 should avoid duplicated time arrays and use chunked storage.
