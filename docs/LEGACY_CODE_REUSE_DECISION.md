# Legacy code reuse decision

## No generator is reusable unchanged

All old generators are monolithic, path-specific and tied to simplified priors or
noise models. The v2 implementation must be modular and tested. The curated
snapshot is evidence, not a vendored runtime dependency.

## Extract after unit testing

| Convention or implementation | Source lineage | Decision |
|---|---|---|
| IMRPhenomXPHM waveform configuration | all generators | retain configurable main option |
| shared intrinsic source parameters across images | all generators | retain invariant |
| amplitude factor `sqrt(abs(mu_i))` | SIS/PM generators | extract into tested physics API |
| saddle Morse factor with index 1/2 | SIS/PM generators | retain with explicit Fourier convention tests |
| separate image arrival times | PDF and later generators | retain; test detector response at each epoch |
| independent image noise seeds | corrected qk/wjx/PDF generators | retain via explicit seed hierarchy |
| memmap plus temporary output | qk/wjx generators | replace with resumable chunked writer and atomic manifests |
| separate noisy/clean/noise arrays | PDF generator | retain as artifact contract |
| observable/truth/diagnostic separation | PDF generator | retain and strengthen with input allowlist/denylist |
| source-level splits and frozen pair manifests | wjx reproducibility code | reuse design, rewrite for posterior task |

## Rewrite for v2

- lens model interfaces and image ordering;
- population/proposal samplers and accepted-bin balancing;
- detector network and real-noise ingestion;
- EM observable likelihood/covariance generation;
- dataset schema, IDs, manifests, checksums and resume state;
- posterior estimator, calibration and evaluation;
- configuration/CLI instead of top-level constants and hard-coded paths.

## Exclude from new implementation

- PI-ResNet/CQT pair classifiers as the posterior model;
- PNG-rendered CQT/Mel pipelines for absolute amplitude inference;
- hard-negative construction as a substitute for posterior validation;
- exact SIS beta/y inputs in deployable presets;
- old point-mass model as a proxy for an extended galaxy lens;
- copied `backup_original` files whose hashes equal current files and therefore do
  not preserve a distinct historical version.

## Required unit tests before extraction

1. signed versus absolute magnification and amplitude factor;
2. SIS analytic identities and limiting behavior;
3. Morse/Fourier phase convention;
4. image and arrival-time ordering;
5. detector-axis labels and permutation behavior;
6. whitening/amplitude preservation;
7. independent noise streams;
8. grouped split non-overlap;
9. truth/observable denylist enforcement;
10. deterministic manifest and resume behavior.
