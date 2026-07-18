# Final-evaluation inference runbook

This runbook implements the frozen RC.6/RC.7 downstream contracts. It is not
an execution authorization. The final pool, selected checkpoints and
calibration/SBC products remain inaccessible until one later machine-readable
gate binds every immutable identity.

## Required entry evidence

The future gate must bind:

- the atomic sealed 20,480-system parent manifest and all 15 namespaces;
- the selected architecture decision and its model-configuration hash;
- checkpoints for seeds 0, 1 and 2;
- same-seed calibration map, independent SBC summary, coverage summary and
  calibration/SBC run-summary hashes;
- one immutable inference commit and normalized CUDA environment;
- one distinct output path for every seed and namespace.

It must explicitly permit final-data unsealing, final-data access,
selected-checkpoint inference, same-seed calibration-map application and
immutable score-artifact creation. Training, tuning, calibration refitting,
architecture/size selection, result-threshold changes and GWOSC/GWTC access
must remain false.

## Execution unit

One job processes exactly one `(model seed, final namespace)` pair. It:

1. resolves the sealed parent without opening Parquet or Zarr;
2. validates all 15 namespace identities and the finalized commitment;
3. validates the selected checkpoint, standardizers, model configuration,
   environment and PSD files;
4. lazily opens one row and one noisy-strain array at a time;
5. applies the frozen namespace and cross-family interpretation;
6. draws exactly 4,096 posterior samples per case in microbatches no larger
   than 512;
7. applies the matching model seed's frozen EM-cell calibration map;
8. writes only per-case scores, labels and immutable identities.

Posterior draws are bounded to a physical batch and are never stored. The
output contains truth log density, NLP per target dimension, CRPS, raw
marginal/joint region scores, calibrated coverage indicators and interval
widths. Truth and diagnostic labels remain offline score fields, not model
inputs.

## Cross-family contexts

The legacy SIE-truth/EPL-assumed namespace is evaluated under the EPL family
condition marginalized over the frozen training slope prior. The reverse cell
uses the SIE family condition. Family-marginalized cells use exactly 2,048 SIE
and 2,048 EPL draws per case and the equal-density log mixture. Component
calibration maps are not combined; the matching seed and EM-cell map is
applied to the resulting mixture scores.

## Failure behavior

The runner fails before data access on any missing or changed identity. It
also fails on namespace/count drift, nonfinite draws or densities, missing
EM-cell maps, duplicate IDs, output reuse, oversized microbatches or a mixed
checkpoint/calibration seed. No failure may open training, refit calibration,
change a threshold or expose final data to model selection.

