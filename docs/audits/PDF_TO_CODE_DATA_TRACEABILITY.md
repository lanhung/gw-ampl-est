# PDF-to-code/data traceability

## Result

The baseline PDF is traceable to a concrete generator, dataset, training entry,
shared model module, checkpoint and validation table. It is not based on the
0222/0228 catalogs.

## Evidence chain

| Object | Verified path/evidence |
|---|---|
| PDF method reference | `docs/reference/baseline_realobs_quality_mu0_report_zh.pdf` |
| Generator used by the exact dataset | `/root/autodl-tmp/tmp/数据生成/SIS_GW_events_randomu_baseline_quality_mu0_before_obsfix.py` |
| Generator SHA256 | `ae3879fbe30137bf72a3d8cc746ab473aeaea593e06897c4e672ff57f650a604` |
| Exact dataset | `/root/autodl-tmp/tmp/数据生成/data_lens_randomu_realobs_quality_mu0` |
| Training entry | `/root/autodl-tmp/tmp/train_mu0_baseline_obs_realobs_quality.py` |
| Training-entry SHA256 | `10777f8888b2f6fca2da24670716d492b6d0de98b200ee4662aaca100f2d1d12` |
| Shared model/training module | `train_mu0_from_wave_obs_randomu_a21friendly_realobs.py` |
| Checkpoint | `runs/mu_direct_baseline_realobs_quality_waveobs/all/best_mu_direct_realobs_quality_all.pt` |
| Validation predictions | same run directory, `mu_direct_realobs_quality_all_val_predictions.csv` |

The checkpoint embeds the exact data path, observable columns, target transform,
model/training configuration and validation metrics. This is stronger evidence
than filename similarity.

## Checkpoint-to-PDF metric verification

| Metric | Checkpoint value | PDF value | Match |
|---|---:|---:|---|
| mu0 MAE | 0.7741573453 | 0.774 | yes |
| mu0 RMSE | 1.3239392042 | 1.324 | yes |
| mu0 MAPE | 13.164715% | 13.16% | yes |
| Pearson correlation | 0.8529922361 | 0.853 | yes |
| Validation rows | 500 | 500 | yes |

The checkpoint was selected on this 20% validation partition by mu0 MAPE. There
is no untouched test set for these reported values.

## Exact dataset contract

- 2,500 accepted SIS pairs from 2,825 attempts, seed 238.
- One simulated ET detector, not H1/L1/V1.
- 4096 Hz, 24-second simulation internally, saved as one-second centered windows.
- Each saved waveform array has shape `(2500, 4096)` and dtype float32.
- Separate noisy, clean and noise arrays exist in raw and whitened forms.
- Noise comes from `set_strain_data_from_power_spectral_density`; it is
  stationary Gaussian ET design noise, not detector strain.
- Waveform: IMRPhenomXPHM.
- Lens: SIS only; A21 stratified over 0.45–0.92, not mu0-balanced.
- mu0 spans 2.509–13.018; only about 130 examples have mu0 >= 10.
- Quality gates include image SNR 12/8, pair SNR 18, delay limits and a window
  signal/noise RMS proxy.

## Observable/truth boundary

The file split is materially better than the old 0222/0228 schema:

- `observable_features.csv`: simulated noisy proxies;
- `lens.csv`: labels and exact SIS variables;
- `diagnostic_features.csv`: privileged truth diagnostics;
- clean/noise strain: diagnostics only.

Nevertheless, the `all` checkpoint consumes highly informative simulated image
geometry, including observed image-position asymmetry. It also consumes peak time
difference. The quality report records Spearman correlation 1.0 between peak-time
difference and true delay and 0.851 between image asymmetry and true y. These are
not literal truth columns, but they create near-analytic shortcuts under exact SIS.

A naive plug-in inversion `1 + 1/y_obs` is numerically unstable because the
simulator clips noisy asymmetry as low as 0.0001; on all 2,500 rows its MAE is
approximately 270.15. This does not weaken the need for an analytic baseline: it
shows that uncertainty propagation and physically bounded likelihoods are needed
instead of direct inversion.

## Before-obsfix versus obsfix

`data_lens_randomu_realobs_quality_mu0(1)` was generated later by
`SIS_GW_events_randomu_realobs_quality_mu0.py`. Selected first/middle/last row
hashes are identical for all waveform arrays, but observable timing diagnostics
changed. The PDF checkpoint embeds the directory without `(1)`, so the latter is
not the evidence source for the published baseline metrics.

## PDF version discrepancy

The repository PDF and remote PDF have different binary and extracted-text
hashes. Their scientific core, metrics and code/data paths match. The repository
copy is a later shorter layout that removes an additional two-dimensional error
landscape discussion and renumbers the final figure. Neither PDF source directory
has Git history, so both are retained as technical references rather than formal
provenance authorities.
