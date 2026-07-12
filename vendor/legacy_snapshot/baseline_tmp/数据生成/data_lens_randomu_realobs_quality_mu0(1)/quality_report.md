# Realobs Quality Mu0 Dataset Report

Generator: `realobs_quality_controlled_mu0_v1`
Attempts: `2825`
Accepted: `2500`
Acceptance rate: `0.8850`

## Reject Counts

| reason | count |
|---|---:|
| `low_pair_network_snr` | 1 |
| `low_snr_image_1` | 2 |
| `low_snr_image_2` | 5 |
| `low_window_signal_noise_rms` | 4 |
| `td_out_of_range` | 313 |

## Diagnostic Spearman Correlations

| pair | corr |
|---|---:|
| `peak_amp_ratio_21_vs_A21_true` | 0.220483 |
| `rms_ratio_21_vs_A21_true` | 0.182119 |
| `energy_ratio_21_vs_R_abs_21_true` | 0.182119 |
| `env_area_ratio_21_vs_A21_true` | 0.180116 |
| `snr_ratio_21_vs_A21_true` | 0.385285 |
| `image_asymmetry_vs_y_true` | 0.851065 |
| `t_d_observed_vs_t_d_true` | 1.000000 |
| `peak_time_diff_vs_t_d_true` | 1.000000 |

## Key Distributions

### `mu0`

`min=2.50949, p05=2.58977, median=3.96477, mean=4.88544, p95=10.1848, max=13.0184`

### `A21_true`

`min=0.450583, p05=0.47721, median=0.703958, mean=0.698478, p95=0.896454, max=0.919984`

### `y_true`

`min=0.083206, p05=0.108875, median=0.337294, mean=0.348667, p95=0.629023, max=0.662476`

### `t_d_true`

`min=9463.8, p05=167850, median=1.82517e+06, mean=2.63745e+06, p95=7.74783e+06, max=9.9943e+06`

### `t_d_observed`

`min=9463.83, p05=167850, median=1.82517e+06, mean=2.63745e+06, p95=7.74783e+06, max=9.9943e+06`

### `snr_1`

`min=12.9543, p05=33.2769, median=97.6599, mean=130.162, p95=329.969, max=1666.66`

### `snr_2`

`min=8.84637, p05=20.4013, median=67.1284, mean=93.3391, p95=263.823, max=985.419`

### `peak_amp_ratio_21`

`min=0.123776, p05=0.373801, median=0.796786, mean=0.879454, p95=1.67401, max=5.0167`

### `rms_ratio_21`

`min=0.195046, p05=0.482705, median=0.870496, mean=0.892308, p95=1.41246, max=3.22172`

### `image_position_asymmetry_observed`

`min=0.0001, p05=0.0655813, median=0.326619, mean=0.338796, p95=0.64067, max=0.924059`
