#!/usr/bin/env python
# coding: utf-8
"""
Train mu0 predictor on the quality-controlled real-observation SIS dataset.

This entry point reuses the existing compact wave+observable model and training
loop, but replaces the dataset paths and observable feature policy for:

    数据生成/SIS_GW_events_randomu_realobs_quality_mu0.py

The important difference from older training configs is the observable feature
set.  The new generator provides higher-quality local/windowed waveform
features, observed time-delay uncertainty, and noisy optical image geometry.
These are real-observation features, not label-side truth columns.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train_mu0_from_wave_obs_randomu_a21friendly_realobs as base  # noqa: E402


DEFAULT_DATA_ROOT = (
    "/root/autodl-tmp/tmp/数据生成/"
    "data_lens_randomu_realobs_quality_mu0"
)
DEFAULT_OUT_DIR = "/root/autodl-tmp/tmp/runs/mu_direct_baseline_realobs_quality_waveobs"


# Existing default columns kept for compatibility with the older model.
LEGACY_WAVE_COLS = [
    "peak_time_diff",
    "env_peak_time_diff",
    "peak_amp_ratio_21",
    "rms_ratio_21",
    "energy_ratio_21",
    "xcorr_lag",
    "norm_xcorr_max",
    "xcorr_lag_win65",
    "norm_xcorr_max_win65",
    "peak_amp_ratio_local_65",
    "rms_ratio_local_65",
    "energy_ratio_local_65",
    "env_area_ratio_21",
    "aligned_l1_residual",
    "aligned_l2_residual",
    "spec_centroid_diff",
    "band_energy_ratio_low_21",
    "band_energy_ratio_mid_21",
    "band_energy_ratio_high_21",
]


TIME_OBS_COLS = [
    "peak_time_diff",
    "env_peak_time_diff",
    "arrival_time_1_sigma",
    "arrival_time_2_sigma",
]


A21_PROXY_COLS = [
    "peak_amp_ratio_21",
    "rms_ratio_21",
    "energy_ratio_21",
    "peak_amp_ratio_local_65",
    "rms_ratio_local_65",
    "energy_ratio_local_65",
    "env_area_ratio_21",
    "win0p25_peak_amp_ratio_21",
    "win0p25_rms_ratio_21",
    "win0p25_energy_ratio_21",
    "win0p25_env_area_ratio_21",
    "win0p5_peak_amp_ratio_21",
    "win0p5_rms_ratio_21",
    "win0p5_energy_ratio_21",
    "win0p5_env_area_ratio_21",
    "win0p9_peak_amp_ratio_21",
    "win0p9_rms_ratio_21",
    "win0p9_energy_ratio_21",
    "win0p9_env_area_ratio_21",
]


WAVE_SHAPE_COLS = [
    "xcorr_lag",
    "norm_xcorr_max",
    "xcorr_lag_win65",
    "norm_xcorr_max_win65",
    "aligned_l1_residual",
    "aligned_l2_residual",
    "spec_centroid_diff",
    "band_energy_ratio_low_21",
    "band_energy_ratio_mid_21",
    "band_energy_ratio_high_21",
]


NOISE_CONTEXT_COLS = [
    "local_noise_rms_1",
    "local_noise_rms_2",
]


OPTICAL_REDSHIFT_COLS = [
    "z_s_observed",
    "z_l_observed",
    "z_l_over_z_s",
    "log1p_z_s_observed",
    "log1p_z_l_observed",
]


OPTICAL_SIGMA_COLS = [
    "sigma_v_observed",
    "log_sigma_v_observed",
]


OPTICAL_SCALE_COLS = [
    "theta_E_observed",
    "image_separation_observed",
]


IMAGE_POSITION_COLS = [
    "theta_plus_abs_observed",
    "theta_minus_abs_observed",
    "image_position_asymmetry_observed",
    "lens_center_x_observed",
    "lens_center_y_observed",
    "image_x_0_observed",
    "image_y_0_observed",
    "image_x_1_observed",
    "image_y_1_observed",
]


OPTIMAL_SNR_COLS = [
    "snr_1",
    "snr_2",
    "snr_ratio_21",
    "snr_sum",
    "snr_diff",
]


def unique_cols(cols: List[str]) -> List[str]:
    out = []
    seen = set()
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def build_quality_obs_cols(preset: str, use_optimal_snr: bool = False) -> List[str]:
    """Return observable columns for a controlled experiment preset."""
    if preset == "legacy":
        cols = list(LEGACY_WAVE_COLS)
        cols += OPTICAL_REDSHIFT_COLS + OPTICAL_SIGMA_COLS + OPTICAL_SCALE_COLS
    elif preset == "a21_wave":
        cols = A21_PROXY_COLS + WAVE_SHAPE_COLS + NOISE_CONTEXT_COLS
    elif preset == "time_lens":
        cols = TIME_OBS_COLS + OPTICAL_REDSHIFT_COLS + OPTICAL_SIGMA_COLS + OPTICAL_SCALE_COLS
    elif preset in {"gw_only", "no_optical"}:
        cols = A21_PROXY_COLS + WAVE_SHAPE_COLS + NOISE_CONTEXT_COLS + TIME_OBS_COLS
    elif preset == "no_image":
        cols = (
            A21_PROXY_COLS
            + WAVE_SHAPE_COLS
            + NOISE_CONTEXT_COLS
            + TIME_OBS_COLS
            + OPTICAL_REDSHIFT_COLS
            + OPTICAL_SIGMA_COLS
            + OPTICAL_SCALE_COLS
        )
    elif preset == "no_time":
        cols = (
            A21_PROXY_COLS
            + WAVE_SHAPE_COLS
            + NOISE_CONTEXT_COLS
            + OPTICAL_REDSHIFT_COLS
            + OPTICAL_SIGMA_COLS
            + OPTICAL_SCALE_COLS
            + IMAGE_POSITION_COLS
        )
    elif preset == "all":
        cols = (
            A21_PROXY_COLS
            + WAVE_SHAPE_COLS
            + NOISE_CONTEXT_COLS
            + TIME_OBS_COLS
            + OPTICAL_REDSHIFT_COLS
            + OPTICAL_SIGMA_COLS
            + OPTICAL_SCALE_COLS
            + IMAGE_POSITION_COLS
        )
    else:
        raise ValueError(f"Unsupported preset: {preset}")

    if use_optimal_snr:
        cols += OPTIMAL_SNR_COLS
    return unique_cols(cols)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train mu0 on quality-controlled real-observation SIS data."
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--preset",
        type=str,
        default="all",
        choices=[
            "all",
            "gw_only",
            "no_optical",
            "no_image",
            "no_time",
            "a21_wave",
            "time_lens",
            "legacy",
        ],
        help=(
            "Observable feature preset. all uses every safe observed chain; "
            "gw_only/no_optical removes all optical observables; "
            "no_image/no_time are ablations; a21_wave isolates waveform/A21 proxies; "
            "time_lens isolates time-delay plus lens-scale observables."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--target-len", type=int, default=4096)
    parser.add_argument("--raw-scale", type=float, default=None)
    parser.add_argument("--use-optimal-snr", action="store_true")
    parser.add_argument("--allow-oracle", action="store_true")
    return parser.parse_args()


def configure_base(args) -> None:
    cfg = base.cfg

    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)
    preset_out_dir = os.path.join(out_dir, args.preset)

    cfg.DATA_ROOT = data_root
    cfg.LENS_CSV = os.path.join(data_root, "lens.csv")
    cfg.LENS_PARAMS_CSV = os.path.join(data_root, "lens_params.csv")
    cfg.OBS_CSV_NOISY_WHITE = os.path.join(data_root, "observable_features.csv")
    cfg.OBS_CSV_CLEAN_WHITE = os.path.join(data_root, "observable_features_clean.csv")
    cfg.OBS_CSV_NOISY_RAW = os.path.join(data_root, "observable_features_raw.csv")
    cfg.OBS_CSV_CLEAN_RAW = os.path.join(data_root, "observable_features_clean_raw.csv")
    cfg.OUT_DIR = preset_out_dir

    cfg.USE_NOISY_WAVE = True
    cfg.USE_WHITENED_WAVE = True
    cfg.WAVE_NORM_MODE = "pair"
    cfg.TARGET_LEN = args.target_len
    cfg.USE_OPTIMAL_SNR_FEATURES = bool(args.use_optimal_snr)
    cfg.ALLOW_ORACLE_FEATURES = bool(args.allow_oracle)

    # These booleans are printed by the base main() and remain semantically
    # useful, although selected_obs_cols is replaced below.
    cfg.USE_OPTICAL_REDSHIFT = args.preset in {"all", "no_image", "no_time", "time_lens", "legacy"}
    cfg.USE_OPTICAL_SIGMA_V = args.preset in {"all", "no_image", "no_time", "time_lens", "legacy"}
    cfg.USE_OPTICAL_IMAGE_GEOMETRY = args.preset in {"all", "no_image", "no_time", "time_lens", "legacy"}
    cfg.USE_OPTICAL_DISTANCE = False

    # Keep oracle switches off.  The new real-observation columns are observed
    # proxies, not the truth columns guarded by these switches.
    cfg.USE_ORACLE_TD = False
    cfg.USE_ORACLE_BETA = False
    cfg.USE_ORACLE_R_ABS = False
    cfg.USE_ORACLE_AMP_RATIO = False
    cfg.USE_ORACLE_Y = False

    cfg.CKPT_NAME = f"best_mu_direct_realobs_quality_{args.preset}.pt"
    cfg.PLOT_NAME = f"mu_direct_realobs_quality_{args.preset}_scatter.png"
    cfg.CSV_NAME = f"mu_direct_realobs_quality_{args.preset}_val_predictions.csv"

    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        cfg.LR = args.lr
    if args.raw_scale is not None:
        cfg.RAW_SCALE = args.raw_scale

    cfg.QUALITY_PRESET = args.preset

    def selected_obs_cols_quality() -> List[str]:
        return build_quality_obs_cols(
            preset=cfg.QUALITY_PRESET,
            use_optimal_snr=cfg.USE_OPTIMAL_SNR_FEATURES,
        )

    base.selected_obs_cols = selected_obs_cols_quality


def main():
    args = parse_args()
    configure_base(args)
    print("=" * 90)
    print("Quality realobs mu0 training entry")
    print(f"Data root: {base.cfg.DATA_ROOT}")
    print(f"Out dir:   {base.cfg.OUT_DIR}")
    print(f"Preset:    {base.cfg.QUALITY_PRESET}")
    print(f"Obs cols:  {len(base.selected_obs_cols())}")
    print("Selected columns:")
    for col in base.selected_obs_cols():
        print(f"  - {col}")
    print("=" * 90)
    base.main()


if __name__ == "__main__":
    main()
