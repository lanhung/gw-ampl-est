#!/usr/bin/env python
# coding: utf-8
"""
Train mu0 predictor on the obsfix quality-controlled real-observation SIS dataset.

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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import WeightedRandomSampler


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train_mu0_from_wave_obs_randomu_a21friendly_realobs as base  # noqa: E402


DEFAULT_DATA_ROOT = (
    "/root/autodl-tmp/tmp/数据生成/"
    "data_lens_randomu_realobs_quality_mu0(1)"
)
DEFAULT_OUT_DIR = "/root/autodl-tmp/tmp/runs/mu_direct_realobs_quality_obsfix_nooptical_waveobs"


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


IMAGE_POSITION_DERIVED_COLS = [
    "image_asymmetry_abs_observed",
    "log_image_position_asymmetry_observed",
    "neg_log_image_position_asymmetry_observed",
    "sis_mu0_proxy_from_image_asymmetry_observed",
    "log_sis_mu0_minus1_proxy_from_image_asymmetry_observed",
]


OPTIMAL_SNR_COLS = [
    "snr_1",
    "snr_2",
    "snr_ratio_21",
    "snr_sum",
    "snr_diff",
]


OPTICAL_FEATURE_PRESETS = {"all", "no_image", "no_time", "time_lens", "legacy"}
OPTICAL_OBS_COLS = set(
    OPTICAL_REDSHIFT_COLS
    + OPTICAL_SIGMA_COLS
    + OPTICAL_SCALE_COLS
    + IMAGE_POSITION_COLS
    + IMAGE_POSITION_DERIVED_COLS
)


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
            + IMAGE_POSITION_DERIVED_COLS
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
            + IMAGE_POSITION_DERIVED_COLS
        )
    else:
        raise ValueError(f"Unsupported preset: {preset}")

    if use_optimal_snr:
        cols += OPTIMAL_SNR_COLS
    return unique_cols(cols)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train mu0 on obsfix quality-controlled real-observation SIS data "
            "generated by 数据生成/SIS_GW_events_randomu_realobs_quality_mu0.py."
        )
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--preset",
        type=str,
        default="no_optical",
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
            "Observable feature preset. This no-optical entry strips optical "
            "observables from every preset; all/no_image/no_time only affect "
            "the remaining GW/time/wave feature groups. "
            "no_image/no_time are ablations; a21_wave isolates waveform/A21 proxies; "
            "time_lens isolates time-delay observables after optical stripping."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--target-len", type=int, default=4096)
    parser.add_argument("--raw-scale", type=float, default=None)
    parser.add_argument("--use-optimal-snr", action="store_true")
    parser.add_argument("--allow-oracle", action="store_true")
    parser.add_argument(
        "--no-physics-derived",
        action="store_true",
        help=(
            "Disable SIS physics-inspired features derived from observed image "
            "asymmetry. Useful for ablation."
        ),
    )
    parser.add_argument(
        "--no-tail-sampler",
        action="store_true",
        help="Disable high-mu0 tail oversampling in the training loader.",
    )
    parser.add_argument(
        "--no-tail-loss",
        action="store_true",
        help="Disable obsfix tail-aware loss and use the base loss.",
    )
    parser.add_argument(
        "--tail-pivot",
        type=float,
        default=6.0,
        help="mu0 value above which the obsfix tail loss/sampler is emphasized.",
    )
    return parser.parse_args()


_BASE_ADD_OPTIONAL_OBSERVABLES = base.add_optional_observables
_BASE_COMPUTE_LOSSES = base.compute_losses
_BASE_SAVE_VAL_CSV = base.save_val_csv


def add_obsfix_physics_observables(obs_df, lens_params_df):
    """Add observed SIS geometry features without using label-side truth."""
    if not bool(getattr(base.cfg, "OBS_FIX_USE_PHYSICS_DERIVED", False)):
        return obs_df.copy()

    obs_df = _BASE_ADD_OPTIONAL_OBSERVABLES(obs_df, lens_params_df)

    if "image_position_asymmetry_observed" not in obs_df.columns:
        theta_cols = {"theta_plus_abs_observed", "theta_minus_abs_observed"}
        if theta_cols <= set(obs_df.columns):
            theta_plus = np.abs(obs_df["theta_plus_abs_observed"].to_numpy(dtype=np.float64))
            theta_minus = np.abs(obs_df["theta_minus_abs_observed"].to_numpy(dtype=np.float64))
            obs_df["image_position_asymmetry_observed"] = (
                np.abs(theta_plus - theta_minus) / np.maximum(theta_plus + theta_minus, 1e-6)
            ).astype(np.float32)

    if "image_position_asymmetry_observed" not in obs_df.columns:
        return obs_df

    # In SIS, image position asymmetry is an observed proxy for y=beta/theta_E,
    # and log(mu0 - 1) = -log(y).  Clipping prevents noisy near-zero astrometry
    # from creating unbounded inputs.
    y_obs = np.abs(
        obs_df["image_position_asymmetry_observed"].to_numpy(dtype=np.float64)
    )
    y_obs = np.clip(y_obs, 1e-3, 0.995)
    mu0_proxy = np.clip(1.0 + 1.0 / y_obs, 2.0, 20.0)

    obs_df["image_asymmetry_abs_observed"] = y_obs.astype(np.float32)
    obs_df["log_image_position_asymmetry_observed"] = np.log(y_obs).astype(np.float32)
    obs_df["neg_log_image_position_asymmetry_observed"] = (-np.log(y_obs)).astype(np.float32)
    obs_df["sis_mu0_proxy_from_image_asymmetry_observed"] = mu0_proxy.astype(np.float32)
    obs_df["log_sis_mu0_minus1_proxy_from_image_asymmetry_observed"] = (
        np.log(np.maximum(mu0_proxy - 1.0, 1e-3))
    ).astype(np.float32)

    return obs_df


def save_val_csv_obsfix(eval_out, save_path, obs_df=None):
    if bool(getattr(base.cfg, "OBS_FIX_DISABLE_OPTICAL_OUTPUT", False)):
        return _BASE_SAVE_VAL_CSV(eval_out, save_path, obs_df=None)
    return _BASE_SAVE_VAL_CSV(eval_out, save_path, obs_df=obs_df)


def compute_losses_obsfix_tail(pred_target, target_transformed, mu0_true):
    """Tail-aware loss aimed at reducing systematic high-mu0 underprediction."""
    mu0_pred = base.target_inverse_torch(pred_target)
    pivot = float(getattr(base.cfg, "OBS_FIX_TAIL_PIVOT", 6.0))
    tail_alpha = float(getattr(base.cfg, "OBS_FIX_TAIL_LOSS_ALPHA", 1.25))
    tail_power = float(getattr(base.cfg, "OBS_FIX_TAIL_LOSS_POWER", 1.35))
    under_weight = float(getattr(base.cfg, "OBS_FIX_UNDERPRED_W", 0.035))

    tail = torch.relu((mu0_true - pivot) / max(pivot, base.cfg.EPS))
    weight = 1.0 + tail_alpha * torch.pow(tail, tail_power)

    trans_err = F.smooth_l1_loss(
        pred_target,
        target_transformed,
        beta=0.10,
        reduction="none",
    )
    trans_loss = (trans_err * weight).mean()

    abs_err = torch.abs(mu0_pred - mu0_true)
    rel_err = abs_err / (torch.abs(mu0_true) + base.cfg.EPS)

    mae = (abs_err * weight).mean()
    mape = (rel_err * weight).mean() * 100.0

    # The baseline scatter shows a one-sided error in the high-mu0 tail.  This
    # term only activates for underprediction above the pivot, so it does not
    # globally push all predictions upward.
    under_rel = torch.relu(mu0_true - mu0_pred) / (torch.abs(mu0_true) + base.cfg.EPS)
    tail_mask = (mu0_true > pivot).to(mu0_true.dtype)
    under_penalty = (under_rel * weight * tail_mask).mean() * 100.0

    loss = (
        trans_loss
        + base.cfg.LOSS_W_MAE * mae
        + base.cfg.LOSS_W_REL * mape
        + under_weight * under_penalty
    )

    return loss, mae, mape


def obsfix_tail_weighted_dataloader(dataset, *args, **kwargs):
    use_tail_sampler = bool(getattr(base.cfg, "OBS_FIX_USE_TAIL_SAMPLER", False))
    is_train_loader = bool(kwargs.get("shuffle", False))

    if use_tail_sampler and is_train_loader and hasattr(dataset, "lens_df"):
        indices = np.asarray(dataset.indices, dtype=np.int64)
        mu0 = dataset.lens_df.iloc[indices]["mu_0"].to_numpy(dtype=np.float64)
        pivot = float(getattr(base.cfg, "OBS_FIX_TAIL_PIVOT", 6.0))
        alpha = float(getattr(base.cfg, "OBS_FIX_TAIL_SAMPLER_ALPHA", 3.0))
        power = float(getattr(base.cfg, "OBS_FIX_TAIL_SAMPLER_POWER", 1.25))

        denom = max(float(np.percentile(mu0, 98) - pivot), 1e-6)
        tail = np.clip((mu0 - pivot) / denom, 0.0, 1.0)
        weights = 1.0 + alpha * np.power(tail, power)

        kwargs["shuffle"] = False
        kwargs["sampler"] = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

    return TorchDataLoader(dataset, *args, **kwargs)


def install_obsfix_training_patches(args) -> None:
    if not args.no_physics_derived:
        base.add_optional_observables = add_obsfix_physics_observables
    else:
        base.add_optional_observables = _BASE_ADD_OPTIONAL_OBSERVABLES

    if not args.no_tail_loss:
        base.compute_losses = compute_losses_obsfix_tail
    else:
        base.compute_losses = _BASE_COMPUTE_LOSSES

    base.DataLoader = obsfix_tail_weighted_dataloader
    base.save_val_csv = save_val_csv_obsfix


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
    optical_input_enabled = False
    cfg.OBS_FIX_USE_PHYSICS_DERIVED = False
    cfg.OBS_FIX_DISABLE_OPTICAL_OUTPUT = True
    cfg.OBS_FIX_USE_TAIL_SAMPLER = not args.no_tail_sampler
    cfg.OBS_FIX_TAIL_PIVOT = float(args.tail_pivot)
    cfg.OBS_FIX_TAIL_SAMPLER_ALPHA = 3.0
    cfg.OBS_FIX_TAIL_SAMPLER_POWER = 1.25
    cfg.OBS_FIX_TAIL_LOSS_ALPHA = 1.25
    cfg.OBS_FIX_TAIL_LOSS_POWER = 1.35
    cfg.OBS_FIX_UNDERPRED_W = 0.035

    # These booleans are printed by the base main() and remain semantically
    # useful, although selected_obs_cols is replaced below.
    cfg.USE_OPTICAL_REDSHIFT = optical_input_enabled
    cfg.USE_OPTICAL_SIGMA_V = optical_input_enabled
    cfg.USE_OPTICAL_IMAGE_GEOMETRY = optical_input_enabled
    cfg.USE_OPTICAL_DISTANCE = False

    # Keep oracle switches off.  The new real-observation columns are observed
    # proxies, not the truth columns guarded by these switches.
    cfg.USE_ORACLE_TD = False
    cfg.USE_ORACLE_BETA = False
    cfg.USE_ORACLE_R_ABS = False
    cfg.USE_ORACLE_AMP_RATIO = False
    cfg.USE_ORACLE_Y = False

    # Slightly larger tabular/wave capacity helps after adding nonlinear SIS
    # geometry features, while the tail-aware sampler keeps the rare high-mu0
    # cases visible during training.
    cfg.WAVE_WIDTH = max(cfg.WAVE_WIDTH, 96)
    cfg.D_MODEL = max(cfg.D_MODEL, 256)
    cfg.OBS_HIDDEN = max(cfg.OBS_HIDDEN, 192)
    cfg.TAIL_ALPHA = max(cfg.TAIL_ALPHA, 0.75)
    cfg.EARLY_STOP_PATIENCE = max(cfg.EARLY_STOP_PATIENCE, 22)
    cfg.MIN_EPOCHS_BEFORE_STOP = max(cfg.MIN_EPOCHS_BEFORE_STOP, 70)

    cfg.CKPT_NAME = f"best_mu_direct_realobs_quality_obsfix_nooptical_{args.preset}.pt"
    cfg.PLOT_NAME = f"mu_direct_realobs_quality_obsfix_nooptical_{args.preset}_scatter.png"
    cfg.CSV_NAME = f"mu_direct_realobs_quality_obsfix_nooptical_{args.preset}_val_predictions.csv"

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
        cols = build_quality_obs_cols(
            preset=cfg.QUALITY_PRESET,
            use_optimal_snr=cfg.USE_OPTIMAL_SNR_FEATURES,
        )
        cols = [col for col in cols if col not in OPTICAL_OBS_COLS]
        return cols

    base.selected_obs_cols = selected_obs_cols_quality


def main():
    args = parse_args()
    install_obsfix_training_patches(args)
    configure_base(args)
    print("=" * 90)
    print("Obsfix tail-physics quality realobs mu0 training entry")
    print(f"Data root: {base.cfg.DATA_ROOT}")
    print(f"Out dir:   {base.cfg.OUT_DIR}")
    print(f"Preset:    {base.cfg.QUALITY_PRESET}")
    print(f"Optical input enabled: {not base.cfg.OBS_FIX_DISABLE_OPTICAL_OUTPUT}")
    print(f"Physics-derived image features: {base.cfg.OBS_FIX_USE_PHYSICS_DERIVED}")
    print(f"Tail sampler: {base.cfg.OBS_FIX_USE_TAIL_SAMPLER}")
    print(f"Tail-aware loss: {not args.no_tail_loss}")
    print(f"Tail pivot mu0: {base.cfg.OBS_FIX_TAIL_PIVOT}")
    print(f"Obs cols:  {len(base.selected_obs_cols())}")
    print("Selected columns:")
    for col in base.selected_obs_cols():
        print(f"  - {col}")
    print("=" * 90)
    base.main()


if __name__ == "__main__":
    main()
