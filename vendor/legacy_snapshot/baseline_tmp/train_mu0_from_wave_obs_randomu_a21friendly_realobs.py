import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# 配置
# ============================================================
@dataclass
class Config:
    # ======================================================
    # 基础运行设置
    # ======================================================
    # 随机种子。建议固定，便于复现实验；常用范围：任意整数。
    SEED: int = 42

    # 是否优先使用 GPU；服务器无 CUDA 时会自动回退 CPU。
    USE_CUDA: bool = True

    # ======================================================
    # 数据路径
    # ======================================================
    # DATA_ROOT 对应 SIS_GW_events_randomu_a21friendly_realobs.py 的 SAVE_DIR。
    DATA_ROOT: str = "/root/autodl-tmp/tmp/数据生成/data_lens_randomu_a21friendly_realobs"
    LENS_CSV: str = "/root/autodl-tmp/tmp/数据生成/data_lens_randomu_a21friendly_realobs/lens.csv"
    LENS_PARAMS_CSV: str = "/root/autodl-tmp/tmp/数据生成/data_lens_randomu_a21friendly_realobs/lens_params.csv"
    OBS_CSV_NOISY_WHITE: str = "/root/autodl-tmp/tmp/数据生成/data_lens_randomu_a21friendly_realobs/observable_features.csv"
    OBS_CSV_CLEAN_WHITE: str = "/root/autodl-tmp/tmp/数据生成/data_lens_randomu_a21friendly_realobs/observable_features_clean.csv"
    OBS_CSV_NOISY_RAW: str = "/root/autodl-tmp/tmp/数据生成/data_lens_randomu_a21friendly_realobs/observable_features_raw.csv"
    OBS_CSV_CLEAN_RAW: str = "/root/autodl-tmp/tmp/数据生成/data_lens_randomu_a21friendly_realobs/observable_features_clean_raw.csv"

    # 输出目录。建议不同实验改成不同目录，避免覆盖 best checkpoint。
    OUT_DIR: str = "/root/autodl-tmp/tmp/runs/mu_direct_randomu_a21friendly_realobs_waveobs"

    # ======================================================
    # 波形输入开关
    # ======================================================
    # True: 使用加噪后的观测波形 SIS_data_strain_*.npy，更接近现实。
    # False: 使用加噪前 clean 波形 SIS_h_strain_*.npy，用于理想上限/消融实验。
    USE_NOISY_WAVE: bool = True

    # True: 使用白化波形；False: 使用 raw 未白化时域波形。
    # 建议默认 True。raw 版本幅度动态范围更大，RAW_SCALE 通常需要重调。
    USE_WHITENED_WAVE: bool = True

    # 波形标准化模式：
    # "pair"  : 两条像共用同一个 mean/std，保留相对振幅；推荐默认。
    # "fixed" : 不做样本 std 归一化，直接除以固定 RAW_SCALE，更多保留绝对振幅。
    # "none"  : 直接输入原波形，通常不稳定，只建议调试。
    WAVE_NORM_MODE: str = "pair"

    # 固定幅度压缩尺度，用于第二个波形通道 tanh(x / RAW_SCALE)。
    # 这个通道保留绝对振幅信息；如果几乎全饱和，调大；如果数值太小，调小。
    # 白化数据常试：0.003, 0.01, 0.03, 0.1；raw 数据需要按实际 RMS 重调。
    RAW_SCALE: float = 0.01

    # ======================================================
    # 光学辅助输入开关
    # ======================================================
    # 默认加入可观测透镜/光学量：红移、速度弥散、Einstein 半径/像间距。
    # 这些是观测量，不是 y/u/mu/R/amp_ratio_true 等标签侧变量。

    # 红移开关：加入 z_s_observed, z_l_observed 及其变换。
    # 红移通常可由光学谱线测量；建议做 GW+optical 实验时优先打开。
    USE_OPTICAL_REDSHIFT: bool = True

    # 速度弥散开关：加入 sigma_v_observed。
    # sigma_v 不是 GW 直接观测量；只有模拟光学光谱测量透镜星系时才打开。
    USE_OPTICAL_SIGMA_V: bool = True

    # 成像几何开关：加入 theta_E_observed / image_separation_observed。
    # 需要光学成像识别透镜像；对 SIS 几何很强，打开后任务会明显更容易。
    USE_OPTICAL_IMAGE_GEOMETRY: bool = True

    # 距离开关：加入 source_luminosity_distance。
    # 真实中通常不是直接精确观测量，除非你明确模拟有外部距离约束。
    USE_OPTICAL_DISTANCE: bool = False

    # 当前 SIS_optimal_SNR_*.npy 是由干净注入 response 计算的 optimal SNR，
    # 在 A21-friendly 数据里容易退化成答案型特征。默认不使用。
    USE_OPTIMAL_SNR_FEATURES: bool = False

    # ======================================================
    # Oracle / 消融实验开关
    # ======================================================
    # 下面两项不是严格 GW 可观测输入，默认关闭。
    # 打开它们只用于判断“如果给模型理论真值/近真值，性能上限能到哪里”。

    # 理论时间延迟真值 t_d。真实训练不要打开；可和 peak_time_diff 对照。
    USE_ORACLE_TD: bool = False

    # 源在透镜平面的真值 beta。它和 y/mu 强相关，属于明显标签侧信息。
    USE_ORACLE_BETA: bool = False

    # 理论绝对放大率比 R = |mu1| / mu0，也保存为 mu_ratio_abs_21/R_abs_21。
    # 对 SIS 来说 mu0 = 2 / (1 - R)，几乎等价于直接给答案。
    USE_ORACLE_R_ABS: bool = False

    # 理论两像振幅比 A21 = sqrt(|mu1| / mu0) = sqrt(R)。
    # 对 SIS 来说 mu0 = 2 / (1 - A21^2)，也是强 oracle 特征。
    USE_ORACLE_AMP_RATIO: bool = False

    # 理论归一化源位置 y = beta / theta_E。
    # 对 SIS 来说 mu0 = 1 + 1 / y，等价于直接给标签侧变量。
    USE_ORACLE_Y: bool = False

    # 防止误把标签侧理论量作为输入。做 oracle/上限消融时才手动改成 True。
    ALLOW_ORACLE_FEATURES: bool = False

    # ======================================================
    # 波形裁剪设置
    # ======================================================
    # 每条波形裁剪/补零后的长度。4096 点约等于 1 秒白化片段。
    # 可试：2048 更快但信息少；8192 更慢但保留更长上下文。
    TARGET_LEN: int = 4096

    # 降采样步长。1 表示不降采样；2/4 可提速但损失高频细节。
    STRIDE: int = 1

    # 推荐目标：log(mu0 - 1)
    # 因为 mu0 = 1 + 1/y，所以 log(mu0-1) = log(1/y)
    # 可选："log_mu0_minus1" 推荐；"log_mu0" 更温和；"mu0" 直接回归较难。
    TARGET_MODE: str = "log_mu0_minus1"

    # ======================================================
    # 模型容量
    # ======================================================
    # 激活函数："relu" 更稳更快；其他值使用 SiLU，可能更平滑但略慢。
    ACTIVATION: str = "relu"

    # 1D CNN 基础通道宽度。常试：32 小模型、64 默认、96/128 大模型。
    WAVE_WIDTH: int = 64

    # 融合 MLP 宽度。常试：128, 192, 256。
    D_MODEL: int = 192

    # 观测特征 MLP 隐层宽度。常试：64, 128, 256。
    OBS_HIDDEN: int = 128

    # 注意：这里不要加入 mu_0、mu_1、u_true、log_u_true、y_true、t_d、
    # beta_true、theta_E_true、sigma_v_true 等理论真值。
    # 默认使用 noisy waveform 提取的可观测统计量，保留 A21 相关的波形幅度信息；
    # 但不默认使用 optimal SNR，因为当前它来自干净注入 response。
    BASE_OBS_COLS: List[str] = field(default_factory=lambda: [
        "peak_time_diff", "env_peak_time_diff",

        "peak_amp_ratio_21", "rms_ratio_21", "energy_ratio_21",

        "xcorr_lag", "norm_xcorr_max",
        "xcorr_lag_win65", "norm_xcorr_max_win65",

        "peak_amp_ratio_local_65", "rms_ratio_local_65", "energy_ratio_local_65",

        "env_area_ratio_21",

        "aligned_l1_residual", "aligned_l2_residual",

        "spec_centroid_diff",
        "band_energy_ratio_low_21",
        "band_energy_ratio_mid_21",
        "band_energy_ratio_high_21",
    ])

    OPTIMAL_SNR_COLS: List[str] = field(default_factory=lambda: [
        "snr_1", "snr_2", "snr_ratio_21", "snr_sum", "snr_diff",
    ])

    # USE_OPTICAL_REDSHIFT=True 时追加。
    OPTICAL_REDSHIFT_COLS: List[str] = field(default_factory=lambda: [
        "z_s_observed", "z_l_observed", "z_l_over_z_s",
        "log1p_z_s_observed", "log1p_z_l_observed",
    ])

    # USE_OPTICAL_SIGMA_V=True 时追加。
    OPTICAL_SIGMA_V_COLS: List[str] = field(default_factory=lambda: [
        "sigma_v_observed", "log_sigma_v_observed",
    ])

    # USE_OPTICAL_IMAGE_GEOMETRY=True 时追加。
    OPTICAL_IMAGE_GEOMETRY_COLS: List[str] = field(default_factory=lambda: [
        "theta_E_observed", "image_separation_observed",
    ])

    # USE_OPTICAL_DISTANCE=True 时追加。
    OPTICAL_DISTANCE_COLS: List[str] = field(default_factory=lambda: [
        "source_luminosity_distance",
    ])

    # USE_ORACLE_TD=True 时追加。
    ORACLE_TD_COLS: List[str] = field(default_factory=lambda: [
        "t_d",
    ])

    # USE_ORACLE_BETA=True 时追加。
    ORACLE_BETA_COLS: List[str] = field(default_factory=lambda: [
        "beta_true",
    ])

    # USE_ORACLE_R_ABS=True 时追加。R_abs_21 和 mu_ratio_abs_21 是同义列；
    # 这里选 R_abs_21，名字更贴近公式 R = |mu1| / mu0。
    ORACLE_R_ABS_COLS: List[str] = field(default_factory=lambda: [
        "R_abs_21",
    ])

    # USE_ORACLE_AMP_RATIO=True 时追加。
    ORACLE_AMP_RATIO_COLS: List[str] = field(default_factory=lambda: [
        "amp_ratio_21_true",
    ])

    # USE_ORACLE_Y=True 时追加。
    ORACLE_Y_COLS: List[str] = field(default_factory=lambda: [
        "y_true",
    ])

    # ======================================================
    # 训练超参数
    # ======================================================
    # 最大训练轮数。配合 early stopping 使用；常试：80, 120, 160, 240。
    EPOCHS: int = 160

    # batch size。显存不够降到 16；显存宽裕可试 64。
    BATCH_SIZE: int = 32

    # 初始学习率。AdamW 常试：3e-5, 1e-4, 3e-4。
    LR: float = 1e-4

    # 权重衰减。常试：1e-5, 1e-4, 5e-4, 1e-3。
    WEIGHT_DECAY: float = 5e-4

    # EMA 模型平滑系数。常试：0.99, 0.995, 0.999。
    EMA_DECAY: float = 0.995

    # 数据划分比例。小数据集可用 0.8/0.2；大数据集可用 0.9/0.1。
    SPLIT: Dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.2})

    # 数据增强默认关闭，因为当前任务依赖幅度和时间延迟。
    # 如打开，只建议轻微 roll/noise，避免破坏物理幅度关系。
    USE_AUGMENTATION: bool = False

    # 单个样本应用增强的概率。常试：0.1-0.3。
    AUG_PROB: float = 0.25

    # 同时平移两条像的最大点数；保持两像相对时间结构不变。
    AUG_ROLL_MAX: int = 8

    # 增强噪声强度，越大噪声越弱。常试：50, 100, 200。
    AUG_NOISE_DIV: float = 100.0

    # loss = transformed-space MAE + LOSS_W_MAE * real-space MAE
    #        + LOSS_W_REL * real-space MAPE。
    # 直接预测 mu 时，相对误差更重要；常试 LOSS_W_REL: 0.02-0.2。
    LOSS_W_MAE: float = 0.02
    LOSS_W_REL: float = 0.08

    # 可选：对高放大率样本加权。mu 尾部更难，也更影响 RMSE。
    USE_TAIL_WEIGHT: bool = True

    # 高 mu 样本权重强度。常试：0.0, 0.2, 0.35, 0.5。
    TAIL_ALPHA: float = 0.35

    # early stopping：至少 MIN_EPOCHS_BEFORE_STOP 之后，连续若干轮无提升则停止。
    EARLY_STOP_PATIENCE: int = 14
    MIN_EPOCHS_BEFORE_STOP: int = 50

    # ReduceLROnPlateau 的最低学习率。
    MIN_LR: float = 5e-7

    # 输出文件名。
    CKPT_NAME: str = "best_mu_direct_realobs_waveobs.pt"
    PLOT_NAME: str = "mu_direct_realobs_waveobs_scatter.png"
    CSV_NAME: str = "mu_direct_realobs_waveobs_val_predictions.csv"

    # 数值稳定项。
    EPS: float = 1e-8


cfg = Config()


# ============================================================
# 工具函数
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_out_dir():
    os.makedirs(cfg.OUT_DIR, exist_ok=True)


def resolve_data_path(path: str) -> str:
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    cand = os.path.join(cfg.DATA_ROOT, base)
    return cand if os.path.exists(cand) else path


def selected_obs_cols() -> List[str]:
    cols = list(cfg.BASE_OBS_COLS)
    if cfg.USE_OPTIMAL_SNR_FEATURES:
        cols += cfg.OPTIMAL_SNR_COLS
    if cfg.USE_OPTICAL_REDSHIFT:
        cols += cfg.OPTICAL_REDSHIFT_COLS
    if cfg.USE_OPTICAL_SIGMA_V:
        cols += cfg.OPTICAL_SIGMA_V_COLS
    if cfg.USE_OPTICAL_IMAGE_GEOMETRY:
        cols += cfg.OPTICAL_IMAGE_GEOMETRY_COLS
    if cfg.USE_OPTICAL_DISTANCE:
        cols += cfg.OPTICAL_DISTANCE_COLS
    if cfg.USE_ORACLE_TD:
        cols += cfg.ORACLE_TD_COLS
    if cfg.USE_ORACLE_BETA:
        cols += cfg.ORACLE_BETA_COLS
    if cfg.USE_ORACLE_R_ABS:
        cols += cfg.ORACLE_R_ABS_COLS
    if cfg.USE_ORACLE_AMP_RATIO:
        cols += cfg.ORACLE_AMP_RATIO_COLS
    if cfg.USE_ORACLE_Y:
        cols += cfg.ORACLE_Y_COLS
    return cols


def validate_no_oracle_leakage(obs_cols: List[str]):
    enabled_oracles = []
    if cfg.USE_ORACLE_TD:
        enabled_oracles.append("USE_ORACLE_TD")
    if cfg.USE_ORACLE_BETA:
        enabled_oracles.append("USE_ORACLE_BETA")
    if cfg.USE_ORACLE_R_ABS:
        enabled_oracles.append("USE_ORACLE_R_ABS")
    if cfg.USE_ORACLE_AMP_RATIO:
        enabled_oracles.append("USE_ORACLE_AMP_RATIO")
    if cfg.USE_ORACLE_Y:
        enabled_oracles.append("USE_ORACLE_Y")

    label_side_cols = {
        "mu_0", "mu_1", "abs_mu_0", "abs_mu_1", "mu_total_abs",
        "mu_ratio_abs_21", "R_abs_21", "amp_ratio_21", "amp_ratio_21_true",
        "u", "log_u", "y", "u_true", "log_u_true", "y_true",
        "beta", "beta_true", "inv_beta", "log_beta", "log_inv_beta",
        "beta_x", "beta_y", "image_x_0", "image_y_0", "image_x_1", "image_y_1",
        "t_d", "theta_E_true", "sigma_v_true",
        "z_l_true", "z_s_true",
    }
    leaked_cols = sorted(set(obs_cols) & label_side_cols)
    privileged_cols = []
    if not cfg.USE_OPTIMAL_SNR_FEATURES:
        privileged_cols = sorted(set(obs_cols) & set(cfg.OPTIMAL_SNR_COLS))

    if cfg.ALLOW_ORACLE_FEATURES:
        return

    if enabled_oracles or leaked_cols or privileged_cols:
        details = []
        if enabled_oracles:
            details.append(f"enabled oracle switches: {enabled_oracles}")
        if leaked_cols:
            details.append(f"label-side input columns: {leaked_cols}")
        if privileged_cols:
            details.append(f"privileged optimal-SNR columns: {privileged_cols}")
        raise RuntimeError(
            "Potential target leakage detected. "
            + "; ".join(details)
            + ". Set ALLOW_ORACLE_FEATURES=True only for explicit oracle ablation."
        )


def resolve_wave_paths():
    if cfg.USE_WHITENED_WAVE:
        prefix = "SIS_data_strain" if cfg.USE_NOISY_WAVE else "SIS_h_strain"
    else:
        prefix = "SIS_data_strain_raw" if cfg.USE_NOISY_WAVE else "SIS_h_strain_raw"

    return (
        resolve_data_path(os.path.join(cfg.DATA_ROOT, f"{prefix}_1.npy")),
        resolve_data_path(os.path.join(cfg.DATA_ROOT, f"{prefix}_2.npy")),
        prefix,
    )


def resolve_obs_path():
    if cfg.USE_NOISY_WAVE and cfg.USE_WHITENED_WAVE:
        path = resolve_data_path(cfg.OBS_CSV_NOISY_WHITE)
        expected_kind = "realobs_a21friendly_noisy_whitened"
    elif (not cfg.USE_NOISY_WAVE) and cfg.USE_WHITENED_WAVE:
        path = resolve_data_path(cfg.OBS_CSV_CLEAN_WHITE)
        expected_kind = "clean_whitened"
    elif cfg.USE_NOISY_WAVE and (not cfg.USE_WHITENED_WAVE):
        path = resolve_data_path(cfg.OBS_CSV_NOISY_RAW)
        expected_kind = "realobs_a21friendly_noisy_raw"
    else:
        path = resolve_data_path(cfg.OBS_CSV_CLEAN_RAW)
        expected_kind = "clean_raw"

    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing observable feature table for {expected_kind}: {path}. "
            "Please regenerate data with 数据生成/SIS_GW_events_randomu_a21friendly_realobs.py."
        )

    return path, expected_kind


def add_optional_observables(obs_df: pd.DataFrame, lens_params_df: pd.DataFrame) -> pd.DataFrame:
    obs_df = obs_df.copy()

    def get_col(preferred, fallback=None):
        if preferred in lens_params_df.columns:
            return lens_params_df[preferred].to_numpy(dtype=np.float32)
        if fallback is not None and fallback in lens_params_df.columns:
            return lens_params_df[fallback].to_numpy(dtype=np.float32)
        raise RuntimeError(f"Missing column in lens_params.csv: {preferred}")

    z_s = get_col("z_s_observed", "z_s")
    z_l = get_col("z_l_observed", "z_l")
    sigma_v = get_col("sigma_v_observed", "sigma_v")

    fill_values = {
        "z_s_observed": z_s,
        "z_l_observed": z_l,
        "z_l_over_z_s": z_l / np.maximum(z_s, cfg.EPS),
        "log1p_z_s_observed": np.log1p(np.maximum(z_s, 0.0)),
        "log1p_z_l_observed": np.log1p(np.maximum(z_l, 0.0)),
        "sigma_v_observed": sigma_v,
        "log_sigma_v_observed": np.log(np.maximum(sigma_v, cfg.EPS)),
    }

    if "theta_E_observed" in lens_params_df.columns:
        fill_values["theta_E_observed"] = lens_params_df["theta_E_observed"].to_numpy(dtype=np.float32)
    elif "theta_E_observed(arcsec)" in lens_params_df.columns:
        fill_values["theta_E_observed"] = lens_params_df["theta_E_observed(arcsec)"].to_numpy(dtype=np.float32)
    elif "theta_E(arcsec)" in lens_params_df.columns:
        fill_values["theta_E_observed"] = lens_params_df["theta_E(arcsec)"].to_numpy(dtype=np.float32)

    if "image_separation_observed" in lens_params_df.columns:
        fill_values["image_separation_observed"] = lens_params_df["image_separation_observed"].to_numpy(dtype=np.float32)
    elif "image_separation_observed(arcsec)" in lens_params_df.columns:
        fill_values["image_separation_observed"] = lens_params_df["image_separation_observed(arcsec)"].to_numpy(dtype=np.float32)
    elif "image_separation(arcsec)" in lens_params_df.columns:
        fill_values["image_separation_observed"] = lens_params_df["image_separation(arcsec)"].to_numpy(dtype=np.float32)

    if "source_luminosity_distance" in lens_params_df.columns:
        fill_values["source_luminosity_distance"] = lens_params_df["source_luminosity_distance"].to_numpy(dtype=np.float32)

    for col, values in fill_values.items():
        if col not in obs_df.columns:
            obs_df[col] = values

    return obs_df


def get_device():
    if cfg.USE_CUDA and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pad_or_trim(x: np.ndarray, target_len: int, stride: int):
    x = x.astype(np.float32)
    n = x.shape[-1]

    if n >= target_len:
        y = x[-target_len:]
    else:
        y = np.pad(x, (target_len - n, 0), mode="constant")

    if stride > 1:
        y = y[::stride]

    return y


def build_split_indices(n: int):
    idx = np.arange(n)
    rng = np.random.RandomState(cfg.SEED)
    rng.shuffle(idx)
    n_train = int(n * cfg.SPLIT["train"])
    return idx[:n_train], idx[n_train:]


def target_transform_np(mu0: np.ndarray):
    mu0 = np.asarray(mu0, dtype=np.float32)

    if cfg.TARGET_MODE == "log_mu0_minus1":
        return np.log(np.maximum(mu0 - 1.0, cfg.EPS)).astype(np.float32)

    if cfg.TARGET_MODE == "log_mu0":
        return np.log(np.maximum(mu0, cfg.EPS)).astype(np.float32)

    if cfg.TARGET_MODE == "mu0":
        return mu0.astype(np.float32)

    raise ValueError(f"Unsupported TARGET_MODE: {cfg.TARGET_MODE}")


def target_inverse_torch(pred: torch.Tensor):
    if cfg.TARGET_MODE == "log_mu0_minus1":
        return 1.0 + torch.exp(pred)

    if cfg.TARGET_MODE == "log_mu0":
        return torch.exp(pred)

    if cfg.TARGET_MODE == "mu0":
        return pred

    raise ValueError(f"Unsupported TARGET_MODE: {cfg.TARGET_MODE}")


def target_inverse_np(pred: np.ndarray):
    pred = np.asarray(pred, dtype=np.float64)

    if cfg.TARGET_MODE == "log_mu0_minus1":
        return 1.0 + np.exp(pred)

    if cfg.TARGET_MODE == "log_mu0":
        return np.exp(pred)

    if cfg.TARGET_MODE == "mu0":
        return pred

    raise ValueError(f"Unsupported TARGET_MODE: {cfg.TARGET_MODE}")


# ============================================================
# 标准化器
# ============================================================
class FeatureNormalizer:
    def __init__(self, columns):
        self.columns = list(columns)
        self.mean_ = None
        self.std_ = None

    def fit(self, df: pd.DataFrame):
        arr = df[self.columns].to_numpy(dtype=np.float32)
        self.mean_ = arr.mean(axis=0)
        self.std_ = arr.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform_row(self, row: pd.Series):
        arr = row[self.columns].to_numpy(dtype=np.float32)
        return (arr - self.mean_) / self.std_

    def state_dict(self):
        return {
            "columns": np.array(self.columns, dtype=object),
            "mean": self.mean_,
            "std": self.std_,
        }


# ============================================================
# 数据集
# ============================================================
class MuDirectCompactDataset(Dataset):
    def __init__(self, wave1, wave2, lens_df, obs_df, obs_norm, indices, mode="train"):
        self.wave1 = wave1
        self.wave2 = wave2
        self.lens_df = lens_df
        self.obs_df = obs_df
        self.obs_norm = obs_norm
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mode = mode

    def __len__(self):
        return len(self.indices)

    def build_wave_pair(self, x1: np.ndarray, x2: np.ndarray):
        x1 = pad_or_trim(x1, cfg.TARGET_LEN, cfg.STRIDE)
        x2 = pad_or_trim(x2, cfg.TARGET_LEN, cfg.STRIDE)

        # 同一事件的两条像共用同一个尺度，避免相对振幅被单独标准化洗掉。
        pair = np.concatenate([x1, x2], axis=0)
        pair_mean = pair.mean()
        pair_std = pair.std() + 1e-8

        def build_one(x):
            if cfg.WAVE_NORM_MODE == "pair":
                shape_ch = (x - pair_mean) / pair_std
            elif cfg.WAVE_NORM_MODE == "fixed":
                shape_ch = x / cfg.RAW_SCALE
            elif cfg.WAVE_NORM_MODE == "none":
                shape_ch = x
            else:
                raise ValueError(f"Unsupported WAVE_NORM_MODE: {cfg.WAVE_NORM_MODE}")

            # 固定尺度幅度通道：保留绝对振幅信息，避免只剩相对幅度。
            amp_ch = np.tanh(x / cfg.RAW_SCALE)
            return np.stack([shape_ch, amp_ch], axis=0).astype(np.float32)

        return build_one(x1), build_one(x2)

    def __getitem__(self, item):
        idx = int(self.indices[item])

        x1 = np.array(self.wave1[idx], copy=True)
        x2 = np.array(self.wave2[idx], copy=True)

        if cfg.USE_AUGMENTATION and self.mode == "train" and np.random.rand() < cfg.AUG_PROB:
            shift = np.random.randint(-cfg.AUG_ROLL_MAX, cfg.AUG_ROLL_MAX + 1)
            x1 = np.roll(x1, shift)
            x2 = np.roll(x2, shift)

            noise1 = np.random.randn(*x1.shape) * (np.std(x1) / cfg.AUG_NOISE_DIV)
            noise2 = np.random.randn(*x2.shape) * (np.std(x2) / cfg.AUG_NOISE_DIV)

            x1 = x1 + noise1
            x2 = x2 + noise2

        w1, w2 = self.build_wave_pair(x1, x2)

        obs_vec = self.obs_norm.transform_row(self.obs_df.iloc[idx]).astype(np.float32)

        mu0 = float(self.lens_df.iloc[idx]["mu_0"])
        mu1 = float(self.lens_df.iloc[idx]["mu_1"])

        y_target = target_transform_np(np.array([mu0], dtype=np.float32))

        return (
            torch.from_numpy(w1),
            torch.from_numpy(w2),
            torch.from_numpy(obs_vec),
            torch.from_numpy(y_target),
            torch.tensor([mu0], dtype=torch.float32),
            torch.tensor([mu1], dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long),
        )


# ============================================================
# 模型结构
# ============================================================
def build_activation():
    if cfg.ACTIVATION.lower() == "relu":
        return nn.ReLU(inplace=True)
    return nn.SiLU()


class WaveEncoder(nn.Module):
    def __init__(self, in_ch=2, width=64):
        super().__init__()
        act = build_activation()

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, 15, stride=2, padding=7),
            nn.BatchNorm1d(width),
            act,
            nn.MaxPool1d(2),

            nn.Conv1d(width, width * 2, 11, stride=2, padding=5),
            nn.BatchNorm1d(width * 2),
            act,

            nn.Conv1d(width * 2, width * 4, 9, stride=2, padding=4),
            nn.BatchNorm1d(width * 4),
            act,

            nn.Conv1d(width * 4, width * 4, 7, stride=2, padding=3),
            nn.BatchNorm1d(width * 4),
            act,

            nn.AdaptiveAvgPool1d(1),
        )

        self.out_dim = width * 4

    def forward(self, x):
        return self.net(x).flatten(1)


class MuDirectCompactRegressor(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        act = build_activation()

        self.wave_encoder = WaveEncoder(in_ch=2, width=cfg.WAVE_WIDTH)

        self.obs_head = nn.Sequential(
            nn.Linear(obs_dim, cfg.OBS_HIDDEN),
            act,
            nn.Linear(cfg.OBS_HIDDEN, cfg.D_MODEL // 2),
            act,
        )

        wave_dim = self.wave_encoder.out_dim
        fusion_dim = wave_dim * 4 + (cfg.D_MODEL // 2)

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, cfg.D_MODEL),
            act,
            nn.Dropout(0.15),
            nn.Linear(cfg.D_MODEL, cfg.D_MODEL // 2),
            act,
            nn.Linear(cfg.D_MODEL // 2, 1),
        )

    def forward(self, w1, w2, obs):
        f1 = self.wave_encoder(w1)
        f2 = self.wave_encoder(w2)

        fd = torch.abs(f1 - f2)
        fp = f1 * f2
        fo = self.obs_head(obs)

        fused = torch.cat([f1, f2, fd, fp, fo], dim=1)

        # 输出的是 transformed target，比如 log(mu0-1)
        return self.head(fused)


# ============================================================
# loss
# ============================================================
def compute_losses(pred_target, target_transformed, mu0_true):
    """
    pred_target: 模型输出的 transformed target
    target_transformed: 真实 transformed target
    mu0_true: 真实 mu0
    """
    mu0_pred = target_inverse_torch(pred_target)

    # transformed space 的 MAE，保证训练稳定
    trans_mae = torch.abs(pred_target - target_transformed).mean()

    # real space 的 MAE/MAPE，直接约束最终 mu0
    abs_err = torch.abs(mu0_pred - mu0_true)
    rel_err = abs_err / (torch.abs(mu0_true) + cfg.EPS)

    if cfg.USE_TAIL_WEIGHT:
        # 高 mu 区域更难，也对最终 RMSE 影响更大
        # 这里用 log 权重，避免过度偏向极端尾部
        weight = 1.0 + cfg.TAIL_ALPHA * torch.log1p(torch.abs(mu0_true))
        abs_err = abs_err * weight
        rel_err = rel_err * weight

    mae = abs_err.mean()
    mape = rel_err.mean() * 100.0

    loss = trans_mae + cfg.LOSS_W_MAE * mae + cfg.LOSS_W_REL * mape

    return loss, mae, mape


# ============================================================
# 评估
# ============================================================
def compute_metrics(pred, true, name):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)

    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    mape = np.mean(np.abs((pred - true) / (np.abs(true) + 1e-8))) * 100.0

    corr = np.corrcoef(pred, true)[0, 1] if len(pred) > 1 else np.nan

    return {
        "name": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "Corr": float(corr),
    }


def print_metric(m):
    print(
        f"[{m['name']}] "
        f"MAE={m['MAE']:.6f} | "
        f"RMSE={m['RMSE']:.6f} | "
        f"MAPE={m['MAPE']:.2f}% | "
        f"Corr={m['Corr']:.4f}"
    )


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()

    all_mu0_pred = []
    all_mu1_pred = []
    all_mu0_true = []
    all_mu1_true = []
    all_idx = []

    total_loss = 0.0
    total_mae = 0.0
    total_mape = 0.0

    for w1, w2, obs, y_trans, mu0_true, mu1_true, idx in tqdm(loader, desc="Validation", leave=False):
        w1 = w1.to(device)
        w2 = w2.to(device)
        obs = obs.to(device)
        y_trans = y_trans.to(device)
        mu0_true = mu0_true.to(device)

        pred_trans = model(w1, w2, obs)
        loss, mae, mape = compute_losses(pred_trans, y_trans, mu0_true)

        mu0_pred = target_inverse_torch(pred_trans)

        # SIS 物理关系：mu0 + mu1 = 2
        mu1_pred = 2.0 - mu0_pred

        total_loss += loss.item()
        total_mae += mae.item()
        total_mape += mape.item()

        all_mu0_pred.append(mu0_pred.cpu().numpy().reshape(-1))
        all_mu1_pred.append(mu1_pred.cpu().numpy().reshape(-1))
        all_mu0_true.append(mu0_true.cpu().numpy().reshape(-1))
        all_mu1_true.append(mu1_true.numpy().reshape(-1))
        all_idx.append(idx.numpy().reshape(-1))

    mu0_pred = np.concatenate(all_mu0_pred)
    mu1_pred = np.concatenate(all_mu1_pred)
    mu0_true = np.concatenate(all_mu0_true)
    mu1_true = np.concatenate(all_mu1_true)
    idx_all = np.concatenate(all_idx)

    n_batches = len(loader)

    return {
        "val_loss": total_loss / n_batches,
        "val_mae_loss": total_mae / n_batches,
        "val_mape_loss": total_mape / n_batches,

        "mu0_pred": mu0_pred,
        "mu1_pred": mu1_pred,
        "mu0_true": mu0_true,
        "mu1_true": mu1_true,
        "idx": idx_all,

        "mu0_metrics": compute_metrics(mu0_pred, mu0_true, "mu0 direct"),
        "mu1_metrics": compute_metrics(mu1_pred, mu1_true, "mu1 from SIS"),
    }


def update_ema_model(ema_model, model, decay):
    msd = model.state_dict()
    for k, v in ema_model.state_dict().items():
        v.copy_(v * decay + msd[k].detach() * (1.0 - decay))


def train_one_epoch(model, ema_model, loader, opt, device, desc):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_mape = 0.0

    for w1, w2, obs, y_trans, mu0_true, _, _ in tqdm(loader, desc=desc, leave=False):
        w1 = w1.to(device)
        w2 = w2.to(device)
        obs = obs.to(device)
        y_trans = y_trans.to(device)
        mu0_true = mu0_true.to(device)

        opt.zero_grad()

        pred_trans = model(w1, w2, obs)
        loss, mae, mape = compute_losses(pred_trans, y_trans, mu0_true)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        update_ema_model(ema_model, model, cfg.EMA_DECAY)

        total_loss += loss.item()
        total_mae += mae.item()
        total_mape += mape.item()

    n_batches = len(loader)

    return (
        total_loss / n_batches,
        total_mae / n_batches,
        total_mape / n_batches,
    )


# ============================================================
# 画图
# ============================================================
def plot_results(eval_out, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mu0_true = eval_out["mu0_true"]
    mu0_pred = eval_out["mu0_pred"]
    mu1_true = eval_out["mu1_true"]
    mu1_pred = eval_out["mu1_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pairs = [
        (mu0_true, mu0_pred, r"$\mu_0$"),
        (mu1_true, mu1_pred, r"$\mu_1$"),
    ]

    for ax, (true, pred, name) in zip(axes, pairs):
        ax.scatter(true, pred, s=8, alpha=0.55)

        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())

        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Pred {name}")
        ax.set_title(f"Direct prediction: {name}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_val_csv(eval_out, save_path, obs_df=None):
    df = pd.DataFrame({
        "event_id": eval_out["idx"],
        "mu0_true": eval_out["mu0_true"],
        "mu0_pred": eval_out["mu0_pred"],
        "mu1_true": eval_out["mu1_true"],
        "mu1_pred": eval_out["mu1_pred"],
    })

    if obs_df is not None:
        event_idx = df["event_id"].to_numpy(dtype=np.int64)
        for col in [
            "z_s_observed", "z_l_observed", "z_l_over_z_s",
            "sigma_v_observed", "theta_E_observed",
            "image_separation_observed", "R_abs_21",
            "u_true", "log_u_true", "y_true",
        ]:
            if col in obs_df.columns:
                df[col] = obs_df.iloc[event_idx][col].to_numpy()

    df["mu0_abs_err"] = np.abs(df["mu0_pred"] - df["mu0_true"])
    df["mu1_abs_err"] = np.abs(df["mu1_pred"] - df["mu1_true"])

    df["mu0_rel_err_percent"] = df["mu0_abs_err"] / (np.abs(df["mu0_true"]) + 1e-8) * 100.0
    df["mu1_rel_err_percent"] = df["mu1_abs_err"] / (np.abs(df["mu1_true"]) + 1e-8) * 100.0

    df.to_csv(save_path, index=False, encoding="utf-8-sig")


# ============================================================
# 主流程
# ============================================================
def main():
    seed_everything(cfg.SEED)
    ensure_out_dir()

    device = get_device()

    print("=" * 90)
    print("Train Direct Mu Compact Regressor - A21-friendly realistic wave+observables")
    print(f"Device:      {device}")
    print(f"Target mode: {cfg.TARGET_MODE}")
    print(f"Out dir:     {cfg.OUT_DIR}")
    print(f"Data root:   {cfg.DATA_ROOT}")

    wave1_path, wave2_path, wave_prefix = resolve_wave_paths()
    obs_path, expected_feature_kind = resolve_obs_path()
    obs_cols = selected_obs_cols()
    validate_no_oracle_leakage(obs_cols)

    print(f"Wave source: {wave_prefix}")
    print(f"Obs table:   {os.path.basename(obs_path)}")
    print(f"Obs kind:    {expected_feature_kind}")
    print(f"Noisy wave:  {cfg.USE_NOISY_WAVE}")
    print(f"Whitened:    {cfg.USE_WHITENED_WAVE}")
    print(f"Wave norm:   {cfg.WAVE_NORM_MODE}")
    print(f"RAW_SCALE:   {cfg.RAW_SCALE}")
    print(f"Use optimal SNR features: {cfg.USE_OPTIMAL_SNR_FEATURES}")
    print(f"Optical redshift:       {cfg.USE_OPTICAL_REDSHIFT}")
    print(f"Optical sigma_v:        {cfg.USE_OPTICAL_SIGMA_V}")
    print(f"Optical image geometry: {cfg.USE_OPTICAL_IMAGE_GEOMETRY}")
    print(f"Optical distance:       {cfg.USE_OPTICAL_DISTANCE}")
    print(f"Oracle t_d:             {cfg.USE_ORACLE_TD}")
    print(f"Oracle beta_true:       {cfg.USE_ORACLE_BETA}")
    print(f"Oracle R_abs_21:        {cfg.USE_ORACLE_R_ABS}")
    print(f"Oracle amp_ratio_21:    {cfg.USE_ORACLE_AMP_RATIO}")
    print(f"Oracle y_true:          {cfg.USE_ORACLE_Y}")
    print(f"Obs dim:     {len(obs_cols)}")
    print("=" * 90)

    wave1 = np.load(wave1_path, mmap_mode="r")
    wave2 = np.load(wave2_path, mmap_mode="r")
    lens_df = pd.read_csv(resolve_data_path(cfg.LENS_CSV))
    lens_params_df = pd.read_csv(resolve_data_path(cfg.LENS_PARAMS_CSV))
    obs_df = pd.read_csv(obs_path)

    valid_len = min(len(wave1), len(wave2), len(lens_df), len(lens_params_df), len(obs_df))

    lens_df = lens_df.iloc[:valid_len].copy().reset_index(drop=True)
    lens_params_df = lens_params_df.iloc[:valid_len].copy().reset_index(drop=True)
    obs_df = obs_df.iloc[:valid_len].copy().reset_index(drop=True)
    obs_df = add_optional_observables(obs_df, lens_params_df)

    if "feature_data_kind" in obs_df.columns:
        kinds = set(obs_df["feature_data_kind"].dropna().astype(str).unique())
        if kinds and kinds != {expected_feature_kind}:
            raise RuntimeError(
                f"Observable feature table kind mismatch: expected "
                f"{expected_feature_kind}, got {sorted(kinds)}"
            )

    for c in ["mu_0", "mu_1"]:
        if c not in lens_df.columns:
            raise RuntimeError(f"Missing column in lens.csv: {c}")

    for c in obs_cols:
        if c not in obs_df.columns:
            raise RuntimeError(f"Missing column in {obs_path}: {c}")

    idx_tr, idx_va = build_split_indices(valid_len)

    obs_norm = FeatureNormalizer(obs_cols).fit(obs_df.iloc[idx_tr])

    tr_ds = MuDirectCompactDataset(
        wave1=wave1,
        wave2=wave2,
        lens_df=lens_df,
        obs_df=obs_df,
        obs_norm=obs_norm,
        indices=idx_tr,
        mode="train",
    )

    va_ds = MuDirectCompactDataset(
        wave1=wave1,
        wave2=wave2,
        lens_df=lens_df,
        obs_df=obs_df,
        obs_norm=obs_norm,
        indices=idx_va,
        mode="val",
    )

    tr_loader = DataLoader(
        tr_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    va_loader = DataLoader(
        va_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = MuDirectCompactRegressor(obs_dim=len(obs_cols)).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    opt = optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.7,
        patience=3,
        min_lr=cfg.MIN_LR,
    )

    best_mape = float("inf")
    best_mae = float("inf")
    wait = 0

    for ep in range(1, cfg.EPOCHS + 1):
        tr_loss, tr_mae, tr_mape = train_one_epoch(
            model=model,
            ema_model=ema_model,
            loader=tr_loader,
            opt=opt,
            device=device,
            desc=f"Epoch {ep}",
        )

        eval_out = evaluate_model(ema_model, va_loader, device)

        mu0_m = eval_out["mu0_metrics"]
        mu1_m = eval_out["mu1_metrics"]

        print(
            f"[Epoch {ep:03d}] "
            f"TrainLoss={tr_loss:.4f} TrainMAE={tr_mae:.6f} TrainMAPE={tr_mape:.2f}% | "
            f"ValLoss={eval_out['val_loss']:.4f} | "
            f"mu0_MAE={mu0_m['MAE']:.6f} mu0_MAPE={mu0_m['MAPE']:.2f}% mu0_Corr={mu0_m['Corr']:.4f} | "
            f"mu1_MAE={mu1_m['MAE']:.6f} mu1_MAPE={mu1_m['MAPE']:.2f}% mu1_Corr={mu1_m['Corr']:.4f} | "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )

        # 主监控：mu0 MAPE；相同情况下看 MAE
        improved = False
        if mu0_m["MAPE"] < best_mape:
            best_mape = mu0_m["MAPE"]
            best_mae = mu0_m["MAE"]
            improved = True
        elif abs(mu0_m["MAPE"] - best_mape) < 1e-6 and mu0_m["MAE"] < best_mae:
            best_mae = mu0_m["MAE"]
            improved = True

        if improved:
            wait = 0

            ckpt_path = os.path.join(cfg.OUT_DIR, cfg.CKPT_NAME)
            torch.save(
                {
                    "model_state": copy.deepcopy(ema_model.state_dict()),
                    "obs_norm": obs_norm.state_dict(),
                    "target_mode": cfg.TARGET_MODE,
                    "obs_cols": obs_cols,
                    "wave_prefix": wave_prefix,
                    "config": cfg.__dict__,
                    "best_mu0_metrics": mu0_m,
                    "best_mu1_metrics": mu1_m,
                },
                ckpt_path,
            )

            plot_path = os.path.join(cfg.OUT_DIR, cfg.PLOT_NAME)
            plot_results(eval_out, plot_path)

            csv_path = os.path.join(cfg.OUT_DIR, cfg.CSV_NAME)
            save_val_csv(eval_out, csv_path, obs_df=obs_df)

            print(f"[Save] Best model saved to: {ckpt_path}")
            print_metric(mu0_m)
            print_metric(mu1_m)

        else:
            wait += 1
            if ep >= cfg.MIN_EPOCHS_BEFORE_STOP and wait >= cfg.EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

        scheduler.step(mu0_m["MAPE"])

    print("Training finished.")


if __name__ == "__main__":
    main()
