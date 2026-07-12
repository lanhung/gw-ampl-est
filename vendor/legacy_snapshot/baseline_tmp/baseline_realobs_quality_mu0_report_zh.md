# 基于真实观测代理特征的 SIS 引力波强透镜放大率预测基线报告

## 摘要

强透镜引力波事件中的两幅像包含由透镜几何、放大率、Morse 相位和时间延迟共同调制的波形信息。对于奇异等温球模型, 第一像放大率 `mu0` 与第二像放大率 `mu1` 满足严格的物理关系 `mu0 + mu1 = 2`, 因而只要能够稳定预测 `mu0`, 就可以由 SIS 关系直接得到 `mu1`。本报告围绕当前 baseline 数据协议和训练代码展开, 重点分析 `数据生成/SIS_GW_events_randomu_baseline_quality_mu0_before_obsfix.py` 的真实观测质量控制数据生成流程, 以及 `train_mu0_baseline_obs_realobs_quality.py` 和其基础训练模块 `train_mu0_from_wave_obs_randomu_a21friendly_realobs.py` 的模型架构、输入特征、训练目标和评估方式。

当前数据生成脚本采用质量控制的真实观测模拟设定: 源参数、透镜红移、速度弥散、天空位置、自旋和外在几何均进行随机采样; 透镜模型使用 SIS 公式; 波形由 `bilby` 和 `IMRPhenomXPHM` 生成; 探测器使用 ET 配置并加入 detector noise; 训练数据保存为以触发峰值为中心的一秒白化观测窗口。与早期课程版 A21-friendly 数据相比, 当前协议显式区分训练可用的观测量、标签真值、诊断性 privileged quantities 和质量控制报告, 更适合评估模型在 noisy real-observation proxy 条件下的可用性。

不过, 当前 baseline 生成脚本并不是直接均匀采样 `mu0`, 而是对两像振幅比 `A21` 做分层均匀采样, 再通过 SIS 公式反推 `y`, `u`, `mu0` 和 `mu1`。由于 `mu0 = 2 / (1 - A21^2)` 是强非线性变换, `A21` 均匀并不等价于 `mu0` 均匀。实际生成的 2500 个样本中, `mu0` 主要集中在 `[2, 4)` 区间, 而 `mu0 >= 10` 的高放大率样本仅约 130 个。这种标签分布不均衡直接导致 baseline 模型在尾端出现系统性低估: 验证集中 `mu0 > 10` 的预测偏差达到约 `-3` 到 `-4`, 散点图表现为高 `mu0` 区间收束性差, `mu1` 负尾端同步回缩。

本文首先介绍物理问题和数据协议设计, 然后详细描述数据生成流程、观测特征体系、训练代码架构、模型结构、损失函数与结果诊断。最后讨论当前 baseline 的局限性, 并提出后续数据协议和训练策略的改进方向, 包括直接按 `mu0` 分层采样、按 accepted count 控制每个尾端区间的样本数、引入分桶加权损失、使用分层划分训练验证集, 以及针对光学几何特征进行泄露风险消融。

## 关键词

引力波强透镜; SIS 模型; 放大率预测; 真实观测代理特征; 质量控制数据生成; 一维卷积网络; 尾部分布不均衡; 回归到均值

## 1. 引言

第三代引力波探测器, 如 Einstein Telescope, 将显著提升远距离双黑洞并合事件的探测能力。随着事件数增加, 强透镜引力波候选事件的识别和参数反演将成为重要问题。对于被星系透镜放大的同一源事件, 不同透镜像共享相同的内禀源参数, 但在振幅、相位、到达时间和探测器响应上存在由透镜几何决定的差异。如何从两幅像的波形和辅助观测量中恢复透镜物理参数, 是强透镜引力波数据分析中的核心任务之一。

本项目目前聚焦于 SIS 双像透镜模型中的放大率预测问题。设第一像为正宇称像, 第二像为负宇称像, 代码中保存的有符号放大率为:

```text
mu0 = mu_0
mu1 = mu_1
```

对于 SIS 双像系统, 它们满足:

```text
mu0 = 1 + 1 / y
mu1 = 1 - 1 / y
mu0 + mu1 = 2
```

其中 `y = beta / theta_E` 是归一化源位置, 双像条件为 `0 < y < 1`。因此, 一旦模型能够预测 `mu0`, 就可以直接得到:

```text
mu1 = 2 - mu0
```

这也是当前训练代码采用的物理约束。

早期实验表明, 如果数据生成过程被构造成 A21-friendly 的理想课程任务, 即两幅像几乎只体现干净的振幅比差异, 模型能够较容易地学到 `mu0`。但这种设置与真实观测存在差距: detector noise、随机天空位置、随机倾角、真实时间延迟引起的探测器响应变化、观测误差和质量筛选都会显著降低从波形振幅比恢复 `mu0` 的稳定性。因此, 当前 baseline 的目标是建立一个更接近真实观测流程的数据协议, 并检验 compact wave+observable 模型在该协议下的表现。

本报告围绕两个问题展开:

1. 当前 baseline 数据生成脚本是否真正实现了目标变量 `mu0` 的均匀覆盖?
2. 当前训练代码在输入、模型结构和损失函数上是否足以应对中部及尾端样本的预测困难?

分析结果显示, 当前数据生成脚本的主要瓶颈是标签分布不均衡。脚本按 `A21` 分层采样, 但 `A21 -> mu0` 是非线性映射, 因而接受后的 `mu0` 分布明显集中在低中值区域。这种数据协议使得模型更容易学好主体区间, 却在高 `mu0` 尾端发生回归到均值。训练脚本虽然使用了 `log(mu0 - 1)` 目标变换和温和的 tail weight, 但不足以抵消尾部样本数量不足带来的系统性偏差。

[图1建议: 项目整体流程图。展示从源参数采样、SIS 透镜采样、双像波形生成、质量控制、观测特征表构建、训练输入、模型预测 `mu0`、由 `mu1 = 2 - mu0` 得到第二像放大率的完整流程。]

## 2. 相关工作与项目演化

### 2.1 从直接回归到物理约束回归

项目早期尝试直接从两条波形回归 `mu0` 和 `mu1` 或二者差值。初始版本采用 1D CNN 处理两通道拼接波形, 以 Huber Loss、相对误差项和 Pearson 相关项构成复合损失。后续 V5 版本转向孪生网络结构, 使用共享编码器分别提取两幅像的特征, 并通过 `[f1, f2, f1 - f2, |f1 - f2|, f1 * f2]` 等关系特征进行融合。这种结构更符合双像任务的对称性和配对性质。

实验记录显示, 单纯提高模型容量并不能稳定解决长尾样本问题。模型在主体区间可以取得较低误差, 但在高放大率尾端仍然容易出现大偏差。后续实验逐渐引入物理参数化思路, 例如预测 `u` 或 `log_u`, 再通过 SIS 公式得到 `mu0` 和 `mu1`。当前 baseline 训练脚本采用的目标是:

```text
target = log(mu0 - 1)
mu0_pred = 1 + exp(model_output)
mu1_pred = 2 - mu0_pred
```

这种目标变换可以压缩 `mu0` 的长尾动态范围, 提升训练稳定性。

### 2.2 A21-friendly 数据与真实观测数据的差异

A21-friendly 数据的核心特点是让两幅像的差异尽量由振幅比决定。理想情况下:

```text
A21 = sqrt(|mu1| / mu0)
mu0 = 2 / (1 - A21^2)
```

如果输入特征中存在干净的两像振幅比或 optimal SNR ratio, 模型几乎可以解析地反推出 `mu0`。这种任务适合作为课程学习或上限实验, 但不能代表真实 noisy detector 数据的难度。

当前 baseline 数据生成脚本试图向真实观测靠近: detector noise 默认打开, 源外在几何随机化, 第二像的 `geocent_time` 默认包含真实时间延迟, 训练窗口从全长 24 秒数据中围绕清洁信号峰值裁剪, 光学红移、速度弥散、Einstein 半径、像位置和到达时间均加入观测误差。同时, 诊断真值被单独写入 `diagnostic_features.csv`, 而训练入口默认关闭 `optimal SNR` 和理论真值列, 以降低标签泄露风险。

### 2.3 目标分布与尾端误差问题

强透镜 SIS 参数之间存在强非线性关系。尤其当 `A21` 接近 1 时:

```text
mu0 = 2 / (1 - A21^2)
```

对 `A21` 的小误差会被放大为 `mu0` 的大误差。其导数为:

```text
dmu0 / dA21 = 4 * A21 / (1 - A21^2)^2
```

这意味着高放大率尾端不仅样本可能更少, 而且预测对观测噪声更加敏感。因此, 如果训练集没有足够的尾端样本, 或损失函数没有对尾端给出足够权重, 模型自然会优先优化样本密集的低中值区间, 在高 `mu0` 区间表现为系统性低估。

## 3. 物理模型与核心公式

### 3.1 SIS 双像几何

SIS lens equation 的一维径向形式可写为:

```text
beta = theta - theta_E * sign(theta)
```

定义无量纲变量:

```text
y = beta / theta_E
x = theta / theta_E
```

则:

```text
y = x - sign(x)
```

当 `0 < y < 1` 时系统产生双像。两幅像的位置为:

```text
x_plus = 1 + y
x_minus = y - 1
```

对应到角位置:

```text
theta_plus = theta_E * (1 + y)
theta_minus = theta_E * (y - 1)
```

如果只看两幅像到透镜中心的绝对角距:

```text
|theta_plus| = theta_E * (1 + y)
|theta_minus| = theta_E * (1 - y)
```

因此:

```text
image_separation = |theta_plus| + |theta_minus| = 2 * theta_E
```

单独的像间距只能给出 `theta_E`, 不能唯一决定 `y` 或 `mu0`。但是如果知道两幅像分别到透镜中心的距离, 则可以得到:

```text
y = (|theta_plus| - |theta_minus|) / (|theta_plus| + |theta_minus|)
```

这说明 `image_position_asymmetry_observed` 在 SIS 设定下是一个非常强的 `y` 代理特征。当前 baseline 的 `all` preset 使用了该类光学像位置特征, 因而应在后续消融中区分含 image geometry 与不含 image geometry 的实验。

### 3.2 放大率、源位置和振幅比关系

当前代码中定义:

```text
u = 1 / y
log_u = log(u)
mu0 = 1 + u
mu1 = 1 - u
```

因为双像条件下 `u > 1`, 所以:

```text
mu0 > 2
mu1 < 0
```

绝对放大率比为:

```text
R_abs_21 = |mu1| / mu0
```

将 `mu0` 和 `mu1` 代入可得:

```text
R_abs_21 = (u - 1) / (u + 1)
         = (1 - y) / (1 + y)
         = (mu0 - 2) / mu0
```

两幅像的振幅比定义为:

```text
A21 = sqrt(R_abs_21)
```

因此:

```text
A21^2 = (mu0 - 2) / mu0
mu0 = 2 / (1 - A21^2)
mu1 = 2 - mu0
```

这是当前任务最核心的公式。只要输入特征能够准确估计 `A21` 或 `y`, `mu0` 就可以被精确反推。反过来, 如果 `A21` 或 `y` 的观测代理受噪声、探测器响应、裁窗误差和质量筛选影响, 高 `mu0` 尾端就会出现明显误差放大。

## 4. Baseline 数据生成协议

### 4.1 脚本定位与输出目标

当前 baseline 数据生成脚本为:

```text
数据生成/SIS_GW_events_randomu_baseline_quality_mu0_before_obsfix.py
```

脚本头部说明其目标是生成 quality-controlled real-observation SIS 数据, 并将输出拆分为四类:

1. 训练可用的观测量。
2. 标签和真值表。
3. 诊断用 privileged quantities。
4. 质量控制报告。

默认核心输出文件包括:

```text
lens.csv
lens_params.csv
observable_features.csv
SIS_data_strain_1.npy
SIS_data_strain_2.npy
SIS_h_strain_1.npy
SIS_h_strain_2.npy
SIS_noise_strain_1.npy
SIS_noise_strain_2.npy
diagnostic_features.csv
quality_report.json
quality_report.md
```

其中 `SIS_data_strain_*.npy` 是训练默认使用的 noisy whitened 一秒窗口。`lens.csv` 保存监督标签和必要的物理真值, `observable_features.csv` 保存训练入口可选择使用的观测代理特征, `diagnostic_features.csv` 保存 `A21_true`, `y_true`, `u_true`, `mu0_true`, `t_d_true` 等不应默认输入模型的诊断列。

[图2建议: 数据文件关系图。左侧为生成过程, 右侧分成 `lens.csv`, `lens_params.csv`, `observable_features.csv`, `diagnostic_features.csv`, `npy waveform arrays`, `quality_report` 六类输出, 用颜色区分训练输入、训练标签和诊断真值。]

### 4.2 配置参数与随机种子

脚本使用 `Config` dataclass 统一管理生成参数。当前 baseline 的关键配置如下:

| 配置项 | 当前值 | 作用 |
|---|---:|---|
| `target_events` | 2500 | 目标接受事件数 |
| `max_attempts` | 25000 | 最大尝试次数 |
| `seed` | 238 | 随机种子 |
| `z_min`, `z_max` | 0.05, 1.5 | 源红移范围 |
| `sigma_v_min`, `sigma_v_max` | 150, 350 km/s | 透镜速度弥散范围 |
| `a21_min`, `a21_max` | 0.45, 0.92 | A21 分层采样范围 |
| `a21_bins` | 8 | A21 分层桶数 |
| `td_min_sec`, `td_max_sec` | 1e2, 1e7 s | 时间延迟质量门 |
| `detector_names` | ET | 探测器 |
| `sampling_frequency` | 4096 Hz | 采样率 |
| `duration` | 24 s | 模拟帧长度 |
| `train_window_seconds` | 1 s | 训练窗口长度 |
| `add_detector_noise` | True | 加入 detector noise |
| `random_spin` | True | 随机自旋 |
| `random_external` | True | 随机天空位置、倾角、偏振角、相位 |
| `random_distance` | True | 随机光度距离 |

脚本通过 `seed_everything()` 同时设置 `numpy` 和 `bilby` 的随机状态, 从而保证同一配置下的数据生成可复现。

### 4.3 源参数采样

源参数由 `draw_source()` 生成。光度距离通过 `bilby.gw.prior.UniformComovingVolume` 在给定红移范围对应的 luminosity distance 区间内采样, 并由 `luminosity_distance_to_redshift` 转换为源红移 `z_s`。

质量采样范围为:

```text
mass_1_source in [20, 70] solar mass
mass_2_source in [10, min(50, mass_1_source)] solar mass
```

随后乘以红移因子得到探测器系质量:

```text
mass_1 = mass_1_source * (1 + z_s)
mass_2 = mass_2_source * (1 + z_s)
```

如果 `random_spin=True`, 自旋大小、倾角和相位角均随机采样; 如果 `random_external=True`, 赤经、赤纬、视线倾角、偏振角和相位也随机采样。当前 baseline 默认开启这些随机项, 因而样本间的 detector response 和投影因子变化比早期课程版更真实。

### 4.4 透镜参数采样与 SIS 计算

透镜参数由 `draw_lens()` 生成。脚本首先在 `[lens_z_min, z_s - gap]` 之间均匀采样透镜红移 `z_l`, 并在 `[150, 350] km/s` 之间均匀采样速度弥散 `sigma_v`。随后使用 `lenstronomy.Cosmo.LensCosmo` 计算 SIS Einstein 半径:

```text
theta_E = sis_sigma_v2theta_E(sigma_v)
```

当前 baseline 的关键点是, 脚本并不直接采样 `mu0` 或 `y`, 而是调用:

```text
draw_a21_stratified(rng, cfg, attempt_id)
```

该函数将 `[a21_min, a21_max]` 等分为 `a21_bins` 个桶, 并通过 `attempt_id % a21_bins` 轮流选择桶, 在桶内均匀采样 `A21`。之后根据 SIS 关系计算:

```text
R = A21^2
y = (1 - R) / (1 + R)
u = 1 / y
mu0 = 1 + u
mu1 = 1 - u
```

等价地:

```text
mu0 = 2 / (1 - A21^2)
```

因此当前协议保证的是尝试阶段的 `A21` 分层覆盖, 而不是 `mu0` 分层覆盖。

源在透镜平面上的位置由随机方位角 `phi` 决定:

```text
beta_x = y * theta_E * cos(phi)
beta_y = y * theta_E * sin(phi)
```

两幅像的位置为:

```text
image_0 = (1 + y) * theta_E * direction
image_1 = (y - 1) * theta_E * direction
```

时间延迟 `t_d` 使用 SIS 物理公式计算, 并通过 `[td_min_sec, td_max_sec]` 进行质量筛选。当前数据中 `t_d` 的范围约为 `9.46e3` 到 `9.99e6` 秒, 中位数约为 `1.83e6` 秒。

[图3建议: `A21 -> mu0` 非线性映射曲线。横轴为 `A21`, 纵轴为 `mu0 = 2 / (1 - A21^2)`, 在 `A21=0.45` 和 `A21=0.92` 处标出当前采样范围, 并标注高 `A21` 区间对应的 `mu0` 快速增长。]

### 4.5 观测误差注入

为了模拟真实观测, 脚本对多个光学和时间观测量加入噪声:

```text
redshift_obs_sigma = 5e-4
sigma_v_obs_frac_sigma = 0.05
theta_E_obs_frac_sigma = 0.02
astrometric_sigma_arcsec = 0.01
lens_center_sigma_arcsec = 0.02
```

红移采用加性高斯噪声并截断为非负; 速度弥散和 Einstein 半径采用乘性分数高斯噪声并限制为正值。像位置和透镜中心位置加入角秒级 astrometric noise。脚本进一步从 noisy image positions 中计算:

```text
theta_plus_abs_observed
theta_minus_abs_observed
image_separation_observed
image_position_asymmetry_observed
mu0_from_image_asymmetry_observed
```

其中:

```text
image_position_asymmetry_observed
  = (theta_plus_abs_observed - theta_minus_abs_observed)
    / image_separation_observed
```

该量在 SIS 下接近 `y`, 因而与 `mu0` 具有强物理联系。当前质量报告显示, `image_position_asymmetry_observed` 与 `y_true` 的 Spearman 相关系数约为 `0.851`。

### 4.6 透镜放大与双像波形生成

脚本通过 `lensed_bbh_source()` 对 `bilby.gw.source.lal_binary_black_hole` 生成的频域 `plus/cross` 波形施加透镜放大因子。对于第 `which_image` 幅像:

```text
amp = sqrt(abs(mu)) * exp[-i * pi * (2 f t_eff + morse_n)]
```

其中:

```text
morse_n = 0.0 for image 0
morse_n = 0.5 for image 1
```

第二像由于负宇称具有 Morse phase。当前配置 `geocent_time_includes_delay=True`, 因此频域放大因子中的 `t_eff` 取 0, 时间延迟主要通过第二像 `geocent_time = source_geocent_time + t_d` 体现。这使得第二像的 detector response 可以随真实到达时间变化, 比强行 same-response 的课程版更接近真实观测。

### 4.7 探测器模拟、白化和裁窗

每幅像由 `simulate_image()` 注入到 ET 探测器中。流程为:

1. 创建 `InterferometerList(["ET"])`。
2. 根据 PSD 设置 24 秒 strain data。
3. 如果 `add_detector_noise=True`, 保留 detector noise; 如果为 False, 将频域噪声置零。
4. 生成频域透镜波形并获取 detector response。
5. 计算 optimal SNR, 作为诊断和消融列保存。
6. 将 response 加入探测器 strain。
7. 得到 raw data、whitened data、signal 和 noise。
8. 根据 clean whitened signal 的最大绝对值位置裁剪一秒窗口。

裁窗使用 `crop_with_padding()` 完成。训练窗口长度为:

```text
4096 samples = 1 second at 4096 Hz
```

每个事件最终保存两幅像的:

```text
data_white
signal_white
noise_white
data_raw
signal_raw
noise_raw
time_window
```

默认训练使用 noisy whitened data:

```text
SIS_data_strain_1.npy
SIS_data_strain_2.npy
```

[图4建议: 单个事件的两幅像一秒白化窗口示例。展示 image 1 和 image 2 的 noisy whitened strain, 标出 peak-centered crop、Morse phase 造成的形态差异以及振幅差异。]

### 4.8 质量控制门

生成脚本不是接受所有候选事件, 而是对 lens 和 simulation 结果进行质量门筛选。主要筛选条件包括:

```text
td_min_sec <= t_d <= td_max_sec
snr_1 >= 12
snr_2 >= 8
sqrt(snr_1^2 + snr_2^2) >= 18
min(window_signal_noise_rms) >= 0.015
arrival time observation error within threshold
```

当前完整数据集的质量报告显示:

```text
Attempts: 2825
Accepted: 2500
Acceptance rate: 0.8850
```

拒绝原因统计为:

| reason | count |
|---|---:|
| `td_out_of_range` | 313 |
| `low_snr_image_2` | 5 |
| `low_window_signal_noise_rms` | 4 |
| `low_snr_image_1` | 2 |
| `low_pair_network_snr` | 1 |

可以看到, 当前质量筛选的主要拒绝原因是时间延迟超出范围, SNR 相关拒绝较少。因此, 尾端样本不足的主因不是质量门大量拒绝高 `mu0` 样本, 而是 `A21` 均匀到 `mu0` 非均匀的映射本身。

### 4.9 观测特征构建

`observable_feature_row()` 从两幅像的一秒 whitened data 和观测透镜参数中构建训练可选特征。特征分为几类。

第一类是波形振幅比和能量比代理:

```text
peak_amp_ratio_21
rms_ratio_21
energy_ratio_21
env_area_ratio_21
peak_amp_ratio_local_65
rms_ratio_local_65
energy_ratio_local_65
win0p25_peak_amp_ratio_21
win0p5_peak_amp_ratio_21
win0p9_peak_amp_ratio_21
...
```

第二类是波形形状和对齐特征:

```text
xcorr_lag
norm_xcorr_max
xcorr_lag_win65
norm_xcorr_max_win65
aligned_l1_residual
aligned_l2_residual
spec_centroid_diff
band_energy_ratio_low_21
band_energy_ratio_mid_21
band_energy_ratio_high_21
```

第三类是噪声上下文:

```text
local_noise_rms_1
local_noise_rms_2
window_snr_like_1
window_snr_like_2
```

第四类是光学和透镜观测代理:

```text
z_s_observed
z_l_observed
z_l_over_z_s
log1p_z_s_observed
log1p_z_l_observed
sigma_v_observed
log_sigma_v_observed
theta_E_observed
image_separation_observed
theta_plus_abs_observed
theta_minus_abs_observed
image_position_asymmetry_observed
lens_center_x_observed
lens_center_y_observed
image_x_0_observed
image_y_0_observed
image_x_1_observed
image_y_1_observed
```

第五类是到达时间观测代理:

```text
arrival_time_1_observed
arrival_time_2_observed
arrival_time_1_sigma
arrival_time_2_sigma
peak_time_diff
env_peak_time_diff
```

第六类是 optimal SNR 诊断列:

```text
snr_1
snr_2
snr_ratio_21
snr_sum
snr_diff
snr_pair
```

训练入口默认不使用 optimal SNR 特征, 只有显式传入 `--use-optimal-snr` 才会加入, 因为这些列由 clean injection response 计算, 在某些课程数据中会退化为答案型特征。

### 4.10 数据集统计与目标分布问题

当前 baseline 数据目录为:

```text
/root/autodl-tmp/tmp/数据生成/data_lens_randomu_realobs_quality_mu0
```

核心统计如下:

| quantity | min | p05 | median | mean | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| `mu0` | 2.509 | 2.590 | 3.965 | 4.885 | 10.185 | 13.018 |
| `mu1` | -11.018 | -8.185 | -1.965 | -2.885 | -0.590 | -0.509 |
| `A21_true` | 0.451 | 0.477 | 0.704 | 0.698 | 0.896 | 0.920 |
| `y_true` | 0.083 | 0.109 | 0.337 | 0.349 | 0.629 | 0.662 |
| `t_d_true` | 9.46e3 | 1.68e5 | 1.83e6 | 2.64e6 | 7.75e6 | 9.99e6 |

`mu0` 固定宽度分桶计数为:

| `mu0` bin | count |
|---|---:|
| `[2, 3)` | 580 |
| `[3, 4)` | 687 |
| `[4, 5)` | 357 |
| `[5, 6)` | 243 |
| `[6, 7)` | 191 |
| `[7, 8)` | 118 |
| `[8, 9)` | 112 |
| `[9, 10)` | 82 |
| `[10, 11)` | 46 |
| `[11, 12)` | 44 |
| `[12, 13)` | 37 |
| `[13, +inf)` | 3 |

这说明当前 baseline 数据明显不是 `mu0` 均匀分布。`mu0 in [2, 4)` 的样本数为 1267, 而 `mu0 >= 10` 的样本数仅约 130。高尾端样本数量不足会直接影响监督学习的误差权衡。

[图5建议: `mu0` 直方图与 `A21` 直方图对比。左图显示 `A21` 各桶相对均衡, 右图显示 `mu0` 明显集中在低中值区间。]

## 5. 训练代码架构

### 5.1 训练入口和基础模块关系

当前 baseline 训练入口为:

```text
train_mu0_baseline_obs_realobs_quality.py
```

该文件本身不重新定义完整模型, 而是作为实验配置入口复用基础训练模块:

```text
train_mu0_from_wave_obs_randomu_a21friendly_realobs.py
```

入口脚本完成三件事:

1. 将数据路径指向当前 quality realobs 数据目录。
2. 定义不同 preset 下的观测特征列组合。
3. 覆盖基础模块中的 `selected_obs_cols()` 函数, 然后调用 `base.main()` 执行完整训练。

默认数据根目录为:

```text
/root/autodl-tmp/tmp/数据生成/data_lens_randomu_realobs_quality_mu0
```

默认输出目录为:

```text
/root/autodl-tmp/tmp/runs/mu_direct_baseline_realobs_quality_waveobs
```

对于默认 `--preset all`, 训练输出位于:

```text
/root/autodl-tmp/tmp/runs/mu_direct_baseline_realobs_quality_waveobs/all
```

保存文件包括:

```text
best_mu_direct_realobs_quality_all.pt
mu_direct_realobs_quality_all_scatter.png
mu_direct_realobs_quality_all_val_predictions.csv
```

### 5.2 特征 preset 设计

入口脚本将可观测特征拆成多个组:

```text
A21_PROXY_COLS
WAVE_SHAPE_COLS
NOISE_CONTEXT_COLS
TIME_OBS_COLS
OPTICAL_REDSHIFT_COLS
OPTICAL_SIGMA_COLS
OPTICAL_SCALE_COLS
IMAGE_POSITION_COLS
OPTIMAL_SNR_COLS
```

支持的 preset 包括:

| preset | 含义 |
|---|---|
| `all` | 使用 waveform proxies、时间观测、红移、速度弥散、尺度和像位置几何 |
| `gw_only` / `no_optical` | 只使用波形统计、噪声上下文和时间观测, 不使用光学量 |
| `no_image` | 使用光学红移、速度弥散和尺度, 但不使用 individual image positions |
| `no_time` | 使用光学和波形特征, 不使用时间延迟相关特征 |
| `a21_wave` | 只隔离波形振幅比和形状代理 |
| `time_lens` | 隔离时间延迟和透镜尺度相关观测量 |
| `legacy` | 保留旧版兼容特征列 |

这一设计适合做消融实验。尤其需要重点比较:

```text
all vs no_image
all vs no_time
all vs no_optical
a21_wave vs time_lens
```

原因是 image position asymmetry 在 SIS 下几乎可以直接给出 `y`, 而 time delay 结合透镜质量尺度也与 `y` 强相关。若目标是评估纯 GW 波形能否恢复 `mu0`, 则必须单独报告 `gw_only` 或 `a21_wave` 结果。

[图6建议: 训练特征 preset 消融示意图。用模块化方块展示每个 preset 开启或关闭哪些观测通道。]

### 5.3 数据读取和样本构造

基础训练模块在 `main()` 中完成数据读取:

```text
wave1 = np.load(SIS_data_strain_1.npy, mmap_mode="r")
wave2 = np.load(SIS_data_strain_2.npy, mmap_mode="r")
lens_df = read_csv(lens.csv)
lens_params_df = read_csv(lens_params.csv)
obs_df = read_csv(observable_features.csv)
```

使用 `mmap_mode="r"` 读取波形数组可以避免一次性将大文件载入内存。脚本随后根据所有表和数组的最短长度截断对齐, 并通过 `add_optional_observables()` 补齐某些旧数据目录中可能缺失的光学列。

数据划分由 `build_split_indices()` 完成。当前实现为固定随机种子 shuffle 后按比例切分:

```text
train : val = 0.8 : 0.2
```

需要注意, 当前划分不是按 `mu0` 分层划分。因此当高 `mu0` 尾端本来就少时, 验证集中每个尾端桶可能只有几个样本, 导致尾端评估方差较大。

每个样本由 `MuDirectCompactDataset` 返回:

```text
w1: image 1 waveform tensor
w2: image 2 waveform tensor
obs_vec: normalized observable vector
y_target: transformed target = log(mu0 - 1)
mu0_true
mu1_true
idx
```

### 5.4 波形预处理

每条输入波形通过 `pad_or_trim()` 裁剪或补零到 `TARGET_LEN`。当前入口默认:

```text
TARGET_LEN = 4096
STRIDE = 1
```

即每条像使用完整一秒窗口。

波形构造采用双通道表示:

1. `shape_ch`: 根据同一事件两幅像的拼接波形计算 pair mean 和 pair std, 然后进行 z-score 标准化。
2. `amp_ch`: 使用固定尺度 `RAW_SCALE` 做 `tanh(x / RAW_SCALE)`, 用于保留绝对振幅信息。

伪代码为:

```text
pair = concat(x1, x2)
pair_mean = mean(pair)
pair_std = std(pair)

shape_ch = (x - pair_mean) / pair_std
amp_ch = tanh(x / RAW_SCALE)
input = stack(shape_ch, amp_ch)
```

这种设计的动机是同时保留两类信息: pair normalization 让网络稳定学习波形形态, fixed amplitude channel 保留放大率相关的绝对幅度线索。

### 5.5 观测特征标准化

观测特征使用 `FeatureNormalizer` 在训练集上计算均值和标准差:

```text
obs_norm.fit(obs_df.iloc[idx_train])
```

验证集使用同一 normalizer 变换, 避免验证信息泄露到训练统计量中。normalizer 状态会随 best checkpoint 一起保存:

```text
"obs_norm": obs_norm.state_dict()
```

### 5.6 防止 oracle 特征泄露

基础模块提供 `validate_no_oracle_leakage()`。该函数维护一组 label-side columns:

```text
mu_0, mu_1, abs_mu_0, abs_mu_1,
mu_ratio_abs_21, R_abs_21, amp_ratio_21_true,
u, log_u, y, u_true, log_u_true, y_true,
beta, beta_true,
t_d, theta_E_true, sigma_v_true,
z_l_true, z_s_true,
...
```

如果这些列出现在训练输入中, 且没有显式设置 `ALLOW_ORACLE_FEATURES=True`, 脚本会报错。默认也不会使用 `OPTIMAL_SNR_COLS`, 因为 optimal SNR 在某些数据协议中接近 clean truth proxy。

这个检查对当前项目非常关键, 因为 SIS 下很多看似普通的物理列都能解析反推 `mu0`。例如 `y_true`, `R_abs_21`, `A21_true`, `mu_ratio_abs_21` 都不应该出现在默认训练输入中。

## 6. 模型结构

### 6.1 总体结构

当前模型类为:

```text
MuDirectCompactRegressor
```

它包含两个主要部分:

1. 共享的一维波形编码器 `WaveEncoder`, 分别编码 image 1 和 image 2。
2. 观测特征 MLP `obs_head`, 编码表格观测量。

融合时使用:

```text
f1 = WaveEncoder(w1)
f2 = WaveEncoder(w2)
fd = abs(f1 - f2)
fp = f1 * f2
fo = obs_head(obs)

fused = concat(f1, f2, fd, fp, fo)
```

随后通过 MLP head 输出一个标量, 即 transformed target:

```text
model_output = log(mu0 - 1)
```

[图7建议: 模型架构图。两条波形进入共享 WaveEncoder, 得到 `f1` 和 `f2`; 观测特征进入 Obs MLP 得到 `fo`; 融合 `[f1, f2, |f1-f2|, f1*f2, fo]`; 最后 MLP 输出 `log(mu0-1)`, 反变换得到 `mu0`, 再由 SIS 关系得到 `mu1`。]

### 6.2 WaveEncoder

`WaveEncoder` 是 compact 1D CNN。默认输入通道数为 2, 即 `shape_ch` 和 `amp_ch`。结构为多层 Conv1d、BatchNorm、激活函数和池化:

```text
Conv1d(in_ch -> width, kernel=15, stride=2)
BatchNorm1d
Activation
MaxPool1d

Conv1d(width -> 2width, kernel=11, stride=2)
BatchNorm1d
Activation

Conv1d(2width -> 4width, kernel=9, stride=2)
BatchNorm1d
Activation

Conv1d(4width -> 4width, kernel=7, stride=2)
BatchNorm1d
Activation

AdaptiveAvgPool1d(1)
```

默认 `WAVE_WIDTH=64`, 因而每条像最终输出维度为 `256`。

该编码器较轻量, 适合 baseline 实验。它主要学习一秒窗口内的局部时域形态、相位翻转和振幅结构, 不负责显式建模长达数天或数月的时间延迟, 因为时间延迟作为观测代理特征通过 `peak_time_diff` 等表格列输入。

### 6.3 观测特征编码器

观测特征通过两层 MLP:

```text
Linear(obs_dim -> OBS_HIDDEN)
Activation
Linear(OBS_HIDDEN -> D_MODEL / 2)
Activation
```

默认:

```text
OBS_HIDDEN = 128
D_MODEL = 192
```

因此 `fo` 维度为 96。

### 6.4 融合预测头

融合向量维度为:

```text
wave_dim * 4 + D_MODEL / 2
```

其中 `wave_dim * 4` 来自:

```text
f1, f2, abs(f1 - f2), f1 * f2
```

预测头为:

```text
Linear(fusion_dim -> D_MODEL)
Activation
Dropout(0.15)
Linear(D_MODEL -> D_MODEL / 2)
Activation
Linear(D_MODEL / 2 -> 1)
```

这种结构相比直接拼接两条波形有两个优势:

1. 共享编码器保证两幅像使用同一特征空间。
2. 差值和乘积特征显式提供两像关系, 有利于学习振幅比和形态差异。

## 7. 训练目标、损失函数与评估

### 7.1 目标变换

训练目标由 `target_transform_np()` 定义。当前默认:

```text
TARGET_MODE = "log_mu0_minus1"
target = log(mu0 - 1)
```

预测后反变换:

```text
mu0_pred = 1 + exp(pred_target)
mu1_pred = 2 - mu0_pred
```

这种目标变换符合 SIS 关系:

```text
mu0 - 1 = u = 1 / y
```

因此:

```text
log(mu0 - 1) = log_u
```

在数值上, 它可以压缩 `mu0` 的长尾范围, 避免模型直接回归高放大率绝对值时被极端样本主导。

### 7.2 损失函数

当前 `compute_losses()` 包含三部分:

1. transformed space 的 MAE:

```text
trans_mae = mean(abs(pred_target - target_transformed))
```

2. real space 的 MAE:

```text
abs_err = abs(mu0_pred - mu0_true)
```

3. real space 的相对误差:

```text
rel_err = abs_err / abs(mu0_true)
```

如果启用 tail weight, 则:

```text
weight = 1 + TAIL_ALPHA * log1p(abs(mu0_true))
```

默认:

```text
USE_TAIL_WEIGHT = True
TAIL_ALPHA = 0.35
LOSS_W_MAE = 0.02
LOSS_W_REL = 0.08
```

最终损失为:

```text
loss = trans_mae + LOSS_W_MAE * weighted_mae + LOSS_W_REL * weighted_mape
```

这种设计兼顾 transformed target 的稳定性和原始 `mu0` 空间的误差约束。但是当前 tail weight 比较温和, 对 `mu0 >= 10` 这种数量明显不足的尾端样本补偿不够。

### 7.3 优化器和训练策略

训练使用:

```text
AdamW
LR = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 32
EPOCHS = 160
EMA_DECAY = 0.995
```

每个 batch 后更新 EMA 模型。验证时使用 EMA 模型, 以提高评估稳定性。学习率调度器为:

```text
ReduceLROnPlateau(factor=0.7, patience=3, min_lr=5e-7)
```

主监控指标是验证集 `mu0` MAPE。如果 MAPE 改善, 保存 best checkpoint、散点图和验证预测 CSV。早停条件为:

```text
ep >= MIN_EPOCHS_BEFORE_STOP and wait >= EARLY_STOP_PATIENCE
```

其中:

```text
MIN_EPOCHS_BEFORE_STOP = 50
EARLY_STOP_PATIENCE = 14
```

### 7.4 评估指标

脚本计算:

```text
MAE
RMSE
MAPE
Corr
```

分别用于 `mu0 direct` 和 `mu1 from SIS`。由于 `mu1_pred = 2 - mu0_pred`, `mu1` 的绝对误差与 `mu0` 基本相同, 但相对误差会因 `mu1` 在接近 0 的低放大率区域绝对值较小而更高。

保存的验证预测表包含:

```text
event_id
mu0_true
mu0_pred
mu1_true
mu1_pred
z_s_observed
z_l_observed
z_l_over_z_s
sigma_v_observed
theta_E_observed
image_separation_observed
mu0_abs_err
mu1_abs_err
mu0_rel_err_percent
mu1_rel_err_percent
```

## 8. Baseline 实验结果与诊断

### 8.1 总体表现

当前 baseline 输出目录为:

```text
/root/autodl-tmp/tmp/runs/mu_direct_baseline_realobs_quality_waveobs/all
```

验证集总体指标为:

| target | MAE | RMSE | MAPE | Corr |
|---|---:|---:|---:|---:|
| `mu0` | 0.774 | 1.324 | 13.16% | 0.853 |
| `mu1` | 0.774 | 1.324 | 26.26% | 0.853 |

从整体指标看, baseline 已经能学到较强的 `mu0` 顺序关系, 相关系数约为 0.853。但是散点图显示, 预测值的动态范围被压缩, 高 `mu0` 区间存在明显低估。

[图8建议: 当前 baseline 的 `mu0` 和 `mu1` 预测散点图。直接引用 `runs/mu_direct_baseline_realobs_quality_waveobs/all/mu_direct_realobs_quality_all_scatter.png`, 并在图注中说明高 `mu0` 与负 `mu1` 尾端出现收缩。]

### 8.2 分桶误差

按真实 `mu0` 固定宽度分桶, 验证集误差如下:

| true `mu0` bin | n | true mean | pred mean | bias | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `[2, 3)` | 114 | 2.707 | 2.835 | 0.128 | 0.192 | 0.418 |
| `[3, 4)` | 130 | 3.443 | 3.464 | 0.021 | 0.369 | 0.579 |
| `[4, 5)` | 67 | 4.518 | 4.477 | -0.041 | 0.561 | 0.757 |
| `[5, 6)` | 57 | 5.467 | 5.027 | -0.440 | 0.761 | 1.002 |
| `[6, 7)` | 42 | 6.470 | 5.802 | -0.668 | 0.952 | 1.238 |
| `[7, 8)` | 33 | 7.544 | 6.075 | -1.469 | 1.570 | 1.867 |
| `[8, 9)` | 22 | 8.554 | 6.766 | -1.788 | 1.921 | 2.513 |
| `[9, 10)` | 14 | 9.457 | 7.625 | -1.832 | 1.848 | 2.032 |
| `[10, 11)` | 6 | 10.434 | 7.157 | -3.277 | 3.277 | 3.324 |
| `[11, 12)` | 10 | 11.558 | 8.025 | -3.533 | 3.533 | 3.939 |
| `[12, 13)` | 5 | 12.370 | 8.087 | -4.283 | 4.283 | 4.894 |

可以看到, `mu0 < 5` 区间的 bias 较小, 而从 `mu0 >= 7` 开始系统性低估迅速加重。`mu0 >= 10` 的 bias 达到 `-3` 以上。模型不是随机失败, 而是在高尾端发生稳定的 regression to mean。

线性拟合关系约为:

```text
mu0_pred ~= 0.632 * mu0_true + 1.349
```

理想斜率应接近 1。当前斜率明显小于 1, 说明预测动态范围被压缩。

[图9建议: 分桶误差柱状图。横轴为 `mu0` bin, 左轴显示样本数, 右轴显示 bias 或 MAE。建议用双轴图展示样本稀疏与尾端误差上升之间的对应关系。]

### 8.3 目标分布不均衡是尾端问题的主要原因

当前数据生成脚本对 `A21` 分层均匀, 但实际 `mu0` 分布不均匀。`A21` 的最后一个桶 `[0.861, 0.92]` 对应 `mu0` 从约 7.76 到 13.02, 覆盖了很宽的 `mu0` 范围。相比之下, 低 `A21` 桶只对应很窄的 `mu0` 区间。例如:

| A21 bin | n | `mu0_min` | `mu0_max` | `mu0_mean` |
|---|---:|---:|---:|---:|
| `(0.449, 0.509]` | 278 | 2.509 | 2.698 | 2.603 |
| `(0.509, 0.568]` | 269 | 2.699 | 2.943 | 2.818 |
| `(0.568, 0.626]` | 278 | 2.958 | 3.286 | 3.123 |
| `(0.626, 0.685]` | 314 | 3.291 | 3.767 | 3.510 |
| `(0.685, 0.744]` | 320 | 3.770 | 4.472 | 4.091 |
| `(0.744, 0.802]` | 335 | 4.478 | 5.612 | 5.024 |
| `(0.802, 0.861]` | 353 | 5.624 | 7.743 | 6.579 |
| `(0.861, 0.920]` | 353 | 7.756 | 13.018 | 9.765 |

这说明最后一个 A21 桶承担了整个高 `mu0` 尾端, 但桶内并没有再按 `mu0` 均匀划分。对于监督学习来说, 这种分布会使模型在高 `mu0` 区间看到的样本更少、变化更剧烈、噪声更敏感。

### 8.4 观测代理相关性降低了波形振幅比的直接可用性

质量报告中的 Spearman 相关显示:

| pair | corr |
|---|---:|
| `peak_amp_ratio_21` vs `A21_true` | 0.220 |
| `rms_ratio_21` vs `A21_true` | 0.182 |
| `energy_ratio_21` vs `R_abs_21_true` | 0.182 |
| `env_area_ratio_21` vs `A21_true` | 0.180 |
| `snr_ratio_21` vs `A21_true` | 0.385 |
| `image_position_asymmetry_observed` vs `y_true` | 0.851 |
| `t_d_observed` vs `t_d_true` | 1.000 |
| `peak_time_diff` vs `t_d_true` | 1.000 |

与课程版不同, 当前 noisy realobs 数据中简单 waveform amplitude ratio 与理论 `A21` 的相关性并不高。这说明 detector noise、随机外在几何、真实到达时间、裁窗和局部噪声会显著干扰从一秒窗口振幅统计恢复 `A21` 的过程。相比之下, 光学像位置不对称度与 `y_true` 相关性较强, 因而 `all` preset 中的光学几何特征可能承担了较多预测信息。

这一点提示后续报告实验必须明确区分:

```text
GW-only baseline
GW + time-delay baseline
GW + optical scale baseline
GW + image geometry baseline
```

否则单个 `all` 结果不能说明模型主要依赖了哪条观测链。

## 9. 当前 baseline 的主要问题

### 9.1 数据生成层面: `mu0` 未均匀覆盖

当前脚本中的注释写明 "Sample A21 directly, then derive y, mu0 and mu1"。这有利于保证两像振幅比范围覆盖, 但不保证 `mu0` 范围覆盖。由于 `mu0(A21)` 在高 `A21` 处快速增长, 直接均匀采 `A21` 会导致 `mu0` 低值区间过密, 高值区间过疏。

如果研究目标是训练一个在完整 `mu0` 范围内稳定的回归器, 则 baseline 数据协议应改为直接按 `mu0` 分层采样, 再反推:

```text
u = mu0 - 1
y = 1 / u
mu1 = 2 - mu0
R = abs(mu1) / mu0
A21 = sqrt(R)
```

如果仍希望保持当前 `A21` 范围, 可以先将 `A21` 范围换算成 `mu0` 范围:

```text
mu0_min = 2 / (1 - a21_min^2)
mu0_max = 2 / (1 - a21_max^2)
```

当前 `A21 in [0.45, 0.92]` 对应:

```text
mu0 in [2.51, 13.02]
```

### 9.2 训练层面: tail weight 不足

当前 tail weight 为:

```text
weight = 1 + 0.35 * log1p(abs(mu0_true))
```

该权重随 `mu0` 增加较慢, 不能抵消尾端样本数量近一个数量级的不足。更合适的策略是按目标分桶反频率加权:

```text
weight_i = 1 / sqrt(count(bin(mu0_i)))
```

或者采用人工分段权重:

```text
mu0 < 6:        weight = 1
6 <= mu0 < 8:  weight = 2
8 <= mu0 < 10: weight = 3
mu0 >= 10:     weight = 5
```

如果担心高权重放大噪声, 可以将 MSE 换成 Huber 或 SmoothL1:

```text
loss = weighted SmoothL1 in transformed space
     + weighted MAE/MAPE in real space
```

### 9.3 数据划分层面: 验证集尾端样本过少

当前训练验证划分是普通随机 shuffle。验证集中 `mu0 >= 10` 的样本数很少, 分桶误差波动较大。建议改为 `mu0` stratified split, 保证每个 `mu0` 区间在训练集和验证集中都有稳定比例。

### 9.4 特征层面: image geometry 需要单独报告泄露风险

`theta_plus_abs_observed`, `theta_minus_abs_observed` 和 `image_position_asymmetry_observed` 在 SIS 下可以直接恢复 `y` 的 noisy proxy。它们不是标签真值, 可以被视为真实光学观测, 但物理上非常接近答案链路。因此报告结果时应明确说明:

1. 如果目标是多信使观测融合, 使用 image geometry 是合理的。
2. 如果目标是检验 GW 波形能否独立预测放大率, 则必须禁用 image geometry。
3. 如果目标是构建最保守的 no-leakage baseline, 则应报告 `no_image` 或 `gw_only` preset。

### 9.5 观测时间延迟的简化

当前 `t_d_observed` 与 `t_d_true` 的 Spearman 相关为 1.0, 说明时间延迟观测在该数据协议中几乎没有排序误差。虽然到达时间加入了 SNR 相关高斯扰动, 但相对于 `t_d` 的动态范围来说误差极小。因此, 时间延迟特征可能比真实低 SNR 条件下更理想。后续如果希望更真实, 可以考虑:

```text
arrival-time uncertainty 随 SNR 和窗口质量更强地变化
引入 peak-finding failure 或多峰歧义
对短延迟和弱第二像场景加入更严格的观测误差模型
```

## 10. 改进方案

### 10.1 数据生成协议改进: 按 `mu0` 分层采样

建议新增 `mu0_min`, `mu0_max`, `mu0_bins`, 并将 `draw_lens()` 中的 A21 采样替换为:

```python
def draw_mu0_stratified(rng, cfg, attempt_id):
    bin_id = attempt_id % max(1, cfg.mu0_bins)
    edges = np.linspace(cfg.mu0_min, cfg.mu0_max, cfg.mu0_bins + 1)
    lo = float(edges[bin_id])
    hi = float(edges[bin_id + 1])
    return float(rng.uniform(lo, hi)), int(bin_id)
```

在 `draw_lens()` 中:

```python
mu0, mu0_bin = draw_mu0_stratified(rng, cfg, attempt_id)
u = mu0 - 1.0
y = 1.0 / u
mu1 = 2.0 - mu0
R = abs(mu1) / mu0
A21 = np.sqrt(R)
```

更严格的做法是按 accepted count 控制每个 bin 的数量, 而不是按 attempt count 轮转。否则质量门拒绝后, 最终接受样本仍可能不均匀。

[图10建议: 新旧采样协议对比图。左侧是 `sample A21 -> derive mu0`, 右侧是 `sample mu0 -> derive A21`; 下方显示两种协议对应的 `mu0` histogram。]

### 10.2 训练划分改进: `mu0` 分层 split

建议将 `build_split_indices()` 改为按 `mu0` 分桶后在每个桶内随机划分。这样可以避免验证集高尾端样本过少, 也让分桶指标更稳定。

### 10.3 损失函数改进: 分桶加权 SmoothL1

当前损失函数可以保留 transformed target, 但权重建议由训练集分布计算:

```text
bin = digitize(mu0_true, mu0_edges)
weight = 1 / sqrt(train_count[bin])
weight = clip(weight, min_w, max_w)
```

然后:

```text
loss_trans = weight * SmoothL1(pred_logu, true_logu)
loss_real = weight * abs(mu0_pred - mu0_true)
loss_rel = weight * abs(mu0_pred - mu0_true) / abs(mu0_true)
loss = loss_trans + alpha * loss_real + beta * loss_rel
```

这样可以直接缓解 `mu0 >= 8` 或 `mu0 >= 10` 的系统性低估。

### 10.4 评估改进: 固定报告分桶指标

建议每次训练都输出:

```text
global MAE/RMSE/MAPE/Corr
per-bin n, bias, MAE, RMSE
slope/intercept of pred vs true
tail-only metrics for mu0 >= 8 and mu0 >= 10
```

仅报告整体 MAE 会掩盖尾端问题。当前 baseline 整体 MAE 为 0.774, 看起来尚可, 但 `mu0 >= 10` 的 bias 已经超过 -3。

### 10.5 模型改进: residual calibration 或 tail expert

如果重新采样和分桶加权后尾端仍然低估, 可以考虑:

1. 在验证集上做 post-hoc calibration:

```text
mu0_calibrated = a * mu0_pred + b
```

2. 训练 residual model:

```text
residual = mu0_true - mu0_pred_baseline
final_pred = baseline_pred + residual_model(features)
```

3. 引入 tail expert:

```text
main head: 主体区间
tail head: 高 mu0 区间
gate head: 判断样本属于主体还是尾端
```

不过这些模型改进应放在数据协议修复之后。否则模型可能只是用复杂结构拟合由采样不均导致的偏差。

## 11. 建议的论文式实验组织

参照 `HBGSA_Paper_English_TwoColumn.tex` 的结构, 后续正式报告可以组织为:

### 11.1 Introduction

说明强透镜引力波放大率预测的重要性, 当前从双像波形和观测代理恢复 `mu0` 的挑战, 以及 baseline 的目标。

### 11.2 Related Work

围绕三个方向展开:

1. 引力波强透镜参数估计。
2. 波形驱动的深度回归模型。
3. 物理约束机器学习和长尾回归。

### 11.3 Methods

分为:

```text
SIS physical formulation
Quality-controlled real-observation dataset generation
Observable feature construction
Wave+observable compact regressor
Training objective and SIS-constrained inference
```

### 11.4 Results

建议至少包含:

```text
Dataset distribution and quality-control summary
Baseline all-preset prediction scatter
Per-bin error analysis
Ablation across feature presets
Tail-sampling diagnosis
```

### 11.5 Discussion

讨论:

```text
Why A21-uniform does not imply mu0-uniform
Why realobs waveform amplitude proxies are weaker than ideal A21 proxies
Which observable chains dominate prediction
How to redesign the next dataset protocol
```

### 11.6 Conclusion

总结当前 baseline 的贡献和局限, 强调下一步应优先修复 `mu0` 分布覆盖。

## 12. 结论

当前 baseline 已经建立了较完整的真实观测代理数据生成和训练管线。数据生成脚本能够从源参数采样、SIS 透镜计算、双像波形注入、detector noise、白化裁窗、观测误差建模、质量控制到多表输出形成闭环。训练代码采用共享波形编码器和观测特征 MLP, 通过 SIS 物理关系只预测 `mu0` 并推导 `mu1`, 整体设计清晰, 也具备多种 feature preset 消融能力。

但当前 baseline 的核心问题是数据协议中 `mu0` 覆盖不均。脚本分层采样的是 `A21`, 而非 `mu0`; 由于 `mu0 = 2 / (1 - A21^2)`, 最终接受样本在低中 `mu0` 区间密集, 高 `mu0` 区间稀疏。验证结果与这一诊断一致: 模型总体相关性较高, 但预测斜率约为 0.632, 高 `mu0` 尾端出现系统性低估, `mu1` 负尾端同步回缩。

因此, 下一阶段最重要的改进不是单纯增大模型, 而是重新设计数据生成协议和训练权重。建议优先实现 `mu0` 分层 accepted sampling, 配合 `mu0` 分层训练验证划分和分桶加权 SmoothL1 loss。完成后再评估 feature preset 消融和模型结构改进, 才能更准确判断当前 wave+observable baseline 在真实观测代理条件下的物理预测能力。

## 附录 A: 当前报告引用的关键文件

```text
参考论文风格:
HBGSA_Paper_English_TwoColumn.tex

数据生成脚本:
数据生成/SIS_GW_events_randomu_baseline_quality_mu0_before_obsfix.py

数据目录:
数据生成/data_lens_randomu_realobs_quality_mu0

训练入口:
train_mu0_baseline_obs_realobs_quality.py

基础训练模块:
train_mu0_from_wave_obs_randomu_a21friendly_realobs.py

训练输出:
runs/mu_direct_baseline_realobs_quality_waveobs/all

参考 markdown:
a21friendly_mu0_training_analysis.md
mu_prediction_factor_formulas.md
实验记录报告.md
数据生成/data_lens_randomu_realobs_quality_mu0/quality_report.md
```

## 附录 B: 推荐插图清单

| 编号 | 图名 | 内容 |
|---|---|---|
| 图1 | 项目整体流程图 | 源采样到模型预测的全流程 |
| 图2 | 数据文件关系图 | 训练输入、标签、诊断真值和质量报告的分离 |
| 图3 | `A21 -> mu0` 非线性映射 | 展示 A21 均匀为何不等于 mu0 均匀 |
| 图4 | 双像一秒白化窗口示例 | 展示两幅像的 noisy whitened strain |
| 图5 | `A21` 与 `mu0` 分布对比 | 直观显示标签分布不均衡 |
| 图6 | feature preset 消融示意图 | 展示 all/no_image/no_time/gw_only 等输入差异 |
| 图7 | 模型架构图 | WaveEncoder、Obs MLP、融合头和 SIS 反变换 |
| 图8 | baseline 预测散点图 | 引用当前 `mu_direct_realobs_quality_all_scatter.png` |
| 图9 | 分桶误差柱状图 | 展示样本数、bias 和 MAE 随 `mu0` bin 的变化 |
| 图10 | 新旧采样协议对比 | A21 分层 vs mu0 分层 |

