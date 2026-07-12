# 远端数据与生成代码审计报告

审计日期：2026-07-12（UTC）  
远端数据位置：`/root/autodl-tmp/qkzhang/`  
审计方式：只读 SSH、文件清单、NPY 头读取、CSV 全量统计、小规模 memmap 数值抽样、代码静态检查、已有 smoke 日志检查。  
本次未修改远端文件，未启动训练或全量重新生成。

## 1. 执行结论

### 1.1 总体判断

当前数据**不适合直接承担**“面向星系尺度强透镜引力波双像的多信使、物理约束、校准后验推断”的主训练集或最终测试集。

它适合保留为：

1. 遗留 SIS/点质量透镜（PM）波形生成基线；
2. SIS 解析恒等式和 Morse 相位约定的回归测试材料；
3. 高斯设计噪声下的早期 smoke/消融数据；
4. 旧的双像配对或时频图分类基线材料。

它不适合直接支持：

1. 多信使绝对放大率后验；
2. 真实 GWTC/GWOSC 数据性能；
3. SIE、幂律椭圆、外剪切或外会聚下的模型不确定性；
4. H1/L1/V1 网络推断；
5. 后验覆盖率或模拟后验校准；
6. 缺失模态、观测协方差和光学成像似然；
7. 论文中的最终科学结论。

### 1.2 最严重的五项问题

| 等级 | 问题 | 对新任务的影响 |
|---|---|---|
| 阻断 | 现存主 NPY 数组没有探测器轴，形状为 `(event, 98304)`，但当前脚本声称输出 `(event, detector, 98304)` | 无法进行可靠的 H1/L1/V1 网络建模；数据与代码版本不一致 |
| 阻断 | 所有噪声由 Bilby 内置设计 PSD 合成，不是 GWOSC 真实 strain | 不能作为真实噪声实验、GWTC anchor injection 或目录应用证据 |
| 阻断 | 所谓“光学/几何”字段主要是精确潜变量：`y`、`beta_x/y`、精确 `theta_E`、精确 `sigma_v`，没有测量误差或协方差 | 若作为输入，SIS 绝对放大率可被解析地直接恢复，形成 oracle/解析捷径 |
| 阻断 | 只有 SIS 和点质量模型；没有星系尺度所需的 SIE+shear、幂律椭圆+shear、`kappa_ext` 或 mass-sheet nuisance | 无法研究核心的模型条件性、失配偏差和覆盖率 |
| 严重 | 0228 的生成代码、随机种子和环境清单未保留；目录没有 Git、manifest 或 checksums | 0228 虽然数值上与 0222 不同，但不能严格复现或追溯 |

## 2. 审计范围和证据边界

根据本仓库现有 `docs/DATA_SOURCES.md`，正式范围是日期标记为 0222 和 0228 的数据。本报告以这两批为核心；0123 仅用于判断版本沿革，未将其纳入推荐的研究数据范围。

本次用户指定的远端根目录是 `/root/autodl-tmp/qkzhang/`，而现有 `docs/DATA_SOURCES.md` 仍记录 `/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main/`。这是一个需要由用户确认并随后修订的数据治理冲突；本报告没有擅自把旧文档中的 authoritative path 改掉。

远端 `/root/autodl-tmp/qkzhang/` 总占用约 425 GB，但其中包含大量材料、DFT、导热、模型和软件文件。与本项目直接相关的主要目录为：

- `SIS_data_0222/`、`SIS_data_0228/`；
- `PM_data_0222/`、`PM_data_0228/`；
- `Unlensed_data_0222/`、`Unlensed_data_0228/`；
- `data_gen_check/`；
- 六套 `dataset_images_*` 派生图像目录。

重要限制：

- 未对约 154 GB 的 0222/0228 波形逐元素全盘扫描；NPY 头和选定事件块已读取，CSV 已全量统计。
- 由于生成时的 Git 提交和原始脚本缺失，不能用密码学证据证明 2 月 NPY 的确切生成程序。
- 当前 `data_gen_check` 是 5 月留下的检查/修订代码；它与 2 月现存数组的输出结构明显不同。

## 3. 数据清单

### 3.1 正式范围内的样本数

| 批次 | SIS 双像对 | PM 双像对 | 独立无透镜事件 | 近似磁盘占用 |
|---|---:|---:|---:|---:|
| 0222 | 2,500 | 2,500 | 5,000 | 约 77 GB |
| 0228 | 2,500 | 2,500 | 5,000 | 约 77 GB |
| 合计 | 5,000 | 5,000 | 10,000 | 约 154 GB |

每个 SIS/PM 批次还包含两套各 5,000 条的 `test_*` 波形。这些不是独立 astrophysical test sources，而是从每个透镜源构造的相似无透镜反例：保持大部分源参数不变，只扰动表观距离和到达时间。它们应被视为旧配对任务的 hard negatives，不能直接视为独立测试总体。

### 3.2 波形数组结构

所有 0222 和 0228 主数据的共同结构为：

- 采样率：4096 Hz；
- 时长：24 s；
- 每条长度：98,304；
- dtype：`float64`；
- 透镜主数组形状：`(2500, 98304)`；
- 透镜 hard-negative 数组形状：`(5000, 98304)`；
- 无透镜数组形状：`(5000, 98304)`；
- SNR 数组形状：`(N,)`。

每类数据分别保存：

- `*_data_strain_*.npy`：白化后的信号加噪声；
- `*_h_strain_*.npy`：白化后的干净注入信号；
- `*_time_array_*.npy`：每事件完整时间轴；
- `*_optimal_SNR_*.npy`：单个标量 SNR；
- CSV 源参数、透镜参数和标签。

抽查事件块均为有限数，没有发现 NaN/Inf；干净信号峰值被统一移到索引约 96,804，即数组尾部前 1,500 个采样点。这是人为对齐，不是原始事件窗口。

### 3.3 探测器轴丢失和代码版本断层

现存数组只含二维 `(event, time)`，没有 detector 字段，也没有 detector manifest。当前 5 月脚本则创建：

```text
(n_events, n_ifos, N), n_ifos = 2, ifos = [H1, L1]
```

并计划写出 `*_optimal_SNR_single_*` 和 `*_optimal_SNR_network_*`。这些新名称在 0222/0228 主目录中不存在。

因此可以确认：

- 现存主数组不是当前脚本所声明的输出版本；
- 当前脚本不能作为现存数组的完整 provenance；
- 现存单通道究竟是 H1、L1、网络组合，还是旧循环中最后一次覆盖的探测器，文件本身无法证明。

结合 5 月“修订后保留 detector 轴”的代码结构，强烈怀疑旧代码曾在 H1/L1 循环内写入同一二维槽位，使后一探测器覆盖前一探测器。但因旧生成脚本未保留，这一点应标为**强推断而非已证明事实**。

## 4. 参数表与物理分布

### 4.1 源参数

源参数 CSV 包含：

```text
luminosity_distance,
mass_1_source, mass_2_source,
a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
ra, dec, theta_jn, psi, phase, geocent_time
```

生成先验的主要特点：

- Planck18 宇宙学；
- `z_source` 约 0.01–1；
- 两个 source-frame 质量各自独立均匀采样于 10–70 `M_sun`；
- 自旋幅度 0–0.99，方向为相应角先验；
- 到达时间均匀覆盖 2015-09 至 2025-12；
- 波形为 `IMRPhenomXPHM`，20 Hz 起始频率。

约一半样本违反 `mass_1_source >= mass_2_source` 的常用排序：

- SIS 0222：1,264/2,500；
- SIS 0228：1,239/2,500；
- PM 0222：1,233/2,500；
- PM 0228：1,250/2,500。

这不一定使波形本身无效，但会让质量比定义、条件先验、目录 posterior 对齐和模型输入约定变得不一致。新数据生成器应在采样后显式排序，并同步交换与黑洞编号绑定的自旋参数，或改为直接采样 `chirp_mass` 和 `q <= 1`。

### 4.2 SIS 透镜数据

SIS CSV 字段为：

```text
lens_params.csv:
z_l, z_s, sigma_v, theta_E(arcsec), y, beta_x, beta_y

lens.csv:
mu_0, mu_1, t_d
```

数值检查确认所有 5,000 个正式范围样本满足：

```text
mu_0 = 1 + 1/y
mu_1 = 1 - 1/y
```

最大绝对数值误差低于 `9e-13`。这部分标签代数是一致的。

但采样协议有明显局限：

- `y ~ Uniform(0.01, 0.3)`；
- `sigma_v ~ Uniform(100, 500) km/s`；
- `z_l` 被确定性设为 `z_s / 2`；
- `theta_E`、`beta_x/y` 和 magnification 全部无噪声；
- 没有选择函数、横截面权重或 astrophysical population prior。

SIS `mu_0` 覆盖：

| 区间 | 0222 样本数 | 0228 样本数 | 合计 |
|---|---:|---:|---:|
| `[0, 5)` | 421 | 449 | 870 |
| `[5, 10)` | 1,187 | 1,184 | 2,371 |
| `[10, 20)` | 491 | 485 | 976 |
| `[20, 50)` | 312 | 278 | 590 |
| `[50, 100)` | 89 | 102 | 191 |
| `>= 100` | 0 | 2 | 2 |

这说明高放大率尾部很稀疏：`mu_0 >= 50` 仅 193/5,000，`>=100` 仅 2/5,000。均匀采样 `y` 并不等价于在 `log(mu)` 或科学目标区间均衡采样，因此会产生明显的尾部条件均值收缩。

`mu_rel = |mu_1|/mu_0` 的范围约 0.539–0.980，中位数约 0.73。数据几乎没有低相对放大率的宽覆盖。

SIS 时间延迟：

| 批次 | 最小 | 中位数 | 95% 分位 | 最大 |
|---|---:|---:|---:|---:|
| 0222 | 0.027 d | 23.30 d | 300.35 d | 600.91 d |
| 0228 | 0.026 d | 25.27 d | 305.71 d | 665.07 d |

时间延迟范围可覆盖一部分 galaxy-lens 搜索窗口，但 `sigma_v` 上限 500 km/s、固定 `z_l=z_s/2` 和无选择效应使其不能被解释为可信的星系透镜总体。

### 4.3 解析捷径和 oracle 输入

当前表中直接保存 `y`、`beta_x/y` 和 `theta_E`，且满足：

```text
y = sqrt(beta_x**2 + beta_y**2) / theta_E
mu_0 = 1 + 1/y
```

因此，只要模型输入包含这几个字段，`mu_0` 就能在机器精度下解析恢复。这不是普通意义上的随机 target leakage，但它是**确定性 oracle shortcut**：

- `beta_x/y` 是源平面潜变量，不是直接观测的光学像位置；
- 数据没有光学成像像素、PSF、lens light、arc/source reconstruction；
- 没有从观测像位置到 `beta` 的 lens-model posterior；
- 没有测量误差或协方差。

所以这批几何字段绝不能直接列入 deployable input allowlist。它们最多用于 privileged diagnostics 或 SIS 解析基线标签验证。

正确的新输入应优先使用有噪声的观测量，例如双像相对 lens center 的位置、Einstein scale 的测量 posterior、速度色散测量及其协方差，而不是精确 `y` 或精确 `beta`。

### 4.4 点质量（PM）数据

PM 模型使用：

- `m_l ~ Uniform(1e8, 1e10) M_sun`；
- `y ~ Uniform(0.01, 0.3)`；
- 点质量双像解析 magnification 和 delay；
- 同样固定 `z_l=z_s/2`。

0222/0228 的延迟中位数分别约 0.359/0.343 d，最大约 1.95/2.04 d。

该数据在数学上可作为点质量双像基线或 OOD 例子，但它不是一个真实的扩展星系质量分布。对本项目的“星系尺度强透镜绝对放大率”主问题，PM 不应与 SIS/SIE/power-law mixture 混作同一 lens-family training population。

## 5. GW 波形和噪声管线

### 5.1 做对的部分

当前 5 月修订代码具有以下可保留思想：

- 使用 `IMRPhenomXPHM`，包含进动和高阶模；
- 两个像共享源参数；
- 第二像的 catalog 到达时间设置为第一像加 `t_d`；
- 单像波形乘 `sqrt(abs(mu_i))`；
- saddle image 使用 `n=1/2`，相位因子为 `exp(-i*pi/2)`；
- 每像使用不同随机种子生成噪声；抽查噪声相关接近 0；
- 5 月修订版区分单探测器 SNR 和网络 SNR；
- 使用 memmap 和临时文件后移动，避免一次性把全部数据装入内存。

### 5.2 不满足新任务的部分

1. **不是真实噪声。** `set_strain_data_from_power_spectral_density` 生成的是依据 Bilby 内置 detector PSD 的合成高斯噪声。随机 `geocent_time` 不会把它变成真实 O1–O4 strain。
2. **没有 GWOSC 数据质量。** 没有 DQ mask、hardware injection veto、GPS segment manifest、校准版本或 checksum。
3. **没有数据驱动 PSD。** 没有 off-source PSD 估计、on-source exclusion 或 PSD mismatch 实验。
4. **探测器不足。** 当前脚本只有 H1/L1；正式任务要求可用时包含 V1，并记录 detector availability。
5. **现存数据丢失 detector 轴。** 即使修订脚本解决了未来输出，旧 NPY 仍不能修复为网络数据。
6. **时间对齐改变表示。** 每个信号峰值被强制移到固定索引，ringdown 后被人为淡出；这适合作为固定窗口分类输入，但需验证不会改变参数后验或制造易学的边界特征。
7. **白化约定未版本化。** 使用 Bilby 白化频域数组后直接 `irfft`，但没有独立测试幅度比例、Parseval/PSD 归一化或跨版本稳定性。
8. **没有校准不确定性。** 不含 detector calibration response nuisance。
9. **没有 selection/detection protocol。** 几乎所有透镜双像都高 SNR；缺少第二像接近阈值或未探测的真实困难区域。

### 5.3 SNR 分布过于容易

现存 SNR 是单个未标明 detector/network 的标量，因此以下数值只能描述旧数组，不能解释为网络 SNR：

| 数据 | 第一像 SNR 中位数 | 第二像 SNR 中位数 | 两像均 `>=8` |
|---|---:|---:|---:|
| SIS 0222 | 115.5 | 98.4 | 99.1% |
| SIS 0228 | 116.7 | 99.7 | 99.0% |
| PM 0222 | 83.7 | 72.0 | 97.8% |
| PM 0228 | 83.8 | 75.5 | 97.8% |

这远不能代表实际 catalog 中低 SNR 第二像、不同 detector network 和 detection selection 的困难情形。新基准必须显式分层 SNR，并包含大量靠近检测阈值的第二像和单像可见场景。

## 6. 多信使信息缺口

当前数据并非真正的 multimessenger dataset。它没有：

- 光学成像像素；
- 图像 PSF、曝光、背景、lens light 或 source light；
- 观测像位置及 astrometric covariance；
- lens center 的测量及协方差；
- 有噪声的 Einstein radius posterior；
- lens/source photometric 或 spectroscopic redshift uncertainty；
- stellar velocity-dispersion aperture、seeing、模板误差；
- time-delay measurement uncertainty；
- missing/censored modality mask；
- lens family uncertainty；
- external shear、external convergence、line-of-sight nuisance；
- mass-sheet transform 或 model discrepancy variable。

SIS 中虽有 `sigma_v`、`z_l/z_s`、`theta_E`、`beta`，但它们全是无误差 latent truth。它们不能被宣传为真实 EM constraints。

## 7. 数据划分、泄漏和独立性风险

### 7.1 已确认

- 0222 和 0228 的 source CSV 不相同，未发现完全相同的整行源参数。
- 0222 SIS 透镜随机序列可由 seed 6130 精确复现前三行；PM 可由 seed 614 精确复现前三行。
- 0228 与 0222 是不同数值样本，但远端没有 0228 对应脚本或 seed 记录，因此“不同随机种子”的具体值无法验证。
- 每个 `test_source_samples_*` 是从对应透镜源衍生；同一基础源产生多个近邻，必须 grouped split。
- `lensed_source_samples.csv` 的前后两半是同一物理源的两个像，不能按行随机切分。

### 7.2 高风险设计

若将不同像、hard negatives、时频图增强版本或同一源的派生数据先拼接再随机按行切分，会产生严重的 source-level leakage。正确 group key 至少应包含：

```text
source_realization_id
lens_realization_id
image_pair_id
noise_segment_id
augmentation_parent_id
```

训练/验证/测试不能共享这些 group。对于 lens generalization 测试，还要按 lens realization 或 lens parameter region 分组。

当前数据没有这些显式 ID，只能依靠行位置隐式配对，脆弱且容易在拼接后丢失。

## 8. 派生时频图像

远端有六套 224×224 RGBA PNG：

- SIS/PM 的 noisy CQT；
- SIS/PM 的 pure CQT；
- SIS 的 noisy/pure Mel。

每套共 5,000 张，目录分为 `lensed` 和 `unlensed`。关键问题：

- 在透镜相关目录中没有找到生成这些 PNG 的脚本；
- 没有 manifest 连接 PNG、原始 NPY、事件 ID、像编号和随机种子；
- 文件名仅为 `pos_XXXX.png` 等弱标识；
- RGBA 渲染图可能包含 colormap、坐标裁剪和逐图归一化，可能破坏幅度信息；
- 不能确认这些图来自 0123、0222 还是其他混合数据。

因此这些 PNG 只适合保留旧分类结果，不适合作为绝对放大率后验的主输入。若确需时频表示，应在新管线中以数值 tensor 生成，明确归一化，并用单元测试证明 `sqrt(mu)` 的相对幅度没有被逐图标准化抹掉。

## 9. 生成代码审计

### 9.1 代码结构

`data_gen_check` 中有六个大型 notebook-export 风格脚本：

- LIGO：SIS、PM、unlensed；
- ET：SIS、PM、unlensed。

所有脚本 AST 可解析。已有 smoke 运行生成了 5 个事件级的小数据，并有 `smoke_qc_summary.csv`，其检查结果为 `overall_ok=True`。

但是 smoke QC 只检查：

- 预期文件是否存在；
- NPY 是否无 NaN/Inf；
- test luminosity distance 是否为正。

它没有检查：

- SIS/PM 解析恒等式；
- detector axis 和 detector labels；
- Morse/parity convention；
- 图像顺序和 time-delay 顺序；
- 幅度是否按 `sqrt(abs(mu))` 缩放；
- whitening 是否保持幅度比例；
- source/lens/noise grouped split；
- PSD 和真实噪声；
- posterior calibration。

### 9.2 可复现性问题

- 目标目录没有 Git 仓库；
- 没有与透镜脚本对应的 `requirements.txt`、lock file、Dockerfile 或 environment YAML；
- 没有数据 manifest 和 checksums；
- 没有记录生成代码 hash 到每批数据；
- 输出路径硬编码为绝对路径；
- 采样参数散落在脚本顶层；
- 脚本导入后即执行，不能作为可测试模块复用；
- 没有 CLI、配置 schema、resume state 或 accepted-count control；
- `backup_original` 与当前六个脚本 SHA256 完全相同，不能提供真正的历史版本；
- 0228 对应代码未找到。

当前可运行 base 环境版本为：Python 3.12.12、NumPy 1.26.4、Pandas 3.0.2、Bilby 2.8.0、Astropy 7.2.0、lenstronomy 1.13.5、GWpy 3.0.13、LAL 7.7.0、LALSimulation 6.2.0。但这些是审计时环境版本，不等于 2 月生成环境版本。

### 9.3 代码中的科学设计问题

1. `z_l=z_s/2` 是确定性简化，过度收缩 lens-redshift 条件空间。
2. `y` 和 lens strength 简单均匀采样，没有 proposal/astrophysical prior 分离。
3. lens truth 和 EM observable 没有区分。
4. 只有 SIS/PM，没有非 SIS 星系模型。
5. 无 detection selection、importance weight 和 posterior-prior correction。
6. 无 calibration split 或 held-out OOD split。
7. test hard negatives通过对 `d_L/sqrt(mu)` 加 `Uniform(-100,100) Mpc` 构造，物理总体和任务定义都较人为。
8. 代码大量逐事件打印，长任务日志巨大且不利于 resume。
9. Bilby 2.8 smoke 日志出现 unused waveform kwargs 的弃用警告；当前仍能运行，但未来版本可能变为错误。
10. 生成时将完整 time array 为每事件重复保存，存储效率较低；应改为共享 sampling metadata 加每事件 start time。

## 10. 对新研究任务逐项适用性

| 目标 | 当前状态 | 结论 |
|---|---|---|
| 遗留 SIS 点回归/解析基线 | 有精确 SIS 标签与干净/带噪波形 | 可保留，但需明确 legacy |
| SIS 解析不确定性传播 | 只有精确 truth，无观测 covariance | 需新建 noisy-observable 层 |
| GW 双像相对信息 | 有成对模拟波形和 Morse 相位 | 可做早期 smoke；旧数组单 detector 未知 |
| 绝对放大率 posterior | 无 posterior target protocol、无 EM likelihood | 不适合 |
| EM-only Bayesian lens model | 无真实成像/运动学似然 | 不适合 |
| SIE/power-law/shear | 无数据无代码 | 不适合 |
| mass-sheet/external convergence | 无 nuisance | 不适合 |
| 模态缺失与 covariance | 无 mask/covariance | 不适合 |
| posterior calibration/SBC | 无 posterior 模型和校准 split | 不适合 |
| 真实 detector noise | 无 GWOSC strain | 不适合 |
| GWTC anchor injection | 无 catalog/PE/segment provenance | 不适合 |
| 实际 GWTC pair scan | 无 catalog 数据 | 不适合 |
| H1/L1/V1 network | 旧数组无 detector 轴；新脚本仅 H1/L1 | 不适合 |
| OOD lens-family mismatch | 仅 SIS/PM，PM 不是扩展星系模型 | 不适合 |

## 11. 建议的数据处置

### 11.1 可以保留

建议将以下内容登记为 immutable legacy artifacts：

- 0222/0228 的 CSV 参数和标签；
- 0222/0228 的干净和合成噪声波形；
- 5 月修订脚本和 smoke 输出；
- SIS/PM 解析公式与 Morse 相位实现。

但应添加 manifest，记录“原始生成代码未知/不完整”“单通道 detector 身份未知”“高斯设计噪声”等限制。

### 11.2 不应进入主结果

- 现存 `*_data_strain*.npy` 不能充当 real-noise benchmark；
- `beta_x/y`、`y` 不能充当 deployable observables；
- 时频 PNG 不能用于绝对幅度主模型，除非重建生成 provenance 并验证归一化；
- PM 数据不能代表扩展星系透镜族；
- hard-negative `test_*` 不能作为独立 held-out astrophysical test set；
- 0228 不能在补齐生成 provenance 前被称为“完全可复现实验批次”。

### 11.3 不要原地修复

不要修改、重命名或覆盖 0222/0228。新项目应在独立目录生成 v2 数据，使用新的 schema 和 manifest。旧数据只通过只读 adapter 访问。

## 12. v2 数据最小设计要求

建议新数据至少拆成下列逻辑表/数组：

### 12.1 Truth 表（privileged，训练输入禁止）

```text
source_truth
lens_truth
image_truth
signed_mu, abs_mu, amplitude_factor
parity, morse_index
true_time_delay
external_convergence
proposal_log_prob
astrophysical_log_prior
```

### 12.2 Observable 表

```text
GW strain by detector and image
detector availability mask
event observed time and uncertainty
observed image positions relative to observed lens center
Einstein-scale posterior or summary
lens/source redshift observations and covariance
velocity-dispersion observation, aperture metadata and uncertainty
modality mask
```

### 12.3 Provenance 表

```text
dataset_version
generator_git_commit
config_hash
seed hierarchy
waveform model and package versions
lens solver and package versions
GWOSC release, event version, GPS segment and checksum
PSD method and off-source interval
calibration/data-quality metadata
source_id, lens_id, pair_id, image_id, noise_segment_id
split assignment
```

### 12.4 必须新增的 lens families

1. SIS，作为解析和极限测试；
2. SIE + external shear；
3. elliptical power-law + external shear；
4. external convergence 或显式 mass-sheet/discrepancy nuisance；
5. lens-family mixture 和 held-out parameter regions。

### 12.5 必须新增的评估分布

- proposal-balanced：按 `log|mu_plus|`、`mu_rel` 等分箱 accepted-count；
- astrophysical evaluation prior；
- balanced tail diagnostic；
- lens-family OOD；
- real-noise GWTC-anchored injection；
- waveform/PSD/calibration mismatch；
- low-SNR second image；
- missing EM modality 和 covariance miscalibration。

## 13. 建议执行优先级

### P0：在任何新训练前完成

1. 为 0222/0228 生成只读 manifest 和文件 checksums；
2. 将旧数组 detector 身份标记为 unknown，禁止伪称 network；
3. 建立 privileged denylist：至少含 `y`、真实 `beta`、真实 magnification、真实 lens parameters；
4. 实现 SIS 解析反演及带观测协方差的 Monte Carlo/Bayesian baseline；
5. 把生成器重构为配置驱动、可测试、可 resume 的模块；
6. 建立 source/lens/noise grouped split；
7. 先生成小规模 SIE/power-law smoke dataset；
8. 写幅度保持、Morse、image ordering、time delay 和 detector-axis 单元测试。

### P1：主方法训练前完成

1. 有噪声 EM observable simulator 与 covariance；
2. lens-family mixture、external convergence 和 discrepancy variable；
3. balanced proposal 与 astrophysical prior 分离；
4. H1/L1/V1 多 detector schema；
5. posterior model、SBC 和 coverage pipeline；
6. Gaussian-noise IID 数据的固定 preregistered splits。

### P2：论文主结果前完成

1. GWOSC real-noise segments、DQ/hardware-injection 筛选和 PSD；
2. GWTC 最新稳定目录 snapshot 与 PE products；
3. GWTC-anchor injections、counterfactual companions 和 actual pair scan 分离；
4. 传统 Bayesian joint baseline；
5. lens/waveform/PSD/EM covariance mismatch 的覆盖率评估。

## 14. 最终判定

最稳妥的研究决策不是继续用现有数据调网络，而是：

> 把 0222/0228 冻结为 legacy synthetic benchmark；保留其 SIS 解析一致性和成对波形生成经验；重新生成带 detector 轴、可观测 EM 噪声、非 SIS 星系透镜族、proposal/prior 分离、group IDs 和完整 provenance 的 v2 数据；真实噪声部分从 GWOSC 独立构建。

现有数据能证明的是“在简化 SIS/PM、设计高斯噪声和精确潜变量条件下生成了双像模拟波形”。它不能证明“模型能从真实 GW+EM 观测中得到校准的、模型条件绝对放大率后验”。两者之间仍需完整的数据与推断管线重建。

## 15. 证据置信度

### 已验证

- 文件数量、大小、NPY shape/dtype；
- CSV 行列和全量分布；
- SIS 解析恒等式；
- 0222 SIS seed 6130、PM seed 614 与透镜参数序列一致；
- 0222/0228 源表不同；
- 当前脚本 AST 可解析；
- 现有 smoke QC 的范围和结果；
- 当前脚本使用设计 PSD 合成噪声、H1/L1、IMRPhenomXPHM；
- 当前数据没有 V1、GWOSC manifest、EM covariance 和非 SIS 扩展星系模型。

### 强推断

- 旧二维数组很可能是旧探测器循环覆盖后只保留一个 detector；
- 现存时频图很可能经历了不适合绝对幅度推断的图像归一化或渲染，但生成脚本缺失，不能确定具体方式。

### 未知/不可恢复

- 0228 的确切随机种子；
- 2 月主 NPY 的确切生成代码 commit 和环境；
- 旧二维数组对应哪个 detector 或何种组合；
- 六套 PNG 对应的源批次和精确变换；
- 所有大 NPY 的逐字节完整性（尚未生成全量 checksums）。
