# C1 数据与切分检查（2023-01 HVFHV 全月面板）

> 分支：`c-lgbm-baseline`　执行：成员 C　生成时间：2026-10-22
> 结论：**通过**。面板 / 特征 / 防泄漏 / 时间切分共 16 项检查全部 PASS，0 FAIL，0 WARN；连同预测文件 5 项，`--what all` 共 21 项全 PASS。
> 机器可读报告：`data/processed/logs/data_checks.json`

## 1. 输入文件与环境

| 文件 | 行数 | 时间范围 | 说明 |
| --- | --- | --- | --- |
| `data/processed/demand_panel.parquet` | 782,688 | 2023-01-01 00:00 ~ 2023-01-31 23:45 | 2976 时间片 × 263 区域完整网格 |
| `data/processed/features.parquet` | 782,688 | 同上 | 18 个特征 + `demand` + `time_idx` |
| `data/processed/feature_list.json` | — | — | 与特征表逐列对应 |
| `data/processed/sample_day.parquet` | 25,248 | 2023-01-01 | 96 片 × 263，本地用 `make_sample` 生成 |
| `data/processed/sample_month.parquet` | 757,440 | 2023-01-01 ~ 01-30 | 2880 片 × 263 |

环境：Windows + Python 3.12.10（仓库内 `.venv`），pandas 2.3.3 / numpy 2.5.4 / pyarrow 26.0.0 / lightgbm 4.7.0 / holidays 0.107.0。

```powershell
# 全量重算特征（5.7 s，全月 782,688 行）
python -m src.features.build_features --config configs/experiments/c_baseline.yaml

# 面板 + 特征 + 防泄漏 + 切分 + 预测的统一校验（含 JSON 报告）
python -m src.data.check_data --config configs/experiments/c_baseline.yaml --what all
```

## 2. 面板结构（`panel` 分区）

| # | 检查项 | 结果 | 实测 |
| --- | --- | --- | --- |
| P1 | 表结构与 dtype 符合接口契约 | PASS | `timestamp=datetime64[ns]`、`zone_id=int16`、`demand=int32` |
| P2 | `(timestamp, zone_id)` 唯一 | PASS | 0 条重复键 |
| P3 | 行数 = 时间片 × 区域 | PASS | 782,688 = 2976 × 263 |
| P4 | 时间步长恒为 15min 且无缺口 | PASS | 2975 个相邻间隔全为 00:15:00，无未对齐时间片 |
| P5 | 区域取值落在 1~263 | PASS | 覆盖 1~263，共 263 个，无 264/265 |
| P6 | `demand` 非负整型、无 NaN | PASS | min=0，max=437，sum=18,478,039，mean=23.61，零需求占比 9.02% |

不存在缺行、重复键或越界区域，**无需**向 A 提 Issue。

## 3. 特征表与防泄漏（`features` / `leak` 分区）

| # | 检查项 | 结果 | 实测 |
| --- | --- | --- | --- |
| F1 | `feature_list.json` 声明的 18 个特征齐全 | PASS | `zone_id`、`lag_1/2/4/8/96/672`、`roll_mean_4/96`、`hour/minute/dow/is_weekend/is_holiday`、`hour_sin/cos`、`dow_sin/cos` |
| F2 | 特征无 `inf` | PASS | 无非有限值 |
| F3 | NaN 仅来自暖机前缀 | PASS | lag_1 0.03%、lag_2 0.07%、lag_4 0.13%、lag_8 0.27%、lag_96 3.23%、lag_672 22.58%、roll_mean_4 0.13%、roll_mean_96 3.23%，且严格等于“每区域前 k 行 / 前 w 行” |
| F4 | 特征不是常量列 | PASS | 18 列全部有变化 |
| L1 | 滞后特征 = `demand.shift(k)` | PASS | 6 个 lag 与逐区域 `shift(k)` 在 1e-5 内完全一致 |
| L2 | 滚动均值 = 先 `shift(1)` 再 `rolling(w).mean` | PASS | `roll_mean_4`、`roll_mean_96` 在 1e-4 内完全一致，当前时刻 `demand` 未参与 |

额外抽查：`corr(demand, lag_1) = 0.963`，滞后特征与标签强相关但构造方式正确，不存在“用 t 时刻需求预测 t 时刻需求”的泄漏。

## 4. 与 A 交付文件的一致性

用仓库内 `demand_panel.parquet` 全量重算特征（5.7 s）后，与 A 直接交付的 `features.parquet` 按 `(timestamp, zone_id)` 对齐比较：

- 形状一致（782,688 × 21），索引集合完全一致；
- 18 个特征 + `demand` + `time_idx` 逐列一致（float 容差 1e-6，整型严格相等）；
- `feature_list.json` 的 `features / lags / rolling_windows` 一致；
- `is_holiday` 的节假日集合一致：2023-01-01、2023-01-02（New Year observed）、2023-01-16（MLK）。

结论：仓库内面板与 A 交付的特征表**同源且可本地复现**，后续 C3/C4 不必再向 A 索要特征文件。

## 5. 时间切分（`split` 分区，70% / 15% / 15%，禁止随机划分）

`src.models.common.time_split_bounds` 给出的边界：

| 集合 | 时间范围 | 时间片 | 行数 | 需求总量 |
| --- | --- | --- | --- | --- |
| train | 2023-01-01 00:00 ~ 2023-01-22 16:30 | 2083 | 547,829 | 12,698,966 |
| val | 2023-01-22 16:45 ~ 2023-01-27 08:00 | 446 | 117,298 | 2,768,853 |
| test | 2023-01-27 08:15 ~ 2023-01-31 23:45 | 447 | 117,561 | 3,010,220 |

| # | 检查项 | 结果 | 实测 |
| --- | --- | --- | --- |
| S1 | 三段非空且按时间戳切分 | PASS | `train_end=2023-01-22 16:45`，`val_end=2023-01-27 08:15` |
| S2 | 三段时间戳互不重叠 | PASS | (train,val)=(val,test)=(train,test)=0 |
| S3 | 严格按时间先后 | PASS | train_max 16:30 < val[16:45 ~ 01-27 08:00] < test_min 01-27 08:15 |
| S4 | 每段都是完整网格 | PASS | 547,829 / 117,298 / 117,561 均为“片数 × 263” |

## 6. 已知问题与对后续实验的影响

1. **`lag_672` 暖机 NaN 22.58%**：每区域前 672 行（7 天）为空，等价于该特征从 01-08 起才有效。基线保留这些行（LightGBM 原生处理缺失），但 C3 对比 `A2/A3` 时样本量并不完全相同，结论里需要注明。
2. **只有单月数据**：`lag_672` 只能覆盖同一周内历史，跨周/跨月季节性无法体现；`val/test` 只覆盖 1 月下旬，指标对节假日（春节等）不具代表性。
3. **`sample_day.parquet` 非 A 直接交付**：由 `src.data.make_sample` 从面板截取（96 × 263 = 25,248 行），与任务书 C1 的期望行数一致。
4. **零需求占比 9.02%**：面板已 zero-fill，不做插值；预测侧按契约 `clip(0, None)`，不再单独处理零膨胀。
5. 若重新生成特征，`feature_list.json` 里的 `panel` 字段会写成本机路径（元信息，不参与训练）。

## 7. 如何复跑

```powershell
# 在仓库根目录执行；用仓库内 .venv 可避免污染全局环境
.\.venv\Scripts\python.exe -m src.data.check_data --config configs/experiments/c_baseline.yaml --what all
```

输出逐项 `PASS/FAIL` 日志与 `data/processed/logs/data_checks.json`；任何一项 FAIL 时命令以退出码 1 结束。
