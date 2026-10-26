# C4 小规模超参数调优（LightGBM 随机搜索）

> 分支：`feature/c-tuning`　执行：成员 C　搜索日期：2026-10-26
> 结论先行：18 组随机搜索把验证集 WAPE 从 16.85% 降到 **16.75%**（−0.10pp），测试集从 15.94% 降到 **15.74%**（−0.21pp）；提升不大但方向稳定。唯一有明显方向性的参数是 **`num_leaves`（越大越好）**，其次是 `learning_rate=0.05`、`colsample_bytree=0.8`；其余参数落在噪声内。最佳配置：`lr=0.05, num_leaves=127, min_child_samples=100, colsample=0.8, subsample=1.0, λ=0`。

## 1. 实验设置

| 项目 | 取值 |
| --- | --- |
| 特征集 | **固定为 C3 推荐的 A2**（15 个特征：`zone_id` + 9 个日历 + `lag_1/2/4/8/96`），由 `c_ablation.yaml` 的 A2 解析，未重新选特征 |
| 数据 | 2023-01 全月面板，782,688 行（2976 片 × 263 区域） |
| 切分 | 按时间 70/15/15：train 547,829 / val 117,298 / test 117,561（与 C1~C3 完全一致） |
| 训练预算 | `n_estimators=6000` + `early_stopping_rounds=100`（与 C3 相同），`metric=mae`，`objective=regression_l1` |
| 设备 | CPU（`n_jobs=-1`），Intel i7-9750H / 16 GB；未使用 GPU 版 LightGBM |
| 选择口径 | **只用验证集**：val WAPE 最小 → val MAE 最小 → trial_id 最小；**搜索阶段不计算任何测试集指标** |
| 环境 | Python 3.12.10 / lightgbm 4.7.0 / pandas 2.3.3 / numpy 2.5.4 / pyarrow 26.0.0 |
| 耗时 | 18 组搜索 39.6 min；最佳组最终训练（含 3 个切分预测）163 s |

## 2. 搜索设计

搜索空间严格按任务书 C4：

| 参数 | 候选值 |
| --- | --- |
| `learning_rate` | 0.03 / 0.05 / 0.08 |
| `num_leaves` | 31 / 63 / 127 |
| `min_child_samples` | 20 / 50 / 100 |
| `colsample_bytree` | 0.7 / 0.8 / 1.0 |
| `subsample` | 0.7 / 0.8 / 1.0 |
| `reg_lambda` | 0 / 1 / 5 |

- **18 组 = 1 个锚点 + 17 组随机采样**：trial 0 固定为 C2/C3 的默认超参（锚点，用于量化“调参到底有没有用”），其余 17 组由 `random.Random(seed=42)` 从 3⁶ = 729 个组合中采样并去重。
- 固定项：`subsample_freq=1`、`reg_alpha=0`（搜索空间外，保持与基线一致）。
- 选参规则在开跑前登记（写进 `c_tuning.yaml` 的 `search.selection`），事后不改。

```powershell
# 抽样计划（秒级，不训练）
.\.venv\Scripts\python.exe -m src.models.run_tuning --config configs/experiments/c_tuning.yaml --dry-run
# 正式搜索（18 组，约 40 min）
.\.venv\Scripts\python.exe -m src.models.run_tuning --config configs/experiments/c_tuning.yaml
# 只重跑最佳组终评（不重跑搜索）
.\.venv\Scripts\python.exe -m src.models.run_tuning --config configs/experiments/c_tuning.yaml --refit
```

## 3. 结果（按验证集 WAPE 升序）

| # | lr | leaves | mcs | colsample | subsample | λ | val MAE | val RMSE | val WAPE | best_iter | 训练 s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **11** | **0.05** | **127** | **100** | **0.8** | **1.0** | **0** | **3.9531** | **6.6064** | **16.75%** | 2122 | 98.3 |
| 7 | 0.03 | 127 | 50 | 0.8 | 0.8 | 0 | 3.9560 | 6.6058 | 16.76% | 3618 | 148.9 |
| 10 | 0.05 | 127 | 20 | 0.8 | 0.7 | 5 | 3.9588 | 6.6171 | 16.77% | 2513 | 112.1 |
| 9 | 0.05 | 63 | 100 | 0.8 | 0.7 | 5 | 3.9616 | 6.6349 | 16.78% | 3713 | 133.2 |
| 17 | 0.03 | 63 | 50 | 0.8 | 1.0 | 5 | 3.9646 | 6.6232 | 16.80% | 5978 | 215.1 |
| 16 | 0.08 | 127 | 20 | 1.0 | 1.0 | 0 | 3.9663 | 6.6213 | 16.80% | 751 | 37.1 |
| 0 | 0.05 | 63 | 50 | 0.8 | 0.8 | 1 | 3.9772 | 6.6614 | 16.85% | 2391 | 89.2 |
| 6 | 0.05 | 31 | 50 | 1.0 | 0.8 | 0 | 3.9840 | 6.6576 | 16.88% | 5987 | 173.4 |
| 5 | 0.08 | 31 | 100 | 1.0 | 1.0 | 5 | 3.9930 | 6.6790 | 16.92% | 2676 | 89.5 |
| 15 | 0.08 | 63 | 100 | 1.0 | 1.0 | 0 | 3.9931 | 6.6957 | 16.92% | 944 | 38.5 |
| 8 | 0.03 | 63 | 20 | 0.7 | 0.8 | 0 | 3.9933 | 6.6881 | 16.92% | 5026 | 180.1 |
| 14 | 0.08 | 63 | 20 | 0.8 | 0.8 | 0 | 3.9973 | 6.6958 | 16.93% | 1771 | 64.1 |
| 1 | 0.08 | 31 | 20 | 1.0 | 0.8 | 0 | 4.0005 | 6.6852 | 16.95% | 4610 | 137.1 |
| 2 | 0.03 | 31 | 100 | 0.7 | 1.0 | 5 | 4.0106 | 6.7123 | 16.99% | 5999 | 195.9 |
| 13 | 0.03 | 31 | 20 | 0.8 | 0.8 | 1 | 4.0236 | 6.7339 | 17.04% | 5973 | 190.8 |
| 12 | 0.08 | 31 | 20 | 1.0 | 0.7 | 1 | 4.0258 | 6.7528 | 17.06% | 3805 | 114.9 |
| 3 | 0.08 | 31 | 100 | 0.8 | 0.7 | 0 | 4.0415 | 6.7938 | 17.12% | 2691 | 91.1 |
| 4 | 0.03 | 31 | 20 | 1.0 | 1.0 | 0 | 4.0886 | 6.8739 | 17.32% | 1136 | 39.3 |

说明：`best_iter=5999/5998/5987` 的 3 组（#2、#13、#17）顶到了 6000 轮上限，仍未完全收敛；它们都不是最优组，故不影响选参结论。

### 3.1 参数敏感性（18 组的均值 / 该值下的最好成绩）

| 参数 | 值 | 组数 | 平均 val WAPE | 最好 val WAPE |
| --- | --- | ---: | ---: | ---: |
| `num_leaves` | 31 / 63 / **127** | 8 / 6 / 4 | 17.03% / 16.87% / **16.77%** | 16.88% / 16.78% / **16.75%** |
| `learning_rate` | 0.03 / **0.05** / 0.08 | 6 / 5 / 7 | 16.97% / **16.81%** / 16.96% | 16.76% / **16.75%** / 16.80% |
| `min_child_samples` | 20 / **50** / 100 | 8 / 4 / 6 | 16.97% / **16.82%** / 16.91% | 16.77% / 16.76% / **16.75%** |
| `colsample_bytree` | 0.7 / **0.8** / 1.0 | 2 / 9 / 7 | 16.95% / **16.87%** / 16.98% | 16.92% / **16.75%** / 16.80% |
| `subsample` | 0.7 / 0.8 / 1.0 | 4 / 7 / 7 | 16.93% / 16.90% / 16.93% | 16.77% / 16.76% / **16.75%** |
| `reg_lambda` | **0** / 1 / 5 | 10 / 3 / 5 | **16.93%** / 16.98% / 16.85% | **16.75%** / 16.85% / 16.77% |

- **唯一单调的参数是 `num_leaves`**：31→63→127 的平均 WAPE 依次下降 0.16pp / 0.10pp，说明在本数据上“更深的树”确实更好（A2 特征少、单棵树容易欠拟合）。
- `learning_rate=0.05` 明显优于 0.03 与 0.08（0.03 需要更多轮、0.08 收敛过快）；`colsample=0.8` 优于 1.0。
- `min_child_samples` / `subsample` / `reg_lambda` 的组间差异 ≤0.05pp，**属于噪声**，不要据此做结论；`reg_lambda=5` 的均值最好但单组最优来自 λ=0，样本量（10/3/5）不足以区分。

## 4. 最佳配置与提升

最佳组 **trial 11**（也是 `configs/experiments/lgbm_best.yaml` 的内容）：

```yaml
objective: regression_l1
metric: mae
n_estimators: 6000
early_stopping_rounds: 100
learning_rate: 0.05
num_leaves: 127
min_child_samples: 100
colsample_bytree: 0.8
subsample: 1.0
subsample_freq: 1
reg_alpha: 0.0
reg_lambda: 0.0
device: cpu
```

| 对比 | val MAE | val WAPE | test MAE | test RMSE | test WAPE | 相对提升（test MAE） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C3 A2（= 锚点 trial 0） | 3.9772 | 16.85% | 4.0824 | 6.5962 | 15.94% | — |
| **C4 最佳（trial 11）** | **3.9531** | **16.75%** | **4.0297** | **6.5034** | **15.74%** | **−1.29%** |
| C2 基线（A4 特征，默认超参） | 3.9879 | 16.89% | 4.0595 | 6.5593 | 15.85% | −0.73%（相对 C2） |

- 相对 A2：val MAE −0.61%、val WAPE −0.10pp；test MAE −1.29%、test RMSE −1.41%、test WAPE −0.21pp。
- 相对 C2 基线：test MAE −0.73%、test RMSE −0.85%、test WAPE −0.12pp。
- 提升幅度不大（约 1%），但**在验证集与测试集、MAE/RMSE/WAPE 三个指标上方向一致**，因此不是偶然波动。

## 5. 测试集一次性评估与可复现性

- 搜索阶段（18 组）只写 `val_*`，`lgbm_tuning.parquet` 中**没有测试集列**；测试集仅在 trial 11 选出后由 `fit_best()` 计算一次，写入 `lgbm_best.metrics.json` 与 `lgbm_best.yaml` 的 `search.test`。
- **内部一致性断言**：`fit_best()` 会用同一组参数重训，并与搜索阶段的 val 指标逐位比对；差值 > 1e-9 直接报错退出（本次通过）。
- **重跑复现**：`--refit` 独立跑两次，train/val/test 的 MAE、RMSE、WAPE 共 9 项**逐位相同**，`best_iteration` 同为 2122；只有 `train_seconds` 因机器负载不同（98.3 s vs 100.8 s）。
- 模型与产物：`data/processed/models/c_best/lgbm_best.txt`（另存 `lgbm_latest.txt` 供 `predict.py`）、`lgbm_best.metrics.json`、`data/processed/feature_list_best.json`（15 特征）、`data/processed/predictions/pred_test_lgbm_best.parquet`。
- 最佳预测已用 C1 的契约校验器复查：`python -m src.data.check_data --config configs/experiments/lgbm_best.yaml --what predictions --predictions .../pred_test_lgbm_best.parquet` → 5/5 PASS（117,561 行 = 447 × 263、非负、无 NaN、复算 MAE 4.0295）。

## 6. 结论

1. **调参有用，但边际收益有限**：18 组只换来 val WAPE −0.10pp / test WAPE −0.21pp。真正的差距来自特征（C3 已证明滞后特征贡献 99%），调参属于“最后 1%”的优化。
2. **只有 `num_leaves` 值得继续试**：31→127 单调改善；若后续还要调，优先把 `num_leaves` 扩到 255、并固定 `lr=0.05、colsample=0.8`，而不是继续扫 `subsample`/`reg_lambda`。
3. 采用 trial 11 作为最终 LightGBM 配置；C5 误差分析即基于该模型。

## 7. 已知问题

- 18 组相对 729 个离散组合只覆盖 ~2.5%，随机搜索的置信度有限；表中的“参数敏感性”只能看趋势，不能做显著性结论。
- 搜索预算固定 6000 轮，3 组（#2/#13/#17）触顶未完全收敛；这会让它们的成绩略偏保守。
- 单月数据（2023-01），结论不保证跨月/跨季节成立。
- 未搜索 `reg_alpha`、`max_depth`、`min_child_weight` 等参数（任务书未要求，且算力有限）。

## 8. 产物

| 产物 | 路径 | 入库 |
| --- | --- | --- |
| 搜索配置 | `configs/experiments/c_tuning.yaml` | 是 |
| 最佳配置 | `configs/experiments/lgbm_best.yaml` | 是 |
| 搜索脚本 | `src/models/run_tuning.py` | 是 |
| 搜索长表（18 行，仅 val） | `data/processed/experiments/lgbm_tuning.parquet` | 否 |
| 搜索清单（含一次性 test 指标） | `data/processed/experiments/lgbm_tuning.manifest.json` | 否 |
| 最佳模型 / 指标 / 预测 | `data/processed/models/c_best/`、`data/processed/predictions/pred_test_lgbm_best.parquet` | 否 |
