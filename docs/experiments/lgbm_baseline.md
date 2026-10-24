# C2 LightGBM 基线（2023-01 全月 HVFHV）

> 分支：`c-lgbm-baseline`　执行：成员 C　tag：`c_baseline`
> 结论：全月基线跑通且可复现。**test MAE 4.0595 / RMSE 6.5593 / WAPE 15.85%**，`best_iteration=1180`，CPU 训练 76.9 s；预测文件满足 `docs/api_contract.md`，B 可直接消费。

## 1. 实验设置

| 项目 | 取值 |
| --- | --- |
| 数据 | 2023-01-01 00:00 ~ 2023-01-31 23:45，2976 时间片 × 263 区域 = 782,688 行 |
| 切分 | 按时间 70/15/15：train < 2023-01-22 16:45；val 至 2023-01-27 08:15；test 为之后 447 片 |
| 样本量 | train 547,829 / val 117,298 / test 117,561 |
| 标签 | `demand`（15min × zone 上车订单数） |
| 特征 | 18 个：`zone_id`（categorical）、`lag_1/2/4/8/96/672`、`roll_mean_4/96`、`hour/minute/dow/is_weekend/is_holiday`、`hour_sin/cos`、`dow_sin/cos` |
| 目标函数 | `regression_l1`，评估 `mae`，提前停止 100 轮 |
| 训练设备 | CPU（`device=cpu`，`n_jobs=-1`）；本机 Intel i7-9750H / 16 GB，不使用 GPU 版 LightGBM |
| 环境 | Python 3.12.10 / lightgbm 4.7.0 / pandas 2.3.3 / numpy 2.5.4 / pyarrow 26.0.0 |
| 随机种子 | 42（`project.seed`） |

超参数（`configs/experiments/c_baseline.yaml`，与 `default.yaml` 一致，仅 `n_estimators` 放宽）：

| 参数 | 值 | 参数 | 值 |
| --- | --- | --- | --- |
| `learning_rate` | 0.05 | `num_leaves` | 63 |
| `min_child_samples` | 50 | `subsample` / `freq` | 0.8 / 1 |
| `colsample_bytree` | 0.8 | `reg_alpha` / `reg_lambda` | 0.0 / 1.0 |
| `n_estimators` | 2000 | `early_stopping_rounds` | 100 |

> 与任务书的差异：任务书 C2 要求 `configs/experiments/c_sample_month.yaml`（30 天样本、800 轮）。本分支改用**全月**，因为 A 交付的特征表本身就是全月（且已验证可由仓库内面板复现，见 [`data_checks.md`](data_checks.md) §4），重算只需 5.7 s，而只有全月才能让 `pred_test.parquet` 覆盖契约要求的完整测试期。`n_estimators` 相应放宽到 2000 并依赖早停。

## 2. 运行命令

```powershell
# 0) 环境（仅首次）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1) 特征（全月，5.7 s）
python -m src.features.build_features --config configs/experiments/c_baseline.yaml

# 2) 训练 + 验证集早停 + 测试集预测（约 2 分钟）
python -m src.models.train_lgbm --config configs/experiments/c_baseline.yaml --tag c_baseline

# 3) 预测文件契约校验（会顺便复算测试集指标）
python -m src.data.check_data --config configs/experiments/c_baseline.yaml --what predictions
```

## 3. 结果

### 3.1 指标

| 集合 | 行数 | MAE | RMSE | WAPE | sMAPE |
| --- | --- | --- | --- | --- | --- |
| train | 547,829 | 3.7554 | 6.2421 | 16.20% | 41.63% |
| val | 117,298 | 3.9879 | 6.6800 | 16.89% | 43.58% |
| **test** | 117,561 | **4.0595** | **6.5593** | **15.85%** | 40.34% |

- `best_iteration = 1180`（`early_stopping_rounds=100` 触发），`train_seconds = 76.9`，整条命令墙钟约 122 s。
- 三段 WAPE 都在 16%~17%，train→val 只升高 0.7pp，**没有明显过拟合**；test 反而略低（15.85%），是测试期（01-27 周五 ~ 01-31 周二）需求波动小于验证期所致，不是信息泄漏（切分与特征均已校验）。
- 测试期真实需求均值 25.61，预测均值 25.59，整体偏差很小；误差主要体现在单点波动而非系统性偏移。

### 3.2 特征重要性（gain）

| 排名 | 特征 | gain 占比 | 排名 | 特征 | gain 占比 |
| --- | --- | --- | --- | --- | --- |
| 1 | `roll_mean_4` | 32.41% | 6 | `hour` | 1.78% |
| 2 | `zone_id` | 24.61% | 7 | `lag_96` | 1.65% |
| 3 | `roll_mean_96` | 20.84% | 8 | `dow` | 1.55% |
| 4 | `lag_1` | 8.42% | 9 | `lag_672` | 1.47% |
| 5 | `lag_2` | 2.63% | 10 | `lag_4` | 1.01% |

解读：短期滚动均值（1h）+ 区域身份 + 日周期滚动均值（24h）三者合计贡献约 78%，日历类特征（`is_weekend`/`is_holiday`/`minute`）几乎无贡献——与“需求主要由近期水平和区域强度决定”的直觉一致。这为 C3 的特征组消融给出预期：`A0`（仅 zone_id + 日历）应当明显差于含滚动特征的 `A4`。

## 4. 预测文件（接口契约校验）

`data/processed/predictions/pred_test_lgbm_c_baseline.parquet`（同时复制为 B 的消费路径 `pred_test.parquet`）：

| 检查项 | 结果 |
| --- | --- |
| 列 | `timestamp`(datetime64[ns])、`zone_id`(int16)、`predicted_demand`(float32)、`model_name`("lgbm") |
| 行数 / 覆盖 | 117,561 = 447 时间片 × 263 区域，2023-01-27 08:15 ~ 2023-01-31 23:45 |
| 取值 | min 0.0，max 386.27，无负值、无 NaN、键唯一 |
| 指标复核 | 用预测文件对 `features.parquet` 的 `demand` 复算：MAE 4.0594 / RMSE 6.5593 / WAPE 15.85%，与 `metrics.json` 一致 |

B 侧只需读取 `data/processed/predictions/pred_test.parquet`，字段与 `docs/api_contract.md` §3.1 完全对应。

## 5. 可复现性

- 同一命令（`--tag c_baseline_repro`）重跑一次，train/val/test 的 MAE、RMSE、WAPE、sMAPE **逐位相同**（LightGBM 固定 seed + 固定线程数下确定）。
- 产物：`data/processed/models/lgbm_c_baseline.txt`（不入库）、`lgbm_c_baseline.metrics.json`（含 `train_seconds`、`rows`、`versions`、`best_iteration`）、`lgbm_feature_importance.parquet`。
- 跨机器复现时线程数可能不同，末位小数可能有微小差异；比较指标请保留 4 位小数。

## 6. 已知问题与下一步

1. **`lag_672` 暖机缺失**：前 7 天该特征为 NaN（全表 22.58%），训练集前一周的样本只能用其余特征；C3 做 `A2/A3` 对比时需说明样本量差异。
2. **单月数据**：只有 31 天，无法评估跨月/节假日鲁棒性；结论只对 2023-01 成立。
3. **误差集中在高需求区域**：Top 区域 `demand` 峰值 437，而预测峰值 386，高需求端被轻度低估；分区域/分时段误差分析留到 C5（`error_by_hour/zone`）。
4. **C3 已完成**：特征组与损失消融（A0~A7）见 [`lgbm_ablation.md`](lgbm_ablation.md) —— 结论是滞后特征贡献 99.2% 的提升，A1~A4 差异在噪声内，推荐 A2（15 特征 + L1）；A4 与本基线逐位一致，可作为本报告结果的交叉验证。
