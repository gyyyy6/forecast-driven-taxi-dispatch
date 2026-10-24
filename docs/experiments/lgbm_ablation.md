# C3 特征组与损失函数消融（2023-01 全月 HVFHV）

> 分支：`feature/c-ablation`　执行：成员 C　生成时间：2026-10-24
> 结论先行：**滞后特征是唯一决定性因素**（A0→A1 贡献了 A0→A4 全部提升的 99.2%）；A1~A4 之间的差异都在 ±0.09pp 内，“特征更多”并不更好；**日历**的不可替代性高于 `zone_id`（剥离代价 1.15pp vs 0.28pp）；**L1 vs L2** 在验证集/测试集上方向相反、量级 ≤1.5%，保留 L1；按预登记规则推荐 **A2（15 特征 + regression_l1）**。

## 1. 实验设置

| 项目 | 取值 |
| --- | --- |
| 数据 | 2023-01-01 ~ 2023-01-31 全月面板，2976 时间片 × 263 区域 = 782,688 行（与 C1/C2 同一份 `features.parquet`） |
| 切分 | 按时间 70/15/15（`src.models.common.time_split_bounds`）：train < 2023-01-22 16:45；val 至 2023-01-27 08:15；test 为之后 447 片 |
| 样本量 | train 547,829 / val 117,298 / test 117,561（8 组完全相同） |
| 特征来源 | 只算一次特征，训练侧按列选取子集 —— 各组特征**取值完全一致**，唯一变量是列集合与损失 |
| 训练预算 | `n_estimators=6000` + `early_stopping_rounds=100`（见下方说明） |
| 超参数 | `learning_rate=0.05`、`num_leaves=63`、`min_child_samples=50`、`subsample=0.8(+freq 1)`、`colsample_bytree=0.8`、`reg_alpha=0.0`、`reg_lambda=1.0`、`n_jobs=-1`、`device=cpu`（与 C2 基线逐项一致） |
| 随机种子 | 42 |
| 早停指标 | 统一 `mae`（保证 8 组的“选择口径”一致；对 L2 组略不利，见 §6） |
| 环境 | Windows + Python 3.12.10，lightgbm 4.7.0 / pandas 2.3.3 / numpy 2.5.4 / pyarrow 26.0.0，Intel i7-9750H CPU |
| 耗时 | 8 组共 1,698.6 s（约 28.3 min，含每组 3 个切分的预测与指标） |

**为什么把训练上限从 2000 提到 6000**：第一次用基线同款 2000 轮跑完时，A0（`best_iteration=2000`）以及 A1/A2/A3（1994/1999/1999）都顶到上限，验证集 l1 仍在缓慢下降，属于“训练预算不足”。放宽到 6000 后 A1/A2/A3 分别在 3452/2391/2594 轮正常早停，A4~A7 的早停点（1180/1624/1238/941）完全不变。因此最终结果用 6000 轮上限，**A4 仍与 C2 基线逐位一致**（见 §6）。

指标口径：表中 `mae / rmse / wape` 是**验证集**（推荐组的唯一选择依据），`test_*` 是留出集评估，`train_*` 只用于观察拟合程度。长表里两套指标并存，见 §8。

## 2. 实验矩阵

特征组（与 `feature_list.json` 的 18 个特征一一对应）：

| 组名 | 列 |
| --- | --- |
| `zone` | `zone_id` |
| `calendar` | `hour, minute, dow, is_weekend, is_holiday, hour_sin, hour_cos, dow_sin, dow_cos` |
| `lag_short` | `lag_1, lag_2, lag_4, lag_8` |
| `lag_day` | `lag_96` |
| `lag_week` | `lag_672` |
| `rolling` | `roll_mean_4, roll_mean_96` |

| id | 特征组 | 特征数 | objective | 说明 |
| --- | --- | ---: | --- | --- |
| A0 | zone + calendar | 10 | regression_l1 | 任务书 A0：仅区域 + 日历 |
| A1 | A0 + lag_short | 14 | regression_l1 | 加 t-1/2/4/8 |
| A2 | A1 + lag_day | 15 | regression_l1 | 加 t-96 |
| A3 | A2 + lag_week | 16 | regression_l1 | 加 t-672 |
| A4 | A3 + rolling | 18 | regression_l1 | 完整特征（= C2 基线） |
| A5 | A4 同特征 | 18 | **regression_l2** | 只换损失 |
| A6 | A4 − zone | 17 | regression_l1 | 任务书可选：剥离 `zone_id` |
| A7 | A4 − calendar | 9 | regression_l1 | 补充：剥离日历 |

> 口径说明：任务书 A0→A1 的写法会同时“去掉 `zone_id`、加上滞后”，无法单独归因。这里按**累积消融**实现（A0 已含 `zone_id`，A1~A4 逐步累加，A4 才是 18 个“完整特征”），并用 A6/A7 从完整模型反向剥离 `zone_id` 与日历，单组贡献因此可分离。

## 3. 运行命令

```powershell
git switch feature/c-ablation
.\.venv\Scripts\python.exe -m src.models.run_ablation --config configs/experiments/c_ablation.yaml --dry-run
.\.venv\Scripts\python.exe -m src.models.run_ablation --config configs/experiments/c_ablation.yaml
# 口径自检只用一组：
.\.venv\Scripts\python.exe -m src.models.run_ablation --config configs/experiments/c_ablation.yaml --only A4 --out data/processed/experiments/_repro.parquet
```

`--dry-run` 会打印 8 组实验的特征列（10/14/15/16/18/18/17/9），并校验 `A0⊂A1⊂A2⊂A3⊂A4`、`A4` 恰好等于全部 18 个特征、A5 与 A4 同特征、A6/A7 为 A4 的单组剥离；`--only` 断点续跑时结果写入 `lgbm_ablation_<ids>.parquet`，不会覆盖主表。

## 4. 结果

| 组 | 特征数 | objective | val MAE | val RMSE | val WAPE | test MAE | test RMSE | test WAPE | best_iter | 训练秒 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 10 | l1 | 5.0638 | 9.9788 | 21.45% | 4.5999 | 8.2822 | 17.96% | 5997 | 226.3 |
| A1 | 14 | l1 | 3.9966 | 6.7035 | 16.93% | 4.0927 | 6.6176 | 15.98% | 3452 | 155.5 |
| **A2** | **15** | **l1** | **3.9772** | **6.6614** | **16.85%** | 4.0824 | 6.5962 | 15.94% | 2391 | 112.8 |
| A3 | 16 | l1 | 3.9902 | 6.6907 | 16.90% | **4.0241** | 6.4630 | **15.72%** | 2594 | 129.9 |
| A4 | 18 | l1 | 3.9879 | 6.6800 | 16.89% | 4.0595 | 6.5593 | 15.85% | 1180 | 62.3 |
| A5 | 18 | l2 | 4.0330 | 6.7592 | 17.09% | 4.0485 | **6.4587** | 15.81% | 1624 | 59.9 |
| A6 | 17 | l1 | 4.0531 | 6.7650 | 17.17% | 4.1052 | 6.6030 | 16.03% | 1238 | 62.0 |
| A7 | 9 | l1 | 4.2585 | 7.0305 | 18.04% | 4.2936 | 6.8644 | 16.77% | 941 | 44.7 |

相对 A4（完整特征）的差值（负值更好）：

| 组 | Δval WAPE (pp) | Δtest WAPE (pp) | Δtest RMSE (%) | 解读 |
| --- | ---: | ---: | ---: | --- |
| A0 | +4.56 | +2.11 | +26.3% | 只有区域+日历时明显不足 |
| A1 | +0.04 | +0.13 | +0.9% | 加短期滞后即追平 |
| A2 | **−0.05** | +0.09 | +0.6% | 验证集最好 |
| A3 | +0.01 | **−0.14** | **−1.5%** | 测试集最好 |
| A5 | +0.19 | −0.04 | −1.5% | 换 L2：验证变差、测试略好 |
| A6 | +0.28 | +0.18 | +0.7% | 去掉 zone_id 的代价 |
| A7 | +1.15 | +0.91 | +4.7% | 去掉日历的代价 |

## 5. 结论

1. **滞后特征是决定性的，且短滞后就够。** A0（区域+日历）→ A1（+`lag_1/2/4/8`）把 val WAPE 从 21.45% 拉到 16.93%，下降 4.52pp，占 A0→A4 全部提升（4.56pp）的 **99.2%**。A0 的 train/val 差距也最大（train WAPE 17.68% vs val 21.45%），说明只有区域×日历时模型只能学到“区域×时段的平均强度”，无法跟随当天的需求波动。
2. **日滞后、周滞后、滚动均值几乎没有增量。** A1→A2（+`lag_96`）−0.08pp、A2→A3（+`lag_672`）+0.06pp、A3→A4（+`rolling`）−0.01pp，全部落在 ±0.09pp；而且排序在 val/test 之间不稳定（A2 验证集最好、A3 测试集最好）。**在单月数据上，18 个特征相对 15 个特征没有可证明的增益。**
3. **日历比 `zone_id` 更不可替代。** 剥离日历（A7）val WAPE +1.15pp、test +0.91pp；剥离 `zone_id`（A6）只有 +0.28pp / +0.18pp。虽然 C2 的 gain 重要性里日历合计约 5%，但滞后特征本身已经携带了“近期水平”，`zone_id` 的基准强度可被部分替代；而 `hour/dow` 提供的是滞后列给不出的结构（尤其 `lag_672` 暖机期和跨日节律）。
4. **L1 vs L2：没有稳定的赢家，保留 L1。** 验证集上 L1 全面更优（MAE 3.9879 vs 4.0330、RMSE 6.6800 vs 6.7592、WAPE 16.89% vs 17.09%）；测试集上 L2 全面略优（MAE 4.0485 vs 4.0595、RMSE 6.4587 vs 6.5593、WAPE 15.81% vs 15.85%）。两组方向相反、最大差异 ≤1.5%，且 A5 的早停仍按 MAE 选择（对 L2 不利），因此**没有证据支持把最终模型切到 L2**；如果后续明确以 RMSE 为目标，应同时把早停指标换成 l2 再比一次。
5. **推荐：A2（`zone_id` + 日历 + `lag_1/2/4/8` + `lag_96`，15 特征，`regression_l1`）** —— val WAPE 16.85% / MAE 3.9772，test WAPE 15.94% / MAE 4.0824。选择依据是预登记规则：**验证集 WAPE 最低，并列时取特征更少者**（见 manifest 的 `recommendation`）。需要强调：A1/A2/A3/A4 的验证差异 ≤0.09pp，属于噪声量级，因此这条推荐的实际含义是“**15 个特征已经足够，不必为 3 个几乎无增量的特征多付训练和推理成本**”，而不是“A2 显著优于 A4”。若团队要求与 C2 基线 / STGCN 严格同特征做 apples-to-apples 对比，用 A4 同样成立。

## 6. 口径自检（为什么可信）

- **A4 与 C2 基线逐位一致**：train/val/test 的 MAE、RMSE、WAPE 共 9 项与 `data/processed/models/lgbm_c_baseline.metrics.json` 的最大绝对差为 **0.0**（best_iteration 同为 1180）。说明消融脚本的参数映射、列顺序、数据切分与基线完全等价，A0/A5/A6/A7 的对比基准可信。
- **可复现**：`--only A4` 重跑一次，`n_features/mae/rmse/wape/test_*/train_*/best_iteration` 全部逐位相同。
- **同预算**：早停统一 100 轮、上限 6000 轮，8 组在同一份数据、同一 seed 下训练。

## 7. 已知问题与局限

1. **A0 仍未早停**：`best_iteration=5997`（顶到 6000），val WAPE 仍在以每千轮约 0.2pp 的速度缓慢下降；但它与 A1 相差 4.52pp，剩余收敛空间远不足以改变结论。A0 的训练秒数（226 s）也因此偏高。
2. **`lag_672` 暖机缺失**：每区域前 672 行（7 天，占全表 22.58%）为空，A3/A4 在训练集 01-01~01-07 的该列无效。A2 恰好是最后一个不含 `lag_672` 的组，因此“A2 验证集最好”可能与暖机期噪声有关；这条假设无法在单月数据上干净地验证（要验证可只保留 01-08 之后的行重跑 A2/A3/A4，留给 C4/C5）。
3. **只有单月数据**：val/test 只覆盖 1 月下旬，季节性、节假日的结论都不具代表性；`lag_672` 也只能看到同一周内历史。
4. **早停指标统一为 MAE**，对 A5(L2) 不利；这会让 L2 的 RMSE 优势被低估，但反过来也避免为单一指标“定制”早停导致组间口径不一致。
5. **A0→A1 不是严格单变量**（A0 与 A1 都含 `zone_id`，差别是 4 个滞后列），这是任务书矩阵的设定；`zone_id` 的单独贡献由 A6 回答。
6. 训练耗时含 3 个切分的预测；6000 棵树时预测开销显著（A0 的 226 s 主要是训练，A1~A3 的预测各约 2 分钟）。

## 8. 产物与复现

| 产物 | 路径 | 是否入库 |
| --- | --- | --- |
| 实验矩阵与特征组 | `configs/experiments/c_ablation.yaml` | 是 |
| 消融脚本 | `src/models/run_ablation.py` | 是 |
| 长表 | `data/processed/experiments/lgbm_ablation.parquet` | 否（`data/` 被 gitignore） |
| 运行清单 | `data/processed/experiments/lgbm_ablation.manifest.json` | 否 |
| 各组测试预测 | `data/processed/experiments/predictions/pred_test_lgbm_<id>.parquet` | 否 |

长表列：`experiment_id, feature_set, objective, n_features, mae, rmse, wape, test_mae, test_rmse, test_wape, train_mae, train_rmse, train_wape, best_iteration, train_seconds, n_train, n_val, n_test, seed`（`mae/rmse/wape` = 验证集）。清单里额外记录 config、seed、上限轮数、切分边界、两个数据文件路径、各组特征列、版本号与 `recommendation`。

> 不会覆盖 B 消费的 `data/processed/predictions/pred_test.parquet`；B 的仿真/看板继续用 C2 基线预测即可，本分支只提供消融证据。
