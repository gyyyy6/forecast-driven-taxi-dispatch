# 成员 C 任务书：LightGBM 基线、消融与调参

> 硬件：GTX 1660 Ti 6G + 16G RAM
> 分支前缀：`feature/c-`
> 核心目标：给团队一个可信、可复现的 LightGBM 基线，并回答“哪些特征/损失/超参数真正有用”。

## 1. 输入与输出

### 输入

- A 提供：`data/processed/demand_panel.parquet`、`features.parquet`、`feature_list.json`
- 小样本：`data/processed/sample_day.parquet`、`sample_month.parquet`
- 配置文件：`configs/default.yaml`
- 接口契约：`docs/api_contract.md`

### 输出

- 模型：`data/processed/models/lgbm_<tag>.txt`（不入库）
- 指标：`data/processed/models/lgbm_<tag>.metrics.json`
- 预测：`data/processed/predictions/pred_test_lgbm_<tag>.parquet`
- 实验汇总：`data/processed/experiments/*.parquet`（长表，便于画图）
- 报告：`docs/experiments/lgbm_baseline.md`、`lgbm_ablation.md`、`lgbm_tuning.md`、`error_analysis.md`
- 最佳配置：`configs/experiments/lgbm_best.yaml`

## 2. 任务清单

### C1. 数据与切分检查（优先级 P0）

**目标**：在训练前确认面板没有缺行、重复键和时间泄漏。

**做法**：

1. 读取 `sample_day.parquet`，检查：
   - 时间片数是否为 96；
   - 区域数是否为 263；
   - 行数是否为 `96 × 263 = 25,248`；
   - `(timestamp, zone_id)` 是否唯一；
   - `demand` 是否非负、零需求比例是否合理。
2. 用 `src.models.common.time_split_bounds` 打印 train/val/test 边界，确认是按时间切分。
3. 抽查 `lag_1`：对同一区域，`lag_1(t) == demand(t-1)`；抽查 `roll_mean_4` 不使用 `demand(t)`。
4. 把结论写入 `docs/experiments/data_checks.md`，附运行命令和实际行数。

**验收**：报告中有“通过/不通过”结论；发现缺行或泄漏时，先开 Issue 给 A，不要直接改数据。

---

### C2. LightGBM 基线（优先级 P0）

**目标**：在 `sample_month` 上跑通基线，并给 B 一份可消费的预测文件。

**做法**：

1. 复制默认配置为 `configs/experiments/c_sample_month.yaml`，至少改：
   ```yaml
   paths:
     panel: data/processed/sample_month.parquet
     features: data/processed/features_sample_month.parquet
     feature_list: data/processed/feature_list_sample_month.json
     model_dir: data/processed/models/c_sample_month
     prediction_dir: data/processed/predictions/c_sample_month
   lgbm:
     n_estimators: 800
     device: cpu
   ```
2. 生成特征：
   ```powershell
   python -m src.features.build_features --config configs/experiments/c_sample_month.yaml
   ```
3. 训练：
   ```powershell
   python -m src.models.train_lgbm --config configs/experiments/c_sample_month.yaml --tag c_baseline
   ```
4. 记录 train/val/test 的 MAE、RMSE、WAPE、best_iteration、训练耗时。
5. 检查预测文件：
   - 列：`timestamp, zone_id, predicted_demand`；
   - 覆盖测试期全部 timestamp × 263 zones；
   - 无负值、无 NaN；
   - `timestamp` 可被 `pd.to_datetime` 解析。

**产出**：

- `docs/experiments/lgbm_baseline.md`
- `pred_test_lgbm_c_baseline.parquet`
- `lgbm_c_baseline.metrics.json`

**验收**：同一命令 + 同一 seed 可复现指标；A/B 能直接读取预测文件。

---

### C3. 特征组与损失函数消融（优先级 P0）

**目标**：回答“滞后、滚动、日历、zone_id 各自贡献多少”，以及 L1/L2 哪个更适合需求预测。

**实验设计**（至少 6 组，全部使用同一时间切分）：

| 组别 | 特征 | objective |
| --- | --- | --- |
| A0 | 仅 `zone_id` + 日历 | `regression_l1` |
| A1 | 日历 + `lag_1/2/4/8` | `regression_l1` |
| A2 | A1 + `lag_96` | `regression_l1` |
| A3 | A2 + `lag_672` | `regression_l1` |
| A4 | A3 + `roll_mean_4/96` | `regression_l1` |
| A5 | A4（完整特征） | `regression_l2` |
| A6（可选） | A4 去掉 `zone_id` | `regression_l1` |

**做法**：

1. 用 `configs/experiments/c_ablation.yaml` 控制特征子集；如果现有脚本不支持，给 `build_features.py` 增加 `--feature-groups` 参数，但**不要破坏默认行为**。
2. 每组训练 1 次，固定 seed；记录 `best_iteration` 和验证集指标。
3. 结果写成长表：
   ```text
   data/processed/experiments/lgbm_ablation.parquet
   列：experiment_id, feature_set, objective, mae, rmse, wape, best_iteration, train_seconds
   ```
4. `docs/experiments/lgbm_ablation.md` 放表格 + 2-3 条结论。

**验收**：至少 6 组可复现结果；明确推荐一组特征集给最终模型。

---

### C4. 小规模超参数调优（优先级 P1）

**目标**：在有限算力下找到稳定提升，而不是盲目堆参数。

**搜索空间建议**（15-20 组，随机搜索）：

```text
learning_rate: [0.03, 0.05, 0.08]
num_leaves: [31, 63, 127]
min_child_samples: [20, 50, 100]
colsample_bytree: [0.7, 0.8, 1.0]
subsample: [0.7, 0.8, 1.0]
reg_lambda: [0, 1, 5]
```

**做法**：

1. 固定 C3 推荐的特征集和时间切分。
2. 用验证集 WAPE/MAE 选择，不使用测试集调参。
3. 记录每组参数与指标到 `data/processed/experiments/lgbm_tuning.parquet`。
4. 最佳参数写入 `configs/experiments/lgbm_best.yaml`，并注明搜索日期、随机种子、试验次数。
5. 1660 Ti 上 LightGBM 用 CPU；不要为了 GPU 版 LightGBM 折腾环境。

**产出**：`docs/experiments/lgbm_tuning.md` + `configs/experiments/lgbm_best.yaml`。

**验收**：最佳配置在测试集上重跑一次，指标与报告一致；调参只使用验证集。

---

### C5. 分时段/分区域误差分析（优先级 P1）

**目标**：告诉 A 和 B 模型在哪些场景不可信，指导仿真设置。

**做法**：

1. 用最佳模型在测试集预测。
2. 按以下维度汇总 MAE/WAPE/偏差（预测-真实）：
   - 小时（0-23）；
   - 工作日/周末；
   - 区域需求分位（低/中/高）；
   - Top 20 高需求区域；
   - 零需求 vs 非零需求。
3. 输出：
   - `data/processed/experiments/error_by_hour.parquet`
   - `data/processed/experiments/error_by_zone.parquet`
   - `docs/experiments/error_analysis.md`
4. 至少回答：
   - 早/晚高峰是否被低估；
   - 零需求区域是否被过度预测；
   - 哪些区域误差最大，是否与机场/曼哈顿核心区相关。

**验收**：报告中有 3 条可执行建议（例如“仿真中峰值需求打折/上调”“某区域需要单独建模”）。

---

### C6. 6G 显存兼容性测试（优先级 P2，保护性任务）

**目标**：确认 C 的 1660 Ti 6G 能跑轻量 STGCN 冒烟实验，避免把不兼容配置交给 A。

**做法**：

1. 用 `sample_month.parquet`，不要用全年数据。
2. 运行 2 个 epoch：
   ```powershell
   python -m src.models.train_stgnn --config configs/default.yaml \
     --panel data/processed/sample_month.parquet \
     --epochs 2 --batch-size 8 --seq-len 12 --hidden 16 --device cuda --tag c_vram_bs8
   ```
3. 再测 `batch_size=16, hidden=32`；记录是否 OOM。
4. 用 `torch.cuda.max_memory_allocated()` 记录峰值显存，写入 `docs/experiments/stgnn_vram.md`。
5. 给出结论：6G 下安全配置 / 需要降级配置。

**注意**：最终 STGCN 由 A 在 4060 上训练；C 只做兼容性验证，不追求指标。

**验收**：报告中有明确的 `batch_size / seq_len / hidden_channels` 推荐值；OOM 时有回退方案。

## 3. 分支与提交

```text
feature/c-lgbm-baseline   -> C1 + C2
feature/c-ablation        -> C3
feature/c-tuning          -> C4 + C5
feature/c-vram-check      -> C6
```

提交前检查：

```powershell
git status
git diff --check
```

不要提交 `data/processed/models/*.txt`、`*.pt`、`data/processed/experiments/*.parquet` 等大文件；报告和配置可以提交。

## 4. 完成定义（DoD）

- [ ] 基线命令、配置、指标、预测文件齐全
- [ ] 消融表至少 6 组，结论明确
- [ ] 调参只用验证集，最佳配置可复现
- [ ] 误差分析有分时段/分区域结论
- [ ] 6G 显存测试有安全配置结论
- [ ] PR 描述包含运行命令与指标
