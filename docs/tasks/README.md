# 成员 B / C 任务分工总览

> 项目：基于时空图神经网络的网约车需求热点预测与轻量级闭环调度仿真
> 更新：2026-10-21
> 角色：A（数据/主模型/集成）、B（仿真/前端）、C（基线/消融/调参）

## 1. 角色边界

| 成员 | 硬件 | 只负责 | 不负责 |
| --- | --- | --- | --- |
| A | RTX 4060 8G + 16G | 全量数据清洗、特征工程、STGCN/Graph WaveNet、最终模型集成 | 仿真策略细节、前端页面 |
| B | Intel Ultra 5 集显 + 16G | SimPy 仿真、Greedy/Forecast 策略、Streamlit 看板 | 任何模型训练、读取全量面板 |
| C | GTX 1660 Ti 6G + 16G | LightGBM 基线、特征/损失消融、调参、误差分析、6G 显存兼容性测试 | 全量 STGCN 训练、前端 |

## 2. 依赖与交付物

| 方向 | 内容 | 文件/格式 |
| --- | --- | --- |
| A → C | 全量面板、特征表、特征清单 | `data/processed/demand_panel.parquet`、`features.parquet`、`feature_list.json` |
| A → B | 测试期预测、区域质心 | `data/processed/predictions/pred_test.parquet`、`data/raw/taxi_zone_centroids.csv` |
| C → A/B | 基线预测、消融与调参结论 | `pred_test_lgbm_<tag>.parquet`、`docs/experiments/*.md`、`data/processed/experiments/*.parquet` |
| B → A/C | 仿真明细、策略对比、看板 | `data/processed/simulation/sim_requests.parquet`、`sim_results.parquet`、`src/visualization/app.py` |

接口字段以 [`docs/api_contract.md`](../api_contract.md) 为准，任何修改先提 PR。

## 3. 数据与算力硬约束

1. **B 的机器不得读取** `demand_panel.parquet`、`features.parquet` 等全量文件；只允许读 `sample_day.parquet`（约 2.5 万行）和小体积预测文件。
2. **C 的 6G 显存**：深度学习实验统一 `batch_size <= 16`，优先 8；`seq_len = 12`；`hidden_channels <= 32`。OOM 时先降 batch，再降 hidden，不要改数据切分。
3. 所有中间数据、样本、预测、仿真结果使用 `.parquet`；禁止用 CSV 传大数据。
4. 时间切分固定为前 70% / 15% / 15%，**禁止随机划分**；特征不得包含未来信息。
5. `data/`、模型权重、截图之外的大文件不入库；提交前检查 `git status`。

## 4. 协作方式

- 从 `dev` 拉分支：
  - C：`feature/c-lgbm-baseline`、`feature/c-ablation`、`feature/c-tuning`
  - B：`feature/b-simulation`、`feature/b-policies`、`feature/b-dashboard`
- 每个 PR 至少 1 人 review；PR 描述写清：数据范围、运行命令、指标、已知问题。
- 每周同步一次：C 报基线/消融指标，B 报仿真场景与看板进度，A 负责合并与接口对齐。
- Issue 标题建议：
  - `[C] LGBM baseline on 2023-01 sample`
  - `[C] Feature/objective ablation`
  - `[C] 6G VRAM compatibility test`
  - `[B] SimPy environment and contract`
  - `[B] Greedy vs forecast experiment matrix`
  - `[B] Streamlit dashboard v1`

## 5. 十周里程碑

| 周次 | 成员 C | 成员 B |
| --- | --- | --- |
| 2 | 数据完整性检查；跑通 LightGBM 基线 | 仿真 fixture；确认输入/输出契约 |
| 3 | 基线指标与首版报告 | Greedy 策略跑通；输出标准仿真结果 |
| 4 | 特征组 / 损失函数消融 | Forecast 预调度策略；同请求流对比 |
| 5 | 小规模调参；保存最佳配置 | 车辆数/等待阈值/预调度比例场景矩阵 |
| 6 | 分区域/分时段误差分析 | Streamlit 看板 v1（预测 + 仿真） |
| 7 | 6G 显存兼容性测试；模型对比 | 看板 v2；仿真实验报告 |
| 8 | 与 A 的 STGCN 预测对比 | 闭环 demo：预测 → 调度 → 指标 |
| 9 | 最终实验表、复现实验命令 | 最终场景、演示脚本、截图 |
| 10 | 报告/答辩中的基线章节 | 报告/答辩中的仿真与前端章节 |

## 6. 集成验收门槛

- C：`pred_test.parquet` 覆盖测试期全部 `(timestamp, zone_id)`；指标可复现；消融表至少 6 组；最佳配置写入 `configs/experiments/`。
- B：1 天仿真在轻薄本上 < 60 秒、内存 < 2GB；同一请求流下可对比 Greedy/Forecast；看板能独立启动并处理缺失文件。
- 联合：A 的最佳预测替换 C 的基线预测后，B 的仿真脚本无需改字段即可运行。

## 7. 详细任务书

- [成员 B 任务书](member_b.md)
- [成员 C 任务书](member_c.md)
