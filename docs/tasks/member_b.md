# 成员 B 任务书：SimPy 仿真、调度策略与 Streamlit 看板

> 硬件：Intel Ultra 5 集显 + 16G RAM
> 分支前缀：`feature/b-`
> 核心目标：在轻薄本上跑通“预测需求 → 调度 → 指标对比 → 可视化”闭环，不碰模型训练。

## 1. 输入与输出

### 输入

- 小样本面板：`data/processed/sample_day.parquet`（约 25,248 行）
- 测试期预测：`data/processed/predictions/pred_test.parquet`
- 区域质心：`data/raw/taxi_zone_centroids.csv`
- 接口契约：`docs/api_contract.md`
- 现有骨架：`src/simulation/env.py`、`src/simulation/policy.py`、`src/visualization/app.py`

### 输出

- 仿真明细：`data/processed/simulation/sim_requests.parquet`
- 策略汇总：`data/processed/simulation/sim_results.parquet`
- 场景结果：`data/processed/simulation/scenarios/*.parquet`
- 报告：`docs/simulation_report.md`、`docs/demo_script.md`
- 看板：`src/visualization/app.py`

## 2. 算力与数据边界

1. **只读小样本**：`sample_day.parquet`、`sample_month` 的按日切片、`pred_test` 的测试期切片。
2. **禁止**读取 `demand_panel.parquet`、`features.parquet`；禁止 import `torch`、`lightgbm`。
3. 目标性能：1 天（96 个时间片、200 辆车）仿真 < 60 秒；Streamlit 启动 < 10 秒，内存 < 2GB。
4. 所有输出用 `.parquet`；看板截图可以放 `docs/assets/`，单张 < 500KB。

## 3. 任务清单

### B1. 仿真输入 fixture 与契约校验（优先级 P0）

**目标**：在 A 的正式预测出来前，也能独立开发和测试。

**做法**：

1. 从 `sample_day.parquet` 生成一个测试用预测文件（只用于本地开发，不入库）：
   - 列名改为 `timestamp, zone_id, predicted_demand`；
   - `predicted_demand = demand.astype("float32")`；
   - 写到 `data/processed/simulation/fixture_pred_day.parquet`。
2. 检查 `env.py` 的 `load_demand` 是否接受该 schema；缺列时给出清晰报错。
3. 准备区域质心：
   - 优先用 A 生成的 `data/raw/taxi_zone_centroids.csv`；
   - 若只有 shapefile，按 README 用 geopandas 生成，确认 263 个区域齐全。
4. 写 `docs/simulation_contract.md`：说明请求、车辆、区域、时间、距离的单位。

**验收**：不依赖模型训练，能用一个 parquet 文件启动仿真。

---

### B2. SimPy 环境完善（优先级 P0）

**目标**：把现有 `env.py` 变成可复现、可解释的离散事件仿真。

**必须建模**：

- 车辆：`vid, zone, free_at, trips, deadhead_km`
- 请求：`request_id, request_time, zone_id, assigned_vehicle, pickup_time, wait_min, served`
- 区域间行程时间：`distance_km / speed_kmh × 60`，同区域最小 1 分钟
- 服务时间：`service_time_min`
- 需求到达：每个 15 分钟片按 `predicted_demand` 生成泊松到达
- 未服务：无空闲车或 ETA > `max_wait_min` 时记 `served=0`
- 空驶：接驾 + 预调度移动都计入 `deadhead_km`

**要求**：

1. 固定 `seed` 后，同一策略重复运行结果一致。
2. 两种策略必须共享同一请求流和初始车辆位置。
3. 仿真结束时间、车辆数、速度、服务时间、最大等待全部走 YAML。
4. 输出字段严格符合 `docs/api_contract.md` 第 4 节。

**验收**：`python -m src.simulation.env --config configs/default.yaml --policy both` 可跑通；`sim_requests.parquet` 与 `sim_results.parquet` 字段正确。

---

### B3. Greedy 最近车辆策略（优先级 P0）

**目标**：建立可信的基线策略。

**做法**：

1. 在 `policy.py` 的 `GreedyPolicy` 中：
   - 只考虑 `free_at <= now` 的车辆；
   - 选择 ETA 最小的车辆；
   - ETA 超过 `max_wait_min` 则拒绝服务。
2. 记录每次接驾的空驶距离。
3. 用 fixture 跑 1 天，人工抽查 3 条请求的派车是否合理。
4. 输出基线指标：服务率、平均等待、P90 等待、空驶里程。

**验收**：基线结果可复现；无“车辆瞬间移动”或“同一辆车同时接两单”。

---

### B4. Forecast-driven 预调度策略（优先级 P0）

**目标**：验证“用未来预测提前把车放到热点”是否真的改善指标。

**做法**：

1. 在 `ForecastPolicy` 中，每个 15 分钟片读取下一时间片的 `predicted_demand`。
2. 计算区域缺口：`需求 - 当前空闲车辆数`；只对正缺口区域预调度。
3. 从供给过剩区域选最近车辆；限制：
   - `max_move_ratio`（默认 0.25）；
   - `min_demand_gain`（默认 1.0）；
   - 预调度 ETA 不超过 15 分钟。
4. 预调度也会占用车辆时间并产生空驶，必须计入指标。
5. 与 Greedy 使用同一请求流、同一初始车辆。

**验收**：`--policy both` 输出两行对比；Forecast 在至少一个场景下服务率或等待时间优于 Greedy，若没有，报告要解释原因。

---

### B5. 场景矩阵实验（优先级 P1）

**目标**：给报告提供“不同供给/等待容忍度下策略是否仍有效”的证据。

**核心场景**（1 天样本即可）：

| 维度 | 取值 |
| --- | --- |
| 车辆数 | 100 / 200 / 300 |
| 最大等待 | 15 / 30 分钟 |
| 预调度比例 | 0.10 / 0.25 / 0.50（仅 Forecast） |

建议先跑 12 组，不要一上来跑几十组：

- Greedy：3 个车辆数 × 2 个等待阈值 = 6 组
- Forecast：`n_vehicles=200, max_wait=30` 下 3 个比例 = 3 组
- 补充：Greedy 最优组合 vs Forecast 最优组合 = 2-3 组

**输出**：

```text
data/processed/simulation/scenarios/
  scenario_<id>_requests.parquet
  scenario_<id>_summary.parquet
```

汇总表列：

```text
scenario_id, policy, n_vehicles, max_wait_min, max_move_ratio,
total_requests, served_rate, avg_wait_min, p90_wait_min, deadhead_km
```

**验收**：每个场景有唯一 ID、可复现命令、汇总行；报告中有场景表与结论。

---

### B6. Streamlit 看板（优先级 P1）

**目标**：让非技术观众 3 分钟内看懂“预测 → 调度 → 改善”。

**页面结构**：

1. **需求热点**
   - 时间滑块（15 分钟粒度）；
   - 有经纬度时用 Plotly Mapbox 散点，没有则用 Top 25 柱状图；
   - 显示总需求、活跃区域数、峰值。
2. **时间趋势**
   - 全网预测需求折线；
   - 可叠加真实需求（如果 parquet 里有 `demand`）。
3. **调度仿真**
   - Greedy vs Forecast 指标表；
   - 等待时间箱线图；
   - 服务率柱状图；
   - 空驶里程对比。
4. **场景对比（可选）**
   - 选择 `scenario_id`，展示不同车辆数/等待阈值。

**要求**：

- 只读 `.parquet`；
- 文件缺失时显示操作提示，不抛异常；
- 不 import 训练代码；
- 侧边栏显示当前数据文件路径与刷新按钮。

**验收**：`streamlit run src/visualization/app.py` 可启动；在 Ultra 5 集显上切换时间片不卡死。

---

### B7. 仿真报告与演示脚本（优先级 P2）

**目标**：把仿真结论沉淀成可写进课程报告的材料。

**报告 `docs/simulation_report.md` 至少包含**：

1. 仿真假设与简化（区域中心、匀速、服务时间固定等）；
2. 请求生成方式（泊松、与预测需求的关系）；
3. Greedy 与 Forecast 的伪代码/流程；
4. 场景表与关键指标；
5. Forecast 有效/无效的原因分析；
6. 局限性（无实时路况、无司机行为、区域粒度过粗）。

**演示脚本 `docs/demo_script.md`**：

- 0:00-0:30 项目目标；
- 0:30-1:30 需求热点与预测；
- 1:30-2:30 Greedy vs Forecast 指标；
- 2:30-3:00 结论与局限。

**验收**：队友按脚本能独立演示；报告中的数字与 `sim_results.parquet` 一致。

## 4. 分支与提交

```text
feature/b-simulation   -> B1 + B2
feature/b-policies     -> B3 + B4
feature/b-scenarios    -> B5
feature/b-dashboard    -> B6 + B7
```

提交前：

```powershell
git status
git diff --check
```

不要提交 `data/processed/simulation/*.parquet` 等运行产物；报告、代码、配置、必要截图可以提交。

## 5. 完成定义（DoD）

- [ ] 1 天仿真在轻薄本上 < 60 秒
- [ ] 同一 seed 下结果可复现
- [ ] Greedy/Forecast 使用同一请求流
- [ ] 输出字段符合 `docs/api_contract.md`
- [ ] 至少 9 个场景结果与汇总表
- [ ] Streamlit 看板可独立启动、处理缺失文件
- [ ] 仿真报告与 3 分钟演示脚本完成
