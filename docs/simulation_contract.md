# 仿真契约（成员 B 维护）

## 1. 单位与语义

| 对象 | 字段 | 单位 / 取值 | 说明 |
| --- | --- | --- | --- |
| 时间 | `request_time`, `pickup_time` | naive datetime | 与 `api_contract.md` §1 一致，不做时区转换 |
| 时段 | `timestamp` | 15 分钟，区间起点 | 96 片/天 |
| 区域 | `zone_id` | 整数 1–263 | 车辆停在区域中心 |
| 车辆 | `free_at` | 仿真分钟（自 start 起算） | 空闲时刻，`= pickup + service_time` |
| 行程时间 | `eta_min` | 分钟 | `60 × dist_km / speed_kmh`，同区域下限 1 分钟 |
| 距离 | `dist_km` | km | 质心 haversine（`taxi_zone_centroids.csv`，lat/lon） |
| 服务时间 | `service_time_min` | 分钟，常数 | YAML |
| 需求 | `predicted_demand` | 单 / 15 分钟 / 区域 | 按泊松过程转成请求到达 |
| 空驶 | `deadhead_km` | km | 接驾 + 预调度移动，全部计入 |

## 2. 规模口径（重要）

`n_vehicles` 是**调度单元数**，不等于纽约在营车辆数。`demand_scale` 是对需求做泊松 thinning 的比例（thinning 后仍是泊松过程）。

| 口径 | fixture | 全天请求 | 200 车实测 | 用途 |
| --- | --- | --- | --- | --- |
| 真实 | `fixture_pred_day.parquet`（scale 1.0） | 630,073 | 服务率 1.6%，耗时 23.3 s | 验证复杂度与耗时上限 |
| 对齐 | `fixture_pred_day_s002.parquet`（scale 0.02） | 12,510 | 服务率 82.2%，耗时 2.5 s | 策略对比 |

报告中所有指标必须注明用了哪个 scale。

## 3. 已实测基线（2023-01-01，scale 0.02，greedy）

| n_vehicles | served_rate | avg_wait | p90_wait | deadhead_km |
| --- | --- | --- | --- | --- |
| 100 | 43.4% | 14.06 min | 25.47 min | 31,695.6 |
| 200 | 82.2% | 6.19 min | 16.44 min | 25,390.3 |
| 300 | 93.7% | 4.26 min | 11.76 min | 18,567.1 |

## 4. 已知简化

区域中心代表整区、匀速 25 km/h、服务时间固定、无实时路况、无司机行为、无拒单、预调度不重算目标缺口。

## 5. 待解决

`ForecastPolicy` 在拥堵工况下几乎不触发预调度（见 issue：源头规则要求本地富余 + 缺口的量纲为“单”而非“车”），当前策略对比结果不能用于结论。