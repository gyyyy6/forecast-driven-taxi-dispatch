# 接口契约（API Contract）

本文件是数据、预测、仿真三个模块之间的**唯一约定**。任何一方修改字段，必须同步更新本文件并在群里通知。

## 1. 时间与空间约定

| 项目 | 约定 |
| --- | --- |
| 时区 | 原始数据为纽约本地时间，统一存为 **naive datetime**，不做 UTC 转换 |
| 时间粒度 | 15 分钟，时间戳为区间**起点**（如 `00:00`, `00:15`） |
| 空间粒度 | Taxi Zone ID，整数 1–263；264/265 视为未知区域并剔除 |
| 缺失需求 | 面板必须补齐为零（zero-fill），不允许缺行 |
| 文件格式 | `.parquet`，`compression="zstd"` |

## 2. 数据 → 模型

### 2.1 需求面板 `demand_panel.parquet`

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `timestamp` | `datetime64[ns]` | 15 分钟片起点 |
| `zone_id` | `int16` | Taxi Zone ID |
| `demand` | `int32` | 该时间片该区域上车订单数 |

唯一键：`(timestamp, zone_id)`；行数应等于 `时间片数 × 263`。

### 2.2 特征表 `features.parquet`

在面板基础上追加特征列，主键不变：

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `lag_1`, `lag_2`, ... | `float32` | 同区域历史需求（15min 的整数倍） |
| `roll_mean_4`, `roll_mean_96` | `float32` | 过去 1h / 24h 均值，均先 shift(1) |
| `hour`, `minute`, `dow` | `int8` | 日历特征 |
| `is_weekend`, `is_holiday` | `int8` | 0/1 |
| `hour_sin`, `hour_cos` | `float32` | 周期编码（可选） |

- `demand` 为标签列，**不得**作为特征输入。
- 所有特征在时刻 `t` 只能使用 `t` 之前的信息。
- 训练/验证/测试边界处允许因滞后产生的 `NaN`；LightGBM 可直接处理，深度模型需丢弃。

## 3. 模型 → 仿真

### 3.1 预测结果 `pred_<split>.parquet`

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `timestamp` | `datetime64[ns]` | 被预测的时间片起点 |
| `zone_id` | `int16` | Taxi Zone ID |
| `predicted_demand` | `float32` | 预测需求，非负 |

- 必须覆盖完整测试期 × 全部区域（缺失区域补 0）。
- 可附加 `model_name`、`split` 列，但**不得改变前三列语义**。
- 保存路径：`data/processed/predictions/pred_{split}.parquet`。

示例：

```python
import pandas as pd
df = pd.DataFrame({
    "timestamp": pd.to_datetime(["2024-12-01 00:00", "2024-12-01 00:00"]),
    "zone_id": pd.array([1, 2], dtype="int16"),
    "predicted_demand": pd.array([12.3, 0.0], dtype="float32"),
})
df.to_parquet("pred_test.parquet", compression="zstd", index=False)
```

## 4. 仿真输出

### 4.1 仿真明细 `sim_requests.parquet`

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | `int64` | 请求 ID |
| `request_time` | `datetime64[ns]` | 请求产生时间 |
| `zone_id` | `int16` | 上车区域 |
| `assigned_vehicle` | `int32` | 车辆 ID，-1 表示未服务 |
| `pickup_time` | `datetime64[ns]` | 实际接驾时间 |
| `wait_min` | `float32` | 等待时长 |
| `served` | `int8` | 是否服务 |
| `policy` | `category` | `greedy` / `forecast` |

### 4.2 汇总指标 `sim_results.parquet`

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `policy` | `str` | 策略名 |
| `served_rate` | `float32` | 完成订单 / 总请求 |
| `avg_wait_min` | `float32` | 平均等待（仅已服务） |
| `p90_wait_min` | `float32` | 90 分位等待 |
| `deadhead_km` | `float32` | 空驶里程 |
| `total_requests` | `int32` | 总请求数 |

## 5. 变更流程

1. 在 `docs/api_contract.md` 提 PR 修改；
2. 至少 1 人 review；
3. 同步更新读写两侧代码与配置；
4. 旧版预测结果作废，重新生成。