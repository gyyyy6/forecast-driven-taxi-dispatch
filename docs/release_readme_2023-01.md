# 2023-01 HVFHV 小数据发布包

本包用于队友快速开始，不包含 451.87 MB 的原始 HVFHV parquet。

## 内容

| 文件 | 行数/说明 |
| --- | --- |
| sample_day.parquet | 25,248 行，2023-01-01 一天，263 区域 |
| sample_month.parquet | 757,440 行，前 30 天，263 区域 |
| demand_panel.parquet | 782,688 行，2023-01 全月 15 分钟面板 |
| demand_panel.meta.json | 面板统计 |
| features.parquet | 782,688 行，18 个特征 |
| feature_list.json | 特征清单 |
| taxi_zone_centroids.csv | 263 个区域质心，lat/lon |
| taxi_zone_lookup.csv | 区域名称/borough |

## 使用

- 成员 C：用 sample_month.parquet / features.parquet 做 LightGBM 基线。
- 成员 B：用 sample_day.parquet + taxi_zone_centroids.csv 做仿真。
- 时间范围：[2023-01-01, 2023-02-01)，15 分钟粒度。
- 完整说明见仓库 docs/data_status_2023-01.md。

## 注意

原始数据与完整处理结果不入 Git；本包用于群文件/GitHub Release 分发。