# 2023-01 HVFHV 数据状态与复现说明

> 状态：**READY**
> 生成日期：2026-10-22
> 生成机器：成员 A（RTX 4060 8G + 16G RAM）
> 数据目录：`data/raw/`、`data/processed/`

## 1. 结论

- 2023-01 HVFHV 原始数据已下载并校验；
- 15 分钟 × Taxi Zone 需求面板已生成；
- 日样本、月样本、特征表已生成；
- 原始数据和中间数据**不提交到 Git**（体积与数据规范原因）；
- 代码、配置、本文档已推送到 GitHub。

## 2. 数据源

| 文件 | 下载地址 | 说明 |
| --- | --- | --- |
| HVFHV 2023-01 | <https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-01.parquet> | 18,479,031 行 |
| Taxi Zone Lookup | <https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv> | 263 个区域 |
| Taxi Zone Shapefile | <https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip> | 解压后 263 个多边形 |

## 3. 文件校验

| 文件 | 大小 | SHA256 |
| --- | ---: | --- |
| `fhvhv_tripdata_2023-01.parquet` | 451.87 MB | `CB066A3FE0770E44DDE67CFC43E8B964B7B3CA7F2C90DC35D2D57CB7E41629B8` |
| `taxi_zone_lookup.csv` | 0.01 MB | `1A99E105092230F8620F301EDCCA7F80D3080642FF404D28ED957D3FA222C8ED` |
| `taxi_zones.zip` | 0.98 MB | `F6D711917BB4340F8F644D5366C51665489EB2D426DD1A4A55677721AE5ADF17` |

Windows 校验命令：

```powershell
Get-FileHash -Algorithm SHA256 data/raw/fhvhv_tripdata_2023-01.parquet
Get-FileHash -Algorithm SHA256 data/raw/taxi_zone_lookup.csv
Get-FileHash -Algorithm SHA256 data/raw/taxi_zones.zip
```

## 4. 本地目录

```text
data/raw/
├── fhvhv_tripdata_2023-01.parquet
├── taxi_zone_lookup.csv
├── taxi_zones.zip
├── taxi_zone_centroids.csv          # 由 shapefile 生成，263 行
└── taxi_zones/
    ├── taxi_zones.shp
    ├── taxi_zones.dbf
    ├── taxi_zones.shx
    └── taxi_zones.prj

data/processed/
├── demand_panel.parquet             # 全量 1 月面板
├── demand_panel.meta.json           # 面板统计
├── sample_day.parquet               # 前 1 天
├── sample_month.parquet             # 前 30 天
├── features.parquet                 # 全量特征
└── feature_list.json                # 18 个特征清单
```

## 5. 清洗口径

- 时间范围：`[2023-01-01 00:00, 2023-02-01 00:00)`；
- 时间粒度：15 分钟，时间戳为区间起点；
- 空间粒度：Taxi Zone ID 1–263；
- 需求定义：每个 `(timestamp, zone_id)` 内 `PULocationID` 的上车订单数；
- 完整面板：2976 个时间片 × 263 区域，缺失补 0；
- 丢弃 `PULocationID = 264/265`（TLC 未知区域）；
- 本次共丢弃 992 条原始记录，均为 264/265；
- 时间戳保持纽约本地 naive 时间，不做 DST 时区转换。

## 6. 处理结果

| 文件 | 行数 | 大小 | 说明 |
| --- | ---: | ---: | --- |
| `demand_panel.parquet` | 782,688 | 0.85 MB | 2976 × 263 |
| `sample_day.parquet` | 25,248 | 0.03 MB | 96 × 263 |
| `sample_month.parquet` | 757,440 | 0.83 MB | 2880 × 263 |
| `features.parquet` | 782,688 | 6.8 MB | 18 个特征 |

面板统计：

```text
开始: 2023-01-01 00:00:00
结束: 2023-01-31 23:45:00
总需求: 18,478,039
零需求比例: 9.02%
平均需求: 23.61
P99 需求: 138
```

特征泄漏抽查：

```text
lag_1 == demand.shift(1)                  True
roll_mean_4 == 过去 4 个 demand 的均值    True
```

## 7. 复现命令

```powershell
# 1) 下载（已下载可跳过）
curl.exe -L --fail --retry 3 -o data/raw/fhvhv_tripdata_2023-01.parquet `
  "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-01.parquet"
curl.exe -L --fail --retry 3 -o data/raw/taxi_zone_lookup.csv `
  "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
curl.exe -L --fail --retry 3 -o data/raw/taxi_zones.zip `
  "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

# 2) 解压 shapefile
Expand-Archive -LiteralPath data/raw/taxi_zones.zip -DestinationPath data/raw -Force

# 3) 生成需求面板
python -m src.data.make_dataset --config configs/default.yaml

# 4) 提取小样本
python -m src.data.make_sample --config configs/default.yaml

# 5) 特征工程
python -m src.features.build_features --config configs/default.yaml
```

生成质心 CSV：

```python
import geopandas as gpd
import pandas as pd

gdf = gpd.read_file("data/raw/taxi_zones/taxi_zones.shp")
c = gdf.to_crs(2263).geometry.centroid
c = gpd.GeoSeries(c, crs=2263).to_crs(4326)
pd.DataFrame({
    "LocationID": gdf["LocationID"],
    "lat": c.y,
    "lon": c.x,
}).to_csv("data/raw/taxi_zone_centroids.csv", index=False)
```

## 8. 给成员 B/C 的提示

### 成员 C

直接使用：

```text
data/processed/sample_month.parquet
data/processed/features.parquet
data/processed/feature_list.json
```

不要重新跑 `make_dataset.py` 覆盖全量面板；需要小样本时用 `make_sample.py`。

### 成员 B

直接使用：

```text
data/processed/sample_day.parquet
data/raw/taxi_zone_centroids.csv
```

注意：

- B 的机器不要读取 `demand_panel.parquet` / `features.parquet`；
- 正式仿真还需要 `pred_test.parquet`，在 C/A 产出预测前，可按 `docs/tasks/member_b.md` 的 B1 从 `sample_day.parquet` 造 fixture；
- `taxi_zone_centroids.csv` 只有 263 行、约 10 KB，因 `data/` 被 gitignore，不在 GitHub 上，建议直接发给 B。

## 9. 注意事项

1. `data/raw/` 和 `data/processed/` 均被 `.gitignore` 忽略，**不要**用 `git add -f` 强行提交原始数据。
2. 重新下载后，先核对 SHA256 再跑清洗。
3. 如果 TLC 后续更新了同一月份文件，校验和可能变化，以实际下载为准。
4. 本月的 `demand_panel.parquet` 只有约 0.85 MB，是因为 parquet + zstd 压缩；原始 parquet 仍有 452 MB。
