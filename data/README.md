# data/ 目录说明

本目录用于本地存放原始数据和中间结果，**不提交到 Git**。

当前 2023-01 数据已经准备完成，详细状态、校验和、复现命令见：

- [`docs/data_status_2023-01.md`](../docs/data_status_2023-01.md)

快速检查：

```powershell
Get-ChildItem data/raw
Get-ChildItem data/processed
```

不要提交：

```text
data/raw/*.parquet
data/processed/*.parquet
data/raw/taxi_zone_centroids.csv
```

小样本和全量面板通过 `make_dataset.py` / `make_sample.py` 在本地生成。