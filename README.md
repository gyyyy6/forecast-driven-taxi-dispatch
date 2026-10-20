# 基于时空图神经网络的网约车需求热点预测与轻量级闭环调度仿真

> 机器学习高级技术及应用 · 10 周课程项目

本项目以 NYC TLC **High Volume For-Hire Vehicle (HVFHV)** 数据为基础，构建一条从原始数据到可视化看板的闭环流水线：

```text
HVFHV 原始数据
   → 15min × Taxi Zone 需求面板
   → 特征工程（严格防泄漏）
   → 需求预测（LightGBM / 轻量 STGCN）
   → 预测结果驱动调度（SimPy）
   → Streamlit 可视化看板
```

## 1. 团队分工与算力约束

| 成员 | 硬件 | 主要职责 | 代码约束 |
| --- | --- | --- | --- |
| A（主机） | RTX 4060 8G + 16G RAM | 数据清洗、全量特征工程、STGCN/Graph WaveNet 训练 | `batch_size` 默认 16，可试 32；序列长度 12 |
| B（副机） | GTX 1660 Ti 6G + 16G RAM | LightGBM 基线、消融实验、调参 | 默认 `batch_size=16`、`seq_len=12`；LightGBM 优先 CPU |
| C（轻薄本） | Intel Ultra 5 集显 + 16G RAM | SimPy 仿真、调度策略、Streamlit 前端 | 只读小样本与预测结果 `.parquet`，不跑全量训练 |

> 约定：深度学习训练只允许在 A 上进行；C 的机器不得加载全量面板。

## 2. 数据

- 数据源：<https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page> → *High Volume For-Hire Vehicle Trip Records*
- 空间粒度：`Taxi Zone ID`（1–263）
- 时间粒度：15 分钟
- 需求定义：某 `(timestamp, zone_id)` 内，以 `PULocationID` 计的**上车订单数**
- 建议同时下载：
  - Taxi Zone Lookup（`taxi_zone_lookup.csv`）
  - Taxi Zone Centroids / Shapefile（仿真距离计算与地图展示）

下载后将原始 `.parquet` 放到：

```text
data/raw/fhvhv_tripdata_2023-01.parquet
data/raw/taxi_zone_lookup.csv
data/raw/taxi_zones.shp  (+ .dbf/.shx/.prj)
```

> 默认配置对应 2023-01 单月数据，时间区间为 [2023-01-01, 2023-02-01)。
> shapefile 需将 .shp/.dbf/.shx/.prj 放在同一目录；若解压到子目录，请同步修改 `paths.zone_shapefile`。

> 质心文件需包含 `LocationID` + (`lat`,`lon`) 或 (`X`,`Y`)。若 TLC 只提供 shapefile，可用 geopandas 生成：
>
> ```python
> import geopandas as gpd, pandas as pd
> gdf = gpd.read_file("data/raw/taxi_zones.shp").to_crs(4326)
> c = gdf.geometry.centroid
> pd.DataFrame({"LocationID": gdf["LocationID"], "lat": c.y, "lon": c.x}).to_csv(
>     "data/raw/taxi_zone_centroids.csv", index=False
> )
> ```

**`data/` 已被 `.gitignore` 忽略，任何数据、模型权重都不允许提交到 Git。**

## 3. 快速开始

推荐 Python **3.11**（PyTorch / PyG 生态兼容性最好；当前机器为 3.13，仅跑 LightGBM/仿真没问题）。

```powershell
# 1) 虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# 成员 A 额外安装（按官网选择 CUDA 12.x）
pip install -r requirements-torch.txt

# 2) 生成需求面板
python -m src.data.make_dataset --config configs/default.yaml

# 2.1) 提取小样本（成员 B/C）
python -m src.data.make_sample --config configs/default.yaml

# 3) 特征工程
python -m src.features.build_features --config configs/default.yaml

# 4) LightGBM 基线
python -m src.models.train_lgbm --config configs/default.yaml

# 5) STGCN（仅成员 A）
python -m src.models.train_stgnn --config configs/default.yaml

# 6) 生成统一格式预测结果
python -m src.models.predict --config configs/default.yaml --model lgbm

# 7) 闭环调度仿真
python -m src.simulation.env --config configs/default.yaml --policy forecast

# 8) 可视化
streamlit run src/visualization/app.py
```

## 4. 目录结构

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-torch.txt
├── data/                     # gitignored：raw / interim / processed
│   ├── raw/
│   ├── interim/
│   └── processed/            # .parquet 面板、特征、预测、仿真结果
├── notebooks/                # EDA 与实验
├── src/
│   ├── config.py             # YAML 配置加载 + 路径解析
│   ├── data/
│   │   └── make_dataset.py   # HVFHV → 15min × zone 面板
│   ├── features/
│   │   └── build_features.py # 滞后、滑窗、日历特征
│   ├── models/
│   │   ├── train_lgbm.py
│   │   ├── train_stgnn.py
│   │   ├── stgnn.py
│   │   └── predict.py
│   ├── simulation/
│   │   ├── env.py            # SimPy 离散事件仿真
│   │   └── policy.py         # Greedy / Forecast-driven
│   └── visualization/
│       └── app.py            # Streamlit + Plotly/Folium
├── configs/
│   └── default.yaml
└── docs/
    └── api_contract.md
```

## 5. 时间切分与防泄漏（硬性规范）

- **严禁随机划分**，一律按 `timestamp` 排序后切分：前 70% 训练、中 15% 验证、后 15% 测试。
- 滞后特征只能使用 `t-1, t-2, ...`；滚动统计必须先 `shift(1)` 再 `rolling`。
- 标准化 / 归一化统计量只能由**训练集**计算，再应用到验证集与测试集。
- 不做跨时间随机插值、不把测试期信息回填到特征。
- 所有中间文件使用 `.parquet`（`zstd` 压缩），禁止用 CSV 传输大文件。

## 6. 接口契约

详见 [`docs/api_contract.md`](docs/api_contract.md)。核心两条：

- 数据 → 模型：`timestamp`, `zone_id`, `demand`, `feature_*`
- 模型 → 仿真：`timestamp`, `zone_id`, `predicted_demand`

## 7. 协作规范

- `main` 只保留可运行版本；每人从 `dev` 拉 `feature/<name>` 分支开发。
- 提交前检查：不含 `data/`、`.pth`、`.venv/`、密钥。
- 大文件通过网盘 / 群文件共享，Git 仓库只放代码与文档。
- 每次实验记录：数据时间范围、特征版本、超参数、随机种子、验证集指标。

## 8. 十周里程碑（建议）

| 周次 | 目标 |
| --- | --- |
| 1 | 仓库、数据下载、字段理解、需求面板 |
| 2 | EDA、时间/空间分布、异常值处理 |
| 3 | 特征工程 + 时间切分 + LightGBM 基线 |
| 4 | 消融实验、滞后窗口与损失函数对比 |
| 5 | 图构建（邻接矩阵）、STGCN 跑通 |
| 6 | 模型调优、预测误差分析 |
| 7 | SimPy 环境与 Greedy 策略 |
| 8 | Forecast-driven 预调度与对比实验 |
| 9 | Streamlit 看板、端到端联调 |
| 10 | 报告、答辩材料、代码整理 |

## 9. 推送到 GitHub

在 GitHub 网页新建**空仓库**（不要勾选自动生成 README），然后：

```powershell
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
git push origin dev
```

首次推送会要求登录。也可使用 SSH：

```powershell
ssh-keygen -t ed25519 -C "your_email@example.com"
# 将 ~/.ssh/id_ed25519.pub 添加到 GitHub → Settings → SSH and GPG keys
git remote add origin git@github.com:<你的用户名>/<仓库名>.git
```
