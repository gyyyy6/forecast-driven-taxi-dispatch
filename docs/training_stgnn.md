# STGCN 图构建、特征张量与训练监控

> 角色：成员 A（RTX 4060 8G）
> 相关代码：`build_graph.py`、`src/features/build_features.py`、`src/models/stgnn.py`、`src/models/train_stgnn.py`

## 1. 构建 263×263 邻接矩阵

`build_graph.py` 支持三种方式：

| 方式 | 命令 | 适用场景 |
| --- | --- | --- |
| 质心 kNN | `python build_graph.py --method centroid` | 默认；不依赖原始 OD 明细，冷启动区域也能连边 |
| OD 共现 | `python build_graph.py --method od` | 有历史订单，想表达真实出行流向 |
| 混合 | `python build_graph.py --method hybrid --alpha 0.5` | 兼顾地理邻近与真实流向，`alpha` 是 OD 权重 |

默认输出：

```text
data/processed/adj_matrix.npy         # float32, (263, 263), 行归一化 + 自环
data/processed/adj_matrix.meta.json   # 方法、参数、nnz、行和、对称性
```

常用参数：

```powershell
# 只构建质心 kNN，检查诊断信息
python build_graph.py --method centroid --k 8

# 构建 OD 矩阵，单独保存，便于对比
python build_graph.py --method od --out data/processed/adj_matrix_od.npy

# 对称归一化（D^-1/2 A D^-1/2），不加自环
python build_graph.py --method centroid --normalize sym --no-self-loops
```

快速检查：

```powershell
python -c "import numpy as np; a=np.load('data/processed/adj_matrix.npy'); print(a.shape, a.dtype, a.min(), a.max(), a.sum(1)[:5])"
```

预期 `shape=(263, 263)`，行和约为 1（`--normalize row` 且加自环时）。

> 质心文件是 `LocationID, lat, lon` 时，脚本会用 haversine 计算米制距离；如果是 `LocationID, X, Y`，则按平面欧氏距离处理。旧代码只认 `X/Y`，会导致 `lat/lon` 文件报错，这里已修正。

## 2. 特征张量：`[T, N, F]` → `[B, T, N, F]`

`build_features.py` 现在同时产出两类文件：

| 文件 | 形状/用途 |
| --- | --- |
| `features.parquet` | 长表 `(timestamp, zone_id, ...)`，LightGBM 使用 |
| `feature_tensor.npy` | `[num_timesteps, num_nodes, num_features]`，STGCN 使用 |
| `feature_tensor.meta.json` | 时间戳、区域顺序、特征名、标准化参数 |

运行：

```powershell
python -m src.features.build_features --config configs/default.yaml
```

预期日志会打印：

```text
STGCN 张量 -> .../feature_tensor.npy, shape=(2976, 263, 17), dtype=float32
DataLoader batch 形状: [batch, 12, 263, 17]
```

为什么磁盘上是 `[T, N, F]` 而不是直接 `[B, T, N, F]`？

- 如果把所有滑窗物化，约 `2964 × 12 × 263 × 17 × 4B ≈ 636 MB`，而且每次改 `seq_len` 都要重建；
- 现在 `TensorWindowDataset.__getitem__` 只切一个窗口 `[T, N, F]`，DataLoader 自动 collate 成 `[B, T, N, F]`，内存更省，`seq_len` 可随时通过命令行改。

关键防泄漏点：

- `lag_*` / `roll_mean_*` 都只用 `t-1` 及之前的数据；
- 张量标准化只用训练期（前 70% 时间片）的均值/方差；
- 月初 `lag_672` 等不足导致的 NaN，在标准化后填 0（训练期均值位置），不会把未来信息带进来；
- `demand` 不作为输入特征，只作为标签。

## 3. 训练 STGCN

全量训练（默认自动读取 `feature_tensor.npy`）：

```powershell
python -m src.models.train_stgnn --config configs/default.yaml --tag full_v1 --plot
```

只用历史 demand（不使用多变量张量）：

```powershell
python -m src.models.train_stgnn --config configs/default.yaml --no-features --tag demand_v1
```

快速冒烟（不会覆盖正式 `pred_test.parquet` / `stgnn_latest.*`）：

```powershell
python -m src.models.train_stgnn --config configs/default.yaml `
  --panel data/processed/sample_day.parquet `
  --epochs 2 --batch-size 8 --device cpu `
  --no-features --no-predict --no-canonical --tag smoke
```

显存不足时：

```powershell
python -m src.models.train_stgnn --batch-size 8 --hidden 16
```

## 4. 监控训练过程

### 4.1 看日志

每个 epoch 会打印：

```text
epoch 001 | train_loss 0.8123 | val_MAE(norm) 0.6543 | lr 1.00e-03 | 42.1s
```

- `train_loss`：SmoothL1，目标已做 per-zone `log1p + z-score`，所以是归一化尺度；
- `val_MAE(norm)`：验证集 L1，同样是归一化尺度，用于早停/选最佳模型；
- 测试集另打印 `MAE / RMSE / WAPE`（原始订单数尺度）。

### 4.2 训练历史文件

每次训练都会写：

```text
data/processed/models/stgnn_<tag>_history.parquet
data/processed/models/stgnn_<tag>_history.csv
data/processed/models/stgnn_<tag>.metrics.json
```

`metrics.json` 里有完整的 `history` 数组，CSV 可直接用 Excel 画图。

### 4.3 TensorBoard

安装：

```powershell
pip install tensorboard
```

启动训练时加 `--tensorboard`：

```powershell
python -m src.models.train_stgnn --config configs/default.yaml --tensorboard --tag full_v1
tensorboard --logdir data/processed/logs
```

代码中记录的量：

| tag | 含义 |
| --- | --- |
| `loss/train` | 每个 epoch 的训练损失 |
| `mae/val_norm` | 验证集归一化 MAE |
| `lr` | 当前学习率 |
| `time/epoch_sec` | 每个 epoch 耗时 |

如果没装 `tensorboard`，脚本只打印 warning，训练继续，不会中断。

### 4.4 简单 Loss 曲线

加 `--plot`：

```powershell
python -m src.models.train_stgnn --config configs/default.yaml --plot --tag full_v1
```

输出：

```text
data/processed/models/stgnn_<tag>_loss.png
```

蓝色是训练 loss，橙色是验证 MAE（双 y 轴），虚线标出最佳 epoch。也可以指定路径：

```powershell
python -m src.models.train_stgnn --plot --plot-out data/processed/logs/full_v1_loss.png
```

### 4.5 代码层面怎么加

如果以后想自己扩展，关键位置只有三处：

1. 训练循环里记录标量：

```python
if writer is not None:
    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("mae/val_norm", val_mae, epoch)
```

2. 用 `torch.utils.tensorboard.SummaryWriter(log_dir=...)` 创建 writer，训练结束后 `writer.close()`；
3. Loss 曲线用 `src/models/common.py::save_training_curve(history, path)`，它内部用 `matplotlib.use("Agg")`，无显示器/服务器环境也能保存 PNG。

## 5. 常见问题

| 现象 | 处理 |
| --- | --- |
| `CUDA 不可用，回退到 CPU` | 当前 PyTorch 是 CPU 版，或驱动不匹配；成员 A 机器按官网装 CUDA 12.x 版 torch |
| 特征张量时间片/时间戳不一致 | 用同一个 `demand_panel.parquet` 重新跑 `build_features.py`，或加 `--no-features` |
| 邻接矩阵维度不是 263 | 检查 `data.zone_min/zone_max` 与 `--out` 文件；重新跑 `build_graph.py` |
| `pred_test.parquet` 被冒烟实验覆盖 | 冒烟时加 `--no-predict --no-canonical` |
| 训练很慢 | 先用 `sample_day.parquet` 冒烟；全量训练确认 `device=cuda`，`num_workers=0`（Windows） |
| 曲线/日志中文乱码 | 终端执行 `chcp 65001`，或直接看 CSV/PNG |