"""轻量 STGCN（纯 PyTorch 实现，无需 PyG/DGL）。

设计:
    - 输入: 过去 T 个时间片、N 个区域的 demand 矩阵
    - 图卷积: 使用稠密邻接矩阵 A（行归一化）聚合邻居
    - 时间卷积: 沿时间轴的 1D 卷积
    - 输出: 下一时间片（或未来 pred_len 个时间片）各区域需求

稠密邻接 263×263 很小，单卡 6G/8G 均可轻松运行。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 图结构
# ---------------------------------------------------------------------------
def row_normalize(adj: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    a = np.asarray(adj, dtype=np.float32).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"邻接矩阵必须为方阵，当前 {a.shape}")
    if add_self_loops:
        np.fill_diagonal(a, 1.0)
    # 只保留非负权重
    a = np.clip(a, 0, None)
    deg = a.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return (a / deg).astype(np.float32)


def build_adjacency_from_centroids(
    centroids_path: str | Path, zones: np.ndarray, k: int = 8
) -> np.ndarray:
    """用 Taxi Zone 质心构造 kNN 高斯核邻接矩阵。"""
    from sklearn.neighbors import NearestNeighbors

    df = pd.read_csv(centroids_path)
    lower = {c.lower(): c for c in df.columns}
    id_col = lower.get("locationid") or lower.get("location_id") or lower.get("zone_id")
    x_col = lower.get("x")
    y_col = lower.get("y")
    if id_col is None or x_col is None or y_col is None:
        raise ValueError(
            f"{centroids_path} 需包含 LocationID/X/Y 列，实际列: {list(df.columns)}"
        )
    df = df.set_index(id_col).reindex(zones)
    if df[[x_col, y_col]].isna().any().any():
        raise ValueError("质心文件缺少部分 Taxi Zone，无法构建邻接矩阵")

    coords = df[[x_col, y_col]].to_numpy(np.float64)
    n = len(coords)
    k = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(coords)
    dist, idx = nn.kneighbors(coords)
    sigma = float(np.median(dist[:, 1:])) if k > 1 else 1.0
    sigma = max(sigma, 1e-6)

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j, d in zip(idx[i], dist[i]):
            adj[i, j] = np.exp(-((d / sigma) ** 2))
    adj = np.maximum(adj, adj.T)
    return row_normalize(adj)


def load_or_build_adjacency(
    adjacency_path: str | Path,
    centroids_path: str | Path,
    zones: np.ndarray,
) -> tuple[np.ndarray, str]:
    adjacency_path = Path(adjacency_path)
    if adjacency_path.exists():
        data = np.load(adjacency_path)
        key = "adjacency" if "adjacency" in data else list(data.keys())[0]
        return row_normalize(data[key]), f"loaded:{adjacency_path}"

    if Path(centroids_path).exists():
        adj = build_adjacency_from_centroids(centroids_path, zones)
        adjacency_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(adjacency_path, adjacency=adj, zones=zones)
        return adj, f"built_from_centroids:{centroids_path}"

    # 兜底：单位矩阵，等价于逐区域时间模型；仅用于先把流程跑通
    return np.eye(len(zones), dtype=np.float32), "identity_fallback"


# ---------------------------------------------------------------------------
# 数据矩阵与窗口
# ---------------------------------------------------------------------------
@dataclass
class ZoneMatrix:
    timestamps: pd.DatetimeIndex
    zones: np.ndarray
    values: np.ndarray  # [T, N] float32


def load_zone_matrix(panel_path: str | Path, zone_min: int = 1, zone_max: int = 263) -> ZoneMatrix:
    df = pd.read_parquet(panel_path, columns=["timestamp", "zone_id", "demand"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    ts = pd.DatetimeIndex(np.sort(df["timestamp"].unique()))
    zones = np.arange(zone_min, zone_max + 1, dtype=np.int16)

    wide = (
        df.pivot(index="timestamp", columns="zone_id", values="demand")
        .reindex(index=ts, columns=zones)
        .fillna(0.0)
    )
    return ZoneMatrix(timestamps=ts, zones=zones, values=wide.to_numpy(np.float32))


def fit_scaler(values: np.ndarray, train_end_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """只用训练期数据拟合 log1p 后的均值/标准差。"""
    x = np.log1p(values[:train_end_idx])
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_scaler(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.log1p(values) - mean) / std).astype(np.float32)


def inverse_scaler(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = values * std + mean
    x = np.clip(x, 0, 20.0)  # log1p(4.8e8)，足够覆盖需求
    return np.expm1(x).astype(np.float32)


def split_indices(n_timesteps: int, train: float, val: float) -> tuple[int, int]:
    n_train = int(n_timesteps * train)
    n_val = int(n_timesteps * val)
    if n_train <= 0 or n_val <= 0 or n_train + n_val >= n_timesteps:
        raise ValueError("时间切分比例无效")
    return n_train, n_train + n_val


def window_starts(
    n_timesteps: int, seq_len: int, pred_len: int, target_start: int, target_end: int
) -> np.ndarray:
    """返回所有窗口起点 s，使预测目标落在 [target_start, target_end)。"""
    first = max(0, target_start - seq_len)
    last = min(n_timesteps - seq_len - pred_len, target_end - seq_len - pred_len)
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64)


class WindowDataset(Dataset):
    """按索引惰性切片，避免把 12×263 的窗口全部物化到内存。"""

    def __init__(self, matrix: np.ndarray, starts: np.ndarray, seq_len: int, pred_len: int):
        self.matrix = matrix
        self.starts = starts
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i])
        x = self.matrix[s : s + self.seq_len]
        y = self.matrix[s + self.seq_len : s + self.seq_len + self.pred_len]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class GraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C] -> [B, T, N, C_out]
        h = self.linear(x)
        h = torch.einsum("nm,btmc->btnc", adj, h)
        return h + self.bias


class STBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.gc = GraphConv(c_in, c_out)
        self.tconv = nn.Conv2d(c_out, c_out, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        self.norm = nn.LayerNorm(c_out)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(c_in, c_out) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gc(x, adj))
        h = h.permute(0, 3, 1, 2)  # [B, C, T, N]
        h = self.tconv(h)
        h = h.permute(0, 2, 3, 1)  # [B, T, N, C]
        h = self.norm(h)
        h = self.dropout(h)
        return F.relu(h + self.residual(x))


class STGCN(nn.Module):
    def __init__(
        self,
        n_zones: int,
        hidden_channels: int = 32,
        pred_len: int = 1,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.input_proj = nn.Linear(1, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                STBlock(
                    hidden_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_channels, pred_len)
        self.n_zones = n_zones

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N] -> [B, T, N, 1]
        h = self.input_proj(x.unsqueeze(-1))
        for block in self.blocks:
            h = block(h, adj)
        # 用最后一个时间片的表示预测未来
        out = self.head(h[:, -1])  # [B, N, pred_len]
        return out


@torch.no_grad()
def predict_matrix(
    model: nn.Module,
    matrix: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    pred_len: int = 1,
) -> np.ndarray:
    """对一个 split 的所有窗口做推理，返回 [n_windows, N, pred_len]。"""
    model.eval()
    outputs: list[np.ndarray] = []
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i : i + batch_size]
        x = np.stack([matrix[s : s + seq_len] for s in batch_starts]).astype(np.float32)
        xb = torch.from_numpy(x).to(device)
        pred = model(xb, model._adj)  # type: ignore[attr-defined]
        outputs.append(pred.cpu().numpy())
    if not outputs:
        return np.empty((0, matrix.shape[1], pred_len), dtype=np.float32)
    return np.concatenate(outputs, axis=0)