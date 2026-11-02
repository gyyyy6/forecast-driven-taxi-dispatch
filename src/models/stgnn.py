"""轻量 STGCN（纯 PyTorch 实现，无需 PyG/DGL）。

设计:
    - 输入: ``[B, T, N, F]``。F=1 时等价于只输入历史 demand；
      F>1 时可输入 build_features.py 导出的多变量特征张量。
    - 图卷积: 使用稠密邻接矩阵 A（默认行归一化 + 自环）聚合邻居。
    - 时间卷积: 沿时间轴的 1D 卷积。
    - 输出: 下一时间片（或未来 pred_len 个时间片）各区域需求 ``[B, N, pred_len]``。

稠密邻接 263×263 很小，单卡 6G/8G 均可轻松运行。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.features.graph import (
    build_centroid_adjacency,
    row_normalize,
)


# ---------------------------------------------------------------------------
# 图结构
# ---------------------------------------------------------------------------
def build_adjacency_from_centroids(
    centroids_path: str | Path, zones: np.ndarray, k: int = 8
) -> np.ndarray:
    """兼容旧接口：用 Taxi Zone 质心构造行归一化 kNN 高斯核邻接矩阵。"""
    adj, _ = build_centroid_adjacency(
        centroids_path, zones, k=k, normalize="row", add_self_loops=True
    )
    return adj


def _load_adjacency_file(path: Path) -> np.ndarray:
    """安全读取 .npy 或 .npz 文件。"""
    data = np.load(path, allow_pickle=False)
    # 区分 npz（有 files 属性）和 npy
    if hasattr(data, "files"):  # npz
        with data:
            key = "adjacency" if "adjacency" in data.files else data.files[0]
            arr = np.asarray(data[key], dtype=np.float32)
    else:
        arr = np.asarray(data, dtype=np.float32)
        
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"邻接矩阵必须为方阵，{path} 实际 shape={arr.shape}")
    return arr


def load_or_build_adjacency(
    adjacency_path: str | Path,
    centroids_path: str | Path,
    zones: np.ndarray,
) -> tuple[np.ndarray, str]:
    adjacency_path = Path(adjacency_path)
    if adjacency_path.exists():
        adj = _load_adjacency_file(adjacency_path)
        if adj.shape != (len(zones), len(zones)):
            raise ValueError(
                f"邻接矩阵形状 {adj.shape} 与区域数 {len(zones)} 不符: {adjacency_path}"
            )
        return row_normalize(adj), f"loaded:{adjacency_path}"

    if Path(centroids_path).exists():
        adj = build_adjacency_from_centroids(centroids_path, zones)
        adjacency_path.parent.mkdir(parents=True, exist_ok=True)
        if adjacency_path.suffix.lower() == ".npy":
            np.save(adjacency_path, adj, allow_pickle=False)
        else:
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


def load_feature_tensor(
    path: str | Path,
    zones: np.ndarray | None = None,
    timestamps: pd.DatetimeIndex | None = None,
) -> tuple[np.ndarray, dict]:
    """读取 ``[T, N, F]`` 特征张量，并校验与当前面板的节点/时间对齐。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"特征张量不存在: {p}（请先运行 build_features.py）")
    arr = np.load(p, allow_pickle=False)
    if arr.ndim != 3:
        raise ValueError(f"特征张量应为 [T, N, F]，实际 shape={arr.shape}")
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("特征张量包含 NaN/Inf，请重新运行 build_features.py")

    meta_path = p.with_suffix(".meta.json")
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if zones is not None:
        z = np.asarray(zones, dtype=np.int64)
        if arr.shape[1] != len(z):
            raise ValueError(f"特征张量节点数 {arr.shape[1]} != {len(z)}")
        meta_zones = meta.get("zones")
        if meta_zones is not None and not np.array_equal(np.asarray(meta_zones, dtype=np.int64), z):
            raise ValueError("特征张量的 zone 顺序与当前面板不一致，请重新运行 build_features.py")

    if timestamps is not None:
        ts = pd.DatetimeIndex(timestamps)
        if arr.shape[0] != len(ts):
            raise ValueError(
                f"特征张量时间片数 {arr.shape[0]} != 面板 {len(ts)}；"
                "请确认两者来自同一面板，或训练时加 --no-features"
            )
        meta_ts = meta.get("timestamps")
        if meta_ts is not None:
            parsed = pd.to_datetime(meta_ts)
            if len(parsed) != len(ts) or not parsed.equals(ts):
                raise ValueError("特征张量的时间戳与当前面板不一致，请重新运行 build_features.py")

    return arr, meta


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
    """仅 demand 输入：``__getitem__`` 返回 ``[T, N]``，batch 后为 ``[B, T, N]``。"""

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


class TensorWindowDataset(Dataset):
    """多变量输入：返回 ``[T, N, F]``，DataLoader 组 batch 后为 ``[B, T, N, F]``。"""

    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        starts: np.ndarray,
        seq_len: int,
        pred_len: int,
    ):
        if features.ndim != 3:
            raise ValueError(f"features 应为 [T, N, F]，实际 {features.shape}")
        if target.ndim != 2:
            raise ValueError(f"target 应为 [T, N]，实际 {target.shape}")
        if features.shape[0] != target.shape[0] or features.shape[1] != target.shape[1]:
            raise ValueError("features 与 target 的时间/节点维度不一致")
        self.features = features
        self.target = target
        self.starts = starts
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i])
        x = self.features[s : s + self.seq_len]
        y = self.target[s + self.seq_len : s + self.seq_len + self.pred_len]
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
        in_features: int = 1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.in_features = int(in_features)
        self.input_proj = nn.Linear(self.in_features, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                STBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_channels, pred_len)
        self.n_zones = n_zones

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # 兼容 [B,T,N] 与 [B,T,N,F]
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        elif x.dim() != 4:
            raise ValueError(f"输入应为 [B,T,N] 或 [B,T,N,F]，实际 {tuple(x.shape)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"输入特征数 {x.shape[-1]} 与模型 in_features={self.in_features} 不一致"
            )
        h = self.input_proj(x)
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