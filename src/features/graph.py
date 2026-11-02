"""图结构构建工具：Taxi Zone 质心 kNN / 历史订单 OD 共现邻接矩阵。

设计约定
--------
- 节点顺序固定为 Taxi Zone 1..263（由 configs/default.yaml 的 data.zone_min/zone_max 决定）。
- 输出矩阵形状为 ``(N, N)``，默认做行归一化并加自环，适合
  ``A[n, m]`` 聚合邻居的 GCN/STGCN 实现。
- 同时支持两种空间关系：
  1. ``centroid``：质心 kNN 高斯核，适合没有 OD 明细或冷启动区域；
  2. ``od``：历史订单起终点共现频次，反映真实出行流向；
  3. ``hybrid``：两者加权融合。
- 结果保存为 ``.npy``（默认 ``data/processed/adj_matrix.npy``），
  并在同目录写 ``*.meta.json`` 记录方法、参数与诊断信息。

用法
----
    python build_graph.py --config configs/default.yaml --method centroid
    python build_graph.py --method od --out data/processed/adj_matrix_od.npy
    python build_graph.py --method hybrid --alpha 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import ensure_parent, load_config, make_logger, resolve

log = make_logger("build_graph")

EARTH_RADIUS_M = 6_371_008.8
PU_COL = "PULocationID"
DO_COL = "DOLocationID"


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
def row_normalize(adj: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """行归一化：每个节点对邻居的权重和为 1。"""
    a = np.asarray(adj, dtype=np.float32).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"邻接矩阵必须为方阵，当前 {a.shape}")
    if add_self_loops:
        np.fill_diagonal(a, 1.0)
    a = np.clip(a, 0, None)
    deg = a.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return (a / deg).astype(np.float32)


def symmetric_normalize(adj: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """对称归一化：D^{-1/2} (A + I) D^{-1/2}。"""
    a = np.asarray(adj, dtype=np.float32).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"邻接矩阵必须为方阵，当前 {a.shape}")
    if add_self_loops:
        np.fill_diagonal(a, 1.0)
    a = np.clip(a, 0, None)
    deg = a.sum(axis=1)
    inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    nz = deg > 0
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    return (a * inv_sqrt[:, None] * inv_sqrt[None, :]).astype(np.float32)


def normalize_adjacency(
    adj: np.ndarray, mode: str = "row", add_self_loops: bool = True
) -> np.ndarray:
    mode = mode.lower()
    if mode == "row":
        return row_normalize(adj, add_self_loops=add_self_loops)
    if mode in {"sym", "symmetric"}:
        return symmetric_normalize(adj, add_self_loops=add_self_loops)
    if mode == "none":
        a = np.asarray(adj, dtype=np.float32).copy()
        if add_self_loops:
            np.fill_diagonal(a, 1.0)
        return a
    raise ValueError(f"未知归一化方式: {mode}（可选 row / sym / none）")


# ---------------------------------------------------------------------------
# 质心
# ---------------------------------------------------------------------------
def load_zone_centroids(
    centroids_path: str | Path, zones: Sequence[int]
) -> tuple[np.ndarray, str]:
    """读取质心并返回 ``(N, 2)`` 坐标与坐标类型 ``degrees`` / ``xy``。

    兼容两种常见格式：
    - ``LocationID, lat, lon``（TLC 常见发布格式）；
    - ``LocationID, X, Y``（shapefile 投影坐标，单位可能是英尺）。
    """
    path = Path(centroids_path)
    if not path.exists():
        raise FileNotFoundError(f"质心文件不存在: {path}")
    df = pd.read_csv(path)
    cols = {str(c).lower(): c for c in df.columns}

    id_col = next(
        (cols[k] for k in ("locationid", "location_id", "zone_id", "zone", "id") if k in cols),
        None,
    )
    lat_col = next((cols[k] for k in ("lat", "latitude") if k in cols), None)
    lon_col = next((cols[k] for k in ("lon", "lng", "longitude") if k in cols), None)
    x_col = cols.get("x")
    y_col = cols.get("y")

    if id_col is None:
        raise ValueError(f"{path} 缺少 LocationID 列，实际列: {list(df.columns)}")

    if lat_col is not None and lon_col is not None:
        coord_cols = [lat_col, lon_col]
        coord_mode = "degrees"
    elif x_col is not None and y_col is not None:
        coord_cols = [x_col, y_col]
        coord_mode = "xy"
    else:
        raise ValueError(
            f"{path} 需包含 lat/lon 或 X/Y 列，实际列: {list(df.columns)}"
        )

    indexed = df.set_index(id_col)
    if indexed.index.duplicated().any():
        raise ValueError(f"{path} 中 LocationID 存在重复")
    indexed = indexed.reindex(list(zones))
    if indexed[coord_cols].isna().any().any():
        missing = indexed.index[indexed[coord_cols].isna().any(axis=1)].tolist()
        raise ValueError(f"{path} 缺少以下 Taxi Zone 的质心: {missing[:20]}")

    coords = indexed[coord_cols].to_numpy(dtype=np.float64)
    return coords, coord_mode


def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """``coords`` 为 ``(N, 2)`` 的 ``[lat, lon]``（度），返回米制距离矩阵。"""
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def euclidean_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def knn_gaussian_adjacency(
    distance: np.ndarray, k: int = 8, sigma: float | None = None
) -> np.ndarray:
    """基于距离矩阵构造对称 kNN 高斯核权重（不含自环）。"""
    d = np.asarray(distance, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"距离矩阵必须为方阵，当前 {d.shape}")
    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=np.float32)
    k = max(1, min(int(k), n - 1))

    order = np.argsort(d, axis=1)
    neighbors = order[:, 1 : k + 1]
    neighbor_dist = np.take_along_axis(d, neighbors, axis=1)

    if sigma is None:
        sigma = float(np.median(neighbor_dist))
    sigma = max(float(sigma), 1e-6)

    weights = np.exp(-((neighbor_dist / sigma) ** 2)).astype(np.float32)
    adj = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), k)
    adj[rows, neighbors.reshape(-1)] = weights.reshape(-1)
    # 对称化，避免单向 kNN 造成孤立/有向偏差
    adj = np.maximum(adj, adj.T)
    return adj


def build_centroid_adjacency(
    centroids_path: str | Path,
    zones: Sequence[int],
    k: int = 8,
    sigma: float | None = None,
    normalize: str = "row",
    add_self_loops: bool = True,
) -> tuple[np.ndarray, dict]:
    coords, coord_mode = load_zone_centroids(centroids_path, zones)
    if coord_mode == "degrees":
        distance = haversine_matrix(coords)
        unit = "meter"
    else:
        distance = euclidean_matrix(coords)
        unit = "xy_unit"

    raw = knn_gaussian_adjacency(distance, k=k, sigma=sigma)
    adj = normalize_adjacency(raw, mode=normalize, add_self_loops=add_self_loops)
    info = {
        "method": "centroid",
        "centroids": str(centroids_path),
        "coord_mode": coord_mode,
        "distance_unit": unit,
        "k": int(k),
        "sigma": None if sigma is None else float(sigma),
        "normalize": normalize,
        "add_self_loops": bool(add_self_loops),
    }
    return adj, info


# ---------------------------------------------------------------------------
# OD 共现
# ---------------------------------------------------------------------------
def _zone_lookup(zones: Sequence[int]) -> np.ndarray:
    """把 Taxi Zone ID 映射到 0..N-1；未知区域映射为 -1。"""
    z = np.asarray(list(zones), dtype=np.int64)
    max_id = max(int(z.max()), 265)
    lookup = np.full(max_id + 1, -1, dtype=np.int64)
    lookup[z] = np.arange(len(z), dtype=np.int64)
    return lookup


def build_od_adjacency(
    files: Iterable[str | Path],
    zones: Sequence[int],
    log1p: bool = True,
    normalize: str = "row",
    add_self_loops: bool = True,
) -> tuple[np.ndarray, dict]:
    """从原始 HVFHV parquet 的 PULocationID / DOLocationID 统计 OD 共现。"""
    files = [Path(f) for f in files]
    if not files:
        raise FileNotFoundError("未提供 OD 原始 parquet 文件")

    n = len(zones)
    lookup = _zone_lookup(zones)
    counts = np.zeros(n * n, dtype=np.int64)
    total_rows = 0
    valid_rows = 0

    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"OD 原始文件不存在: {path}")
        df = pd.read_parquet(path, columns=[PU_COL, DO_COL])
        total_rows += len(df)
        pu = pd.to_numeric(df[PU_COL], errors="coerce").to_numpy(dtype=np.float64)
        do = pd.to_numeric(df[DO_COL], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(pu) & np.isfinite(do)
        pu_i = pu[finite].astype(np.int64)
        do_i = do[finite].astype(np.int64)
        in_range = (pu_i >= 0) & (pu_i < len(lookup)) & (do_i >= 0) & (do_i < len(lookup))
        pu_i = pu_i[in_range]
        do_i = do_i[in_range]
        pu_idx = lookup[pu_i]
        do_idx = lookup[do_i]
        known = (pu_idx >= 0) & (do_idx >= 0)
        pu_idx = pu_idx[known]
        do_idx = do_idx[known]
        valid_rows += len(pu_idx)
        counts += np.bincount(pu_idx * n + do_idx, minlength=n * n)
        log.info(
            "  OD %s: %d 行，有效 %d 行，累计有效 %d",
            path.name,
            len(df),
            len(pu_idx),
            valid_rows,
        )

    mat = counts.reshape(n, n).astype(np.float64)
    np.fill_diagonal(mat, 0.0)  # 自环由归一化阶段统一添加
    if log1p:
        mat = np.log1p(mat)
    adj = normalize_adjacency(mat.astype(np.float32), mode=normalize, add_self_loops=add_self_loops)
    info = {
        "method": "od",
        "files": [str(f) for f in files],
        "total_rows": int(total_rows),
        "valid_rows": int(valid_rows),
        "nonzero_pairs": int(np.count_nonzero(counts)),
        "log1p": bool(log1p),
        "normalize": normalize,
        "add_self_loops": bool(add_self_loops),
    }
    return adj, info


# ---------------------------------------------------------------------------
# 保存与校验
# ---------------------------------------------------------------------------
def validate_adjacency(adj: np.ndarray, n_zones: int) -> dict:
    a = np.asarray(adj, dtype=np.float32)
    if a.shape != (n_zones, n_zones):
        raise ValueError(f"邻接矩阵形状应为 {(n_zones, n_zones)}，实际 {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("邻接矩阵包含 NaN/Inf")
    if (a < 0).any():
        raise ValueError("邻接矩阵包含负权重")
    row_sums = a.sum(axis=1)
    return {
        "shape": [int(a.shape[0]), int(a.shape[1])],
        "nnz": int(np.count_nonzero(a)),
        "density": float(np.count_nonzero(a) / a.size),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "row_sum_mean": float(row_sums.mean()),
        "symmetric": bool(np.allclose(a, a.T, atol=1e-5)),
    }


def save_adjacency(
    adj: np.ndarray,
    out_path: str | Path,
    meta: dict,
    zones: Sequence[int],
) -> tuple[Path, Path]:
    path = Path(out_path)
    if path.suffix.lower() != ".npy":
        path = path.with_suffix(".npy")
    path = ensure_parent(path)
    np.save(path, np.asarray(adj, dtype=np.float32), allow_pickle=False)

    meta_path = path.with_suffix(".meta.json")
    payload = dict(meta)
    payload.update(
        {
            "preprocessed": True,
            "zones": [int(z) for z in zones],
            "saved_to": str(path),
            "diagnostics": validate_adjacency(adj, len(zones)),
        }
    )
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path, meta_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 263x263 Taxi Zone adjacency matrix.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--method", choices=["centroid", "od", "hybrid"], default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--centroids", default=None)
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--raw-glob", default=None)
    p.add_argument("--k", type=int, default=None, help="质心 kNN 邻居数")
    p.add_argument("--sigma", type=float, default=None, help="高斯核带宽；默认取 kNN 距离中位数")
    p.add_argument("--normalize", choices=["row", "sym", "none"], default=None)
    p.add_argument("--no-self-loops", action="store_true")
    p.add_argument("--no-od-log1p", action="store_true")
    p.add_argument("--alpha", type=float, default=None, help="hybrid 中 OD 的权重")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    gcfg = cfg.get("graph", {})
    dcfg = cfg.get("data", {})

    zone_min = int(dcfg.get("zone_min", 1))
    zone_max = int(dcfg.get("zone_max", 263))
    zones = np.arange(zone_min, zone_max + 1, dtype=np.int16)
    n_zones = len(zones)

    method = args.method or gcfg.get("method", "centroid")
    out_path = args.out or cfg["paths"].get("adjacency", "data/processed/adj_matrix.npy")
    centroids_path = resolve(args.centroids or cfg["paths"]["zone_centroids"])
    raw_dir = resolve(args.raw_dir or cfg["paths"]["raw_dir"])
    raw_glob = args.raw_glob or cfg["paths"].get("raw_glob", "*.parquet")
    k = int(args.k if args.k is not None else gcfg.get("k", 8))
    sigma = args.sigma if args.sigma is not None else gcfg.get("sigma")
    normalize = args.normalize or gcfg.get("normalize", "row")
    add_self_loops = not args.no_self_loops and bool(gcfg.get("add_self_loops", True))
    od_log1p = not args.no_od_log1p and bool(gcfg.get("od_log1p", True))
    alpha = float(args.alpha if args.alpha is not None else gcfg.get("hybrid_alpha", 0.5))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--alpha 必须在 [0, 1] 内")

    log.info(
        "构建邻接矩阵: method=%s, zones=%d..%d (%d), k=%d, normalize=%s",
        method, zone_min, zone_max, n_zones, k, normalize,
    )

    if method == "centroid":
        adj, info = build_centroid_adjacency(
            centroids_path, zones, k=k, sigma=sigma,
            normalize=normalize, add_self_loops=add_self_loops,
        )
    elif method == "od":
        files = sorted(raw_dir.glob(raw_glob))
        adj, info = build_od_adjacency(
            files, zones, log1p=od_log1p,
            normalize=normalize, add_self_loops=add_self_loops,
        )
    else:
        files = sorted(raw_dir.glob(raw_glob))
        c_adj, c_info = build_centroid_adjacency(
            centroids_path, zones, k=k, sigma=sigma,
            normalize="row", add_self_loops=False,
        )
        o_adj, o_info = build_od_adjacency(
            files, zones, log1p=od_log1p,
            normalize="row", add_self_loops=False,
        )
        blended = (alpha * o_adj + (1.0 - alpha) * c_adj).astype(np.float32)
        adj = normalize_adjacency(blended, mode=normalize, add_self_loops=add_self_loops)
        info = {
            "method": "hybrid",
            "alpha_od": alpha,
            "centroid": c_info,
            "od": o_info,
            "normalize": normalize,
            "add_self_loops": bool(add_self_loops),
        }

    diag = validate_adjacency(adj, n_zones)
    log.info(
        "矩阵 shape=%s, nnz=%d, density=%.4f, row_sum=[%.4f, %.4f], symmetric=%s",
        diag["shape"], diag["nnz"], diag["density"],
        diag["row_sum_min"], diag["row_sum_max"], diag["symmetric"],
    )
    path, meta_path = save_adjacency(adj, out_path, info, zones)
    log.info("已保存 %s", path)
    log.info("元信息 %s", meta_path)


if __name__ == "__main__":
    main()