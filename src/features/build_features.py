"""在需求面板上构造滞后、滚动与日历特征（严格防泄漏）。

用法::
    python -m src.features.build_features --config configs/default.yaml

输出::
    data/processed/features.parquet          # 长表，供 LightGBM 使用
    data/processed/feature_list.json         # 长表特征清单
    data/processed/feature_tensor.npy        # [T, N, F]，供 STGCN 使用
    data/processed/feature_tensor.meta.json  # 张量元信息（时间戳/区域/特征/标准化参数）

关于 STGCN 的维度
------------------
磁盘上的张量是时间优先的 ``[num_timesteps, num_nodes, num_features]``。
训练时 ``TensorWindowDataset.__getitem__`` 切出 ``[seq_len, num_nodes, num_features]``，
DataLoader 自动 collate 成 ``[batch, seq_len, num_nodes, num_features]``。
这样无需把所有滑窗物化到磁盘，内存占用可控。

防泄漏要点
----------
- lag_k 使用同区域 t-k 的真实需求；
- 滚动均值必须先 shift(1)，不能把当前时刻 demand 算进去；
- 张量标准化参数只使用训练期时间片估计，再应用到验证/测试期；
- 不使用任何未来窗口统计。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import holidays
import numpy as np
import pandas as pd

from src.config import ensure_parent, load_config, make_logger, resolve

log = make_logger("build_features")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build leakage-safe features and STGCN tensor.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--input", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--feature-list-out", default=None)
    p.add_argument("--tensor-out", default=None)
    p.add_argument("--tensor-meta-out", default=None)
    p.add_argument("--no-tensor", action="store_true", help="只生成长表，不导出 [T,N,F] 张量")
    p.add_argument("--no-standardize", action="store_true", help="张量不做标准化（默认使用训练期统计量）")
    p.add_argument("--seq-len", type=int, default=None, help="仅写入元信息，便于核对 batch 形状")
    p.add_argument("--pred-len", type=int, default=None)
    return p.parse_args()


def add_calendar(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype("int8")
    df["minute"] = ts.dt.minute.astype("int8")
    df["dow"] = ts.dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")

    years = sorted(ts.dt.year.unique().tolist())
    try:
        ny_holidays = holidays.US(years=years, subdiv="NY")
        holiday_dates = pd.to_datetime(sorted(ny_holidays.keys()))
        df["is_holiday"] = ts.dt.normalize().isin(holiday_dates).astype("int8")
    except Exception as exc:  # holidays 版本差异时降级
        log.warning("holidays 计算失败，is_holiday 置 0: %s", exc)
        df["is_holiday"] = np.int8(0)

    if cfg.get("cyclic", True):
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7).astype("float32")
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7).astype("float32")
    return df


def add_history_features(df: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("zone_id", sort=False)["demand"]

    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag).astype("float32")

    for w in windows:
        # 关键：先 shift(1)，再 rolling，确保只使用 t 之前的信息
        df[f"roll_mean_{w}"] = (
            g.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=w).mean())
            .astype("float32")
        )
    return df


def _is_demand_like(feature: str) -> bool:
    """需求类特征先 log1p 再标准化，减轻长尾/量级差异。"""
    return feature.startswith("lag_") or feature.startswith("roll_mean_")


def build_feature_tensor(
    df: pd.DataFrame,
    features: list[str],
    zones: np.ndarray,
    train_ratio: float = 0.70,
    standardize: bool = True,
) -> tuple[np.ndarray, dict]:
    """把长表转换为 ``[T, N, F]`` 的 float32 张量。

    标准化统计量只在训练期时间片上估计；NaN（例如月初 lag_672 不足）
    在标准化后填 0，即训练期均值位置。
    """
    if df.duplicated(["timestamp", "zone_id"]).any():
        raise ValueError("输入表存在重复的 (timestamp, zone_id)，无法 pivot 成张量")

    timestamps = pd.DatetimeIndex(np.sort(df["timestamp"].unique()))
    zones = np.asarray(zones, dtype=np.int64)
    n_t, n_z, n_f = len(timestamps), len(zones), len(features)
    tensor = np.empty((n_t, n_z, n_f), dtype=np.float32)

    means = np.zeros(n_f, dtype=np.float64)
    stds = np.ones(n_f, dtype=np.float64)
    log_features: list[str] = []
    n_train = max(1, int(n_t * float(train_ratio)))

    for k, feature in enumerate(features):
        wide = (
            df.pivot(index="timestamp", columns="zone_id", values=feature)
            .reindex(index=timestamps, columns=zones)
        )
        values = wide.to_numpy(dtype=np.float64)

        if _is_demand_like(feature):
            values = np.log1p(np.clip(values, 0.0, None))
            log_features.append(feature)

        if standardize:
            train_values = values[:n_train]
            valid = np.isfinite(train_values)
            if valid.any():
                train_valid = train_values[valid]
                mean = float(train_valid.mean())
                std = float(train_valid.std(ddof=0))
                if not np.isfinite(std) or std < 1e-6:
                    std = 1.0
            else:
                mean, std = 0.0, 1.0
            values = (values - mean) / std
        else:
            mean, std = 0.0, 1.0

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        tensor[:, :, k] = values.astype(np.float32)
        means[k] = mean
        stds[k] = std

    meta = {
        "shape": [int(n_t), int(n_z), int(n_f)],
        "timestamps": [str(t) for t in timestamps],
        "zones": [int(z) for z in zones],
        "features": list(features),
        "log_features": log_features,
        "standardized": bool(standardize),
        "train_ratio": float(train_ratio),
        "mean": means.tolist(),
        "std": stds.tolist(),
        "layout": "X[timestamp, zone, feature]",
        "note": "DataLoader 组 batch 后形状为 [batch, seq_len, num_nodes, num_features]",
    }
    return tensor, meta


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    fcfg = cfg.get("features", {})
    dcfg = cfg.get("data", {})
    scfg = cfg.get("split", {})
    panel_path = resolve(args.input or cfg["paths"]["panel"])
    out_path = ensure_parent(args.out or cfg["paths"]["features"])

    if not panel_path.exists():
        raise FileNotFoundError(f"需求面板不存在: {panel_path}，请先运行 make_dataset.py")

    if args.input and not args.tensor_out:
        log.warning("检测到 --input；建议同时指定 --tensor-out，避免覆盖主面板的 feature_tensor.npy")
    if args.input and not args.feature_list_out:
        log.warning("检测到 --input；建议同时指定 --feature-list-out，避免覆盖主特征清单")

    log.info("读取面板 %s", panel_path)
    df = pd.read_parquet(panel_path)
    required = {"timestamp", "zone_id", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"面板缺少列: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["zone_id"] = df["zone_id"].astype("int16")
    df["demand"] = df["demand"].astype("int32")
    log.info(
        "面板 %d 行，%d 区域，%s ~ %s",
        len(df), df["zone_id"].nunique(), df["timestamp"].min(), df["timestamp"].max(),
    )

    lags = [int(x) for x in fcfg.get("lags", [1, 2, 4, 8, 96, 672])]
    windows = [int(x) for x in fcfg.get("rolling_windows", [4, 96])]

    df = add_history_features(df, lags, windows)
    df = add_calendar(df, fcfg)
    df["time_idx"] = (
        (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 900
    ).astype("int32")

    feature_cols = (
        ["zone_id"]
        + [f"lag_{x}" for x in lags]
        + [f"roll_mean_{x}" for x in windows]
        + ["hour", "minute", "dow", "is_weekend", "is_holiday"]
    )
    if fcfg.get("cyclic", True):
        feature_cols += ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    df.to_parquet(out_path, compression="zstd", index=False)
    log.info("已写入 %s（%.1f MB，%d 行）", out_path, out_path.stat().st_size / 1e6, len(df))

    list_path = ensure_parent(args.feature_list_out or cfg["paths"]["feature_list"])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel": str(panel_path),
        "lags": lags,
        "rolling_windows": windows,
        "features": feature_cols,
        "target": "demand",
        "keys": ["timestamp", "zone_id"],
        "n_rows": int(len(df)),
    }

    # ---- STGCN tensor: [T, N, F] ----
    if not args.no_tensor:
        model_features = [c for c in feature_cols if c not in {"zone_id", "demand"}]
        zones = np.arange(
            int(dcfg.get("zone_min", 1)), int(dcfg.get("zone_max", 263)) + 1, dtype=np.int16
        )
        tensor, tensor_meta = build_feature_tensor(
            df,
            features=model_features,
            zones=zones,
            train_ratio=float(scfg.get("train", 0.70)),
            standardize=not args.no_standardize,
        )

        tensor_path = ensure_parent(
            args.tensor_out or cfg["paths"].get("feature_tensor", "data/processed/feature_tensor.npy")
        )
        if tensor_path.suffix.lower() != ".npy":
            tensor_path = tensor_path.with_suffix(".npy")
        np.save(tensor_path, tensor, allow_pickle=False)

        seq_len = int(args.seq_len or cfg.get("stgnn", {}).get("seq_len", 12))
        pred_len = int(args.pred_len or cfg.get("stgnn", {}).get("pred_len", 1))
        tensor_meta.update(
            {
                "path": str(tensor_path),
                "panel": str(panel_path),
                "seq_len": seq_len,
                "pred_len": pred_len,
                "expected_batch_shape": ["batch", seq_len, len(zones), len(model_features)],
            }
        )
        meta_path = (
            ensure_parent(args.tensor_meta_out)
            if args.tensor_meta_out
            else tensor_path.with_suffix(".meta.json")
        )
        meta_path.write_text(
            json.dumps(tensor_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        payload.update(
            {
                "tensor_features": model_features,
                "tensor_shape": [int(x) for x in tensor.shape],
                "tensor_path": str(tensor_path),
                "tensor_meta": str(meta_path),
            }
        )
        log.info(
            "STGCN 张量 -> %s, shape=%s, dtype=%s",
            tensor_path, tensor.shape, tensor.dtype,
        )
        log.info(
            "DataLoader batch 形状: [batch, %d, %d, %d]",
            seq_len, len(zones), len(model_features),
        )
        if tensor_meta["standardized"]:
            log.info("张量标准化：仅用前 %.0f%% 训练期时间片估计均值/方差", 100 * float(scfg.get("train", 0.70)))

    list_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("特征清单 -> %s（%d 个特征）", list_path, len(feature_cols))

    nan_ratio = df[feature_cols].isna().mean().sort_values(ascending=False).head(5)
    log.info("NaN 比例 Top5:\n%s", nan_ratio.to_string())


if __name__ == "__main__":
    main()