"""在需求面板上构造滞后、滚动与日历特征（严格防泄漏）。

用法:
    python -m src.features.build_features --config configs/default.yaml

防泄漏要点:
    - lag_k 使用同区域 t-k 的真实需求；
    - 滚动均值必须先 shift(1)，不能把当前时刻 demand 算进去；
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
    p = argparse.ArgumentParser(description="Build leakage-safe features.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--input", default=None)
    p.add_argument("--out", default=None)
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    fcfg = cfg.get("features", {})
    panel_path = resolve(args.input or cfg["paths"]["panel"])
    out_path = ensure_parent(args.out or cfg["paths"]["features"])

    if not panel_path.exists():
        raise FileNotFoundError(f"需求面板不存在: {panel_path}，请先运行 make_dataset.py")

    log.info("读取面板 %s", panel_path)
    df = pd.read_parquet(panel_path)
    required = {"timestamp", "zone_id", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"面板缺少列: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["zone_id"] = df["zone_id"].astype("int16")
    df["demand"] = df["demand"].astype("int32")
    log.info("面板 %d 行，%d 区域，%s ~ %s", len(df), df["zone_id"].nunique(),
             df["timestamp"].min(), df["timestamp"].max())

    lags = [int(x) for x in fcfg.get("lags", [1, 2, 4, 8, 96, 672])]
    windows = [int(x) for x in fcfg.get("rolling_windows", [4, 96])]

    df = add_history_features(df, lags, windows)
    df = add_calendar(df, fcfg)
    df["time_idx"] = ((df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 900).astype("int32")

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

    list_path = ensure_parent(cfg["paths"]["feature_list"])
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
    list_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("特征清单 -> %s（%d 个特征）", list_path, len(feature_cols))

    nan_ratio = df[feature_cols].isna().mean().sort_values(ascending=False).head(5)
    log.info("NaN 比例 Top5:\n%s", nan_ratio.to_string())


if __name__ == "__main__":
    main()