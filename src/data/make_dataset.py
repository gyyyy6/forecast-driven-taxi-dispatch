"""HVFHV 原始 parquet → 15 分钟 × Taxi Zone 需求面板。

用法:
    python -m src.data.make_dataset --config configs/default.yaml
    python -m src.data.make_dataset --raw-dir data/raw \
        --start 2023-01-01 --end 2023-02-01 \
        --out data/processed/demand_panel.parquet

约定:
    - 时间区间为 [start, end)，时间戳为纽约本地 naive 时间；
    - demand = 每个时间片、每个 PULocationID 的上车订单数；
    - 输出为完整面板：所有时间片 × 1..263 区域，缺失补 0。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import ensure_parent, load_config, make_logger, resolve

log = make_logger("make_dataset")

PICKUP_COL = "pickup_datetime"
ZONE_COL = "PULocationID"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 15-min demand panel from HVFHV parquet.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--raw-glob", default=None, help="默认读取 paths.raw_glob")
    p.add_argument("--out", default=None)
    p.add_argument("--start", default=None, help="闭区间起点，如 2024-01-01")
    p.add_argument("--end", default=None, help="开区间终点，如 2025-01-01")
    p.add_argument("--freq", default=None, help="默认 15min")
    p.add_argument("--zone-min", type=int, default=None)
    p.add_argument("--zone-max", type=int, default=None)
    return p.parse_args()


def aggregate_file(path: Path, start: pd.Timestamp | None, end: pd.Timestamp | None, freq: str) -> pd.DataFrame:
    """单文件聚合。TLC 按月提供文件，逐月读取可把内存压在几百 MB。"""
    df = pd.read_parquet(path, columns=[PICKUP_COL, ZONE_COL])
    df = df.rename(columns={ZONE_COL: "zone_id"})
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    if df["pickup_datetime"].dt.tz is not None:
        df["pickup_datetime"] = df["pickup_datetime"].dt.tz_localize(None)
    df = df.dropna(subset=["pickup_datetime", "zone_id"])

    if start is not None:
        df = df[df["pickup_datetime"] >= start]
    if end is not None:
        df = df[df["pickup_datetime"] < end]
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "zone_id", "demand"])

    df["zone_id"] = df["zone_id"].astype("int16")
    df["timestamp"] = df["pickup_datetime"].dt.floor(freq)
    return (
        df.groupby(["timestamp", "zone_id"], observed=True)
        .size()
        .rename("demand")
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dcfg = cfg.get("data", {})

    raw_dir = resolve(args.raw_dir or cfg["paths"]["raw_dir"])
    raw_glob = args.raw_glob or cfg["paths"].get("raw_glob", "*.parquet")
    out_path = ensure_parent(args.out or cfg["paths"]["panel"])
    freq = args.freq or dcfg.get("freq", "15min")
    zone_min = args.zone_min or dcfg.get("zone_min", 1)
    zone_max = args.zone_max or dcfg.get("zone_max", 263)
    start = pd.Timestamp(args.start) if args.start else (pd.Timestamp(dcfg["start"]) if dcfg.get("start") else None)
    end = pd.Timestamp(args.end) if args.end else (pd.Timestamp(dcfg["end"]) if dcfg.get("end") else None)

    files = sorted(raw_dir.glob(raw_glob))
    if not files:
        raise FileNotFoundError(
            f"{raw_dir} 下没有匹配 {raw_glob} 的文件。请先从 NYC TLC 下载 HVFHV 数据。"
        )
    log.info("发现 %d 个原始文件（%s），时间范围 [%s, %s)", len(files), raw_glob, start, end)

    parts: list[pd.DataFrame] = []
    for i, f in enumerate(files, 1):
        part = aggregate_file(f, start, end, freq)
        log.info("  [%d/%d] %s -> %d 行", i, len(files), f.name, len(part))
        if not part.empty:
            parts.append(part)

    if not parts:
        raise ValueError("过滤后没有数据，请检查 start/end 与文件时间范围。")

    agg = pd.concat(parts, ignore_index=True)
    del parts
    agg = (
        agg.groupby(["timestamp", "zone_id"], observed=True, as_index=False)["demand"]
        .sum()
    )

    data_start = start or agg["timestamp"].min()
    data_end = end or (agg["timestamp"].max() + pd.Timedelta(freq))
    if end is not None:
        data_end = end

    zones = range(int(zone_min), int(zone_max) + 1)
    full_index = pd.MultiIndex.from_product(
        [pd.date_range(data_start, data_end, freq=freq, inclusive="left"), zones],
        names=["timestamp", "zone_id"],
    )
    log.info("补齐完整面板: %d 个时间片 × %d 个区域 = %d 行",
             len(full_index) // len(list(zones)), len(list(zones)), len(full_index))

    panel = (
        agg.set_index(["timestamp", "zone_id"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    panel["zone_id"] = panel["zone_id"].astype("int16")
    panel["demand"] = panel["demand"].astype("int32")
    panel = panel.sort_values(["timestamp", "zone_id"], ignore_index=True)

    panel.to_parquet(out_path, compression="zstd", index=False)
    log.info("已写入 %s（%.1f MB）", out_path, out_path.stat().st_size / 1e6)

    meta = {
        "source_files": [f.name for f in files],
        "rows": int(len(panel)),
        "zones": int(panel["zone_id"].nunique()),
        "timestamps": int(panel["timestamp"].nunique()),
        "start": str(panel["timestamp"].min()),
        "end": str(panel["timestamp"].max()),
        "freq": freq,
        "total_demand": int(panel["demand"].sum()),
        "zero_ratio": float((panel["demand"] == 0).mean()),
        "demand_mean": float(panel["demand"].mean()),
        "demand_p99": float(panel["demand"].quantile(0.99)),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("面板元信息: %s", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()