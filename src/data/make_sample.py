"""从全量需求面板中提取小样本，供成员 B/C 和快速实验使用。

用法:
    python -m src.data.make_sample --config configs/default.yaml
    python -m src.data.make_sample --start "2023-01-15" --out-dir data/processed

默认输出:
    data/processed/sample_day.parquet    前 1 天
    data/processed/sample_month.parquet  前 30 天

说明:
    - 默认读取 configs/default.yaml 的 paths.panel；
    - 只保留 timestamp / zone_id / demand 三列；
    - 时间窗口为 [start, start + N days)，按时间顺序截取，不做随机抽样。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ensure_dir, load_config, make_logger, resolve

log = make_logger("make_sample")

REQUIRED_COLUMNS = ["timestamp", "zone_id", "demand"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract day/month samples from the demand panel.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--input", default=None, help="默认读取 paths.panel")
    p.add_argument("--out-dir", default=None, help="默认 data/processed")
    p.add_argument("--start", default=None, help="样本起点，默认面板最早时间")
    p.add_argument("--day-days", type=int, default=1, help="day 样本天数，默认 1")
    p.add_argument("--month-days", type=int, default=30, help="month 样本天数，默认 30")
    return p.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"全量面板不存在: {path}，请先运行 make_dataset.py")
    try:
        df = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    except ImportError as exc:
        raise SystemExit(
            "读取 parquet 失败：缺少 pyarrow。请执行 `pip install pyarrow`。"
        ) from exc

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"面板缺少列: {missing}；实际列: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["timestamp", "zone_id"], ignore_index=True)
    return df


def save_sample(name: str, df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        raise ValueError(f"{name} 样本为空，请检查 start 是否超出面板时间范围")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)
    log.info(
        "%s -> %s | shape=%s | %s ~ %s | zones=%d",
        name,
        path,
        df.shape,
        df["timestamp"].min(),
        df["timestamp"].max(),
        df["zone_id"].nunique(),
    )
    print(
        f"[{name}] shape={df.shape}, "
        f"time={df['timestamp'].min()} ~ {df['timestamp'].max()}, "
        f"zones={df['zone_id'].nunique()}, file={path}"
    )


def main() -> None:
    args = parse_args()
    if args.day_days < 1 or args.month_days < 1:
        raise ValueError("day-days / month-days 必须 >= 1")
    if args.month_days < args.day_days:
        raise ValueError("month-days 不能小于 day-days")

    cfg = load_config(args.config)
    input_path = resolve(args.input or cfg["paths"]["panel"])
    out_dir = (
        ensure_dir(args.out_dir)
        if args.out_dir
        else ensure_dir(Path(cfg["paths"]["panel"]).parent)
    )

    log.info("读取全量面板 %s", input_path)
    df = load_panel(input_path)
    if df.empty:
        raise ValueError("全量面板为空")

    panel_start, panel_end = df["timestamp"].min(), df["timestamp"].max()
    start = pd.Timestamp(args.start) if args.start else panel_start
    if start < panel_start or start > panel_end:
        raise ValueError(
            f"start={start} 超出面板范围 [{panel_start}, {panel_end}]"
        )

    day_end = start + pd.Timedelta(days=args.day_days)
    month_end = start + pd.Timedelta(days=args.month_days)

    day = df[(df["timestamp"] >= start) & (df["timestamp"] < day_end)].copy()
    month = df[(df["timestamp"] >= start) & (df["timestamp"] < month_end)].copy()

    print(f"full panel shape={df.shape}, time={panel_start} ~ {panel_end}")
    save_sample("sample_day", day, out_dir / "sample_day.parquet")
    save_sample("sample_month", month, out_dir / "sample_month.parquet")


if __name__ == "__main__":
    main()