"""从 sample_day.parquet 生成仿真用预测文件（仅本地开发，不入库）。

用法:
    python -m src.simulation.make_fixture --config configs/default.yaml
    python -m src.simulation.make_fixture --scale 0.02 \
        --out data/processed/simulation/fixture_pred_day_s002.parquet
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import ensure_dir, load_config, make_logger, resolve

log = make_logger("make_fixture")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a prediction fixture from sample_day.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--src", default="data/processed/sample_day.parquet")
    ap.add_argument("--out", default="data/processed/simulation/fixture_pred_day.parquet")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="需求抽稀比例（泊松 thinning），1.0 表示不缩放")
    args = ap.parse_args()

    if args.scale <= 0:
        raise ValueError("--scale 必须 > 0")

    load_config(args.config)          # 顺带校验 YAML 可读
    src = resolve(args.src)
    if not src.exists():
        raise FileNotFoundError(f"缺少样本 {src}，请先解压 A 的 release")

    df = pd.read_parquet(src, columns=["timestamp", "zone_id", "demand"])
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["timestamp"]),
            "zone_id": pd.array(df["zone_id"], dtype="int16"),
            "predicted_demand": (df["demand"].astype("float64") * args.scale)
            .clip(lower=0).astype("float32"),
            "model_name": f"fixture_sample_day_scale{args.scale:g}",
        }
    ).sort_values(["timestamp", "zone_id"], ignore_index=True)

    if len(out) != 96 * 263:
        log.warning("行数 %d != 96x263，仿真会缺区域（缺的补 0）", len(out))
    if out["predicted_demand"].isna().any():
        raise ValueError("predicted_demand 出现 NaN")

    path = resolve(args.out)
    ensure_dir(path.parent)
    out.to_parquet(path, compression="zstd", index=False)

    per_slice = out.groupby("timestamp")["predicted_demand"].sum()
    log.info("fixture -> %s", path)
    log.info("行数 %d | 时间片 %d | 区域 %d | 需求合计 %.0f",
             len(out), out["timestamp"].nunique(),
             out["zone_id"].nunique(), out["predicted_demand"].sum())
    log.info("每片需求: 均值 %.0f | 峰值 %.0f（%s）| 非零区域占比 %.1f%%",
             per_slice.mean(), per_slice.max(), per_slice.idxmax(),
             (out["predicted_demand"] > 0).mean() * 100)
    print(out.dtypes.to_string())
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()