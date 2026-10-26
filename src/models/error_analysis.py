"""成员 C / C5：最佳模型的分时段 / 分区误差分析。

回答任务书的三个问题：
    - 早/晚高峰是否被低估（按小时、按真实需求档位看偏差 bias = pred - true）；
    - 零需求区域是否被过度预测（零需求行的预测均值与 bias）；
    - 哪些区域误差最大，是否与机场 / 曼哈顿核心区相关（结合 TLC zone lookup 的 Borough/Zone）。

用法:
    python -m src.models.error_analysis --config configs/experiments/lgbm_best.yaml
    python -m src.models.error_analysis --config configs/experiments/lgbm_best.yaml \
        --pred data/processed/predictions/pred_test_lgbm_best.parquet

产出:
    data/processed/experiments/error_by_hour.parquet     segment(all/weekday/weekend) × hour(0-23)
    data/processed/experiments/error_by_zone.parquet     263 个区域（含 borough / zone 名 / 需求分位）
    data/processed/experiments/error_by_segment.parquet  需求档位 / 零需求 / 区域需求三分位 / borough
    data/processed/experiments/error_analysis.manifest.json

指标口径:
    mae / rmse / wape = 常规误差；bias = mean(pred - true)（负值=低估）；
    under_share = 预测值小于真实值的行占比；pred_over_true = 预测均值 / 真实均值。
    真实需求全为 0 的分组 WAPE 无定义，记为 NaN（用 bias / mae 解读）。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import ensure_dir, ensure_parent, load_config, make_logger, resolve
from src.models.common import regression_metrics, save_json, time_split_bounds

log = make_logger("error_analysis")

METRIC_COLUMNS = [
    "n",
    "mae",
    "rmse",
    "wape",
    "bias",
    "pred_mean",
    "true_mean",
    "pred_over_true",
    "under_share",
]
DEMAND_BINS = [-0.1, 0.0, 5.0, 20.0, 50.0, np.inf]
DEMAND_LABELS = ["0", "1-5", "6-20", "21-50", ">50"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hourly / zonal error analysis for the tuned LightGBM model.")
    p.add_argument("--config", default="configs/experiments/lgbm_best.yaml")
    p.add_argument("--pred", default=None, help="预测文件，默认 prediction_dir/pred_test_lgbm_best.parquet")
    p.add_argument("--out-dir", default=None, help="默认取 config 的 out_dir")
    p.add_argument("--zone-lookup", default=None, help="默认取 config 的 paths.zone_lookup")
    p.add_argument("--top", type=int, default=20, help="Top N 高需求区域")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def seg_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    n = int(len(true))
    if n == 0:
        return {c: np.nan for c in METRIC_COLUMNS} | {"n": 0}
    err = pred - true
    abs_err = np.abs(err)
    denom = float(np.abs(true).sum())
    true_mean = float(true.mean())
    return {
        "n": n,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(abs_err.sum() / denom) if denom > 0 else float("nan"),
        "bias": float(err.mean()),
        "pred_mean": float(pred.mean()),
        "true_mean": true_mean,
        "pred_over_true": float(pred.mean() / true_mean) if true_mean > 0 else float("nan"),
        "under_share": float((pred < true).mean()),
    }


def load_lookup(path) -> pd.DataFrame:
    if path is None or not path.exists():
        log.warning("zone lookup 不存在（%s），区域名称将缺失", path)
        return pd.DataFrame(columns=["zone_id", "borough", "zone"])
    df = pd.read_csv(path)
    df = df.rename(columns={"LocationID": "zone_id", "Borough": "borough", "Zone": "zone"})
    keep = [c for c in ("zone_id", "borough", "zone") if c in df.columns]
    df = df[keep].copy()
    df["zone_id"] = df["zone_id"].astype("int64")
    log.info("已加载 zone lookup：%d 个区域", len(df))
    return df


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def hourly_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    rows = []
    segments = {
        "all": df,
        "weekday": df[~df["is_weekend"]],
        "weekend": df[df["is_weekend"]],
    }
    for name, sub in segments.items():
        for hour in range(24):
            part = sub[sub["hour"] == hour]
            rows.append({"segment": name, "hour": hour, **seg_metrics(part["demand"], part["predicted_demand"])})
    return pd.DataFrame(rows)[["segment", "hour", *METRIC_COLUMNS]]


def zone_table(df: pd.DataFrame, lookup: pd.DataFrame, top: int) -> pd.DataFrame:
    g = df.groupby("zone_id", observed=True)
    table = pd.DataFrame(
        {
            "n": g.size().astype(int),
            "mae": g.apply(lambda s: seg_metrics(s["demand"], s["predicted_demand"])["mae"], include_groups=False),
            "rmse": g.apply(lambda s: seg_metrics(s["demand"], s["predicted_demand"])["rmse"], include_groups=False),
            "wape": g.apply(lambda s: seg_metrics(s["demand"], s["predicted_demand"])["wape"], include_groups=False),
            "bias": g.apply(lambda s: seg_metrics(s["demand"], s["predicted_demand"])["bias"], include_groups=False),
            "pred_mean": g["predicted_demand"].mean(),
            "true_mean": g["demand"].mean(),
            "true_sum": g["demand"].sum(),
            "under_share": g.apply(
                lambda s: float((s["predicted_demand"] < s["demand"]).mean()), include_groups=False
            ),
        }
    ).reset_index()
    table["pred_over_true"] = table["pred_mean"] / table["true_mean"].replace(0, np.nan)
    table["rank_by_demand"] = table["true_sum"].rank(ascending=False, method="first").astype(int)
    table["is_top20"] = table["rank_by_demand"] <= top
    # 区域需求三分位：按均值排名切三段，避免并列均值导致 qcut 失败
    rank = table["true_mean"].rank(method="first")
    table["demand_quantile"] = pd.qcut(rank, 3, labels=["low", "mid", "high"]).astype(str)
    table = table.merge(lookup, on="zone_id", how="left")
    table["borough"] = table["borough"].fillna("unknown")
    table["zone"] = table["zone"].fillna("")
    cols = [
        "zone_id", "borough", "zone", "demand_quantile", "rank_by_demand", "is_top20",
        *METRIC_COLUMNS, "true_sum",
    ]
    return table[cols].sort_values("rank_by_demand", ignore_index=True)


def segment_tables(df: pd.DataFrame, zones: pd.DataFrame, top: int) -> pd.DataFrame:
    df = df.copy()
    df["demand_level"] = pd.cut(df["demand"], bins=DEMAND_BINS, labels=DEMAND_LABELS, right=True)
    df["zero_flag"] = np.where(df["demand"] == 0, "zero", "nonzero")
    df["day_type"] = np.where(df["timestamp"].dt.dayofweek >= 5, "weekend", "weekday")
    zone_meta = zones[["zone_id", "demand_quantile", "borough", "is_top20"]]
    df = df.merge(zone_meta, on="zone_id", how="left")

    rows = []

    def add(kind: str, name: str, part: pd.DataFrame) -> None:
        rows.append({"segment_type": kind, "segment": name, **seg_metrics(part["demand"], part["predicted_demand"])})

    for level in DEMAND_LABELS:
        add("demand_level", level, df[df["demand_level"].astype(str) == level])
    for flag in ("zero", "nonzero"):
        add("zero_flag", flag, df[df["zero_flag"] == flag])
    for q in ("low", "mid", "high"):
        add("demand_tertile", q, df[df["demand_quantile"] == q])
    for dt in ("weekday", "weekend"):
        add("day_type", dt, df[df["day_type"] == dt])
    for b, part in df.groupby("borough", observed=True):
        add("borough", str(b), part)
    add("top20", f"top{top}", df[df["is_top20"] == True])  # noqa: E712
    add("top20", f"rest", df[df["is_top20"] != True])  # noqa: E712
    add("overall", "all", df)
    return pd.DataFrame(rows)[["segment_type", "segment", *METRIC_COLUMNS]]


def markdown(table: pd.DataFrame, columns: list[str], floatfmt: dict[str, str] | None = None) -> str:
    fmt = floatfmt or {}
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [head, sep]
    for _, r in table.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            if c in fmt:
                cells.append("—" if pd.isna(v) else fmt[c].format(v))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    features_path = resolve(cfg["paths"]["features"])
    pred_path = resolve(args.pred) if args.pred else resolve(cfg["paths"]["prediction_dir"]) / "pred_test_lgbm_best.parquet"
    out_dir = ensure_dir(args.out_dir or cfg["paths"].get("out_dir", "data/processed/experiments"))
    lookup_path = resolve(args.zone_lookup or cfg["paths"].get("zone_lookup", "data/raw/taxi_zone_lookup.csv"))
    if not features_path.exists():
        raise FileNotFoundError(f"特征表不存在: {features_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")

    scfg = cfg.get("split") or {}
    log.info("读取特征表 %s", features_path)
    feats = pd.read_parquet(features_path, columns=["timestamp", "zone_id", "demand"])
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    train_end, val_end = time_split_bounds(feats["timestamp"], scfg.get("train", 0.70), scfg.get("val", 0.15))
    test = feats[feats["timestamp"] >= val_end].copy()

    log.info("读取预测 %s", pred_path)
    pred = pd.read_parquet(pred_path)
    missing = {"timestamp", "zone_id", "predicted_demand"} - set(pred.columns)
    if missing:
        raise ValueError(f"预测文件缺少列 {sorted(missing)}")
    pred = pred[["timestamp", "zone_id", "predicted_demand"]].copy()
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])

    df = test.merge(pred, on=["timestamp", "zone_id"], how="left", validate="one_to_one")
    if df["predicted_demand"].isna().any():
        raise ValueError(f"{int(df['predicted_demand'].isna().sum())} 行预测缺失，覆盖不完整")
    if len(df) != len(test) or df["timestamp"].nunique() != test["timestamp"].nunique():
        raise ValueError(f"合并后行数 {len(df)} != 测试集 {len(test)}")
    df["predicted_demand"] = df["predicted_demand"].astype("float64")
    df["demand"] = df["demand"].astype("float64")
    log.info("测试集 %d 行，%d 片 × %d 区域（%s ~ %s）", len(df), df["timestamp"].nunique(),
             df["zone_id"].nunique(), df["timestamp"].min(), df["timestamp"].max())

    lookup = load_lookup(lookup_path)
    overall = seg_metrics(df["demand"], df["predicted_demand"])
    overall_full = regression_metrics(df["demand"].to_numpy(), df["predicted_demand"].to_numpy())

    by_hour = hourly_table(df)
    by_zone = zone_table(df, lookup, args.top)
    by_segment = segment_tables(df, by_zone, args.top)

    hour_path = ensure_parent(out_dir / "error_by_hour.parquet")
    zone_path = ensure_parent(out_dir / "error_by_zone.parquet")
    seg_path = ensure_parent(out_dir / "error_by_segment.parquet")
    by_hour.to_parquet(hour_path, compression="zstd", index=False)
    by_zone.to_parquet(zone_path, compression="zstd", index=False)
    by_segment.to_parquet(seg_path, compression="zstd", index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "inputs": {"features": str(features_path), "prediction": str(pred_path), "zone_lookup": str(lookup_path)},
        "split": {"train_end": str(train_end), "val_end": str(val_end)},
        "test": {
            "rows": int(len(df)),
            "timestamps": int(df["timestamp"].nunique()),
            "zones": int(df["zone_id"].nunique()),
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
        "overall": {**overall, **{k: overall_full[k] for k in ("smape",)}},
        "outputs": {"by_hour": str(hour_path), "by_zone": str(zone_path), "by_segment": str(seg_path)},
        "bias_definition": "mean(pred - true); negative = under-forecast",
    }
    manifest_path = out_dir / "error_analysis.manifest.json"
    save_json(manifest, manifest_path)

    # ---------- 打印便于写进报告的表格 ----------
    log.info("整体：MAE=%.4f RMSE=%.4f WAPE=%.2f%% bias=%.4f under_share=%.2f%%",
             overall["mae"], overall["rmse"], overall["wape"] * 100, overall["bias"], overall["under_share"] * 100)
    print("\n### 整体")
    print(markdown(pd.DataFrame([{"scope": "test", **overall}]), ["scope", *METRIC_COLUMNS],
                   {"mae": "{:.4f}", "rmse": "{:.4f}", "wape": "{:.2%}", "bias": "{:+.4f}",
                    "pred_over_true": "{:.4f}", "under_share": "{:.2%}"}))

    print("\n### 按小时（all）")
    print(markdown(by_hour[by_hour.segment == "all"],
                   ["hour", "n", "mae", "wape", "bias", "pred_mean", "true_mean"],
                   {"wape": "{:.2%}", "bias": "{:+.4f}"}))

    print("\n### 工作日 vs 周末")
    print(markdown(by_segment[by_segment.segment_type == "day_type"],
                   ["segment", "n", "mae", "rmse", "wape", "bias", "pred_mean", "true_mean"],
                   {"wape": "{:.2%}", "bias": "{:+.4f}"}))

    print("\n### 需求档位")
    print(markdown(by_segment[by_segment.segment_type == "demand_level"],
                   ["segment", "n", "mae", "wape", "bias", "pred_over_true", "under_share"],
                   {"wape": "{:.2%}", "bias": "{:+.4f}", "pred_over_true": "{:.3f}", "under_share": "{:.1%}"}))

    print("\n### 零需求 vs 非零")
    print(markdown(by_segment[by_segment.segment_type == "zero_flag"],
                   ["segment", "n", "mae", "bias", "pred_mean", "true_mean", "pred_over_true"],
                   {"bias": "{:+.4f}", "pred_over_true": "{:.3f}"}))

    print("\n### 区域需求三分位")
    print(markdown(by_segment[by_segment.segment_type == "demand_tertile"],
                   ["segment", "n", "mae", "wape", "bias"], {"wape": "{:.2%}", "bias": "{:+.4f}"}))

    print("\n### Borough")
    print(markdown(by_segment[by_segment.segment_type == "borough"],
                   ["segment", "n", "mae", "wape", "bias"], {"wape": "{:.2%}", "bias": "{:+.4f}"}))

    print(f"\n### Top {args.top} 高需求区域")
    top = by_zone[by_zone.is_top20].sort_values("rank_by_demand").head(args.top)
    print(markdown(top, ["rank_by_demand", "zone_id", "borough", "zone", "true_mean", "mae", "wape", "bias"],
                   {"true_mean": "{:.2f}", "wape": "{:.2%}", "bias": "{:+.4f}"}))

    print("\n### 误差最大的 15 个区域（真实需求均值 >= 5 单/片，按 WAPE）")
    meaningful = by_zone[by_zone.true_mean >= 5]
    worst = meaningful.sort_values("wape", ascending=False).head(15)
    print(markdown(worst, ["zone_id", "borough", "zone", "true_mean", "mae", "wape", "bias"],
                   {"true_mean": "{:.2f}", "wape": "{:.2%}", "bias": "{:+.4f}"}))

    print("\n### 绝对误差最大的 15 个区域（按 MAE）")
    worst_abs = by_zone.sort_values("mae", ascending=False).head(15)
    print(markdown(worst_abs, ["zone_id", "borough", "zone", "true_mean", "mae", "wape", "bias"],
                   {"true_mean": "{:.2f}", "wape": "{:.2%}", "bias": "{:+.4f}"}))

    print("\n### 误差最小（相对）的 10 个高需求区域（Top20 内按 WAPE 升序）")
    best = by_zone[by_zone.is_top20].sort_values("wape").head(10)
    print(markdown(best, ["zone_id", "borough", "zone", "true_mean", "mae", "wape", "bias"],
                   {"true_mean": "{:.2f}", "wape": "{:.2%}", "bias": "{:+.4f}"}))

    print("\n### Top20 vs 其余（区域需求分桶）")
    print(markdown(by_segment[by_segment.segment_type == "top20"],
                   ["segment", "n", "mae", "wape", "bias"], {"wape": "{:.2%}", "bias": "{:+.4f}"}))

    log.info("输出 -> %s / %s / %s", hour_path.name, zone_path.name, seg_path.name)
    log.info("清单 -> %s", manifest_path)


if __name__ == "__main__":
    main()
