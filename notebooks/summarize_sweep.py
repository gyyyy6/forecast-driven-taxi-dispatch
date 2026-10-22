"""汇总 sweep* 目录下的策略对比结果。

用法:
    python notebooks\summarize_sweep.py
    python notebooks\summarize_sweep.py --pattern "sweep2_*"
    python notebooks\summarize_sweep.py --pattern "sweep*"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/simulation")
    ap.add_argument("--pattern", default="sweep*", help="匹配结果目录名")
    ap.add_argument("--out", default="sweep_summary.parquet")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob(f"{args.pattern}/sim_results.parquet"))
    if not files:
        raise SystemExit(f"没有找到 {root}\\{args.pattern}\\sim_results.parquet")

    frames = []
    for p in files:
        df = pd.read_parquet(p)
        df["out_dir"] = p.parent.name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["policy"] = all_df["policy"].astype(str)

    print("=" * 110)
    print("明细")
    print("=" * 110)
    cols = ["out_dir", "n_vehicles", "policy", "total_requests",
            "served_rate", "avg_wait_min", "p90_wait_min", "deadhead_km"]
    print(all_df[cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    def pivot(value, scale=1.0, digits=2):
        t = all_df.pivot_table(
            index=["out_dir", "n_vehicles"], columns="policy",
            values=value, aggfunc="first", observed=True,
        ) * scale
        if {"greedy", "forecast"} <= set(t.columns):
            t["diff"] = t["forecast"] - t["greedy"]
        return t.round(digits)

    print("\n服务率（%）")
    print(pivot("served_rate", 100).to_string())
    print("\n平均等待（min）")
    print(pivot("avg_wait_min").to_string())
    print("\nP90 等待（min）")
    print(pivot("p90_wait_min").to_string())

    dh = pivot("deadhead_km", 1.0, 1)
    if "diff" in dh.columns:
        dh["diff_pct"] = (dh["diff"] / dh["greedy"] * 100).round(2)
    print("\n空驶（km）")
    print(dh.to_string())

    out = root / args.out
    all_df.to_parquet(out, compression="zstd", index=False)
    print("\n已写出", out)


if __name__ == "__main__":
    main()