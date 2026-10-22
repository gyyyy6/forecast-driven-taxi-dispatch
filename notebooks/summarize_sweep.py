"""汇总 sweep_* 目录下的策略对比结果。"""
from pathlib import Path

import pandas as pd

root = Path("data/processed/simulation")
files = sorted(root.glob("sweep_*/sim_results.parquet"))
if not files:
    raise SystemExit("没有找到 sweep_*/sim_results.parquet")

rows = []
for p in files:
    df = pd.read_parquet(p)
    df["out_dir"] = p.parent.name
    rows.append(df)
all_df = pd.concat(rows, ignore_index=True)

cols = ["out_dir", "n_vehicles", "policy", "total_requests",
        "served_rate", "avg_wait_min", "p90_wait_min", "deadhead_km"]
print("=" * 100)
print(all_df[cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

print("\n" + "=" * 100)
print("服务率对比（%）")
piv = all_df.pivot_table(index="n_vehicles", columns="policy",
                         values="served_rate", aggfunc="first") * 100
if {"greedy", "forecast"} <= set(piv.columns):
    piv["diff_pp"] = piv["forecast"] - piv["greedy"]
print(piv.to_string(float_format=lambda x: f"{x:.2f}"))

print("\n平均等待对比（min）")
piv2 = all_df.pivot_table(index="n_vehicles", columns="policy",
                          values="avg_wait_min", aggfunc="first")
if {"greedy", "forecast"} <= set(piv2.columns):
    piv2["diff_min"] = piv2["forecast"] - piv2["greedy"]
print(piv2.to_string(float_format=lambda x: f"{x:.2f}"))

print("\n空驶里程对比（km）")
piv3 = all_df.pivot_table(index="n_vehicles", columns="policy",
                          values="deadhead_km", aggfunc="first")
if {"greedy", "forecast"} <= set(piv3.columns):
    piv3["diff_pct"] = (piv3["forecast"] / piv3["greedy"] - 1) * 100
print(piv3.to_string(float_format=lambda x: f"{x:,.1f}"))

all_df.to_parquet(root / "sweep_summary.parquet", compression="zstd", index=False)
print("\n已写出", root / "sweep_summary.parquet")