"""对比两个场景的分区域服务率：检验收紧匹配半径的公平性代价。"""
from pathlib import Path

import pandas as pd

SIM = Path("data/processed/simulation/scenarios")
DEMAND = Path("data/processed/simulation/fixture_pred_day_s002.parquet")
PAIRS = [("s_g_n200_w30", "s_g_n200_w15")]

demand = pd.read_parquet(DEMAND)
need = demand.groupby("zone_id")["predicted_demand"].sum()
top25 = set(need.nlargest(25).index)

for a, b in PAIRS:
    print("=" * 70)
    for name in (a, b):
        req = pd.read_parquet(SIM / f"scenario_{name}_requests.parquet")
        g = req.groupby("zone_id")["served"].mean() * 100
        inside = g[g.index.isin(top25)].mean()
        outside = g[~g.index.isin(top25)].mean()
        print(f"{name:22s} 总体 {req.served.mean()*100:6.2f}% | "
              f"Top25 区 {inside:6.2f}% | 其他区 {outside:6.2f}%")