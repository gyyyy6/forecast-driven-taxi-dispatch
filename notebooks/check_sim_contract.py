"""仿真输出契约自检。用法: python notebooks\check_sim_contract.py [结果目录]"""
import sys
from pathlib import Path

import pandas as pd

SIM = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/simulation")
SERVICE_MIN = 12.0            # configs/default.yaml: simulation.service_time_min

REQ_COLS = ["request_id", "request_time", "zone_id", "assigned_vehicle",
            "pickup_time", "wait_min", "served", "policy"]
SUM_COLS = ["policy", "served_rate", "avg_wait_min", "p90_wait_min",
            "deadhead_km", "total_requests"]

print("结果目录:", SIM.resolve())
req = pd.read_parquet(SIM / "sim_requests.parquet")
res = pd.read_parquet(SIM / "sim_results.parquet")

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if extra else ""))
    ok = ok and bool(cond)

miss = [c for c in REQ_COLS if c not in req.columns]
check("sim_requests 列齐全", not miss, f"缺 {miss}")

if not miss:
    s1 = req[req.served == 1]
    s0 = req[req.served == 0]
    print(f"总请求 {len(req):,} | 已服务 {len(s1):,} | 未服务 {len(s0):,}")
    check("request_id 唯一", req.request_id.is_unique)
    check("served ∈ {0,1}", set(req.served.unique()) <= {0, 1}, str(sorted(req.served.unique())))
    check("zone_id ∈ 1..263", req.zone_id.between(1, 263).all())
    check("request_time 单调不减", req.request_time.is_monotonic_increasing)
    check("已服务均有 pickup_time", s1.pickup_time.notna().all())
    check("未服务 assigned_vehicle 全为 -1", (s0.assigned_vehicle == -1).all())

    if len(s1):
        err = (s1.wait_min - (s1.pickup_time - s1.request_time).dt.total_seconds() / 60).abs().max()
        check("wait_min == pickup - request", err < 1e-2, f"max_err={err:.4f}")
        check("wait_min >= 1（同区保底 1 分钟）", (s1.wait_min >= 1 - 1e-6).all(),
              f"min={s1.wait_min.min():.3f}")
        check("wait_min <= max_wait(30)", (s1.wait_min <= 30.0 + 1e-6).all(),
              f"max={s1.wait_min.max():.3f}")
        v = s1.sort_values(["assigned_vehicle", "pickup_time"])
        gap = v.groupby("assigned_vehicle")["pickup_time"].diff().dt.total_seconds() / 60
        clash = int((gap < SERVICE_MIN - 1e-6).sum())
        check("同一辆车不同时接两单", clash == 0, f"{clash} 处冲突 / {len(v)} 条")

miss2 = [c for c in SUM_COLS if c not in res.columns]
check("sim_results 列齐全", not miss2, f"缺 {miss2}")
if not miss2:
    for _, r in res.iterrows():
        sub = req[req.policy.astype(str) == str(r.policy)]
        check(f"[{r.policy}] total_requests 与明细行数一致",
              int(r.total_requests) == len(sub), f"{int(r.total_requests)} vs {len(sub)}")
        if len(sub):
            check(f"[{r.policy}] served_rate 一致",
                  abs(float(r.served_rate) - float(sub.served.mean())) < 1e-6)
    print("\nsim_results:")
    print(res.to_string(index=False))

groups = [g.reset_index(drop=True) for _, g in req.sort_values(["policy", "request_time"])
          .groupby(req["policy"].astype(str))]
if len(groups) == 2:
    check("两策略共享同一请求流",
          groups[0][["request_time", "zone_id"]].equals(groups[1][["request_time", "zone_id"]]),
          f"{len(groups[0])} vs {len(groups[1])}")
else:
    print(f"（只有 {len(groups)} 个策略，跳过请求流一致性检查——这是单策略冒烟，正常）")

s1 = req[req.served == 1]
if len(s1):
    print("\n等待时间分布（已服务）:")
    print(s1.wait_min.describe(percentiles=[0.5, 0.9]).to_string())
    print("\n每车完成单量:")
    print(s1.groupby("assigned_vehicle").size().describe().to_string())

print("\n结论:", "全部 PASS" if ok else "有 FAIL，先修再往下")