"""成员 B / B5：场景矩阵批量仿真。

产出:
    data/processed/simulation/scenarios/
        scenario_<id>_requests.parquet
        scenario_<id>_summary.parquet
        scenarios_summary.parquet
        scenarios_manifest.json

用法:
    python -m src.simulation.scenarios --config configs/default.yaml \
        --pred data/processed/simulation/fixture_pred_day_s002.parquet \
        --start "2023-01-01 00:00" --end "2023-01-02 00:00"
    python -m src.simulation.scenarios --only s_g_n200_w30     # 只跑一个, 先验证
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ensure_dir, load_config, make_logger, resolve
from src.simulation.env import (
    DispatchSimulation,
    generate_requests,
    load_demand,
    load_geo,
)
from src.simulation.policy import make_policy

log = make_logger("scenarios")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    policy: str
    n_vehicles: int
    max_wait_min: float
    max_move_ratio: float | None


def build_scenarios() -> list[Scenario]:
    """B5 场景矩阵：7 组 greedy + 6 组 forecast = 13 组。"""
    out: list[Scenario] = []
    # Greedy 基线：3 车辆数 × 2 等待阈值
    for n in (100, 200, 300):
        for w in (30, 15):
            out.append(Scenario(f"s_g_n{n}_w{w}", "greedy", n, w, None))
    out.append(Scenario("s_g_n150_w30", "greedy", 150, 30, None))
    # Forecast 比例扫描：n=200, w=30
    for r in (0.10, 0.25, 0.50):
        tag = f"r{int(round(r * 100)):03d}"
        out.append(Scenario(f"s_f_n200_w30_{tag}", "forecast", 200, 30, r))
    # Forecast 补充：ratio=0.10（目前负作用最小）下换车辆数
    for n in (100, 150, 300):
        out.append(Scenario(f"s_f_n{n}_w30_r010", "forecast", n, 30, 0.10))
    # Forecast 在收紧等待阈值下（w=15）
    for n in (100, 200):
        out.append(Scenario(f"s_f_n{n}_w15_r010", "forecast", n, 15, 0.10))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run B5 scenario matrix.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--pred", default="data/processed/simulation/fixture_pred_day_s002.parquet")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--only", default=None, help="只跑 scenario_id 含该子串的场景")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_all = load_config(args.config)
    base = dict(cfg_all.get("simulation", {}))

    start = pd.Timestamp(args.start or base.get("start"))
    end = pd.Timestamp(args.end or base.get("end"))
    horizon = (end - start).total_seconds() / 60.0
    if horizon <= 0:
        raise ValueError("仿真结束时间必须晚于开始时间")

    zone_min = int(cfg_all["data"].get("zone_min", 1))
    zone_max = int(cfg_all["data"].get("zone_max", 263))

    pred_path = resolve(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")

    geo = load_geo(cfg_all["paths"]["zone_centroids"], zone_min, zone_max)
    demand = load_demand(pred_path, start, end, zone_min, zone_max)
    log.info("区间 %s ~ %s | %d 个时间片 | 预测 %s",
             start, end, len(demand), pred_path.name)

    out_dir: Path = (
        Path(args.out_dir) if args.out_dir
        else resolve(cfg_all["paths"]["simulation_dir"]) / "scenarios"
    )
    ensure_dir(out_dir)

    scenarios = build_scenarios()
    if args.only:
        scenarios = [s for s in scenarios if args.only in s.scenario_id]
    if not scenarios:
        raise SystemExit("没有匹配的场景")
    log.info("待跑场景 %d 个 -> %s", len(scenarios), out_dir)

    summary_rows: list[dict] = []
    manifest: list[dict] = []

    for i, sc in enumerate(scenarios, 1):
        cfg = dict(base)
        cfg["n_vehicles"] = sc.n_vehicles
        cfg["max_wait_min"] = sc.max_wait_min
        if sc.max_move_ratio is not None:
            cfg["max_move_ratio"] = sc.max_move_ratio

        seed = int(cfg.get("seed", 42))
        rng = np.random.default_rng(seed)
        requests = generate_requests(demand, rng)          # 与 env.py 完全一致

        first = demand.iloc[: min(4, len(demand))].sum(axis=0).to_numpy(np.float64)
        weights = first / first.sum() if first.sum() > 0 else np.full(len(first), 1 / len(first))
        initial = rng.choice(geo.zones, size=sc.n_vehicles, p=weights)

        policy = make_policy(sc.policy, **cfg)
        sim = DispatchSimulation(requests, initial, geo, policy, cfg, start, horizon)
        records, summary = sim.run(demand.index, demand)

        records.to_parquet(out_dir / f"scenario_{sc.scenario_id}_requests.parquet",
                           compression="zstd", index=False)

        row = {
            "scenario_id": sc.scenario_id,
            "policy": sc.policy,
            "n_vehicles": sc.n_vehicles,
            "max_wait_min": float(sc.max_wait_min),
            "max_move_ratio": sc.max_move_ratio,
            "total_requests": int(summary["total_requests"]),
            "served_rate": float(summary["served_rate"]),
            "avg_wait_min": float(summary["avg_wait_min"]),
            "p90_wait_min": float(summary["p90_wait_min"]),
            "deadhead_km": float(summary["deadhead_km"]),
        }
        pd.DataFrame([row]).to_parquet(
            out_dir / f"scenario_{sc.scenario_id}_summary.parquet",
            compression="zstd", index=False,
        )
        summary_rows.append(row)
        manifest.append({**asdict(sc), "seed": seed, "requests": len(requests),
                         "start": str(start), "end": str(end), "pred": str(pred_path)})

        log.info("[%2d/%2d] %-22s 服务率 %5.2f%% | 等待 %5.2f | P90 %5.2f | 空驶 %8.1f km",
                 i, len(scenarios), sc.scenario_id, row["served_rate"] * 100,
                 row["avg_wait_min"], row["p90_wait_min"], row["deadhead_km"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(out_dir / "scenarios_summary.parquet",
                          compression="zstd", index=False)
    (out_dir / "scenarios_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "config": args.config,
                "reproduce": ("python -m src.simulation.scenarios --config "
                              f"{args.config} --pred {args.pred} "
                              f'--start "{start}" --end "{end}"'),
                "scenarios": manifest,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    piv = summary_df.pivot_table(index=["n_vehicles", "max_wait_min"], columns="policy",
                                 values="served_rate", aggfunc="first", observed=True) * 100
    if {"greedy", "forecast"} <= set(piv.columns):
        piv["diff_pp"] = (piv["forecast"] - piv["greedy"]).round(2)
    print("\n服务率（%），同 (车辆数, 等待阈值) 下对比")
    print(piv.round(2).to_string())
    print("\n场景汇总表（前 10 行）")
    print(summary_df.head(10).to_string(index=False))
    print("\n已写出:", out_dir)


if __name__ == "__main__":
    main()