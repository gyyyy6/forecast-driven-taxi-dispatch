"""SimPy 离散事件仿真：预测需求 → 车辆调度 → 服务指标。

用法:
    python -m src.simulation.env --config configs/default.yaml --policy both
    python -m src.simulation.env --config configs/default.yaml --policy forecast \
        --pred data/processed/predictions/pred_test.parquet

说明:
    - 需求来自模型预测（predicted_demand），按 15 分钟片转成泊松到达；
    - 两种策略共享同一请求流与初始车辆位置，保证对比公平；
    - 区域距离优先使用 TLC 质心文件（X/Y 为纽约州平面坐标，单位英尺）。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simpy

from src.config import ensure_dir, load_config, make_logger, resolve
from src.simulation.policy import DispatchPolicy, Vehicle, make_policy

log = make_logger("simulation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run dispatch simulation.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--pred", default=None)
    p.add_argument("--policy", default="both", choices=["greedy", "forecast", "both"])
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--n-vehicles", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-move-ratio", type=float, default=None)
    p.add_argument("--min-demand-gain", type=float, default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


@dataclass
class Geo:
    zones: np.ndarray
    dist_km: np.ndarray

    def travel_min(self, a: int, b: int, speed_kmh: float) -> float:
        offset = int(self.zones[0])
        d = float(self.dist_km[int(a) - offset, int(b) - offset])
        return max(60.0 * d / max(speed_kmh, 1e-6), 1.0)

    def travel_km(self, a: int, b: int) -> float:
        offset = int(self.zones[0])
        return float(self.dist_km[int(a) - offset, int(b) - offset])


def load_geo(path: str | Path, zone_min: int, zone_max: int) -> Geo:
    """读取 TLC 质心，构造区域距离矩阵（km）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"缺少质心文件: {path}\n"
            "请从 NYC TLC 下载 Taxi Zone Centroids（含 LocationID/X/Y），"
            "或提供 lat/lon 列。没有距离矩阵无法进行调度仿真。"
        )
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    id_col = low.get("locationid") or low.get("location_id") or low.get("zone_id")
    lat_col, lon_col = low.get("lat"), low.get("lon") or low.get("lng")
    x_col, y_col = low.get("x"), low.get("y")
    if id_col is None:
        raise ValueError(f"质心文件缺少 LocationID 列: {list(df.columns)}")

    zones = np.arange(zone_min, zone_max + 1)
    df = df.set_index(id_col).reindex(zones)
    needed = [lat_col, lon_col] if (lat_col and lon_col) else [x_col, y_col]
    if any(c is None for c in needed):
        raise ValueError("质心文件需包含 lat/lon 或 X/Y 列")
    if df[needed].isna().any().any():
        raise ValueError("质心文件缺少部分 Taxi Zone，无法构造完整距离矩阵")

    if lat_col and lon_col:
        lat = np.radians(df[lat_col].to_numpy(np.float64))[:, None]
        lon = np.radians(df[lon_col].to_numpy(np.float64))[:, None]
        dlat = lat - lat.T
        dlon = lon - lon.T
        a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
        dist = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    elif x_col and y_col:
        xy = df[[x_col, y_col]].to_numpy(np.float64)
        dist_ft = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
        dist = dist_ft * 0.0003048  # feet -> km
    else:
        raise ValueError("质心文件需包含 lat/lon 或 X/Y 列")
    return Geo(zones=zones, dist_km=dist.astype(np.float32))


def load_demand(path: str | Path, start: pd.Timestamp, end: pd.Timestamp, zone_min: int, zone_max: int) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["timestamp", "zone_id", "predicted_demand"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    if df.empty:
        raise ValueError(f"{path} 在 [{start}, {end}) 内没有预测数据")
    zones = np.arange(zone_min, zone_max + 1)
    wide = (
        df.pivot_table(index="timestamp", columns="zone_id", values="predicted_demand", aggfunc="sum")
        .reindex(columns=zones)
        .fillna(0.0)
        .sort_index()
    )
    return wide.clip(lower=0)


def generate_requests(demand: pd.DataFrame, rng: np.random.Generator) -> list[tuple[float, int]]:
    """把每个 15 分钟片的需求转成泊松到达，返回 [(分钟偏移, 区域)]。"""
    times = demand.index
    step_min = float((times[1] - times[0]).total_seconds() / 60.0) if len(times) > 1 else 15.0
    t0 = times[0]
    requests: list[tuple[float, int]] = []
    for ts, row in demand.iterrows():
        offset0 = (ts - t0).total_seconds() / 60.0
        for zone, mu in row.items():
            mu = float(mu)
            if mu <= 0:
                continue
            for _ in range(rng.poisson(mu)):
                requests.append((offset0 + rng.uniform(0, step_min), int(zone)))
    requests.sort(key=lambda x: x[0])
    return requests


class DispatchSimulation:
    def __init__(
        self,
        requests: list[tuple[float, int]],
        initial_zones: np.ndarray,
        geo: Geo,
        policy: DispatchPolicy,
        cfg: dict,
        start: pd.Timestamp,
        horizon_min: float,
    ):
        self.env = simpy.Environment()
        self.requests = requests
        self.geo = geo
        self.policy = policy
        self.cfg = cfg
        self.start = start
        self.horizon = horizon_min
        self.speed = float(cfg.get("speed_kmh", 25))
        self.service = float(cfg.get("service_time_min", 12))
        self.reposition_every = float(cfg.get("dispatch_interval_min", 15))
        self.vehicles = [Vehicle(vid=i, zone=int(z)) for i, z in enumerate(initial_zones)]
        self.records: list[dict] = []
        self.reposition_events: list[dict] = []
        self._next_request_id = 0

        self._demand_by_step: dict[int, dict[int, float]] = {}
        self._demand_times = None  # set by caller if forecast is used

    def travel_min(self, a: int, b: int) -> float:
        return self.geo.travel_min(a, b, self.speed)

    def travel_km(self, a: int, b: int) -> float:
        return self.geo.travel_km(a, b)

    def _request(self, arrival_min: float, zone: int):
        env = self.env
        yield env.timeout(max(0.0, arrival_min - env.now))
        now = env.now
        idle = [v for v in self.vehicles if v.free_at <= now + 1e-9]
        vehicle, eta = self.policy.assign(zone, now, idle, self.travel_min)
        rid = self._next_request_id
        self._next_request_id += 1

        if vehicle is None:
            self.records.append(
                {
                    "request_id": rid,
                    "request_time": self.start + pd.Timedelta(minutes=now),
                    "zone_id": np.int16(zone),
                    "assigned_vehicle": np.int32(-1),
                    "pickup_time": pd.NaT,
                    "wait_min": np.float32(np.nan),
                    "served": np.int8(0),
                    "policy": self.policy.name,
                }
            )
            return

        pickup_min = now + eta
        km = self.travel_km(vehicle.zone, zone)
        vehicle.free_at = pickup_min + self.service
        vehicle.zone = zone
        vehicle.trips += 1
        vehicle.deadhead_km += km
        self.records.append(
            {
                "request_id": rid,
                "request_time": self.start + pd.Timedelta(minutes=now),
                "zone_id": np.int16(zone),
                "assigned_vehicle": np.int32(vehicle.vid),
                "pickup_time": self.start + pd.Timedelta(minutes=pickup_min),
                "wait_min": np.float32(eta),
                "served": np.int8(1),
                "policy": self.policy.name,
            }
        )

    def _reposition_loop(self, demand_times: pd.DatetimeIndex, demand: pd.DataFrame):
        """每 15 分钟，用下一时间片的预测需求做一次预调度。"""
        step = self.reposition_every
        while self.env.now + step <= self.horizon:
            yield self.env.timeout(step)
            now = self.env.now
            idx = int(round(now / step))
            next_idx = min(idx, len(demand_times) - 1)
            next_demand = {int(z): float(v) for z, v in demand.iloc[next_idx].items()}
            idle = [v for v in self.vehicles if v.free_at <= now + 1e-9]
            moves = self.policy.reposition(now, idle, next_demand, self.travel_min, self.travel_km)
            for vehicle, target, eta in moves:
                from_zone = vehicle.zone
                km = self.travel_km(from_zone, target)
                vehicle.free_at = now + eta
                vehicle.zone = target
                vehicle.deadhead_km += km
                self.reposition_events.append(
                    {
                        "time": self.start + pd.Timedelta(minutes=now),
                        "vehicle_id": vehicle.vid,
                        "from_zone": int(from_zone),
                        "to_zone": int(target),
                        "eta_min": float(eta),
                        "deadhead_km": float(km),
                        "policy": self.policy.name,
                    }
                )

    def run(self, demand_times: pd.DatetimeIndex, demand: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        for arrival, zone in self.requests:
            if arrival <= self.horizon:
                self.env.process(self._request(arrival, zone))
        if self.policy.name == "forecast":
            self.env.process(self._reposition_loop(demand_times, demand))
        self.env.run(until=self.horizon)

        records = pd.DataFrame(self.records)
        if records.empty:
            summary = {"policy": self.policy.name, "total_requests": 0, "served_rate": 0.0,
                       "avg_wait_min": float("nan"), "p90_wait_min": float("nan"), "deadhead_km": 0.0}
            return records, summary

        served = records[records["served"] == 1]
        summary = {
            "policy": self.policy.name,
            "total_requests": int(len(records)),
            "served_rate": float(len(served) / len(records)),
            "avg_wait_min": float(served["wait_min"].mean()) if len(served) else float("nan"),
            "p90_wait_min": float(served["wait_min"].quantile(0.9)) if len(served) else float("nan"),
            "deadhead_km": float(sum(v.deadhead_km for v in self.vehicles)),
            "n_vehicles": len(self.vehicles),
        }
        return records, summary


def main() -> None:
    args = parse_args()
    cfg_all = load_config(args.config)
    cfg = dict(cfg_all.get("simulation", {}))
    if args.start:
        cfg["start"] = args.start
    if args.end:
        cfg["end"] = args.end
    if args.n_vehicles:
        cfg["n_vehicles"] = args.n_vehicles
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_move_ratio is not None:
        cfg["max_move_ratio"] = args.max_move_ratio
    if args.min_demand_gain is not None:
        cfg["min_demand_gain"] = args.min_demand_gain

    start = pd.Timestamp(cfg["start"])
    end = pd.Timestamp(cfg["end"])
    horizon = (end - start).total_seconds() / 60.0
    if horizon <= 0:
        raise ValueError("仿真结束时间必须晚于开始时间")

    pred_path = resolve(args.pred or cfg_all["visualization"]["prediction_path"])
    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_path}，请先运行 predict.py")

    zone_min = int(cfg_all["data"].get("zone_min", 1))
    zone_max = int(cfg_all["data"].get("zone_max", 263))
    geo = load_geo(cfg_all["paths"]["zone_centroids"], zone_min, zone_max)
    demand = load_demand(pred_path, start, end, zone_min, zone_max)
    log.info("仿真区间 %s ~ %s，需求 %d 个时间片", start, end, len(demand))

    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    requests = generate_requests(demand, rng)
    log.info("生成请求 %d 条", len(requests))

    # 初始车辆按首小时需求分布投放
    first = demand.iloc[: min(4, len(demand))].sum(axis=0).to_numpy(np.float64)
    weights = first / first.sum() if first.sum() > 0 else np.full(len(first), 1 / len(first))
    initial = rng.choice(geo.zones, size=int(cfg.get("n_vehicles", 200)), p=weights)

    policies = ["greedy", "forecast"] if args.policy == "both" else [args.policy]
    all_records, summaries = [], []
    for name in policies:
        policy = make_policy(name, **cfg)
        sim = DispatchSimulation(
            requests=requests,
            initial_zones=initial,
            geo=geo,
            policy=policy,
            cfg=cfg,
            start=start,
            horizon_min=horizon,
        )
        records, summary = sim.run(demand.index, demand)
        all_records.append(records)
        summaries.append(summary)
        log.info(
            "%s: 服务率 %.1f%%, 平均等待 %.1f min, P90 %.1f min, 空驶 %.1f km",
            name,
            summary["served_rate"] * 100,
            summary["avg_wait_min"],
            summary["p90_wait_min"],
            summary["deadhead_km"],
        )

    out_dir = ensure_dir(args.out_dir or cfg_all["paths"]["simulation_dir"])
    records_df = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    if not records_df.empty:
        records_df["policy"] = records_df["policy"].astype("category")
    summary_df["policy"] = summary_df["policy"].astype("category")
    records_df.to_parquet(out_dir / "sim_requests.parquet", compression="zstd", index=False)
    summary_df.to_parquet(out_dir / "sim_results.parquet", compression="zstd", index=False)
    for name, rec in zip(policies, all_records):
        rec.to_parquet(out_dir / f"sim_requests_{name}.parquet", compression="zstd", index=False)
        summary_df[summary_df["policy"] == name].to_parquet(
            out_dir / f"sim_results_{name}.parquet", compression="zstd", index=False
        )
    (out_dir / "sim_config.json").write_text(
        json.dumps({**cfg, "start": str(start), "end": str(end), "prediction": str(pred_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("结果 -> %s", out_dir)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()