"""调度策略：最近车辆贪心 vs 基于预测的预调度。

策略只依赖:
    - 当前空闲车辆及其区域
    - 请求区域
    - 区域间行程时间矩阵
    - （仅 forecast）下一时间片的预测需求

设计要点（v2）:
    1. 缺口量纲一致：需求是"单/15min"，必须折算成"需要多少辆车"
       （单车 15 分钟最多完成 15 / (service_time + 平均接驾) 单）；
    2. 取车不再要求"本地有富余"——稀缺工况下该集合恒为空，
       改为只从"比目标区更不缺车"的区域抽车（比较需求/供给比）；
    3. 比例与阈值全部由 YAML 注入，便于场景实验。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

TravelTimeFn = Callable[[int, int], float]


@dataclass
class Vehicle:
    vid: int
    zone: int
    free_at: float = 0.0
    trips: int = 0
    deadhead_km: float = 0.0


class DispatchPolicy(ABC):
    name: str = "base"

    def __init__(self, max_wait_min: float = 30.0):
        self.max_wait_min = max_wait_min

    @abstractmethod
    def assign(
        self, request_zone: int, now: float, idle: list[Vehicle], travel_min: TravelTimeFn
    ) -> tuple[Vehicle | None, float]:
        """返回 (车辆, ETA 分钟)；无可用车辆返回 (None, inf)。"""

    def reposition(
        self,
        now: float,
        idle: list[Vehicle],
        next_demand: dict[int, float],
        travel_min: TravelTimeFn,
        travel_km: Callable[[int, int], float],
    ) -> list[tuple[Vehicle, int, float]]:
        """返回 [(车辆, 目标区域, 行程时间)]。默认不调度。"""
        return []


class GreedyPolicy(DispatchPolicy):
    """最近空闲车辆 + 等待上限拒绝。"""

    name = "greedy"

    def assign(self, request_zone, now, idle, travel_min):
        if not idle:
            return None, float("inf")
        etas = np.array([travel_min(v.zone, request_zone) for v in idle], dtype=np.float64)
        i = int(np.argmin(etas))
        if etas[i] > self.max_wait_min:
            return None, float("inf")
        return idle[i], float(etas[i])


class ForecastPolicy(GreedyPolicy):
    """贪心匹配 + 基于下一时间片预测的预调度。

    预调度只在"目标区需求折算成车辆数后仍缺车"时触发，
    并且只从需求/供给比更低的区域抽车，避免把稀缺资源从更缺的地方挪走。
    """

    name = "forecast"

    def __init__(
        self,
        max_wait_min: float = 30.0,
        service_time_min: float = 12.0,
        max_move_ratio: float = 0.25,
        min_demand_gain: float = 0.3,
        lookahead_steps: int = 1,
        mean_pickup_min: float = 5.0,
    ):
        super().__init__(max_wait_min=max_wait_min)
        self.service_time_min = float(service_time_min)
        self.max_move_ratio = float(max_move_ratio)
        self.min_demand_gain = float(min_demand_gain)
        self.lookahead_steps = int(lookahead_steps)
        self.mean_pickup_min = float(mean_pickup_min)

    # ---- 内部工具 ----
    @property
    def cycle_min(self) -> float:
        """单车完成一单所需时间（服务 + 平均接驾）。"""
        return self.service_time_min + self.mean_pickup_min

    def _needed_vehicles(
        self, demand_by_zone: dict[int, float], zones
    ) -> dict[int, float]:
        """把"单 / 15 分钟 / 区"折算成"需要的车辆数"。"""
        factor = self.cycle_min / 15.0
        return {z: float(demand_by_zone.get(z, 0.0)) * factor for z in zones}

    # ---- 预调度 ----
    def reposition(self, now, idle, next_demand, travel_min, travel_km):
        if not idle or not next_demand:
            return []

        idle_by_zone: dict[int, list[Vehicle]] = {}
        for v in idle:
            idle_by_zone.setdefault(v.zone, []).append(v)

        supply = {z: len(vs) for z, vs in idle_by_zone.items()}
        zones = {int(z) for z in next_demand} | set(supply)
        need = self._needed_vehicles(next_demand, zones)

        # 只对正缺口区域预调度
        deficit = {
            z: need[z] - supply.get(z, 0)
            for z in zones
            if need[z] - supply.get(z, 0) >= self.min_demand_gain
        }
        if not deficit:
            return []

        max_moves = max(1, int(len(idle) * self.max_move_ratio))
        targets = sorted(deficit, key=deficit.get, reverse=True)
        max_eta = min(self.max_wait_min, 15.0)

        moved: set[int] = set()
        moves: list[tuple[Vehicle, int, float]] = []

        for target in targets:
            if len(moves) >= max_moves:
                break

            # 目标区"每辆车要扛多少单"，越小说明越不缺车
            tgt_util = need[target] / max(supply.get(target, 0) + 1, 1)

            candidates: list[tuple[float, Vehicle]] = []
            for z, vs in idle_by_zone.items():
                if z == target:
                    continue
                src_util = need.get(z, 0.0) / max(supply.get(z, 0), 1)
                if src_util >= tgt_util:
                    continue                      # 不从更缺车的区域抽车
                for v in vs:
                    if v.vid in moved:
                        continue
                    candidates.append((travel_min(z, target), v))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[0])
            eta, vehicle = candidates[0]
            if eta > max_eta:
                continue                          # 预调度必须在一个时间片内到位

            moves.append((vehicle, target, eta))
            moved.add(vehicle.vid)
            supply[vehicle.zone] = supply.get(vehicle.zone, 1) - 1
            supply[target] = supply.get(target, 0) + 1

        return moves


def make_policy(name: str, **kwargs) -> DispatchPolicy:
    """从配置构造策略；未知键忽略，缺失键走默认值。"""
    name = str(name).lower()
    max_wait = float(kwargs.get("max_wait_min", 30.0))

    if name == "greedy":
        return GreedyPolicy(max_wait_min=max_wait)

    if name == "forecast":
        return ForecastPolicy(
            max_wait_min=max_wait,
            service_time_min=float(kwargs.get("service_time_min", 12.0)),
            max_move_ratio=float(kwargs.get("max_move_ratio", 0.25)),
            min_demand_gain=float(kwargs.get("min_demand_gain", 0.3)),
            lookahead_steps=int(kwargs.get("lookahead_steps", 1)),
            mean_pickup_min=float(kwargs.get("mean_pickup_min", 5.0)),
        )

    raise ValueError(f"未知策略: {name}")