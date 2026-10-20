"""调度策略：最近车辆贪心 vs 基于预测的预调度。

策略只依赖:
    - 当前空闲车辆及其区域
    - 请求区域
    - 区域间行程时间矩阵
    - （仅 forecast）下一时间片的预测需求
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
    """在贪心匹配基础上，把空闲车辆预调度到下一时间片的高需求区域。

    预调度只移动“当前区域供给相对需求过剩”的车辆，并限制最大移动比例，
    避免为了预测噪声付出过高空驶成本。
    """

    name = "forecast"

    def __init__(
        self,
        max_wait_min: float = 30.0,
        max_move_ratio: float = 0.25,
        min_demand_gain: float = 1.0,
        lookahead_steps: int = 1,
    ):
        super().__init__(max_wait_min=max_wait_min)
        self.max_move_ratio = max_move_ratio
        self.min_demand_gain = min_demand_gain
        self.lookahead_steps = lookahead_steps

    def reposition(self, now, idle, next_demand, travel_min, travel_km):
        if not idle or not next_demand:
            return []
        idle_by_zone: dict[int, list[Vehicle]] = {}
        for v in idle:
            idle_by_zone.setdefault(v.zone, []).append(v)

        supply = {z: len(vs) for z, vs in idle_by_zone.items()}
        zones = set(next_demand) | set(supply)
        # 需求相对供给的缺口；只考虑正缺口区域
        deficit = {
            z: float(next_demand.get(z, 0.0)) - float(supply.get(z, 0))
            for z in zones
            if float(next_demand.get(z, 0.0)) - float(supply.get(z, 0)) >= self.min_demand_gain
        }
        if not deficit:
            return []

        max_moves = max(1, int(len(idle) * self.max_move_ratio))
        targets = sorted(deficit, key=deficit.get, reverse=True)
        moved: set[int] = set()
        moves: list[tuple[Vehicle, int, float]] = []

        for target in targets:
            if len(moves) >= max_moves:
                break
            candidates = []
            for z, vs in idle_by_zone.items():
                if z == target:
                    continue
                surplus = supply.get(z, 0) - float(next_demand.get(z, 0.0))
                if surplus <= 0:
                    continue
                for v in vs:
                    if v.vid in moved:
                        continue
                    candidates.append((travel_min(z, target), v))
            candidates.sort(key=lambda x: x[0])
            if not candidates:
                continue
            eta, vehicle = candidates[0]
            # 预调度必须在下一时间片到来前完成，最多等一个 15 分钟片
            if eta > min(self.max_wait_min, 15.0):
                continue
            moves.append((vehicle, target, eta))
            moved.add(vehicle.vid)
            supply[vehicle.zone] = supply.get(vehicle.zone, 1) - 1
            supply[target] = supply.get(target, 0) + 1
        return moves


def make_policy(name: str, max_wait_min: float) -> DispatchPolicy:
    if name == "greedy":
        return GreedyPolicy(max_wait_min=max_wait_min)
    if name == "forecast":
        return ForecastPolicy(max_wait_min=max_wait_min)
    raise ValueError(f"未知策略: {name}")