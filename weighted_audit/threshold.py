"""基于验证集的加权成本阈值选择。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data import Record

ALL_NEGATIVE = math.inf


class CostError(ValueError):
    """成本配置不合法。"""


@dataclass(frozen=True)
class Costs:
    false_positive: float
    false_negative: float

    def __post_init__(self) -> None:
        for name in ("false_positive", "false_negative"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise CostError(f"成本 {name} 必须为非负有限值，得到 {value!r}")
        if self.false_positive == 0.0 and self.false_negative == 0.0:
            raise CostError("误报与漏报成本不能同时为零")


def weighted_cost(records: list[Record], threshold: float, costs: Costs) -> float:
    """计算给定阈值下的加权误报与漏报总成本。"""
    total = 0.0
    for rec in records:
        predicted = 1 if rec.probability >= threshold else 0
        if predicted == 1 and rec.label == 0:
            total += costs.false_positive * rec.weight
        elif predicted == 0 and rec.label == 1:
            total += costs.false_negative * rec.weight
    return total


def candidates(records: list[Record]) -> list[float]:
    """候选阈值：0、验证集不同概率、全部判负。"""
    values = {0.0}
    values.update(rec.probability for rec in records)
    values.add(ALL_NEGATIVE)
    return sorted(values)


def select_threshold(records: list[Record], costs: Costs) -> tuple[float, float]:
    """返回 (阈值, 验证集最小加权成本)；并列时取更高阈值。"""
    best_threshold = 0.0
    best_cost = math.inf
    for threshold in candidates(records):
        cost = weighted_cost(records, threshold, costs)
        # 升序遍历，成本相等时后者（更高阈值）胜出；全部判负最高。
        if cost <= best_cost:
            best_cost = cost
            best_threshold = threshold
    return best_threshold, best_cost
