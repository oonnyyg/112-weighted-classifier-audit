"""CSV 数据读取与校验。"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    """单条评估记录。"""

    record_id: str
    label: int
    probability: float
    weight: float


class DataError(ValueError):
    """输入数据不合法。"""


def _parse_label(raw: str, row_no: int) -> int:
    text = raw.strip()
    if text in ("0", "1"):
        return int(text)
    raise DataError(f"第 {row_no} 行：标签必须为 0 或 1，得到 {raw!r}")


def _parse_probability(raw: str, row_no: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise DataError(f"第 {row_no} 行：概率不是数值：{raw!r}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise DataError(f"第 {row_no} 行：概率必须为 [0,1] 内的有限值，得到 {raw!r}")
    return value


def _parse_weight(raw: str, row_no: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise DataError(f"第 {row_no} 行：权重不是数值：{raw!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise DataError(f"第 {row_no} 行：权重必须为非负有限值，得到 {raw!r}")
    return value


def load_csv(path: str) -> list[Record]:
    """读取 CSV，要求列：id,label,probability,weight。"""
    records: list[Record] = []
    seen: set[str] = set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "label", "probability", "weight"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise DataError(f"{path}：缺少必需列 {sorted(required)}")
        for row_no, row in enumerate(reader, start=2):
            record_id = (row["id"] or "").strip()
            if not record_id:
                raise DataError(f"第 {row_no} 行：标识不能为空")
            if record_id in seen:
                raise DataError(f"第 {row_no} 行：标识重复：{record_id!r}")
            seen.add(record_id)
            records.append(
                Record(
                    record_id=record_id,
                    label=_parse_label(row["label"], row_no),
                    probability=_parse_probability(row["probability"], row_no),
                    weight=_parse_weight(row["weight"], row_no),
                )
            )
    if not records:
        raise DataError(f"{path}：数据集不能为空")
    return records


def check_disjoint(validation: list[Record], test: list[Record]) -> None:
    """校验验证集与测试集标识不重叠。"""
    val_ids = {r.record_id for r in validation}
    overlap = val_ids & {r.record_id for r in test}
    if overlap:
        raise DataError(f"验证集与测试集共享标识：{sorted(overlap)[:5]}")
