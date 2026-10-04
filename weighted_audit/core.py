"""Core logic for weighted binary classifier audit."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass

REQUIRED_COLUMNS = ("id", "label", "probability", "weight")


@dataclass(frozen=True)
class Record:
    id: str
    label: int
    probability: float
    weight: float


def load_records(path):
    """Load and validate a CSV of records."""
    records = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        for lineno, row in enumerate(reader, start=2):
            rid = (row["id"] or "").strip()
            if not rid:
                raise ValueError(f"{path}:{lineno}: empty id")
            try:
                label = int(row["label"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{lineno}: label must be 0 or 1") from None
            if label not in (0, 1):
                raise ValueError(f"{path}:{lineno}: label must be 0 or 1")
            try:
                prob = float(row["probability"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{lineno}: invalid probability") from None
            if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
                raise ValueError(f"{path}:{lineno}: probability must be finite and in [0, 1]")
            try:
                weight = float(row["weight"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{lineno}: invalid weight") from None
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"{path}:{lineno}: weight must be positive and finite")
            records.append(Record(rid, label, prob, weight))
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    ids = [r.id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    return records


def validate_disjoint(validation, test):
    shared = {r.id for r in validation} & {r.id for r in test}
    if shared:
        raise ValueError(f"validation and test share ids: {sorted(shared)[:5]}")


def validate_costs(fp_cost, fn_cost):
    for name, value in (("fp_cost", fp_cost), ("fn_cost", fn_cost)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be non-negative and finite")
    if fp_cost == 0.0 and fn_cost == 0.0:
        raise ValueError("fp_cost and fn_cost cannot both be zero")


def threshold_candidates(validation):
    """Candidates: 0.0, each distinct probability, and the all-negative sentinel (inf)."""
    probs = sorted({r.probability for r in validation})
    return [0.0] + probs + [math.inf]


def weighted_cost(records, threshold, fp_cost, fn_cost):
    total = 0.0
    for r in records:
        predicted = r.probability >= threshold
        if predicted and r.label == 0:
            total += fp_cost * r.weight
        elif not predicted and r.label == 1:
            total += fn_cost * r.weight
    return total


def select_threshold(validation, fp_cost, fn_cost):
    """Minimize weighted validation cost; ties go to the higher threshold.

    The all-negative sentinel (inf) ranks above every other candidate on ties.
    """
    best = None
    best_cost = None
    for cand in threshold_candidates(validation):
        cost = weighted_cost(validation, cand, fp_cost, fn_cost)
        if best_cost is None or cost < best_cost or (cost == best_cost and cand > best):
            best = cand
            best_cost = cost
    return best, best_cost
