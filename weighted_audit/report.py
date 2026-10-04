"""汇总评估并导出 JSON 与 CSV。"""

from __future__ import annotations

import csv
import json

from .data import Record
from .metrics import (
    Prediction,
    brier_score,
    confusion_matrix,
    f1,
    precision,
    predict_all,
    recall,
    roc_auc,
)
from .threshold import ALL_NEGATIVE, Costs, select_threshold


def _threshold_label(threshold: float) -> str:
    if threshold == ALL_NEGATIVE:
        return "all_negative"
    return repr(threshold)


def evaluate(
    validation: list[Record],
    test: list[Record],
    costs: Costs,
) -> tuple[dict, list[Prediction]]:
    """在验证集上选阈值，冻结后评估测试集，返回 (汇总, 明细)。"""
    threshold, validation_cost = select_threshold(validation, costs)
    predictions = predict_all(test, threshold, costs)
    conf = confusion_matrix(test, threshold)
    f1_value, f1_note = f1(conf)
    auc_value, auc_note = roc_auc(test)
    total_cost = sum(p.cost for p in predictions)
    summary = {
        "costs": {
            "false_positive": costs.false_positive,
            "false_negative": costs.false_negative,
        },
        "threshold": None if threshold == ALL_NEGATIVE else threshold,
        "threshold_rule": _threshold_label(threshold),
        "validation": {
            "size": len(validation),
            "weighted_cost": validation_cost,
        },
        "test": {
            "size": len(test),
            "confusion": {
                "true_positive": conf.true_positive,
                "false_positive": conf.false_positive,
                "true_negative": conf.true_negative,
                "false_negative": conf.false_negative,
            },
            "precision": precision(conf),
            "recall": recall(conf),
            "f1": f1_value,
            "f1_note": f1_note,
            "roc_auc": auc_value,
            "roc_auc_note": auc_note,
            "brier_score": brier_score(test),
            "total_cost": total_cost,
        },
    }
    return summary, predictions


def write_summary_json(summary: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_predictions_csv(predictions: list[Prediction], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "probability", "weight", "predicted", "cost"])
        for p in predictions:
            writer.writerow(
                [p.record_id, p.label, repr(p.probability), repr(p.weight), p.predicted, repr(p.cost)]
            )
