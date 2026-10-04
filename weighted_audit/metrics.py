"""冻结阈值后的加权评估指标。"""

from __future__ import annotations

from dataclasses import dataclass

from .data import Record
from .threshold import Costs


@dataclass(frozen=True)
class Prediction:
    record_id: str
    label: int
    probability: float
    weight: float
    predicted: int
    cost: float


@dataclass(frozen=True)
class Confusion:
    true_positive: float
    false_positive: float
    true_negative: float
    false_negative: float


def predict_all(records: list[Record], threshold: float, costs: Costs) -> list[Prediction]:
    """按冻结阈值生成逐条预测与成本。"""
    results = []
    for rec in records:
        predicted = 1 if rec.probability >= threshold else 0
        cost = 0.0
        if predicted == 1 and rec.label == 0:
            cost = costs.false_positive * rec.weight
        elif predicted == 0 and rec.label == 1:
            cost = costs.false_negative * rec.weight
        results.append(
            Prediction(
                record_id=rec.record_id,
                label=rec.label,
                probability=rec.probability,
                weight=rec.weight,
                predicted=predicted,
                cost=cost,
            )
        )
    return results


def confusion_matrix(records: list[Record], threshold: float) -> Confusion:
    """加权混淆矩阵。"""
    tp = fp = tn = fn = 0.0
    for rec in records:
        predicted = 1 if rec.probability >= threshold else 0
        if predicted == 1 and rec.label == 1:
            tp += rec.weight
        elif predicted == 1 and rec.label == 0:
            fp += rec.weight
        elif predicted == 0 and rec.label == 0:
            tn += rec.weight
        else:
            fn += rec.weight
    return Confusion(tp, fp, tn, fn)


def precision(conf: Confusion) -> float | None:
    denom = conf.true_positive + conf.false_positive
    if denom == 0.0:
        return None
    return conf.true_positive / denom


def recall(conf: Confusion) -> float | None:
    denom = conf.true_positive + conf.false_negative
    if denom == 0.0:
        return None
    return conf.true_positive / denom


def f1(conf: Confusion) -> tuple[float | None, str | None]:
    """F1 = 2TP / (2TP + FP + FN)；分母为零时返回原因。"""
    denom = 2.0 * conf.true_positive + conf.false_positive + conf.false_negative
    if denom == 0.0:
        return None, "F1 分母为零：加权真阳性、误报与漏报均为零"
    return 2.0 * conf.true_positive / denom, None


def roc_auc(records: list[Record]) -> tuple[float | None, str | None]:
    """加权 ROC AUC：正负样本对按权重乘积计分，同分计半分。"""
    positives = [r for r in records if r.label == 1]
    negatives = [r for r in records if r.label == 0]
    if not positives or not negatives:
        return None, "测试集只包含一种真实类别，无法计算 ROC AUC"
    score = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            pair_weight = pos.weight * neg.weight
            total += pair_weight
            if pos.probability > neg.probability:
                score += pair_weight
            elif pos.probability == neg.probability:
                score += 0.5 * pair_weight
    if total == 0.0:
        return None, "正负样本对总权重为零，无法计算 ROC AUC"
    return score / total, None


def brier_score(records: list[Record]) -> float:
    """加权 Brier 均方误差，基于原始概率而非二值化预测。"""
    total_weight = sum(r.weight for r in records)
    if total_weight == 0.0:
        return 0.0
    return sum(r.weight * (r.probability - r.label) ** 2 for r in records) / total_weight
