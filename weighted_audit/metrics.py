"""Weighted metrics computed on the frozen threshold."""

from __future__ import annotations


def predict(records, threshold, fp_cost, fn_cost):
    """Per-record predictions and costs at the frozen threshold."""
    rows = []
    for r in records:
        predicted = r.probability >= threshold
        if predicted and r.label == 1:
            outcome, cost = "tp", 0.0
        elif predicted and r.label == 0:
            outcome, cost = "fp", fp_cost * r.weight
        elif not predicted and r.label == 1:
            outcome, cost = "fn", fn_cost * r.weight
        else:
            outcome, cost = "tn", 0.0
        rows.append(
            {
                "id": r.id,
                "label": r.label,
                "probability": r.probability,
                "weight": r.weight,
                "predicted": int(predicted),
                "outcome": outcome,
                "cost": cost,
            }
        )
    return rows


def confusion(rows):
    tp = sum(r["weight"] for r in rows if r["outcome"] == "tp")
    fp = sum(r["weight"] for r in rows if r["outcome"] == "fp")
    fn = sum(r["weight"] for r in rows if r["outcome"] == "fn")
    tn = sum(r["weight"] for r in rows if r["outcome"] == "tn")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(cm):
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    denom = 2 * tp + fp + fn
    f1 = (2 * tp) / denom if denom > 0 else None
    return precision, recall, f1


def roc_auc(records):
    """Weighted ROC AUC: pairs scored by weight product, ties count half.

    Returns None when only one true class is present.
    """
    positives = [r for r in records if r.label == 1]
    negatives = [r for r in records if r.label == 0]
    if not positives or not negatives:
        return None
    score = 0.0
    total = 0.0
    for p in positives:
        for n in negatives:
            w = p.weight * n.weight
            total += w
            if p.probability > n.probability:
                score += w
            elif p.probability == n.probability:
                score += 0.5 * w
    return score / total


def brier(records):
    """Weighted mean squared error of the raw probabilities."""
    total_w = sum(r.weight for r in records)
    return sum(r.weight * (r.probability - r.label) ** 2 for r in records) / total_w
