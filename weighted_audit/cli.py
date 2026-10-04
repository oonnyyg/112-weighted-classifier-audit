"""Command line entry point for the weighted classifier audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys

from .core import (
    load_records,
    select_threshold,
    validate_costs,
    validate_disjoint,
)
from .metrics import brier, confusion, predict, precision_recall_f1, roc_auc


def build_parser():
    parser = argparse.ArgumentParser(
        prog="weighted-audit",
        description="Pick a business threshold on validation data, then audit a test set.",
    )
    parser.add_argument("--validation", required=True, help="validation CSV path")
    parser.add_argument("--test", required=True, help="test CSV path")
    parser.add_argument("--fp-cost", required=True, type=float, help="false positive cost")
    parser.add_argument("--fn-cost", required=True, type=float, help="false negative cost")
    parser.add_argument("--summary", required=True, help="output JSON summary path")
    parser.add_argument("--details", required=True, help="output CSV details path")
    return parser


def run(args):
    validate_costs(args.fp_cost, args.fn_cost)
    validation = load_records(args.validation)
    test = load_records(args.test)
    validate_disjoint(validation, test)

    threshold, val_cost = select_threshold(validation, args.fp_cost, args.fn_cost)
    all_negative = math.isinf(threshold)

    rows = predict(test, threshold, args.fp_cost, args.fn_cost)
    cm = confusion(rows)
    precision, recall, f1 = precision_recall_f1(cm)
    auc = roc_auc(test)
    brier_score = brier(test)
    total_cost = sum(r["cost"] for r in rows)

    summary = {
        "threshold": None if all_negative else threshold,
        "threshold_all_negative": all_negative,
        "fp_cost": args.fp_cost,
        "fn_cost": args.fn_cost,
        "validation_cost": val_cost,
        "test": {
            "n": len(test),
            "confusion": cm,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f1_note": None
            if f1 is not None
            else "f1 undefined: 2*tp + fp + fn weights sum to zero",
            "roc_auc": auc,
            "roc_auc_note": None
            if auc is not None
            else "roc_auc undefined: only one true class present",
            "brier": brier_score,
            "total_cost": total_cost,
        },
    }

    with open(args.summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    with open(args.details, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "label", "probability", "weight", "predicted", "outcome", "cost"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
