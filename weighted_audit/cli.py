"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys

from .data import DataError, check_disjoint, load_csv
from .report import evaluate, write_predictions_csv, write_summary_json
from .threshold import CostError, Costs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weighted-audit",
        description="带权二分类质量评估：验证集选阈值，测试集独立评估。",
    )
    parser.add_argument("--validation", required=True, help="验证集 CSV 路径")
    parser.add_argument("--test", required=True, help="测试集 CSV 路径")
    parser.add_argument("--fp-cost", type=float, required=True, help="误报成本（非负）")
    parser.add_argument("--fn-cost", type=float, required=True, help="漏报成本（非负）")
    parser.add_argument("--summary-json", required=True, help="JSON 汇总输出路径")
    parser.add_argument("--predictions-csv", required=True, help="CSV 明细输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        costs = Costs(false_positive=args.fp_cost, false_negative=args.fn_cost)
        validation = load_csv(args.validation)
        test = load_csv(args.test)
        check_disjoint(validation, test)
    except (DataError, CostError) as exc:
        print(f"输入错误：{exc}", file=sys.stderr)
        return 2
    summary, predictions = evaluate(validation, test, costs)
    write_summary_json(summary, args.summary_json)
    write_predictions_csv(predictions, args.predictions_csv)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
