import csv
import json
import os
import tempfile
import unittest

from weighted_audit.cli import main


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("id,label,probability,weight\n")
        for row in rows:
            handle.write(",".join(str(x) for x in row) + "\n")


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = self.dir.name
        self.val = os.path.join(d, "val.csv")
        self.test = os.path.join(d, "test.csv")
        self.summary = os.path.join(d, "summary.json")
        self.preds = os.path.join(d, "preds.csv")
        write_csv(
            self.val,
            [("v1", 1, 0.9, 1.0), ("v2", 0, 0.2, 1.0), ("v3", 1, 0.6, 2.0)],
        )
        write_csv(
            self.test,
            [("t1", 1, 0.8, 1.0), ("t2", 0, 0.7, 2.0), ("t3", 0, 0.1, 1.0)],
        )

    def tearDown(self):
        self.dir.cleanup()

    def run_cli(self, *extra):
        return main(
            [
                "--validation", self.val,
                "--test", self.test,
                "--fp-cost", "1",
                "--fn-cost", "1",
                "--summary-json", self.summary,
                "--predictions-csv", self.preds,
                *extra,
            ]
        )

    def test_end_to_end(self):
        self.assertEqual(self.run_cli(), 0)
        with open(self.summary, encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertEqual(summary["threshold"], 0.6)
        test = summary["test"]
        self.assertEqual(test["confusion"]["true_positive"], 1.0)
        self.assertEqual(test["confusion"]["false_positive"], 2.0)
        self.assertEqual(test["confusion"]["true_negative"], 1.0)
        self.assertEqual(test["confusion"]["false_negative"], 0.0)
        with open(self.preds, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        detail_cost = sum(float(r["cost"]) for r in rows)
        self.assertAlmostEqual(detail_cost, test["total_cost"])
        # 明细成本 = fp_cost*FP + fn_cost*FN，与混淆矩阵一致。
        conf = test["confusion"]
        expected = conf["false_positive"] * 1 + conf["false_negative"] * 1
        self.assertAlmostEqual(test["total_cost"], expected)

    def test_shared_id_returns_error(self):
        write_csv(self.test, [("v1", 1, 0.5, 1.0)])
        self.assertEqual(self.run_cli(), 2)

    def test_zero_costs_returns_error(self):
        code = main(
            [
                "--validation", self.val,
                "--test", self.test,
                "--fp-cost", "0",
                "--fn-cost", "0",
                "--summary-json", self.summary,
                "--predictions-csv", self.preds,
            ]
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
