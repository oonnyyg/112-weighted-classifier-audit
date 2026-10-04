import csv
import json
import os
import tempfile
import unittest

from weighted_audit.cli import main


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "label", "probability", "weight"])
        writer.writerows(rows)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.val = os.path.join(d, "val.csv")
        self.test = os.path.join(d, "test.csv")
        self.summary = os.path.join(d, "summary.json")
        self.details = os.path.join(d, "details.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, fp="1", fn="1"):
        return main([
            "--validation", self.val,
            "--test", self.test,
            "--fp-cost", fp,
            "--fn-cost", fn,
            "--summary", self.summary,
            "--details", self.details,
        ])

    def test_end_to_end(self):
        write_csv(self.val, [
            ["v1", 1, 0.9, 1.0],
            ["v2", 0, 0.2, 1.0],
        ])
        write_csv(self.test, [
            ["t1", 1, 0.8, 2.0],
            ["t2", 0, 0.7, 1.0],
            ["t3", 0, 0.1, 1.0],
        ])
        self.assertEqual(self.run_cli(), 0)
        with open(self.summary, encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertEqual(summary["threshold"], 0.9)
        cm = summary["test"]["confusion"]
        self.assertEqual(cm, {"tp": 0.0, "fp": 0.0, "fn": 2.0, "tn": 2.0})
        self.assertEqual(summary["test"]["total_cost"], 2.0)
        with open(self.details, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(
            sum(float(r["cost"]) for r in rows), summary["test"]["total_cost"]
        )

    def test_shared_ids_rejected(self):
        write_csv(self.val, [["x", 1, 0.5, 1.0]])
        write_csv(self.test, [["x", 0, 0.5, 1.0]])
        self.assertEqual(self.run_cli(), 2)

    def test_empty_test_rejected(self):
        write_csv(self.val, [["v", 1, 0.5, 1.0]])
        write_csv(self.test, [])
        self.assertEqual(self.run_cli(), 2)

    def test_invalid_probability_rejected(self):
        write_csv(self.val, [["v", 1, 1.5, 1.0]])
        write_csv(self.test, [["t", 0, 0.5, 1.0]])
        self.assertEqual(self.run_cli(), 2)

    def test_single_class_test_auc_null(self):
        write_csv(self.val, [["v1", 1, 0.9, 1.0], ["v2", 0, 0.1, 1.0]])
        write_csv(self.test, [["t1", 1, 0.8, 1.0], ["t2", 1, 0.4, 1.0]])
        self.assertEqual(self.run_cli(), 0)
        with open(self.summary, encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertIsNone(summary["test"]["roc_auc"])
        self.assertIsNotNone(summary["test"]["roc_auc_note"])


if __name__ == "__main__":
    unittest.main()
