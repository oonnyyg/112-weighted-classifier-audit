import math
import unittest

from weighted_audit.core import Record
from weighted_audit.metrics import (
    brier,
    confusion,
    predict,
    precision_recall_f1,
    roc_auc,
)


def rec(rid, label, prob, weight=1.0):
    return Record(rid, label, prob, weight)


class MetricsTests(unittest.TestCase):
    def test_confusion_and_scores_weighted(self):
        data = [
            rec("a", 1, 0.9, 2.0),  # tp
            rec("b", 0, 0.8, 1.0),  # fp
            rec("c", 1, 0.2, 3.0),  # fn
            rec("d", 0, 0.1, 4.0),  # tn
        ]
        rows = predict(data, 0.5, 1.0, 1.0)
        cm = confusion(rows)
        self.assertEqual(cm, {"tp": 2.0, "fp": 1.0, "fn": 3.0, "tn": 4.0})
        precision, recall, f1 = precision_recall_f1(cm)
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 2 / 5)
        self.assertAlmostEqual(f1, 4 / (4 + 1 + 3))
        self.assertAlmostEqual(sum(r["cost"] for r in rows), 4.0)

    def test_f1_none_when_denominator_zero(self):
        cm = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 5.0}
        _, _, f1 = precision_recall_f1(cm)
        self.assertIsNone(f1)

    def test_auc_probability_ties_count_half(self):
        data = [rec("p", 1, 0.5), rec("n", 0, 0.5)]
        self.assertAlmostEqual(roc_auc(data), 0.5)

    def test_auc_weighted_pairs(self):
        data = [
            rec("p1", 1, 0.9, 2.0),
            rec("p2", 1, 0.4, 1.0),
            rec("n1", 0, 0.5, 3.0),
        ]
        # pairs: (p1,n1) win w=6, (p2,n1) lose w=3 -> 6/9
        self.assertAlmostEqual(roc_auc(data), 6 / 9)

    def test_auc_none_single_class(self):
        self.assertIsNone(roc_auc([rec("a", 1, 0.3), rec("b", 1, 0.8)]))
        self.assertIsNone(roc_auc([rec("a", 0, 0.3)]))

    def test_brier_uses_raw_probabilities(self):
        data = [rec("a", 1, 0.9, 1.0), rec("b", 0, 0.6, 3.0)]
        expected = (1 * 0.1**2 + 3 * 0.6**2) / 4
        self.assertAlmostEqual(brier(data), expected)

    def test_all_negative_threshold(self):
        data = [rec("a", 1, 0.9), rec("b", 0, 0.1)]
        rows = predict(data, math.inf, 1.0, 2.0)
        self.assertTrue(all(r["predicted"] == 0 for r in rows))
        self.assertAlmostEqual(sum(r["cost"] for r in rows), 2.0)


if __name__ == "__main__":
    unittest.main()
