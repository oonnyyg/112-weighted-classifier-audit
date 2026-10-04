import unittest

from weighted_audit.data import Record
from weighted_audit.metrics import (
    Confusion,
    brier_score,
    confusion_matrix,
    f1,
    precision,
    recall,
    roc_auc,
)


def rec(rid, label, prob, weight=1.0):
    return Record(rid, label, prob, weight)


class ConfusionTests(unittest.TestCase):
    def test_weighted_confusion(self):
        records = [
            rec("a", 1, 0.9, 2.0),
            rec("b", 0, 0.8, 1.0),
            rec("c", 0, 0.1, 3.0),
            rec("d", 1, 0.2, 4.0),
        ]
        conf = confusion_matrix(records, 0.5)
        self.assertEqual(conf, Confusion(2.0, 1.0, 3.0, 4.0))
        self.assertAlmostEqual(precision(conf), 2.0 / 3.0)
        self.assertAlmostEqual(recall(conf), 2.0 / 6.0)
        value, note = f1(conf)
        self.assertAlmostEqual(value, 4.0 / 9.0)
        self.assertIsNone(note)

    def test_f1_zero_denominator(self):
        value, note = f1(Confusion(0.0, 0.0, 5.0, 0.0))
        self.assertIsNone(value)
        self.assertIsNotNone(note)


class RocAucTests(unittest.TestCase):
    def test_weighted_auc(self):
        records = [
            rec("p1", 1, 0.9, 2.0),
            rec("n1", 0, 0.1, 1.0),
            rec("n2", 0, 0.95, 1.0),
        ]
        auc, note = roc_auc(records)
        # 正样本权重 2：胜 (0.9>0.1) 计 2，负 (0.9<0.95) 计 0，总权重 4。
        self.assertAlmostEqual(auc, 0.5)
        self.assertIsNone(note)

    def test_tied_probabilities_score_half(self):
        records = [rec("p", 1, 0.5, 2.0), rec("n", 0, 0.5, 3.0)]
        auc, note = roc_auc(records)
        self.assertAlmostEqual(auc, 0.5)

    def test_single_class_returns_none(self):
        records = [rec("a", 1, 0.5), rec("b", 1, 0.7)]
        auc, note = roc_auc(records)
        self.assertIsNone(auc)
        self.assertIsNotNone(note)


class BrierTests(unittest.TestCase):
    def test_weighted_brier_uses_probabilities(self):
        records = [rec("a", 1, 0.8, 1.0), rec("b", 0, 0.6, 3.0)]
        expected = (1.0 * 0.04 + 3.0 * 0.36) / 4.0
        self.assertAlmostEqual(brier_score(records), expected)


if __name__ == "__main__":
    unittest.main()
