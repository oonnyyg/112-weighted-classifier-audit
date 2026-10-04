import unittest

from weighted_audit.data import Record
from weighted_audit.threshold import (
    ALL_NEGATIVE,
    CostError,
    Costs,
    candidates,
    select_threshold,
)


def rec(rid, label, prob, weight=1.0):
    return Record(rid, label, prob, weight)


class CostsTests(unittest.TestCase):
    def test_negative_cost_rejected(self):
        with self.assertRaises(CostError):
            Costs(false_positive=-1.0, false_negative=1.0)

    def test_both_zero_rejected(self):
        with self.assertRaises(CostError):
            Costs(false_positive=0.0, false_negative=0.0)

    def test_one_zero_allowed(self):
        Costs(false_positive=0.0, false_negative=1.0)


class CandidateTests(unittest.TestCase):
    def test_candidates_include_zero_probs_and_all_negative(self):
        records = [rec("a", 1, 0.3), rec("b", 0, 0.3), rec("c", 1, 0.8)]
        cands = candidates(records)
        self.assertEqual(cands, [0.0, 0.3, 0.8, ALL_NEGATIVE])


class SelectThresholdTests(unittest.TestCase):
    def test_basic_selection(self):
        records = [rec("a", 1, 0.9), rec("b", 0, 0.1)]
        costs = Costs(1.0, 1.0)
        threshold, cost = select_threshold(records, costs)
        self.assertEqual(threshold, 0.9)
        self.assertEqual(cost, 0.0)

    def test_tie_prefers_higher_threshold(self):
        # 概率同分且成本对称：候选 0、0.5、全部判负成本均为 1，取最高。
        records = [rec("a", 1, 0.5), rec("b", 0, 0.5)]
        costs = Costs(1.0, 1.0)
        threshold, cost = select_threshold(records, costs)
        self.assertEqual(cost, 1.0)
        self.assertEqual(threshold, ALL_NEGATIVE)

    def test_tie_among_numeric_thresholds(self):
        # 候选 0、0.5、0.9 成本均为 1，全部判负为 2，取最高数值阈值 0.9。
        records = [
            rec("p1", 1, 0.5),
            rec("p2", 1, 0.9),
            rec("n1", 0, 0.5, weight=0.0),
            rec("n2", 0, 0.8),
        ]
        costs = Costs(1.0, 1.0)
        threshold, cost = select_threshold(records, costs)
        self.assertEqual(cost, 1.0)
        self.assertEqual(threshold, 0.9)

    def test_all_negative_is_highest_candidate(self):
        # 漏报成本为零时，全部判负成本最低且为最高候选。
        records = [rec("a", 1, 0.9), rec("b", 0, 0.8)]
        costs = Costs(1.0, 0.0)
        threshold, cost = select_threshold(records, costs)
        self.assertEqual(threshold, ALL_NEGATIVE)
        self.assertEqual(cost, 0.0)

    def test_weighted_costs_drive_selection(self):
        records = [rec("a", 1, 0.7, weight=3.0), rec("b", 0, 0.2, weight=1.0)]
        costs = Costs(1.0, 1.0)
        threshold, cost = select_threshold(records, costs)
        self.assertEqual(threshold, 0.7)
        self.assertEqual(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
