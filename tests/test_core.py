import math
import unittest

from weighted_audit.core import (
    Record,
    select_threshold,
    threshold_candidates,
    validate_costs,
    validate_disjoint,
    weighted_cost,
)


def rec(rid, label, prob, weight=1.0):
    return Record(rid, label, prob, weight)


class ThresholdSelectionTests(unittest.TestCase):
    def test_candidates_include_zero_probs_and_sentinel(self):
        data = [rec("a", 0, 0.2), rec("b", 1, 0.2), rec("c", 1, 0.7)]
        cands = threshold_candidates(data)
        self.assertEqual(cands, [0.0, 0.2, 0.7, math.inf])

    def test_tie_picks_higher_threshold(self):
        # fp_cost == fn_cost == 1, one positive at 0.5, one negative at 0.5:
        # threshold 0.5 costs 1 (fp), threshold inf costs 1 (fn) -> tie -> inf.
        data = [rec("p", 1, 0.5), rec("n", 0, 0.5)]
        thr, cost = select_threshold(data, 1.0, 1.0)
        self.assertEqual(thr, math.inf)
        self.assertEqual(cost, 1.0)

    def test_tie_among_finite_picks_higher(self):
        # One positive at 0.3, one negative at 0.8; fp=fn=1.
        # t=0.3: fp cost 1; t=0.8: fn cost 1; tie -> higher (0.8), not inf (cost 1 too,
        # inf wins over 0.8? inf cost = fn of positive = 1 -> tie with 0.8 -> inf wins).
        data = [rec("p", 1, 0.3), rec("n", 0, 0.8)]
        thr, _ = select_threshold(data, 1.0, 1.0)
        self.assertEqual(thr, math.inf)

    def test_all_negative_when_fn_cheap(self):
        data = [rec("p", 1, 0.9), rec("n", 0, 0.1)]
        thr, cost = select_threshold(data, fp_cost=100.0, fn_cost=0.0)
        self.assertEqual(thr, math.inf)
        self.assertEqual(cost, 0.0)

    def test_zero_fp_cost_tie_picks_highest_zero_cost_threshold(self):
        data = [rec("p", 1, 0.9), rec("n", 0, 0.1)]
        thr, cost = select_threshold(data, fp_cost=0.0, fn_cost=100.0)
        # 0.0, 0.1 and 0.9 all cost 0; tie rule picks the highest.
        self.assertEqual(thr, 0.9)
        self.assertEqual(cost, 0.0)

    def test_weighted_cost_uses_weights(self):
        data = [rec("n", 0, 0.9, weight=2.0), rec("p", 1, 0.1, weight=3.0)]
        self.assertEqual(weighted_cost(data, 0.5, 1.0, 1.0), 5.0)


class ValidationTests(unittest.TestCase):
    def test_costs_must_not_both_be_zero(self):
        with self.assertRaises(ValueError):
            validate_costs(0.0, 0.0)
        with self.assertRaises(ValueError):
            validate_costs(-1.0, 1.0)
        validate_costs(0.0, 1.0)

    def test_disjoint_ids(self):
        with self.assertRaises(ValueError):
            validate_disjoint([rec("a", 0, 0.1)], [rec("a", 1, 0.2)])
        validate_disjoint([rec("a", 0, 0.1)], [rec("b", 1, 0.2)])


if __name__ == "__main__":
    unittest.main()
