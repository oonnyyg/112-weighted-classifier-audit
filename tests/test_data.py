import unittest

from weighted_audit.data import DataError, Record, check_disjoint, load_csv


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("id,label,probability,weight\n")
        for row in rows:
            handle.write(",".join(str(x) for x in row) + "\n")


class LoadCsvTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.dir = tempfile.TemporaryDirectory()
        self.path = f"{self.dir.name}/data.csv"

    def tearDown(self):
        self.dir.cleanup()

    def test_valid_load(self):
        write_csv(self.path, [("a", 1, 0.7, 2.0), ("b", 0, 0.2, 1.5)])
        records = load_csv(self.path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0], Record("a", 1, 0.7, 2.0))

    def test_empty_dataset_rejected(self):
        write_csv(self.path, [])
        with self.assertRaises(DataError):
            load_csv(self.path)

    def test_duplicate_id_rejected(self):
        write_csv(self.path, [("a", 1, 0.5, 1.0), ("a", 0, 0.4, 1.0)])
        with self.assertRaises(DataError):
            load_csv(self.path)

    def test_bad_label_rejected(self):
        write_csv(self.path, [("a", 2, 0.5, 1.0)])
        with self.assertRaises(DataError):
            load_csv(self.path)

    def test_probability_out_of_range_rejected(self):
        write_csv(self.path, [("a", 1, 1.5, 1.0)])
        with self.assertRaises(DataError):
            load_csv(self.path)

    def test_non_finite_probability_rejected(self):
        write_csv(self.path, [("a", 1, "nan", 1.0)])
        with self.assertRaises(DataError):
            load_csv(self.path)

    def test_negative_weight_rejected(self):
        write_csv(self.path, [("a", 1, 0.5, -1.0)])
        with self.assertRaises(DataError):
            load_csv(self.path)


class DisjointTests(unittest.TestCase):
    def test_shared_id_rejected(self):
        val = [Record("a", 1, 0.5, 1.0)]
        test = [Record("a", 0, 0.5, 1.0)]
        with self.assertRaises(DataError):
            check_disjoint(val, test)

    def test_disjoint_ok(self):
        check_disjoint([Record("a", 1, 0.5, 1.0)], [Record("b", 0, 0.5, 1.0)])


if __name__ == "__main__":
    unittest.main()
