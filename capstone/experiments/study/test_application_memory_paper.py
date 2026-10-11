"""Guard scientific denominators and fail closed on incomplete plot inputs."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('paper_plot',
    Path(__file__).with_name('plot-application-memory-paper.py'))
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class MeasurementGuards(unittest.TestCase):
    def test_denominator_includes_first_issues(self):
        bins = [2, 3] + [0]*30
        values = plot.cdf(bins, 10)
        self.assertEqual(list(plot.EDGES[:3]), [1, 3, 7])
        self.assertEqual(list(values[:3]), [20, 50, 50])
        self.assertEqual(values[-1], 50)  # Do not renormalize to reused issues.

    def test_overflow_and_bad_counts_are_rejected(self):
        for bins, issues in [([0]*31, 10), ([-1, 2]+[0]*30, 10),
                             ([11]+[0]*31, 10), ([0]*31+[1], 10),
                             ([0]*32, 0)]:
            with self.subTest(bins=bins, issues=issues), self.assertRaises(ValueError):
                plot.cdf(bins, issues)

    @staticmethod
    def rows():
        return [dict(arm=arm, rep=rep, issues=10, reuses=5, bins=[2, 3]+[0]*30)
                for arm in plot.ARMS for rep in range(3)]

    def test_missing_and_duplicate_process_are_rejected(self):
        rows = self.rows()
        for bad in (rows[:-1], rows[:-1]+[copy.deepcopy(rows[0])]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plot.checked_cell(bad)

    def test_histogram_reconciliation_and_demand_are_checked(self):
        for key, value in [('issues', 11), ('reuses', 4)]:
            rows = self.rows()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                plot.checked_cell(rows)

    def test_variation_cannot_silently_be_plotted_as_one_curve(self):
        rows = self.rows()
        rows[0]['bins'] = [1, 4]+[0]*30
        with self.assertRaisesRegex(ValueError, 'repetitions differ'):
            plot.checked_cell(rows)


if __name__ == '__main__':
    unittest.main()
