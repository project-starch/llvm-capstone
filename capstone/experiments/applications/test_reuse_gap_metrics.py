import unittest

from reuse_gap_metrics import parse_reuse_gap


class ReuseGapMetricsTest(unittest.TestCase):
    def test_valid_report_and_reconciliation(self):
        line = ('PYM_REUSE_GAP attempts=6 issues=5 releases=3 reuses=2 '
                'distinct=3 capacity=8 error=0 bins=1,1,' + ','.join(['0'] * 30))
        report = parse_reuse_gap(line, 'PYM_REUSE_GAP')
        self.assertEqual(report['bins'][:2], [1, 1])
        self.assertEqual(report['reuses'], 2)
        self.assertEqual(parse_reuse_gap('backend> ' + line, 'PYM_REUSE_GAP'), report)

    def test_rejects_missing_duplicate_and_error(self):
        line = ('PG_REUSE_GAP attempts=6 issues=5 releases=3 reuses=2 '
                'distinct=3 capacity=8 error=0 bins=1,1,' + ','.join(['0'] * 30))
        for text in ('', line + '\n' + line, line.replace('error=0', 'error=1'),
                     line.replace('reuses=2', 'reuses=3'),
                     line + ' reuses=2',
                     line.replace('capacity=8', 'capacity=7')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_reuse_gap(text, 'PG_REUSE_GAP')


if __name__ == '__main__':
    unittest.main()
