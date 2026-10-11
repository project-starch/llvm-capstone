"""Guard the reuse denominator and reject misleading platform comparisons."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('analysis', Path(__file__).with_name('analyze-reuse.py'))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row():
    counts = dict(phase='exit', allocations=5, frees=5, unique=2, reused=3,
                  live=0, peak=64, blocks=0, peak_blocks=2, requested=160,
                  failures=0, errors=0, unknown=0, observer=24,
                  reuse1=1, reuse8=2, reuse64=0, reuse512=0, reuse4096=0, reuse_more=0)
    return dict(point=dict(application='fixture', size=1, batches=1, retained=0,
                           arm='capstone-sublet', expected_phases=['exit']),
                repetition=0, status='pass', allocations=[counts],
                memory=[dict(phase='exit', live=0)], image_sha256='image', stdout_sha256='output')


class ReuseTests(unittest.TestCase):
    def test_cdf_denominator_includes_never_reused_allocations(self):
        r = row()
        analysis.check(r)
        metrics = analysis.extract(r)['metrics']
        self.assertEqual(metrics['reuse1_fraction'], .2)
        self.assertEqual(metrics['reuse8_fraction'], .6)
        self.assertEqual(metrics['reuse_more_fraction'], .6)

    def test_failed_attempt_remains_without_metrics(self):
        r = row()
        r.update(status='signal', allocations=[], memory=[])
        result = analysis.extract(r)
        self.assertEqual(result['status'], 'signal')
        self.assertNotIn('metrics', result)

    def test_all_primary_repeats_checked(self):
        old, first, second = row(), row(), row()
        second['repetition'] = 1
        second['memory'][0]['live'] = 8
        second['image_sha256'] = 'changed'
        second['point']['environment'] = {'WORKLOAD_POLICY': 'changed'}
        checks = analysis.invariance([old], [first, second])
        self.assertEqual([r['equal'] for r in checks], [True, False])
        self.assertEqual(checks[1]['differing'], ['memory', 'image_sha256', 'point.environment'])

    def test_invalid_observation_cannot_be_invariant(self):
        for modify in ('errors', 'missing_phase'):
            r = copy.deepcopy(row())
            if modify == 'errors': r['allocations'][0]['errors'] = 1
            else: r['point']['expected_phases'].insert(0, 'startup')
            with self.subTest(modify=modify), self.assertRaises(ValueError):
                analysis.invariance([r], [row()])


if __name__ == '__main__':
    unittest.main()
