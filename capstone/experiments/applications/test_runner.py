import unittest
from run import verdict, parse_memory

class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.point = dict(expected_stdout='EXP-OK\n', expected_phases=['startup', 'exit'])
        self.err = 'EXP-MEM phase=startup live=0 peak=0 pool=4096\nEXP-MEM phase=exit live=0 peak=128 pool=4096\n'
        self.exit = dict(kind='exit', value=0)
    def check(self, rc=0, timeout=False, stdout='EXP-OK\n', stderr=None, result=None):
        return verdict(self.point, rc, timeout, stdout, self.err if stderr is None else stderr,
                       self.exit if result is None else result)
    def test_valid(self): self.assertEqual(self.check(), 'pass')
    def test_wrong_oracle(self): self.assertEqual(self.check(stdout='EXP-NO\n'), 'oracle-mismatch')
    def test_missing_completion(self): self.assertEqual(self.check(stderr=self.err.splitlines()[0]), 'missing-phases')
    def test_timeout(self): self.assertEqual(self.check(timeout=True), 'timeout')
    def test_signal(self): self.assertEqual(self.check(rc=139, result=dict(kind='signal', value=11)), 'signal')
    def test_missing_result(self): self.assertEqual(self.check(result={}), 'missing-exit-evidence')
    def test_impossible_counter(self): self.assertEqual(self.check(stderr=self.err.replace('live=0 peak=128', 'live=256 peak=128')), 'bad-metrics')
    def test_missing_counter(self): self.assertEqual(self.check(stderr=self.err.replace('live=0 ', '')), 'bad-metrics')
    def test_non_numeric_counter(self): self.assertEqual(self.check(stderr=self.err.replace('live=0', 'live=bad')), 'bad-metrics')
    def test_negative_counter(self): self.assertEqual(self.check(stderr=self.err.replace('live=0', 'live=-1')), 'bad-metrics')
    def test_postgres_error_after_oracle(self):
        point = dict(expected_values=['ok'], expected_phases=['startup', 'exit'])
        self.assertEqual(verdict(point, 0, False, '1: oracle = "ok"',
                                 self.err+'FATAL: incomplete transaction\n', self.exit), 'oracle-mismatch')

if __name__ == '__main__': unittest.main()
