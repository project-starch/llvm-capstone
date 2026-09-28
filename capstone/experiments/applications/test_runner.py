import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from run import verdict, parse_memory, fresh_tree_paths, prepare_fresh_tree, tree_digest

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
        point = dict(application='postgres', expected_values=['ok'],
                     expected_phases=['startup', 'exit'])
        self.assertEqual(verdict(point, 0, False, '1: oracle = "ok"',
                                 self.err+'FATAL: incomplete transaction\n', self.exit), 'oracle-mismatch')
    def test_postgres_error_after_stdout_hash(self):
        stdout = b'EXP-OK\n'
        point = dict(application='postgres', expected_stdout_sha256=hashlib.sha256(stdout).hexdigest(),
                     expected_stdout_bytes=len(stdout), expected_phases=['startup', 'exit'])
        self.assertEqual(verdict(point, 0, False, stdout.decode(),
                                 self.err+'FATAL: incomplete transaction\n', self.exit, stdout),
                         'oracle-mismatch')

class FreshTreeTest(unittest.TestCase):
    def test_each_attempt_starts_from_the_same_directory(self):
        with TemporaryDirectory() as temporary:
            share = Path(temporary)
            source = share / 'baseline'
            source.mkdir()
            (source / 'data').write_text('original')
            specification = dict(source='baseline', destination='working')
            expected = tree_digest(source)
            owned = set()
            prepare_fresh_tree(share, specification, expected, owned)
            (share / 'working/data').write_text('changed')
            (share / 'working/new').write_text('new')
            prepare_fresh_tree(share, specification, expected, owned)
            self.assertEqual((share / 'working/data').read_text(), 'original')
            self.assertFalse((share / 'working/new').exists())
            self.assertEqual((source / 'data').read_text(), 'original')

    def test_foreign_destination_is_preserved(self):
        with TemporaryDirectory() as temporary:
            share = Path(temporary)
            (share / 'baseline').mkdir()
            (share / 'working').mkdir()
            (share / 'working/keep').write_text('mine')
            with self.assertRaisesRegex(RuntimeError, 'not created by this campaign'):
                prepare_fresh_tree(share, dict(source='baseline', destination='working'),
                                   tree_digest(share / 'baseline'), set())
            self.assertEqual((share / 'working/keep').read_text(), 'mine')

    def test_escape_and_changed_source_are_rejected(self):
        with TemporaryDirectory() as temporary:
            share = Path(temporary)
            source = share / 'baseline'
            source.mkdir()
            (source / 'data').write_text('original')
            with self.assertRaises(ValueError):
                fresh_tree_paths(share, dict(source='baseline', destination='../outside'))
            (share / 'alias').symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                fresh_tree_paths(share, dict(source='baseline', destination='alias/new'))
            expected = tree_digest(source)
            (source / 'data').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                prepare_fresh_tree(share, dict(source='baseline', destination='working'),
                                   expected, set())
            self.assertFalse((share / 'working').exists())

    def test_source_symlink_is_rejected(self):
        with TemporaryDirectory() as temporary:
            share = Path(temporary)
            source = share / 'baseline'
            source.mkdir()
            (source / 'external').symlink_to('/etc/passwd')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                tree_digest(source)

if __name__ == '__main__': unittest.main()
