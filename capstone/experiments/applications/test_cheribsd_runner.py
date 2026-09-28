#!/usr/bin/env python3
"""Reject incomplete application output, policy mismatches and broken accounting."""
import importlib.util
from pathlib import Path
import unittest
import shlex
import tempfile

spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('cheribsd-run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.point = dict(expected_stdout='OK\n', expected_phases=['exit'], revocation=1)
        self.metrics = ('EXP-CHERI phase=exit revocation=1 heap_error=0 shadow_error=0 '
                        'allocated=32 active=64 resident=128\nEXP-GUEST-EXIT 0\n')

    def test_valid(self):
        self.assertEqual(runner.verdict(self.point, 0, 'OK\n', self.metrics), 'pass')

    def test_valid_disabled_process(self):
        self.point['revocation'] = 0
        self.assertEqual(runner.verdict(self.point, 0, 'OK\n', self.metrics.replace('revocation=1', 'revocation=0')), 'pass')

    def test_reject_false_passes(self):
        cases = [(255, 'OK\n', self.metrics),
                 (0, 'wrong\n', self.metrics),
                 (0, 'OK\n', self.metrics.replace('revocation=1', 'revocation=0')),
                 (0, 'OK\n', self.metrics.replace('shadow_error=0', 'shadow_error=14')),
                 (0, 'OK\n', self.metrics.replace('allocated=32', 'allocated=256')),
                 (0, 'OK\n', self.metrics.replace('allocated=32', 'allocated=-1')),
                 (0, 'OK\n', self.metrics.replace('allocated=32', 'allocated=x')),
                 (0, 'OK\n', self.metrics.replace('allocated=32', '')),
                 (0, 'OK\n', self.metrics.replace('phase=exit', 'phase=startup')),
                 (0, 'OK\n', self.metrics.replace('EXP-GUEST-EXIT 0', '')),
                 (0, 'OK\n', self.metrics.replace('EXIT 0', 'EXIT 162')),
                 (0, 'OK\n', self.metrics+'EXP-GUEST-EXIT 0\n')]
        for rc, out, err in cases:
            with self.subTest(rc=rc, out=out, err=err):
                self.assertNotEqual(runner.verdict(self.point, rc, out, err), 'pass')

    def test_ffmpeg_pool_requires_complete_reuse_and_phase_ledgers(self):
        point = dict(self.point, application='ffmpeg', arm='poisoncap-temporal',
                     nested_allocator='ffmpeg-pool', mode=2, revocation=0,
                     expected_phases=['before-0', 'released-0'])
        metric = ('EXP-CHERI phase=before-0 revocation=0 heap_error=0 shadow_error=0 '
                  'allocated=32 active=64 resident=128\n')
        stderr = (metric + metric.replace('before-0', 'released-0') +
                  'FFPOOL-POLICY mode=2 payload_reservation=4194304\n' +
                  'FFPOOL-MEM phase=before-0 payload_used=0\n' +
                  'FFPOOL-MEM phase=released-0 payload_used=16\n' +
                  'FF2-GAP-TOTAL issues=3 reuses=2 observer=128\n' +
                  ''.join(f'FF2-GAP pair={i} a={2 if i == 0 else 0} b=0\n'
                          for i in range(16)) + 'EXP-GUEST-EXIT 0\n')
        self.assertEqual(runner.verdict(point, 0, 'OK\n', stderr), 'pass')
        # A short quarantined workload can legitimately have no reissues.
        no_reuse = stderr.replace('reuses=2', 'reuses=0').replace('pair=0 a=2', 'pair=0 a=0')
        self.assertEqual(runner.verdict(point, 0, 'OK\n', no_reuse), 'pass')
        for broken in (stderr.replace('mode=2', 'mode=0'),
                       stderr.replace('reuses=2', 'reuses=3'),
                       stderr.replace('FF2-GAP pair=15', 'FF2-GAP pair=14'),
                       stderr.replace('FFPOOL-MEM phase=released-0',
                                      'FFPOOL-MEM phase=before-0')):
            self.assertEqual(runner.verdict(point, 0, 'OK\n', broken), 'bad-inner-metrics')

    def test_cpython_requires_matching_inner_policy_report(self):
        point = dict(self.point, application='cpython', arm='poisoncap-temporal',
                     nested_allocator='cpython-pymalloc', mode=1, revocation=0)
        stderr = (self.metrics.replace('revocation=1', 'revocation=0') +
                  'PYM_INTERPRETER_POLICY mode=1 payload_reservation=67108864 '
                  'metadata_reservation=16777216\n'
                  'PYM_POISONCAP mode=1 sweeps=2 poison_bytes=16 clear_bytes=16 '
                  'zeroed_bytes=16\n')
        self.assertEqual(runner.verdict(point, 0, 'OK\n', stderr), 'pass')
        for broken in (stderr.replace('mode=1', 'mode=0'),
                       stderr.replace('sweeps=2', 'sweeps=0'),
                       stderr.replace('metadata_reservation=16777216',
                                      'metadata_reservation=0')):
            self.assertEqual(runner.verdict(point, 0, 'OK\n', broken),
                             'bad-inner-metrics')


class PolicyTests(unittest.TestCase):
    def point(self, arm, enabled):
        return dict(arm=arm, revocation=enabled, environment={}, argv=['/tmp/app', 'a b'])

    def test_default_remains_unmodified(self):
        self.assertEqual(runner.policy_environment(self.point('cheribsd-default', 1)), {})

    def test_published_ffmpeg_policy_requires_outer_revocation(self):
        point = dict(self.point('poisoncap-temporal', 1), application='ffmpeg',
                     nested_allocator='ffmpeg-pool', mode=2,
                     nested_policy='published-sqlite-thresholds-corrected-v1')
        point['argv'] = ['/tmp/ffmpeg', 'input.mkv', '1', '2']
        self.assertEqual(runner.policy_environment(point), {'_RUNTIME_REVOCATION_ENABLE': '1'})
        with self.assertRaises(ValueError):
            runner.policy_environment(dict(point, revocation=0))

    def test_published_policy_rejects_unreported_sweeps_and_threshold_changes(self):
        point = dict(nested_allocator='mruby-gc',
                     nested_policy='published-sqlite-thresholds-corrected-v1')
        report = ('MRB_GC_STUDY policy=1 quarantine_limit=4096 minimum_held=16777216 '
                  'full_drains=3 threshold_drains=0 sweeps=3 quarantine=2 peak_quarantine=4096\n')
        self.assertTrue(runner.published_policy_valid(point, report))
        for bad in ('', report.replace('sweeps=3', 'sweeps=4'),
                    report.replace('minimum_held=16777216', 'minimum_held=1048576'),
                    report.replace('peak_quarantine=4096', 'peak_quarantine=4097')):
            self.assertFalse(runner.published_policy_valid(point, bad))
        self.assertFalse(runner.published_policy_valid(dict(nested_allocator='mruby-gc'), report))

    def test_on_off_are_isolated_process_switches(self):
        for enabled, suffix in ((1, 'on'), (0, 'off')):
            point = self.point('cheribsd-revocation-'+suffix, enabled)
            command = shlex.split(runner.application_command(point, 90))
            switch = '_RUNTIME_REVOCATION_' + ('ENABLE' if enabled else 'DISABLE') + '=1'
            self.assertIn(switch, command)
            self.assertIn('-i', command)
            self.assertEqual(command[-1], 'a b')

    def test_ffmpeg_transferred_policy_rejects_eager_reuse_and_hidden_sweeps(self):
        point = dict(nested_allocator='ffmpeg-pool',
                     nested_policy='published-sqlite-thresholds-corrected-v1')
        report = ('FF2_POISONCAP policy=1 quarantine_limit=4096 minimum_held=16777216 '
                  'full_drains=0 threshold_drains=1 teardown_drains=2 sweeps=3 '
                  'qcount=1 quarantine=16 held=32 peak_held=16777216 '
                  'peak_quarantine=4194304 legacy_reuse_drains=0\n')
        self.assertTrue(runner.published_policy_valid(point, report))
        for bad in (report.replace('legacy_reuse_drains=0', 'legacy_reuse_drains=1'),
                    report.replace('sweeps=3', 'sweeps=4'),
                    report.replace('qcount=1', 'qcount=4097'),
                    report.replace('quarantine=16 ', 'quarantine=48 ')):
            self.assertFalse(runner.published_policy_valid(point, bad))

    def test_policy_labels_and_overrides_cannot_disagree(self):
        bad = [self.point('cheribsd-default', 0), self.point('cheribsd-revocation-off', 1),
               self.point('unknown', 1)]
        for name in ('_RUNTIME_REVOCATION_DISABLE', 'MALLOC_CONF', '_RUNTIME_ABORT_DISABLE'):
            point = self.point('cheribsd-revocation-on', 1)
            point['environment'][name] = '1'
            bad.append(point)
        for point in bad:
            with self.subTest(point=point), self.assertRaises(ValueError):
                runner.policy_environment(point)

    def test_ffmpeg_poisoncap_uses_explicit_nested_mode_and_disabled_outer_revocation(self):
        for arm, mode in (('poisoncap-spatial', 0), ('poisoncap-temporal', 2)):
            point = self.point(arm, 0)
            point.update(application='ffmpeg', nested_allocator='ffmpeg-pool',
                         mode=mode, argv=['/tmp/ffmpeg', '/tmp/input.mkv', '1', str(mode)])
            env = runner.policy_environment(point)
            self.assertEqual(env, {'_RUNTIME_REVOCATION_DISABLE': '1'})
            for value in (1, 3):
                broken = dict(point, mode=value)
                with self.assertRaises(ValueError):
                    runner.policy_environment(broken)
            with self.assertRaises(ValueError):
                runner.policy_environment(dict(point, revocation=1))

    def test_cpython_poisoncap_mode_is_runner_owned(self):
        for arm, mode in (('poisoncap-spatial', 0), ('poisoncap-temporal', 1)):
            point = self.point(arm, 0)
            point.update(application='cpython', nested_allocator='cpython-pymalloc',
                         mode=mode)
            env = runner.policy_environment(point)
            self.assertEqual(env['PYM_POISONCAP_MODE'], str(mode))
            self.assertEqual(env['_RUNTIME_REVOCATION_DISABLE'], '1')
            with self.assertRaises(ValueError):
                runner.policy_environment(dict(point, mode=1-mode))
            with self.assertRaises(ValueError):
                runner.policy_environment(dict(point, environment={'PYM_POISONCAP_MODE': str(mode)}))


class GuestPanicTests(unittest.TestCase):
    def test_serial_panic_is_distinct_from_application_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            serial = Path(tmp)/'serial.log'
            self.assertIsNone(runner.guest_panic(serial))
            serial.write_bytes(b'boot\r\r\nStarting sshd.\r\r\n'
                               b'panic: Poison probe missing page 0x41400480\r\r\n')
            self.assertEqual(runner.guest_panic(serial),
                             'panic: Poison probe missing page 0x41400480')


if __name__ == '__main__':
    unittest.main()
