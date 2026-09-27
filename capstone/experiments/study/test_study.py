"""Guard pairing, artifact identity and resumable campaign denominators."""
import json
from pathlib import Path
import tempfile
import unittest

import study


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = dict(applications={'mruby': {'version': 'test'}}, suites=[dict(
            id='suite', application='mruby', cases=['lists'], source={'revision': 'a'*40})])
        self.plan = study.make_plan(self.catalog, 'outer-malloc', ['suite'], 2, 7)
        self.key = 'suite/lists@upstream'
        reference = self.root/'reference'
        reference.write_text('OK\n')
        data = self.root/'input'
        data.write_text('fixed workload')
        binding = dict(source={'revision': 'a'*40}, application_version='test',
            profile='outer-malloc', parameters={'work': 'upstream-default'}, adaptation='fixture',
            oracle_reference=str(reference), oracle_reference_sha256=study.file_hash(reference),
            oracle={'expected_stdout': 'OK\n', 'expected_phases': ['startup', 'exit']},
            input_sha256={'script': study.file_hash(data)}, arms={})
        for arm in study.ARMS:
            name = study.platform(arm) if arm.startswith('cheribsd') else arm
            binary = self.root/name
            binary.write_text(name)
            manifest = self.root/(name+'.json')
            manifest.write_text(json.dumps(dict(application='mruby', heap='sublet' if arm == 'capstone-sublet' else 'level0',
                nested='none', image_sha256=study.file_hash(binary), allocations=True, allocations_sha256='observer')))
            binding['arms'][arm] = dict(binary=str(binary), build_manifest=str(manifest),
                inputs={'script': str(data)}, resources={'observer_bytes': 100},
                qualification_evidence='fixture-correctness',
                point=dict(application='mruby', environment={}, argv=['/tmp/app'], files={str(binary): '/tmp/app'}))
        self.bindings = {self.key: binding}

    def test_complete_reproducible_plan_and_explicit_profile(self):
        self.assertEqual(len(self.plan['cells']), 8)
        self.assertEqual(self.plan, study.make_plan(self.catalog, 'outer-malloc', ['suite'], 2, 7))
        other = study.make_plan(self.catalog, 'nested', ['suite'], 2, 7)
        self.assertNotEqual(self.plan['plan_id'], other['plan_id'])
        self.plan['cells'].pop()
        with self.assertRaises(ValueError): study.check_plan(self.plan)

    def test_missing_workload_keeps_denominator(self):
        points, blocked = study.emit(self.plan, {}, 'capstone', {})
        self.assertEqual(points, [])
        self.assertIn(self.key, blocked)
        self.assertEqual(sum(r['counts']['unqualified'] for r in study.summary(self.plan, {}, {})), 8)

    def test_poisoncap_is_a_distinct_matched_platform(self):
        plan = study.make_plan(self.catalog, 'nested', ['suite'], 2, 7, comparison='nested-poisoncap')
        self.assertEqual(len(plan['cells']), 8)
        self.assertEqual({c['platform'] for c in plan['cells']}, {'capstone', 'poisoncap'})
        self.assertEqual({c['arm'] for c in plan['cells']}, set(study.COMPARISONS['nested-poisoncap']))
        self.assertEqual(sum(r['counts']['unqualified'] for r in study.summary(plan, {}, {})), 8)
        with self.assertRaises(ValueError):
            study.make_plan(self.catalog, 'outer-malloc', ['suite'], 2, 7, comparison='nested-poisoncap')
        with self.assertRaises(ValueError): study.platform('invented-arm')

    def test_poisoncap_cannot_reuse_default_cheribsd_qualification(self):
        plan = study.make_plan(self.catalog, 'nested', ['suite'], 2, 7, comparison='nested-poisoncap')
        for guest in ('capstone', 'poisoncap'):
            points, blocked = study.emit(plan, {}, guest, {})
            self.assertEqual(points, [])
            self.assertIn('not yet qualified', blocked[self.key])
        with self.assertRaisesRegex(ValueError, 'not yet qualified'):
            study.emit(plan, self.bindings, 'poisoncap', {})

    def poisoncap_fixture(self):
        plan = study.make_plan(self.catalog, 'nested', ['suite'], 2, 7,
                               comparison='nested-poisoncap')
        source = self.root/'input'
        reference = self.root/'ppm'
        reference.write_bytes(b'P6\n\x00\xff\x01')
        evidence = self.root/'evidence'
        evidence.write_text('complete nested fixture')
        platform_files = {}
        for name in ('capstone_qemu', 'capstone_kernel', 'poisoncap_qemu',
                     'poisoncap_kernel', 'poisoncap_libc'):
            path = self.root/name
            path.write_text(name)
            platform_files[name] = dict(path=str(path), sha256=study.file_hash(path))
        binding = dict(comparison='nested-poisoncap', source={'revision': 'a'*40},
            application_version='test', profile='nested',
            parameters={'work': 'upstream-default'}, adaptation='GC slot observer fixture',
            nested_allocator='mruby-gc', oracle_reference=str(reference),
            oracle_reference_sha256=study.file_hash(reference),
            oracle={'expected_stdout_sha256': study.file_hash(reference),
                    'expected_stdout_bytes': reference.stat().st_size,
                    'expected_phases': ['startup', 'exit']},
            input_sha256={'script': study.file_hash(source)},
            platform_files=platform_files, arms={})
        for arm in study.COMPARISONS['nested-poisoncap']:
            poison = arm.startswith('poisoncap')
            name = 'poisoncap' if poison else arm
            binary = self.root/name
            binary.write_text(name)
            manifest = self.root/(name+'.json')
            manifest.write_text(json.dumps(dict(app='mruby' if poison else None,
                application='mruby', platform='poisoncap' if poison else 'capstone',
                nested_gc='poisoncap' if poison else None,
                heap='level0', nested='mruby' if arm == 'capstone-sublet' else 'none',
                gc_gaps=True, image_sha256=study.file_hash(binary))))
            guest_source = '/tmp/ao.rb' if poison else '/mnt/host/study/input'
            point = dict(application='mruby', environment={},
                argv=(['/tmp/mruby', guest_source, '8'] if poison else [guest_source, '8']),
                files={str(source): guest_source})
            if poison:
                point['files'][str(binary)] = '/tmp/mruby'
                point['nested_allocator'] = 'mruby-gc'
            binding['arms'][arm] = dict(binary=str(binary), build_manifest=str(manifest),
                inputs={'script': str(source)}, resources={'observer_bytes': 8320},
                qualification_evidence=dict(path=str(evidence),
                    sha256=study.file_hash(evidence)), point=point)
        return plan, {self.key: binding}

    def test_qualified_poisoncap_binding_emits_only_real_nested_arms(self):
        plan, bindings = self.poisoncap_fixture()
        cap, blocked = study.emit(plan, bindings, 'capstone', {})
        poison, blocked_poison = study.emit(plan, bindings, 'poisoncap', {})
        self.assertEqual((len(cap), len(poison), blocked, blocked_poison), (4, 4, {}, {}))
        self.assertTrue(all(p['gc_gaps'] and p['expected_stdout_bytes'] == 6 for p in cap))
        self.assertEqual({(p['arm'], p['expected_min_sweeps']) for p in poison},
                         {('poisoncap-spatial', 0), ('poisoncap-temporal', 1)})
        self.assertEqual(len({p['study_artifact']['binary_sha256'] for p in poison}), 1)

    def test_poisoncap_binding_rejects_unmatched_binary_and_workload(self):
        plan, bindings = self.poisoncap_fixture()
        item = bindings[self.key]['arms']['poisoncap-temporal']
        item['point']['argv'][-1] = '16'
        with self.assertRaises(ValueError): study.emit(plan, bindings, 'poisoncap', {})
        item['point']['argv'][-1] = '8'
        item['point']['files'].pop(str(self.root/'input'))
        with self.assertRaises(ValueError): study.emit(plan, bindings, 'poisoncap', {})

    def test_poisoncap_binding_rejects_changed_platform_or_oracle(self):
        plan, bindings = self.poisoncap_fixture()
        kernel = self.root/'poisoncap_kernel'
        kernel.write_text('different kernel')
        with self.assertRaises(ValueError): study.emit(plan, bindings, 'capstone', {})
        kernel.write_text('poisoncap_kernel')
        reference = self.root/'ppm'
        reference.write_bytes(b'changed PPM')
        with self.assertRaises(ValueError): study.emit(plan, bindings, 'poisoncap', {})

    def test_explicit_size_matrix_changes_plan_and_denominator(self):
        matrix = {'suite/lists': {'small': {'operations': 100}, 'large': {'operations': 1000}}}
        plan = study.make_plan(self.catalog, 'nested', ['suite'], 2, 7, matrix)
        self.assertEqual(len(plan['cells']), 16)
        self.assertNotEqual(plan['plan_id'], self.plan['plan_id'])
        with self.assertRaises(ValueError):
            study.make_plan(self.catalog, 'nested', ['suite'], 2, 7, {})

    def test_missing_control_is_not_qualified(self):
        del self.bindings[self.key]['arms']['cheribsd-revocation-off']
        with self.assertRaises(ValueError): study.emit(self.plan, self.bindings, 'capstone', {})

    def test_outer_malloc_cannot_be_labeled_nested(self):
        self.plan = study.make_plan(self.catalog, 'nested', ['suite'], 2, 7)
        self.bindings[self.key]['profile'] = 'nested'
        with self.assertRaises(ValueError): study.emit(self.plan, self.bindings, 'capstone', {})

    def test_changed_input_or_binary_is_rejected(self):
        for file in ('input', 'capstone'):
            path = self.root/file
            original = path.read_bytes()
            path.write_bytes(b'changed')
            with self.subTest(file=file), self.assertRaises(ValueError):
                study.emit(self.plan, self.bindings, 'capstone', {})
            path.write_bytes(original)

    def test_cheribsd_on_off_cannot_change_workload(self):
        self.bindings[self.key]['arms']['cheribsd-revocation-off']['point']['argv'].append('extra')
        with self.assertRaises(ValueError): study.emit(self.plan, self.bindings, 'cheribsd', {})

    def test_observer_identity_is_shared(self):
        path = self.root/'capstone.json'
        manifest = study.read(path)
        manifest['allocations_sha256'] = 'changed'
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError): study.emit(self.plan, self.bindings, 'capstone', {})

    def test_oracle_is_backed_by_preserved_output(self):
        self.bindings[self.key]['oracle']['expected_stdout'] = 'different\n'
        with self.assertRaises(ValueError): study.emit(self.plan, self.bindings, 'capstone', {})

    def recorded_failure(self):
        points, _ = study.emit(self.plan, self.bindings, 'capstone', {})
        row = dict(point=points[0], repetition=0, status='signal')
        path = self.root/'runs.jsonl'
        path.write_text(json.dumps(row)+'\n')
        return path, row

    def test_resume_preserves_failure_without_retry(self):
        path, row = self.recorded_failure()
        rows = study.attempts(self.plan, [path])
        remaining, _ = study.emit(self.plan, self.bindings, 'capstone', rows)
        self.assertEqual(len(remaining), 3)
        self.assertNotIn(row['point']['id'], [p['id'] for p in remaining])
        summary = study.summary(self.plan, rows, self.bindings)
        self.assertEqual(sum(r['counts'].get('signal', 0) for r in summary), 1)

    def test_duplicate_or_foreign_attempts_are_rejected(self):
        path, row = self.recorded_failure()
        with self.assertRaises(ValueError): study.attempts(self.plan, [path, path])
        row['point']['study']['plan_id'] = 'another-plan'
        path.write_text(json.dumps(row)+'\n')
        with self.assertRaises(ValueError): study.attempts(self.plan, [path])

    def test_resume_cannot_mix_changed_bindings(self):
        path, _ = self.recorded_failure()
        rows = study.attempts(self.plan, [path])
        self.bindings[self.key]['parameters']['operations'] = 200
        with self.assertRaises(ValueError): study.emit(self.plan, self.bindings, 'capstone', rows)


if __name__ == '__main__':
    unittest.main()
