#!/usr/bin/env python3
"""Source reuse and disk refusal regressions (no compiler or benchmark builds)."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('census', Path(__file__).with_name('milestone-census.py'))
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


class CensusTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('CAPSTONE_TMP_ROOT'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_disk_reserve(self):
        with patch.object(census.shutil, 'disk_usage') as usage:
            usage.return_value.free = 15 * 1024**3 - 1
            with self.assertRaisesRegex(ValueError, 'less than 15 GiB'):
                census.disk(self.root)
            usage.return_value.free += 1
            census.disk(self.root)

    def test_compiler_disk_reserve(self):
        with patch.object(census.shutil, 'disk_usage') as usage:
            usage.return_value.free = 20 * 1024**3 - 1
            with self.assertRaisesRegex(ValueError, 'less than 20 GiB'):
                census.disk(self.root, compiler=True)

    def test_perl_slots_are_consistent_and_other_arms_unchanged(self):
        arms = {'platform': {'cpu_props': ['x-capstone-supervisor-slots=1']},
                'arms': {'cap': {'cpu_props': ['x-capstone-supervisor-slots=32'],
                                 'module_params': 'debug=1 supervisor_slots=32'},
                         'rv64': {}}}
        perl = census.benchmark_arms(arms, '500.perlbench_r')
        self.assertEqual(perl['platform']['cpu_props'], ['x-capstone-supervisor-slots=2'])
        self.assertEqual(perl['arms']['cap']['cpu_props'], [])
        self.assertEqual(perl['arms']['cap']['module_params'], 'debug=1 supervisor_slots=2')
        self.assertEqual(perl['arms']['rv64']['module_params'], 'supervisor_slots=2')
        for arm in perl['arms'].values():
            self.assertEqual(arm['share_model'], 'mapped-xattr')
            self.assertEqual(arm['stage_image'], 'tmpfs')
        self.assertEqual(arms['platform']['cpu_props'], ['x-capstone-supervisor-slots=1'])
        self.assertEqual(census.benchmark_arms(arms, '557.xz_r'), arms)
        self.assertEqual(set(census.SLICES), set(census.BENCHES))

    def test_staged_perl_path_survives_child_prefix(self):
        wrapper = Path(__file__).with_name('capstone-run.sh').read_text()
        block = wrapper.split('  case "${words[0]}" in\n', 1)[1].split('  esac', 1)[0]
        block = 'case "${words[0]}" in\n' + block + 'esac\n'
        staged = '/tmp/stage-0001-perlbench_r_base.fixture'
        guest = '/mnt/host/cache/SPEC17/benchspec/CPU/500.perlbench_r/run/run_base_test_fixture.0000'
        for original in ('../exe/perlbench_r', '/mnt/host/absolute/perlbench_r'):
            output = subprocess.check_output(['bash', '-c',
                'words=("$1"); staged=$2\n' + block + 'printf "%s" "${words[0]}"',
                'fixture', original, staged], text=True)
            if original.startswith('/'):
                self.assertEqual(output, staged)
            else:
                self.assertFalse(output.startswith('/'))
                self.assertEqual(os.path.normpath(guest + '/' + output), staged)
                self.assertEqual(os.path.normpath(guest + '/t/../' + output), staged)
        # The old absolute program word fails specifically after t/TEST adds ../.
        self.assertNotEqual(os.path.normpath(guest + '/t/../' + staged), staged)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True).strip()

    def commit(self):
        self.git('add', '.')
        self.git('-c', 'user.name=test', '-c', 'user.email=fixture.invalid',
                 'commit', '-qm', 'fixture')
        return self.git('rev-parse', 'HEAD')

    def test_reuse_tracks_inputs_but_not_launcher_or_gates(self):
        self.git('init', '-q')
        for path in census.COMPILER + census.SDK:
            dest = self.root / path
            if dest.suffix:
                dest.parent.mkdir(parents=True, exist_ok=True)
            else:
                dest.mkdir(parents=True, exist_ok=True)
                dest /= 'input.c'
            dest.write_text('input\n')
        virt = self.root / census.VIRTUAL
        virt.mkdir(parents=True)
        (virt / 'heap.c').write_text('heap\n')
        first = census.identities(self.root, self.commit())
        (virt / 'gates').mkdir()
        (virt / 'gates/test.py').write_text('gate\n')
        (virt / 'README.md').write_text('docs\n')
        (virt / 'exec.c').write_text('launcher\n')
        (virt / 'test-registry.c').write_text('launcher test\n')
        (virt / 'build-adapter.sh').write_text('adapter build\n')
        self.assertEqual(first, census.identities(self.root, self.commit()))
        (virt / 'vm.h').write_text('new SDK header\n')
        header = census.identities(self.root, self.commit())
        self.assertNotEqual(first, header)
        (virt / 'heap.c').write_text('changed heap\n')
        second = census.identities(self.root, self.commit())
        self.assertNotEqual(header, second)
        (self.root / 'llvm/input.c').write_text('changed compiler\n')
        self.assertNotEqual(second, census.identities(self.root, self.commit()))

    def test_spec_copy_is_private_and_keeps_suite_metadata(self):
        source = self.root / 'source'
        exe = source / 'benchspec/CPU/502.gcc_r/exe/app'
        exe.parent.mkdir(parents=True)
        exe.write_text('original')
        (source / 'benchspec/CPU/other').mkdir()
        (exe.parent.parent / 'run').mkdir()
        target = self.root / 'target'
        census.copy_spec(source, target)
        copied = target / exe.relative_to(source)
        copied.write_text('changed')
        self.assertEqual(exe.read_text(), 'original')
        self.assertTrue((target / 'benchspec/CPU/other').exists())
        self.assertFalse((target / 'benchspec/CPU/502.gcc_r/run').exists())

    def test_plain_verification_metadata_is_required(self):
        kit, tree = self.root / 'kit', self.root / 'tree'
        for path in (kit, tree):
            (path / 'config').mkdir(parents=True)
            (path / 'config/riscv-plain.cfg').write_text('flags=baseline\n')
        with self.assertRaisesRegex(ValueError, 'missing executable'):
            census.check_plain_config(kit, tree)
        (tree / 'config/riscv-plain.cfg').write_text('flags=baseline\n__HASH__\nexehash=fixture\n')
        census.check_plain_config(kit, tree)
        (kit / 'config/riscv-plain.cfg').write_text('flags=changed\n')
        with self.assertRaisesRegex(ValueError, 'config changed'):
            census.check_plain_config(kit, tree)


if __name__ == '__main__':
    unittest.main()
