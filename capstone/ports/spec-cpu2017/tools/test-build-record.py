#!/usr/bin/env python3
"""Synthetic source-selection/build-record regression; no SPEC or target compiler."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
BENCH = "500.perlbench_r"
spec = importlib.util.spec_from_file_location("build_record", HERE / "build-record.py")
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)


def put(path, value, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    if executable:
        path.chmod(0o755)


def json_file(path, value):
    put(path, json.dumps(value, indent=2) + "\n")


class BuildRecordTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="spec-build-recipe-", dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.kit, self.tree, self.sdk = [self.root / name for name in ("kit", "SPEC17", "sdk")]
        for name in ("build.sh", "build-record.py"):
            destination = self.kit / "tools" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(HERE / name, destination)
        destination = self.kit / "config/capstone-virtual.cfg"
        destination.parent.mkdir()
        shutil.copy2(HERE.parent / "config/capstone-virtual.cfg", destination)
        put(self.kit / "tools/build-host-ldecod.sh", '#!/bin/sh\nprintf "synthetic decoder\\n" > "$2"\n')
        put(self.kit / "tools/spec-extract.py", 'print("synthetic original MANIFEST verified")\n')
        self.old = self.package("capstone", "old synthetic source\n")
        self.new = self.package("capstone_provenance_v2", "new synthetic source\n")
        self.manifest = self.kit / "manifest.lock.json"
        json_file(self.manifest, {"workloads": {"benchmarks": [BENCH]}, "arms": {},
                                 "srcalt": {BENCH: self.old[0]}, "srcalt_sha256": {BENCH: self.old[1]}})
        self.manifest_bytes = self.manifest.read_bytes()
        self.overlay = self.root / "selection.json"
        self.selection = {"schema_version": 1, "base_manifest_sha256": record.digest(self.manifest),
                          "source_alternatives": {BENCH: {"name": "capstone_provenance_v2",
                          "package": self.new[0], "sha256": self.new[1], "object_count": 1,
                          "replacement_sha256": {"fixture.c": hashlib.sha256(b"new synthetic source\n").hexdigest()}}}}
        json_file(self.overlay, self.selection)
        self.tree.mkdir()
        (self.tree / "config").mkdir()
        put(self.tree / "version.txt", "synthetic\n")
        put(self.tree / "shrc", f'export PATH="{self.tree}/bin:$PATH"\n')
        for name, command in (("specxz", "xz"), ("spectar", "tar")):
            put(self.tree / "bin" / name, f'#!/bin/sh\nexec {command} "$@"\n', True)
        put(self.tree / "bin/runcpu", '''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
args=sys.argv[1:]; defs={args[i+1].split('=',1)[0]:args[i+1].split('=',1)[1] for i,x in enumerate(args) if x=='--define'}
assert os.environ.get('CAPSTONE_SDK') == defs['sdk']
root=Path.cwd(); bench='500.perlbench_r'; label=defs['arm_label']; name=defs['perl_srcalt']
base=root/'benchspec/CPU'/bench; build=base/'build'/('build_base_'+label+'.0000'); build.mkdir(parents=True)
source=(base/'src/src.alt'/name/'fixture.c').read_bytes(); (build/'fixture.c').write_bytes(source)
(build/'fixture.o').write_bytes(b'object:'+source); (build/'perlbench_r').write_bytes(b'binary:'+source)
cc=str(Path(defs['sdk'])/'capstone-cc'); (build/'make.out').write_text(cc+' -O2 -c -o fixture.o fixture.c\\n'+cc+' -o perlbench_r fixture.o\\n')
exe=base/'exe'; exe.mkdir(); (exe/('perlbench_r_base.'+label)).write_bytes((build/'perlbench_r').read_bytes())
(root/'invoked.json').write_text(json.dumps(args))
''', True)
        put(self.sdk / "capstone-cc", "#!/bin/sh\necho synthetic compiler\n", True)
        config = {"profile": "virtual", "heap_lifetimes": True, "heap_scrub_free": False,
                  "cc": str(self.sdk / "capstone-cc"), "musl": str(self.sdk / "musl")}
        for name in ("ld", "runtime", "libc", "builtins", "script"):
            config[name] = str(self.sdk / name)
            put(self.sdk / name, "synthetic " + name)
        for name in ("arch/capstone64", "arch/generic", "obj/include", "include"):
            put(self.sdk / "musl" / name / "fixture.h", "/* synthetic */\n")
        json_file(self.sdk / "sdk.json", config)
        self.out = self.root / "record"

    def package(self, name, source):
        content = self.root / ("package-" + name)
        put(content / f"benchspec/CPU/{BENCH}/src/src.alt/{name}/fixture.c", source)
        relative = f"packages/{BENCH}.{name}.tar.xz"
        path = self.kit / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:xz") as archive:
            archive.add(content / "benchspec", arcname="benchspec")
        return relative, record.digest(path)

    def build(self, overlay=True, accepted=True):
        env = os.environ.copy()
        env.pop("CAPSTONE_SPEC_SOURCE_SELECTION", None)
        env.pop("BENCHMARKS", None)
        env["CAPSTONE_SPEC_PACKAGES"] = str(self.kit)
        env["CAPSTONE_SDK"] = str(self.root / "unselected-sdk")
        if overlay:
            env["CAPSTONE_SPEC_SOURCE_SELECTION"] = str(self.overlay)
        result = subprocess.run(["bash", str(self.kit / "tools/build.sh"), str(self.kit),
                                 str(self.tree), str(self.sdk), str(self.out), "native_fixture"],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode == 0, accepted, result.stdout + result.stderr)
        self.assertEqual(self.manifest.read_bytes(), self.manifest_bytes)
        return result

    def test_default_build_keeps_historical_selection_and_freezes_single_arm(self):
        self.build(overlay=False)
        frozen = json.loads((self.out / "frozen-native_fixture.json").read_text())[BENCH]
        self.assertEqual(frozen["source_alternative"]["name"], "capstone")
        self.assertEqual(set(frozen["objects"]), {"fixture.o"})
        self.assertEqual(len(frozen["compile_commands"]), 1)
        self.assertEqual(len(frozen["link_commands"]), 1)
        self.assertIn("cc", frozen["sdk_inputs"]["artifacts"])

    def test_explicit_source_selection_builds_new_source_and_records_identity(self):
        self.build()
        frozen = json.loads((self.out / "frozen-native_fixture.json").read_text())[BENCH]
        self.assertEqual(frozen["source_alternative"], self.selection["source_alternatives"][BENCH])
        self.assertIn("perl_srcalt=capstone_provenance_v2", json.loads((self.tree / "invoked.json").read_text()))
        build = self.tree / frozen["build_directory"]
        self.assertEqual((build / "fixture.o").read_bytes(), b"object:new synthetic source\n")
        self.assertEqual(frozen["sources"]["fixture.c"], record.digest(build / "fixture.c", "sha512"))

    def test_wrong_package_hash_is_rejected_before_install_or_build(self):
        self.selection["source_alternatives"][BENCH]["sha256"] = "0" * 64
        json_file(self.overlay, self.selection)
        self.build(accepted=False)
        self.assertFalse((self.tree / "invoked.json").exists())

    def test_package_must_install_the_selected_name(self):
        self.selection["source_alternatives"][BENCH].update(package=self.old[0], sha256=self.old[1])
        json_file(self.overlay, self.selection)
        self.build(accepted=False)
        self.assertFalse((self.tree / "invoked.json").exists())

    def test_existing_same_label_build_is_rejected_before_reuse(self):
        (self.tree / f"benchspec/CPU/{BENCH}/build/build_base_native_fixture.0000").mkdir(parents=True)
        self.build(accepted=False)
        self.assertFalse((self.tree / "invoked.json").exists())

    def test_overlay_rejects_wrong_base_and_non_source_fields(self):
        for key, value in (("base_manifest_sha256", "0" * 64), ("platform", {})):
            changed = dict(self.selection)
            changed[key] = value
            json_file(self.overlay, changed)
            with self.assertRaises(ValueError):
                record.select(self.manifest, self.overlay)

    def test_subset_requires_only_selected_source_packages(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["workloads"]["benchmarks"].append("502.gcc_r")
        manifest["srcalt"]["502.gcc_r"] = "packages/unavailable-gcc.tar.xz"
        manifest["srcalt_sha256"]["502.gcc_r"] = "2" * 64
        json_file(self.manifest, manifest)
        selected = record.select(self.manifest, benchmarks=[BENCH])
        self.assertEqual(list(selected["source_alternatives"]), [BENCH])
        self.assertEqual(selected["benchmarks"], [BENCH])
        self.assertEqual(len(record.select(self.manifest)["source_alternatives"]), 2)

    def test_published_overlay_matches_qualified_package_and_unchanged_lock(self):
        port = HERE.parent.parent / "perl/provenance"
        manifest = HERE.parent / "manifest.lock.json"
        selected = record.select(manifest, port / "srcalt-selection.json", [BENCH])
        qualified = json.loads((port / "qualification.json").read_text())["source"]
        item = selected["source_alternatives"][BENCH]
        self.assertEqual(item["sha256"], qualified["package_sha256"])
        self.assertEqual(item["replacement_sha256"], qualified["replacement_after_sha256"])
        self.assertEqual(item["object_count"], 69)
        self.assertEqual(selected["base_manifest_sha256"], "ee5662584d4aea719e1b1f8fcea4bf892fad105ff94d005ee035dc45f18c49e9")

    def test_overlay_rejects_unsafe_identity_and_paths(self):
        for key, value in (("name", "../bad"), ("package", "../outside.tar.xz")):
            changed = json.loads(json.dumps(self.selection))
            changed["source_alternatives"][BENCH][key] = value
            json_file(self.overlay, changed)
            with self.assertRaises(ValueError):
                record.select(self.manifest, self.overlay)

    def test_wrong_replacement_source_cannot_freeze(self):
        self.selection["source_alternatives"][BENCH]["replacement_sha256"]["fixture.c"] = "0" * 64
        json_file(self.overlay, self.selection)
        self.build(accepted=False)
        self.assertFalse((self.out / "frozen-native_fixture.json").exists())

    def test_wrong_object_count_cannot_freeze(self):
        self.selection["source_alternatives"][BENCH]["object_count"] = 69
        json_file(self.overlay, self.selection)
        self.build(accepted=False)
        self.assertFalse((self.out / "frozen-native_fixture.json").exists())

    def test_sdk_changes_and_uncompiled_objects_cannot_freeze(self):
        self.build()
        selected = json.loads((self.out / "source-selection.json").read_text())
        sdk_before = json.loads((self.out / "sdk-inputs-native_fixture.json").read_text())
        put(self.sdk / "libc", "changed SDK")
        with self.assertRaisesRegex(ValueError, "SDK inputs changed"):
            record.freeze(self.tree, "native_fixture", [BENCH], selected, self.sdk, sdk_before)
        build = self.tree / f"benchspec/CPU/{BENCH}/build/build_base_native_fixture.0000"
        with (build / "make.out").open("a") as stream:
            stream.write("capstone-cc missing.o -o another\n")
        with self.assertRaisesRegex(ValueError, "no recorded fresh compilation"):
            record.freeze(self.tree, "native_fixture", [BENCH], selected, self.sdk, record.sdk_inputs(self.sdk))


if __name__ == "__main__":
    unittest.main()
