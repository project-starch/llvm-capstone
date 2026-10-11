"""Regression checks for the four-arm build gate."""

import importlib.util
import hashlib
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "build_comparability", Path(__file__).with_name("check-build-comparability.py")
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture():
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    arms = {}
    for name, platform in module.ARMS.items():
        arms[name] = {
            "platform": platform,
            "upstream_sha256": digest("upstream"),
            "driver_sha256": digest("driver"),
            "workload_sha256": digest("input"),
            "runtime_sqlite_config": {"lookaside": [0, 0], "temp_store": 3},
            "compiler_sha256": digest(platform + "-compiler"),
            "target": platform + "-target",
            "vfs": platform + "-vfs",
            "base_port_patch_sha256": digest(platform + "-patch"),
            "port_source_sha256": digest(platform + "-source"),
            "protection_patch_sha256": digest(name + "-protection"),
            "binary_sha256": digest(name + "-binary"),
            "compile_argv": ["clang", "-O0", "-DSQLITE_THREADSAFE=0",
                             "-DSQLITE_ENABLE_MEMSYS5=1", "sqlite3.c"],
            "driver_compile_argv": ["clang", "-O1", "-DSQLITE_STUDY_ORACLE=1",
                                    "speedtest1.c"],
            "link_argv": ["clang", "sqlite3.o", "speedtest1.o", "-o", "app"],
        }
    arms["capstone"]["compile_argv"].append("-DSQLITE_OS_OTHER=1")
    arms["capstone-sublet"]["compile_argv"].append("-DSQLITE_OS_OTHER=1")
    return {"schema": 1, "arms": arms}


class BuildComparabilityTests(unittest.TestCase):
    def test_platform_vfs_exception_does_not_hide_sqlite_mismatch(self):
        data = fixture()
        self.assertEqual(module.audit(data), [])
        data["arms"]["poisoncap-temporal"]["compile_argv"].append(
            "-DSQLITE_DEFAULT_LOOKASIDE=1200,40"
        )
        self.assertTrue(any("SQLITE_DEFAULT_LOOKASIDE" in issue
                            for issue in module.audit(data)))

    def test_unknown_optimization_and_source_drift_fail(self):
        data = fixture()
        data["arms"]["poisoncap-spatial"]["compile_argv"].remove("-O0")
        self.assertTrue(any("optimization" in issue
                            for issue in module.audit(data)))
        data = fixture()
        data["arms"]["poisoncap-spatial"]["upstream_sha256"] = hashlib.sha256(b"other").hexdigest()
        self.assertTrue(any("upstream_sha256" in issue
                            for issue in module.audit(data)))


if __name__ == "__main__":
    unittest.main()
