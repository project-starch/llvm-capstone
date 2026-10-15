"""The application SDK is a build input: a rebuilt SDK must invalidate a port.

capstone-cc compiles against the SDK's musl headers and links its runtime,
libc, builtins and link script without any of them appearing on a command line
CMake generates. The defect this guards is silent: before
capstone-application-sdk.cmake, `cmake --build` on a port's existing build tree
reported "no work to do" after the SDK had been rebuilt and kept executables
built against the previous runtime. test_a_rebuilt_sdk_is_invisible_at_head is
that negative control, run against the toolchain as committed at HEAD~ of this
change -- without it, a test that only asserts a rebuild cannot distinguish the
registration from CMake rebuilding for an unrelated reason.

A fake SDK stands in for a built one: the toolchain reads sdk.json, runs the
compiler it names with -print-resource-dir, and hashes the files it names, so a
host clang and six placeholder files exercise every path. Nothing here needs a
Capstone toolchain, which is why it runs with the host tests.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

TOOLCHAINS = Path(__file__).resolve().parents[1] / "cmake/toolchains"
REPO = Path(__file__).resolve().parents[4]
SDK_FILES = ("libapplication-runtime.a", "libc-capstone.a",
             "libcapstone-application-builtins.a", "link.ld")


def tool(*names):
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


class SdkDependencies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmake = tool("cmake")
        cls.ninja = tool("ninja")
        cls.clang = tool("clang", "clang-18")
        cls.ar = tool("llvm-ar", "llvm-ar-18", "ar")
        cls.ranlib = tool("llvm-ranlib", "llvm-ranlib-18", "ranlib")
        if not all((cls.cmake, cls.ninja, cls.clang, cls.ar, cls.ranlib)):
            raise unittest.SkipTest("needs cmake, ninja, clang and ar on the host")

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="sdk-depends-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.sdk = self.root / "sdk"
        self.sdk.mkdir()
        # CMAKE_AR and CMAKE_RANLIB come from the toolchain's LLVM build
        # directory, which a host-only run does not have.
        llvm = self.root / "llvm/bin"
        llvm.mkdir(parents=True)
        (llvm / "llvm-ar").symlink_to(self.ar)
        (llvm / "llvm-ranlib").symlink_to(self.ranlib)
        self.llvm = llvm.parent
        driver = self.sdk / "capstone-cc"
        driver.write_text(f'#!/bin/sh\nexec {self.clang} "$@"\n')
        driver.chmod(0o755)
        for name in SDK_FILES:
            (self.sdk / name).write_text("first\n")
        (self.sdk / "sdk.json").write_text(json.dumps({
            "version": 1, "application_abi": 2, "profile": "virtual",
            "allocator": "mallocng", "codegen": ["-mllvm", "-capstone-gp-free"],
            "cc": self.clang, "ld": self.clang, "musl": str(self.sdk / "musl"),
            "script": str(self.sdk / "link.ld"),
            "runtime": str(self.sdk / "libapplication-runtime.a"),
            "libc": str(self.sdk / "libc-capstone.a"),
            "builtins": str(self.sdk / "libcapstone-application-builtins.a"),
        }) + "\n")
        self.port = self.root / "port"
        self.port.mkdir()
        # Two targets: one with a source of its own, and one standing in for a
        # port that links prebuilt upstream objects and so has no compile step
        # of its own to retrigger.
        (self.port / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.25)\n"
            "project(sdk_depends LANGUAGES C)\n"
            "add_executable(with-source main.c)\n"
            "add_subdirectory(nested)\n")
        (self.port / "main.c").write_text("int main(void) { return 0; }\n")
        (self.port / "nested").mkdir()
        (self.port / "nested/CMakeLists.txt").write_text(
            "add_executable(in-subdirectory other.c)\n")
        (self.port / "nested/other.c").write_text("int main(void) { return 0; }\n")

    def configure(self, build, toolchain=TOOLCHAINS / "capstone-application.cmake"):
        return subprocess.run(
            [self.cmake, "-S", self.port, "-B", build, "-G", "Ninja",
             "-DCMAKE_TOOLCHAIN_FILE=" + str(toolchain),
             "-DCAPSTONE_SDK=" + str(self.sdk),
             "-DCAPSTONE_LLVM_BUILD_DIR=" + str(self.llvm)],
            capture_output=True, text=True, cwd=self.root)

    def build(self, build):
        done = subprocess.run([self.cmake, "--build", build],
                              capture_output=True, text=True, cwd=self.root)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout

    def head_toolchain(self):
        """The toolchain as committed before this change, for the control."""
        previous = self.root / "at-head"
        (previous / "toolchains").mkdir(parents=True)
        revision = subprocess.run(
            ["git", "-C", REPO, "log", "-1", "--format=%H", "--",
             "capstone/ports/common/cmake/toolchains/capstone-application-sdk.cmake"],
            capture_output=True, text=True, check=True).stdout.strip()
        if not revision:
            raise unittest.SkipTest("the registration is not committed yet")
        for name in ("cmake/Workspace.cmake", "cmake/toolchains/capstone-application.cmake",
                     "cmake/toolchains/capstone-application-rules.cmake"):
            source = subprocess.run(
                ["git", "-C", REPO, "show",
                 f"{revision}~1:capstone/ports/common/{name}"],
                capture_output=True, text=True)
            if source.returncode:
                raise unittest.SkipTest("no pre-change toolchain to control against")
            (previous / Path(name).relative_to("cmake")).write_text(source.stdout)
        return previous / "toolchains/capstone-application.cmake"

    def rebuild_the_sdk(self):
        """What a fresh `cmake --build` of the SDK leaves behind: new archives."""
        for name in SDK_FILES:
            (self.sdk / name).write_text("second\n")

    def identity(self, build):
        graph = (Path(build) / "build.ninja").read_text()
        found = {line.split("CAPSTONE_SDK_IDENTITY=")[1].split()[0]
                 for line in graph.splitlines() if "CAPSTONE_SDK_IDENTITY=" in line}
        self.assertEqual(len(found), 1, found)
        return found.pop()

    def test_the_sdk_is_a_link_dependency_of_every_executable(self):
        build = self.root / "build"
        configured = self.configure(build)
        self.assertEqual(configured.returncode, 0, configured.stderr)
        graph = (build / "build.ninja").read_text()
        for target in ("build with-source:", "build nested/in-subdirectory:"):
            line = next(l for l in graph.splitlines() if l.startswith(target))
            for name in ("sdk.json", "capstone-cc", *SDK_FILES):
                self.assertIn(str(self.sdk / name), line, name)

    def test_a_missing_sdk_artifact_is_named_at_configure_time(self):
        (self.sdk / "libapplication-runtime.a").unlink()
        configured = self.configure(self.root / "build")
        self.assertNotEqual(configured.returncode, 0)
        self.assertIn("names a runtime that does not exist", configured.stderr)

    def test_a_rebuilt_sdk_recompiles_and_relinks(self):
        build = self.root / "build"
        self.assertEqual(self.configure(build).returncode, 0)
        self.build(build)
        self.assertIn("no work to do", self.build(build).lower())
        before = self.identity(build)
        self.rebuild_the_sdk()
        output = self.build(build)
        self.assertEqual(self.identity(build) != before, True, "SDK identity did not move")
        for expected in ("Building C object", "with-source", "in-subdirectory"):
            self.assertIn(expected, output, output)

    def test_a_rebuilt_sdk_is_invisible_at_head(self):
        build = self.root / "build-at-head"
        configured = self.configure(build, toolchain=self.head_toolchain())
        self.assertEqual(configured.returncode, 0, configured.stderr)
        self.build(build)
        self.rebuild_the_sdk()
        self.assertIn("no work to do", self.build(build).lower(),
                      "the control no longer reproduces the defect")


class SourceListing(unittest.TestCase):
    """A cache key over "every source under these paths" must never hash nothing.

    Twice a listing failed and left a libc/runtime key with no runtime source in it, so a
    change there silently reused the old runtime: rg absent on a host (2026-10-01, 163
    sources) and git ls-files in a tree export with no .git (2026-10-13, measured on p12:
    269 files in a checkout, 0 there). Swapping the tool was the fix both times and could
    not work, because the bug is that an empty listing is accepted.
    """

    ENV = REPO / "capstone/tests/capstone-test-env.sh"
    PATHS = "capstone/runtime capstone/ports/common/application"

    def listing(self, root=None):
        override = f"CAPSTONE_REPO_ROOT={root} " if root else ""
        script = (f'source "{self.ENV}" >/dev/null 2>&1; '
                  f'{override}capstone_source_listing {self.PATHS}')
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    def test_a_checkout_lists_its_sources(self):
        done = self.listing()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertGreater(len(done.stdout.split()), 200, "a checkout has hundreds of these")

    def test_a_tree_export_without_git_falls_back_to_find(self):
        root = Path(tempfile.mkdtemp(prefix="export-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for rel in ("capstone/runtime/a.c", "capstone/ports/common/application/b.sh"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text("x\n")
        self.assertFalse((root / ".git").exists())
        done = self.listing(root)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(sorted(done.stdout.split()),
                         ["capstone/ports/common/application/b.sh", "capstone/runtime/a.c"])

    def test_an_empty_listing_is_an_error_not_an_empty_key(self):
        root = Path(tempfile.mkdtemp(prefix="empty-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for rel in ("capstone/runtime", "capstone/ports/common/application"):
            (root / rel).mkdir(parents=True)
        done = self.listing(root)
        self.assertEqual(done.returncode, 1, f"an empty listing must fail: {done!r}")
        self.assertEqual(done.stdout, "")
        self.assertIn("no files under", done.stderr)

    def test_no_cache_key_pipes_a_bare_listing_into_its_digest(self):
        # The regression this guards: a bare `git ls-files ... |` inside the digest block,
        # where a non-zero exit is invisible.
        offenders = []
        for env in sorted(REPO.glob("capstone/ports/*/app/deps/env.sh")):
            text = env.read_text()
            if "git ls-files" in text and "capstone_source_listing" not in text:
                offenders.append(str(env.relative_to(REPO)))
        self.assertEqual(offenders, [], "these compute a key from an unchecked listing")
        # and the check must have looked at the two files that had the defect
        found = {str(p.relative_to(REPO)) for p in REPO.glob("capstone/ports/*/app/deps/env.sh")}
        for required in ("capstone/ports/memcached/app/deps/env.sh",
                         "capstone/ports/wireshark/app/deps/env.sh"):
            self.assertIn(required, found)


if __name__ == "__main__":
    unittest.main()
