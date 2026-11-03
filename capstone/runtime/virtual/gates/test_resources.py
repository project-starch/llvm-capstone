#!/usr/bin/env python3
"""Process tests for arm leases and dispatch; no QEMU or application checks mocked as passes.

Run on a host with both documented snapshot CPU sets available. All test coordination
files live in a temporary HOME, separate from the host's production locks.
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "arm-resources.py"
WORKER = """
import json, os, pathlib, subprocess, sys, time
p = pathlib.Path(sys.argv[1])
p.write_text(json.dumps(dict(start=time.time(), env={k: v for k, v in os.environ.items()
    if k.startswith('GATE_') or k in ('TMPDIR', 'XDG_CACHE_HOME')}, cpus=sorted(os.sched_getaffinity(0)))))
time.sleep(float(sys.argv[2]))
d = json.loads(p.read_text()); d['end'] = time.time(); p.write_text(json.dumps(d))
"""


class Resources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = dict(os.environ, HOME=str(self.root), PHASE="gates")
        for key in ("GATE_SLOT", "GATE_PARALLEL", "QEMU_LOCK", "GATE_ARM_PARENT"):
            self.env.pop(key, None)
        self.children = []

    def tearDown(self):
        for p in self.children:
            if p.poll() is None:
                p.terminate()
            p.communicate(timeout=10)
        self.tmp.cleanup()

    def command(self, name, duration="0", slot="0"):
        env = dict(self.env, GATE_SLOT=slot)
        cmd = [sys.executable, str(RESOURCES), str(self.root / name), sys.executable,
               "-c", WORKER, str(self.root / (name + ".json")), duration]
        return cmd, env

    def start(self, name, duration="30", slot="0"):
        cmd, env = self.command(name, duration, slot)
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.children.append(p)
        self.ready(name, p)
        return p

    def ready(self, name, p):
        deadline = time.monotonic() + 10
        while not (self.root / (name + ".json")).exists():
            if p.poll() is not None:
                self.fail(p.communicate())
            if time.monotonic() > deadline:
                self.fail("child did not start")
            time.sleep(0.02)

    def run_one(self, name, slot="0", **extra):
        cmd, env = self.command(name, slot=slot)
        env.update(extra)
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10)

    def test_two_slots_and_real_driver_collision(self):
        self.start("first")
        other = self.run_one("second", "1")
        self.assertEqual(other.returncode, 0, other.stderr)
        a, b = (json.loads((self.root / (n + ".json")).read_text()) for n in ("first", "second"))
        self.assertFalse(set(a["cpus"]) & set(b["cpus"]))
        self.assertNotEqual(a["env"]["TMPDIR"], b["env"]["TMPDIR"])
        self.assertEqual(a["env"]["GATE_MEMCACHED_PORT"], "21300")
        self.assertEqual(b["env"]["GATE_MEMCACHED_PORT"], "21301")
        env = dict(self.env, GATE_SLOT="0", TREE=str(self.root), PLATFORM=str(self.root),
                   LLVM=str(self.root), HOOKS=str(self.root / "hooks.ko"), CROSS_GCC="unused")
        collision = subprocess.run(["bash", str(HERE / "run-arm.sh"), str(self.root / "collision")],
                                   env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(collision.returncode, 75, collision.stderr)
        self.assertIn("resource busy: gate-slot-0.lock", collision.stderr)
        self.assertFalse((self.root / "collision").exists())

    def test_output_collision_across_slots(self):
        self.start("same")
        r = self.run_one("same", "1")
        self.assertEqual(r.returncode, 75)
        self.assertIn("resource busy: .arm.lock", r.stderr)

    def test_bad_configuration_and_busy_port(self):
        for slot in ("-1", "2", "01", "x", "999999999999"):
            r = self.run_one("bad", slot)
            self.assertEqual(r.returncode, 75)
            self.assertFalse((self.root / "bad").exists())
        r = self.run_one("private", QEMU_LOCK=str(self.root / "private.lock"))
        self.assertEqual(r.returncode, 75)
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 21300)); server.listen()
            r = self.run_one("port")
            self.assertEqual(r.returncode, 75)
            self.assertFalse((self.root / "port").exists())

    def test_interrupt_releases_lease_and_child_group(self):
        p = self.start("interrupt")
        child_pid = int(subprocess.check_output(["ps", "--ppid", str(p.pid), "-o", "pid="], text=True))
        p.send_signal(signal.SIGTERM)
        p.communicate(timeout=10)
        self.assertEqual(p.returncode, 75)
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)
        r = self.run_one("reused")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_killed_supervisor_keeps_child_lease(self):
        p = self.start("orphan", duration="2")
        p.kill()
        # Its child still owns the inherited lease even without its supervisor.
        r = self.run_one("replacement")
        self.assertEqual(r.returncode, 75, r.stderr)
        self.assertIn("resource busy: gate-slot-0.lock", r.stderr)
        p.communicate(timeout=10)
        r = self.run_one("after-orphan")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_command_status_is_preserved(self):
        r = subprocess.run([sys.executable, str(RESOURCES), str(self.root / "exit"),
                            "bash", "-c", "exit 23"], env=self.env, capture_output=True)
        self.assertEqual(r.returncode, 23)

    def test_dispatch_refuses_unsupported_and_conflicting_configuration(self):
        driver = self.root / "driver.sh"
        marker = self.root / "executed"
        driver.write_text(f'#!/bin/bash\ntouch "{marker}"\n')
        cmd = ["bash", str(HERE / "gate-queue.sh"), "bash", str(driver), str(self.root / "out")]
        env = dict(self.env, GATE_PARALLEL="1")
        r = subprocess.run(cmd, env=env, capture_output=True)
        self.assertEqual(r.returncode, 2)
        self.assertFalse(marker.exists())
        driver.write_text('# GATE_SLOT_PROTOCOL=1\n' + driver.read_text())
        for extra in (dict(GATE_SLOTS="0"), dict(GATE_SLOTS="3"), dict(GATE_SLOT="0")):
            r = subprocess.run(cmd, env=dict(env, **extra), capture_output=True)
            self.assertEqual(r.returncode, 2)
            self.assertFalse(marker.exists())

    def test_dispatch_parallel_and_legacy_exclusion(self):
        driver = self.root / "driver.sh"
        driver.write_text('#!/bin/bash\n# GATE_SLOT_PROTOCOL=1\nexec python3 "$TEST_RESOURCES" "$1" '
                          'python3 "$TEST_WORKER" "$1.json" 1\n')
        worker = self.root / "worker.py"; worker.write_text(WORKER)
        env = dict(self.env, GATE_PARALLEL="1", GATE_SLOTS="2", TEST_RESOURCES=str(RESOURCES),
                   TEST_WORKER=str(worker))
        for name in ("a", "b"):
            p = subprocess.Popen(["bash", str(HERE / "gate-queue.sh"), "bash", str(driver), str(self.root / name)],
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.children.append(p)
        for name, p in zip(("a", "b"), self.children):
            self.ready(name, p)
        pending = subprocess.Popen(["bash", str(HERE / "gate-queue.sh"), "bash", str(driver), str(self.root / "pending")],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.children.append(pending)
        legacy = subprocess.Popen(["bash", str(HERE / "gate-queue.sh"), "bash", str(driver), str(self.root / "legacy")],
                                  env=dict(env, GATE_PARALLEL="0"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.children.append(legacy)
        for p in self.children:
            _, err = p.communicate(timeout=15)
            self.assertEqual(p.returncode, 0, err)
        a, b, c = (json.loads((self.root / (n + ".json")).read_text()) for n in ("a", "b", "legacy"))
        self.assertLess(max(a["start"], b["start"]), min(a["end"], b["end"]))
        self.assertGreater(c["start"], max(a["end"], b["end"]))
        self.assertNotIn("GATE_SLOT", c["env"])
        pending_result = json.loads((self.root / "pending.json").read_text())
        self.assertGreater(pending_result["start"], c["end"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
