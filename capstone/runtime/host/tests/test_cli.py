import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from capstone_vm import cli


class TransportTests(unittest.TestCase):
    def test_qemu_uses_recorded_environment_on_restart(self):
        with patch.dict(os.environ, {"CAPSTONE_GP_NONLIN": "0",
                                    "CAPSTONE_TAGWATCH": "1",
                                    "UNRELATED_SETTING": "retained"}):
            environment = cli.qemu_process_environment({"CAPSTONE_GP_NONLIN": "1"})
        self.assertEqual(environment["CAPSTONE_GP_NONLIN"], "1")
        self.assertNotIn("CAPSTONE_TAGWATCH", environment)
        self.assertEqual(environment["UNRELATED_SETTING"], "retained")

    def test_arguments_round_trip_through_remote_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "assets").mkdir()
            (state / "config.json").write_text(json.dumps({"port": 2222}))
            arguments = ["/mnt/host/app.dom", "", "two words", "line\nbreak",
                         "'quoted'", "$(touch should-not-exist)", "; exit 98"]
            seen = []
            real_popen = subprocess.Popen

            def execute(command, **kwargs):
                words = shlex.split(command[-1])
                self.assertEqual(words[:2], ["exec", "env"])
                script = words[-1]
                words = shlex.split(script.split("; exec ", 1)[1])
                self.assertEqual(words[0], "capstone-job")
                self.assertEqual(words[2:5], ["--", "capstone-exec", "--"])
                result_file = state / "assets" / Path(words[1]).relative_to("/mnt/control")
                result_file.write_text('{"version":1,"kind":"exit","value":139}')
                # Exercise a real POSIX shell too, without executing capstone-exec.
                capture = "python3 -c 'import json,sys;print(json.dumps(sys.argv[1:]))' "
                with patch.object(subprocess, "Popen", real_popen):
                    result = subprocess.check_output(["sh", "-c", capture +
                        shlex.join(words[5:])], text=True)
                seen.extend(json.loads(result))
                class Process:
                    def wait(self): return 139
                    def poll(self): return 139
                return Process()

            with patch.object(cli, "running", return_value=True), patch.object(cli.subprocess, "Popen", execute):
                rc = cli.main(["--state", directory, "run", "--", *arguments])
            self.assertEqual(seen, arguments)
            self.assertEqual(rc, 139)

    def test_restart_preserves_paths_and_recorded_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            identity = {"files": {k: {"path": "/tmp/" + k, "sha256": "old"}
                        for k in ("qemu", "kernel", "firmware", "rootfs")},
                        "share": "/tmp/share", "memory": "4G", "cma_mib": 1024,
                        "process_cache_mib": 768,
                        "environment": {"CAPSTONE_REV_NODES": "65536"}}
            (state / "config.json").write_text(json.dumps({"identity": identity}))
            with patch.object(cli, "running", return_value=False), patch.object(cli, "start", return_value=0) as start:
                self.assertEqual(cli.main(["--state", directory, "restart"]), 0)
                args, received = start.call_args.args
                self.assertEqual(received, state)
                self.assertEqual(args.environment, identity["environment"])
                self.assertEqual(args.cma_mib, 1024)
                self.assertEqual(args.process_cache_mib, 768)
                self.assertEqual(args.rootfs, Path("/tmp/rootfs"))
                self.assertEqual(args.port, 0)
                self.assertFalse(hasattr(args, "launcher"))

    def test_restart_preserves_virtual_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            identity = {"files": {k: {"path": "/tmp/" + k, "sha256": "old"}
                        for k in ("qemu", "kernel", "firmware", "rootfs", "launcher", "module")},
                        "share": "/tmp/share", "memory": "2G", "cma_mib": 768,
                        "process_cache_mib": 384, "environment": {}, "profile": "virtual"}
            (state / "config.json").write_text(json.dumps({"identity": identity}))
            with patch.object(cli, "running", return_value=False), patch.object(cli, "start", return_value=0) as start:
                self.assertEqual(cli.main(["--state", directory, "restart"]), 0)
                args, _ = start.call_args.args
                self.assertEqual(args.profile, "virtual")
                self.assertEqual(args.launcher, Path("/tmp/launcher"))
                self.assertEqual(args.module, Path("/tmp/module"))

    def test_restart_preserves_exact_bounds_and_only_when_set(self):
        for recorded in (True, None):
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                identity = {"files": {k: {"path": "/tmp/" + k, "sha256": "old"}
                            for k in ("qemu", "kernel", "firmware", "rootfs", "launcher", "module")},
                            "share": "/tmp/share", "memory": "2G", "cma_mib": 768,
                            "process_cache_mib": 384, "environment": {}, "profile": "virtual"}
                if recorded:
                    identity["exact_bounds"] = True
                (state / "config.json").write_text(json.dumps({"identity": identity}))
                with patch.object(cli, "running", return_value=False), patch.object(cli, "start", return_value=0) as start:
                    self.assertEqual(cli.main(["--state", directory, "restart"]), 0)
                    args, _ = start.call_args.args
                    self.assertEqual(args.exact_bounds, bool(recorded))

    def test_restart_preserves_share_model_and_only_when_set(self):
        for recorded in ("mapped-xattr", None):
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                identity = {"files": {k: {"path": "/tmp/" + k, "sha256": "old"}
                            for k in ("qemu", "kernel", "firmware", "rootfs", "launcher", "module")},
                            "share": "/tmp/share", "memory": "2G", "cma_mib": 768,
                            "process_cache_mib": 384, "environment": {}, "profile": "virtual"}
                if recorded:
                    identity["share_model"] = recorded
                (state / "config.json").write_text(json.dumps({"identity": identity}))
                with patch.object(cli, "running", return_value=False), patch.object(cli, "start", return_value=0) as start:
                    self.assertEqual(cli.main(["--state", directory, "restart"]), 0)
                    args, _ = start.call_args.args
                    self.assertEqual(args.share_model, recorded or "none")

    def test_share_model_reaches_qemu_and_the_identity(self):
        for model in ("none", "mapped-xattr"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ("qemu", "kernel", "firmware", "rootfs"):
                    (root / name).write_text("x")
                (root / "share").mkdir()
                state = root / "s"
                state.mkdir(mode=0o700)
                real_popen = subprocess.Popen

                def popen(command, *args, **kwargs):
                    # Only QEMU's start is stopped; ssh-keygen runs for real.
                    if command[0] == str(root / "qemu"):
                        raise RuntimeError("stop")
                    return real_popen(command, *args, **kwargs)
                with patch.dict(os.environ, {"CAPSTONE_QEMU_LOCK": str(root / "lock")}), \
                     patch.object(cli.subprocess, "Popen", side_effect=popen):
                    with self.assertRaisesRegex(RuntimeError, "stop"):
                        cli.start(cli.argparse.Namespace(
                            qemu=root / "qemu", kernel=root / "kernel", firmware=root / "firmware",
                            rootfs=root / "rootfs", share=root / "share", profile="physical",
                            share_model=model, memory="2G", cma_mib=512, process_cache_mib=384,
                            port=0, boot_timeout=1), state)
                config = json.loads((state / "config.json").read_text())
                virtfs = [config["command"][i + 1] for i, word in enumerate(config["command"]) if word == "-virtfs"]
                self.assertIn(f"mount_tag=hostshare,security_model={model},", virtfs[0])
                self.assertIn("mount_tag=control,security_model=none,", virtfs[1])
                self.assertEqual(config["identity"].get("share_model"), None if model == "none" else model)

    def test_unknown_share_model_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("qemu", "kernel", "firmware", "rootfs"):
                Path(directory, name).write_text("x")
            with self.assertRaisesRegex(cli.VMError, "Share model"):
                cli.start(cli.argparse.Namespace(
                    qemu=Path(directory, "qemu"), kernel=Path(directory, "kernel"),
                    firmware=Path(directory, "firmware"), rootfs=Path(directory, "rootfs"),
                    share=Path(directory), profile="physical", share_model="passthrough", memory="2G",
                    cma_mib=512, process_cache_mib=384, port=0, boot_timeout=1), Path(directory))

    def test_exact_bounds_is_refused_on_the_physical_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("qemu", "kernel", "firmware", "rootfs"):
                Path(directory, name).write_text("x")
            with self.assertRaises(cli.VMError):
                cli.start(cli.argparse.Namespace(
                    qemu=Path(directory, "qemu"), kernel=Path(directory, "kernel"),
                    firmware=Path(directory, "firmware"), rootfs=Path(directory, "rootfs"),
                    share=Path(directory), profile="physical", exact_bounds=True, memory="2G",
                    cma_mib=512, process_cache_mib=384, port=0, boot_timeout=1), Path(directory))

    def test_up_defaults_to_the_virtual_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            platform = [f"--{name}={directory}/{name}" for name in ("qemu", "kernel", "firmware", "rootfs")]
            for extra, profile, exact_bounds in (([], "virtual", False),
                                                 (["--profile", "physical"], "physical", False),
                                                 (["--exact-bounds"], "virtual", True)):
                with patch.object(cli, "start", return_value=0) as start:
                    self.assertEqual(cli.main(["--state", directory, "up", *platform,
                                               f"--share={directory}", *extra]), 0)
                    args, _ = start.call_args.args
                    self.assertEqual((args.profile, args.exact_bounds), (profile, exact_bounds))

    def test_restart_reads_a_record_without_profile_as_physical(self):
        # Records written before the profile field existed were all physical VMs.
        with tempfile.TemporaryDirectory() as directory:
            identity = {"files": {k: {"path": "/tmp/" + k, "sha256": "old"}
                        for k in ("qemu", "kernel", "firmware", "rootfs")},
                        "share": "/tmp/share", "memory": "2G", "environment": {}}
            Path(directory, "config.json").write_text(json.dumps({"identity": identity}))
            with patch.object(cli, "running", return_value=False), patch.object(cli, "start", return_value=0) as start:
                self.assertEqual(cli.main(["--state", directory, "restart"]), 0)
                args, _ = start.call_args.args
                self.assertEqual((args.profile, args.exact_bounds), ("physical", False))

    def virtual_start(self, directory, **settings):
        for name in ("qemu", "kernel", "firmware", "rootfs", "launcher", "capstone-job", "module"):
            Path(directory, name).write_text(name)
        return cli.argparse.Namespace(
            qemu=Path(directory, "qemu"), kernel=Path(directory, "kernel"),
            firmware=Path(directory, "firmware"), rootfs=Path(directory, "rootfs"),
            share=Path(directory), memory="2G", environment={},
            cma_mib=512, process_cache_mib=384, port=0, boot_timeout=1, **settings)

    def test_virtual_start_needs_its_launcher_and_module(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.virtual_start(directory)
            with self.assertRaisesRegex(cli.VMError, "launcher and module"):
                cli.start(args, Path(directory))

    def test_virtual_start_records_exact_bounds_only_when_set(self):
        # A running VM with the same identity is reused, so the identity start() computes is
        # compared without booting: exact_bounds is in it exactly when the VM gets the mode.
        for setting, recorded in ((True, True), (False, False)):
            with tempfile.TemporaryDirectory() as directory:
                args = self.virtual_start(directory, launcher=Path(directory, "launcher"),
                                          module=Path(directory, "module"), exact_bounds=setting)
                files = {name: Path(directory, file).resolve() for name, file in
                         (("qemu", "qemu"), ("kernel", "kernel"), ("firmware", "firmware"),
                          ("rootfs", "rootfs"), ("launcher", "launcher"), ("module", "module"),
                          ("job_helper", "capstone-job"))}
                identity = {"files": {name: {"path": str(path), "sha256": cli.fingerprint(path)}
                                      for name, path in files.items()},
                            "share": str(Path(directory).resolve()), "memory": "2G", "cma_mib": 512,
                            "process_cache_mib": 384,
                            "environment": {"CAPSTONE_GP_NONLIN": "1", "CAPSTONE_REV_NODES": "65536"},
                            "profile": "virtual"}
                if recorded:
                    identity["exact_bounds"] = True
                Path(directory, "config.json").write_text(json.dumps({"port": 1, "identity": identity}))
                with patch.object(cli, "running", return_value=True), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.start(args, Path(directory)), 0)

    def test_qmp_events_and_partial_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            server = socket.socket(socket.AF_UNIX)
            server.bind(str(state / "qmp.sock"))
            server.listen()
            failures = []

            def serve():
                try:
                    connection, _ = server.accept()
                    with connection, connection.makefile("rwb", buffering=0) as stream:
                        stream.write(b'{"QMP": {}}\n')
                        for name in ("qmp_capabilities", "query-status"):
                            request = json.loads(stream.readline())
                            self.assertEqual(request["execute"], name)
                            stream.write(b'{"event":"TEST"}\n')
                            reply = json.dumps({"id": name, "return": {"running": True}}).encode() + b"\n"
                            stream.write(reply[:5])
                            stream.write(reply[5:])
                except BaseException as error:
                    failures.append(error)

            worker = threading.Thread(target=serve)
            worker.start()
            try:
                self.assertTrue(cli.qmp(state, "query-status")["running"])
            finally:
                server.close()
                worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertFalse(failures)

    def test_qmp_connect_waits_for_a_full_backlog(self):
        # QEMU's QMP listener has a backlog of one; concurrent runs must wait
        # for it, not fail with EAGAIN.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qmp.sock"
            server = socket.socket(socket.AF_UNIX)
            server.bind(str(path))
            server.listen(1)
            server.settimeout(0.2)
            stop = threading.Event()

            def serve():
                while not stop.is_set():
                    try:
                        connection, _ = server.accept()
                    except socket.timeout:
                        continue
                    stop.wait(0.05)
                    connection.close()

            worker = threading.Thread(target=serve)
            worker.start()
            outcomes = []

            def client():
                try:
                    cli.qmp_connect(path).close()
                    outcomes.append("connected")
                except OSError as error:
                    outcomes.append(error.errno)

            clients = [threading.Thread(target=client) for _ in range(6)]
            try:
                for thread in clients:
                    thread.start()
                for thread in clients:
                    thread.join(10)
            finally:
                stop.set()
                worker.join(5)
                server.close()
            self.assertEqual(outcomes, ["connected"] * 6)

    def test_paused_vm_still_owns_session(self):
        with patch.object(cli, "qmp", return_value={"running": False, "status": "paused"}):
            self.assertTrue(cli.running(Path("unused")))

    def test_unresponsive_vm_is_not_treated_as_stopped(self):
        with patch.object(cli, "qmp", side_effect=TimeoutError("unresponsive")):
            with self.assertRaises(TimeoutError):
                cli.running(Path("unused"))

    def test_private_state_required(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            state.chmod(0o755)
            with self.assertRaises(cli.VMError), cli.lock(state):
                pass

    def test_stopped_vm_is_not_restarted_by_execution(self):
        with patch.object(cli, "running", return_value=False), patch.object(cli.subprocess, "call") as execute:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["run", "program"]), 1)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
