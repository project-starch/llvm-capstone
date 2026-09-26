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
                        "share": "/tmp/share", "memory": "4G",
                        "environment": {"CAPSTONE_REV_NODES": "65536"}}
            (state / "config.json").write_text(json.dumps({"identity": identity}))
            with patch.object(cli, "running", return_value=False), patch.object(cli, "start", return_value=0) as start:
                self.assertEqual(cli.main(["--state", directory, "restart"]), 0)
                args, received = start.call_args.args
                self.assertEqual(received, state)
                self.assertEqual(args.environment, identity["environment"])
                self.assertEqual(args.rootfs, Path("/tmp/rootfs"))
                self.assertEqual(args.port, 0)
                self.assertFalse(hasattr(args, "launcher"))

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
