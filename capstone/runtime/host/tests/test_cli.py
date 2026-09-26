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
            (state / "config.json").write_text(json.dumps({"port": 2222}))
            arguments = ["/mnt/host/app.dom", "", "two words", "line\nbreak",
                         "'quoted'", "$(touch should-not-exist)", "; exit 98"]
            seen = []

            def execute(command):
                words = shlex.split(command[-1])
                self.assertEqual(words[:3], ["exec", "capstone-exec", "--"])
                # Exercise a real POSIX shell too, without executing capstone-exec.
                capture = "python3 -c 'import json,sys;print(json.dumps(sys.argv[1:]))' "
                result = subprocess.check_output(["sh", "-c", capture +
                    shlex.join(words[3:])], text=True)
                seen.extend(json.loads(result))
                return 139

            with patch.object(cli, "running", return_value=True), patch.object(cli.subprocess, "call", execute):
                rc = cli.main(["--state", directory, "run", "--", *arguments])
            self.assertEqual(seen, arguments)
            self.assertEqual(rc, 139)

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
