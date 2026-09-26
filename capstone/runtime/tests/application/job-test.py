#!/usr/bin/env python3
"""Require real waitpid distinctions and signal forwarding from the SSH helper."""
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
with tempfile.TemporaryDirectory() as directory:
    result = Path(directory) / "result.json"
    command = [sys.argv[1], str(result), "--"]
    for script, kind, value in (("exit 139", "exit", 139), ("kill -SEGV $$", "signal", 11)):
        process = subprocess.run([*command, "sh", "-c", script], timeout=5)
        assert process.returncode == 139
        assert json.loads(result.read_text()) == {"version": 1, "kind": kind, "value": value}
        result.unlink()
    for stop in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        # The child reports readiness after installing its own handlers. Start
        # from a new interpreter so inherited Python signal policy cannot hide
        # what the job helper forwards to its process group.
        program = ("import signal,time; "
                   "[signal.signal(s,signal.SIG_DFL) for s in (1,2,15)]; "
                   "print('ready',flush=True); time.sleep(30)")
        process = subprocess.Popen([*command, sys.executable, "-c", program], stdout=subprocess.PIPE)
        assert process.stdout.readline() == b"ready\n"
        process.send_signal(stop)
        assert process.wait(timeout=5) == 128 + stop
        assert json.loads(result.read_text()) == {"version": 1, "kind": "signal", "value": stop}
        result.unlink()
        process.stdout.close()
print("waitpid exit/signal distinction and forwarding: PASS")
