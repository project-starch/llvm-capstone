"""Host-side file staging, fingerprints and serialized QEMU execution for ports."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def stage_run(results, prefix, inputs):
    """Keep a separate copy of every input for every attempt, including failures."""
    results = Path(results).resolve()
    results.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=prefix, dir=results))
    share = run / "share"
    share.mkdir()
    for name, source in inputs.items():
        if Path(name).name != name:
            raise ValueError("Staged input names must be basenames")
        shutil.copy(source, share / name)
    return run, {name: digest(share / name) for name in inputs}


def run_guest(run, command, done, *, env=None, timeout_multiplier=12, lock_timeout=60):
    """Run one guest command; callers interpret their own application output."""
    env = dict(os.environ if env is None else env, CAPSTONE_REPO_ROOT=str(REPO_ROOT))
    lock = Path(
        env.get("CAPSTONE_QEMU_LOCK", str(Path.home() / ".capstone-locks/qemu.lock"))
    )
    lock.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [
            "flock",
            "-x",
            "-w",
            str(lock_timeout),
            str(lock),
            sys.executable,
            str(REPO_ROOT / "capstone/tests/runtime-qemu/run-domain-smoke.py"),
            "--share-dir",
            str(Path(run) / "share"),
            "--log-file",
            str(Path(run) / "serial.log"),
            "--kernel-arg",
            "cma=512M",
            "--timeout-multiplier",
            str(timeout_multiplier),
            "--guest-command",
            command,
            "--success-marker",
            done,
        ],
        env=env,
    )


def write_checksums(directory):
    directory = Path(directory)
    (directory / "SHA256SUMS").write_text(
        "".join(
            f"{digest(p)}  {p.name}\n"
            for p in sorted(directory.iterdir())
            if p.is_file() and p.name != "SHA256SUMS"
        )
    )


def write_replay_verdict(run, *, passed, runner_exit, **details):
    """Common result envelope; each port still owns its execution oracle.

    A trace hash binds the result to the staged bytes when a validated trace
    sidecar exists. Dedicated lifetime fixtures and deliberate malformed-input
    controls need not be trace replays and keep that association absent.
    """
    run = Path(run)
    trace = None
    sidecar = run / "trace.json"
    if sidecar.exists():
        inspection = json.loads(sidecar.read_text())
        if (
            inspection.get("schema") == "capstone.trace-inspection/v1"
            and inspection.get("validation", {}).get("complete") is True
        ):
            trace = inspection["trace"]
        elif passed:
            raise ValueError("cannot pass a replay whose trace validation failed")
    write_json(
        run / "verdict.json",
        {
            **details,
            "schema": "capstone.replay-verdict/v1",
            "passed": bool(passed),
            "runner_exit": runner_exit,
            "trace": trace,
        },
    )
