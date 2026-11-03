#!/usr/bin/env python3
"""Own an arm's output and optional snapshot slot for the lifetime of its children."""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


# Match capstone_qemu_domain in capstone-test-env.sh. More domains need qualification.
SLOTS = (("A", "28-29,60-61", 21300), ("B", "30-31,62-63", 21301))


def lock(path, held):
    handle = path.open("a+b")
    held.append(handle)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"resource busy: {path.name}") from None


def main():
    out = Path(sys.argv[1]).resolve()
    command = sys.argv[2:]
    env = dict(os.environ)
    phase = env.get("PHASE", "all")
    if phase not in ("all", "build", "gates"):
        raise RuntimeError("PHASE must be all, build or gates")
    slot = env.get("GATE_SLOT")
    if slot is not None and slot not in ("0", "1"):
        raise RuntimeError("GATE_SLOT must be 0 or 1 (the two snapshot domains)")
    held = []
    child = None
    try:
        resources = dict(slot=slot, phase=phase, pid=os.getpid(), start=time.time())
        if slot is not None and phase != "build":
            domain, cpus, port = SLOTS[int(slot)]
            root = Path.home() / ".capstone-locks"
            root.mkdir(parents=True, exist_ok=True)
            qemu_lock = root / "qemu.lock"
            if env.get("QEMU_LOCK", str(qemu_lock)) != str(qemu_lock):
                raise RuntimeError("slotted arms require the global QEMU lock; unset QEMU_LOCK")
            lock(root / f"gate-slot-{slot}.lock", held)
            # A stale server or a non-gate process must cause a refusal before any gate.
            # The slot lease prevents another arm racing this check; outsiders still must
            # respect the documented port reservation for the duration of an arm.
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
            allowed = set(os.sched_getaffinity(0))
            selected = set()
            for part in cpus.split(","):
                bounds = [int(n) for n in part.split("-")]
                selected.update(range(bounds[0], bounds[-1] + 1))
            if not selected <= allowed:
                raise RuntimeError(f"slot {slot} CPUs {cpus} are not available")
            os.sched_setaffinity(0, selected)
            env.update(GATE_DOMAIN=domain, GATE_MEMCACHED_PORT=str(port))
            resources.update(domain=domain, cpus=cpus, memcached_port=port)
        else:
            env.pop("GATE_DOMAIN", None)
            env.pop("GATE_MEMCACHED_PORT", None)
        out.mkdir(parents=True, exist_ok=True)
        lock(out / ".arm.lock", held)
        # Never overwrite the record of a previous attempt with a rejected rerun.
        if (out / ("gates" if phase == "gates" else "build")).exists():
            raise RuntimeError(f"{phase} output exists: use a fresh OUT")
        private = out / (f"slot-{slot}" if slot is not None else "serial")
        for name in ("tmp", "cache", "state", "crash"):
            (private / name).mkdir(parents=True, exist_ok=True)
        env.update(TMPDIR=str(private / "tmp"), TMP=str(private / "tmp"), TEMP=str(private / "tmp"),
                   XDG_CACHE_HOME=str(private / "cache"), XDG_STATE_HOME=str(private / "state"),
                   CLANG_CRASH_DIAGNOSTICS_DIR=str(private / "crash"),
                   PYTHONDONTWRITEBYTECODE="1", CCACHE_DISABLE="1",
                   GATE_ARM_PARENT=str(os.getpid()))
        record = out / f"resources-{phase}.json"
        record.write_text(json.dumps(resources, indent=2) + "\n")
        print("gate-resources: " + json.dumps(resources, sort_keys=True), flush=True)

        def interrupted(signum, frame):
            raise InterruptedError(f"signal {signum}")

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, interrupted)
        # The arm shell also holds the leases: an uncatchable death of this
        # supervisor must not admit a replacement while that shell still runs.
        child = subprocess.Popen(command, env=env, start_new_session=True,
                                 pass_fds=tuple(handle.fileno() for handle in held))
        rc = child.wait()
        resources.update(end=time.time(), returncode=rc)
        record.write_text(json.dumps(resources, indent=2) + "\n")
        return rc
    finally:
        # Kill only the process group created above, including a native server left by
        # a failed harness. Keep leases until cleanup completes, even on interruption.
        if child is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                if sig == signal.SIGTERM:
                    time.sleep(0.2)
            child.wait()
        for handle in reversed(held):
            handle.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as error:
        print(f"gate-resources: REFUSED: {error}", file=sys.stderr)
        sys.exit(75)
