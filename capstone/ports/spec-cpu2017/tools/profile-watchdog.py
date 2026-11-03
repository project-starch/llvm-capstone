#!/usr/bin/env python3
"""Bound one profile invocation and reap its detached VM (Linux only).

Run as a separate process, never inside the census worker threads: subreaper
ownership belongs to a process. PID file descriptors prevent PID reuse races.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def children():
    """Snapshot only our descendants, including adopted double-forked children."""
    parents = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                parents[int(entry.name)] = int(fields[1])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    owned = {os.getpid()}
    while True:
        more = {pid for pid, parent in parents.items() if parent in owned}
        if more <= owned:
            return owned - {os.getpid()}
        owned |= more


def signal_children(signum, records):
    for pid in children():
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            # Recheck ancestry AFTER opening the stable process handle.
            if pid not in children():
                continue
            command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
            records.append({'pid': pid, 'signal': signum,
                            'command': [x.decode(errors='replace') for x in command if x]})
            signal.pidfd_send_signal(fd, signum)
        except (FileNotFoundError, ProcessLookupError):
            pass
        finally:
            os.close(fd)


def reap():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if not pid:
                return
        except ChildProcessError:
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if not 0 < args.seconds < float('inf') or not args.command:
        ap.error('a finite positive deadline and command are required')
    command = args.command[1:] if args.command[0] == '--' else args.command
    if args.out.exists():
        ap.error('refusing an existing output directory')
    # PR_SET_CHILD_SUBREAPER: the persistent VM must reparent here, not to PID 1.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'cannot become child subreaper')
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support is required')
    interrupted = []
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda sig, frame: interrupted.append(sig))
    started = time.monotonic()
    proc = subprocess.Popen(command, start_new_session=True)
    reason = None
    while proc.poll() is None:
        if interrupted:
            reason = f'watchdog interrupted by signal {interrupted[0]}'
            break
        if time.monotonic() - started >= args.seconds:
            reason = f'wall-time limit exceeded ({args.seconds:g}s)'
            break
        time.sleep(0.1)
    rc = proc.poll()
    # Even a normally exiting shell can leave a VM behind after a failed boot.
    reap()
    remaining = children()
    if remaining and reason is None:
        reason = 'profile exited with live descendants'
    signals = []
    if remaining:
        signal_children(signal.SIGTERM, signals)
        grace = time.monotonic() + 3
        while children() and time.monotonic() < grace:
            proc.poll()
            reap()
            time.sleep(0.05)
        # Repeat to catch children born while their parents were being stopped.
        deadline = time.monotonic() + 5
        while children() and time.monotonic() < deadline:
            signal_children(signal.SIGKILL, signals)
            proc.poll()
            reap()
            time.sleep(0.05)
    proc.poll()
    reap()
    remaining = sorted(children())
    if remaining:
        reason = f'UNRESOLVED: descendants survived cleanup: {remaining}'
    if rc and reason is None:
        reason = f'profile exited {rc}'
    args.out.mkdir(parents=True, exist_ok=True)
    if reason:
        path = args.out / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        result.update(verdict='FAIL', reasons=result.get('reasons', []) + [reason])
        path.write_text(json.dumps(result, indent=2) + '\n')
    record = dict(limit_seconds=args.seconds, elapsed_seconds=time.monotonic()-started,
                  returncode=rc, reason=reason, signals=signals, remaining=remaining)
    (args.out / 'watchdog.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    return 1 if reason else 0


if __name__ == '__main__':
    raise SystemExit(main())
