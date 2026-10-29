#!/usr/bin/env python3
"""Run PoisonCap's own speedtest1 on their platform, as their driver runs it: no arguments.

    run.py OUT --binary FILE [--args "..."] [--timeout S] [--platform DIR] [--port N]
           [--kernel-root DIR] [--revocation-default-off] [--pre CMD ...]
           [--after-start CMD] [--poll S] [--console-echo]

One boot of their image, single-user through platform/poisoncap_guest.py (the multi-user boot of this
build panics). The binary is copied in and run once in an empty directory, with the given
arguments (none by default: their main() configures memsys5 itself, and speedtest1's defaults are
--size 100 --testset main and a temporary database).

The libc revocation default (security.cheri.runtime_revocation_default) is as booted unless
--revocation-default-off sets it to 0 from the single-user shell, before sshd starts; it applies to
processes exec'd after it is set. --pre commands run after the copy (e.g. setting it to 1 for the run),
--after-start CMD runs 2 s after the run's shell has backgrounded speedtest1 (no handshake: it is a
delay, not a guarantee that speedtest1 has exec'd). Their memsys5 calls cheri_revoke itself, whatever
the default.

OUT/result.json, every field a reader needs to bind the run to its inputs:
  invocation, runner/binary/kernel sha256; each stage reached ('stage') and, if one failed, the error;
  the revocation sysctls before and at the run; the exit status ("no exit ..." when the run's ssh
  returned without the status marker); host seconds of the run's ssh call (not a panic timestamp);
  speedtest1's stdout/stderr, or with --poll and a dead guest the last copy and when it was taken;
  'tests' (completed lines) and 'last_started' (the last test started but not finished in that stdout,
  with the time the stdout was observed); every poll's time and outcome; the --after-start status;
  the console of the run's window, scrubbed of user@host strings and home paths, as matched lines and
  as its last 300 lines. The serial log itself stays in OUT/vm/ and is not a result.

--console-echo is NOT NEUTRAL: it pipes stdout through tee, and with it their speedtest1 died at
start-up on a poison fault (2 of 2 runs, 2026-10-25) where it does not with stdout to a file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform'))
import poisoncap_guest  # noqa: E402
from records import CONSOLE, scrub  # noqa: E402

TEST = re.compile(r'^\s*(\d{3}) - (.*?)\.{3,}\s*([0-9.]+)s\s*$', re.M)
STARTED = re.compile(r'^\s*(\d{3}) - (.*?)\.{3,}', re.M)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--args', default='', help='speedtest1 arguments; none by default, as their driver')
    p.add_argument('--timeout', type=int, default=4 * 3600)
    poisoncap_guest.add_platform_arguments(p, port=10089)
    p.add_argument('--revocation-default-off', action='store_true',
                   help='set the libc revocation default to 0 from the single-user shell, before sshd starts. '
                        'With it left on, this build panics (debug kernel) or stops answering (NODEBUG) during '
                        'the copy-in, before speedtest1 runs')
    p.add_argument('--kernel-root', type=Path,
                   help='boot KERNEL_ROOT/boot/kernel/kernel instead of the platform rootfs\'s (e.g. '
                        'platform/build-nodebug-kernel.sh\'s); the disk image is the platform\'s either way')
    p.add_argument('--pre', action='append', default=[],
                   help='a guest command run (and recorded) after the copy, before speedtest1; repeatable')
    p.add_argument('--after-start', metavar='CMD',
                   help='run CMD in the guest 2 s after speedtest1 was backgrounded (a delay, no handshake), '
                        'e.g. setting the libc default back to 0 so that processes exec\'d later (the --poll '
                        'copies) run without libc revocation')
    p.add_argument('--poll', type=int, default=0, metavar='S',
                   help='every S seconds copy out.txt (and after-start.txt) out of the guest over a second ssh; '
                        'stdout stays a file')
    p.add_argument('--console-echo', action='store_true',
                   help='also tee speedtest1\'s output to the guest console. NOT NEUTRAL, see the module text')
    a = p.parse_args()
    if a.console_echo and a.after_start:
        p.error('--console-echo and --after-start cannot be combined')
    return a


def run_command(a, command):
    """The guest shell line that runs speedtest1 in /tmp/st and prints __EXIT=<status>: with stdout and
    stderr to files, with --after-start's command 2 s after it was backgrounded, or (--console-echo)
    through tee to the console, with speedtest1's own status as __RC."""
    if a.console_echo:
        return (f'cd /tmp/st && chmod 755 speedtest1 && {{ timeout {a.timeout} {command}; '
                f'echo "__RC=$?"; }} 2>&1 | tee out.txt > /dev/console; echo "__EXIT=$?"')
    if a.after_start:
        # cd on its own: `a && b && (c) &` would background the whole && list.
        return (f'cd /tmp/st || exit 9; chmod 755 speedtest1; '
                f'( timeout {a.timeout} {command} > out.txt 2> err.txt; echo "__EXIT=$?" ) & '
                f'sleep 2; {a.after_start} > after-start.txt 2>&1; echo "__AFTER=$?" >> after-start.txt; '
                f'sysctl security.cheri.runtime_revocation_default >> after-start.txt 2>&1; wait')
    return (f'cd /tmp/st && chmod 755 speedtest1 && timeout {a.timeout} {command} '
            f'> out.txt 2> err.txt; echo "__EXIT=$?"')


class Poller:
    """Every INTERVAL s, copies out.txt and after-start.txt out of the guest over a second ssh; keeps the
    last successful copy (snapshot) and every attempt's time and outcome (polls)."""

    def __init__(self, guest, interval):
        self.guest, self.interval = guest, interval
        self.polls = []
        self.snapshot = {'text': '', 'after': '', 'at': None}
        self.stopped = threading.Event()
        self.thread = None

    def start(self, t0):
        self.t0 = t0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=90)

    def _loop(self):
        while not self.stopped.wait(self.interval):
            t = round(time.monotonic() - self.t0, 1)
            try:
                q = self.guest.ssh('cat /tmp/st/out.txt; echo __SPLIT__; cat /tmp/st/after-start.txt 2>/dev/null',
                                   timeout=60)
            except subprocess.TimeoutExpired:
                self.polls.append([t, 'timeout'])
                continue
            if q.returncode == 0 and '__SPLIT__' in q.stdout:
                out, after = q.stdout.split('__SPLIT__\n', 1) if '__SPLIT__\n' in q.stdout else (q.stdout, '')
                if not self.stopped.is_set():
                    self.snapshot.update(text=out, after=after, at=t)
                self.polls.append([t, 'ok'])
            else:
                self.polls.append([t, f'rc={q.returncode}'])


def finish_record(res, a, serial, before, started, exit_seen, poller):
    """What the record gets whether or not the run finished: the polls; without an exit status, the last
    copied stdout; the console of the run's window (or of the whole boot if the run never started);
    the completed tests and the last one started."""
    if poller:
        res['polls'] = poller.polls
        if started and not exit_seen:
            res['stdout'] = poller.snapshot['text']
            res['stdout_from'] = f"last successful poll at {poller.snapshot['at']} s"
            if a.after_start:
                res['after_start'] = {'command': a.after_start, 'observed': res['stdout_from'],
                                      'output': poller.snapshot['after'].strip()}
    if serial.exists():
        text = serial.read_bytes().decode(errors='replace').replace('\r', '')
        window = text[before:] if started else text
        res['console_scope'] = 'run window' if started else 'whole boot (the run never started)'
        lines = [scrub(l.strip()) for l in window.splitlines()]
        res['console'] = [l for l in lines if CONSOLE.search(l)][:200]
        res['console_tail'] = lines[-300:]
        if a.console_echo and started and not exit_seen:
            res['stdout'], res['stdout_from'] = scrub(window), 'serial console'
    out = res.get('stdout') or ''
    res['tests'] = [{'id': int(i), 'name': n.strip(), 'seconds': float(s)} for i, n, s in TEST.findall(out)]
    started_tests = STARTED.findall(out)
    if started_tests and (not res['tests'] or int(started_tests[-1][0]) != res['tests'][-1]['id']):
        res['last_started'] = {'id': int(started_tests[-1][0]), 'name': started_tests[-1][1].strip(),
                               'observed': res.get('stdout_from', 'at exit')}


def main():
    a = parse_args()
    (a.out / 'vm').mkdir(parents=True, exist_ok=False)
    kernel_root = a.kernel_root or a.platform / 'work/output/rootfs-riscv64-purecap'
    guest = poisoncap_guest.new_guest(a.platform, a.out / 'vm', a.port, kernel_root=kernel_root,
                                      revocation_default_off=a.revocation_default_off)
    serial = a.out / 'vm/serial.log'
    command = './speedtest1' + (' ' + a.args if a.args else '')
    res = {'invocation': scrub(' '.join(sys.argv)), 'command': command, 'timeout': a.timeout,
           'runner_sha256': sha256(__file__), 'binary_sha256': sha256(a.binary),
           'kernel_sha256': sha256(kernel_root / 'boot/kernel/kernel'),
           'kernel_root': scrub(str(a.kernel_root)) if a.kernel_root else 'platform rootfs',
           'revocation_default_off': a.revocation_default_off,
           'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': []}
    poller = Poller(guest, a.poll) if a.poll else None
    t0 = None
    m = None
    before = 0
    try:
        res['stage'].append('boot')
        guest.start()
        res['stage'].append('sysctl')
        res['sysctl'] = guest.ssh('sysctl security.cheri.runtime_revocation_default '
                                  'kern.features.cheri_revoke').stdout.strip()
        res['stage'].append('copy')
        guest.ssh('mkdir -p /tmp/st && rm -rf /tmp/st/*')
        guest.copy(a.binary, 'root@127.0.0.1:/tmp/st/speedtest1')
        res['stage'].append('pre')
        res['pre'] = []
        for cmd in a.pre:
            r = guest.ssh(cmd)
            res['pre'].append({'command': cmd, 'rc': r.returncode, 'output': (r.stdout + r.stderr).strip()})
        res['sysctl_at_run'] = guest.ssh('sysctl security.cheri.runtime_revocation_default '
                                         'debug.witness.watch 2>&1').stdout.strip()
        res['stage'].append('run')
        before = serial.stat().st_size
        t0 = time.monotonic()
        if poller:
            poller.start(t0)
        r = guest.ssh(run_command(a, command), timeout=a.timeout + 300)
        res['seconds'] = round(time.monotonic() - t0, 1)
        if poller:
            poller.stop()
        res['ssh_rc'] = r.returncode
        m = re.search(r'^__EXIT=(\d+)$', r.stdout or '', re.M)
        res['exit'] = int(m.group(1)) if m else 'no exit (ssh: %s)' % scrub((r.stderr or '').strip()[-200:])
        if m:
            res['stage'].append('fetch')
            q = guest.ssh('cat /tmp/st/out.txt', timeout=300)
            res['stdout'], res['stdout_from'] = q.stdout, ('at exit' if q.returncode == 0 else f'fetch rc={q.returncode}')
            res['stderr'] = guest.ssh('cat /tmp/st/err.txt 2>/dev/null', timeout=300).stdout
            if a.after_start:
                res['after_start'] = {'command': a.after_start, 'observed': 'at exit',
                                      'output': guest.ssh('cat /tmp/st/after-start.txt', timeout=60).stdout.strip()}
            if a.console_echo:
                rc = re.search(r'^__RC=(\d+)$', res['stdout'], re.M)
                res['exit'] = int(rc.group(1)) if rc else 'unknown (console-echo: no __RC line)'
    except Exception as e:  # noqa: BLE001 -- the record must say where and why a run stopped
        res['error'] = scrub(f'{res["stage"][-1] if res["stage"] else "setup"}: {type(e).__name__}: {e}')[:600]
        if t0 is not None and 'seconds' not in res:
            res['seconds'] = round(time.monotonic() - t0, 1)
    finally:
        if poller:
            poller.stop()
        guest.close()
        res['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        finish_record(res, a, serial, before, t0 is not None, m is not None, poller)
        (a.out / 'result.json').write_text(json.dumps(res, indent=1) + '\n')
    print(f"exit={res.get('exit')} seconds={res.get('seconds')} tests={len(res['tests'])} "
          f"error={res.get('error', '-')}")
    print(' '.join(str(t['id']) for t in res['tests']))
    return 0 if res.get('exit') == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
