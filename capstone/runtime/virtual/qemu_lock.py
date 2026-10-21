"""The QEMU serialisation lock, as a reader-writer lock with a turnstile (2026-10-21).

$CAPSTONE_QEMU_LOCK is the one global lock (capstone/tests/capstone-test-env.sh). A caller that sets nothing else takes
it EXCLUSIVE, and so excludes every other QEMU session. A snapshot-class session -- the rootfs attached with -snapshot,
a private read-only stage disk, no network, output only on its serial line (this directory's run-staged.py and run.py)
-- may instead set $CAPSTONE_QEMU_DOMAIN_LOCK to a per-domain lock file: it then holds the global lock SHARED and the
domain lock EXCLUSIVE, so one session runs per domain, domains run in parallel, and an exclusive holder of the global
lock still excludes all of them and waits for them.

A TURNSTILE (<global>.turnstile) keeps a writer from starving behind back-to-back readers: Linux flock gives no writer
preference, so without it an exclusive caller could wait until both domains happened to be idle at once.
  reader:    turnstile EX -> global SH -> release turnstile -> domain EX
  exclusive: turnstile EX -> global EX (waits HERE, holding the turnstile) -> release turnstile
A waiting exclusive holds the turnstile, so no new reader passes it, and it is granted as soon as the readers already
inside finish. A caller that takes the global lock EXCLUSIVE without the turnstile (capstone_vm/cli.py, un-migrated
scripts) is still correct, only not protected from starvation. flock(2) throughout: lockf/fcntl record locks do not
interoperate with it. Prints the mode and the time the lock was granted, so a run's wait can be read from its log."""
import fcntl
import os
import time
from contextlib import contextmanager


@contextmanager
def qemu_lock():
    asked = time.time()
    path = os.environ['CAPSTONE_QEMU_LOCK']
    domain = os.environ.get('CAPSTONE_QEMU_DOMAIN_LOCK', '').strip()
    with open(path, 'a+b') as g:
        with open(path + '.turnstile', 'a+b') as t:
            fcntl.flock(t, fcntl.LOCK_EX)
            fcntl.flock(g, fcntl.LOCK_SH if domain else fcntl.LOCK_EX)
        # the turnstile is released here, with the global lock held
        if not domain:
            print(f'qemu-lock: global exclusive granted {time.time():.1f} (waited {time.time() - asked:.1f} s)', flush=True)
            yield
            return
        with open(domain, 'a+b') as d:
            fcntl.flock(d, fcntl.LOCK_EX)
            print(f'qemu-lock: global shared + domain {os.path.basename(domain)} exclusive granted {time.time():.1f} '
                  f'(waited {time.time() - asked:.1f} s)', flush=True)
            yield
