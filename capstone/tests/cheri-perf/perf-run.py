#!/usr/bin/env python3
"""Boot CheriBSD purecap under CHERI-QEMU, run the revoke-cost guest script, and
capture the serial output. QEMU argv is read from a file (one token per line),
produced by run.sh from `cheribuild.py run-riscv64-purecap --pretend`, so we boot
exactly the emulator/kernel/image cheribuild would, under our own automation.

Adapted from cheri-baseline/cheri-run.py for the perf microbenchmark.
"""
import sys
import pexpect


def main():
    if len(sys.argv) < 3:
        print("usage: perf-run.py <qemu-argv-file> <out-log>", file=sys.stderr)
        return 2
    argv_file, out_log = sys.argv[1], sys.argv[2]
    with open(argv_file) as f:
        argv = [ln.rstrip("\n") for ln in f if ln.strip()]
    cmd, args = argv[0], argv[1:]

    logf = open(out_log, "w")
    q = pexpect.spawn(cmd, args, encoding="utf-8", timeout=1200)
    q.logfile = logf
    try:
        q.expect(r"login:", timeout=1800)
        q.sendline("root")
        i = q.expect([r"Password:", r"# "], timeout=180)
        if i == 0:
            q.sendline("")
            q.expect(r"# ", timeout=60)
        q.sendline("stty -echo 2>/dev/null; true")
        q.expect(r"# ", timeout=30)
        # The guest script sets sysctls + runs the bench for all three configs.
        # Large ITERS under TCG can be slow -- give it room.
        q.sendline("sh /root/cheri-perf/run-in-guest.sh")
        q.expect(r"===CHERI-PERF-END===", timeout=3600)
        q.expect(r"# ", timeout=60)
        q.sendline("poweroff 2>/dev/null || halt -p 2>/dev/null || shutdown -p now")
        q.expect(pexpect.EOF, timeout=120)
    except (pexpect.EOF, pexpect.TIMEOUT) as exc:
        logf.flush()
        print(f"[perf-run] stopped: {type(exc).__name__}", file=sys.stderr)
        try:
            q.terminate(force=True)
        except Exception:
            pass
        return 1
    finally:
        logf.flush()
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
