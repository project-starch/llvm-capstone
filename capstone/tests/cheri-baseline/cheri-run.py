#!/usr/bin/env python3
"""Boot CheriBSD purecap under CHERI-QEMU, run the corpus guest script for both
configs (A = spatial only, B = +revocation sweep), capture the serial output.

The QEMU argv is read from a file (one token per line), produced by the wrapper
from `cheribuild.py run-riscv64-purecap --pretend`. This keeps us using exactly
the emulator/kernel/image cheribuild would boot, but under our own automation so
we can script the two configs and capture results deterministically.
"""
import sys
import pexpect

def main():
    if len(sys.argv) < 3:
        print("usage: cheri-run.py <qemu-argv-file> <out-log>", file=sys.stderr)
        return 2
    argv_file, out_log = sys.argv[1], sys.argv[2]
    with open(argv_file) as f:
        argv = [ln.rstrip("\n") for ln in f if ln.strip()]
    cmd, args = argv[0], argv[1:]

    logf = open(out_log, "w")
    q = pexpect.spawn(cmd, args, encoding="utf-8", timeout=600)
    q.logfile = logf

    try:
        # CheriBSD boots to a login prompt. Boot can be slow under TCG.
        q.expect(r"login:", timeout=900)
        q.sendline("root")
        # root has no password in cheribuild images; land on the shell prompt.
        i = q.expect([r"Password:", r"# "], timeout=120)
        if i == 0:
            q.sendline("")           # empty password
            q.expect(r"# ", timeout=60)
        # quieten the console a touch
        q.sendline("stty -echo 2>/dev/null; sysctl kern.consmsgbuf_size >/dev/null 2>&1; true")
        q.expect(r"# ", timeout=30)

        for cfg in ("spatial", "temporal", "eager"):
            q.sendline(f"sh /root/cheri-baseline/run-in-guest.sh {cfg}")
            q.expect(rf"===CHERI-BASELINE-END cfg={cfg}===", timeout=600)
            q.expect(r"# ", timeout=30)

        q.sendline("poweroff 2>/dev/null || halt -p 2>/dev/null || shutdown -p now")
        q.expect(pexpect.EOF, timeout=120)
    except (pexpect.EOF, pexpect.TIMEOUT) as exc:
        logf.flush()
        print(f"[cheri-run] stopped: {type(exc).__name__}", file=sys.stderr)
        try:
            q.terminate(force=True)
        except Exception:
            pass
        return 1
    finally:
        logf.flush(); logf.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
