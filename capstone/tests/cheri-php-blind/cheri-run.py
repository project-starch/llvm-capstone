#!/usr/bin/env python3
"""Boot CheriBSD purecap under CHERI-QEMU and run the PHP-allocator rows for all
three revocation configs, capturing the serial output.

Adapted from capstone/tests/cheri-baseline/cheri-run.py. It is a separate file
rather than a parameter to that one because that script is the committed driver
for two other experiments and its end-marker is part of their contract.

The QEMU argv is read from a file (one token per line), as produced by
provision-cheri-vehicle.sh, so we boot exactly the emulator/kernel/image the
vehicle recipe built, under our own automation.
"""
import sys
import pexpect

END = "===PHP-BLIND-END cfg={cfg}==="


def main():
    if len(sys.argv) < 3:
        print("usage: cheri-run.py <qemu-argv-file> <out-log> [guest-dir]",
              file=sys.stderr)
        return 2
    argv_file, out_log = sys.argv[1], sys.argv[2]
    guest_dir = sys.argv[3] if len(sys.argv) > 3 else "/root/cheri-php-blind"
    with open(argv_file) as f:
        argv = [ln.rstrip("\n") for ln in f if ln.strip()]
    cmd, args = argv[0], argv[1:]

    logf = open(out_log, "w")
    q = pexpect.spawn(cmd, args, encoding="utf-8", timeout=900)
    q.logfile = logf

    try:
        q.expect(r"login:", timeout=1200)
        q.sendline("root")
        i = q.expect([r"Password:", r"# "], timeout=180)
        if i == 0:
            q.sendline("")
            q.expect(r"# ", timeout=120)
        q.sendline("stty -echo 2>/dev/null; true")
        q.expect(r"# ", timeout=60)

        for cfg in ("spatial", "temporal", "eager"):
            q.sendline(f"sh {guest_dir}/run-in-guest.sh {cfg}")
            # 22 binaries per pass, each with a 30 s guest-side timeout.
            q.expect(END.format(cfg=cfg), timeout=1800)
            q.expect(r"# ", timeout=120)

        q.sendline("poweroff 2>/dev/null || halt -p 2>/dev/null || shutdown -p now")
        q.expect(pexpect.EOF, timeout=180)
    except (pexpect.EOF, pexpect.TIMEOUT) as exc:
        logf.flush()
        print(f"[cheri-run] stopped: {type(exc).__name__}", file=sys.stderr)
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
