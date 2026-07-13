#!/usr/bin/env python3
"""Boot CheriBSD purecap and run one guest command, capture output, power off."""
import sys, pexpect
argv_file, out_log, guest_cmd = sys.argv[1], sys.argv[2], sys.argv[3]
argv = [l.rstrip("\n") for l in open(argv_file) if l.strip()]
logf = open(out_log, "w")
q = pexpect.spawn(argv[0], argv[1:], encoding="utf-8", timeout=600)
q.logfile = logf
try:
    q.expect(r"login:", timeout=900)
    q.sendline("root")
    i = q.expect([r"Password:", r"# "], timeout=120)
    if i == 0:
        q.sendline(""); q.expect(r"# ", timeout=60)
    q.sendline(guest_cmd)
    q.expect(r"__ONESHOT_DONE__", timeout=600)
    q.expect(r"# ", timeout=30)
    q.sendline("poweroff 2>/dev/null || halt -p 2>/dev/null || shutdown -p now")
    q.expect(pexpect.EOF, timeout=120)
except (pexpect.EOF, pexpect.TIMEOUT) as e:
    print(f"[oneshot] {type(e).__name__}", file=sys.stderr)
finally:
    logf.flush(); logf.close()
