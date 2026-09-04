#!/usr/bin/env python3
"""Count how many times a variant reaches the S-12 faulting instruction, under QEMU.

WHY THIS EXISTS. The bug is ~54% per board draw, and the baseline executes the faulting
instruction about five times per run (the function runs `3 + plan depth` times). So the
per-execution rate is ~0.145 -- and a deletion that corrupts the function's return value gives the
variant ONE exposure where the baseline got five. Such a variant then wedges only ~15% of the time
EVEN IF EVERY DELETED INSTRUCTION IS IRRELEVANT.

A single clean board draw from such a variant reads as "this instruction is required". It is not
evidence of anything. This tool measures the denominator so that reading cannot be made.

QEMU does not reproduce the fault, and after NOPing the rest of the function runs with
uninitialised locals so its eventual crash is uninformative. Neither matters: what QEMU gives,
exactly and for free, is the count of executions before the program dies -- which is precisely the
exposure a board draw would get.

WHAT IS COUNTED. Entries to `sqlite3WhereCodeOneLoopStart`, not executions of the faulting
instruction itself. The window from entry to fault is branch-free and NOP does not branch, so the
two are equal for every variant this campaign produces, and the entry block is far easier to
identify unambiguously in a trace.

THE ADDRESS IS DISCOVERED, NEVER ASSUMED. The monitor picks the domain's load base at runtime and
it differs between runs and between arms; a hardcoded address would silently count nothing and
report 0, which this campaign would read as "inadmissible" -- a wrong answer that looks like a
decision. So pass 1 translates with `-d in_asm` and finds the block whose disassembly carries the
function's prologue signature; pass 2 counts that block's executions with `-d exec,nochain`.

A count of 0 is reported as an ERROR, not as a result. "Never reached" and "the tracer found
nothing" are different claims and only one of them is about the variant.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.environ.get("CAPSTONE_ROOT") or subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
    cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
QEMU = f"{ROOT}/capstone/capstone-qemu/build/qemu-system-riscv64"
SMOKE = f"{ROOT}/capstone/tests/runtime-qemu/run-domain-smoke.py"
# The FAULTING instruction's encoding, as QEMU prints it: one word, big-endian text, no spaces.
# Searched for by ENCODING rather than by address because the monitor picks the domain's load base
# at runtime -- under QEMU it maps at 0x101c00000, nothing like the 0x828xxxxx the board reports --
# and a hardcoded address finds nothing and reports 0, which this campaign would read as
# "inadmissible". QEMU's disassembler renders Capstone opcodes as `illegal`; it still executes them.
FAULT_ENC = "0b07275b"
# The window is 35 instructions before the fault, so entry = fault - 35*4.
WINDOW_INSNS = 35


def run_traced(dom, test, host, extra, log_path, timeout):
    share = tempfile.mkdtemp(prefix="s12expo-")
    try:
        shutil.copy(dom, os.path.join(share, "sqlite_silicon.dom"))
        shutil.copy(host, os.path.join(share, "sqlite_host.user"))
        shutil.copy(test, os.path.join(share, "case.test"))
        cmd = [
            "python3", SMOKE, "--share-dir", share,
            "--guest-command",
            "cp /mnt/host/sqlite_host.user /tmp/h.user && chmod 0755 /tmp/h.user && "
            "/tmp/h.user /mnt/host/sqlite_silicon.dom --slt /mnt/host/case.test",
            "--timeout-multiplier", "12",
        ]
        # `--qemu-extra-arg=-d`, not `--qemu-extra-arg -d`: argparse consumes a bare leading-dash
        # value as an option and fails with "expected one argument".
        for e in extra:
            cmd.append(f"--qemu-extra-arg={e}")
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    finally:
        shutil.rmtree(share, ignore_errors=True)
    return log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dom")
    ap.add_argument("--test", default=f"{ROOT}/capstone/benchmarks/sqlite/slt/dd2_join.test")
    ap.add_argument("--host", default="/tmp/capstone/sqlite-slt2/sqlite_host.user")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--entry", default=None,
                    help="skip the locate pass and use this hex entry address. Valid ONLY "
                         "because NOP patching preserves image size and layout, so every variant "
                         "loads at the same base; verify it once per batch against a full run.")
    a = ap.parse_args()

    for p in (QEMU, SMOKE, a.dom, a.test, a.host):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    tmp = tempfile.mkdtemp(prefix="s12expo-log-")
    try:
        if a.entry:
            entry = int(a.entry, 16)
        else:
            entry = None
        # Pass 1 -- translation only, to LOCATE the function. in_asm emits once per translated
        # block, so this is bounded (~40 MB). Skipped when --entry is supplied.
        p1 = os.path.join(tmp, "in_asm.log")
        if entry is None:
            run_traced(a.dom, a.test, a.host,
                       ["-d", "in_asm,nochain", "-D", p1],
                       p1, a.timeout)
        if entry is None and (not os.path.exists(p1) or os.path.getsize(p1) == 0):
            print("ERROR: pass 1 produced no trace -- the tracer did not run, which is NOT the "
                  "same as the variant never reaching the function. Check the qemu -D path is "
                  "writable from inside the runner.", file=sys.stderr)
            return 2

        for line in ([] if entry is not None else open(p1, errors="replace")):
            m = re.match(r"^0x([0-9a-f]+):\s+" + FAULT_ENC + r"\b", line)
            if m:
                entry = int(m.group(1), 16) - WINDOW_INSNS * 4
                break
        if entry is None:
            print(f"ERROR: encoding {FAULT_ENC} (the faulting instruction) was not found in the "
                  f"translation trace. The variant may never have called the function, or the "
                  f"patch removed it. Either way this is not an exposure count.", file=sys.stderr)
            return 2

        # Pass 2 -- count executions of that block.
        p2 = os.path.join(tmp, "exec.log")
        run_traced(a.dom, a.test, a.host,
                   ["-d", "exec,nochain",
                    "-dfilter", f"0x{entry:x}..0x{entry + (WINDOW_INSNS + 4) * 4:x}", "-D", p2],
                   p2, a.timeout)
        n = 0
        if os.path.exists(p2):
            for line in open(p2, errors="replace"):
                # The guest PC sits INSIDE the bracket group and carries no 0x prefix:
                # `Trace 0: 0x77.. [00000000/0000000101cf4788/../..]`. Matching on "0x<addr>"
                # counts zero and looks like "never reached".
                if f"{entry:x}" in line:
                    n += 1

        print(f"exposure: {n}   (entries to {'sqlite3WhereCodeOneLoopStart'} at 0x{entry:x})")
        if n == 0:
            print("  ERROR: zero. Reported as an error, not a result -- 'never reached' and 'the "
                  "tracer found nothing' are different claims.", file=sys.stderr)
            return 1
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
