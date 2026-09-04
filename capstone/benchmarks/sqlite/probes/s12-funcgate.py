#!/usr/bin/env python3
"""Does a NOP-patched variant still BEHAVE like the base, under QEMU?

WHY. The static gates in s12-minimize.py are sound in the register direction and UNSOUND in one
specific memory direction: the frame-slot scan is linear and ignores control flow, so when it sees
a later store to the same slot it clears the violation -- even though that store may sit inside a
branch that is not taken on this input. [28], [30] and [33] are cleared exactly that way, by stores
520-1220 instructions downstream, and whether those execute is a question about the program, not
about the disassembly.

Rather than model the control flow, run it. QEMU never reproduces S-12, which is precisely what
makes it the right instrument here: any behavioural difference it reports is caused by the CUT and
by nothing else. A variant whose QEMU output is byte-identical to the base's is behaviour-
preserving on this input, and its board result is therefore readable -- a wedge is the defect, not
the deletion. A variant whose output differs has corrupted the program, and its board result would
be another EJF: a deterministic fault somewhere unrelated that reads like a signature.

This is the gate that would have refused EJF before it cost three draws.

A run that produces NO output is an ERROR, not a difference -- "the harness failed" and "the
variant behaved differently" are different claims and only one is about the variant.
"""
import argparse, hashlib, os, shutil, subprocess, sys, tempfile

ROOT = "/home/alexey/dev/llvm-capstone"
SMOKE = f"{ROOT}/capstone/tests/runtime-qemu/run-domain-smoke.py"
HOST = "/tmp/capstone/sqlite-slt2/sqlite_host.user"
TEST = f"{ROOT}/capstone/benchmarks/sqlite/slt/dd2_join.test"


def run(dom, timeout, tag):
    """Run one image and return its GUEST SERIAL LOG.

    The serial log, not the runner's stdout. The runner reports pass/fail; the log carries the
    guest's own markers (`SQ: G/enter`, `SQ: H/return`, the SLT-SUMMARY counters), which is what
    "did this variant behave like the base" actually means. Passing --log-file also matches the
    invocation in benchmarks/sqlite/run-sqlite-slt.sh, which is the one known to work.
    """
    share = tempfile.mkdtemp(prefix="s12fg-")
    log = f"/tmp/capstone/funcgate-{tag}.log"
    try:
        shutil.copy(dom, os.path.join(share, "sqlite_silicon.dom"))
        shutil.copy(HOST, os.path.join(share, "sqlite_host.user"))
        shutil.copy(TEST, os.path.join(share, "case.test"))
        cmd = ["python3", SMOKE, "--share-dir", share, "--log-file", log,
               "--timeout-multiplier", "12", "--guest-command",
               "cp /mnt/host/sqlite_host.user /tmp/h.user && chmod 0755 /tmp/h.user && "
               "/tmp/h.user /mnt/host/sqlite_silicon.dom --slt /mnt/host/case.test"]
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    finally:
        shutil.rmtree(share, ignore_errors=True)
    return open(log, errors="replace").read() if os.path.exists(log) else ""


def signature(out):
    """The guest-visible result lines, with per-run noise removed.

    Addresses, timings and the domain load base differ between runs of the SAME image, so a raw
    diff reports every run as different and the gate becomes vacuous in the refusing direction --
    which is the safe direction but makes it useless. Keep the lines the host program prints about
    the TEST, drop everything else.
    """
    keep = []
    for line in out.splitlines():
        s = line.strip()
        if any(k in s for k in ("SLT", "slt:", "ok ", "FAIL", "PASS", "mismatch",
                                "statements", "queries", "result", "RESULT", "retval")):
            keep.append(s)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    ref_raw = run(a.base, a.timeout, "base")
    ref = signature(ref_raw)
    if not ref:
        print("ERROR: the BASE produced no recognisable result lines. The harness did not work; "
              "this is not a statement about any variant. Raw tail:\n" +
              "\n".join(ref_raw.splitlines()[-25:]), file=sys.stderr)
        return 2
    print(f"base {hashlib.sha256(open(a.base,'rb').read()).hexdigest()[:12]}  "
          f"{len(ref)} result lines")
    for l in ref:
        print("   |", l)

    rc = 0
    for v in a.variants:
        out = run(v, a.timeout, os.path.basename(v).replace(".dom", ""))
        sig = signature(out)
        tag = hashlib.sha256(open(v, "rb").read()).hexdigest()[:12]
        if not sig:
            print(f"\n{os.path.basename(v)} {tag}: ERROR -- no result lines at all. Ambiguous "
                  f"between a harness failure and a variant that died before printing; NOT a "
                  f"verdict.")
            rc = max(rc, 2)
        elif sig == ref:
            print(f"\n{os.path.basename(v)} {tag}: BEHAVIOUR-PRESERVING -- identical result "
                  f"lines. Board verdict would be readable.")
        else:
            print(f"\n{os.path.basename(v)} {tag}: DIVERGED -- the cut changed the program.")
            for l in sig[:8]:
                print("   >", l)
            rc = max(rc, 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
