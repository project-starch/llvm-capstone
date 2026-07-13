#!/usr/bin/env python3
"""Classify the CHERI baseline serial log into the per-row security table.

Three configs are run in one boot:
  spatial  - revocation OFF               (CHERI spatial safety only)
  temporal - revocation ON, async sweep   (realistic CHERI temporal deployment)
  eager    - revocation ON, revoke/free   (aggressive synchronous upper bound)

Verdict (from the task taxonomy):
  BLOCKED-SYNC   - faults in `spatial`: caught AT the offending access with no
                   revocation at all (null-deref / out-of-bounds).
  BLOCKED-SWEEP  - no spatial fault, but a revocation config faults: the dangling
                   capability is caught only by a sweep, not at the contract point.
                   (async) = caught even in the realistic default; (eager) = only
                   with revoke-on-every-free, which is not the default.
  MISS           - survives every config: the defect is not caught.

On CheriBSD a process killed by signal S exits 128+S; a capability fault is
SIGPROT (34) -> 162. NULL deref through an untagged cap is also SIGPROT.
"""
import sys, re

SIG = {162: "SIGPROT", 139: "SIGSEGV", 138: "SIGBUS", 134: "SIGABRT",
       132: "SIGILL", 136: "SIGFPE", 137: "SIGKILL", 159: "SIGSYS"}
CFGS = ("spatial", "temporal", "eager")

def label(rc):
    if rc is None:
        return "no-data"
    if rc >= 128:
        return SIG.get(rc, f"SIG{rc-128}")
    return f"exit{rc}"

def is_fault(rc):
    return rc is not None and rc >= 128

def main():
    serial, rowstsv = sys.argv[1], sys.argv[2]
    text = open(serial, errors="replace").read()

    rows = []
    for ln in open(rowstsv):
        if ln.startswith("#") or not ln.strip():
            continue
        rows.append(ln.rstrip("\n").split("\t"))

    rc = {}
    for m in re.finditer(r"ROW (\d+) cfg=(\w+) rc=(\d+) out=\[(.*?)\]", text):
        rc[(m.group(1), m.group(2))] = int(m.group(3))

    print("# sanity (purecap SQLite must run clean, else results are invalid):")
    for m in re.finditer(r"SANITY (\S+) rc=(\d+) out=\[(.*?)\]", text):
        name, code = m.group(1), int(m.group(2))
        tag = "OK" if code == 0 else f"FAULT({label(code)})"
        print(f"#   {name:<16} rc={code} {tag}")
    print()
    print("# config reality (per pass):")
    for cfg in CFGS:
        knobs = re.findall(rf"===CHERI-BASELINE-BEGIN cfg={cfg}===(.*?)--- runs", text, re.S)
        blob = knobs[0] if knobs else ""
        kn = " ".join(re.findall(r"KNOB (\S+)", blob))
        st = " ".join(re.findall(r"STATUS (.+)", blob))
        print(f"#   {cfg:<9} {kn}  {st}".rstrip())
    print()

    hdr = ("row", "dir", "host-oracle", "spatial", "temporal", "eager", "verdict")
    print("{:<3} {:<34} {:<18} {:<9} {:<9} {:<9} {}".format(*hdr))
    print("-" * 118)
    for f in rows:
        newrow, dirn, oracle = f[0], f[1], f[2]
        rcs = {c: rc.get((newrow, c)) for c in CFGS}
        def is_abort(rc):
            return rc == 134  # SIGABRT: allocator double-free detection
        if is_fault(rcs["spatial"]):
            verdict = "BLOCKED-SYNC"
        elif is_abort(rcs["temporal"]) or is_abort(rcs["eager"]):
            verdict = "BLOCKED-ABORT(double-free)"
        elif is_fault(rcs["temporal"]):
            verdict = "BLOCKED-SWEEP(async)"
        elif is_fault(rcs["eager"]):
            verdict = "BLOCKED-SWEEP(eager)"
        elif any(v is None for v in rcs.values()):
            verdict = "NO-DATA"
        else:
            verdict = "MISS"
        print("{:<3} {:<34} {:<18} {:<9} {:<9} {:<9} {}".format(
            newrow, dirn, oracle,
            label(rcs["spatial"]), label(rcs["temporal"]), label(rcs["eager"]),
            verdict))

if __name__ == "__main__":
    main()
