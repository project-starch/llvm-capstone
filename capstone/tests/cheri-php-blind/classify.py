#!/usr/bin/env python3
"""Classify the CheriBSD purecap serial log into the per-row verdict table.

Six passes are run in one boot: three revocation configs x two allocator arms.

  spatial  - revocation OFF             (CHERI spatial safety only)
  temporal - revocation ON, async sweep (realistic CHERI temporal deployment)
  eager    - revocation ON, revoke/free (aggressive synchronous upper bound)

  stockalloc - PHP 5.0.0's allocator as shipped
  patched    - the two sed patches the corpus applies so ASAN can see the defects
               (ZEND_DISABLE_MEMORY_CACHE 0->1, REAL_SIZE(size) -> (size))

WHERE a fault happens is as important as whether one happens, because
shutdown_memory_manager() drains the size-class cache and free()s every block.
A fault during shutdown is NOT a catch of the defect: the program has already
consumed the stale data. Each row therefore prints phase markers and the guest
runner reports the last one reached.

Verdicts:
  BLOCKED-SYNC     - faults in `spatial`, at the offending access. Caught at the
                     contract point with no revocation at all.
  BLOCKED-SWEEP(c) - no spatial fault, but config c faults at the access.
  BLOCKED-ABORT    - allocator aborted (SIGABRT), e.g. double free.
  LATE@SHUT        - no fault at the access; faults only in shutdown. The defect
                     ran to completion; this is a late catch, not a catch.
  MISS             - survives every config at the access and at shutdown.
"""
import sys, re, collections

SIG = {162: "SIGPROT", 139: "SIGSEGV", 138: "SIGBUS", 134: "SIGABRT",
       132: "SIGILL", 136: "SIGFPE", 137: "SIGKILL", 159: "SIGSYS"}
CFGS = ("spatial", "temporal", "eager")
ARMS = ("stockalloc", "patched")

# A fault during these phases is a catch AT the offending access.
USE_PHASES = {"USE-BEGIN"}
# A fault during these is a late catch only.
SHUT_PHASES = {"SHUT-BEGIN"}


def label(rc):
    if rc is None:
        return "no-data"
    return SIG.get(rc, f"SIG{rc-128}") if rc >= 128 else f"exit{rc}"


def is_fault(rc):
    return rc is not None and rc >= 128


def verdict(res):
    """res: {cfg: (rc, last_phase)} for one (row, arm)."""
    if any(res.get(c) is None for c in CFGS):
        return "NO-DATA", ""
    where = {c: res[c][1] for c in CFGS}
    rc = {c: res[c][0] for c in CFGS}

    def fault_at_use(c):
        return is_fault(rc[c]) and where[c] in USE_PHASES

    def fault_at_shut(c):
        return is_fault(rc[c]) and where[c] in SHUT_PHASES

    if rc["spatial"] == 134 or rc["temporal"] == 134 or rc["eager"] == 134:
        return "BLOCKED-ABORT", "allocator abort (double free)"
    if fault_at_use("spatial"):
        return "BLOCKED-SYNC", "at the access, no revocation needed"
    if fault_at_use("temporal"):
        return "BLOCKED-SWEEP(async)", "at the access, realistic default"
    if fault_at_use("eager"):
        return "BLOCKED-SWEEP(eager)", "at the access, revoke-on-every-free only"
    if any(fault_at_shut(c) for c in CFGS):
        cs = ",".join(c for c in CFGS if fault_at_shut(c))
        return "LATE@SHUT", f"only during shutdown ({cs}) - defect already ran"
    if any(is_fault(rc[c]) for c in CFGS):
        cs = ",".join(f"{c}:{where[c]}" for c in CFGS if is_fault(rc[c]))
        return "FAULT-ELSEWHERE", cs
    return "MISS", ""


def main():
    if len(sys.argv) < 3:
        print("usage: classify.py <serial.log> <rows.tsv>", file=sys.stderr)
        return 2
    serial, rowstsv = sys.argv[1], sys.argv[2]
    text = open(serial, errors="replace").read()

    rows = []
    for ln in open(rowstsv):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        while len(f) < 7:
            f.append("")
        rows.append(f[:7])

    # ROW <row> arm=<arm> cfg=<cfg> rc=<n> last=<phase> out=[...]
    got = {}
    pat = re.compile(r"ROW (\S+) arm=(\S+) cfg=(\S+) rc=(\d+) last=(\S+) out=\[(.*?)\]")
    for m in pat.finditer(text):
        got[(m.group(1), m.group(2), m.group(3))] = (int(m.group(4)), m.group(5))
    missing = re.findall(r"ROW (\S+) arm=(\S+) cfg=(\S+) BIN-MISSING", text)

    if not got:
        print("ERROR: no ROW result lines in %s -- nothing was measured" % serial,
              file=sys.stderr)
        return 3

    print("# config reality (per pass) -- a row verdict is void if these are wrong:")
    for cfg in CFGS:
        blob = re.findall(rf"===PHP-BLIND-BEGIN cfg={cfg}===(.*?)--- runs", text, re.S)
        b = blob[0] if blob else ""
        kn = " ".join(re.findall(r"KNOB (\S+)", b))
        st = " ".join(re.findall(r"STATUS (.+)", b))
        print(f"#   {cfg:<9} {kn}  {st}".rstrip())
    if missing:
        print("# BIN-MISSING: " + ", ".join(f"{r}/{a}/{c}" for r, a, c in missing))
    print()

    hdr = ("row", "arm", "spatial", "temporal", "eager", "verdict", "predicted", "")
    print("{:<24} {:<11} {:<16} {:<16} {:<16} {:<22} {:<16} {}".format(*hdr))
    print("-" * 150)

    tally = collections.Counter()
    surprises = []
    for row, models, ncases, klass, es, ep, note in rows:
        for arm in ARMS:
            res = {c: got.get((row, arm, c)) for c in CFGS}
            v, why = verdict(res)
            pred = es if arm == "stockalloc" else ep
            cells = []
            for c in CFGS:
                if res[c] is None:
                    cells.append("no-data")
                else:
                    cells.append(f"{label(res[c][0])}@{res[c][1]}")
            flag = ""
            if pred and pred not in ("DONE",):
                base = v.split("(")[0]
                if not (v == pred or base == pred.split("(")[0]):
                    flag = "<< UNEXPECTED"
                    surprises.append((row, arm, pred, v))
            elif pred == "DONE" and v != "MISS":
                flag = "<< UNEXPECTED"
                surprises.append((row, arm, pred, v))
            print("{:<24} {:<11} {:<16} {:<16} {:<16} {:<22} {:<16} {}".format(
                row, arm, cells[0], cells[1], cells[2], v, pred or "-", flag))
            if klass in ("temporal", "spatial"):
                tally[(klass, arm, v.split("(")[0])] += 1
    print("-" * 150)

    print("\n# modelled-defect rows only (the 31), by arm and verdict:")
    for klass in ("temporal", "spatial"):
        for arm in ARMS:
            got_any = {k[2]: v for k, v in tally.items() if k[0] == klass and k[1] == arm}
            if got_any:
                s = " ".join(f"{k}={v}" for k, v in sorted(got_any.items()))
                print(f"#   {klass:<9} {arm:<11} {s}")

    if surprises:
        print("\n# PREDICTION MISSES (pre-registered vs measured) -- investigate before reporting:")
        for row, arm, pred, v in surprises:
            print(f"#   {row} [{arm}] predicted {pred}, measured {v}")
    else:
        print("\n# every row matched its pre-registered prediction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
