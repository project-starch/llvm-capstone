#!/usr/bin/env python3
"""b4-judge-t1a.py <spec.txt> [--twin]: judge part 1 of B's first board boot (PREREG Addendum 14) against module-ids55
PREREG section 2 (predictions 1 and 3) and addendum 2, i.e. against the twin's F0 arm.
Every reading is printed with its value. Exit 0: every prediction holds (part 2 may boot); 1: a reading is off the
prediction (HOLD); 2: no data (the stage's DONE marker or a section is absent).
--twin: read a twin transcript (f55/live-F0/spec.txt, which has no contract section and reads NON-LINEAR by the 6.1
kernel, addendum 2): the contract and mapping-word checks are reported but not counted. That is the known-good input
for the dry run; a board transcript is always read without it."""
import re, sys
from pathlib import Path

args = [x for x in sys.argv[1:] if not x.startswith("--")]
twin = "--twin" in sys.argv
if len(args) != 1:
    sys.exit("usage: b4-judge-t1a.py <spec.txt> [--twin]")
p = Path(args[0])
if not p.is_file():
    print(f"NO DATA: {p} does not exist"); sys.exit(2)
t = p.read_text(errors="replace").replace("\r", "")
bad, notes = [], []


def check(name, ok, val, counted=True):
    tag = "OK " if ok else ("OFF" if counted else "N/A")
    print(f"  {tag} {name}: {val}")
    if not ok and counted:
        bad.append(name)


if "@@SMOKE-DONE" not in t:
    print(f"NO DATA: no @@SMOKE-DONE in {p}"); sys.exit(2)

par = dict(re.findall(r"^@@SMOKE-PARAM (\w+)=(\S+)", t, re.M))
check("param ids", par.get("ids") == "55", par.get("ids"))
check("param index_bits", par.get("index_bits") == "25", par.get("index_bits"))
check("param node_flat_records", par.get("node_flat_records") == "1048576", par.get("node_flat_records"))

# The CMA region: the board kernel's reserved-mem line, else the twin's cma= line, else the DT's 256 MiB at 0xac000000.
m = re.search(r"created CMA memory pool at 0x([0-9a-f]+), size (\d+) MiB", t) or re.search(r"cma: Reserved (\d+) MiB at 0x([0-9a-f]+)", t)
if m and m.re.pattern.startswith("created"):
    lo, size = int(m.group(1), 16), int(m.group(2)) << 20; src = "the kernel's reserved-mem line"
elif m:
    lo, size = int(m.group(2), 16), int(m.group(1)) << 20; src = "the kernel's cma: Reserved line"
else:
    lo, size = 0xac000000, 256 << 20; src = "NOT READ: the DT's pool (0xac000000, 256 MiB) as the runtime lane reports it"
print(f"  --- CMA region [{lo:#x}, {lo + size:#x}) from {src}")
al = re.findall(r"capstone_vm: flat table (\d+) records, (\d+) bytes, from (\w+), pa ([0-9a-f]+), (NON-LINEAR|linear) mapping", t)
uniq = sorted(set(al))
check("allocation line present, one distinct", len(uniq) == 1, uniq)
if uniq:
    rec, nb, frm, pa, mp = uniq[0]; pa = int(pa, 16)
    check("allocation records/bytes", (rec, nb) == ("1048576", "16777216"), (rec, nb))
    check("allocation from CMA", frm == "CMA", frm)
    check("allocation pa inside the CMA region", lo <= pa and pa + int(nb) <= lo + size, hex(pa))
    check("allocation mapping word (NON-LINEAR on the board is a STOP)", mp == "linear", mp, counted=not twin)
check("no ALLOCATION FAILED line", "ALLOCATION FAILED" not in t, t.count("ALLOCATION FAILED"))

r = re.search(r"^RESULT vm-contract retval=(\d+)$", t, re.M)
check("Tier 1 contract (ok exit 0 + VIRTUAL_APPLICATION_OK; stale exit 139 + cause 25)", bool(r) and r.group(1) == "0",
      r.group(1) if r else "ABSENT", counted=not twin)


def smoke(n):
    s = re.search(rf"^@@SMOKE-BEGIN live {n}$(.*?)^@@SMOKE-END live {n} rc=(\d+)$", t, re.M | re.S)
    if not s:
        return None
    body, rc = s.group(1), int(s.group(2))
    st = re.search(r"STATS nodes=(\d+) node_capacity=(\d+)", body)
    fl = re.search(r"FAULT-LINES (\d+)", body)
    return dict(body=body, rc=rc, nodes=int(st.group(1)) if st else None, cap=int(st.group(2)) if st else None,
                faults=int(fl.group(1)) if fl else None)


for n in (1000, 400000, 1100000):
    s = smoke(n)
    if s is None:
        check(f"live {n}", False, "SECTION ABSENT"); continue
    if n < 1100000:
        ok = f"LIVE-OK {n} bad=0" in s["body"] and s["rc"] == 0 and s["faults"] == 0
        check(f"live {n} LIVE-OK bad=0 rc 0, no fault line", ok, f"rc {s['rc']} faults {s['faults']} nodes {s['nodes']}")
        if n == 400000:
            # The identity reading: 414,217 records means slots above 2^18, which bitstream 3's 18-bit index cannot hold.
            check("live 400000 nodes in [400000, 430000], capacity 1048576",
                  s["nodes"] is not None and 400000 <= s["nodes"] <= 430000 and s["cap"] == 1048576, (s["nodes"], s["cap"]))
    else:
        mn = re.search(r"LIVE-NULL at (\d+)", s["body"])
        at = int(mn.group(1)) if mn else None
        check("live 1100000 LIVE-NULL index in [990000, 1047000]", at is not None and 990000 <= at <= 1047000, at)
        check("live 1100000 rc 3", s["rc"] == 3, s["rc"])
        check("live 1100000 nodes in [1040000, 1048576]", s["nodes"] is not None and 1040000 <= s["nodes"] <= 1048576, s["nodes"])
        check("live 1100000 no domain fault", s["faults"] == 0, s["faults"])

print(f"B4-T1A {'AS PREDICTED' if not bad else 'OFF THE PREDICTION: ' + ', '.join(bad)}")
sys.exit(1 if bad else 0)
