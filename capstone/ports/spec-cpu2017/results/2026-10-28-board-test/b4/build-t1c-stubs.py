#!/usr/bin/env python3
"""build-t1c-stubs.py <t1b board.txt> <F0r runpy.log> <out dir>: dry-run transcripts for b4-t1c (PREREG Addendum 17),
made from t1b's REAL board transcript by asserted edits, one file per case:
- pos:        the stage marker reads t1c; `@@PARAM resolve_dirty=Y` and `@@PARAM resolve_pte=Y` after @@MODTEXT;
              `@@T1C-RD resolve_dirty=Y resolve_pte=Y` after the marker; vm's S-24 lines (cause 15, Segmentation fault, exit 139) replaced by F0r's own vm tail
              (VM_CONTRACT_FAIL:execute through mapping PCC, its stats line, exit 1); the S-24 KMSG bound line dropped;
              after VIRTUAL_RUNTIME_DONE the RD2, clock and spin-pair lines (spin: 5 s not ready, 15 s ready).
- noparam:    pos without the @@PARAM line (boardvm refuses; the judge reads READBACK FAIL).
- paramN:     pos with every resolve_dirty read-back N.
- pteN:       pos with every resolve_pte read-back N.
- s24:        pos with t1b's S-24 vm lines kept (the cells and P2/P3 go OFF).
- nospin:     pos without the spin and clock lines (reported NO DATA; grading unaffected)."""
import re, sys
from pathlib import Path

src, f0r, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
t = src.read_text(errors="replace")
nl = "\r\n" if "\r\n" in t else "\n"
L = t.split(nl)
tw = f0r.read_text(errors="replace").replace("\r", "").splitlines()
k = tw.index("VM_EXIT:vm:1")
vm_tail = tw[k - 2:k + 1]
assert vm_tail[0] == "VM_CONTRACT_FAIL:execute through mapping PCC" and vm_tail[1].startswith("CAPSTONE_VM_STATS "), vm_tail


def one(pred):
    idx = [i for i, s in enumerate(L) if pred(s)]
    assert len(idx) == 1, idx
    return idx[0]


def build(param="Y", rd="Y", pte="Y", fix_vm=True, spin=True):
    x = list(L)
    i = [n for n, s in enumerate(x) if s.startswith("@@B4-STAGE t1b")]
    assert len(i) == 1
    x[i[0]] = "@@B4-STAGE t1c"
    x.insert(i[0] + 1, f"@@T1C-RD resolve_dirty={rd} resolve_pte={pte}")
    if fix_vm:
        a = [n for n, s in enumerate(x) if s.startswith("capstone-exec: domain fault cause=15 pc=0x3fb8411a48")]
        assert len(a) == 1 and x[a[0] + 1] == "Segmentation fault" and x[a[0] + 2] == "VM_EXIT:vm:139"
        x[a[0]:a[0] + 3] = vm_tail
        b = [n for n, s in enumerate(x) if "resolved 65 times with no change" in s]
        assert len(b) == 1
        del x[b[0]]
    if param is not None:
        m = [n for n, s in enumerate(x) if s.startswith("@@MODTEXT ")]
        assert len(m) == 1
        x[m[0] + 1:m[0] + 1] = [f"@@PARAM resolve_dirty={param}", f"@@PARAM resolve_pte={pte}"]
    d = [n for n, s in enumerate(x) if s == "VIRTUAL_RUNTIME_DONE"]
    e = [n for n, s in enumerate(x) if s == "@@SPEC-DONE b4-t1b directed"]
    assert len(d) == 1 and len(e) == 1 and e[0] == d[0] + 1
    add = [f"@@T1C-RD2 resolve_dirty={rd} resolve_pte={pte}"]
    if spin:
        add += ["@@T1C-CLOCK rc=0 w0=1793400000000000000 w1=1793400007100000000 u0=2000.00 u1=2007.10 "
                "launch_ns=3781543440 elapsed_ns=6382554640",
                "@@T1C-SPIN wait=5 rc=143 ready=0", "Terminated",
                "@@T1C-SPIN wait=15 rc=143 ready=1", "@@T1C-SPIN15-OUT VIRTUAL_SPIN_READY"]
    add += ["@@SPEC-DONE b4-t1c directed"]
    x[e[0]:e[0] + 1] = add
    return nl.join(x)


out.mkdir(parents=True, exist_ok=True)
cases = {"pos": build(), "noparam": build(param=None), "paramN": build(param="N", rd="N"),
         "pteN": build(pte="N"),
         "s24": build(fix_vm=False), "nospin": build(spin=False)}
for n, s in cases.items():
    (out / f"{n}.board.txt").write_text(s)
    print(f"stub {n}: {len(s.splitlines())} lines")
