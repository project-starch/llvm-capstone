#!/usr/bin/env python3
"""rfix-grade.py <PREREG.md> <session dir>: grade the R-58 / R-59 / Q-24 bare kit (llvm-capstone land/rtl-fixes-kit
30117b74cc8f, capstone/tests/rtl-smoke/r58-r59-q24-bare-2026-10-26) by its own PREREG table, read from the PREREG file the
session ran (each image's board_rec, SR and vector; "the same as X" rows take X's). The recorder lines are `SR <16 hex>`
(the count) and `SV <16 hex>` (the values) in <session dir>/<image>/uart.txt, as va-alloc-grade.py reads them.
Rules (the PREREG's): VOID unless va-runv reads SR 15 and its vector exactly; R-58 CONFIRMED by va-srevroot's exact
vector, R-59 by BOTH r59 images, Q-24 by q24-deleg-on with q24-deleg as its control (a q24-deleg reading that differs voids
the Q-24 pair). Any other vector is a FINDING, printed value by value. Exit 0 = every fix CONFIRMED, 1 = any FINDING or
voided pair, 2 = VOID or NO DATA."""
import re
import sys
from pathlib import Path


def table(prereg):
    rows = {}
    for l in open(prereg):
        if not l.startswith("| `"):
            continue
        c = [x.strip() for x in l.split("|")[1:-1]]
        img = c[0].split("`")[1].removesuffix(".bin")
        m = re.search(r"`([^`]+)`", c[4])
        rows[img] = (int(c[3]), [int(x, 16) for x in m.group(1).split()] if m else c[4])
    for k, (n, v) in rows.items():
        if isinstance(v, str):
            m = re.match(r"the same as (\S+)", v)
            rows[k] = (n, rows[m.group(1)][1])
    return rows


def read(sess, img):
    p = Path(sess) / img / "uart.txt"
    if not p.is_file():
        return None
    t = p.read_text(errors="replace")
    return re.findall(r"SR ([0-9A-Fa-f]{16})", t), [int(v, 16) for v in re.findall(r"SV ([0-9A-Fa-f]{16})", t)]


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    rows = table(sys.argv[1]); sess = sys.argv[2]; ok = {}
    for img, (n, want) in rows.items():
        r = read(sess, img)
        if r is None:
            print(f"RFIX {img} NO DATA"); ok[img] = None; continue
        sr, sv = r
        vals = " ".join(f"{v:#x}" for v in sv)
        if len(sr) == 1 and int(sr[0], 16) == n and sv == want:
            print(f"RFIX {img} EXACT: {vals}"); ok[img] = True
        else:
            d = [f"{i}: read {a:#x} want {b:#x}" for i, (a, b) in enumerate(zip(sv, want)) if a != b]
            print(f"RFIX {img} DIFFERS: SR {sr} ({len(sv)} values, want {n}) :: {'; '.join(d) or 'length only'} :: {vals}"); ok[img] = False
    if not ok.get("va-runv"):
        print("RFIX VOID: the control va-runv did not read its vector exactly" if ok.get("va-runv") is False else "RFIX VOID: no control reading")
        sys.exit(2)
    rc = 0
    for fix, imgs, ctl in (("R-58", ["va-srevroot"], None), ("R-59", ["va-runv-r59", "va-runv-r59-low"], None),
                           ("Q-24", ["q24-deleg-on"], "q24-deleg")):
        if ctl and not ok.get(ctl):
            print(f"RFIX {fix} VOID: its control {ctl} did not read its vector"); rc = 1; continue
        if all(ok.get(i) for i in imgs):
            print(f"RFIX {fix} CONFIRMED ON SILICON ({', '.join(imgs)} exact)")
        else:
            print(f"RFIX {fix} FINDING: {', '.join(i for i in imgs if not ok.get(i))} not exact"); rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
