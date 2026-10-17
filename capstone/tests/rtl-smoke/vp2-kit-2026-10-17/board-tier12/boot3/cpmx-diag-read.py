#!/usr/bin/env python3
"""cpmx-diag-read.py <board.txt> [--selftest]: read the CPMX diagnostic (capstone-sbi vp2/cpmx-diag, CAPSTONE_CPMX_DIAG)
from a board transcript and name the branch it supports.

The dump follows MTVL/CPMX's trigger, printed by report_cpmx_diag:
- RGNN, then per live region RDID RDSL RDBA RDEN RDEH RDPM;
- per loaded CPMP entry CPID (id | region << 8) CPBA CPEN CPPM, and CPHT when that entry covers mtval.
MTVL follows the dump.

Branches, for mtval M:
  covered-loaded   a LOADED CPMP entry covers M (CPHT): the fetch was refused by an entry that covers it. Its perms are
                   printed; with X set this is the RTL's fetch-side check (the RTL lane), without X the monitor's piece.
  covered-unloaded a live region not in any CPMP entry covers M: swap_cpmp's lookup should have found it (the monitor).
  uncovered        no live region and no loaded entry covers M: a hole in the region table (the monitor's carves).
  NO DATA          no dump in the transcript (exit 2): the fault did not recur, or the firmware lacked the define.
"""
import re, sys

TAGS = ("RGNN", "RDID", "RDSL", "RDBA", "RDEN", "RDEH", "RDPM", "CPID", "CPBA", "CPEN", "CPPM", "CPHT", "MTVL", "CPMX")


def parse(text):
    text = text.replace("\r", "")
    rows = [(m.group(1), int(m.group(2), 16)) for m in re.finditer(r"^(%s):([0-9A-F]{8})$" % "|".join(TAGS), text, re.M)]
    # the LAST dump: from the last RGNN that a CPMX follows
    starts = [i for i, (t, _) in enumerate(rows) if t == "RGNN"]
    dumps = []
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(rows)
        seg = rows[i:end]
        if any(t == "CPMX" for t, _ in seg) and any(t == "MTVL" for t, _ in seg):
            dumps.append(seg)
    if not dumps:
        return None
    seg = dumps[-1]
    regions, cpmps, cur, hits, mtval = [], [], None, [], None
    for t, v in seg:
        if t == "RDID": cur = {"id": v}; regions.append(cur)
        elif t == "RDSL": cur["slot"] = v
        elif t == "RDBA": cur["base"] = v
        elif t == "RDEN": cur["end_lo"] = v
        elif t == "RDEH": cur["end"] = (v << 32) | cur["end_lo"]
        elif t == "RDPM": cur["perm"] = v
        elif t == "CPID": cur = {"cpmp": v & 0xff, "region": v >> 8}; cpmps.append(cur)
        elif t == "CPBA": cur["base"] = v
        elif t == "CPEN": cur["end"] = v if v else 1 << 32
        elif t == "CPPM": cur["perm"] = v
        elif t == "CPHT": hits.append(v)
        elif t == "MTVL": mtval = v
    return {"regions": regions, "cpmps": cpmps, "hits": hits, "mtval": mtval, "n": seg[0][1]}


def branch(d):
    m = d["mtval"]
    unloaded = [r for r in d["regions"] if r.get("slot") == 0xFFFFFFFF and r["base"] <= m < r["end"]]
    if d["hits"]:
        c = [c for c in d["cpmps"] if c["cpmp"] in d["hits"]]
        return "covered-loaded", f"CPMP entr{'y' if len(c) == 1 else 'ies'} " + ", ".join(
            f"{x['cpmp']} (region {x['region']}, [{x['base']:#x}, {x['end']:#x}), perm {x['perm']:#x})" for x in c)
    if unloaded:
        return "covered-unloaded", ", ".join(f"region {r['id']} [{r['base']:#x}, {r['end']:#x}) perm {r['perm']:#x}" for r in unloaded)
    return "uncovered", f"no live region or loaded entry covers {m:#x} ({len(d['regions'])} live regions, {len(d['cpmps'])} loaded entries)"


def selftest():
    def dump(regions, cpmps, hit, m):
        L = ["RGNN:%08X" % len(regions)]
        for i, (sl, b, e, p) in enumerate(regions):
            L += ["RDID:%08X" % i, "RDSL:%08X" % sl, "RDBA:%08X" % b, "RDEN:%08X" % (e & 0xffffffff), "RDEH:%08X" % (e >> 32), "RDPM:%08X" % p]
        for c, r, b, e, p in cpmps:
            L += ["CPID:%08X" % (c | r << 8), "CPBA:%08X" % b, "CPEN:%08X" % (e & 0xffffffff), "CPPM:%08X" % p]
            if b <= m < e: L.append("CPHT:%08X" % c)
        return "\n".join(L + ["MTVL:%08X" % m, "CPMX:00000002", "INJS:00000002"])
    M = 0x81d2e940
    cases = [(dump([(3, 0x80100000, 0xac000000, 7)], [(3, 0, 0x80100000, 0xac000000, 7)], True, M), "covered-loaded"),
             (dump([(0xFFFFFFFF, 0x80100000, 0xac000000, 7)], [], False, M), "covered-unloaded"),
             (dump([(0xFFFFFFFF, 0x80100000, 0x81d2e000, 7), (0xFFFFFFFF, 0x81d2f000, 0xac000000, 7)], [], False, M), "uncovered"),
             (dump([(0xFFFFFFFF, 0x8008a780, 1 << 40, 7)], [], False, M), "covered-unloaded")]
    bad = 0
    for t, want in cases:
        got = branch(parse(t))[0]; bad += got != want
        print(f"{'ok  ' if got == want else 'FAIL'} want {want:17} got {got}")
    got = parse("MTVL:81D2E940\nCPMX:00000002\n")
    print(f"{'ok  ' if got is None else 'FAIL'} want NO DATA          got {'NO DATA' if got is None else 'a dump'}"); bad += got is not None
    print("PASS" if not bad else f"{bad} FAILED"); return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]: sys.exit(selftest())
    d = parse(open(sys.argv[1], errors="replace").read())
    if d is None: print("CPMX-DIAG NO DATA: no region dump followed by MTVL/CPMX"); sys.exit(2)
    for r in d["regions"]:
        print(f"  region {r['id']:3} slot {('-' if r['slot'] == 0xFFFFFFFF else r['slot']):>2} [{r['base']:#012x}, {r['end']:#012x}) perm {r['perm']:#x}")
    for c in d["cpmps"]:
        print(f"  cpmp {c['cpmp']:2} region {c['region']:3} [{c['base']:#012x}, {c['end']:#012x}) perm {c['perm']:#x}")
    k, why = branch(d)
    print(f"CPMX-DIAG mtval {d['mtval']:#x}: BRANCH {k}: {why}")
