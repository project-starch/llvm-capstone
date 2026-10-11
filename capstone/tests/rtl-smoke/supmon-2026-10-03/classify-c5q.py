#!/usr/bin/env python3
"""Verdict for boot supmon-c5q against PREREG.md (C5q). Reads THIS boot only (the console replays the previous one
first: everything before the last 'OpenSBI v' is dropped). Usage: classify-c5q.py <board-supmon-c5q dir>.
Exit 0 = every prediction holds, 1 = a prediction refuted, 2 = no data / incomplete run (never a pass)."""
import ast
import re
import sys

C3_RECORD = 2551615035
ORACLE = "112006 38bb59fd"


def transcript(d):
    chunks = []
    for line in open(d + "/driver.log", errors="replace"):
        m = re.match(r"\[fpga\] \[uart\] (.*)$", line.rstrip("\n"))
        if m:
            try:
                chunks.append(ast.literal_eval(m.group(1)))
            except Exception:
                chunks.append(m.group(1))
    t = "".join(chunks)
    k = t.rfind("OpenSBI v")
    return t[k:] if k >= 0 else ""


def main(d):
    cur = transcript(d)
    if not cur:
        print("NO-RESULT: no boot in driver.log")
        return 2
    s = cur.replace("\r", "")
    # per-test sections: from a START marker (the echoed command line also contains it -- take the line that
    # STARTS with ###) to the matching END
    tests = re.findall(r"^### TEST (\d)/5 START [^\n]*\n(.*?)^### TEST \1/5 END [^\n]*?rc=(\d+) ###", s, re.S | re.M)
    out, bad = [], []
    bt = re.findall(r"BT0(\d):", s)
    out.append(f"boot trace {['BT0' + x for x in bt]}; Linux login {'login:' in s}")
    if bt != ["0", "1", "2", "3"]:
        bad.append("boot trace")
    if len(tests) != 5:
        print("\n".join(out))
        print(f"NO-RESULT: {len(tests)} of 5 tests completed")
        return 2
    cyc = {}
    for n, body, rc in tests:
        n = int(n)
        if n in (1, 5):
            r = re.findall(r"RESULT k800 retval=(\S+) cycles=(\d+)", body)
            ok = r and r[0][0] == "4" and rc == "0"
            out.append(f"test {n} k800: {r} rc={rc} -> {'ok' if ok else 'MISS'}")
            if not ok:
                bad.append(f"k800 test {n}")
            continue
        h = re.findall(r"Verification Hash: (\d+ \w{8})", body)
        heap = re.findall(r"HEAP \d+ DROPPED \d+ RC \d+", body)
        c = re.findall(r"SPEEDTEST1-CYCLES (\d+)", body)
        supm = re.findall(r"SUPM:([0-9A-F]+)", body)
        supn = re.findall(r"SUPN:([0-9A-F]+)", body)
        supk = re.findall(r"SUPK:([0-9A-F]+)", body)
        supa = re.findall(r"SUPA:([0-9A-F]+)", body)
        run = n - 1
        want_m = "00000001" if run == 2 else "00000000"
        ok = (h == [ORACLE] and heap == ["HEAP 2097152 DROPPED 0 RC 0"] and rc == "0" and supm == [want_m] and len(c) == 1)
        if run == 2:
            ok = ok and len(supk) == 1 and int(supk[0], 16) == 0 and len(supn) == 1 and int(supn[0], 16) > 0 and not supa
        else:
            ok = ok and not supn and not supk
        if c:
            cyc[run] = int(c[0])
        out.append(f"run {run}: hash {h} {heap} rc={rc} SUPM {supm} SUPN {[int(x, 16) for x in supn]} SUPK {supk} "
                   f"SUPA {supa} cycles {c} -> {'ok' if ok else 'MISS'}")
        if not ok:
            bad.append(f"speedtest run {run}")
    if len(cyc) == 3:
        p1, s2, p3 = cyc[1], cyc[2], cyc[3]
        spread = abs(p1 - p3) / min(p1, p3) * 100
        rec = max(abs(p1 - C3_RECORD), abs(p3 - C3_RECORD)) / C3_RECORD * 100
        base = (p1 + p3) / 2
        ovh = (s2 - base) / base * 100
        out.append(f"plain runs: {p1:,} / {p3:,} (spread {spread:.4f} %, worst vs C3 record {rec:.4f} %)")
        out.append(f"supervised run: {s2:,}; overhead vs the plain mean {s2 - base:+,.0f} cycles = {ovh:+.4f} %")
        if spread >= 0.1 or rec >= 0.1:
            bad.append("plain-run cycles")
        if not (0 < ovh < 1):
            bad.append("overhead outside (0, 1) %")
        n2 = [int(x, 16) for x in re.findall(r"SUPN:([0-9A-F]+)", tests[2][1])]
        if n2 and n2[0] > 0:
            out.append(f"per preemption: {(s2 - base) / n2[0]:,.0f} cycles over {n2[0]} preemptions")
    print("\n".join(out))
    print("VERDICT: " + ("all predictions hold" if not bad else f"REFUTED: {bad}"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
