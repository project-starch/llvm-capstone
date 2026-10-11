#!/usr/bin/env python3
"""Verdict for boot supmon-c5f (and C5u-shaped boots) against PREREG.md: 6 tests -- k800; speedtest x4 with SUPM 0x000
(plain), 0x001 (A, loud), 0x111 (C, quiet + fence before each arm), 0x011 (B, quiet); k800. Reads THIS boot only.
Exit 0 = every prediction holds, 1 = refuted, 2 = no data / incomplete (never a pass)."""
import ast
import re
import sys

ORACLE = "112006 38bb59fd"
WANT_SUPM = {2: "00000000", 3: "00000001", 4: "00000111", 5: "00000011"}


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
    return t[k:].replace("\r", "") if k >= 0 else ""


def main(d):
    s = transcript(d)
    if not s:
        print("NO-RESULT: no boot in driver.log")
        return 2
    tests = re.findall(r"^### TEST (\d)/6 START [^\n]*\n(.*?)^### TEST \1/6 END [^\n]*?rc=(\d+) ###", s, re.S | re.M)
    out, bad, cyc = [f"boot trace {re.findall(r'BT0\d', s)}; login {'login:' in s}"], [], {}
    if len(tests) != 6:
        print("\n".join(out))
        print(f"NO-RESULT: {len(tests)} of 6 tests completed")
        return 2
    for n, body, rc in tests:
        n = int(n)
        if n in (1, 6):
            r = re.findall(r"RESULT k800 retval=(\S+) cycles=(\d+)", body)
            ok = bool(r) and r[0][0] == "4" and rc == "0"
            out.append(f"test {n} k800 {r} rc={rc} -> {'ok' if ok else 'MISS'}")
            bad += [] if ok else [f"k800 test {n}"]
            continue
        h = re.findall(r"Verification Hash: (\d+ \w{8})", body)
        heap = re.findall(r"HEAP \d+ DROPPED \d+ RC \d+", body)
        c = re.findall(r"SPEEDTEST1-CYCLES (\d+)", body)
        supm = re.findall(r"SUPM:([0-9A-F]+)", body)
        supn = [int(x, 16) for x in re.findall(r"SUPN:([0-9A-F]+)", body)]
        supk = re.findall(r"SUPK:([0-9A-F]+)", body)
        ok = h == [ORACLE] and heap == ["HEAP 2097152 DROPPED 0 RC 0"] and rc == "0" and supm == [WANT_SUPM[n]] and len(c) == 1
        if n > 2:   # supervised: the final SUPK is 0, and preemptions happened
            ok = ok and len(supn) == 1 and supn[0] > 0 and supk and int(supk[-1], 16) == 0
        if c:
            cyc[n] = int(c[0])
        out.append(f"test {n} SUPM {supm} hash {h} rc={rc} SUPN {supn} final SUPK {supk[-1:] } cycles {c} -> {'ok' if ok else 'MISS'}")
        bad += [] if ok else [f"speedtest test {n}"]
    if len(cyc) == 4:
        base = cyc[2]
        for n, name in ((3, "A loud (UART inside the bracket)"), (4, "C quiet + fence per arm"), (5, "B quiet")):
            out.append(f"{name}: {cyc[n]:,} cycles = {(cyc[n] - base) / base * 100:+.3f} % against plain {base:,}")
    print("\n".join(out))
    print("VERDICT: " + ("all predictions hold" if not bad else f"REFUTED: {bad}"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
