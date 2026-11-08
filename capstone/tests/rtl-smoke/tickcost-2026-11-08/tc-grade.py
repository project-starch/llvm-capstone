#!/usr/bin/env python3
"""Grade one tickcost boot from its board transcript (guest/board.txt), against PREREG.md.

usage: tc-grade.py <board.txt> E1|E2

Stage output is read from the raw console lines AND the base64 tails of the driver's @@HB heartbeats (a wedged stage
never prints its output block). Exit 0 = every arm read as pre-registered; 1 = a FINDING (an arm that should return
did not, or the wedge reads differently from the prediction: printed); 2 = VOID or no data (says where it looked).
The 224 verdict (does E reproduce the cvprof wedges' 0xff) is printed as a reading, not graded.
"""
import base64, re, sys

ARMS_RETURN = ["A", "B", "C", "D"]


def text_of(path):
    raw = open(path, errors="replace").read().replace("\r", "")
    tails = []
    for m in re.finditer(r"^@@HB \d+ .*?tail=([A-Za-z0-9+/=]+)\s*$", raw, re.M):
        try:
            tails.append(base64.b64decode(m.group(1)).decode(errors="replace"))
        except Exception:
            pass
    return raw, raw + "\n" + "\n".join(tails)


def is_insn(w):
    if w == 0 or w >> 32:
        return False
    if (w & 3) != 3:
        return w < 0x10000            # a compressed instruction
    return (w & 0x7f) in (0x03, 0x07, 0x0f, 0x13, 0x17, 0x1b, 0x23, 0x27, 0x2f, 0x33, 0x37, 0x3b, 0x43, 0x47, 0x4b,
                          0x4f, 0x53, 0x5b, 0x63, 0x67, 0x6f, 0x73, 0x0b, 0x2b, 0x7b)


def main(argv):
    if len(argv) != 3 or argv[2] not in ("E1", "E2"):
        print(__doc__); return 2
    path, earm = argv[1], argv[2]
    try:
        raw, text = text_of(path)
    except OSError as e:
        print(f"VOID: cannot read {path}: {e}"); return 2
    if "@@TC INSMOD-FAIL" in text:
        print("VOID: insmod failed"); return 2
    if not re.search(r"tickcost: loaded", text):
        print(f"VOID: no 'tickcost: loaded' in {path} (raw console or heartbeat tails)"); return 2
    # Arm completion comes from the KERNEL's console lines (KERN_EMERG, printed at once), paired in order: each
    # `tickcost: enable` is an arm (B, C, D, then E), and it returned iff a `tickcost: disable` follows it. The
    # @@TC markers reach the transcript only through heartbeat tails, which stop at the last heartbeat before a wedge
    # (8 Nov: they ended at arm C's start), so they are not used for completion. A stage-side marker is shown only.
    ev = [(m.group(1), m.group(2)) for m in re.finditer(
        r"tickcost: (enable|disable) mode=(\d)", text.split("===== vbatch wedge apertures =====")[0])]
    arms, cur = [], None
    for kind, mode in ev:
        if kind == "enable":
            cur = {"mode": mode, "returned": False, "line": None}; arms.append(cur)
        elif cur is not None and not cur["returned"]:
            cur["returned"] = True
    dis = [m.group(0) for m in re.finditer(r"tickcost: disable mode=\d calls=\d+ maxbody_us=\d+.*", text)]
    names = ["B", "C", "D", earm]
    alive = {"A": "@@TC A ALIVE" in text or bool(ev)}
    for i, n in enumerate(names):
        alive[n] = i < len(arms) and arms[i]["returned"]
    modes_ok = [a["mode"] for a in arms] == ["0", "1", "0", "2" if earm == "E1" else "3"][:len(arms)]
    started_e = bool(re.search(r"tickcost: stuck (rdtime|register)", text)) and len(arms) >= 4
    scans = re.findall(r"tickcost: scan (user|kernel) rep=(\d) probes=(\d+) ns=(\d+)", text)
    print(f"arms by enable/disable: {[(n, a['mode'], a['returned']) for n, a in zip(names, arms)]}  modes in order {modes_ok}"
          f"  E stuck line {started_e}  scan lines {len(set(scans))}")
    for s in sorted(set(scans)):
        print(f"  scan {s[0]} rep {s[1]} probes {s[2]} {int(s[3]) / 1e6:.3f} ms")
    for d in dis:
        print("  " + d)
    if len(set(scans)) != 6:
        print("VOID: arm A did not print six scan lines"); return 2
    if not modes_ok:
        print("VOID: the arms did not run in the pre-registered order/modes"); return 2
    if not alive["B"]:
        print("VOID: arm B (3 ms callback) did not return"); return 2
    finding = []
    for a in ("C", "D"):
        if not alive[a]:
            finding.append(f"arm {a} did not return (the slow-tick path stopped this kernel)")
    if finding:
        for f in finding:
            print("FINDING: " + f)
        return 1
    if not started_e:
        print(f"VOID: arm {earm} never reached its stuck line"); return 2
    if alive[earm]:
        finding.append(f"arm {earm} RETURNED, but its callback never returns by construction")
    # the wedge block (the driver's last one)
    blocks = raw.split("===== vbatch wedge apertures =====")
    if len(blocks) < 2:
        print("VOID: no wedge aperture block"); return 2 if not finding else 1
    w = blocks[-1]
    tl = re.search(r"^sw=255 TRAP LOG \{seen,mcause\[6:0\]\} 0x([0-9a-f]{2})", w, re.M)
    tval = re.search(r"^trap-log tval 0x([0-9a-f]+)", w, re.M)
    mepc = re.search(r"^trap-log mepc 0x([0-9a-f]+)", w, re.M)
    s224 = [int(v, 16) for v in re.findall(r"^sw=224 (?:repeat \d+ )?\{?[^\n]*?0x([0-9a-f]{2}) [01]{8}$", w, re.M)]
    print(f"trap log {tl.group(1) if tl else 'UNREAD'}  mepc {mepc.group(1) if mepc else 'UNREAD'}  "
          f"tval {tval.group(1) if tval else 'UNREAD'}")
    print(f"224 samples ({len(s224)}): " + " ".join(f"{v:02x}" for v in s224))
    for u in re.findall(r"^UNSTABLE .*$", w, re.M):
        print("  " + u)
    if not tl or tl.group(1) != "85":
        finding.append(f"trap log {tl.group(1) if tl else 'UNREAD'}, predicted 0x85 (the STI that entered the tick)")
    if not tval or not is_insn(int(tval.group(1), 16)):
        finding.append("tval does not decode as an instruction")
    if s224:
        n = len(s224)
        exc = sum(v >> 7 & 1 for v in s224); privm = sum(v & 1 for v in s224); ff = sum(v == 0xff for v in s224)
        print(f"224: ex_commit.valid {exc}/{n}, privM {privm}/{n}, 0xff {ff}/{n}")
        print("READING: " + ("this stuck-in-the-tick arm REPRODUCES the cvprof wedges' 0xff" if ff * 2 > n
                             else "this stuck-in-the-tick arm does NOT reproduce the cvprof wedges' 0xff"))
        if earm == "E2" and (exc or privm):
            finding.append("E2 (register loop, no traps) read ex_commit.valid or privM set, predicted 0")
    else:
        finding.append("aperture 224 unread")
    if finding:
        for f in finding:
            print("FINDING: " + f)
        return 1
    print("AS PRE-REGISTERED: A-D returned; " + earm + " stuck with trap log 0x85 and an instruction tval")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
