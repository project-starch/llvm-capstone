#!/usr/bin/env python3
"""Grade the GDB reader's validation boot (PREREG addendum 2) from its board transcript.

usage: gdb-grade.py <board.txt> E1|E2

Reads the `tickcost-addr tick|cvprof_body|ktime_get 0x...` console lines and the LAST "===== ... wedge gdb ====="
block (`GDB pc sample i 0x...`, `GDB <csr> 0x...`, `GDB insn ...`). E2: READER VALID (exit 0) iff every pc sample lies
in [tick + 0x154, tick + 0x160), the register loop; INVALID (exit 1) on any sample outside it, 0xca11ab1ebadcab1e, or
UNREAD. E1: each sample is classified (the .L83 loop tick + 0x170..0x17c, ktime_get's first 0x400 bytes, M-mode
firmware [0x80000000, 0x80200000), or OTHER) and printed with mstatus.MPP; exit 0 if every sample is in one of the
three, 1 otherwise. Exit 2 = VOID or no data (no addresses printed, no gdb block, a probe failure line).
"""
import re
import sys

JUNK = 0xca11ab1ebadcab1e


def main(argv):
    if len(argv) != 3 or argv[2] not in ("E1", "E2"):
        print(__doc__); return 2
    path, arm = argv[1], argv[2]
    try:
        text = open(path, errors="replace").read().replace("\r", "")
    except OSError as e:
        print(f"VOID: cannot read {path}: {e}"); return 2
    addr = {m.group(1): int(m.group(2), 16)
            for m in re.finditer(r"tickcost-addr (tick|cvprof_body|ktime_get) 0x([0-9a-f]+)", text)}
    if "tick" not in addr or addr["tick"] == 0:
        print(f"VOID: no nonzero 'tickcost-addr tick' line in {path} (kptr_restrict? the console?)"); return 2
    if addr["tick"] >= 0xffffffff80000000:
        print(f"VOID: tick {addr['tick']:#x} is not in module space (riscv64 modules lie below the kernel image at "
              "0xffffffff80000000): a kernel symbol was picked"); return 2
    blocks = re.split(r"===== .*? wedge gdb =====", text)
    if len(blocks) < 2:
        print("VOID: no wedge gdb block"); return 2
    g = blocks[-1].split("=====")[0]
    if "GDB probe FAILED" in g:
        print("READER INVALID (the probe failed): " + re.search(r"GDB probe FAILED.*", g).group(0)); return 1
    pcs = [(int(i), v) for i, v in re.findall(r"GDB pc sample (\d+) (0x[0-9a-f]+|UNREAD)", g)]
    csr = dict(re.findall(r"GDB (ra|sp|mcause|mepc|mtval|mstatus|scause|sepc) (0x[0-9a-f]+|UNREAD)", g))
    t = addr["tick"]
    print("addresses: " + " ".join(f"{k} {v:#x}" for k, v in sorted(addr.items())))
    print("csrs: " + " ".join(f"{k} {v}" for k, v in csr.items()))
    mpp = None
    if csr.get("mstatus", "UNREAD") != "UNREAD":
        mpp = (int(csr["mstatus"], 16) >> 11) & 3
        print(f"mstatus.MPP = {mpp}")
    if not pcs:
        print("VOID: the gdb block has no pc samples"); return 2

    def cls(v):
        if v == "UNREAD":
            return "UNREAD"
        x = int(v, 16)
        if x == JUNK:
            return "JUNK"
        if t + 0x154 <= x < t + 0x160:
            return "L85-register-loop"
        if t + 0x170 <= x < t + 0x17c:
            return "L83-rdtime-loop"
        if "ktime_get" in addr and addr["ktime_get"] <= x < addr["ktime_get"] + 0x400:
            return "ktime_get"
        if 0x80000000 <= x < 0x80200000:
            return "firmware-M"
        return "OTHER"
    kinds = []
    for i, v in pcs:
        k = cls(v); kinds.append(k)
        off = f" (tick+{int(v, 16) - t:#x})" if v != "UNREAD" and k.startswith("L8") else ""
        print(f"  sample {i}: {v} {k}{off}")
    for ln in re.findall(r"GDB insn .*", g):
        print("  " + ln)
    if arm == "E2":
        if all(k == "L85-register-loop" for k in kinds) and len(kinds) == 8:
            print("READER VALID: 8/8 samples in the register loop"); return 0
        print(f"READER INVALID: {sum(k == 'L85-register-loop' for k in kinds)}/{len(kinds)} in the loop; kinds {kinds}")
        return 1
    ok = all(k in ("L83-rdtime-loop", "ktime_get", "firmware-M") for k in kinds)
    print(("E1 AS PRE-REGISTERED" if ok else "E1 FINDING: a sample outside loop / ktime_get / firmware") +
          f": " + ", ".join(f"{k} {kinds.count(k)}" for k in sorted(set(kinds))))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
