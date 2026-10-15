#!/usr/bin/env python3
"""No floating point in a silicon domain image (B5, docs/plans/b0-silicon-delegated-runtime.md).

A delegated domain starts with mstatus.FS Off and the domain switcher saves no FP state, so a single F/D instruction
traps to a monitor that cannot serve it, and the board wedges. Every silicon build is soft-float (+m +a, no +f/+d);
this checks the IMAGE instead of trusting the flags.

Walks every executable section's instruction stream (16-bit compressed and 32-bit forms) and reports:
  - LOAD-FP 0x07, STORE-FP 0x27, FMADD/FMSUB/FNMSUB/FNMADD 0x43/0x47/0x4b/0x4f, OP-FP 0x53;
  - compressed c.fld / c.fsd / c.fldsp / c.fsdsp (RV64: quadrant 0 and 2, funct3 001 and 101);
  - a CSR instruction on fflags, frm or fcsr (0x001-0x003).
Exit 0 = none found; 1 = found (each listed); 2 = the image could not be read. Self-test: --self-test."""
import struct
import sys

FP_OPCODES = {0x07: "LOAD-FP", 0x27: "STORE-FP", 0x43: "FMADD", 0x47: "FMSUB", 0x4b: "FNMSUB", 0x4f: "FNMADD",
              0x53: "OP-FP"}


def exec_sections(data):
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a little-endian ELF64")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3a)
    for i in range(shnum):
        off = shoff + i * shentsize
        _name, sh_type, flags, addr, sh_offset, size = struct.unpack_from("<IIQQQQ", data, off)
        if sh_type == 1 and flags & 0x4 and size:            # PROGBITS, SHF_EXECINSTR
            yield addr, data[sh_offset:sh_offset + size]


def scan(code, base):
    hits, i = [], 0
    while i + 2 <= len(code):
        h, = struct.unpack_from("<H", code, i)
        if h & 3 != 3:                                         # 16-bit compressed
            quad, f3 = h & 3, h >> 13
            if (quad in (0, 2)) and f3 in (1, 5):
                hits.append((base + i, h, "c.fld/c.fsd" + ("sp" if quad == 2 else "")))
            i += 2
            continue
        if i + 4 > len(code):
            break
        w, = struct.unpack_from("<I", code, i)
        op = w & 0x7f
        if op in FP_OPCODES:
            hits.append((base + i, w, FP_OPCODES[op]))
        elif op == 0x73 and (w >> 12) & 7 and (w >> 20) in (1, 2, 3):
            hits.append((base + i, w, "csr fflags/frm/fcsr"))
        i += 4
    return hits


def self_test():
    bad = 0
    fadd_d = struct.pack("<I", 0x02b57553)                    # fadd.d fa0, fa0, fa1  (OP-FP)
    fld = struct.pack("<I", 0x00053507)                       # fld fa0, 0(a0)        (LOAD-FP)
    frcsr = struct.pack("<I", 0x00302573)                     # csrr a0, fcsr
    c_fld = struct.pack("<H", 0x2108)                         # c.fld fa0, 0(a0)
    add = struct.pack("<I", 0x00b50533)                       # add a0, a0, a1
    c_add = struct.pack("<H", 0x952e)                         # c.add a0, a1
    for name, buf, want in (("fadd.d", fadd_d, 1), ("fld", fld, 1), ("csrr fcsr", frcsr, 1), ("c.fld", c_fld, 1),
                            ("add", add, 0), ("c.add", c_add, 0), ("mixed", c_add + fadd_d + add, 1)):
        got = len(scan(buf, 0))
        if got != want:
            print(f"self-test FAIL: {name} gave {got} hits, want {want}")
            bad += 1
    print("no-fp-gate self-test: " + ("every control fires" if not bad else f"{bad} FAILED"))
    return 1 if bad else 0


def main(argv):
    if argv[1:] == ["--self-test"]:
        return self_test()
    rc = 0
    for path in argv[1:]:
        try:
            data = open(path, "rb").read()
            hits = [h for addr, code in exec_sections(data) for h in scan(code, addr)]
        except (OSError, ValueError, struct.error) as e:
            print(f"no-fp-gate: {path}: cannot read ({e})")
            return 2
        if hits:
            rc = 1
            print(f"no-fp-gate: {path}: {len(hits)} floating-point instruction(s)")
            for addr, w, what in hits[:20]:
                print(f"  {addr:#x}: {w:#010x} {what}")
        else:
            print(f"no-fp-gate: {path}: no floating-point instructions")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
