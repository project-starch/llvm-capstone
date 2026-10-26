#!/usr/bin/env python3
"""r64-census.py <image | @list file> ...: static reach of the divide/sqrt defects R-64 and R-66 (filed on dev
db755fec4825 and 5d1435c245d9). Per distinct image (by sha256), from `llvm-objdump -d --mattr=+f,+d` (without the
attribute every FP instruction reads 0):
- R-64 channels: flag READS (frflags, frcsr, csrr* of fflags/fcsr with a non-zero rd), flag writes, rounding-mode
  writes (fsrm, fsrmi, fscsr, csrw* of frm/fcsr), and every fdiv/fsqrt with its rm field decoded from the ENCODING
  (byte or word objdump format), so a static mode (rm != 7) counts whatever the mnemonic printing does;
- R-66: every fdiv.s / fsqrt.s classified by the last writer of each source f register, by a LINEAR backward scan
  inside its function (control flow ignored, so this is a candidate list, not a proof): D (an FP64 op, fld or
  fmv.d.x: not NaN-boxed, the R-66 signature), S (an FP32 op or flw), or ? (live into the function, or after a call).
Every hit is reported with its enclosing function, so the runtime's fenv (fegetenv, __fe_getround) can be told from
program code.
Images are ELF e_machine 243 (RISC-V) or 259 (Capstone .dom); relocatable objects (ET_REL other than .ko) are build
intermediates and skipped, with the count printed. Two controls run first and must read exactly: a synthetic object
and fp-divsqrt.dom (fdiv.d 1, fsqrt.d 1, both dyn). Exit 2 on a control failure, no image, an objdump failure, or an
fdiv/fsqrt line whose encoding does not parse. Written 2026-10-26."""
import collections
import hashlib
import multiprocessing
import re
import subprocess
import sys
from pathlib import Path

H = Path("/mnt/hdd") / Path.home().name / "capstone"
O = str(H / "cc-prmerged/bin/llvm-objdump")
MC = str(H / "cc-prmerged/bin/llvm-mc")
FPDIV_CTL = str(H / "perf/fpdiag/fp-divsqrt.dom")
INS = re.compile(r"^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} ){4}|(?:[0-9a-f]{2} ){2}|[0-9a-f]{8}\s|[0-9a-f]{4}\s)\s*([a-z.0-9]+)\s*(.*)$")
FN = re.compile(r"^[0-9a-f]+ <(.+)>:")
CSR = r"(fflags|frm|fcsr|0x1|0x2|0x3)\b"
# the runtime fenv functions every image carries (the perf census found exactly these four)
FENV = {"fegetenv", "fesetenv", "__fe_getround", "__fe_raise_inexact"}
RMNAME = {0: "rne", 1: "rtz", 2: "rdn", 3: "rup", 4: "rmm", 7: "dyn"}
FREG = re.compile(r"^(f\d+|ft\d+|fs\d+|fa\d+)$")
FALIAS = {**{f"ft{i}": f"f{i}" for i in range(8)}, **{f"fs{i}": f"f{8 + i}" for i in range(2)},
          **{f"fa{i}": f"f{10 + i}" for i in range(8)}, **{f"fs{i}": f"f{16 + i}" for i in range(2, 12)},
          **{f"ft{i}": f"f{20 + i}" for i in range(8, 12)}}


def freg(r):
    r = r.strip()
    return FALIAS.get(r, r) if FREG.match(r) else None


def word(enc):
    p = enc.split()
    return int("".join(reversed(p)), 16) if len(p) in (2, 4) else int(p[0], 16)


def dest_prec(mn):
    """Precision an f-register-writing mnemonic leaves in its destination: 'D', 'S' or 'H'."""
    p = mn.split(".")
    if p[0] in ("fld", "c"):
        return "D" if mn.endswith("fld") or mn == "c.fld" or mn == "c.fldsp" else None
    if p[0] == "flw":
        return "S"
    if p[0] in ("fcvt", "fmv") and len(p) == 3:
        x = p[1]
    else:
        x = p[-1]
    return {"d": "D", "s": "S", "w": "S", "h": "H", "q": "D"}.get(x)


CALLER_SAVED = {f"f{i}" for i in (*range(0, 8), *range(10, 18), *range(28, 32))}


def fwriter(mn, ops, opl):
    """(register, class) if this instruction writes an f register, else None."""
    if opl and freg(opl[0]) and mn.startswith(("f", "c.f")) and not ("(" in ops and not re.match(r"^(c\.)?fl", mn)):
        return freg(opl[0]), dest_prec(mn) or "?"
    return None


def dataflow_d_sites(body):
    """body: [(addr, mn, ops)] of one function. Reaching definitions of every f register over the function's CFG
    (branch targets from the operand, fall-through unless j/jr/ret/tail; a call leaves the caller-saved registers
    'call'; entry leaves every register 'in'). Returns {addr: classes reaching the sources} for each fdiv.s/fsqrt.s."""
    idx = {a: i for i, (a, _, _) in enumerate(body)}
    def succ(i):
        a, mn, ops = body[i]
        tgt = re.search(r"(?:^|,\s*)0x([0-9a-f]+)(?:\s|$)", ops)   # the target, never the <sym+0x..> offset
        t = idx.get(int(tgt[1], 16)) if tgt else None
        if mn in ("ret", "jr", "tail") or (mn == "jalr" and ops.startswith("zero")) or mn.startswith("c.jr"):
            return []
        if mn in ("j", "c.j") or (mn == "jal" and ops.startswith("zero")):
            return [t] if t is not None else []
        out = [i + 1] if i + 1 < len(body) else []
        if mn.startswith(("b", "c.b")) and t is not None:
            out.append(t)
        return out
    def transfer(i, st):
        a, mn, ops = body[i]
        opl = [o.strip() for o in ops.split(",")]
        if mn in ("jal", "jalr", "call") or mn == "c.jalr":
            if not ops.startswith("zero"):
                st = dict(st); [st.__setitem__(r, frozenset({"call"})) for r in CALLER_SAVED]
            return st
        w = fwriter(mn, ops, opl)
        if w:
            st = dict(st); st[w[0]] = frozenset({w[1]})
        return st
    ALL = [f"f{i}" for i in range(32)]
    IN = [None] * len(body)
    IN[0] = {r: frozenset({"in"}) for r in ALL}
    work = [0]
    while work:
        i = work.pop()
        out = transfer(i, IN[i])
        for j in succ(i):
            if IN[j] is None:
                IN[j] = dict(out); work.append(j)
            else:
                merged = {r: IN[j][r] | out[r] for r in ALL}
                if merged != IN[j]:
                    IN[j] = merged; work.append(j)
    res = {}
    for i, (a, mn, ops) in enumerate(body):
        if mn in ("fdiv.s", "fsqrt.s"):
            if IN[i] is None:        # entered only indirectly (a jump table): no state, so never silently clean
                res[a] = (mn, ops, frozenset({"unreached"})); continue
            opl = [o.strip() for o in ops.split(",")]
            srcs = [freg(o) for o in opl[1:3 if mn == "fdiv.s" else 2] if freg(o)]
            res[a] = (mn, ops, frozenset().union(*[IN[i][r] for r in srcs]))
    return res


def census(img):
    out = subprocess.run([O, "-d", "--mattr=+f,+d", img], capture_output=True, text=True)
    if out.returncode:
        return ("ERROR", f"objdump rc {out.returncode} on {img}: {out.stderr[:200]}")
    if "Disassembly" not in out.stdout:
        return (collections.Counter({"no-code": 1}), {})
    fn = "?"; c = collections.Counter(); where = collections.defaultdict(set); last = {}
    # every f-register writer per function, whatever the path: the conservative R-66 check of a linear "S"
    fwrites = collections.defaultdict(lambda: collections.defaultdict(set)); lin_s = []
    bodies = collections.defaultdict(list)
    for ln in out.stdout.splitlines():
        m = FN.match(ln)
        if m:
            if not m[1].startswith(".L"):
                fn = m[1]; last = {}
            continue
        m = INS.match(ln)
        if not m:
            if re.search(r"\t(fdiv|fsqrt)\.", ln):
                return ("ERROR", f"unparsed divsqrt line in {img}: {ln}")
            continue
        enc, mn, ops = m[1], m[2], m[3].strip()
        opl = [o.strip() for o in ops.split(",")]
        bodies[fn].append((int(ln.split(":")[0], 16), mn, ops))
        key = None
        if mn in ("frflags", "frcsr"):
            key = "flag-read"
        elif mn in ("fsrm", "fsrmi", "fscsr"):
            key = "rm-write"
        elif mn in ("fsflags", "fsflagsi"):
            key = "flag-write"
        elif mn == "frrm":
            key = "rm-read"
        elif mn.startswith("csr"):
            if mn in ("csrr", "csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci") and \
                    re.search(r"^[a-z0-9]+,\s*" + CSR, ops) and not ops.startswith("zero"):
                key = "flag-read" if re.search(r",\s*(fflags|fcsr|0x1|0x3)\b", ops) else "rm-read"
            elif re.search(r"^" + CSR, ops) and mn in ("csrw", "csrwi", "csrs", "csrsi", "csrc", "csrci"):
                key = "rm-write" if re.search(r"^(frm|fcsr|0x2|0x3)\b", ops) else "flag-write"
        if key:
            c[key] += 1; where[f"{key} {mn} {ops}"].add(fn)
        if re.match(r"^(fdiv|fsqrt)\.[sdhq]$", mn):
            rm = (word(enc) >> 12) & 7
            c[f"{mn} rm={RMNAME.get(rm, rm)}"] += 1
            if rm != 7:
                where[f"STATIC-RM {mn} {RMNAME.get(rm, rm)}"].add(fn)
            if mn in ("fdiv.s", "fsqrt.s"):
                srcs = [freg(o) for o in opl[1:3 if mn == "fdiv.s" else 2]]
                cls = [last.get(s, "?") for s in srcs if s]
                k = "D" if "D" in cls else ("?" if "?" in cls or "H" in cls else "S")
                c[f"r66 {mn} src={k}"] += 1
                if k == "S":
                    lin_s.append((fn, mn, ops, ln.split(':')[0].strip(), [s for s in srcs if s]))
                if k != "S":
                    where[f"R66-{'CANDIDATE' if k == 'D' else 'UNRESOLVED'} {mn} {ops} @{ln.split(':')[0].strip()}"].add(fn)
        # f-register writers (after the R-66 read of this instruction's sources)
        if mn in ("jal", "jalr", "call", "tail") or mn.startswith("c.jal"):
            last = {r: v for r, v in last.items() if r in {f"f{i}" for i in (8, 9, *range(18, 28))}}
        elif opl and freg(opl[0]) and mn.startswith(("f", "c.f")) and not ("(" in ops and not re.match(r"^(c\.)?fl", mn)):
            p = dest_prec(mn)
            last[freg(opl[0])] = p or "?"
            fwrites[fn][freg(opl[0])].add(p or "?")
    for f, mn, ops, at, srcs in lin_s:
        if any("D" in fwrites[f][r] for r in srcs):
            c[f"r66 {mn} S-with-D-writer-in-fn"] += 1
            where[f"R66-PATHCHECK {mn} {ops} @{at}"].add(f)
    # the sound check: reaching definitions over each function's CFG, wherever the function has an FP64 writer
    for f, body in bodies.items():
        if not any("D" in v for v in fwrites[f].values()):
            continue
        for a, (mn, ops, cls) in dataflow_d_sites(body).items():
            if "unreached" in cls:
                c[f"r66 {mn} unreached (dataflow)"] += 1
                where[f"R66-DATAFLOW-UNREACHED {mn} {ops} @{a:x}"].add(f)
            if "D" in cls:
                c[f"r66 {mn} D-reaches (dataflow)"] += 1
                where[f"R66-DATAFLOW-D {mn} {ops} @{a:x} reaching {sorted(cls)}"].add(f)
    return (c, where)


def main():
    tmp = Path("/tmp") / "r64ctl-census"
    tmp.mkdir(exist_ok=True)
    (tmp / "ctl.s").write_text("\t.text\n\t.globl f\nf:\n\tfrflags a0\n\tfsrmi 2\n\tfsrm a1\n\tcsrr a2, fflags\n"
                               "\tfdiv.d fa0, fa1, fa2, rtz\n\tfsqrt.s fa0, fa1, rup\n\tfdiv.s fa0, fa1, fa2\n"
                               "\tfmv.d.x fa3, a0\n\tfdiv.s fa4, fa3, fa2\n\tflw fa5, 0(a0)\n\tfsqrt.s fa6, fa5\n"
                               "\tfld fa7, 8(a0)\n\tfsw fa5, 0(a1)\n\tfdiv.s ft0, fa5, fa7\n"
                               "\tfcvt.s.d ft1, fa7\n\tfsqrt.s ft2, ft1\n\tcall g\n\tfsqrt.s ft3, ft1\n"
                               "\tcsrsi mip, 0x2\n\tcsrci mip, 0x1\n\tcsrrs a3, fcsr, zero\n"
                               "\tflw fs1, 0(a0)\n\tfsqrt.s fs2, fs1\n\tfld fs1, 8(a0)\n\tret\n"
                               "\t.globl g3\ng3:\n\tflw fs3, 0(a0)\n\tbeqz a1, 1f\n\tfld fs3, 8(a0)\n1:\n\tfsqrt.s fs4, fs3\n\tret\n"
                               "\t.globl g4\ng4:\n\tfld fs5, 0(a0)\n\tj 2f\n3:\n\tfsqrt.s fs6, fs5\n\tret\n2:\n\tflw fs5, 0(a0)\n\tj 3b\n"
                               "\t.globl g5\ng5:\n\tfld fs7, 0(a0)\n\tjr a2\n\tfsqrt.s fs8, fs7\n\tret\n")
    subprocess.run([MC, "-triple=riscv64", "-mattr=+f,+d,-relax", "-filetype=obj", str(tmp / "ctl.s"), "-o", str(tmp / "ctl.o")],
                   check=True)
    want = {"flag-read": 3, "rm-write": 2, "fdiv.d rm=rtz": 1, "fsqrt.s rm=rup": 1, "fdiv.s rm=dyn": 3,
            "r66 fsqrt.s src=?": 2, "r66 fdiv.s src=?": 1, "r66 fdiv.s src=D": 2, "r66 fsqrt.s src=S": 3,
            "r66 fsqrt.s S-with-D-writer-in-fn": 1,
            "fsqrt.s rm=dyn": 7, "r66 fsqrt.s src=D": 3, "r66 fdiv.s D-reaches (dataflow)": 2,
            "r66 fsqrt.s D-reaches (dataflow)": 1, "r66 fsqrt.s unreached (dataflow)": 1}
    got = census(str(tmp / "ctl.o"))
    df = sorted(k for k in (got[1] if got[0] != "ERROR" else {}) if k.startswith("R66-DATAFLOW-D"))
    want_df = ["R66-DATAFLOW-D fdiv.s fa4, fa3, fa2 @20 reaching ['D', 'in']",
               "R66-DATAFLOW-D fdiv.s ft0, fa5, fa7 @34 reaching ['D', 'S']",
               "R66-DATAFLOW-D fsqrt.s fs4, fs3 @74 reaching ['D', 'S']"]
    if df != want_df:
        print(f"CONTROL FAILED dataflow: {df} != {want_df}"); sys.exit(2)
    if got[0] == "ERROR" or dict(got[0]) != want:
        print(f"CONTROL FAILED synthetic: {got[0] if got[0] == 'ERROR' else dict(got[0])} != {want}"); sys.exit(2)
    got = census(FPDIV_CTL)
    if got[0] == "ERROR" or got[0]["fdiv.d rm=dyn"] != 1 or got[0]["fsqrt.d rm=dyn"] != 1 or \
            any("rm=" in k and "dyn" not in k for k in got[0]):
        print(f"CONTROL FAILED fp-divsqrt.dom: {got}"); sys.exit(2)
    print("CONTROLS fired: synthetic object (every channel, R-66 D/S/?) and fp-divsqrt.dom (fdiv.d 1, fsqrt.d 1, dyn)")
    imgs = []
    for a in sys.argv[1:]:
        imgs += [l.strip() for l in open(a[1:]) if l.strip()] if a.startswith("@") else [a]
    seen = {}; other = collections.Counter()
    for p in imgs:
        b = Path(p).read_bytes()
        if b[:4] != b"\x7fELF":
            other["not ELF"] += 1; continue
        mach, etype = int.from_bytes(b[18:20], "little"), int.from_bytes(b[16:18], "little")
        if mach not in (243, 259):
            other[f"e_machine {mach}"] += 1; continue
        if etype == 1 and not p.endswith(".ko") and not p.endswith(".ko.stock"):
            other["relocatable .o"] += 1; continue
        seen.setdefault(hashlib.sha256(b).hexdigest(), p)
    print(f"SKIPPED: {dict(other)}")
    if not seen:
        print("NO DATA: no image among the arguments"); sys.exit(2)
    order = sorted(seen.items(), key=lambda x: x[1])
    with multiprocessing.Pool(8) as pool:
        res = pool.map(census, [p for _, p in order], chunksize=16)
    errs = [r[1] for r in res if r[0] == "ERROR"]
    tot = collections.Counter(); allw = collections.defaultdict(set); imgs_with = collections.Counter()
    for (h, p), r in zip(order, res):
        if r[0] == "ERROR":
            continue
        c, w = r
        print(f"IMAGE {h[:16]} {p} {dict(sorted(c.items()))}")
        for k, v in sorted(w.items()):
            rest = sorted(v - FENV)
            if rest:
                print(f"HIT {h[:16]} {p} {k} <- {', '.join(rest)[:300]}")
        tot.update(c)
        for k in c:
            imgs_with[k] += 1
        for k, v in w.items():
            allw[k] |= v
    print(f"TOTAL {len(imgs)} paths, {len(seen)} distinct images: {dict(sorted(tot.items()))}")
    print(f"IMAGES WITH: {dict(sorted(imgs_with.items()))}")
    for k, v in sorted(allw.items()):
        print(f"  {k} <- {', '.join(sorted(v))[:400]}")
    for e in errs:
        print(f"NO DATA: {e}")
    sys.exit(2 if errs else 0)


if __name__ == "__main__":
    main()
