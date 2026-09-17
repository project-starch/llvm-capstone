#!/usr/bin/env python3
"""Positive control for movc-cfg-scan.py's reaching-definition walk.

The scanner had no test. It acquired one because an indirect call was found to terminate the backward
walk on its own target register: `jalr rs` is the pseudo for `jalr ra, rs, 0`, and reading operand 0 as
the def made it look like a definition of rs. The walk stopped there and recorded a 'cap' reaching
definition that does not exist, so the reaching-def union was a LOWER BOUND and a site could read
MIXED, or fail to classify, on the strength of a call it merely passed through. It bit only when the
movc's source is callee-saved, because scan() already short-circuits caller-saved registers at its
'call' branch -- which is why it survived every real image anyone had looked at.

This test feeds synthetic disassembly and asserts the verdict FLIPS: shape A must classify INT-ONLY,
and it does so only if the walk passes through the jalr. Run it after any change to defs_reads.
"""
import re, subprocess, sys, types, pathlib

SCANNER = pathlib.Path(__file__).with_name("movc-cfg-scan.py")

SHAPES = {
    # name: (disassembly body, expected verdict substring)
    "A: int def reached THROUGH an indirect call on a callee-saved reg": ("""\
0000000000010000 <shapeA>:
   10000: 01 02 03 04   mv	s3, a0
   10004: 01 02 03 04   jalr	s3
   10008: 01 02 03 04   movc	a0, s3
   1000c: 01 02 03 04   mv	a1, s3
   10010: 01 02 03 04   ret
""", "INT-ONLY"),
    # control: a genuine capability def must still NOT be called INT-ONLY
    "B: genuine capability def of the source": ("""\
0000000000010000 <shapeB>:
   10000: 01 02 03 04   ldc	cs3, 0(sp)
   10004: 01 02 03 04   movc	a0, s3
   10008: 01 02 03 04   mv	a1, s3
   1000c: 01 02 03 04   ret
""", None),
}

def run_shape(text):
    ns = {}
    # Execute only the definitions ABOVE scan(): the scanner's argv check runs at module level and
    # exits, so importing the whole file would end the test rather than run it.
    src = SCANNER.read_text().split("def scan(")[0]
    assert "def defs_reads" in src and "def analyse" in src, "prelude split lost the functions under test"
    fake = types.SimpleNamespace(stdout=text)
    real = subprocess.run
    exec(compile(src, str(SCANNER), "exec"), ns)
    subprocess.run = lambda *a, **k: fake
    try:
        out = []
        ns["print"] = lambda *a, **k: out.append(" ".join(str(x) for x in a))
        funcs = ns["parse"]("ignored")
        insns = list(funcs.values())[0]
        succ, pred = ns["analyse"](insns)
        verdicts = []
        for i, (a, op, ops) in enumerate(insns):
            if op != "movc":
                continue
            r = ns["regs_in"](ops)
            if len(r) < 2 or r[0] == r[1]:
                continue
            rd, rs = r[0], r[1]
            kinds, seen, stack = set(), set(), list(pred[i])
            while stack:
                j = stack.pop()
                if j in seen: continue
                seen.add(j); a2, op2, ops2 = insns[j]
                d, _ = ns["defs_reads"](op2, ops2)
                if op2 in ("jal","jalr","call","cjalr","cjal") and rs in ns["CALLER_SAVED"]:
                    kinds.add("call"); continue
                if d == rs:
                    if op2 in ns["INT_OPS"] and not (op2=="li" and re.search(r",\s*(0x0|0)$", ops2)):
                        kinds.add("int")
                    elif op2 == "li": kinds.add("zero")
                    else: kinds.add("cap")
                    continue
                if not pred[j]: kinds.add("entry")
                stack.extend(pred[j])
            verdicts.append("INT-ONLY" if kinds and kinds <= {"int"} else ",".join(sorted(kinds)) or "none")
        return verdicts
    finally:
        subprocess.run = real

def main():
    bad = 0
    for name, (text, want) in SHAPES.items():
        got = run_shape(text)
        if want is None:
            ok = got and "INT-ONLY" not in got
            print(f"  {'ok ' if ok else 'FAIL'} {name}\n        verdicts={got}  (must NOT be INT-ONLY)")
        else:
            ok = got and want in got
            print(f"  {'ok ' if ok else 'FAIL'} {name}\n        verdicts={got}  (want {want})")
        bad += 0 if ok else 1
    if not SHAPES:
        print("no shapes defined -- that is an error, not a pass", file=sys.stderr); return 2
    print(f"\nmovc-cfg-scan-selftest: {'PASS' if not bad else str(bad)+' FAILURE(S)'}")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
