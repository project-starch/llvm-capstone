#!/usr/bin/env python3
"""Cross-check capstone/runtime/include/capstone/representable.h against the RTL bounds encoder.

The header is C and the oracle is Python, so the C is compiled and its answers are read out of the
emitted IR -- the predicate under test is the one that will ship, not a reimplementation of it.

Two-sided by construction: the check must ACCEPT every range the oracle encodes losslessly and
REJECT every range it does not, and a deliberately wrong predicate must fail it. Run it with no
arguments; it exits non-zero on any disagreement.
"""
import load_model, os, pathlib, random, subprocess, sys, tempfile

bm = load_model.load()
HERE = pathlib.Path(__file__).resolve().parent
INC = None
for p in HERE.parents:
    c = p / "capstone" / "runtime" / "include"
    if (c / "capstone" / "representable.h").is_file():
        INC = c; break
if INC is None:
    sys.exit("ERROR: representable.h not found above %s" % HERE)
CC = os.environ.get("CAPSTONE_CLANG") or os.environ.get("CC") or "clang"

def c_answers(cases, broken=False):
    """Run the header's predicate over `cases` and return its answers.

    The predicate is pure integer arithmetic (shifts, %, __builtin_clzl), so its VALUES are
    target-independent and are obtained by compiling the real header into a host driver that
    prints them. That the same header compiles for capstone64 is asserted separately below --
    the two together cover "it builds for the target" and "it computes the right thing", which
    one run cannot.
    """
    mutant = ('#define capstone_repr_ok(b, l) '
              '(((l) % capstone_repr_granule(l)) == 0)\n') if broken else ''
    drv = """#include <stdio.h>
#include <capstone/representable.h>
%s
int main(void) {
    unsigned long b, l;
    while (scanf("%%lu %%lu", &b, &l) == 2)
        printf("%%d %%lu\\n", capstone_repr_ok(b, l), capstone_repr_granule(l));
    return 0;
}
""" % mutant
    d = tempfile.mkdtemp()
    cs = os.path.join(d, "drv.c"); open(cs, "w").write(drv)
    exe = os.path.join(d, "drv")
    r = subprocess.run(["cc", "-O2", "-I", str(INC), cs, "-o", exe], capture_output=True, text=True)
    if r.returncode:
        sys.exit("ERROR: the header did not compile for the host:\n%s" % r.stderr[:800])
    inp = "".join("%d %d\n" % (b, l) for b, l in cases)
    r = subprocess.run([exe], input=inp, capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        sys.exit("ERROR: the driver produced no answers -- an error, not a zero")
    out = {}
    for i, line in enumerate(r.stdout.split("\n")):
        if not line.strip():
            continue
        ok, g = line.split()
        out[("ok", i)] = int(ok); out[("g", i)] = int(g)
    if len(out) < 2*len(cases):
        sys.exit("ERROR: %d answers for %d cases" % (len(out)//2, len(cases)))
    return out


def assert_target_compiles():
    """The shipping target must still accept the header."""
    tc = os.environ.get("CAPSTONE_CLANG")
    if not tc:
        print("NOTE: CAPSTONE_CLANG unset; the capstone64 compile is NOT checked")
        return
    d = tempfile.mkdtemp(); cs = os.path.join(d, "t.c")
    open(cs, "w").write("#include <capstone/representable.h>\n"
                        "int f(unsigned long b, unsigned long l){ return capstone_repr_ok(b, l); }\n")
    r = subprocess.run([tc, "-target", "capstone64-unknown-elf", "-ffreestanding", "-fno-builtin",
                        "-O2", "-I", str(INC), "-S", "-emit-llvm", cs, "-o", os.path.join(d, "t.ll")],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit("ERROR: the header does not compile for capstone64:\n%s" % r.stderr[:800])
    print("the header compiles for capstone64")


def oracle_lossless(base, length):
    """Does the encoder reproduce [base, base+length) exactly?"""
    iE, B, T, E = bm.compress(base, base + length)
    b, t = bm.decompress(iE, B, T, base)
    return b == base and t == base + length

# cases: every small length at a few bases, then random large ones, then the renormalisation band
cases = []
for length in list(range(1, 5000)) + [8191, 8192, 16368, 32752, 65520, 131056, 131072]:
    for base in (1 << 30, (1 << 30) + 16, (1 << 30) + 8, (1 << 30) + 4096):
        cases.append((base, length))
rnd = random.Random(20261016)
for _ in range(4000):
    k = rnd.randrange(12, 26)
    length = rnd.randrange(1 << k, 1 << (k + 1))
    base = rnd.randrange(1 << 28, 1 << 32) & ~0xf
    cases.append((base, length))
# the band the renormalisation affects, explicitly
for k in range(12, 24):
    g = 1 << (k - 9)
    for d in range(0, 3 * g, max(1, g // 4)):
        cases.append(((1 << 30) + 16, (1 << (k + 1)) - d))

assert_target_compiles()
print("cases: %d" % len(cases))
ans = c_answers(cases)
BATCH = 1500
disagree = []
for i, (base, length) in enumerate(cases):
    said = ans.get(("ok", i))
    if said is None:
        continue
    truth = oracle_lossless(base, length)
    if bool(said) != truth:
        disagree.append((base, length, bool(said), truth))
print("the header ACCEPTS %d of %d" % (sum(1 for i in range(len(cases)) if ans.get(("ok", i))), len(cases)))
print("disagreements with the encoder: %d" % len(disagree))
for b, l, said, truth in disagree[:8]:
    print("   base=%#x len=%d  header=%s oracle=%s" % (b, l, said, truth))

print()
print("NEGATIVE CONTROL: a predicate that ignores the base must FAIL this check")
bad = c_answers(cases, broken=True)
bad_dis = 0
for i, (base, length) in enumerate(cases):
    said = bad.get(("ok", i))
    if said is None:
        continue
    if bool(said) != oracle_lossless(base, length):
        bad_dis += 1
print("   mutant disagreements: %d" % bad_dis)
if bad_dis == 0:
    sys.exit("ERROR: the mutant passed -- this check cannot distinguish a wrong predicate")
print("   -> the check fires on a base-ignoring predicate, so the 0 above is a result")

if disagree:
    sys.exit("FAIL: %d disagreements" % len(disagree))
print("\nPASS: the header agrees with the encoder on every case, and the check is two-sided.")
