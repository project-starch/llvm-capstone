#!/usr/bin/env python3
"""Build the pointer benchmarks of the llvm-test-suite as hosted Capstone applications.

    ptrbench.py <llvm-test-suite checkout> <application SDK dir> <out dir> [name ...]

Olden (bh, bisort, em3d, health, mst, perimeter, power, treeadd, tsp, voronoi), Ptrdist
(anagram, bc, ft, ks, yacr2) and MallocBench (cfrac, espresso): the C programs the
memory-safety literature measures pointer and allocation behaviour on. Each is compiled
unchanged, twice: with the host compiler as the reference, and with the SDK's capstone-cc as
a domain image whose malloc is the SDK's heap (sublet: one revocation node per allocation).
The reference runs here with the chosen arguments; its stdout is what the guest run is
compared with byte for byte.

Writes <out>/<name>.dom, <out>/native/<name>, <out>/native/<name>.out, <out>/inputs/<name>/
(the input files, which the guest reaches through the share), and <out>/programs.json: for
each program the guest working directory, command and stdin, for run-mix.py --programs.
The guest paths assume <out> is published as /mnt/host/bench.

A program that fails to build or to run natively is reported and left out; the exit status
is the number of programs left out.
"""
import json
import pathlib
import shutil
import subprocess
import sys

MB = "MultiSource/Benchmarks"
# name: (dir, sources or None for all .c, cppflags, ldflags, args, stdin, input files)
# Arguments are the test-suite's SMALL_PROBLEM_SIZE ones (or the default where there is no
# small one): a run that takes well under a second natively and minutes under the emulator.
# Two are smaller still: the sublet heap hands out whole atoms from a pool the launcher caps
# at 256 MiB of grant, so a program may hold at most ~520,000 live blocks; treeadd 20 (1 M
# nodes) and bisort 700000 get NULL from malloc there and fault on it.
PROGRAMS = {
    "treeadd":   (f"{MB}/Olden/treeadd",   None, ["-DTORONTO"], [],      ["18"],                         None, []),
    "bh":        (f"{MB}/Olden/bh",        None, ["-DTORONTO"], ["-lm"], ["2000", "5"],                  None, []),
    "bisort":    (f"{MB}/Olden/bisort",    None, ["-DTORONTO"], ["-lm"], ["250000"],                     None, []),
    "em3d":      (f"{MB}/Olden/em3d",      None, ["-DTORONTO"], [],      ["256", "250", "35"],           None, []),
    "health":    (f"{MB}/Olden/health",    None, ["-DTORONTO"], ["-lm"], ["8", "15", "1"],               None, []),
    "mst":       (f"{MB}/Olden/mst",       None, ["-DTORONTO"], [],      ["1000"],                       None, []),
    "perimeter": (f"{MB}/Olden/perimeter", None, ["-DTORONTO"], [],      ["9"],                          None, []),
    "power":     (f"{MB}/Olden/power",     None, ["-DTORONTO"], ["-lm"], [],                             None, []),
    "tsp":       (f"{MB}/Olden/tsp",       None, ["-DTORONTO"], ["-lm"], ["102400"],                     None, []),
    # voronoi's myalign() uses memalign, which the sublet heap has not (linking pulls musl's
    # allocator in beside it); the program's own fallback over malloc is selected instead
    "voronoi":   (f"{MB}/Olden/voronoi",   None, ["-DTORONTO", "-DMEMALIGN_IS_NOT_AVAILABLE"], ["-lm"],
                  ["10000", "20", "32", "7"], None, []),
    "anagram":   (f"{MB}/Ptrdist/anagram", None, [],            [],      ["words", "2"],                 "input.OUT", ["words", "input.OUT"]),
    "bc":        (f"{MB}/Ptrdist/bc",      None, [],            [],      [],                             "primes.b", ["primes.b"]),
    "ft":        (f"{MB}/Ptrdist/ft",      None, [],            [],      ["1500", "100000"],             None, []),
    "ks":        (f"{MB}/Ptrdist/ks",      None, [],            [],      ["KL-4.in"],                    None, ["KL-4.in"]),
    "yacr2":     (f"{MB}/Ptrdist/yacr2",   None, ["-DTODD"],    [],      ["input2.in"],                  None, ["input2.in"]),
    "cfrac":     (f"{MB}/MallocBench/cfrac", ["cfrac.c", "pops.c", "pconst.c", "pio.c", "pabs.c", "pneg.c", "pcmp.c",
                  "podd.c", "phalf.c", "padd.c", "psub.c", "pmul.c", "pdivmod.c", "psqrt.c", "ppowmod.c", "atop.c",
                  "ptoa.c", "itop.c", "utop.c", "ptou.c", "errorp.c", "pfloat.c", "pidiv.c", "pimod.c", "picmp.c",
                  "primes.c", "pcfrac.c", "pgcd.c"],
                  ["-DNOMEMOPT"], ["-lm"], ["41757646344123832613190542166099121"], None, []),
    "espresso":  (f"{MB}/MallocBench/espresso", None, ["-DNOMEMOPT"], [], ["-t", "INPUT/largest.espresso"], None,
                  ["INPUT/largest.espresso"]),
}
# Left out: MallocBench/p2c, whose native run with the test-suite's `-v < INPUT/mf.p` did not
# finish in 10 minutes here; gs, gawk, make and perl, which are their own ports' size.
# The programs are 1990s C: K&R definitions, implicit int, common symbols, no prototypes,
# so they compile as gnu89 with warnings off; cfrac and p2c are modern enough for the default.
CFLAGS = ["-O2", "-fno-common", "-Wno-everything"]   # capstone-cc takes -W..., not -w
# -fcommon crashes the capstone64 backend on a pointer-typed common symbol (`char *x;`:
# llvm_unreachable "Unknown section kind", TargetLoweringObjectFileImpl.cpp:631, toolchain
# 68c75ed3). bh relies on common symbols (one variable defined in several files), so its
# domain links with duplicate definitions allowed, and its reference is built with -fcommon.
MODERN = {"cfrac"}
# ft, bisort and espresso draw random numbers from the C library, and glibc's and musl's
# generators differ, so the reference and the domain would print different graphs. Both get
# the same generator instead: musl's rand (a 64-bit LCG), under names that match the
# declarations the programs see (int rand, long random, void srand/srandom).
RAND_DEFS = ["-Drand=ptrbench_rand32", "-Drandom=ptrbench_rand31", "-Dsrand=ptrbench_seed",
             "-Dsrandom=ptrbench_seed"]
RAND_PROGRAMS = {"ft", "espresso"}       # bisort has a random() of its own
RAND_C = """#include <stdint.h>
static uint64_t ptrbench_state;
void ptrbench_seed(unsigned s) { ptrbench_state = s - 1; }
int ptrbench_rand32(void) { ptrbench_state = 6364136223846793005ULL * ptrbench_state + 1; return (int)(ptrbench_state >> 33); }
long ptrbench_rand31(void) { return ptrbench_rand32(); }
"""
NATIVE_EXTRA = {"bh": ["-fcommon"]}
# Source adaptations, applied to a copy, the same for the reference and the domain. voronoi
# encodes an edge's rotation in the low bits of its pointer and computes the sibling by
# integer arithmetic cast back to a pointer; on Capstone such a pointer has no capability
# and its first use faults (cause 24). The arithmetic stays, the pointer is moved to the
# computed address by pointer arithmetic, which keeps the capability. The same change the
# CHERI ports of this program make. Inline functions, not macros: sym()'s argument has side
# effects at one call (sym(connect_right(...))), and the original macro evaluated it once.
ADAPT = {
    "voronoi": {
        "defines.h": [
            ("#define sym(a) ((QUAD_EDGE) (((uptrint) (a)) ^ 2*SIZE))",
             "static __inline__ QUAD_EDGE rebase_edge(QUAD_EDGE a, uptrint addr)\n"
             "{ return (QUAD_EDGE) ((char *) a + ((long) addr - (long) (uptrint) a)); }\n"
             "static __inline__ QUAD_EDGE sym_edge(QUAD_EDGE a)\n"
             "{ uptrint x = (uptrint) a; return rebase_edge(a, x ^ 2*SIZE); }\n"
             "static __inline__ QUAD_EDGE rot_edge(QUAD_EDGE a)\n"
             "{ uptrint x = (uptrint) a; return rebase_edge(a, ((x + 1*SIZE) & ANDF) | (x & ~ANDF)); }\n"
             "static __inline__ QUAD_EDGE rotinv_edge(QUAD_EDGE a)\n"
             "{ uptrint x = (uptrint) a; return rebase_edge(a, ((x + 3*SIZE) & ANDF) | (x & ~ANDF)); }\n"
             "#define sym(a) sym_edge(a)"),
            ("#define rot(a) ((QUAD_EDGE) ( (((uptrint) (a) + 1*SIZE) & ANDF) | ((uptrint) (a) & ~ANDF) ))",
             "#define rot(a) rot_edge(a)"),
            ("#define rotinv(a) ((QUAD_EDGE) ( (((uptrint) (a) + 3*SIZE) & ANDF) | ((uptrint) (a) & ~ANDF) ))",
             "#define rotinv(a) rotinv_edge(a)"),
        ],
        "newvor.c": [
            ("e = (QUAD_EDGE) ((uptrint) e ^ ((uptrint) e & ANDF));", "e = rebase_edge(e, (uptrint) e ^ ((uptrint) e & ANDF));"),
            ("temp = (QUAD_EDGE) ((uptrint) temp+SIZE);", "temp = (QUAD_EDGE) ((char *) temp + SIZE);"),
            ("onext(temp) = (QUAD_EDGE) ((uptrint) ans + 3*SIZE);", "onext(temp) = (QUAD_EDGE) ((char *) ans + 3*SIZE);"),
            ("onext(temp) = (QUAD_EDGE) ((uptrint) ans + 2*SIZE);", "onext(temp) = (QUAD_EDGE) ((char *) ans + 2*SIZE);"),
            ("onext(temp) = (QUAD_EDGE) ((uptrint) ans + 1*SIZE);", "onext(temp) = (QUAD_EDGE) ((char *) ans + 1*SIZE);"),
            ("ptr = (QUAD_EDGE) ((uptrint) ptr & ~ANDF);", "ptr = rebase_edge(ptr, (uptrint) ptr & ~ANDF);"),
        ],
    },
}


def adapted_dir(root, d, name, out):
    """A copy of the program's directory with its adaptations applied; the original otherwise."""
    if name not in ADAPT:
        return root / d
    copy = out / "src" / name
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(root / d, copy)
    for fname, edits in ADAPT[name].items():
        text = (copy / fname).read_text()
        for old, new in edits:
            if old not in text:
                sys.exit(f"{name}: adaptation target not found in {fname}: {old[:50]!r}")
            text = text.replace(old, new)
        (copy / fname).write_text(text)
    return copy
DOMAIN_EXTRA = {"bh": ["-Wl,--allow-multiple-definition"]}


def sources(root, d, listed):
    if listed:
        return [str(root / d / s) for s in listed]
    return sorted(str(p) for p in (root / d).glob("*.c"))


def main():
    root, sdk, out = (pathlib.Path(a) for a in sys.argv[1:4])
    only = set(sys.argv[4:])
    out.mkdir(parents=True, exist_ok=True)
    (out / "native").mkdir(exist_ok=True)
    # a run for some names updates those entries and keeps the rest
    mpath = out / "programs.json"
    manifest = json.load(open(mpath)) if only and mpath.exists() else {}
    failed = []
    rand_c = out / "ptrbench_rand.c"
    rand_c.write_text(RAND_C)
    for name, (d, listed, cpp, ld, args, stdin, inputs) in PROGRAMS.items():
        if only and name not in only:
            continue
        srcdir = adapted_dir(root, d, name, out)
        src = [str(srcdir / s) for s in listed] if listed else sorted(str(p) for p in srcdir.glob("*.c"))
        if not src:
            print(f"{name}: no sources under {d}"); failed.append(name); continue
        if name in RAND_PROGRAMS:
            src.append(str(rand_c))
            cpp = [*cpp, *RAND_DEFS]
        indir = out / "inputs" / name
        indir.mkdir(parents=True, exist_ok=True)
        for f in inputs:
            (indir / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / d / f, indir / f)
        # the reference: the host compiler, the same flags
        native = out / "native" / name
        std = [] if name in MODERN else ["-std=gnu89"]
        r = subprocess.run(["cc", *CFLAGS, *NATIVE_EXTRA.get(name, []), *std, *cpp, "-I", str(srcdir), *src,
                            "-o", str(native), *ld],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"{name}: native build failed:\n{r.stderr[-800:]}"); failed.append(name); continue
        try:
            with open(indir / stdin, "rb") if stdin else open("/dev/null", "rb") as fin:
                run = subprocess.run([str(native), *args], cwd=indir, stdin=fin, capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            print(f"{name}: native run did not finish in 300 s"); failed.append(name); continue
        if run.returncode:
            print(f"{name}: native run exit {run.returncode}: {run.stderr[-300:]!r}"); failed.append(name); continue
        if not run.stdout:
            print(f"{name}: native run printed nothing"); failed.append(name); continue
        (out / "native" / f"{name}.out").write_bytes(run.stdout)
        # the domain image
        dom = out / f"{name}.dom"
        r = subprocess.run([str(sdk / "capstone-cc"), *CFLAGS, *std, *cpp, "-I", str(srcdir), *src,
                            *DOMAIN_EXTRA.get(name, []), "-o", str(dom), *ld],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"{name}: domain build failed:\n{r.stderr[-1200:]}"); failed.append(name); continue
        manifest[name] = {
            "cwd": f"/mnt/host/bench/inputs/{name}",
            "cmd": " ".join([f"/mnt/host/bench/{name}.dom", *args]),
            "stdin": f"/mnt/host/bench/inputs/{name}/{stdin}" if stdin else None,
            "native_out": f"native/{name}.out",
            "sources": len(src),
        }
        print(f"{name}: ok ({len(src)} files, reference {len(run.stdout)} bytes)")
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(manifest)} programs built, {len(failed)} left out: {' '.join(failed)}")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
