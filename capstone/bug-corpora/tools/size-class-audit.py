#!/usr/bin/env python3
"""Say, BEFORE a CheriBSD boot is spent, which spatial cases that boot cannot possibly catch.

THE PROBLEM. `malloc` bounds a capability to the allocator's USABLE size, not to the requested
size, and jemalloc's usable size is a step function. A crossing shorter than the gap to the next
size class therefore lands inside the same allocation's own slack, and the CheriBSD arm reads
CLEAN -- for a reason that is a property of the allocator's granularity and has nothing to do with
the defect. A clean arm that could never have been anything else is not a negative result; it is
an unproven instrument, and this tree has a standing rule against treating one as the other.

WHAT THIS TOOL DOES. For each case it measures, rather than infers:

  * the crossed allocation's size in bytes, by interposing on malloc/calloc at run time;
  * the crossing's size in bytes, from the case's own printed `extent` and element width;
  * jemalloc's usable size for that request.

and reports ABSORBED when crossing <= usable - requested.

CALIBRATION, because a model of someone else's allocator is a claim. Four in-guest readings of
the usable size are recorded in `memcached/plain-heap-repros/00`'s case.json -- 1 -> 16, 9 -> 16,
17 -> 32, 8192 -> 8192 -- and `jemalloc_usable` reproduces all four. It did NOT at first: it
returned 8 for a 1-byte request, which those readings refute. The verdict was then checked against
the cases in this tree that have been MEASURED in-guest:
`ffmpeg/plain-heap-repros/01_bcbf3a5630_vf_scale_format_list_compaction`, whose case.json records
a CheriBSD run with revocation on as "NOT CAUGHT, which REFUTES this arm's committed prediction",
naming a 24-byte request and the size class as the mechanism. This tool independently predicts
ABSORBED for that case from the arithmetic alone. It likewise predicts ABSORBED for
`memcached/plain-heap-repros/00` (measured NOT CAUGHT) and `faults` for
`memcached/plain-heap-repros/01` (measured CAUGHT, SIGPROT with the bounds si_code). Three
agreements with real in-guest readings, in both directions, is the only reason to trust the rows
that have not been measured.

WHAT IT IS NOT. It does not say a case is bad. A case whose CheriBSD arm is absorbed is still a
real defect with a real native differential -- it simply means the Capstone arms are the
discriminating ones there, and the case.json should SAY so rather than hedge. Where the defect
allows it, the better fix is to size the reduction's allocation onto a class boundary, which is
what `wireshark/plain-heap-repros/02` does deliberately.

Usage:  python3 tools/size-class-audit.py [corpus ...]      (default: every plain-heap corpus)
Exit:   0 clean, 1 if any case could not be resolved -- "no data" is an ERROR here, never a pass.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

TOOLS = pathlib.Path(__file__).resolve().parent
CORPORA = TOOLS.parent

# stdio allocates exactly this for its buffer before the case runs. It is excluded only when
# another candidate exists, so a case that genuinely allocates 4096 is still resolvable.
STDIO_BUF = 4096
ELEM_OK = {1, 2, 4, 8, 16}

SPY = r"""
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
static void *(*real_malloc)(size_t);
static void *(*real_calloc)(size_t, size_t);
static __thread int in_hook;
static void init(void) {
  if (!real_malloc) real_malloc = dlsym(RTLD_NEXT, "malloc");
  if (!real_calloc) real_calloc = dlsym(RTLD_NEXT, "calloc");
}
void *malloc(size_t n) {
  init();
  void *p = real_malloc(n);
  if (!in_hook) { in_hook = 1; fprintf(stderr, "ALLOC %zu\n", n); in_hook = 0; }
  return p;
}
void *calloc(size_t a, size_t b) {
  init();
  if (!real_calloc) return NULL;            /* during dlsym bootstrap */
  void *p = real_calloc(a, b);
  if (!in_hook) { in_hook = 1; fprintf(stderr, "ALLOC %zu\n", a * b); in_hook = 0; }
  return p;
}
"""


def jemalloc_usable(n):
    """CheriBSD malloc's usable size: 16-byte MINIMUM, then quantum(16)-spaced to 128, then four
    classes per doubling group (spacing = 2^floor(log2 n) / 4).

    The 16-byte minimum is MEASURED, not assumed, and it is not jemalloc's generic 8: the four
    readings below were taken in-guest and are recorded in
    `memcached/plain-heap-repros/00_ddee3e2_authfile_scan_past_calloc`'s case.json, which states
    "calloc(1,1) and calloc(1,9) both return length=16, calloc(1,17) returns 32, calloc(1,8192)
    returns exactly 8192". This function's first draft returned 8 for a 1-byte request and so
    disagreed with the first of those; a capability's bounds must also be representable, which is
    the likely reason the floor is a full quantum."""
    if n <= 16:
        return 16
    if n <= 128:
        return (n + 15) // 16 * 16
    spacing = (1 << (n.bit_length() - 1)) // 4
    return (n + spacing - 1) // spacing * spacing


def build_spy(tmp):
    src, so = tmp / "allocspy.c", tmp / "allocspy.so"
    src.write_text(SPY)
    subprocess.run(["cc", "-O1", "-fPIC", "-shared", "-o", str(so), str(src), "-ldl"], check=True)
    # Positive control: the interposer must SEE an allocation it is pointed at. Build the probe
    # at -O0 -- at -O1 the compiler elides a malloc/free pair and the control fails for a reason
    # that says nothing about the interposer. That happened while writing this.
    probe = tmp / "probe.c"
    probe.write_text("#include <stdlib.h>\n#include <stdio.h>\n"
                     "int main(void){volatile unsigned char*p=malloc(40);p[0]=1;"
                     "printf(\"%d\\n\",p[0]);free((void*)p);return 0;}\n")
    subprocess.run(["cc", "-O0", "-o", str(tmp / "probe"), str(probe)], check=True)
    r = subprocess.run([str(tmp / "probe")], capture_output=True, text=True,
                       env=dict(os.environ, LD_PRELOAD=str(so)))
    if "ALLOC 40" not in r.stderr:
        sys.exit("size-class-audit: the allocation interposer failed its positive control "
                 "(a known 40-byte allocation was not seen); refusing to report")
    return so


def audit(corpus, so, tmp, rows, unresolved):
    decl = json.loads((corpus / "corpus.json").read_text())
    shared = corpus / "shared"
    for d in sorted(corpus.glob("[0-9][0-9]_*")):
        if not d.is_dir():
            continue
        n = int(d.name.split("_")[0])
        exe = tmp / f"{corpus.parent.name}-{d.name}"
        b = subprocess.run(["cc", "-O0", "-g", "-o", str(exe), str(d / "case.c"),
                            str(shared / "driver.c"), f"-I{shared}"],
                           capture_output=True, text=True)
        if b.returncode:
            unresolved.append((corpus, d.name, "build failed"))
            continue
        r = subprocess.run([str(exe), "buggy", str(n)], capture_output=True, text=True,
                           env=dict(os.environ, LD_PRELOAD=str(so)))
        allocs = [int(x) for x in re.findall(r"^ALLOC (\d+)$", r.stderr, re.M)]
        m = re.search(r"cap=(-?\d+) touched=(-?\d+)(?: extent=(-?\d+))?", r.stdout)
        if not m or not allocs:
            unresolved.append((corpus, d.name,
                               "no cap= line" if not m else "no allocation seen"))
            continue
        cap, touched = int(m.group(1)), int(m.group(2))
        extent = int(m.group(3)) if m.group(3) is not None else None
        cand = sorted({a for a in allocs if cap > 0 and a % cap == 0 and (a // cap) in ELEM_OK})
        if len(cand) > 1 and STDIO_BUF in cand:
            cand = [a for a in cand if a != STDIO_BUF]
        note = ""
        if not cand:
            unresolved.append((corpus, d.name,
                               f"cap={cap} divides none of {sorted(set(allocs))}"))
            continue
        if len(cand) > 1:
            # The crossed object is the one the index ran off, which in every case here is the
            # SHORTER array -- but that is a heuristic, so it is reported rather than hidden.
            note = f" (ambiguous: {cand}, took smallest)"
        size = cand[0]
        esz = size // cap
        if extent is None:
            # Some drivers print no `extent` (it is the UNREDUCED span, which not every corpus
            # records). The REDUCED crossing is still derivable: `touched` is the index the
            # consumer reached, so it ran (touched - cap + 1) elements past the end.
            if touched < cap:
                unresolved.append((corpus, d.name,
                                   f"no extent= and touched={touched} < cap={cap}: "
                                   f"this arm did not cross, so there is nothing to absorb"))
                continue
            extent = touched - cap + 1
            note += " (crossing derived from touched-cap+1; driver prints no extent)"
        cross = abs(extent) * esz
        usable = jemalloc_usable(size)
        slack = usable - size
        absorbed = touched >= 0 and cross <= slack   # a below-base crossing is never absorbed
        rows.append((corpus.parent.name, d.name, size, esz, cross, usable, slack, absorbed, note))


def main(argv):
    targets = [pathlib.Path(a) for a in argv[1:]] or sorted(
        p.parent for p in CORPORA.glob("*/plain-heap-repros/corpus.json"))
    rows, unresolved = [], []
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        so = build_spy(tmp)
        for c in targets:
            audit(c if c.is_absolute() else CORPORA / c, so, tmp, rows, unresolved)

    print(f"{'program':<11}{'case':<48}{'bytes':>6}{'el':>4}{'cross':>7}"
          f"{'usable':>7}{'slack':>6}  verdict")
    print("-" * 110)
    for p, name, size, esz, cross, usable, slack, absorbed, note in rows:
        print(f"{p:<11}{name:<48}{size:>6}{esz:>4}{cross:>7}{usable:>7}{slack:>6}  "
              f"{'ABSORBED -- CheriBSD arm cannot fire' if absorbed else 'faults'}{note}")
    print(f"\nresolved {len(rows)}, absorbed {sum(1 for r in rows if r[7])}")
    if unresolved:
        print(f"\nUNRESOLVED -- reported, never defaulted to a pass ({len(unresolved)}):")
        for c, name, why in unresolved:
            print(f"  {c.parent.name}/{c.name}/{name}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
