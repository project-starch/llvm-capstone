#!/usr/bin/env python3
"""test-retry-streak.py <capstone-sbi worktree>: a host test of the fetch-retry cap's LOGIC (579ef1c), on the monitor's
own source text: fetch_retry_covered is cut from sbi_capstone.c (from `#define FETCH_RETRY_MIN_INSTRET` to the
function's closing brace), compiled with the host cc against stubs for the CSR reads, the CPMP slots, the report and
the asm, and driven through refusal sequences.

It proves the C, not the generated code: capstone-c's output is checked only by the firmware build (the dom_stack
gate, the FRTC tag present). The silicon positive control is the batch bitstream's two-pc arm.

Scenarios (each prints its result list; R = retried, I = injected, C = FRTC reported):
  alt       two pcs alternating, 10 instructions apart: with the cap, never more than FETCH_RETRY_MAX_STREAK retries
            in a row, then FRTC and an injection;
  same      one pc twice, 10 apart: the per-pc guard injects the second, with no FRTC;
  progress  two pcs alternating, 5000 apart: every refusal is retried (progress resets the streak);
  nocover   a pc no executable slot covers: injected, with no report.
The negative control is the same harness built WITHOUT -DCAPSTONE_FETCH_RETRY_CAP: `alt` must then retry every
refusal, or this test cannot tell a working cap from none. Exit 0 only when every expectation holds on both builds.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS = r'''
#include <stdio.h>
#include <string.h>
#define unsigned unsigned long
#define CPMP_COUNT 4
#define CAUSE_FETCH_ACCESS 1
#define __linear
typedef struct { unsigned b, e, p; } slot_t;
static slot_t slots[CPMP_COUNT];
static long cpmp_region[CPMP_COUNT];
static unsigned sim_mstatus, sim_mcause, sim_mepc, sim_minstret;
#define C_READ_CSR(csr, v) ((v) = sim_##csr)
static void *read_cpmp(unsigned i) { return &slots[i]; }
static void write_cpmp(unsigned i, void *c) { (void)i; (void)c; }
#define cap_base(c) (((slot_t *)(c))->b)
#define cap_end(c) (((slot_t *)(c))->e)
#define __capfield(c, n) (((slot_t *)(c))->p)
static unsigned cpmp_unvouched(void) { return 0; }
static char log_[4096]; static int nlog;
#define CAPSTONE_TAG_FRTY 1
#define CAPSTONE_TAG_FRTC 2
static void capstone_report(unsigned tag, unsigned v) { (void)v; log_[nlog++] = tag == CAPSTONE_TAG_FRTC ? 'C' : 'Y'; }
#define ASM_STUB(s) ((void)0)
unsigned fetch_retry_pc;
unsigned fetch_retry_ret;
#ifdef CAPSTONE_FETCH_RETRY_CAP
unsigned fetch_retry_streak;
#endif
@@BODY@@
#undef unsigned
int main(int argc, char **argv) {
    unsigned long step = strtoul(argv[2], 0, 0); int n = atoi(argv[3]);
    cpmp_region[0] = 0; slots[0].b = 0x1000; slots[0].e = 0x9000; slots[0].p = 1;
    cpmp_region[1] = cpmp_region[2] = cpmp_region[3] = -1;
    sim_mcause = CAUSE_FETCH_ACCESS; sim_mstatus = 0; sim_minstret = 100000;
    for (int k = 0; k < n; k++) {
        unsigned long pc = !strcmp(argv[1], "same") ? 0x2000 : !strcmp(argv[1], "nocover") ? 0xa000 : (k & 1 ? 0x3000 : 0x2000);
        sim_mepc = pc; sim_minstret += step; nlog = 0;
        unsigned long r = fetch_retry_covered(pc);
        printf("%c%s", r ? 'R' : 'I', memchr(log_, 'C', nlog) ? "C" : "");
        printf(k + 1 < n ? " " : "\n");
    }
    return 0;
}
'''


def body(src):
    s = src.read_text()
    a = s.index("#define FETCH_RETRY_MIN_INSTRET")
    f = s.index("static unsigned fetch_retry_covered", a)
    m = re.compile(r"^}\n", re.M).search(s, f)
    if not m:
        sys.exit("NO DATA: fetch_retry_covered has no closing brace at column 0")
    text = s[a:m.end()]
    text, n = re.subn(r'__asm__ volatile\("[^"]*"\);', "ASM_STUB(0);", text)
    if n < 2:
        sys.exit(f"NO DATA: expected the sfence/fence.i/strobe asm in fetch_retry_covered, found {n}")
    return text


def build(work, text, cap):
    c = work / f"h{int(cap)}.c"
    c.write_text("#include <stdlib.h>\n" + HARNESS.replace("@@BODY@@", text))
    exe = work / f"h{int(cap)}"
    cmd = ["cc", "-O0", "-w", "-o", str(exe), str(c)] + (["-DCAPSTONE_FETCH_RETRY_CAP"] if cap else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"BUILD FAILED (cap={cap}):\n{r.stderr[-2000:]}")
    return exe


def run(exe, scen, step, n):
    return subprocess.run([str(exe), scen, str(step), str(n)], capture_output=True, text=True, check=True).stdout.split()


def longest_run(seq, sym):
    best = cur = 0
    for x in seq:
        cur = cur + 1 if x == sym else 0
        best = max(best, cur)
    return best


def main():
    src = Path(sys.argv[1]) / "sbi_capstone.c"
    text = body(src)
    if "CAPSTONE_FETCH_RETRY_CAP" not in text:
        sys.exit("NO DATA: this monitor has no CAPSTONE_FETCH_RETRY_CAP block")
    ok = True
    with tempfile.TemporaryDirectory() as t:
        work = Path(t)
        on, off = build(work, text, True), build(work, text, False)
        res = {}
        for name, exe in (("cap", on), ("nocap", off)):
            res[name] = {"alt": run(exe, "alt", 10, 12), "same": run(exe, "same", 10, 4),
                         "progress": run(exe, "progress", 5000, 12), "nocover": run(exe, "nocover", 10, 3)}
            for k, v in res[name].items():
                print(f"{name:6} {k:9} {' '.join(v)}")
        c = res["cap"]

        def expect(cond, what):
            nonlocal ok
            print(("PASS " if cond else "FAIL ") + what)
            ok &= bool(cond)
        expect(longest_run([x[0] for x in c["alt"]], "R") <= 2, "alt with the cap: at most 2 retries in a row")
        expect(c["alt"][:3] == ["R", "R", "IC"], "alt with the cap: R R then FRTC and an injection")
        expect(sum(x == "IC" for x in c["alt"]) == 4, "alt with the cap: one FRTC per three refusals (4 in 12)")
        expect(c["same"][:2] == ["R", "I"], "same pc: the per-pc guard injects the second, with no FRTC")
        expect(all(x == "R" for x in c["progress"]), "progress between refusals: every refusal retried")
        expect(all(x == "I" for x in c["nocover"]), "no covering slot: injected, no report")
        expect(all(x == "R" for x in res["nocap"]["alt"]), "NEGATIVE CONTROL: without the cap, alt retries every refusal")
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
