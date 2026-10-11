#!/usr/bin/env python3
"""Create the three plain-TEMPORAL corpora: the one cell of the inventory that is empty.

Every temporal corpus in this tree sits on a NESTED allocator -- FFmpeg's AVBufferPool,
Wireshark's wmem, memcached's slabs/cache -- because that is what the temporal hunts were aimed
at. So `temporal x plain allocator` reads 0 for all three programs, and that zero is a property of
which corpora exist, not of the upstream software: all three free direct allocations and use them
afterwards.

The observable is ALIASING, not merely "we read after the free". A read of freed memory is
undefined but usually quiet, so a case that only recorded the ordering would be asserting its own
construction. Instead each case frees the object, takes a FRESH allocation of the same size --
which glibc's tcache satisfies from the very chunk just freed -- and shows the stale pointer now
reads the new object's contents. That is deterministic, it is what makes a use-after-free
dangerous, and it is two-sided: the fixed arm's stale pointer either does not exist or does not
alias.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent   # capstone/bug-corpora

PROGS = {
    "ffmpeg": dict(
        macro="FFT", low="fft", tag="FFmpeg",
        allocs="av_malloc / av_mallocz / av_calloc / av_malloc_array, freed with av_free / av_freep",
        sibling="../pool-repros",
        sibling_why=("that corpus's objects come from an AVBufferPool or an AVRefStructPool, which "
                     "recycles them; the platform allocator only ever saw the pool's block. Here "
                     "the object IS the platform allocation."),
        pin="9.0.1"),
    "wireshark": dict(
        macro="WST", low="wst", tag="Wireshark",
        allocs="g_malloc / g_malloc0 / g_new / g_new0 / g_strdup, freed with g_free",
        sibling="../wmem-repros",
        sibling_why=("that corpus's objects are chunks wmem's BLOCK or BLOCK_FAST allocator carved "
                     "out of a block g_malloc handed out. Here the object IS the g_malloc, with no "
                     "wmem layer -- which is why these cases sit in wiretap and wsutil rather than "
                     "in epan."),
        pin="4.6.8"),
    "memcached": dict(
        macro="MCT", low="mct", tag="memcached",
        allocs="malloc / calloc / realloc / strdup, freed with free",
        sibling="../allocator-repros",
        sibling_why=("that corpus's objects are slab chunks or cache.c objects, both carved by "
                     "memcached's own suballocators. Here the object IS the malloc, which is where "
                     "the proxy and the parser keep their buffers."),
        pin="1.6.45"),
}

CORPUS_H = '''/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * PLAIN-TEMPORAL corpus, and deliberately a sibling of {sibling} rather than part
 * of it: {sibling_why}
 *
 * It exists because the inventory's temporal x PLAIN-allocator cell was EMPTY for all three
 * target programs, while every temporal corpus in the tree sat on a nested allocator. That zero
 * was a property of which corpora existed, not of the upstream software -- {tag} frees direct
 * allocations and uses them afterwards like any C program.
 *
 * THE OBSERVABLE IS ALIASING, NOT ORDERING. A read of freed memory is undefined but usually
 * quiet, so a case that only recorded "the access came after the free" would be asserting its own
 * construction rather than measuring anything. Each case instead:
 *
 *   1. frees the object,
 *   2. takes a FRESH allocation of the same size, which glibc's tcache satisfies from the very
 *      chunk just freed, and fills it with a marker,
 *   3. reads through the STALE pointer and finds the marker.
 *
 * The stale pointer now names a different live object. That is deterministic, it is the thing
 * that makes a use-after-free exploitable, and it is two-sided: under the upstream fix the stale
 * pointer either is not retained or is not followed, so no marker is seen.
 *
 * The contract is ../../SCHEMA.md. shared/driver.c supplies main(); a case supplies its sequence
 * inside {macro}_CASE(NN) and fills the outcome the driver prints.
 */
#ifndef {macro}_CORPUS_H
#define {macro}_CORPUS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define {macro}_CASE(n)                                                            \\
  const int {low}_case_number = (n);                                             \\
  void {low}_case_run(int fixed, struct {low}_outcome *o)

/* What a case observed.
 *   `aliased`  THE CLAIM: the stale pointer read storage that now belongs to a
 *              different live object. This is what the verdict turns on.
 *   `freed`    the lifetime ender ran. Recorded so a case that never freed is
 *              visibly INCONCLUSIVE rather than quietly passing.
 *   `marker`   what the fresh object was filled with, and
 *   `observed` what the stale read returned. Equal means aliased.
 *   `bytes`    the allocation's size, which is what makes the reuse land. */
struct {low}_outcome {{
  int aliased;
  int freed;
  int damage;
  unsigned long bytes;
  unsigned marker, observed;
  const char *defect_text, *fixed_text;
}};

extern const int {low}_case_number;
void {low}_case_run(int fixed, struct {low}_outcome *o);

_Noreturn void {low}_fail(unsigned code);
#define CHECK(c, n)                                                            \\
  do {{                                                                         \\
    if (!(c))                                                                  \\
      {low}_fail(n);                                                            \\
  }} while (0)

/* A premise that holds only WITHOUT AddressSanitizer: that the allocator handed
 * the released chunk back. ASan quarantines freed memory precisely so it cannot
 * be reused, which is what lets it fault at the labelled probe instead -- so
 * asserting reuse under ASan would fail the case for the very reason the ASan
 * arm exists. The plain build observes the aliasing; the ASan build observes the
 * fault; neither alone is the result. */
#if defined(__SANITIZE_ADDRESS__)
#define CHECK_REUSE(c, n) ((void)0)
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define CHECK_REUSE(c, n) ((void)0)
#else
#define CHECK_REUSE(c, n) CHECK(c, n)
#endif
#else
#define CHECK_REUSE(c, n) CHECK(c, n)
#endif

/* The accesses through the stale pointer, labelled so a sanitiser's report or a
 * capability fault can be required to land HERE rather than merely somewhere in
 * the program.
 *
 * DECLARED here and DEFINED once in shared/driver.c, never `static` in the
 * header: a per-translation-unit copy cannot be resolved unambiguously by
 * `supervise`, which is why some older corpora's CheriBSD rows read
 * "attribution: not established". */
unsigned {low}_read_probe(const volatile unsigned char *p);
void {low}_write_probe(volatile unsigned char *p, unsigned char v);

#define read_probe {low}_read_probe
#define write_probe {low}_write_probe

#endif
'''

DRIVER_C = '''/* main() for the {tag} plain-temporal corpus. One program per case, run twice against the same
 * binary: the control arm (fixed) first, then the buggy one. The allocator under test is the
 * platform's -- that is the whole point of the corpus.
 */
#include "corpus.h"

_Noreturn void {low}_fail(unsigned code) {{
  fprintf(stderr, "CONTROL-FAILED %u\\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}}

/* The labelled accesses, defined ONCE here so the symbol is resolvable from the
 * image and a fault can be required to land at it. */
__attribute__((noinline, used)) unsigned
{low}_read_probe(const volatile unsigned char *p) {{
  return *p;
}}
__attribute__((noinline, used)) void
{low}_write_probe(volatile unsigned char *p, unsigned char v) {{
  *p = v;
}}

int main(int argc, char **argv) {{
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != {low}_case_number) {{
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\\n",
            {low}_case_number, argv[2]);
    return 75;
  }}
  struct {low}_outcome o = {{0}};
  printf("case=%d arm=%s\\n", {low}_case_number, fixed ? "fixed" : "buggy");
  {low}_case_run(fixed, &o);
  printf("bytes=%lu freed=%d marker=0x%02x observed=0x%02x aliased=%d damage=%d\\n",
         o.bytes, o.freed, o.marker, o.observed, o.aliased, o.damage);
  /* A case that never reached its lifetime ender has measured nothing, whichever
   * arm it is -- that is INCONCLUSIVE, never a pass. */
  if (!o.freed)
    printf("VERDICT INCONCLUSIVE the lifetime ender did not run\\n");
  else if (!fixed && o.aliased)
    printf("VERDICT DEFECT-REPRODUCED %s\\n", o.defect_text);
  else if (fixed && !o.aliased)
    printf("VERDICT FIXED %s\\n", o.fixed_text);
  else
    printf("VERDICT INCONCLUSIVE\\n");
  return fixed ? !!o.aliased : !o.aliased;
}}
'''

RUN_NATIVE = '''#!/usr/bin/env bash
# One program per case, built twice: plain, and with AddressSanitizer.
#
#   plain  gives the fix differential -- the buggy arm's stale pointer reads the marker a LATER
#          allocation wrote, the fixed arm's does not. The control arm runs FIRST; if it does not
#          hold, nothing below is a verdict.
#   asan   gives native-detect, two-sided, keyed on `heap-use-after-free`. Keying on the string
#          "AddressSanitizer" would also match LeakSanitizer's summary, so a merely leaky probe
#          would read as a detection on every arm.
#
# NOTE the two builds measure DIFFERENT things on purpose. Under ASan the freed chunk goes to
# quarantine, so the fresh allocation does NOT reuse it and the aliasing cannot happen -- the
# buggy arm aborts at the labelled probe instead, which is the detection being measured. The
# plain build is where the aliasing is observed. Neither alone is the result.
#
# An infrastructure failure exits 75 and prints no verdict.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
ROOT=$HERE/..
OUT=${{1:-${{CAPSTONE_TMP_ROOT:-/tmp/capstone}}/{low}-plain-temporal-repros}}
mkdir -p "$OUT"
CC=${{CC:-cc}}

shopt -s nullglob
dirs=("$ROOT"/[0-9][0-9]_*)
if [ ${{#dirs[@]}} -eq 0 ]; then
  echo "CONTROL-FAILED no case directories under $ROOT" >&2; exit 75
fi

status=0
for dir in "${{dirs[@]}}"; do
  name=$(basename "$dir")
  n=${{name%%_*}}; n=${{n#0}}; n=${{n:-0}}

  # -O0: at higher levels the compiler may fold a store into a freed object, and the aliasing the
  # case is built to observe would stop being observable for a reason unrelated to the defect.
  "$CC" -O0 -g -o "$OUT/$name" "$dir/case.c" "$ROOT/shared/driver.c" -I"$ROOT/shared" \\
    || {{ echo "CONTROL-FAILED build $name" >&2; exit 75; }}
  "$CC" -O0 -g -fsanitize=address -o "$OUT/$name.asan" "$dir/case.c" "$ROOT/shared/driver.c" \\
    -I"$ROOT/shared" || {{ echo "CONTROL-FAILED asan build $name" >&2; exit 75; }}

  # --- control arm first, plain build
  fixed_out=$("$OUT/$name" fixed "$n" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo "CONTROL-FAILED $name fixed arm rc=$rc" >&2; exit 75; fi
  case "$fixed_out" in
    *"VERDICT FIXED"*) ;;
    *) echo "CONTROL-FAILED $name control did not hold: $fixed_out" >&2; exit 75 ;;
  esac

  # The control arm held, so a failure here is the CASE not reproducing, not the
  # instrument failing. Still exit 75 -- a case whose buggy arm does not reproduce has
  # produced no verdict -- but do not label it CONTROL-FAILED, which names the wrong arm.
  buggy_out=$("$OUT/$name" buggy "$n" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    echo "DID-NOT-REPRODUCE $name buggy arm rc=$rc (the fixed-arm control DID hold)" >&2
    exit 75
  fi

  # --- native-detect, both directions. The buggy arm is EXPECTED to abort.
  asan_fixed=$("$OUT/$name.asan" fixed "$n" 2>&1); arc=$?
  asan_buggy=$("$OUT/$name.asan" buggy "$n" 2>&1)

  fixed_clean=0
  case "$asan_fixed" in
    *"heap-use-after-free"*|*"double-free"*|*"heap-buffer-overflow"*) ;;
    *) [ $arc -eq 0 ] && fixed_clean=1 ;;
  esac
  buggy_seen=NO-REPORT
  case "$asan_buggy" in
    *"heap-use-after-free"*) buggy_seen=heap-use-after-free ;;
    *"double-free"*)         buggy_seen=double-free ;;
  esac

  printf '%s plain=%s asan-buggy=%s asan-fixed=%s\\n' "$name" \\
    "$(case "$buggy_out" in *"VERDICT DEFECT-REPRODUCED"*) echo reproduced;; *) echo NO;; esac)" \\
    "$buggy_seen" \\
    "$([ $fixed_clean -eq 1 ] && echo silent || echo REPORTED-OR-FAILED)"

  case "$buggy_out" in *"VERDICT DEFECT-REPRODUCED"*) ;; *) status=1 ;; esac
  [ "$buggy_seen" = NO-REPORT ] && status=1
  [ $fixed_clean -eq 1 ] || status=1
done
exit $status
'''

README = '''# {tag} plain-temporal repros — lifetime defects on a DIRECT allocation

**The cell this corpus exists to fill.** Every temporal corpus in this tree sits on a *nested*
allocator — FFmpeg's `AVBufferPool`, Wireshark's wmem, memcached's slabs and `cache.c` — because
that is what the temporal hunts were aimed at. Recomputing the inventory's four cells from each
case's `nested` boolean therefore gave **temporal × plain allocator = 0 for all three target
programs**. That zero was a property of which corpora existed, not of the upstream software.

**What belongs here:** the freed object came straight from {allocs}, with **no inner allocator
between it and the platform**. Its sibling is `{sibling}`, and the difference is not which bound is
crossed but **who allocated the object**: {sibling_why}

## The observable is ALIASING, not ordering

A read of freed memory is undefined but usually quiet, so a case that only recorded "the access
came after the free" would be asserting its own construction. Each case here instead frees the
object, takes a **fresh allocation of the same size** — which glibc's tcache satisfies from the
very chunk just freed — fills it with a marker, and then reads through the **stale** pointer and
finds the marker. The stale pointer now names a different live object.

That is deterministic, it is what makes a use-after-free exploitable rather than merely undefined,
and it is two-sided: under the upstream fix the stale pointer either is not retained or is not
followed, so no marker is seen.

**The plain and ASan builds measure different things, on purpose.** Under ASan the freed chunk goes
to quarantine, so the fresh allocation does not reuse it and the aliasing cannot happen — the buggy
arm aborts at the labelled probe instead, which is the `native-detect` reading. The aliasing is
observed in the plain build. Neither alone is the result, which is why `runners/run-native.sh`
reports both columns.

## Liveness

Recorded, never required. Most cases here are fix-reversals against the {pin} pin, the convention
the rest of this tree already follows.

## Cases

| shape | cases |
|---|---|

Each row's `case.json` carries the upstream fix, the exact lifetime ender, the access that follows
it, and a `nested: false` with the reason — the field the inventory's cell counts are computed from.
'''


def make(prog, d):
    c = ROOT / prog / "plain-temporal-repros"
    (c / "shared").mkdir(parents=True, exist_ok=True)
    (c / "runners").mkdir(parents=True, exist_ok=True)
    (c / "shared" / "corpus.h").write_text(CORPUS_H.format(**d))
    (c / "shared" / "driver.c").write_text(DRIVER_C.format(**d))
    r = c / "runners" / "run-native.sh"
    r.write_text(RUN_NATIVE.format(**d))
    r.chmod(0o755)
    rp = c / "README.md"
    if not rp.exists():
        rp.write_text(README.format(**d))   # never clobber a README carrying shape-table rows
    decl = {
        "program": prog,
        "boundary": "the platform allocator's own bound -- a direct allocation, no inner layer",
        "title": f"Lifetime defects on a direct {prog} allocation, with no suballocator between it "
                 f"and the platform",
        "upstream": {"version": d["pin"]},
        "cases": 0,
        "case_schema": "case-json",
        "case_macro": d["macro"],
        "required_arms": ["spatial", "sublet", "poisoncap-spatial", "poisoncap-protected",
                          "cheribsd-revocation", "native-detect", "native-fix-differential"],
        "arm_keys": {},
        "status": "planned",
        "live_in_pin_recorded": True,
        "expect_live_in_pin": {},
        "runners": [f"capstone/bug-corpora/{prog}/plain-temporal-repros/runners/run-native.sh"],
        "checker": "capstone/bug-corpora/tools/check-corpus.py",
        "evidence": [],
        "advisories": [],
        "shape_table": True,
    }
    (c / "corpus.json").write_text(json.dumps(decl, indent=2, ensure_ascii=False) + "\n")
    return c


for prog, d in PROGS.items():
    print("created", make(prog, d))
