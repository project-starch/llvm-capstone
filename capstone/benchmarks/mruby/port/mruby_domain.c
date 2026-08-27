/* mruby inside a Capstone domain: entry point and staged bring-up.
 *
 * The ladder bisects startup. EVERY stage returns a marker, so a run always yields
 * a result rather than a wedge -- the rule this project learned the expensive way:
 * a wedged domain emits nothing, so a run that only ever fails tells you one bit
 * per boot. Stages are cumulative and ascending, and all of them live in ONE image
 * selected at run time; see the note below.
 *
 * CALLS, in order. Marked ones carry the 0x6D52 tag; the rest return a raw 32-bit
 * number, which is why the runner labels by POSITION and not by tag.
 *
 *    0  anchor                   -- &domain_main, so a trap pc can be mapped back
 *    1  entry                    -- was the domain entered, does cap-init run,
 *                                   does the return channel work?      [stage 0]
 *    2  the OUTER allocator      -- malloc/realloc/free from cap_heap  [stage 1]
 *    3  narrowing control        -- the LENGTH of a malloc(64) capability: 64 when
 *                                   narrowing is on, the whole arena when it is
 *                                   not. Raw. Without it every verdict below rests
 *                                   on a build flag nobody checked.
 *    4  mrb_open_core            -- a VM on the outer allocator alone   [stage 2]
 * 5-20  md_first[0..7]           -- the FIRST call's geometry, recorded whatever
 *                                   it looked like; low half then high half each
 * 21-36 md_viol[0..7]            -- the first call whose clear would NOT fit
 *   37  probe call count         -- how often mrb_vm_run reached the stack clear
 *   38  probe violation count    -- how many of those would have stored OOB. ZERO
 *                                   with a nonzero 37 means md_viol is zeros that
 *                                   MEAN zero; md_first still carries real data,
 *                                   which is exactly why both are reported.
 *   39  mrb_gc_add_region        -- the GC heap becomes ONE region     [stage 3]
 *   40  run embedded bytecode    -- returns WHAT RUBY COMPUTED         [stage 4]
 *
 * WHY THE REGION MATTERS AND IS NOT AN OPTIMISATION. src/gc.c:1508 reads
 * `if (dead_slot && !page->region)`, so region pages are NEVER returned to the
 * allocator. With a region-backed heap the whole engine heap is one capability,
 * no page ever reaches free(), and a use-after-free on an RVALUE yields a pointer
 * that is tagged, in bounds, and unrevocable. That is the blind spot being
 * measured. Stage 3 exists to prove the region actually took, because if mruby
 * silently falls back to malloc for pages the measurement changes underneath.
 */
#include <stddef.h>
#include <stdint.h>

/* The amalgamated header, not the split ones: rake emits a single mruby.h and
   there is no mruby/gc.h to include. */
#include "mruby.h"

/* ONE IMAGE, ALL STAGES, selected at RUN TIME by a static counter.
 *
 * MD_STAGE used to be a compile-time switch, which meant five builds of the same
 * 85000-line translation unit at eight to ten minutes each. The loader calls a
 * domain repeatedly and the entry glue does NOT re-run initialisers between calls,
 * so a static counter survives -- proved by the WAMR port's anchor rung before it
 * was relied on here. Invoke with `capstone-test.user mruby.dom 6`.
 *
 * Call 0 is the ANCHOR and not a stage. A trap reports its pc as a RUNTIME address
 * and nothing prints the load base, so a fault is otherwise unmappable; worse, the
 * pc names the TRANSLATION BLOCK'S ENTRY rather than the faulting instruction (see
 * ref/HOW-TO-RUN-ON-QEMU.md), so the base is needed before the read-forward can
 * even start. Returning the low 32 bits of &domain_main gives it.
 *
 * Ascending order is deliberate: a fault takes the domain with it, so everything
 * after the first one is lost -- and that first one IS the bisection point.
 */

/* 0x6D52 is "mR". The tag exists so a bare small integer coming from anywhere
   else in the image cannot be mistaken for a result -- the same reason WAMR's
   domain tags its answers with 0x5741. */
#define MD_MARK(stage, n) \
    do { *res = 0x6D520000u | ((unsigned)(stage) << 8) | (unsigned)(n); return; } while (0)
#define MD_OK   0x01u
#define MD_FAIL 0xEEu

#ifndef MD_REGION_BYTES
#define MD_REGION_BYTES (512u * 1024u)
#endif

/* The GC region. Static and 16-aligned: it must be one capability, and mruby
   stores capability-bearing RVALUEs in it. */
static char md_region[MD_REGION_BYTES] __attribute__((aligned(16)));

/* port/md_probe.c. Declared rather than headered: one instrument, one reader. */
#include "capstone_setjmp.h"
extern jmp_buf md_escape;
extern int md_escape_armed;
extern void md_probe_set_heap(void *);
extern unsigned long md_first[8];
extern unsigned long md_last[8];
extern unsigned long md_reread_differs;
extern int md_probe_selftest(void *);
extern unsigned long md_viol[8];
extern unsigned long md_probe_calls;
extern unsigned long md_probe_violations;

#include "md_specimen.h"   /* generated by tools/gen-specimen.sh from a .rb file */

void
domain_main(unsigned *res, unsigned func)
{
    static unsigned nth;
    static mrb_state *mrb;
    unsigned call = nth++;

    (void)func;

    switch (call) {
    case 0:
        /* THE ANCHOR. Not a stage. */
        *res = (unsigned)(uintptr_t)&domain_main;
        return;

    case 1:
        /* Touches nothing: proves entry, cap-init and the return channel. */
        MD_MARK(0, MD_OK);

    case 2: {
        /* The OUTER allocator alone, before any mruby code runs. It narrows every
           result to the request, so this also proves the narrowing path does not
           fault on its own traffic. The realloc is included deliberately: it is the
           operation that must re-widen a user pointer before handing it to umm. */
        void *a = malloc(64), *b;

        if (!a)
            MD_MARK(1, MD_FAIL);
        ((char *)a)[0] = 0x5A;
        ((char *)a)[63] = (char)0xA5;
        b = realloc(a, 4096);
        if (!b)
            MD_MARK(1, 0x02u);
        if (((char *)b)[0] != 0x5A || ((char *)b)[63] != (char)0xA5)
            MD_MARK(1, 0x03u);   /* realloc lost the contents */
#ifdef MD_PROBE_STACK
        /* Hand the probe the arena base HERE, before any mruby code runs. It used
           to find this itself with a malloc, which put an allocation between the
           capability it measured and the one mrb_vm_run then re-loaded. */
        md_probe_set_heap(b);
#endif
        free(b);
        MD_MARK(1, MD_OK);
    }

    case 3: {
        /* THE NARROWING CONTROL, and it is not optional. Every verdict below
           depends on whether the outer allocator narrows, and a build where the
           knob silently did not take looks exactly like a result. So ask the
           capability itself: 64 for the narrowed arm, the whole arena for the wide
           one. Reported raw, not as a marker. */
        void *p = malloc(64);

        if (!p) {
            *res = 0xFFFFFFFFu;
            return;
        }
        *res = (unsigned)(__builtin_capstone_cap_get_end((char *)p)
                          - __builtin_capstone_cap_get_base((char *)p));
        free(p);
        return;
    }

    case 4: {
        /* POSITIVE CONTROL FOR THE PROBE, before any mruby code runs. It hands the
           predicate a capability deliberately too small for the frame it describes
           and returns 1 if the predicate noticed. Three predicates in a row have
           now judged a frame healthy that the very next instructions faulted on, so
           "no violation" has to be distinguishable from "cannot see one". A 0 here
           invalidates every verdict the probe makes below. */
        void *p = malloc(64);
        int fires;

        if (!p) {
            *res = 0xFFFFFFFFu;
            return;
        }
        fires = md_probe_selftest(p);
        free(p);
        *res = (unsigned)fires;
        return;
    }

    case 5:
        /* A VM on the outer allocator alone. mrb_open_core builds the whole class
           hierarchy, runs mrblib through mrb_vm_run, and is thousands of
           allocations -- the first place a pointer-model defect has room to show.
           With MD_PROBE_STACK the probe ESCAPES here rather than letting the fault
           happen: it arms a setjmp, the first bad stack frame longjmps back, and
           this call returns 0x77 with the geometry recorded for the rungs below.
           Clamping alone was tried and was not enough -- mruby ran on and died
           further in, so the reporting rungs were never reached. */
#ifdef MD_PROBE_STACK
        if (setjmp(md_escape) != 0)
            MD_MARK(2, 0x77u);   /* the probe jumped out; read calls 21-36 */
        md_escape_armed = 1;
#endif
        mrb = mrb_open_core();
#ifdef MD_PROBE_STACK
        md_escape_armed = 0;
#endif
        if (!mrb)
            MD_MARK(2, MD_FAIL);
        MD_MARK(2, MD_OK);

    /* Calls 5..36: md_first then md_viol, low half then high half of each word.
       Split because these are 33-bit addresses reported through a 32-bit channel,
       and a silently truncated distance would read like "the VM stack is in the
       heap" when it is nowhere near it. BOTH arrays are reported because the first
       version of this probe recorded only on a violation it could fail to detect,
       and came back with eight zeros that were indistinguishable from a finding. */
    case  6: case  7: case  8: case  9: case 10: case 11: case 12: case 13:
    case 14: case 15: case 16: case 17: case 18: case 19: case 20: case 21:
    case 22: case 23: case 24: case 25: case 26: case 27: case 28: case 29:
    case 30: case 31: case 32: case 33: case 34: case 35: case 36: case 37:
    case 38: case 39: case 40: case 41: case 42: case 43: case 44: case 45:
    case 46: case 47: case 48: case 49: case 50: case 51: case 52: case 53: {
        unsigned i = call - 6;
        unsigned long *a = (i < 16) ? md_first : (i < 32) ? md_last : md_viol;

        *res = (i & 1) ? (unsigned)(a[(i % 16) / 2] >> 32)
                       : (unsigned)a[(i % 16) / 2];
        return;
    }

    case 54:
        *res = (unsigned)md_probe_calls;
        return;

    case 55:
        /* The instrument's own honesty check. Zero violations with a nonzero call
           count means the clear never went out of bounds and every word above is a
           zero that means zero -- NOT a probe that failed to fire. Zero calls means
           mrb_vm_run was never reached, which is a different result entirely. */
        *res = (unsigned)md_probe_violations;
        return;

    case 56:
        /* THE MEASUREMENT THIS BUILD EXISTS FOR: how many frames read
           c->ci->stack twice inside the probe and got two different
           capabilities. Nonzero turns an inference into an observation. */
        *res = (unsigned)md_reread_differs;
        return;

    case 57: {
        /* The GC heap becomes ONE region we own. Reports the PAGE COUNT rather than
           OK: a region that yielded fewer pages than expected still "works" and
           silently sends later allocations to malloc, which would change what is
           being measured without saying so. Zero pages means it did not take. */
        int pages;

        if (!mrb)
            MD_MARK(3, MD_FAIL);
        pages = mrb_gc_add_region(mrb, md_region, sizeof(md_region));
        *res = 0x6D520000u | (3u << 8) | ((unsigned)(pages < 0 ? 0 : pages) & 0xFFu);
        return;
    }

    case 58: {
        /* Run the embedded bytecode and return WHAT RUBY COMPUTED, so a zero from a
           failed run cannot be mistaken for a legitimate result. */
        mrb_value v;
        unsigned out;

        if (!mrb)
            MD_MARK(4, MD_FAIL);
        v = mrb_load_irep(mrb, md_specimen);
        if (mrb->exc)
            MD_MARK(4, 0x03u);   /* Ruby raised */
        out = mrb_integer_p(v) ? (unsigned)mrb_integer(v) : 0xFEu;
        *res = 0x6D520000u | (4u << 8) | (out & 0xFFu);
        return;
    }

    default:
        if (mrb)
            mrb_close(mrb);
        mrb = NULL;
        MD_MARK(9, MD_OK);
    }
}
