/* Why mrb_vm_run's stack_clear stores out of bounds: an INSTRUMENT, not a fix.
 *
 * The fault is a store at c->ci->stack + stack_keep*sizeof(mrb_value) through a
 * capability whose bounds are far too small. Observing it from outside gives one
 * bit per boot, so this converts the fault into a RETURNED WRONG ANSWER: it clamps
 * the clear to what the capability actually permits, records the geometry, and
 * lets the domain run on and report.
 *
 * IT RECORDS UNCONDITIONALLY, and that is the whole lesson from its first
 * version. That one recorded only when it judged the clear out of bounds, and it
 * judged with `end - cursor` in unsigned arithmetic -- which cannot be trusted for
 * exactly the capability under investigation, whose cursor may sit outside its own
 * bounds and make the subtraction wrap to something enormous. It came back with
 * eight zeros and no way to tell "nothing was wrong" from "the test could not
 * fire". So: snapshot the FIRST call always, snapshot the first VIOLATING call
 * separately, and compute the room without a subtraction that can wrap.
 *
 * Deliberately free of mruby types: it takes void* and long, so its position in
 * the amalgamation does not matter and it cannot drag mruby's headers in.
 *
 * EVERY VALUE IS RELATIVE. The domain's return channel is 32 bits and these
 * addresses are 33, so absolute values would silently truncate. Lengths and
 * offsets are small and exact, and they are what the question is about anyway.
 */

#define MD_W 8
#define MD_VALUE_SIZE 32u   /* sizeof(mrb_value) here; the emitted `slli 5` agrees */

/* Leave after this many calls even if nothing looks wrong. THE ESCAPE MUST NOT
   DEPEND ON THE PROBE'S OWN JUDGEMENT: three runs were spent on a predicate that
   said the frame was fine while the very next instructions faulted on it, and each
   time the result was no numbers at all rather than numbers that disagreed with
   me. Escaping on call 1 regardless guarantees a reading; raise it to walk further
   in once the first frame is understood. */
#ifndef MD_ESCAPE_AFTER
#define MD_ESCAPE_AFTER 1
#endif

/* THE MEASUREMENT HAS TO COME BACK, and clamping was not enough to make it.
 *
 * Clamping keeps the clear inside the capability, but mruby then runs on with a
 * stack it believes it cleared and does not fault where the probe can see it --
 * the domain died later inside mrb_open_core, so the ladder never reached the
 * rungs that report, and three runs produced no numbers at all. A domain cannot
 * return from the middle of a library call, so the probe jumps out instead: the
 * ladder arms a setjmp before mrb_open_core, the first bad frame longjmps back to
 * it, and the run continues to the reporting rungs with the snapshot intact.
 *
 * That is this project's own rule about instrumentation, applied to itself: prefer
 * turning a fault into a wrong answer that RETURNS over observing the fault. */
#include "capstone_setjmp.h"

jmp_buf md_escape;
int md_escape_armed;

/* Read by mruby_domain.c and reported one half-word per ladder call. */
unsigned long md_first[MD_W];      /* the first call, whatever it looked like */
unsigned long md_viol[MD_W];       /* the first call whose clear would not fit */
unsigned long md_probe_calls;      /* how many times mrb_vm_run reached the clear */
unsigned long md_probe_violations; /* how many of those would have stored OOB */

static unsigned long md_arena_base;

static unsigned long
md_base(void *p)
{
    return __builtin_capstone_cap_get_base((char *)p);
}

static unsigned long
md_end(void *p)
{
    return __builtin_capstone_cap_get_end((char *)p);
}

static unsigned long
md_cur(void *p)
{
    return __builtin_capstone_cap_get_cursor((char *)p);
}

/* THE PROBE MUST NOT TOUCH THE HEAP. Its first version called malloc(16) here to
   find the arena base, and mrb_vm_run RE-LOADS c->ci->stack after the probe
   returns -- so the value it measured and the value the loop used were separated
   by an allocation and a free. An instrument that mutates the thing it is
   measuring cannot be believed, whichever way the numbers come out. The domain
   hands the base in from ladder call 2 instead, before any mruby code runs. */
void
md_probe_set_heap(void *p)
{
    md_arena_base = md_base(p);
}

static void
md_snap(unsigned long *w, void *sp, void *stbase, long nregs, long stack_keep)
{
    w[0] = md_end(sp) - md_base(sp);          /* length of ci->stack */
    w[1] = md_cur(sp) - md_base(sp);          /* cursor into it, signed on the way out */
    w[2] = (unsigned long)nregs;
    w[3] = (unsigned long)stack_keep;
    w[4] = stbase ? md_end(stbase) - md_base(stbase) : 0;
    w[5] = stbase ? md_base(sp) - md_base(stbase) : 0;
    /* Signed distance from the heap. Small => the VM stack is in our arena; huge
       => this capability never came from malloc at all, which is a completely
       different bug from one with the wrong length. */
    w[6] = md_base(sp) - md_arena_base;
    w[7] = (unsigned long)__builtin_capstone_cap_get_tag((char *)sp);
}

long
md_probe_stack(void *sp, void *stbase, long nregs, long stack_keep)
{
    unsigned long b, e, c;
    long avail;

    md_probe_calls++;

    if (!sp)
        return nregs;

    if (md_probe_calls == 1)
        md_snap(md_first, sp, stbase, nregs, stack_keep);

    /* MIRROR WHAT THE LOOP DOES. The first version asked whether nregs elements fit
       from the cursor, which is not the question: mruby derives the count as
       nregs - stack_keep and walks until the cursor reaches the limit, so a nregs
       BELOW stack_keep puts the limit under the start and the loop never
       terminates -- a case that check could not see at all, and did not. It passed
       the frame through and the run faulted on the very first store.

       So compute the two addresses the loop will actually touch and test those. */
    b = md_base(sp);
    e = md_end(sp);
    c = md_cur(sp);

    {
        unsigned long lo = c + (unsigned long)stack_keep * MD_VALUE_SIZE; /* first store */
        unsigned long hi = c + (unsigned long)nregs * MD_VALUE_SIZE;      /* one past last */
        int bad = (nregs < stack_keep)          /* limit below start: runs away */
               || (nregs < 0) || (stack_keep < 0)
               || (c < b) || (c > e)            /* cursor outside its own bounds */
               || (lo < b) || (hi > e);

        avail = (c < b || c >= e) ? 0 : (long)((e - c) / MD_VALUE_SIZE);

        if (bad) {
            md_probe_violations++;
            if (md_probe_violations == 1)
                md_snap(md_viol, sp, stbase, nregs, stack_keep);

            /* Clamp to something that cannot fault and cannot run away: never
               below stack_keep, never past the end. */
            nregs = (avail < stack_keep) ? stack_keep : avail;
        }

        /* LEAVE with the numbers -- on a bad frame, or simply once enough frames
           have been seen. Clamping alone did not get them back: mruby ran on with a
           stack it believed cleared and died further inside mrb_open_core, before
           the ladder reached a rung that reports. */
        if (md_escape_armed
            && (bad || md_probe_calls >= (unsigned long)MD_ESCAPE_AFTER)) {
            md_escape_armed = 0;
            longjmp(md_escape, bad ? 2 : 1);
        }
    }

    return nregs;
}
