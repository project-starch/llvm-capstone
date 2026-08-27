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
#ifndef MD_PROBE_SKIP_CLEAR
#define MD_PROBE_SKIP_CLEAR 0
#endif

/* MD_PROBE_FORCE_STACK: store stbase -- a capability the probe has just measured
   as 4096 bytes with its cursor at its base -- into c->ci->stack before returning,
   and clamp nregs to what fits in it. The clear that follows is then provably safe
   IF it reads the field the probe wrote.
   This is the last question left. Everything else has been measured: the frame is
   healthy when the probe reads it (twice, with the same two loads the clear uses),
   the predicate fires on a frame it must reject, the probe preserves s0-s3 and
   touches no other callee-saved register, it does not touch the heap, and the
   domain stack sits 183076 bytes BELOW the arena and grows away from it. If a
   forced-good field still faults with 80-byte bounds, the load is not reading what
   was written, and that is no longer an mruby question. */
#ifndef MD_PROBE_FORCE_STACK
#define MD_PROBE_FORCE_STACK 0
#endif

#ifndef MD_ESCAPE_AFTER
#define MD_ESCAPE_AFTER 1000000
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
unsigned long md_last[MD_W];       /* rolling: the most recent call seen */
unsigned long md_viol[MD_W];       /* the first call whose clear would not fit */
unsigned long md_probe_calls;      /* how many times mrb_vm_run reached the clear */
unsigned long md_probe_violations; /* how many of those would have stored OOB */
unsigned long md_reread_differs;   /* frames where reading c->ci->stack TWICE differed */

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
    /* WHERE THE STACK IS, relative to the heap. mrb_vm_run saves capabilities --
       proc among them, and sizeof(struct RProc) is exactly the 80 bytes the fault
       reports -- into its own frame on entry. If the domain stack has descended
       into the carved arena, that save lands on the callinfo array and overwrites
       ci->stack with a saved RProc. A small or negative number here says so; a
       number larger than the arena says the stack is safely elsewhere. */
    {
        char probe_local;

        w[7] = md_cur(&probe_local) - md_arena_base;
    }
}

long md_probe_stack(void *ctx, long nregs, long stack_keep);

/* POSITIVE CONTROL for the predicate below. Hands it a capability deliberately too
   small for the frame it describes and returns 1 if the predicate noticed. Without
   this, "no violation" and "cannot detect a violation" are the same observation --
   and the second has been true three times already. Runs before any mruby code, so
   it cannot disturb what is being measured; it restores the counters afterwards. */
int
md_probe_selftest(void *heap_ptr)
{
    unsigned long cur = md_cur(heap_ptr);
    void *tiny = __builtin_capstone_cap_shrink(heap_ptr, cur, cur + 16);
    unsigned long saved_calls = md_probe_calls, saved_viol = md_probe_violations;
    int armed = md_escape_armed;
    int fired;

    /* A fake context: slot 1 = stbase, slot 3 = ci; and a fake callinfo whose
       slot 3 is the deliberately tiny capability. Built on the stack so the
       control needs nothing from mruby and can run before it. */
    void *fake_ci[4] = { 0, 0, 0, tiny };
    void *fake_ctx[4] = { 0, heap_ptr, 0, fake_ci };

    md_escape_armed = 0;              /* must not jump out of the control itself */
    md_probe_stack(fake_ctx, 100, 0); /* 100 * 32 bytes into 16: must be bad */
    fired = (md_probe_violations > saved_viol);

    md_probe_calls = saved_calls;
    md_probe_violations = saved_viol;
    md_escape_armed = armed;
    md_viol[0] = 0;                   /* the control must not leave its own frame behind */
    return fired;
}

/* Takes the CONTEXT, not ci->stack, and loads the field ITSELF -- twice.
 *
 * The caller used to hand in c->ci->stack. The probe measured a healthy 4096-byte
 * capability, and the clear four instructions later faulted on an 80-byte one,
 * through the same register and the same two loads. Everything else was ruled out:
 * the predicate fires on a frame it must reject (md_probe_selftest), this function
 * saves and restores only s0-s3 and never touches the caller's s7, and skipping the
 * clear entirely turns the fault into a hang, so the probe is provably in that
 * path. What was left was an inference -- that the memory changes across the call --
 * and an inference is not a measurement. So: read c->ci->stack here, do the work,
 * then read it AGAIN as late as possible and report whether the two differ.
 *
 * Offsets are in capability-sized slots: mrb_context.stbase is 0x10 (slot 1) and
 * .ci is 0x30 (slot 3); mrb_callinfo.stack is 0x30 (slot 3). All four were taken
 * from -Xclang -fdump-record-layouts, not from reading the struct.
 */
long
md_probe_stack(void *ctx, long nregs, long stack_keep)
{
    void **cx = (void **)ctx;
    void *ci, *sp, *sp2, *stbase;
    unsigned long b, e, c;
    long avail;
    int bad, moved;

    md_probe_calls++;

    if (!ctx)
        return nregs;
    ci = cx[3];
    stbase = cx[1];
    if (!ci)
        return nregs;
    sp = ((void **)ci)[3];
    if (!sp)
        return nregs;

    if (md_probe_calls == 1)
        md_snap(md_first, sp, stbase, nregs, stack_keep);
    md_snap(md_last, sp, stbase, nregs, stack_keep);

    b = md_base(sp);
    e = md_end(sp);
    c = md_cur(sp);
    {
        unsigned long lo = c + (unsigned long)stack_keep * MD_VALUE_SIZE;
        unsigned long hi = c + (unsigned long)nregs * MD_VALUE_SIZE;

        bad = (nregs < stack_keep) || (nregs < 0) || (stack_keep < 0)
           || (c < b) || (c > e) || (lo < b) || (hi > e)
           || (stbase && (md_base(sp) < md_base(stbase)
                          || md_end(sp) > md_end(stbase)));
        avail = (c < b || c >= e) ? 0 : (long)((e - c) / MD_VALUE_SIZE);
    }

    /* THE SECOND READ. Same two loads the clear is about to do, as late as this
       function can do them. */
    sp2 = ((void **)cx[3])[3];
    moved = (md_base(sp2) != b) || (md_end(sp2) != e) || (md_cur(sp2) != c);
    if (moved) {
        md_reread_differs++;
        if (md_reread_differs == 1)
            md_snap(md_viol, sp2, stbase, nregs, stack_keep);
    }
    else if (bad && md_probe_violations == 0) {
        md_snap(md_viol, sp, stbase, nregs, stack_keep);
    }
    if (bad)
        md_probe_violations++;

    if (bad || moved)
        nregs = (avail < stack_keep) ? stack_keep : avail;

#if MD_PROBE_FORCE_STACK
    if (stbase) {
        long room = (long)((md_end(stbase) - md_cur(stbase)) / MD_VALUE_SIZE);

        ((void **)cx[3])[3] = stbase;          /* c->ci->stack = c->stbase */
        stack_keep = 0;
        nregs = (room < 1) ? 0 : (room > 4 ? 4 : room);
        md_snap(md_viol, stbase, stbase, nregs, stack_keep);
    }
#endif

    if (md_escape_armed
        && (bad || moved || md_probe_calls >= (unsigned long)MD_ESCAPE_AFTER)) {
        md_escape_armed = 0;
        longjmp(md_escape, moved ? 3 : bad ? 2 : 1);
    }

    return nregs;
}
