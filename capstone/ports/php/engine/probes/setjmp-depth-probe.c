/* Does longjmp survive a DEEP unwind on this ABI?
 *
 * probes/setjmp-probe.c proved longjmp correct from 3 frames. Rung B's watchdog fires
 * hundreds of frames down inside zend_startup, and after its longjmp the domain faults in
 * domain_main with sp ~144 bytes BELOW the stack base. Two candidates: longjmp is wrong at
 * depth, or the static jmp_buf was clobbered during startup. This probe isolates the first.
 *
 * res layout:  (frames_reached << 8) | flags
 *   bit0 first setjmp returned 0
 *   bit1 longjmp delivered its value
 *   bit2 sp is back INSIDE the stack bounds after the unwind
 *   bit3 sp is back at (or above) its pre-recursion cursor
 *   bit4 a capability held across the unwind still dereferences
 *   bit5 a local in the setjmp frame is intact
 * All six set => 0x3F. Anything less names the property that broke.
 */
#include <setjmp.h>

#ifndef PROBE_DEPTH
#define PROBE_DEPTH 200
#endif

static jmp_buf jb;
static unsigned char region[64] __attribute__((aligned(16)));
static unsigned long frames_reached;
static unsigned long sp_at_setjmp;

static void deep(int n)
{
    volatile char pad[1024];          /* ~1 KB per frame, so 200 frames is ~200 KB */
    pad[0] = (char)n; pad[1023] = (char)n;
    frames_reached++;
    if (n > 0) { deep(n - 1); }
    else { longjmp(jb, 9); }
}

void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    unsigned flags = 0;
    volatile unsigned long guard = 0xA5A5A5A5UL;
    char *cap = (char *)region;
    cap[0] = 'q';

    void *fp = __builtin_frame_address(0);
    unsigned long base = __builtin_capstone_cap_get_base(fp);
    unsigned long end  = __builtin_capstone_cap_get_end(fp);
    sp_at_setjmp = __builtin_capstone_cap_get_cursor(fp);

    int r = setjmp(jb);
    if (r == 0) {
        flags |= 1u;
        frames_reached = 0;
        deep(PROBE_DEPTH);
    }
    if (r == 9) { flags |= 2u; }

    {
        void *fp2 = __builtin_capstone_cap_get_tag(__builtin_frame_address(0))
                        ? __builtin_frame_address(0) : (void *)0;
        if (fp2) {
            unsigned long c = __builtin_capstone_cap_get_cursor(fp2);
            if (c >= base && c < end)   { flags |= 4u; }
            if (c >= sp_at_setjmp)      { flags |= 8u; }
        }
    }
    if (cap[0] == 'q')            { flags |= 16u; }
    if (guard == 0xA5A5A5A5UL)    { flags |= 32u; }

    *res = (unsigned)((frames_reached & 0xFFFFFFUL) << 8) | flags;
}
