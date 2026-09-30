/* Find the recursive call site by walking the frame chain.
 *
 * zend_startup consumes ~2.6 MB of stack. The largest real frame in the image is 4,240
 * bytes (zend_register_interfaces: a -0x7f0 immediate PLUS a -2208 register adjustment --
 * clang splits frames over ~2 KB, which an immediate-only scan undercounts). So ~600-1300
 * nested calls: recursion.
 *
 * -finstrument-functions is unusable here (it crashes clang on this target), and the fault
 * itself gives one pc at translation-block granularity. So: sample from a function the
 * engine calls often, and when the stack crosses a floor, WALK THE SAVED FRAME POINTERS and
 * report the return address that occurs MOST OFTEN. In a recursion the same site repeats,
 * so the mode of the chain names it in a single boot.
 *
 * FRAME LAYOUT, verified against compiled prologues of two different sizes:
 *     cincoffsetimm sp, sp, -N
 *     stc ra, (N-0x10)(sp)       -> ra      at s0 - 0x10
 *     stc s0, (N-0x20)(sp)       -> prev s0 at s0 - 0x20
 *     movc s0, sp ; cincoffsetimm s0, s0, N   (so s0 == sp on entry)
 * The offsets are independent of N, so one walker handles every frame.
 */
#define NOINSTR __attribute__((no_instrument_function))

#define PHP_DEPTH_MAX_FRAMES 1024u
#define PHP_DEPTH_SLOTS      24u

unsigned long php_depth_floor;
unsigned long php_depth_low;
unsigned long php_depth_mallocs;
int           php_depth_armed;
void        (*php_depth_trip_fn)(void);

unsigned long php_depth_frames;        /* how many frames the walk saw */
unsigned long php_depth_mode_ra;       /* the most frequent return address */
unsigned long php_depth_mode_count;    /* how often it appeared */

NOINSTR void php_depth_walk(void)
{
    unsigned long ras[PHP_DEPTH_SLOTS];
    unsigned long cnt[PHP_DEPTH_SLOTS];
    unsigned used = 0;
    unsigned i;                 /* gnu89 hoists for-init declarations to function scope */
    unsigned long bestc = 0, bestra = 0;
    for (i = 0; i < PHP_DEPTH_SLOTS; i++) { ras[i] = 0; cnt[i] = 0; }

    void *fp = __builtin_frame_address(0);
    unsigned long base = __builtin_capstone_cap_get_base(fp);
    unsigned long end  = __builtin_capstone_cap_get_end(fp);
    unsigned long frames = 0;

    while (frames < PHP_DEPTH_MAX_FRAMES) {
        unsigned long s0 = __builtin_capstone_cap_get_cursor(fp);
        if (s0 < base + 0x20UL || s0 >= end) { break; }
        void **ra_slot   = (void **)((char *)fp - 0x10);
        void **prev_slot = (void **)((char *)fp - 0x20);
        void *ra   = *ra_slot;
        void *prev = *prev_slot;
        if (!__builtin_capstone_cap_get_tag(ra)) { break; }
        unsigned long rav = __builtin_capstone_cap_get_cursor(ra);

        for (i = 0; i < used; i++) { if (ras[i] == rav) { cnt[i]++; break; } }
        if (i == used && used < PHP_DEPTH_SLOTS) { ras[used] = rav; cnt[used] = 1; used++; }

        frames++;
        if (!__builtin_capstone_cap_get_tag(prev)) { break; }
        unsigned long pv = __builtin_capstone_cap_get_cursor(prev);
        if (pv <= s0 || pv >= end) { break; }    /* must move UP the stack */
        fp = prev;
    }

    for (i = 0; i < used; i++) { if (cnt[i] > bestc) { bestc = cnt[i]; bestra = ras[i]; } }
    php_depth_frames     = frames;
    php_depth_mode_ra    = bestra;
    php_depth_mode_count = bestc;
}

/* ---------------------------------------------------------------------------
 * REPORT A 64-BIT VALUE THROUGH A DELIBERATE FAULT.
 *
 * Every other channel out of a deep, wedged domain has failed: *res is only read by the
 * host after a clean domreturn, the output sink writes to a static buffer nobody reads,
 * env->pc is a translation-block boundary rather than the faulting instruction, and the
 * one recovery path (longjmp) is itself suspect here.
 *
 * But the monitor prints `badaddr` on every capability fault. csdebuggencap mints a
 * capability over an ARBITRARY range, so: mint one over [v, v+8), step the cursor past the
 * end, and store. The access is out of bounds, the domain halts, and the monitor prints
 * badaddr = v + 64. Subtract 64 and the value comes back out intact.
 *
 * QEMU-only (csdebuggencap is a debug op), which is fine -- this is a diagnostic.
 * --------------------------------------------------------------------------- */
NOINSTR static void *php_gencap(unsigned long b, unsigned long e)
{
    void *c;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x40, %0, %1, %2" : "=r"(c) : "r"(b), "r"(e));
    return c;
}

NOINSTR void php_fault_report(unsigned long v)
{
    volatile char *p = (volatile char *)php_gencap(v, v + 8UL);
    p[64] = 1;                 /* out of bounds on purpose: badaddr == v + 64 */
    for (;;) { }               /* unreachable; keeps the compiler from tail-merging it */
}
