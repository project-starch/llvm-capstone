/* How much stack does a domain actually have, and which way does it grow?
 *
 * Every rung-B fault put sp/s0 within ~100 bytes of the stack capability's BASE rather
 * than its top, which is the signature of a stack that is nearly empty at entry -- not of
 * deep recursion. Measure it instead of inferring it.
 *
 * res = (headroom_KB << 8) | flags
 *   bit0  cursor is inside [base,end)
 *   bit1  a 4 KB probe frame still leaves the cursor inside the bounds
 *   bit2  recursion to depth 64 survived
 */
static unsigned long g_base, g_end, g_cur, g_deepest;

static void recurse(int n)
{
    volatile char pad[512];
    pad[0] = (char)n; pad[511] = (char)n;
    unsigned long c = __builtin_capstone_cap_get_cursor(__builtin_frame_address(0));
    if (c < g_deepest) { g_deepest = c; }
    if (n > 0) { recurse(n - 1); }
}

void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    void *fp = __builtin_frame_address(0);
    g_base = __builtin_capstone_cap_get_base(fp);
    g_end  = __builtin_capstone_cap_get_end(fp);
    g_cur  = __builtin_capstone_cap_get_cursor(fp);
    g_deepest = g_cur;

    unsigned flags = 0;
    if (g_cur >= g_base && g_cur < g_end) { flags |= 1u; }
    if (g_cur > g_base + 4096UL) { flags |= 2u; }

    recurse(64);
    if (g_deepest > g_base) { flags |= 4u; }

    /* headroom = how far the cursor sits ABOVE the base, in KB: the room a
     * downward-growing stack actually has. */
    unsigned long headroom = (g_cur > g_base) ? (g_cur - g_base) : 0UL;
    *res = (unsigned)((headroom >> 10) << 8) | flags;
}
