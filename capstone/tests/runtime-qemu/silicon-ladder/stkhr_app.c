/* Measurement probe: how much stack does a domain actually get?
   lcc(rd, sp, 2) = cursor, 3 = base; the difference is the headroom a recursion
   has to live in. Encoded by hand with the same .insn form the entry glue uses.
   The global exists so the build's gp[i] gate has something to see. */
static volatile unsigned stkhr_touch;

void domain_main(unsigned *res, unsigned func) {
    unsigned long cur, base;
    (void)func;
    stkhr_touch = 1;
    __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x2" : "=r"(cur));
    __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x3" : "=r"(base));
    *res = (unsigned)(cur - base) + stkhr_touch - 1;
}
