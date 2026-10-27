/* Step-0 probe: does split -> mrev -> delin -> revoke make the alias fault?
 * -DPROBE_NO_REVOKE is the control. */
static unsigned char arena[4096] __attribute__((aligned(16)));

static inline void *gencap(unsigned long b, unsigned long e) {
  void *c;                       /* csdebuggencap: QEMU debug op, LINEAR [b,e) */
  __asm__ volatile(".insn r 0x5b, 0x1, 0x40, %0, %1, %2" : "=r"(c) : "r"(b), "r"(e));
  return c;
}
static inline void *split(void **lo, unsigned long mid) {
  void *hi; void *l = *lo;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x06, %0, %1, %2" : "=&r"(hi), "+r"(l) : "r"(mid));
  *lo = l; return hi;
}

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  unsigned long base = __builtin_capstone_cap_get_cursor((void *)&arena[0]);
  void *lin = gencap(base, base + sizeof(arena));

  unsigned long end = __builtin_capstone_cap_get_end(lin);
  void *hi    = split(&lin, end - 64);                    /* fresh node, LIN */
  void *rev   = __builtin_capstone_cap_mrev(hi);          /* senior handle */
  char *alias = (char *)__builtin_capstone_cap_delin(hi); /* what a caller gets */

  alias[0] = 'a';                 /* live: must work */
#ifndef PROBE_NO_REVOKE
  __builtin_capstone_cap_revoke(rev);
#endif
  alias[1] = 'b';                 /* stale: must fault when revoked */

  *res = 0xD1;                    /* reached only if the access returned */
}
