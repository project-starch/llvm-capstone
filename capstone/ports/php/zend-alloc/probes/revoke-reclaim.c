/* Does REVOKE hand authority back, i.e. can a freed range be re-minted?
 *
 * This decides whether revoke-on-free and address reuse are really mutually
 * exclusive under a one-way-SPLIT arena, or whether that is just how
 * revoke_on_free_alloc.h happens to be written (it DISCARDS revoke's return
 * value at :148 and comments "the arena is NOT reclaimed").
 *
 * Reporting, not asserting: the default arm never dereferences the returned
 * capability, so it always completes and can report what came back.
 * -DRECLAIM_WRITE is the arm that actually tries to use it.
 */
static unsigned char arena[4096] __attribute__((aligned(16)));

static inline void *gencap(unsigned long b, unsigned long e) {
  void *c;
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
  unsigned long abase = __builtin_capstone_cap_get_cursor((void *)&arena[0]);
  void *lin = gencap(abase, abase + sizeof(arena));

  unsigned long end = __builtin_capstone_cap_get_end(lin);
  unsigned long mid = end - 64;

  void *hi    = split(&lin, mid);
  void *rev   = __builtin_capstone_cap_mrev(hi);
  char *alias = (char *)__builtin_capstone_cap_delin(hi);
  alias[0] = 'a';                       /* live */

  void *back = __builtin_capstone_cap_revoke(rev);   /* CAPTURE the return */

  unsigned code = 0;
  code |= (unsigned)(__builtin_capstone_cap_get_tag(back) & 1u);          /* bit0: tagged? */
  if (__builtin_capstone_cap_get_base(back) == mid)  { code |= 2u; }      /* bit1: base ok */
  if (__builtin_capstone_cap_get_end(back)  == end)  { code |= 4u; }      /* bit2: end ok  */

#ifdef RECLAIM_WRITE
  /* If revoke really returns authority, this re-mint should work and the write
   * should land. If it faults, reuse via this route is impossible. */
  if (code & 1u) {
    char *p2 = (char *)__builtin_capstone_cap_delin(back);
    p2[1] = 'b';
    code |= 8u;                         /* bit3: re-minted write SURVIVED */
  }
#endif

  *res = 0xE0u + code;
}
