/* If a freed range is re-minted, does the OLD alias stay dead?
 *
 * This is what decides whether an allocator may recycle addresses. In a classic
 * allocator, reuse is exactly what turns a dangling pointer into a type-confusion
 * bug: the stale pointer starts aliasing the NEW occupant. With capabilities the
 * old alias should remain dead, because its revocation node was revoked and the
 * re-mint derives a different one.
 *
 *   default        alias2 (the new occupant) is used; alias1 is never touched -> completes
 *   -DTOUCH_STALE  alias1 is written AFTER the range is re-minted
 *                    faults  -> reuse is SAFE, the cache can be kept
 *                    survives-> reuse RESURRECTS stale pointers, cache must go
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

  /* --- allocation 1 --- */
  void *hi1    = split(&lin, end - 64);
  void *rev1   = __builtin_capstone_cap_mrev(hi1);
  char *alias1 = (char *)__builtin_capstone_cap_delin(hi1);
  alias1[0] = 'a';

  /* --- free 1: revoke, and KEEP the reclaimed authority --- */
  void *back = __builtin_capstone_cap_revoke(rev1);

  unsigned code = 0;
  if (!(__builtin_capstone_cap_get_tag(back) & 1u)) { *res = 0xF0u; return; }

  /* --- allocation 2, same address range, fresh handle --- */
  void *rev2   = __builtin_capstone_cap_mrev(back);
  char *alias2 = (char *)__builtin_capstone_cap_delin(back);
  alias2[2] = 'c';                    /* the new occupant works */
  code |= 1u;

#ifdef TOUCH_STALE
  alias1[3] = 'x';                    /* THE QUESTION: stale pointer, reused range */
  code |= 2u;                         /* set only if that access RETURNED */
#endif

  /* --- free 2: is the recycled allocation still independently revocable? --- */
  __builtin_capstone_cap_revoke(rev2);
  code |= 4u;

  *res = 0xA0u + code;
}
