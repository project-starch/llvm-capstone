#include <capstone/capability.h>

/* This is built with the existing Capstone compiler, without ISA intrinsics.
 * Returning a scalar keeps the bootstrap independent of the application SDK.
 * Each variant other than the default and RENEW ends in a fault that entry.inc
 * and run.sh place exactly; see run.sh for the expected cause of each. */
#if defined(STALE_ALIAS)
/* Release through an offset copy of the child, then hand the child itself
 * back: the bootstrap dereferences it, which must fault as a dead lifetime
 * (25). The copy names the same lifetime as the child. */
void *sublet_wrappers(void *backing) {
  unsigned char *pool = capstone_cap_derive(backing, 128, 512);
  unsigned char *child = capstone_cap_derive(pool, 16, 32);
  unsigned char *client = child + 4;
  client[0] = 42;
  capstone_cap_release(client);
  return child;
}
#elif defined(RENEW_STALE)
/* Renew the child and hand the old reference back: it names the lifetime that
 * ended, so the bootstrap's dereference must fault (25). */
void *sublet_wrappers(void *backing) {
  unsigned char *pool = capstone_cap_derive(backing, 128, 512);
  unsigned char *child = capstone_cap_derive(pool, 16, 32);
  child[0] = 42;
  unsigned char *renewed = capstone_cap_renew(child);
  renewed[0] = 43;
  return child;
}
#else
unsigned long sublet_wrappers(void *backing) {
#if defined(BAD_OFFSET)
  capstone_cap_derive(backing, ~0UL, 16);
  return 4;
#elif defined(LINEAR_PARENT)
  /* backing is linear here: refused at the zero-size CDERIVE (26), before
   * any instruction that would move it out of its register. */
  capstone_cap_derive(backing, 0, 16);
  return 5;
#elif defined(LEGACY_FORM)
  /* The two-operand CREVOKE of the capstone-sublet bitstream names a register
   * in rs2. This model refuses it as illegal (2), so it cannot end the
   * lifetime of what it meant as the parent. */
  unsigned char *pool = capstone_cap_derive(backing, 128, 512);
  unsigned char *child = capstone_cap_derive(pool, 16, 32);
  capstone_cap_revoke_child(pool, child);
  return 6;
#elif defined(RENEW)
  /* A renewal keeps the bytes, the bounds and the place under the pool, and
   * the new lifetime ends like any other. */
  unsigned char *pool = capstone_cap_derive(backing, 128, 512);
  unsigned char *child = capstone_cap_derive(pool, 16, 32);
  child[0] = 42;
  unsigned char *renewed = capstone_cap_renew(child);
  if (renewed[0] != 42) return 7;
  renewed[31] = 43;
  if (pool[47] != 43) return 8;
  capstone_cap_release(renewed);
  return 0;
#else
  unsigned char *pool = capstone_cap_derive(backing, 128, 512);
  unsigned char *child = capstone_cap_derive(pool, 16, 32);
  unsigned long rights;
  __asm__ volatile(".insn r 0x5b, 1, 4, %0, %1, x5"
                   : "=r"(rights) : "r"(child));
  if (rights != (CAPSTONE_PERM_READ | CAPSTONE_PERM_WRITE)) return 1;
  child[0] = 42;
  if (pool[16] != 42) return 2;
  capstone_cap_release(child);
  child = capstone_cap_derive(pool, 16, 32);
  if (child[0] != 42) return 3;
  capstone_cap_release(child);
  return 0;
#endif
}
#endif
