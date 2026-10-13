#include <capstone/capability.h>

/* This is built with the existing Capstone compiler, without ISA intrinsics.
 * Returning a scalar keeps the bootstrap independent of the application SDK.
 * Each variant other than the default ends in a fault that entry.inc and
 * run.sh place exactly; see run.sh for the expected cause of each. */
#ifdef STALE_ALIAS
/* Release through the client alias, then hand the child itself back: the
 * bootstrap dereferences it, which must fault as a dead lifetime (25). */
void *sublet_wrappers(void *backing) {
  unsigned char *pool = capstone_cap_derive(backing, 128, 512);
  unsigned char *child = capstone_cap_derive(pool, 16, 32);
  unsigned char *client = capstone_cap_without_manage(child);
  client[0] = 42;
  capstone_cap_revoke_child(pool, client);
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
#elif defined(NO_MANAGE)
  /* A reference without MANAGE cannot end a child (27). */
  unsigned char *pool = capstone_cap_derive(backing, 128, 512);
  unsigned char *child = capstone_cap_derive(pool, 16, 32);
  void *client = capstone_cap_without_manage(pool);
  capstone_cap_revoke_child(client, child);
  return 6;
#else
  unsigned char *pool = capstone_cap_derive(backing, 128, 512);
  unsigned char *child = capstone_cap_derive(pool, 16, 32);
  void *client = capstone_cap_without_manage(child);
  unsigned long rights;
  __asm__ volatile(".insn r 0x5b, 1, 4, %0, %1, x5"
                   : "=r"(rights) : "r"(client));
  if (rights != (CAPSTONE_PERM_READ | CAPSTONE_PERM_WRITE)) return 1;
  child[0] = 42;
  if (pool[16] != 42) return 2;
  capstone_cap_revoke_child(pool, client);
  child = capstone_cap_derive(pool, 16, 32);
  if (child[0] != 42) return 3;
  capstone_cap_revoke_child(pool, child);
  return 0;
#endif
}
#endif
