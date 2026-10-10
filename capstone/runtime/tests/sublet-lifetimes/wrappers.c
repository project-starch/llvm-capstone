#include <capstone/capability.h>

/* This is built with the existing Capstone compiler, without ISA intrinsics.
 * Returning a scalar keeps the bootstrap independent of the application SDK. */
unsigned long sublet_wrappers(void *backing) {
#ifdef BAD_OFFSET
  capstone_cap_derive(backing, ~0UL, 16);
  return 4;
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
