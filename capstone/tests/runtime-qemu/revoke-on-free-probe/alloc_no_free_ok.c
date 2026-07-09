// Phase 0 control for alloc_use_after_free_fault: without the free, the alias
// reads back. Proves the fault was the free, not the allocator handing out a
// dead or consumed capability (a LINEAR cap passed by value is consumed by copy,
// task-005 C3; the delin in rof_malloc is what makes the alias survivable).
//
// Expected: no fault, retval 0x0812005e.
#include "rof_probe_domain.h"

void domain_main(void *arg, unsigned func) {
  if (rof_probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  rof_init(rof_probe_grant);

  volatile char *a = (volatile char *)rof_malloc(64);
  volatile char *b = (volatile char *)rof_malloc(64);
  a[0] = (char)ROF_PROBE_SENTINEL_A;
  b[0] = (char)ROF_PROBE_SENTINEL_B;

  volatile char v = a[0]; /* no free: still live */
  (void)b;

  *res = (ROF_RET_NO_FREE_OK & ~0xffu) | (unsigned char)v; /* 0x0812005e */
}
