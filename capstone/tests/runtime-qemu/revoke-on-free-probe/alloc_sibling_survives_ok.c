// Phase 0 control: freeing one allocation does not disturb the others, and the
// allocator keeps working after a free.
//
//   a = malloc(64); b = malloc(64)
//   free(a)                       -- revoke A only
//   read b                        -- B untouched by A's revoke
//   c = malloc(64); write/read c  -- a fresh allocation still works
//
// This is the property the whole allocator rests on: per-allocation revocation
// nodes, so one free is not an over-broad sweep. If free(a) had revoked a node
// senior to B or to the arena, reading b or allocating c would fault.
//
// Expected: no fault, retval 0x0813003c (B's sentinel; 0x081300bb if B or C
// read wrong).
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

  rof_free((void *)a); /* revoke A only */

  volatile char vb = b[0]; /* B survives A's revoke */

  volatile char *c = (volatile char *)rof_malloc(64); /* allocator still works */
  c[0] = (char)0x77;
  volatile char vc = c[0];

  if ((unsigned char)vb != ROF_PROBE_SENTINEL_B || (unsigned char)vc != 0x77) {
    *res = (ROF_RET_SIBLING_SURVIVES_OK & ~0xffu) | 0xbbu;
    return;
  }
  *res = (ROF_RET_SIBLING_SURVIVES_OK & ~0xffu) | (unsigned char)vb; /* 0x0813003c */
}
