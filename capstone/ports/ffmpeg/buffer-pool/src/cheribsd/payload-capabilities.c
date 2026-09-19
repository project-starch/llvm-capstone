/* Bounded leases from the shared reusable arena. The default is spatial;
 * FFPOOL_PICASSO explicitly integrates libc-owned colors for pool returns.
 */
#include "payload-backend.h"
#include <cheri/cheric.h>
#ifdef FFPOOL_PICASSO
#include <malloc_np.h>
#include <stdio.h>
#include <stdlib.h>
/* Diagnostic exported by the installed artifact: zero reads busy color IDs. */
extern int malloc2(size_t);
static size_t tokens_live, tokens_peak, tokens_issued, tokens_freed;
void ff2_picasso_checkpoint(unsigned long round) {
  printf("FF2_COLORS round=%lu busy=%d live=%zu peak=%zu issued=%zu freed=%zu\n",
         round, malloc2(0), tokens_live, tokens_peak, tokens_issued, tokens_freed);
  fflush(stdout);
}
#endif

#ifndef __CHERI_PURE_CAPABILITY__
#error "This backend requires CHERI purecap"
#endif

static unsigned char *arena;
static size_t bounds_slack, max_bounds_slack, leases;

void ff2_payload_init(void *p, size_t n) {
#ifdef FFPOOL_PICASSO
  printf("FF2_PICASSO otype_bits=%d recycle_threshold=%d token_bytes=64\n",
         CHERI_OTYPE_BITS, CHERI_OTYPE_USER_MAX - 2000);
#endif
  arena = p;
  ff2_pool_init_region((uintptr_t)p, n);
}
void ff2_payload_carve(struct payload_block *b, size_t offset) {
  b->region.c = arena + offset;
}
void ff2_payload_prepare_backing(struct payload_block *b, unsigned mode) {
#ifdef FFPOOL_PICASSO
  if (mode != 2 || !malloc_revoke_enabled())
#else
  if (mode != 0)
#endif
    ff2_fail(320); /* Refuse to label spatial execution as temporal. */
  b->full_alias = cheri_setbounds(b->region.c, b->requested);
  if (!cheri_gettag(b->full_alias) ||
      cheri_getbase(b->full_alias) != cheri_getaddress(b->region.c) ||
      cheri_getlen(b->full_alias) < b->requested ||
      cheri_getlen(b->full_alias) > b->rounded)
    ff2_fail(321);
}
void *ff2_payload_issue_pointer(struct payload_block *b, unsigned mode) {
#ifdef FFPOOL_PICASSO
  if (mode != 2)
#else
  if (mode != 0)
#endif
    ff2_fail(320);
  size_t slack = cheri_getlen(b->full_alias) - b->requested;
  bounds_slack += slack;
  if (slack > max_bounds_slack)
    max_bounds_slack = slack;
  leases++;
#ifdef FFPOOL_PICASSO
  /* A real libc allocation owns each color. Freeing this token invalidates
   * every pointer with that color, including the bounded payload lease.
   * Token storage is adapter overhead, separate from the payload arena. */
  void *color_owner = malloc(64);
  if (!color_owner || cheri_gettype(color_owner) < 0)
    ff2_fail(322);
  b->outer.c = color_owner;
  tokens_issued++;
  if (++tokens_live > tokens_peak)
    tokens_peak = tokens_live;
  void *p = __builtin_cheri_cc_set_type(b->full_alias, cheri_gettype(color_owner));
  return cheri_andperm(p, ~CHERI_PERM_SW_VMEM);
#else
  return b->full_alias;
#endif
}
int ff2_payload_same_authority(const void *a, const void *b) {
  return __builtin_cheri_equal_exact(a, b);
}
void ff2_payload_return_lease(struct payload_block *b, unsigned mode) {
#ifdef FFPOOL_PICASSO
  if (mode != 2 || !b->outer.c)
    ff2_fail(323);
  free(b->outer.c);
  b->outer.c = NULL;
  b->alias = NULL;
  tokens_live--;
  tokens_freed++;
#else
  (void)b;
  (void)mode;
#endif
}
void ff2_payload_free_backing(struct payload_block *b, unsigned mode) {
#ifdef FFPOOL_PICASSO
  ff2_payload_return_lease(b, mode);
#else
  (void)b;
  (void)mode;
#endif
}
void ff2_payload_report_stats(struct ff2_header *report) {
  /* Extension fields; existing event wire format and mode 0 stay unchanged. */
  report->reserved[0] = leases;
  report->reserved[1] = bounds_slack;
  report->reserved[2] = max_bounds_slack;
  report->reserved[3] = sizeof(void *);
#ifdef FFPOOL_PICASSO
  ff2_picasso_checkpoint(0);
#endif
}
