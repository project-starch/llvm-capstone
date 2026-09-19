/* Capstone bounds/backing protection; mode 2 adds Sublet pool leases. */
#include "../allocators/sublet/pool-leases.h"
#include "alias-scatter.h"
#include "payload-backend.h"
#include <sublet/sublet.h>

static capstone_cap_slot remaining;
static size_t init_bytes;

void ff2_alias_scatter_setup(struct ff2_alias_scatter *s) {
  uintptr_t parent_base = capstone_cap_base(&remaining);
  sublet_carve(&remaining, parent_base + 4096, &s->parent);
  sublet_handle(&s->parent, &s->parent_handle);
  s->child_address = capstone_cap_base(&s->parent);
  sublet_carve(&s->parent, s->child_address + 64, &s->child);
  s->child_alias = sublet_take(&s->child);

  uintptr_t sibling_base = capstone_cap_base(&remaining);
  sublet_carve(&remaining, sibling_base + 1024, &s->sibling);
  s->sibling_alias = sublet_take(&s->sibling);
}

void ff2_alias_scatter_transition(struct ff2_alias_scatter *s, unsigned mode,
                                  unsigned reuse) {
  if (!mode) {
    s->new_alias = s->child_alias;
    return;
  }
  /* No child handle is returned, revoked or cleared by the child. */
  capstone_cap_revoke(&s->parent_handle);
  if (!reuse)
    return;
  if (capstone_cap_type(&s->parent_handle) == CAPSTONE_CAP_UNINITIALIZED)
    capstone_cap_initialize_zero(&s->parent_handle);
  capstone_cap_move(&s->parent_handle, &s->parent);
  sublet_carve(&s->parent, s->child_address + 64, &s->new_child);
  s->new_alias = sublet_take(&s->new_child);
  if ((uintptr_t)s->new_alias != s->child_address)
    ff2_fail(324);
}

void ff2_payload_init(void *p, size_t n) {
  capstone_cap_store(&remaining, p);
  uintptr_t payload_base = capstone_cap_base(&remaining);
  if (capstone_cap_type(&remaining) != CAPSTONE_CAP_LINEAR ||
      capstone_cap_end(&remaining) - payload_base != n)
    ff2_fail(302);
  ff2_pool_init_region(payload_base, n);
}
static void *spatial_alias(struct payload_block *b) {
  void *p = capstone_cap_delinearize(&b->region);
  sublet_stats.delin++;
  return p;
}
void *ff2_payload_issue_pointer(struct payload_block *b, unsigned mode) {
  void *p = mode == 2 ? ff2_sublet_issue(b) : b->full_alias;
  return __builtin_capstone_cap_shrink(p, b->address,
                                       b->address + b->requested);
}
/* Address equality cannot authorize free. Compare the tagged current lease;
 * LCC's validity selector is unimplemented in the pinned QEMU. */
int ff2_payload_same_authority(const void *a, const void *b) {
  capstone_cap_slot aa, bb;
  capstone_cap_store(&aa, (void *)a);
  capstone_cap_store(&bb, (void *)b);
  if (capstone_cap_type(&aa) != CAPSTONE_CAP_NONLINEAR ||
      capstone_cap_type(&bb) != CAPSTONE_CAP_NONLINEAR)
    return 0;
  const volatile uint64_t *x = (const volatile uint64_t *)&aa;
  const volatile uint64_t *y = (const volatile uint64_t *)&bb;
  return x[0] == y[0] && x[1] == y[1];
}
void ff2_payload_carve(struct payload_block *b, size_t offset) {
  (void)offset;
  sublet_carve(&remaining, b->address + b->rounded, &b->region);
}
void ff2_payload_prepare_backing(struct payload_block *b, unsigned mode) {
  if (mode)
    sublet_handle(&b->region, &b->outer);
  if (mode != 2 || !b->full_alias) {
    if (mode != 2 && !b->full_alias)
      b->full_alias = spatial_alias(b);
  }
}
void ff2_payload_return_lease(struct payload_block *b, unsigned mode) {
  if (mode == 2)
    ff2_sublet_return(b);
}
void ff2_payload_free_backing(struct payload_block *b, unsigned mode) {
  if (mode) {
    unsigned long before = sublet_stats.init;
    sublet_give_to(&b->outer, &b->region);
    if (sublet_stats.init != before)
      init_bytes += b->rounded;
    b->full_alias = b->alias = NULL;
  }
}
void ff2_payload_report_stats(struct ff2_header *h) {
  h->split = sublet_stats.split;
  h->mrev = sublet_stats.mrev;
  h->delin = sublet_stats.delin;
  h->revoke = sublet_stats.revoke;
  h->init = sublet_stats.init;
  h->init_bytes = init_bytes;
  ff2_sublet_add_stats(h);
}
