/* Adapt the existing nested ports' grant layout to the SDK's single grant.
 * No authority is duplicated. mruby receives a 32 MiB slot pool. CPython's
 * pymalloc takes its regions from mallocng and derives its blocks (CDERIVE),
 * so it has no grant. */
#include <capstone/capability.h>
#include <stdlib.h>
void *__real___capstone_region(unsigned);

#if defined(EXP_HEAP_AND_POOL)
static capstone_cap_slot pools[2];
static int ready;
void *__wrap___capstone_region(unsigned index) {
  if (index > 1) return 0;
  if (!ready) {
    capstone_cap_store(&pools[0], __real___capstone_region(0));
    if (capstone_cap_type(&pools[0]) != CAPSTONE_CAP_LINEAR ||
        capstone_cap_end(&pools[0])-capstone_cap_base(&pools[0]) <
          PORT_HEAP_REGION_BYTES + PORT_INNER_REGION_BYTES) abort();
    capstone_cap_split(&pools[0], capstone_cap_base(&pools[0]) + PORT_HEAP_REGION_BYTES, &pools[1]);
    /* The driver rounds the combined grant up to a power of two. Pool
     * adapters require their exact requested extent, not that surplus. */
    unsigned long end = capstone_cap_base(&pools[1]) + PORT_INNER_REGION_BYTES;
    if (end < capstone_cap_end(&pools[1])) {
      capstone_cap_slot surplus;
      capstone_cap_split(&pools[1], end, &surplus);
      capstone_cap_store(&surplus, 0);
    }
    ready = 1;
  }
  void *p = pools[index].c;
  pools[index].c = 0;
  return p;
}
#else
void *__wrap___capstone_region(unsigned index) {
  return index == 1 ? __real___capstone_region(0) : 0;
}
#endif
