/* Adapt the existing nested ports' grant layout to the SDK's single grant.
 * No authority is duplicated: the CPython grant is split into its required
 * 64 MiB payload and the remaining metadata backing (at least 16 MiB).
 * The driver rounds the 80 MiB request to 128 MiB; the metadata backend uses
 * 16 MiB of the 64 MiB remainder. mruby receives a 32 MiB slot pool. */
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
#elif defined(EXP_PYMALLOC)
static capstone_cap_slot parts[2];
static int ready;
void *__wrap___capstone_region(unsigned index) {
  if (index > 1) return 0;
  if (!ready) {
    capstone_cap_store(&parts[0], __real___capstone_region(0));
    if (capstone_cap_type(&parts[0]) != CAPSTONE_CAP_LINEAR ||
        capstone_cap_end(&parts[0])-capstone_cap_base(&parts[0]) < (80UL << 20)) abort();
    capstone_cap_split(&parts[0], capstone_cap_base(&parts[0])+(64UL << 20), &parts[1]);
    ready = 1;
  }
  if (index == 1) return capstone_cap_delinearize(&parts[1]);
  void *p = parts[0].c;
  parts[0].c = 0;
  return p;
}
#else
void *__wrap___capstone_region(unsigned index) {
  return index == 1 ? __real___capstone_region(0) : 0;
}
#endif
