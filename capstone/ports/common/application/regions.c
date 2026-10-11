/* Adapt the existing nested ports' grant layout to the SDK's single grant.
 * No authority is duplicated. mruby receives a 32 MiB slot pool. CPython's
 * pymalloc and PostgreSQL's memory contexts take their memory from mallocng
 * and derive their blocks (CDERIVE), so they have no grant. */
#include <capstone/capability.h>
#include <stdlib.h>
void *__real___capstone_region(unsigned);

void *__wrap___capstone_region(unsigned index) {
  return index == 1 ? __real___capstone_region(0) : 0;
}
