#include <stddef.h>
#include <stdio.h>
#include <string.h>

/* libcapstone's header requires its caller to provide size_t. */
#include "capstone/domain-fault.h"
#include "libcapstone.h"

int main(int argc, char **argv) {
  if (argc != 2 || capstone_init())
    return 2;
  dom_id_t domain = create_dom(argv[1], NULL);
  region_id_t region = create_region(4096);
  if ((long)domain < 0 || (long)region < 0)
    return 1;
  volatile unsigned long *observed = map_region(region, 4096);
  if (!observed || observed == (void *)-1)
    return 1;
  memset((void *)observed, 0, 4096);
  shared_region_annotated(domain, region, 1, 0);
  for (int i = 0; i < 3; ++i) {
    unsigned long result = call_dom(domain);
    if (result != CAPSTONE_DOMAIN_FAULT_RETVAL || observed[0] != 1 ||
        observed[1] != 0)
      return 1;
    printf("FAULT_REENTRY %d RESULT %lu ENTERED %lu AFTER %lu\n", i, result,
           observed[0], observed[1]);
  }
  capstone_cleanup();
  puts("__CAPSTONE_FAULT_REENTRY_DONE__");
  return 0;
}
