/* Minimal allocator-independent consumer of the shared Linux fault policy. */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

/* libcapstone's header requires its caller to provide size_t. */
#include "capstone/linux-domain-fault.h"
#include "libcapstone.h"

static void cleanup(void *unused) {
  (void)unused;
  capstone_cleanup();
}

int main(int argc, char **argv) {
  if (argc != 2 || capstone_init())
    return 2;
  dom_id_t domain = create_dom(argv[1], NULL);
  region_id_t region = create_region(4096);
  if ((long)domain < 0 || (long)region < 0)
    return 1;
  void *shared = map_region(region, 4096);
  if (!shared || shared == (void *)-1)
    return 1;
  memset(shared, 0, 4096);
  shared_region_annotated(domain, region, 1, 0);
  unsigned long result = call_dom(domain);
  /* Handle the result before consuming anything the domain may have written. */
  capstone_domain_exit_on_fault(result, cleanup, NULL);
  cleanup(NULL);
  if (result)
    return 1;
  puts("RUNTIME_CLIENT HEALTHY");
  return 0;
}
