#include <stdio.h>
#include <string.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "revoke_cost_probe.h"

/* Controller for the temporal-safety overhead microbenchmark (Capstone side).
 * Grants one LINEAR arena to the domain and makes the CALL; the domain measures
 * its one allocator config and dumps the counters to the serial log via
 * csdebugcountprint, which run-revoke-cost-probe.sh greps. Identical setup to
 * borrow-cost-probe/borrow_cost_probe_guest.c; only the .dom differs per build. */

#define print_nobuf(...)  \
  do {                    \
    printf(__VA_ARGS__);  \
    fflush(stdout);       \
  } while (0)

#define TAG "revoke-cost-probe"

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "%s: %s (observed=0x%016lx)\n", TAG, message, observed);
  fflush(stderr);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <revoke_cost.dom>\n", argv[0]);
    return 2;
  }

  if (capstone_init()) {
    fprintf(stderr, "%s: failed to initialize Capstone\n", TAG);
    return 1;
  }

  dom_id_t dom_id = create_dom(argv[1], NULL);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);
  print_nobuf("%s: created domain ID = %lu\n", TAG, dom_id);

  region_id_t region_id = create_region(ROF_COST_REGION_SIZE);
  unsigned char *region_bytes =
      (unsigned char *)map_region(region_id, ROF_COST_REGION_SIZE);
  if (!region_bytes)
    return fail_cleanup("map_region failed", 0);
  memset(region_bytes, 0, ROF_COST_REGION_SIZE);
  print_nobuf("%s: created arena region ID = %lu\n", TAG, region_id);

  /* Hand the arena over LINEAR + RW, monitor keeps no revocation handle. */
  shared_region_annotated(dom_id, region_id, ROF_COST_ANNOTATION_PERM_INOUT,
                          ROF_COST_ANNOTATION_REV_TRANSFERRED);
  print_nobuf("%s: arena transferred to domain (LIN, RW)\n", TAG);

  unsigned long retval = call_dom(dom_id);
  print_nobuf("%s: call retval = 0x%08lx\n", TAG, retval);
  if ((unsigned)retval != ROF_COST_RET_OK)
    return fail_cleanup("domain did not report measurement OK", retval);

  print_nobuf("%s: measurement complete\n", TAG);
  capstone_cleanup();
  return 0;
}
