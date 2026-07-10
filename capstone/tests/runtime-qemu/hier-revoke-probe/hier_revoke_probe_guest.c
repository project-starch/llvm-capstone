#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "hier_revoke_probe.h"

/* Controller for the Phase-0 hierarchical revoke probe. Creates one region and
 * TRANSFERS it (linear, RW) into a single domain, which carves per-connection
 * sub-arenas out of it and runs the parent-close-cascades-to-child lifecycle
 * intra-domain. No host read of the arena (fault probes revoke part of it). */

#define print_nobuf(...) \
  do {                   \
    printf(__VA_ARGS__); \
    fflush(stdout);      \
  } while (0)

#define TAG "hier-revoke-probe"

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "%s: %s (observed=0x%016lx)\n", TAG, message, observed);
  fflush(stderr);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <domain.dom>\n", argv[0]);
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

  region_id_t region_id = create_region(HIER_PROBE_REGION_SIZE);
  unsigned char *region_bytes =
      (unsigned char *)map_region(region_id, HIER_PROBE_REGION_SIZE);
  if (!region_bytes)
    return fail_cleanup("map_region failed", 0);
  memset(region_bytes, 0, HIER_PROBE_REGION_SIZE);
  print_nobuf("%s: created region ID = %lu\n", TAG, region_id);

  shared_region_annotated(dom_id, region_id, HIER_PROBE_ANNOTATION_PERM_INOUT,
                          HIER_PROBE_ANNOTATION_REV_TRANSFERRED);
  print_nobuf("%s: region transferred to domain (LIN, RW)\n", TAG);

  unsigned long retval = call_dom(dom_id);
  print_nobuf("%s: call retval = 0x%08lx\n", TAG, retval);

  capstone_cleanup();
  return 0;
}
