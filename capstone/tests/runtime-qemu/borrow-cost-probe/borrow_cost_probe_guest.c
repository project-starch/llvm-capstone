#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "borrow_cost_probe.h"

/* Controller for the borrow-path cost measurement (task-014, deliverable 2).
 *
 * Ordinary guest-Linux helper (buildroot gcc). It only sets the domain up: it
 * creates a region, transfers it LINEAR + RW into the domain (no
 * monitor-retained revocation handle, so the domain can mrev it), and makes the
 * single measurement CALL. The domain does all the measuring and dumps the
 * results through the Capstone debug counters (csdebugcountprint), which land in
 * the QEMU serial log; run-borrow-cost-probe.sh greps the counter lines.
 *
 * See intra-domain-mrev-revoke-probe/intra_domain_mrev_revoke_probe_guest.c for
 * the delivery-path rationale -- this controller is the same shape minus the
 * opt-in host read-back (this probe never reads the region from Linux, so the
 * swap_cpmp lcc guard-rail is never in play).
 */

#define print_nobuf(...) \
  do {                   \
    printf(__VA_ARGS__); \
    fflush(stdout);      \
  } while (0)

#define TAG "borrow-cost-probe"

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "%s: %s (observed=0x%016lx)\n", TAG, message, observed);
  fflush(stderr);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <borrow_cost.dom>\n", argv[0]);
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

  region_id_t region_id = create_region(BORROW_COST_REGION_SIZE);
  unsigned char *region_bytes =
      (unsigned char *)map_region(region_id, BORROW_COST_REGION_SIZE);
  if (!region_bytes)
    return fail_cleanup("map_region failed", 0);
  memset(region_bytes, 0, BORROW_COST_REGION_SIZE);
  print_nobuf("%s: created region ID = %lu\n", TAG, region_id);

  /* Hand the region over LINEAR + RW, monitor keeps no revocation handle. */
  shared_region_annotated(dom_id, region_id, BORROW_COST_ANNOTATION_PERM_INOUT,
                          BORROW_COST_ANNOTATION_REV_TRANSFERRED);
  print_nobuf("%s: region transferred to domain (LIN, RW)\n", TAG);

  unsigned long retval = call_dom(dom_id);
  print_nobuf("%s: call retval = 0x%08lx\n", TAG, retval);
  if ((unsigned)retval != BORROW_COST_RET_OK)
    return fail_cleanup("domain did not report measurement OK", retval);

  print_nobuf("%s: measurement complete\n", TAG);
  capstone_cleanup();
  return 0;
}
