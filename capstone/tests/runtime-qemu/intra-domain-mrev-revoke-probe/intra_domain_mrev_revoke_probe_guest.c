#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "intra_domain_mrev_revoke_probe.h"

/* Controller for the single-domain held-cap probe.
 *
 * Unlike the borrow-revoke probes, the controller is NOT the revoker: it only
 * creates the region and TRANSFERS it, linear and RW, into the domain. From
 * there the domain owns the arena outright -- REV_TRANSFERRED leaves the monitor
 * with no revocation handle -- and mints its own MREV, uses an alias, and
 * revokes. The whole borrow-revoke lifecycle happens inside one domain.
 *
 * The controller keeps an ordinary Linux mmap of the region (map_region), which
 * is a page-table mapping rather than a capability descendant, so it can read
 * the bytes back and report whether the domain's live write landed. That read is
 * opt-in ("read-arena"), because it is only SAFE while some capability over the
 * region still has a live revocation node:
 *
 *   REV_TRANSFERRED leaves a stale duplicate of the region capability in the
 *   monitor's regions[] and drops the region's cpmp entry, so the next host
 *   access to those pages takes a cpmp miss and the monitor's swap_cpmp() calls
 *   cap_base(regions[id]) to find the covering region. If the domain has revoked
 *   that lineage, reloading the stale cap yields an UNTAGGED value and the lcc
 *   inside cap_base() trips `helper_cslcc: Assertion rs1_v->tag failed` -- QEMU
 *   aborts. Guard rail, not a mechanism failure; see README.md, "Gap found".
 *
 * Domain payloads are domain_main-style .dom images built with the Capstone
 * clang, loaded with create_dom(path, NULL) -- no .smode payload. A probe that
 * takes a capability fault halts the domain and QEMU exits, so for the fault
 * probes this program never reaches its own exit; the evidence is the monitor's
 * fault line in the serial log. See run-intra-domain-mrev-revoke-probe.sh.
 */

#define print_nobuf(...) \
  do {                   \
    printf(__VA_ARGS__); \
    fflush(stdout);      \
  } while (0)

#define TAG "intra-domain-mrev-revoke-probe"

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "%s: %s (observed=0x%016lx)\n", TAG, message, observed);
  fflush(stderr);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    fprintf(stderr, "usage: %s <domain.dom> [read-arena]\n", argv[0]);
    return 2;
  }
  /* Only pass read-arena for probes that leave the arena's revocation node
   * live. See the header comment. */
  int read_arena = (argc == 3 && strcmp(argv[2], "read-arena") == 0);

  if (capstone_init()) {
    fprintf(stderr, "%s: failed to initialize Capstone\n", TAG);
    return 1;
  }

  dom_id_t dom_id = create_dom(argv[1], NULL);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);
  print_nobuf("%s: created domain ID = %lu\n", TAG, dom_id);

  region_id_t region_id = create_region(PROBE_REGION_SIZE);
  unsigned char *region_bytes =
      (unsigned char *)map_region(region_id, PROBE_REGION_SIZE);
  if (!region_bytes)
    return fail_cleanup("map_region failed", 0);
  memset(region_bytes, 0, PROBE_REGION_SIZE);
  print_nobuf("%s: created region ID = %lu\n", TAG, region_id);

  /* Hand the region over: LINEAR (so the domain can MREV it) and RW (so
   * cstighten does not silently delinearise it), with no monitor-retained
   * revocation handle. This domain entry is where the domain captures the cap. */
  shared_region_annotated(dom_id, region_id, PROBE_ANNOTATION_PERM_INOUT,
                          PROBE_ANNOTATION_REV_TRANSFERRED);
  print_nobuf("%s: region transferred to domain (LIN, RW)\n", TAG);

  /* The single domain call. Fault probes never come back from this. */
  unsigned long retval = call_dom(dom_id);
  print_nobuf("%s: call retval = 0x%08lx\n", TAG, retval);
  if (read_arena)
    print_nobuf("%s: arena[%u] after call = 0x%02x\n", TAG, PROBE_OFFSET,
                region_bytes[PROBE_OFFSET]);

  capstone_cleanup();
  return 0;
}
