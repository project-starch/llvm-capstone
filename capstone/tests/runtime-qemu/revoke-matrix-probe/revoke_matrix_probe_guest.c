#include <stdio.h>
#include <stdlib.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "revoke_matrix_probe.h"

/* Lender/controller (shared by all matrix cases): lend a region as a revocable
 * borrow, revoke it between two domain calls. The borrower's round-2 use of the
 * delegated cap is the use-after-revoke; the case selects how it held the cap. */

#define print_nobuf(...)         \
  do {                           \
    printf(__VA_ARGS__);         \
    fflush(stdout);              \
  } while (0)

#define PROBE_ANNOTATION_PERM_OUT 0x2
#define PROBE_ANNOTATION_REV_BORROWED 0x1

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "revoke-matrix-probe: %s (observed=0x%016lx)\n", message,
          observed);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <smode-domain-path>\n", argv[0]);
    return 2;
  }
  if (capstone_init()) {
    fprintf(stderr, "revoke-matrix-probe: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);

  region_id_t region_id = create_region(REVOKE_MATRIX_REGION_SIZE);
  unsigned long *region_words =
      (unsigned long *)map_region(region_id, REVOKE_MATRIX_REGION_SIZE);
  if (!region_words)
    return fail_cleanup("map_region failed", 0);
  region_words[0] = 0;

  print_nobuf("revoke-matrix-probe: case %d\n", REVOKE_MATRIX_CASE);
  shared_region_annotated(dom_id, region_id, PROBE_ANNOTATION_PERM_OUT,
                          PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf("revoke-matrix-probe: region borrowed to domain\n");

  unsigned long ret1 = call_dom(dom_id);
  print_nobuf("revoke-matrix-probe: round 1 retval = 0x%lx\n", ret1);
  if (region_words[0] != REVOKE_MATRIX_SENTINEL_STAGE1)
    return fail_cleanup("stage 1 sentinel mismatch (borrow not live?)",
                        region_words[0]);

  print_nobuf("revoke-matrix-probe: revoking borrowed region\n");
  revoke_region(region_id);
  print_nobuf("revoke-matrix-probe: region revoked\n");

  print_nobuf("revoke-matrix-probe: entering round 2 (use-after-revoke)\n");
  unsigned long ret2 = call_dom(dom_id);
  print_nobuf("revoke-matrix-probe: round 2 returned 0x%lx\n", ret2);

  if (region_words[0] == REVOKE_MATRIX_SENTINEL_STAGE2) {
    print_nobuf("revoke-matrix-probe: NO-TRAP-GAP use-after-revoke store landed\n");
  } else {
    print_nobuf("revoke-matrix-probe: use-after-revoke did not update lender view "
                "(word=0x%016lx)\n",
                region_words[0]);
  }

  capstone_cleanup();
  return 0;
}
