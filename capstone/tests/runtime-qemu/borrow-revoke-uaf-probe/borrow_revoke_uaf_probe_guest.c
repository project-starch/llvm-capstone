#include <stdio.h>
#include <stdlib.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "borrow_revoke_uaf_probe.h"

/* Lender / controller. Owns the region, lends it to the domain as a revocable
 * borrow, then revokes it between the two domain calls. The borrower's round-2
 * dereference of its cached pointer is the use-after-revoke under test. */

#define print_nobuf(...)         \
  do {                           \
    printf(__VA_ARGS__);         \
    fflush(stdout);              \
  } while (0)

/* Annotation constants (see hostcall-stdout-probe header). REV_BORROWED is the
 * *revocable* borrow: the lender retains a revocation capability so that
 * revoke_region() later has a REV cap to act on. REV_SHARED (0x2) is a
 * non-revocable share and makes revoke_region() assert in helper_csrevoke. */
#define PROBE_ANNOTATION_PERM_OUT 0x2
#define PROBE_ANNOTATION_REV_BORROWED 0x1

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "borrow-revoke-uaf-probe: %s (observed=0x%016lx)\n", message,
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
    fprintf(stderr, "borrow-revoke-uaf-probe: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);

  region_id_t region_id = create_region(BORROW_REVOKE_UAF_REGION_SIZE);
  unsigned long *region_words =
      (unsigned long *)map_region(region_id, BORROW_REVOKE_UAF_REGION_SIZE);
  if (!region_words)
    return fail_cleanup("map_region failed", 0);

  region_words[0] = 0;

  print_nobuf("borrow-revoke-uaf-probe: created domain ID = %lu\n", dom_id);
  print_nobuf("borrow-revoke-uaf-probe: created region ID = %lu\n", region_id);

  /* Lend the region to the borrower as a revocable borrow. */
  shared_region_annotated(dom_id, region_id, PROBE_ANNOTATION_PERM_OUT,
                          PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf("borrow-revoke-uaf-probe: region borrowed to domain\n");

  /* Round 1: borrow live. Borrower caches the pointer and writes stage 1. */
  unsigned long ret1 = call_dom(dom_id);
  print_nobuf("borrow-revoke-uaf-probe: round 1 retval = 0x%lx\n", ret1);
  print_nobuf("borrow-revoke-uaf-probe: word after round 1 = 0x%016lx\n",
              region_words[0]);
  if (region_words[0] != BORROW_REVOKE_UAF_SENTINEL_STAGE1)
    return fail_cleanup("stage 1 sentinel mismatch (borrow not live?)",
                        region_words[0]);

  /* End the borrow. The lender keeps its own mapping; the capability the
   * borrower cached should be invalidated. */
  print_nobuf("borrow-revoke-uaf-probe: revoking borrowed region\n");
  revoke_region(region_id);
  print_nobuf("borrow-revoke-uaf-probe: region revoked\n");

  /* Round 2: borrower dereferences its cached pointer (use-after-revoke).
   * EXPECT a deterministic capability fault. The serial log carries the QEMU
   * "[CAPSTONE] Cap mem access ..." diagnostic; the run wrapper classifies on
   * it. If we return here AND the lender observes the stage-2 sentinel, the
   * use-after-revoke was NOT trapped (a no-trap gap). */
  print_nobuf("borrow-revoke-uaf-probe: entering round 2 (use-after-revoke)\n");
  unsigned long ret2 = call_dom(dom_id);
  print_nobuf("borrow-revoke-uaf-probe: round 2 returned 0x%lx\n", ret2);
  print_nobuf("borrow-revoke-uaf-probe: word after round 2 = 0x%016lx\n",
              region_words[0]);

  if (region_words[0] == BORROW_REVOKE_UAF_SENTINEL_STAGE2) {
    print_nobuf("borrow-revoke-uaf-probe: NO-TRAP-GAP use-after-revoke store "
                "landed (stage-2 sentinel visible)\n");
  } else {
    print_nobuf("borrow-revoke-uaf-probe: use-after-revoke did not update the "
                "lender's view (word=0x%016lx)\n",
                region_words[0]);
  }

  capstone_cleanup();
  return 0;
}
