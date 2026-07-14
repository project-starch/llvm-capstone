#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "../runtime-qemu/borrow-cost-probe/borrow_cost_probe.h"

/* Controller for the borrow-path cost measurement -- RTL/FPGA variant (task-016).
 *
 * Hardware port of borrow-cost-probe/borrow_cost_probe_guest.c. Identical setup
 * (create domain, create + share a LINEAR RW region, CALL), but instead of the
 * domain dumping results via the QEMU csdebugcount serial op, the domain writes
 * them into the shared region and THIS controller reads them back and printf()s
 * them -- which lands on the FPGA UART (the platform's Terminal tab).
 *
 * The per-operation cost is (variant_total - empty_total) / iters, computed here
 * so the UART line is directly the paper number. run-... / a human copies the
 * "RESULT" lines off the Terminal tab.
 *
 * UNTESTED IN-SANDBOX: build + run on the caplifive toolchain / FPGA.
 */

#define print_nobuf(...) \
  do {                   \
    printf(__VA_ARGS__); \
    fflush(stdout);      \
  } while (0)

#define TAG "borrow-cost-fpga"

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "%s: %s (observed=0x%016lx)\n", TAG, message, observed);
  fflush(stderr);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <borrow_cost_fpga.dom>\n", argv[0]);
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

  /* Read the eight results the domain wrote to the region base. */
  const unsigned long *r = (const unsigned long *)region_bytes;
  unsigned long iters = r[BORROW_COST_SLOT_ITERS];
  unsigned long empty = r[BORROW_COST_SLOT_EMPTY];
  unsigned long raw = r[BORROW_COST_SLOT_RAW];
  unsigned long borrow = r[BORROW_COST_SLOT_BORROW];
  unsigned long copy = r[BORROW_COST_SLOT_COPY];
  unsigned long copy_bytes = r[BORROW_COST_SLOT_COPY_BYTES];
  unsigned long copy2 = r[BORROW_COST_SLOT_COPY2];
  unsigned long copy2_bytes = r[BORROW_COST_SLOT_COPY2_BYTES];

  if (iters == 0)
    return fail_cleanup("domain wrote no results (iters=0)", 0);

  /* Raw totals (cycles) -- so the trace is auditable, not just the derived cost. */
  print_nobuf("%s: RAW iters=%lu empty=%lu raw=%lu borrow=%lu copy%lu=%lu copy%lu=%lu\n",
              TAG, iters, empty, raw, borrow, copy_bytes, copy, copy2_bytes, copy2);

  /* Per-operation cycle cost = (variant - empty) / iters. Integer math on the
   * deltas; the raw line above lets anyone recompute with rounding if needed. */
  unsigned long raw_pp = (raw - empty) / iters;
  unsigned long borrow_pp = (borrow - empty) / iters;
  unsigned long copy_pp = (copy - empty) / iters;
  unsigned long copy2_pp = (copy2 - empty) / iters;

  print_nobuf("%s: RESULT cycles/op  raw=%lu  borrow=%lu  copy@%luB=%lu  copy@%luB=%lu\n",
              TAG, raw_pp, borrow_pp, copy_bytes, copy_pp, copy2_bytes, copy2_pp);
  if (raw_pp > 0) {
    print_nobuf("%s: RESULT vs-raw     borrow=%lu.%02lux  copy@%luB=%lu.%02lux  copy@%luB=%lu.%02lux\n",
                TAG, borrow_pp / raw_pp, (borrow_pp * 100 / raw_pp) % 100,
                copy_bytes, copy_pp / raw_pp, (copy_pp * 100 / raw_pp) % 100,
                copy2_bytes, copy2_pp / raw_pp, (copy2_pp * 100 / raw_pp) % 100);
  }

  print_nobuf("%s: measurement complete\n", TAG);
  capstone_cleanup();
  return 0;
}
