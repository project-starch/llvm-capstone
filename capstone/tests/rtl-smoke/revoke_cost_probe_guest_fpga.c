#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "../runtime-qemu/revoke-cost-probe/revoke_cost_probe.h"

/* Controller for the temporal-safety overhead microbenchmark -- RTL/FPGA variant
 * (task-016). Hardware port of revoke-cost-probe/revoke_cost_probe_guest.c.
 * Identical setup (create domain, grant a LINEAR arena, CALL) but, because the
 * FPGA core has no csdebugcount serial op, the domain writes its 4 counters into
 * a RETAINED (REV_SHARED) results region and THIS controller reads them back and
 * printf()s them -- which lands on the FPGA UART (the platform's Terminal tab).
 *
 * One allocator config per .dom build (bump/norevoke/revoke); run all three and
 * diff the per-op numbers to get the breakdown:
 *   alloc-side overhead = norevoke - bump      (make each allocation revocable)
 *   revoke overhead     = revoke   - norevoke  (the free-time revoke, the O(1) op)
 *   total temporal cost = revoke   - bump
 *
 * UNTESTED IN-SANDBOX: build + run on the caplifive toolchain / FPGA.
 */

#define print_nobuf(...)  \
  do {                    \
    printf(__VA_ARGS__);  \
    fflush(stdout);       \
  } while (0)

#define TAG "revoke-cost-fpga"

/* REV_SHARED: the host keeps a valid mapping after sharing (unlike
 * REV_TRANSFERRED 0x3), so it can read the results back. */
#define ROF_COST_ANNOTATION_REV_SHARED 0x2u

/* The results region only holds 4 unsigned longs (32 B); keep it a single page
 * rather than a second ROF_COST_REGION_SIZE (256 KiB) arena -- two large regions
 * starve the domain's shared-region space and the arena arrives unusable. */
#define ROF_RESULTS_REGION_SIZE 4096u

static const char *mode_name(unsigned long m) {
  switch (m) {
    case ROF_COST_MODE_BUMP:     return "bump";
    case ROF_COST_MODE_NOREVOKE: return "norevoke";
    case ROF_COST_MODE_REVOKE:   return "revoke";
    default:                     return "?";
  }
}

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "%s: %s (observed=0x%016lx)\n", TAG, message, observed);
  fflush(stderr);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <revoke_cost_fpga_{bump,norevoke,revoke}.dom>\n",
            argv[0]);
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

  /* TWO regions (see revoke_cost_fpga.c): (1) the LINEAR arena the allocator
   * SPLIT/mrev/revokes, handed REV_TRANSFERRED; (2) the results region, handed
   * REV_SHARED so the host RETAINS its mapping and can read the 4 counters back
   * after the call. A single REV_TRANSFERRED region cannot be read back by the
   * host (task-007 host-landmine; see the borrow-cost port RESULTS.md). */
  region_id_t arena_id = create_region(ROF_COST_REGION_SIZE);
  unsigned char *arena_bytes =
      (unsigned char *)map_region(arena_id, ROF_COST_REGION_SIZE);
  if (!arena_bytes)
    return fail_cleanup("map_region (arena) failed", 0);
  memset(arena_bytes, 0, ROF_COST_REGION_SIZE);

  region_id_t results_id = create_region(ROF_RESULTS_REGION_SIZE);
  unsigned char *region_bytes =
      (unsigned char *)map_region(results_id, ROF_RESULTS_REGION_SIZE);
  if (!region_bytes)
    return fail_cleanup("map_region (results) failed", 0);
  memset(region_bytes, 0, ROF_RESULTS_REGION_SIZE);
  print_nobuf("%s: created arena region ID = %lu, results region ID = %lu\n",
              TAG, arena_id, results_id);

  /* Share ORDER is load-bearing: arena first (REV_TRANSFERRED, domain owns it),
   * results second (REV_SHARED, host retains its mapping). */
  shared_region_annotated(dom_id, arena_id, ROF_COST_ANNOTATION_PERM_INOUT,
                          ROF_COST_ANNOTATION_REV_TRANSFERRED);
  shared_region_annotated(dom_id, results_id, ROF_COST_ANNOTATION_PERM_INOUT,
                          ROF_COST_ANNOTATION_REV_SHARED);
  print_nobuf("%s: arena transferred (LIN, RW); results shared (host retains)\n",
              TAG);

  unsigned long retval = call_dom(dom_id);
  print_nobuf("%s: call retval = 0x%08lx\n", TAG, retval);
  if ((unsigned)retval != ROF_COST_RET_OK)
    return fail_cleanup("domain did not report measurement OK", retval);

  const unsigned long *r = (const unsigned long *)region_bytes;
  unsigned long iters = r[ROF_COST_SLOT_ITERS];
  unsigned long empty = r[ROF_COST_SLOT_EMPTY];
  unsigned long allocfree = r[ROF_COST_SLOT_ALLOCFREE];
  unsigned long mode = r[ROF_COST_SLOT_MODE];

  if (iters == 0)
    return fail_cleanup("domain wrote no results (iters=0)", 0);

  /* Raw totals (cycles) so the trace is auditable, not just the derived cost. */
  print_nobuf("%s: RAW mode=%s iters=%lu empty=%lu allocfree=%lu\n",
              TAG, mode_name(mode), iters, empty, allocfree);

  /* Per-operation cycle cost = (allocfree - empty) / iters. */
  unsigned long perop = (allocfree - empty) / iters;
  print_nobuf("%s: RESULT cycles/op  mode=%s  alloc_free=%lu\n",
              TAG, mode_name(mode), perop);

  print_nobuf("%s: measurement complete\n", TAG);
  capstone_cleanup();
  return 0;
}
