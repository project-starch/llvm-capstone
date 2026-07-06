#include <stdio.h>
#include <stdlib.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "sqlite_borrow_revoke_probe.h"

/* Engine (lender/controller): owns the current row buffer, lends it to the host
 * binding as a revocable borrow, then revokes it at sqlite3_step. See the header
 * for the mapping to diesel RUSTSEC-2021-0037. */

#define print_nobuf(...)  \
  do {                    \
    printf(__VA_ARGS__);  \
    fflush(stdout);       \
  } while (0)

/* PERM_IN (0x0) grants the borrower READ authority: the engine produces the
 * column data and the host reads it (E->H borrow). (PERM_OUT would be write-only,
 * the wrong direction for a column read.) REV_BORROWED (0x1) makes it revocable. */
#define PROBE_ANNOTATION_PERM_IN 0x0
#define PROBE_ANNOTATION_REV_BORROWED 0x1

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "sqlite-borrow-revoke: %s (observed=0x%016lx)\n", message,
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
    fprintf(stderr, "sqlite-borrow-revoke: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);

  /* The engine's buffer for the current row. */
  region_id_t region_id = create_region(SQLITE_BORROW_REGION_SIZE);
  unsigned long *row_buffer =
      (unsigned long *)map_region(region_id, SQLITE_BORROW_REGION_SIZE);
  if (!row_buffer)
    return fail_cleanup("map_region failed", 0);
  row_buffer[0] = SQLITE_BORROW_COLUMN_VALUE; /* sqlite3_column_text() content */

  /* Lend the column buffer to the host binding as a revocable borrow. */
  shared_region_annotated(dom_id, region_id, PROBE_ANNOTATION_PERM_IN,
                          PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf("sqlite-borrow-revoke: column buffer borrowed to host\n");

  /* Round 1: the host reads column_text while the row is current (borrow live)
   * and returns the value it read, proving the borrow was delivered. */
  unsigned long r1 = call_dom(dom_id);
  print_nobuf("sqlite-borrow-revoke: round 1 (read before step) retval = 0x%016lx\n",
              r1);
  if (r1 != SQLITE_BORROW_COLUMN_VALUE)
    return fail_cleanup("round 1 did not read the column (borrow not live?)", r1);
  print_nobuf("sqlite-borrow-revoke: host read column OK before step\n");

  /* sqlite3_step: advance the row -> the previous row buffer's borrow ends. */
  revoke_region(region_id);
  print_nobuf("sqlite-borrow-revoke: step revoked the column borrow\n");

  /* Round 2: the host re-reads its CACHED column pointer = the diesel UAF. */
  print_nobuf("sqlite-borrow-revoke: entering round 2 (use-after-free read)\n");
  unsigned long r2 = call_dom(dom_id);
  print_nobuf("sqlite-borrow-revoke: round 2 returned 0x%016lx\n", r2);

  if (r2 == SQLITE_BORROW_FAULT_SENTINEL) {
    print_nobuf("sqlite-borrow-revoke: use-after-free read TRAPPED "
                "(domain faulted, ret=0x%lx)\n", r2);
  } else if (r2 == SQLITE_BORROW_COLUMN_VALUE) {
    print_nobuf("sqlite-borrow-revoke: NO-TRAP-GAP use-after-free read returned "
                "stale column value\n");
  } else {
    print_nobuf("sqlite-borrow-revoke: round 2 unexpected retval 0x%016lx\n", r2);
  }

  capstone_cleanup();
  return 0;
}
