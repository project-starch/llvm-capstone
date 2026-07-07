#include <stdio.h>
#include <stdlib.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "sqlite_hier_revoke_probe.h"

/* Engine (lender): lends a parent (connection) region and a child (statement
 * value) region, both revocable; "closes" the connection by revoking the PARENT
 * and checks whether the child borrow is cascaded. See the header. */

#define print_nobuf(...)  \
  do {                    \
    printf(__VA_ARGS__);  \
    fflush(stdout);       \
  } while (0)

/* PERM_IN (read) + REV_BORROWED (revocable linear borrow). */
#define PROBE_ANNOTATION_PERM_IN 0x0
#define PROBE_ANNOTATION_REV_BORROWED 0x1

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "sqlite-hier-revoke: %s (observed=0x%016lx)\n", message,
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
    fprintf(stderr, "sqlite-hier-revoke: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);

  /* Parent = the connection authority; child = a statement/value buffer. */
  region_id_t parent = create_region(SQLITE_HIER_REGION_SIZE);
  region_id_t child = create_region(SQLITE_HIER_REGION_SIZE);
  unsigned long *child_buf =
      (unsigned long *)map_region(child, SQLITE_HIER_REGION_SIZE);
  if (!child_buf)
    return fail_cleanup("map_region(child) failed", 0);
  child_buf[0] = SQLITE_HIER_COLUMN_VALUE;

  /* Lend parent first, then child, so the borrower's REGION_COUNT-1 query
   * resolves to the child (the value it reads). */
  shared_region_annotated(dom_id, parent, PROBE_ANNOTATION_PERM_IN,
                          PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf("sqlite-hier-revoke: connection (parent) borrowed to host\n");
  shared_region_annotated(dom_id, child, PROBE_ANNOTATION_PERM_IN,
                          PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf("sqlite-hier-revoke: statement value (child) borrowed to host\n");

  /* Round 1: the host reads the statement value and caches its pointer. */
  unsigned long r1 = call_dom(dom_id);
  print_nobuf("sqlite-hier-revoke: round 1 retval = 0x%016lx\n", r1);
  if (r1 != SQLITE_HIER_COLUMN_VALUE)
    return fail_cleanup("round 1 did not read the child value", r1);
  print_nobuf("sqlite-hier-revoke: host read statement value OK before close\n");

  /* sqlite3_close(connection): revoke the PARENT only. */
  revoke_region(parent);
  print_nobuf("sqlite-hier-revoke: close revoked the connection (parent)\n");

  /* Round 2: the host re-reads its cached CHILD pointer (use-after-close). */
  unsigned long r2 = call_dom(dom_id);
  print_nobuf("sqlite-hier-revoke: round 2 returned 0x%016lx\n", r2);

  if (r2 == SQLITE_HIER_FAULT_SENTINEL) {
    print_nobuf("sqlite-hier-revoke: use-after-close TRAPPED via hierarchical "
                "cascade (parent revoke invalidated child)\n");
  } else if (r2 == SQLITE_HIER_COLUMN_VALUE) {
    print_nobuf("sqlite-hier-revoke: NO-CASCADE parent revoke did not invalidate "
                "child (independent rev roots; needs derived-child firmware)\n");
  } else {
    print_nobuf("sqlite-hier-revoke: round 2 unexpected retval 0x%016lx\n", r2);
  }

  capstone_cleanup();
  return 0;
}
