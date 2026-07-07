#include <stdio.h>
#include <stdlib.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "sqlite_hier_child_revoke_probe.h"

/* Engine (lender/controller): owns the connection's backing store, derives a
 * child (statement value) borrow *from inside it* via share_child_region, then
 * "closes" the connection by revoking the PARENT. If the hierarchical cascade
 * works, the child borrow the host cached is invalidated. See the header for the
 * mapping to the use-after-close cve-repros rows. */

#define print_nobuf(...)  \
  do {                    \
    printf(__VA_ARGS__);  \
    fflush(stdout);       \
  } while (0)

/* PERM_IN (read): the engine produces the column data and the host reads it. */
#define PROBE_ANNOTATION_PERM_IN 0x0

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "sqlite-hier-child: %s (observed=0x%016lx)\n", message,
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
    fprintf(stderr, "sqlite-hier-child: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);

  /* Parent = the connection's backing store; the column value lives inside it. */
  region_id_t parent = create_region(SQLITE_HIER_CHILD_PARENT_SIZE);
  unsigned long *conn =
      (unsigned long *)map_region(parent, SQLITE_HIER_CHILD_PARENT_SIZE);
  if (!conn)
    return fail_cleanup("map_region(parent) failed", 0);
  conn[SQLITE_HIER_CHILD_OFFSET / sizeof(unsigned long)] =
      SQLITE_HIER_CHILD_COLUMN_VALUE;

  /* Lend a child sub-window [CHILD_OFFSET, +CHILD_LEN) DERIVED from the parent, so
   * a later revoke of the parent (sqlite3_close) cascades to it. */
  share_child_region(dom_id, parent, SQLITE_HIER_CHILD_OFFSET,
                     SQLITE_HIER_CHILD_LEN, PROBE_ANNOTATION_PERM_IN);
  print_nobuf("sqlite-hier-child: statement value (child) derived + borrowed from "
              "connection\n");

  /* Round 1: the host reads the child value while the connection is open and
   * caches the pointer (the binding that outlives the close). */
  unsigned long r1 = call_dom(dom_id);
  print_nobuf("sqlite-hier-child: round 1 retval = 0x%016lx\n", r1);
  if (r1 != SQLITE_HIER_CHILD_COLUMN_VALUE)
    return fail_cleanup("round 1 did not read the child value", r1);
  print_nobuf("sqlite-hier-child: host read statement value OK before close\n");

  /* sqlite3_close(connection): revoke the PARENT. With a derived child this
   * cascades: __revoke(parent_rev) invalidates the junior child. */
  revoke_region(parent);
  print_nobuf("sqlite-hier-child: close revoked the connection (parent)\n");

  /* Round 2: the host re-reads its CACHED child pointer = the use-after-close. */
  print_nobuf("sqlite-hier-child: entering round 2 (use-after-close read)\n");
  unsigned long r2 = call_dom(dom_id);
  print_nobuf("sqlite-hier-child: round 2 returned 0x%016lx\n", r2);

  if (r2 == SQLITE_HIER_CHILD_FAULT_SENTINEL) {
    print_nobuf("sqlite-hier-child: use-after-close TRAPPED via hierarchical "
                "cascade (parent revoke invalidated the derived child)\n");
  } else if (r2 == SQLITE_HIER_CHILD_COLUMN_VALUE) {
    print_nobuf("sqlite-hier-child: NO-CASCADE-GAP child still readable after "
                "parent revoke\n");
  } else {
    print_nobuf("sqlite-hier-child: round 2 unexpected retval 0x%016lx\n", r2);
  }

  capstone_cleanup();
  return 0;
}
