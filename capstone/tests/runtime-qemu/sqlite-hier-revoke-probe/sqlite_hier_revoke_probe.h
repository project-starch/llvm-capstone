#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_SQLITE_HIER_REVOKE_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_SQLITE_HIER_REVOKE_PROBE_H

/* Stage-2 experiment for the use-after-close class (cve-repros rows 4/8/10/12,
 * HIERARCHICAL-REVOKE): does revoking a PARENT borrow cascade to a CHILD borrow?
 *
 *   parent (connection) == a region lent to the host as a revocable borrow
 *   child  (statement/value buffer) == a second region lent to the host
 *   sqlite3_close(connection) == revoke_region(parent)
 *
 * The host reads+caches the child value (round 1); the engine then "closes" the
 * connection by revoking the PARENT; round 2 the host re-reads the cached CHILD
 * pointer. If the parent revoke cascades (the rev-tree is a single depth-ordered
 * list, so a senior revoke may invalidate junior handles), the round-2 read
 * faults == use-after-close trapped by hierarchical cascade. If not, the child
 * read still succeeds == parent/child are independent rev roots and a faithful
 * cascade needs a monitor extension (child split-derived from the parent).
 *
 * This probe deliberately uses ONLY existing lender ops (create_region,
 * shared_region_annotated, revoke_region) so it needs no firmware change: it is
 * the cheap feasibility test that decides whether the hierarchical shape needs
 * new monitor work.
 */

#define SQLITE_HIER_REGION_SIZE 4096UL
#define SQLITE_HIER_COLUMN_VALUE 0xC01A0DEDC01A0DEDUL
#define SQLITE_HIER_FAULT_SENTINEL 0x0FA017EDUL

#endif
