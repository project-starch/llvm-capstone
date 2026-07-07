#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_SQLITE_HIER_CHILD_REVOKE_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_SQLITE_HIER_CHILD_REVOKE_PROBE_H

/* Stage-2 "after" for the use-after-close class (cve-repros rows 4/5/7/8/9/10/12,
 * HIERARCHICAL-REVOKE): a statement/value pointer that lives *inside* a SQLite
 * connection, dereferenced after sqlite3_close(connection). The proposal's H
 * primitive: closing the connection must invalidate every pointer beneath it.
 *
 * This is the POSITIVE counterpart to sqlite-hier-revoke-probe, which showed that
 * two independent create_region()s are independent rev-tree roots -> revoking the
 * parent does NOT cascade to the child. Here the child is derived from the parent
 * with the new monitor op share_child_region(): the monitor mints a senior
 * revocation handle on the parent (retained by the engine) and __split()s the
 * child out of the parent's own capability, so the child is junior in the parent's
 * rev lineage. revoke_region(parent) == sqlite3_close then cascades:
 * __revoke(parent_rev) invalidates the derived child.
 *
 *   parent (connection)       == create_region(); the connection's backing store
 *   child  (statement value)  == [CHILD_OFFSET, CHILD_OFFSET+CHILD_LEN) inside it,
 *                                shared via share_child_region()
 *   sqlite3_close(connection) == revoke_region(parent)
 *
 * Flow: the engine writes the column value inside the connection at CHILD_OFFSET
 * and shares the child; round 1 the host reads it and caches the pointer; the
 * engine "closes" (revokes the parent); round 2 the host re-reads its CACHED CHILD
 * pointer = the use-after-close. With the hierarchical cascade the cached
 * capability reloads untagged and the read faults; the monitor terminates the
 * domain and the engine observes the fault sentinel instead of a stale value.
 * Safe-fail: the use-after-close becomes a deterministic trap.
 */

#define SQLITE_HIER_CHILD_PARENT_SIZE 4096UL
/* The column value sits inside the connection object (not at offset 0), so the
 * child is a genuine sub-window carved with a head + tail split. 8-byte aligned. */
#define SQLITE_HIER_CHILD_OFFSET 64UL
#define SQLITE_HIER_CHILD_LEN 8UL
#define SQLITE_HIER_CHILD_COLUMN_VALUE 0xC01A0DEDC01A0DEDUL
#define SQLITE_HIER_CHILD_FAULT_SENTINEL 0x0FA017EDUL

#endif
