#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_HIER_REVOKE_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_HIER_REVOKE_PROBE_H

/* Phase-0 feasibility probe for the HIERARCHICAL revoke-on-free allocator
 * (task 010, checkpoint H). Proves the tree PRIMITIVE in isolation, before any
 * SQLite, answering the one mechanism question the whole checkpoint rests on:
 *
 *   Does REVOKE of a connection's senior MREV handle sweep a child that was
 *   SPLIT off that connection's sub-arena AFTER the MREV -- while a sibling
 *   connection's child survives?
 *
 * If yes, hierarchical ownership (parent connection revoke cascades to child
 * statements, scoped, not a global wipe) is expressible intra-domain with only
 * SPLIT + MREV + REVOKE -- no monitor/region change. If no, that wall is the
 * checkpoint's finding. See revoke_on_free_hier_alloc.h.
 *
 * Receive protocol: identical to revoke-on-free-probe -- the delivered
 * REGION_SHARE capability IS domain_main's first argument on the func==1 entry.
 */

#define HIER_PROBE_REGION_SIZE (64u * 1024u)
#define HIER_PROBE_DPI_REGION_SHARE 1u

/* shared_region_annotated(): PERM_INOUT (RW) + REV_TRANSFERRED. */
#define HIER_PROBE_ANNOTATION_PERM_INOUT 0x1u
#define HIER_PROBE_ANNOTATION_REV_TRANSFERRED 0x3u

/* Per-connection sub-arena size carved off the main grant. */
#define HIER_PROBE_SUBARENA 0x2000u

#define HIER_PROBE_SENTINEL_A 0x5Eu
#define HIER_PROBE_SENTINEL_B 0x3Cu

/* Return codes (a domain that reaches its *res store did NOT fault). */
#define HIER_RET_CHILD_REVOKED_NOTRAP 0x08700000u
#define HIER_RET_NO_CLOSE_OK 0x0872005eu        /* control: no close, child live */
#define HIER_RET_SIBLING_SURVIVES_OK 0x0873003cu /* close A; B's child survives */

#endif
