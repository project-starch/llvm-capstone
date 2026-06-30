#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_BORROW_REVOKE_UAF_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_BORROW_REVOKE_UAF_PROBE_H

/* M0 borrow->revoke->use-after-revoke probe.
 *
 * Demonstrates the load-bearing claim of the SQLite marshalling direction
 * (design/sqlite-marshalling-feasibility.md): a borrowed region capability,
 * once the lender revokes it, can no longer be dereferenced by the borrower.
 *
 * Topology mirrors shared-region-probe: the .user controller is the LENDER
 * (it owns the region and calls revoke_region), the .smode payload is the
 * BORROWER running inside the domain (it caches the delegated pointer in
 * round 1 and dereferences the cached pointer in round 2, after the lender
 * has revoked).
 */

#define BORROW_REVOKE_UAF_REGION_SIZE 4096UL

/* Stage-1 sentinel the borrower writes while the borrow is live (round 1). */
#define BORROW_REVOKE_UAF_SENTINEL_STAGE1 0x1111111111111111UL
/* Stage-2 sentinel the borrower would write IF the post-revoke deref did not
 * fault (round 2). Observing this value in the lender's mapping is the
 * negative outcome: use-after-revoke was NOT trapped. */
#define BORROW_REVOKE_UAF_SENTINEL_STAGE2 0x2222222222222222UL

/* dom_return values, so the serial log shows which round completed. */
#define BORROW_REVOKE_UAF_RET_ROUND1 0x101UL
#define BORROW_REVOKE_UAF_RET_ROUND2 0x202UL

#endif
