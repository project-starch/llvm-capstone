#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_REVOKE_ON_FREE_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_REVOKE_ON_FREE_PROBE_H

/* Phase-0 feasibility probe for the revoke-on-free allocator (task 008).
 *
 * Proves the allocator PRIMITIVE in isolation, before any SQLite: allocate two
 * buffers from one monitor-granted arena, free (revoke) the first, and show its
 * cached alias faults while the second buffer still works and a third allocation
 * succeeds. This is the allocator analogue of the held-cap probe's
 * held_arena_survives_revoke.
 *
 * Receive protocol: identical to intra-domain-mrev-revoke-probe -- the delivered
 * REGION_SHARE capability IS domain_main's first argument on the func==1 entry.
 * Restated here (a few lines) rather than pulling in that probe's header, whose
 * comment block is about a different probe; the allocator itself is the shared
 * code (revoke_on_free_alloc.h).
 */

#define ROF_PROBE_REGION_SIZE (64u * 1024u)
#define ROF_PROBE_DPI_REGION_SHARE 1u

/* shared_region_annotated(): PERM_INOUT (RW) + REV_TRANSFERRED. */
#define ROF_PROBE_ANNOTATION_PERM_INOUT 0x1u
#define ROF_PROBE_ANNOTATION_REV_TRANSFERRED 0x3u

#define ROF_PROBE_SENTINEL_A 0x5Eu
#define ROF_PROBE_SENTINEL_B 0x3Cu

/* Return codes (a domain that reaches its *res store did NOT fault). */
#define ROF_RET_USE_AFTER_FREE_NOTRAP 0x08100000u
#define ROF_RET_NO_FREE_OK 0x0812005eu       /* control: no free, alias live */
#define ROF_RET_SIBLING_SURVIVES_OK 0x0813003cu /* free A; B + a 3rd malloc live */

#endif
