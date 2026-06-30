#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_REVOKE_MATRIX_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_REVOKE_MATRIX_PROBE_H

/* Revocation enforcement test matrix (cases 2 and 3 of
 * agent-handoff/design/revocation-enforcement-proposal.md §6). Extends the M0
 * borrow-revoke-uaf-probe with different ways the borrower holds the delegated
 * capability across the revoke:
 *
 *   CASE 2 (memory-stored): the borrowed cap lives in a .bss pointer slot and is
 *           reloaded on the round-2 dereference.
 *   CASE 3 (explicit stc/ldc): round 1 stores the borrowed cap into a separate
 *           capability slot (stc); round 2 reloads it (ldc) and dereferences.
 *
 * Both exercise the cap-load untag enforcement point. Until the recording-side
 * fix lands (pending author), revocation marks nothing invalid, so every case is
 * expected to show the NO-TRAP gap (round-2 store lands). After the fix they must
 * fault. CASE 4 (senior-cascade sub-cap) is deferred: it needs SHRINK in the
 * borrower, which the buildroot gcc used here cannot emit (no Capstone builtins).
 */

#ifndef REVOKE_MATRIX_CASE
#define REVOKE_MATRIX_CASE 2
#endif

#define REVOKE_MATRIX_REGION_SIZE 4096UL
#define REVOKE_MATRIX_SENTINEL_STAGE1 0x1111111111111111UL
#define REVOKE_MATRIX_SENTINEL_STAGE2 0x2222222222222222UL
#define REVOKE_MATRIX_RET_ROUND1 0x101UL
#define REVOKE_MATRIX_RET_ROUND2 0x202UL

#endif
