#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_INTRA_DOMAIN_MREV_REVOKE_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_INTRA_DOMAIN_MREV_REVOKE_PROBE_H

/* Single-domain held-cap BORROW-REVOKE probe (row3 Option B, the "gold
 * standard" shape). See README.md.
 *
 * ONE domain receives a REAL monitor-granted linear capability, MREVs it, uses
 * an alias derived from it, REVOKEs at a lifecycle point, and the cached alias
 * faults. Stronger than the two-entity Option A probe (revoke is intra-domain,
 * over a held capability) and than the task-005 codegen spike (which minted its
 * arena with the csdebuggencap debug op; this one comes down the real delivery
 * path).
 *
 * Shared by the .user controller (buildroot gcc) and the .dom domain payloads
 * (Capstone clang). Keep it free of capability builtins.
 */

#define PROBE_REGION_SIZE 4096UL

/* Byte the probes write through the held capability, and the value written. */
#define PROBE_OFFSET 8u
#define PROBE_SENTINEL_LIVE 0x5Eu

/* Where held_split_sibling_ok cuts the arena: low half keeps [base, MID),
 * high half gets [MID, end) with a fresh revocation node. */
#define PROBE_SPLIT_MID 2048UL

/* DPI function codes. The monitor enters the domain once per operation and
 * passes the code as domain_main's second (scalar) argument. Mirrors
 * caplifive-buildroot/package/capstone-sbi-domain/capstone-sbi/sbi_capstone.h.
 */
#define PROBE_DPI_CALL 0x0u
#define PROBE_DPI_REGION_SHARE 0x1u

/* Region annotations (same monitor header).
 *
 * REV_TRANSFERRED: the region is handed over LINEAR with no monitor-retained
 * revocation handle -- the domain owns full authority over it, which is exactly
 * the single-domain shape. (REV_BORROWED would also deliver a LIN cap but the
 * monitor would keep an __mrev senior to it.)
 *
 * PERM_INOUT (RW), not PERM_IN: helper_cstighten silently DELINEARISES a LIN
 * capability whose permissions do not allow write (op_helper.c, "immutable
 * linear capability can be safely invalidated without scrubbing the data"), and
 * helper_csmrev asserts CAP_TYPE_LIN. A read-only grant is therefore not
 * MREV-able and would abort the emulator rather than fault cleanly.
 */
#define PROBE_ANNOTATION_PERM_INOUT 0x1u
#define PROBE_ANNOTATION_REV_TRANSFERRED 0x3u

/* domain_main return values. Only the OK probes ever return: a probe that takes
 * a capability fault halts the domain, and the guest never gets back to the
 * shell. Each value is unique so a marker match cannot pass on the wrong probe.
 */
#define PROBE_RET_NO_REVOKE_OK 0x22300000u   /* | sentinel  -> 0x2230005e */
#define PROBE_RET_UNRELATED_OK 0x22310033u
#define PROBE_RET_REVOKE_NOTRAP 0x22320000u  /* fault probe: never returned */
#define PROBE_RET_MEM_NOTRAP 0x22330000u     /* fault probe: never returned */
#define PROBE_RET_AMBIENT_NOTRAP 0x22340000u /* fault probe: never returned */
#define PROBE_RET_SPLIT_OK 0x22350044u
#define PROBE_RET_LIFECYCLE_NOTRAP 0x22360000u /* fault probe: never returned */
#define PROBE_RET_ARENA_SURVIVES_OK 0x22370077u

#endif
