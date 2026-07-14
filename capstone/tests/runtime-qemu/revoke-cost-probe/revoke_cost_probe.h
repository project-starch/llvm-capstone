#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_REVOKE_COST_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_REVOKE_COST_PROBE_H

/* Temporal-safety overhead microbenchmark -- Capstone side (PI 2026-07-14).
 *
 * The malloc/free-heavy companion to the borrow-cost probe. Where borrow-cost
 * measures one borrow op, this measures the cost of the *revoke-on-free
 * allocator* over a malloc/free-heavy workload, so it can be compared to CHERI's
 * revocation-sweep cost on the same class of workload (plan
 * perf-cheri-vs-capstone-qemu.md). Three configs of the SAME alloc/touch/free
 * loop, one per domain build (-DROF_COST_MODE), each bracketed by the csrdicount
 * readout under -icount:
 *
 *   BUMP     (0): plain bump allocator over a broad NONLIN heap cap -- interior
 *                 pointers, no per-object capability machinery, free is a no-op.
 *                 The UNPROTECTED baseline (Capstone analogue of CHERI
 *                 revocation-off): spatial bounds at heap granularity only.
 *   NOREVOKE (1): the revoke-on-free allocator (revoke_on_free_alloc.h) with the
 *                 revoke suppressed (rof_no_revoke=1) -- SPLIT+mrev+delin per
 *                 malloc, but free does not revoke. Isolates the ALLOCATION-side
 *                 capability cost.
 *   REVOKE   (2): the full revoke-on-free allocator -- SPLIT+mrev+delin per
 *                 malloc, REVOKE per free. Full temporal safety.
 *
 * Breakdown (the PI's "where does the cost come from"):
 *   alloc-side overhead = NOREVOKE - BUMP   (make each allocation revocable)
 *   revoke overhead     = REVOKE   - NOREVOKE (the free-time revoke)
 *   total temporal cost = REVOKE   - BUMP
 *
 * FUNCTIONAL-MODEL PROXY, NOT SILICON TIMING (same caveat as borrow-cost): this
 * is a deterministic dynamic-instruction count under -icount, an honest overhead
 * proxy, not cycle-accurate timing.
 *
 * Shared by the .user controller (buildroot gcc) and the .dom domain payload
 * (Capstone clang). Keep it free of capability builtins.
 */

/* Arena must exceed ITERS*BLOCK: neither allocator coalesces (rof SPLITs one-way
 * from the top; bump only advances), so the arena monotonically shrinks even
 * though every block is freed each iteration. 512*64 = 32 KiB << 256 KiB. */
#define ROF_COST_REGION_SIZE (256u * 1024u)
#define ROF_COST_ITERS 512u
#define ROF_COST_BLOCK 64u

/* Config selector (compile-time, -DROF_COST_MODE=...). */
#define ROF_COST_MODE_BUMP 0
#define ROF_COST_MODE_NOREVOKE 1
#define ROF_COST_MODE_REVOKE 2

/* DPI codes + region annotations: one LINEAR arena, RW, monitor keeps no
 * revocation handle (so the domain can SPLIT/mrev it). Same as the rof probe. */
#define ROF_COST_DPI_REGION_SHARE 1u
#define ROF_COST_ANNOTATION_PERM_INOUT 0x1u
#define ROF_COST_ANNOTATION_REV_TRANSFERRED 0x3u

/* csdebugcount slots the domain writes and csdebugcountprint dumps. */
#define ROF_COST_SLOT_ITERS 0
#define ROF_COST_SLOT_EMPTY 1
#define ROF_COST_SLOT_ALLOCFREE 2
#define ROF_COST_SLOT_MODE 3

#define ROF_COST_RET_OK 0x22390000u

#endif
