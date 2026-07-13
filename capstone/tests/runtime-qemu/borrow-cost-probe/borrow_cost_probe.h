#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_BORROW_COST_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_BORROW_COST_PROBE_H

/* Borrow-path cost measurement (paper deliverable 2 -- task-014).
 *
 * ONE domain receives a real monitor-granted LINEAR arena (the same
 * REGION_SHARE / REV_TRANSFERRED delivery the intra-domain-mrev-revoke probe
 * uses) and runs three variants of the SAME boundary operation -- borrow one
 * result word across the host/engine boundary and use it -- differing only in
 * how the lend is protected:
 *
 *   1. RAW pointer  -- today's zero-copy path: dereference the shared word.
 *   2. CAPABILITY BORROW -- mint a revocation cap (mrev) + delegate a working
 *      cap (delin) + access it + revoke. The paper's mechanism, exactly the ops
 *      the intra-domain validation used.
 *   3. COPY baseline -- the TRANSIENT-style defensive copy the mechanism
 *      replaces: word-copy the payload into a private buffer, then read it.
 *
 * Each variant runs a fixed-count inner loop bracketed by the csrdicount
 * emulator readout (raw retired-instruction count under -icount), plus an empty
 * calibration loop of the same shape. The per-operation instruction count is
 * (variant_total - empty_total) / N. Results are reported through the Capstone
 * debug counters (csdebugcount / csdebugcountprint) so they land in the serial
 * log; the run script greps them.
 *
 * FUNCTIONAL-MODEL PROXY, NOT SILICON TIMING: QEMU is an ISA/functional model
 * with no pipeline, cache, or cycle model. csrdicount is a deterministic
 * dynamic-instruction count, which is an honest overhead proxy but NOT a
 * cycle-accurate timing measurement. See RESULTS.md.
 *
 * Shared by the .user controller (buildroot gcc) and the .dom domain payload
 * (Capstone clang). Keep it free of capability builtins.
 */

#define BORROW_COST_REGION_SIZE 4096UL

/* Copy-baseline payload sizes, in bytes (representative borrowed result values,
 * e.g. a cached SQLite column/row buffer -- 256 B matches the arena carve the
 * scaffold probes use). Two sizes make the point empirical: copy cost is
 * O(size) while borrow and raw are O(1). */
#define BORROW_COST_COPY_BYTES 256UL
#define BORROW_COST_COPY_BYTES_2 1024UL
/* Buffers must hold the larger payload. */
#define BORROW_COST_BUF_BYTES BORROW_COST_COPY_BYTES_2

/* Inner-loop iteration count. Deterministic on the functional model, so this
 * only needs to be large enough to amortise the two bracket reads; kept well
 * under the emulator's 10000-node revocation-tree pool (the borrow loop mints
 * one mrev node per iteration). */
#define BORROW_COST_ITERS 1024UL

/* DPI function codes -- the monitor passes the code as domain_main's second
 * (scalar) argument. Same values as the intra-domain probe. */
#define BORROW_COST_DPI_CALL 0x0u
#define BORROW_COST_DPI_REGION_SHARE 0x1u

/* Region annotations: hand the arena over LINEAR + RW with no monitor-retained
 * revocation handle, so the domain can mrev it. Same as the intra-domain probe;
 * see its header for why PERM_INOUT (not PERM_IN) is required for mrev. */
#define BORROW_COST_ANNOTATION_PERM_INOUT 0x1u
#define BORROW_COST_ANNOTATION_REV_TRANSFERRED 0x3u

/* Debug-counter slots the domain writes and csdebugcountprint dumps. */
#define BORROW_COST_SLOT_ITERS 0
#define BORROW_COST_SLOT_EMPTY 1
#define BORROW_COST_SLOT_RAW 2
#define BORROW_COST_SLOT_BORROW 3
#define BORROW_COST_SLOT_COPY 4
#define BORROW_COST_SLOT_COPY_BYTES 5
#define BORROW_COST_SLOT_COPY2 6
#define BORROW_COST_SLOT_COPY2_BYTES 7

/* domain_main success return value (measurement completed, no fault). */
#define BORROW_COST_RET_OK 0x22380000u

#endif
