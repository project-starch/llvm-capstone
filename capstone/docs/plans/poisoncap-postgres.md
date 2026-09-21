# PoisonCap PostgreSQL integration

Work branch: `ports/14-poisoncap-postgres`. Based on the verified FFmpeg and pymalloc
PoisonCap integrations and current `dev`. Reuse the published compiler, QEMU, CheriBSD
image and platform controls; keep platform sources outside Git.

## Objective

Give the eight PostgreSQL memory-context defects a PoisonCap arm. They currently have
one pairing only — the port's `spatial` baseline against Capstone/Sublet — so the
four-arm temporal-safety matrix in [the comparison plan](temporal-security-comparison.md)
has no measured PoisonCap column for them. This lane produces that column, under
attacker class A-1 (buggy consumer): every case is an upstream defect, reduced, not an
adversary choosing a moment.

## Lifetime policy

The hosted backend implements the existing `pg_subpool` manager-hook ABI, so the
PostgreSQL managers (`aset.c`, `slab.c`, `mcxt.c`) are the same patched sources the
other arms use. What changes is underneath:

Metadata — pools, blocks, chunk entries and context headers — is static and lives
outside revocable storage. Backing blocks come from `posix_memalign` at representable
alignment and are retained until process exit in **both** modes, so libc free or libc
revocation can never explain a protected/unprotected difference. Each chunk is bounded
exactly at carve; unrepresentable geometry fails closed rather than widening into a
neighbour. Hand-out strips `CHERI_PERM_POISON` and `CHERI_PERM_SW_VMEM` from the client
copy, so a consumer can neither poison nor re-widen.

Retiring storage — chunk drop, block free, managed reset — poisons every 16-byte
granule, completes a synchronous kernel sweep, clears poison, and then zeros the
storage. The final zeroing is not redundant: clearing poison does not erase poison
capabilities still stored in those bytes, and a later sweep that finds them revokes
unrelated fresh storage. That failure was paid for once already in the pymalloc arm.

Mode 0 and mode 1 differ in exactly one respect: mode 0 skips invalidation. Layout,
allocation policy, block retention and the binary are otherwise identical.

## Predicted readings, written before the first run

Recorded here so that a wrong prediction stays visible, and so the run is not merely
confirmatory. The discriminator is not "did it fault" but "did it fault **at the probe**":
`supervise` resolves `pg_defect_probe` from the child's own map and the target ELF, and
prints both the expected address and the trap PC. A fault anywhere else is a different
event wearing the right exit status.

| # | upstream row | mode 0 | mode 1 | notes |
|---|---|---|---|---|
| 0 | tuplestore double `pfree` | **uncertain** | **uncertain** | the stale access *is* the second free, so there is no probe. The adapter's own `active[]` bookkeeping may refuse it (exit 1) in both modes, which would make this case a non-pair. |
| 1 | vacuum `dead_items` | complete | fault at probe | |
| 2 | TidStore 2024 instance | complete | fault at probe | context deleted before the struct is freed |
| 3 | `live_parts` alias | complete | fault at probe | |
| 4 | child `SpecialJoinInfo` | complete | fault at probe | |
| 5 | WindowAgg partition reset | complete | fault at probe | reset, not free |
| 6 | pgoutput grandchild teardown | complete | fault at probe | no fresh allocation afterwards |
| 7 | reorderbuffer slab reuse | complete | fault at probe | asserts deterministic same-address reuse |

**H-REUSE, the hypothesis this run is actually worth spending a boot on.** The manager's
own `chunks[i].slot` points into the storage that `invalidate()` poisons, and the sweep
revokes capabilities pointing at poisoned memory wherever it finds them — including a
static table. `pg_subpool_hand` returns a permission-stripped copy of that stored slot.
If the sweep reaches it, then in mode 1 **chunk reuse through the manager's free list
hands out an untagged capability**, and cases 1, 3, 4, 5 and 7 fault at their
`fresh[0] = N` write rather than at the probe. Cases 2 and 6 tear down a whole context
and would be unaffected.

That reading and the intended one produce the **same exit status**, 162. Only the
PC comparison separates them, which is why the runner treats a missing or mismatched
`SUPERVISE expect` line as a failure and not as a detail. If H-REUSE holds, the finding
is a defect in the adapter, not in the eight defects.

## Acceptance gates

1. Platform controls pass: purecap ABI with libc revocation off, and an exact
   out-of-bounds read trapping as SIGPROT (exit 162). The second is the positive
   control for fault detection itself — without it a clean mode-0 arm proves nothing.
2. The four direct-link manager examples pass unchanged under the PoisonCap backend.
3. All sixteen defect arms RUN in one boot. The suite continues past a failing oracle,
   because "every other arm would have rejected this too" is information a suite that
   stops at the first failure cannot produce.
4. Every mode-1 fault is matched against `pg_defect_probe`, and the pairing is only
   claimed for cases whose mode-0 arm completed at the same access site.
5. The replay driver processes the same trace in both modes, with sweep, poison, clear
   and zero counts recorded.

## Why this adapter deviates from the paper, and why each deviation is forced

PoisonCap is arXiv:2605.13210, "Efficient Hierarchical Temporal Safety for
CHERI". Its flow is: poison on free, which alone prevents a dangling access; the
freed allocation is quarantined; a revocation sweep runs when quarantined
memory crosses a threshold, and MUST run before the storage is reallocated;
detox by the allocator at reallocation; and no zeroing at all --
poisoning replaces the zero pass, which is the paper's claim to costing nothing
over a zero-before-reuse baseline. Delegation to a nested allocator is by
BOUNDS: an access to poisoned memory is permitted only through a pointer
broader than the poison capability it references, so a sub-allocator's heap
capability survives while its issued chunks do not.

This adapter follows none of those three timings, and a mode built to follow
them literally was measured on 2026-09-20: it paired 2 of 8 where the shipped
policy pairs 8 of 8. The three deviations are consequences of this allocator,
not caution:

1. **Detox at free, not at reallocation.** PostgreSQL's `aset` keeps its
   size-class free list INSIDE the freed chunk. Leave the poison in place and
   the manager traps on its own bookkeeping at the next `pfree`/`palloc` --
   which is what happened to cases 1, 2, 3, 4 and 7, each faulting in the
   reallocation path before its stale access was even reached. The pymalloc
   port states the same requirement in one line.
2. **Sweep at free, because there is no quarantine.** The paper does not
   sweep per free: poison traps a dangling access on its own, and the sweep is
   threshold-driven, triggered by the allocator once quarantined memory
   reaches a fraction of the heap. What the paper does require is that the
   pass "must occur before safe reallocation", because it is the pass that
   invalidates a capability whose bounds match the poison. This adapter has no
   quarantine layer -- `aset` owns the free list and reissues a chunk
   immediately -- so "before reallocation" collapses onto "at free". Case 5
   measured what happens without it: `poison_bytes=80 detox_bytes=80
   sweeps=0`, the same storage detoxed for its new owner, and the stale read
   then succeeded with no fault. The deviation is the missing quarantine, not
   the sweep itself.
3. **Zeroing on top of poisoning.** `cclearpoison` clears the access state but
   does not erase the poison capability the instruction stored in memory. Left
   there, a later sweep finds it and revokes a fresh, unrelated allocation --
   paid for once already in the pymalloc arm.

**Consequence for cost.** The counters this adapter reports are therefore an
upper bound on a policy the paper does not propose: a sweep per free plus a
zero pass the paper removes. They must never be quoted as PoisonCap overhead.
Measuring that would need a quarantine layer and an allocator whose free list
is not in-band, which is a different experiment.

## Scope

A trusted, serial adapter. This is not whole-backend protection, not isolation of
hostile nested managers (class A-3), and the synchronous per-free sweep is a
deliberately conservative policy, not a lower bound for PoisonCap. QEMU elapsed time
is not hardware cost.
