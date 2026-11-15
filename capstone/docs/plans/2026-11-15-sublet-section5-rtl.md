# Sublet paper Section 5 on silicon: what the RTL lacks, and a phased plan

Status: proposal, 2026-11-15. Nothing is implemented. The scope and the order are decisions for the project lead,
together with the external collaborator.

## Why

The external collaborator asked for the RTL to implement Section 5 of the Sublet paper (`paper-nested-allocators`,
`sections/05-implementation.tex`). Two parts of their stack are ready:
- **QEMU:** done, on capstone-qemu `origin/address-space-root`, commits 4bfbf02fbe..9289114e06, 13-14 Nov.
- **Kernel:** reported ready, but not on any remote we read.

The expectation is that the paged lifetime table will cost some performance, and that silicon numbers on it are the
real numbers for the paper. Today's silicon (bitstream F, capstone-ariane a948e4134) uses a flat table.

## Gap, row by row

The source is a read-only gap analysis of F (a948e4134) against Section 5 and the QEMU branch. The key file:line
quotes were re-read before writing this.

| # | Section 5 element | F today | Change |
|---|---|---|---|
| 1 | Capability format: address 38 bits zero-extended, slot 31, generation 32; no type, rights or linear/sealed | 55-bit id (slot 25, generation 30; `ariane_pkg.sv:211-213`); 39-bit sign-extended cursor; perm and cap_type in the metadata | large: every id lane widens 55 to 63 bits, and the compress/decompress functions change |
| 2 | 128-bit record: generation 32, previous 31, next 31, depth 31; no parent, VALID, LINEAR or FREE flag | 137-bit `rev_node_t` with parent, valid, free and linear | large, with two semantic flips: (A) the generation advances at release, not at allocation; (B) the free list links through `next`, not `prev` |
| 3 | Paged table: 4 KiB pages, root, two directory levels and record pages, walk 5:9:9:8, physical pointers | flat table only (`ex_stage.sv:1905-1906`: `srevroot + slot*16`); a PAGED header is refused (`capstone_rev_node.anvil:301`) | large: a new walker of 3 dependent reads with validity and reachability faults (29) |
| 4 | `srevroot` with flush | exists (CSR 0x5C1); a base change flushes the verdict cache and the trackers | small, plus a wait for dirty record-cache lines (with row 5) |
| 5 | Record cache: 2-way, 8 KiB, 64 B lines of 4 records, LRU, write-back | none: records go through the shared write-through L1D. The LSU has a 64x4 verdict cache (live bit only, round-robin replacement) | large: a new block on the access path's dominant slack term |
| 6 | Record TLB, 16 entries, record-page addresses | none | new; small in structure, large in integration |
| 7 | CDERIVE: free head, then the never-used slot, then cause 30 for the kernel to grow the table | a hardware free list and bump allocator already exist in both arms; cause 30 on exhaustion. There is no grow path: cause 30 resumes a supervised context. A 256-slot reserve applies | medium: drop type/MANAGE/parent, add the no-fault failure (an untagged 0) and NOGROW, define who answers cause 30 |
| 8 | CREVOKE: one pass over the preorder segment, operand included | the segment walk and the two-write splice exist. It is a two-operand instruction (parent + child) and does not release its operand. Retirement is at 2^30-1 | medium-large |
| 9 | CRENEW | absent; its funct7 (0x54) is CSNODE in F's decoder | large: new instruction and walk variant |
| 10 | Access ordering and replay | the verdict-cache miss replay (HPM event 23) and per-record invalidation broadcasts exist | medium if the record cache sits behind the same replay path, large otherwise |
| 11 | Deleted state and instructions (QEMU): types, rights, SEAL, DELIN, CSRUNV, supervisor and others | all present; the supervisor and context machinery spans the commit, CSR, switcher and PMP paths | large in effort, mostly deletion; the biggest risk |

## Synthesis risk (each needs a synthesis-only gate with per-module LUT deltas)

1. **The access-path lifetime compare** (`load_store_unit.sv` verdict cache, "the dominant slack term"). A record-cache
   hit test there is the worst case. Earlier variants cost WNS -24.5 ns and +22.7 k LUTs.
2. **The rev unit's memory write-request selector** (the R-35 -27.7 ns path). The walker, the cache fill/evict
   engine and CRENEW each add send sites to it.
3. **compress/decompress of the format.** A trial once took technology mapping to 94 GB on two round trips.
4. **The load-data cone at `rd_ctag`** (on the standing loop list), for the zero-extend.
5. **`csr_regfile`** (several loop-list stems). The `srevroot` wait must be a registered stall.

## Proposed phases

The phases are ordered so that the collaborator's question ("what does paging cost?") is answered first, and
cheaply.

- **Phase 1: the paged table, with dynamic growth** (approved by the lead 2026-11-15).
  - The format and instructions stay as in F.
  - Add a walker (5:9:9:8) and a 16-entry record TLB in front of the existing record reads and writes.
  - **Growth, revised the same day:** the collaborator asked for the table to grow dynamically, as the paper
    says. The first scope built the tree up front, and that did not match. The table starts with one batch, and
    cause 30 makes the kernel add a batch, following the collaborator's own protocol: llvm-capstone
    `native-kernel-new-qemu`, `capstone/kernel/patches/0011-riscv-mm-sublet-table.patch`, "The growth protocol".
  - The current SPEC and nested stacks run on it unchanged. The difference against F is the paging cost.
  - Estimate: several days for the rtl lane, plus a synthesis-only gate and a full build.
- **Phase 2: the record cache** (2-way, 8 KiB, 64 B lines, write-back). It is measured against Phase 1. The design
  question is whether it replaces the verdict cache or sits behind it.
- **Phase 3: format and record semantics.**
  - 63-bit ids, the 128-bit record, generation at release, the `next` link;
  - single-operand CREVOKE, CRENEW, hardware CSMINT;
  - the no-fault CDERIVE failure and cause 30 growth.
  - This needs the collaborator's kernel.
- **Phase 4: the deletions** (types, rights, CSRUNV, supervisor). The SPEC native route runs on CSRUNV, so it is
  rebuilt on the collaborator's kernel before any table is re-measured.

## The collaborator's kernel needs more than Phase 1

The collaborator's stack is the llvm-capstone branch `native-kernel-new-qemu`: Linux `riscv-sublet-*` patches,
`capstone_vm.c` removed, and QEMU at `35de150ce` (CSRUNV and the supervisor deleted). Its table code checks at boot
that the platform declares the **63-bit Sublet format** (`sublet_ready()`). It also enters protected processes by
the trap route, not CSRUNV.

So running and measuring that kernel on silicon needs Phases 3 and 4 (format, record semantics, deletions, trap
route), not only the table. Phase 1 measures the paging and growth cost on today's stack, as a proxy for that.

## Open decisions

- Scope: Section 5 only, or the whole QEMU branch, which also has trap-route / protected U and tag retention in
  sscratch/mscratch.
- What M-mode uses for a table after the change (QEMU keeps a host tree for M and for S with no table).
- Whether the record cache replaces the verdict cache.
- The source of the kernel: it is needed from Phase 3, and for Phase 1's tree layout if it is not in capstone_vm.
- The baseline: the collaborator also asked for "a good baseline". Today's riscv-base has I$ 16 KiB 4-way and DTLB 16
  against Capstone's 32 KiB 8-way and 64. A riscv-base bitstream with the same I$ and DTLB needs one synthesis and
  about 17 board-hours for the 8 rows.
