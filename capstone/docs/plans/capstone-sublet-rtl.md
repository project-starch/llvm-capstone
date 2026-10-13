# capstone-sublet: the RTL side of the Sublet lifetime extension, and the FP-fixes bitstream

*RTL lane, 2026-10-13. Branches in capstone-ariane: `fp-fixes` (the next bitstream's candidates) and
`capstone-sublet` (the ISA extension, on top of fp-fixes). Hashes below are the lane branches' as pushed; the
landing on the shared branch squashes per logical change.*

## Why

The Sublet lifetime extension (llvm-capstone #216 / capstone-qemu #21, `docs/design/virtual-capstone/isa.md`
§"Sublet child lifetimes") is the part of the new processor model that an allocator needs on silicon and that does
not depend on the virtual profile's open questions (translation in C mode, the per-context table, exact bounds).
It is two instructions over the existing revocation-node list, a fourth permission bit, a parent link in the node
record, and two semantic alignments of existing instructions. The application ports and mallocng are migrating to
it (#220, #221, #222), so a bitstream carrying it is what lets the board run the migrated allocators.

Separately, three defects found on `caplifive_r51r52_0568f93a9` block every hard-float native process or shorten
capabilities, and go in the next bitstream regardless.

## Branch `fp-fixes` (off 0568f93a9, the resident bitstream's tree)

| Commit | What | Evidence |
|---|---|---|
| cc1159a51 | R-53 (was S-19): `commit_stage.sv` dirty_fp_state parenthesised as upstream CVA6 #2936; no FP instruction ever set mstatus.FS Dirty, so Linux skipped the FP save at every context switch | s19-fs-* arms read FS 2 -> 3 with SD 1 (Clean on the base tree); rv64ud 12/12 |
| ae21e121e | R-52 residual: `issue_read_operands.sv` third-operand forwarding compared the producer's rd against the CONSUMER's rd, so `fadd fa0,fa0,fa5` took fa5 from the in-flight fdiv that writes fa0 | r52-resid-a reads 0x406e0db8fc0a7d07 (= 240.42883112, the board lane's expected value); order B/D controls equal |
| 2a3a87dfc | R-54: `compress_bounds` takes E + 1 when the rounded-up span reaches 1024 granules, instead of wrapping to half the span | bounds-renorm reads 0x80faa000 covering (0x80f6a000 without); bounds-model.py 0 of 400,000 x 3 short (1,050 without); QEMU lockstep: capstone-qemu bounds-encoder-renormalise aad781e15acd |
| 4a4261bb4 | Option (b): the ENCODER stops using the cursorless scheme (bit 27 always 0 from the core; the decoder still reads cursorless words). Closes R-11's truncation at base; every at-base top now rounds OUTWARD as QEMU's | cursorless-off reads 0x80408000 / 5008 (0x80407ffc / 5003 on 2a3a87dfc, B4's H shape); lint UNUSEDSIGNAL 833, the one new name compress_cap's cap.cursor |

Gates on every commit: rtl-lint-gate.sh at baseline (per-commit lint for cc1159a51, ae21e121e, 8cda78a46 too),
the 95-test sweep identical to 0568f93a9 on 2a3a87dfc and 328f61761, the FP / bounds / sup arms as above.

**The lead's choice for the FP bitstream** (nothing is synthesised until the lead says so):

- **(a) = 2a3a87dfc**: R-53 + rs3 + renormalisation. R-11's truncation stays on the heap and outside the glue
  (B4's H reads [+0, +65532) at base when it straddles a 4 MiB boundary; covering once moved).
- **(b) = 4a4261bb4**: (a) plus the encoder change. No truncation anywhere; but where the cursorless scheme gave an
  EXACT top, a glue-carved global's capability now reaches into its upper neighbour by under a granule (the board
  lane's table: k800 64 B, s06agg 112 B, r1_slots_pools 2,560 / 496 B, live16 1,008 B, SQLite cells 4,096 B,
  b0-hello 384 B, b1-thread 2,048 B, memcached 32,768 B) and level0 objects of 4 KiB and up round outward.
  check-repr --renorm over the 11 acceptance images (the board lane, b0-silicon-delegated-runtime.md): under (b) no
  fault on any image; under (a) every verdict as on 0568f93a9, R-11's existing trap bases included (stale-full
  r1_slots_pools AT RISK at 7 of 255 bases, the SQLite cells cell5-memsys5-O2 and cell6-sublet-O2 at 14 of 253
  each), because the cursorless scheme truncates at base and the renormalisation does not reach it.
- r54-bounds as the before/after arm (the board lane's corrected prediction): 0568f93a9 39, (a) 0, (b) 64.
- Monitor side (the board lane owns): no change for the renormalisation (every monitor capability is exact under
  every scheme); under (b) the C-13 granule rule leaves the CODE region's top rounding into the seal region for
  code-heavy layouts (none of the 11 images), fix = size the granule from max(split_size's granule, repr_gran).
- QEMU side (the helper lane owns): aad781e15acd lands only with the flash that carries 2a3a87dfc.

## Branch `capstone-sublet` (on top of fp-fixes)

| Commit | What |
|---|---|
| 124d38406 | The load-side S-06 sim invariant moves into `load_unit.sv`, scoped to LDC returns. A plain scalar load's dcache user lane is the testbench's preloaded ELF row (`ariane_tb.sv` USER_MEM), not the granule's metadata; the 28-bit format decoded that row as type 4 and the check never fired, the 27-bit format decodes it as NOT_CAP and it fired on the format tree's first sweep. Positive control: the inverted check aborts linear-clear-audit on its first LDC. |
| d34c26002 | The capability word becomes `{revnode_id 30 \| perm 4 \| cap_type 3 \| bounds 27}`, QEMU's widths and order; MANAGE = perm bit 3; the cursorless type and the decoder's branch are gone; TIGHTEN takes four rights bits, CAPPERM accepts 15, LCC field 5 reads four bits; `CAP_PERM_MANAGE` / `CAP_PERM_ALL` in asm_insn.h. perm4-roundtrip reads 15 / 7 / 15 / 8 / 27 / 5008. |
| b330433fe | Node record gains `parent` (16-bit bare index, MSB field, packed into the b word's spare bits [46:31]); rev-node unit gains `derive_req` (init_req's shape: child after the parent at depth + 1, never linear, parent recorded; status 2 at the depth maximum) and `kill_req` (child live under its generation, a direct non-linear child of the parent and not the parent; invalidated and pushed to the free list, then KILL_NODE, REVOKE_NODE line for line, walks the junior run and splices child and run out: QEMU's cap_rev_retire); decoder funct7 0x51 CDERIVE rd,rs1,rs2 and 0x52 CREVOKE rs1,rs2 (rd must be x0), both allowed under supervision; dyn unit CDERIVE / CREVOKE with QEMU's check order; the two ops join every op table (check_cap_op, check_op_cap_dyn, the issue serialisation set, the tracer's group 5). Arm sublet-derive-kill (19 readings). |

| 52ad6e5ea | A non-linear SPLIT keeps its node (no sibling). Load-bearing with CDERIVE on this RTL: a sibling is inserted right after the parent at the parent's depth, so it would sit between a NONLIN parent and its depth + 1 children and stop the parent's revoke walk; it would also outlive CREVOKE of the half it was cut from. Arm sublet-split-shares-node (the escape reads 0 on the old tree, 25 on the new). |
| 11c405304 | CREVOKE joins R-45's commit-flush list (`commit_stage.sv`, the fix that already orders younger loads behind REVOKE and DROP, on silicon since 2026-09-29), so a load right after CREVOKE refetches and is denied. Found by the port of QEMU's cases: the `ld` right after CREVOKE executed ~400 cycles before the kill finished and committed in order one cycle after it. A first version was an issue-side barrier over every tree mutation, withdrawn once the board lane pointed at R-45; SPLIT/DELIN/MREV/CDERIVE end no node and stay unordered. QEMU's sublet-lifetimes cases 0-34 and 37 ported as sublet-cases.S (36 of 36, plus the wrong-PC negative control that must fail). |

| 464abe2ce | MREV deepens its whole subtree (deepen_req: a read-only overflow check before the handle, a write walk after) and a LINEAR split's sibling goes BEFORE the parent (QEMU's cap_rev_mrev and dup_before). The claim-auditor refuted the earlier "deepening is unreachable" argument: this RTL duplicates LINEAR capabilities (R-21: TIGHTEN/SHRINKTO return rs1 unchanged), so a node can carry a LINEAR (MREV) and a NONLIN duplicate (DELIN, CDERIVE); mrev twice then revoke of the inner handle orphaned the child. The helper lane found the same duplicate in QEMU (SHRINKTO; probe capstone/runtime/tests/sublet-lifetimes/mrev-over-derived), so QEMU exercises its deepening walk too. Arm sublet-mrev-orphan reads 0 (the orphan) on the step-4 tree and 25 here. |

Divergences recorded (helper lane's Q-15 and this doc): no representability check in CDERIVE (the RTL rounds outward
at compression; QEMU's compact profile refuses with 29; the migrated allocators run on QEMU's exact-bounds profile
and never see the refusal); TIGHTEN asking for a right the source lacks raises 29 here and in the academic spec, 26
in QEMU (sublet case 26); the physical QEMU profile raises 2 for CDERIVE/CREVOKE and must flip in the same step as
the bitstream that carries them.

Still to do on the branch, in order:

1. The testharness fix: `ariane_tb.cpp` prints SUCCESS when the cycle limit expires with no tohost write, so a hung
   or over-budget test exits 0 and the exit-code sweep cannot see a new hang. Its own commit: a time-limit exit that
   prints FAILED, with the three supervised arms that legitimately need more budget (sup-mswap-capstl ~4.6M cycles,
   sup-stcsb-ev-q5k ~5.3M, -q2k ~8M, the board lane's reading of the 2026-10-10 logs) given 6M / 6M / 9M so their full
   readings become their baseline. No sweep test reaches the limit on any tree gated this cycle.
2. Gates before synthesis: lint at baseline by name, the sweep identical, the sup list identical plus the new arms,
   claim-auditor on the kill walk's bound and the parent link's generation rule, then synthesis on the lead's word
   with the pre-registration pushed first.

## Lessons recorded this cycle (memory, not rules)

- A committed file must equal the tested file: 711d7b44b carried a half edit that does not parse while every gate
  had run on the working tree; after `commit -o`, `git diff HEAD -- <paths>` must show no hunk of the change.
- The scratch arm runner reports RC=1 for a test absent from its list exactly like a failure; "identical" across
  two trees from such arms is void. The sweep's list is the instrument for the bounds-regression tests.
- A sim-only check that fires after an ENCODING change: print its inputs on the old tree first.
