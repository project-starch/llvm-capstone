# Can the RTL be brought to the virtual Capstone profile? An audit, 2026-10-10

*RTL lane, on the lead's instruction. Read-only; nothing implemented. Sources: the virtual-Capstone guide at `dev`
(`docs/design/virtual-capstone/{README,isa,ownership,runtime,guarantees,development}.md`, 2026-10-07/08), the
transferable ISA amendment (`docs/plans/virtual-capstone-isa.patch` at `e03131496304`, 425 lines), capstone-qemu
`virtual-capstone` 1a6dd20732 (#13..#20: 77 files, +9,880 lines over `c128-qemu-merge`), the 2026-10-07 stack-audit note,
and the RTL at 0568f93a9 (an rtl-oracle read of seven questions, the three load-bearing quotes re-checked by hand). The
helper lane confirmed that no written owner of the model's semantics and no written intent for silicon exist; both are
the lead's to state. The guide itself says: "This is a QEMU implementation, not a released RTL interface."*

## The two profiles in RTL terms

| | Physical profile (the RTL today) | Virtual profile (QEMU `virtual-capstone`) |
|---|---|---|
| C mode | M-mode privilege + capability memory; sticky `capmode` | the same encoding plus a virtual-context flag |
| Capability addresses | physical; cursor is the paddr, no translation in C mode | VIRTUAL in the context's address space; translated under `satp` with USER PTE permissions, data and fetch |
| Lifetime table | one global pool, 65,536 records at a hardcoded physical base, records of 94 bits with a 14-bit generation | per address space, kernel RAM at CSR `srevroot`, 16-byte records {prev, next, depth, flags}, flat or paged 5:9:9:8, no generation, 31-bit ids |
| Enforcement for S/U | CPMP (16 physical entries) | none: applications never run in S/U; the virtual C context is the application |
| Entry and events | CALL/RETURN with a seal; cssupervise + escape; 3 event kinds; one armed context | `CSRUNV` with a 576/640/656-byte physical frame; actions 0..3; 5 event kinds incl. resumable page fault and node pressure; 32 saved contexts |
| Minting / retirement | CAPCREATE/CAPTYPE/CAPBOUND/CAPPERM in M mode; REVOKE walks | `CSMINT` from a descriptor, `CSRETIRE` of an ancestor (walk + unlink), trusted S/scalar-M only |
| ID recycling | hardware free list + generation bits (R-12 reclaimer) | software namespace sweep of every tagged page, then recycle; "no generation bits in this prototype" |
| Memory encoding | 128 bits: cursor 64 + {30-bit node, 3 perm, 3 type, 28 compressed bounds}; tag = 1 bit per granule in a shadow tag region | 128 bits with 27 bounds / 31 node; **plus**, per tagged granule, a 24-byte uncompressed {cursor, base, end} in the physical tag map, authoritative under the opt-in exact-bounds profile that ABI v5 / local mallocng REQUIRE |
| Fault causes | 24..30 on base 23 (R-24) | 24..30, same names |

## Verdict

**Partly, and not as a port.** Three of the seven areas already agree or are close (cause numbering, the capability
word's field split, the R+W rule for consuming loads; the supervised-CALL machinery is the same shape as `CSRUNV`
minus two event kinds). Two areas are structural rewrites of units that have each cost a bitstream cycle or more this
month: address translation of capability accesses in C mode (the RTL's C mode is M-mode, and M-mode bypasses the MMU
by the base core's translation-enable rule, with no capability term), and the lifetime table (a different data
structure, per context, in guest RAM, optionally paged, reached by a dependent two-level walk on every node access).
And one requirement cannot be met by any change inside the current capability format: the exact-bounds profile keeps
24 bytes of uncompressed bounds beside every tagged granule, and the software stack as landed (ABI v5, local mallocng,
all eight application ports) runs only under that profile. The RTL carries one tag bit per 16-byte granule and nothing
else; honouring that profile means either a wider in-memory capability (256-bit, i.e. two granules per pointer, with
every load/store/forwarding path, the write buffer and the tag region reworked) or a side metadata store the size of a
quarter of memory. That is a hardware architecture decision, not an RTL change, and it is the first thing the lead has
to rule on; the alternative is the one the physical profile already follows, representable bounds enforced by the
allocator (R-33's rule), which the virtual runtime explicitly chose not to do.

## Item by item, with the evidence

**A. Lifetime table — DIVERGENT, a rewrite of the rev-node unit's memory interface.** The RTL's node records live in
ordinary RAM reached through D-cache ports 2 and 3 (`cva6.sv:2306,2308`; `ex_stage.sv:1340-1355` builds plain
`dcache_req_i_t` requests), at a hardcoded base `CAP_REVNODE_MEM_BASE = 56'hBFF0_0000` (`ariane_pkg.sv:602`), indexed
by `index * 16` (`ex_stage.sv:1312-1313`); the record is 94 bits {free 1, generation 14, depth 17, prev 30, next 30,
valid 1, linear 1} (`capstone_unit.anvilh:602-610`), the pool 65,536 (`REVNODE_HEAD_BITS = 16`). No `srevroot`, no
per-context root, no directory. What the virtual profile needs: a CSR-selected base swapped per context (cheap: a
register in place of the constant, saved and restored by the switcher like `offsetmmu` is), the spec's 16-byte record
format (a re-layout of the Anvil unit's reads and writes; the depth-ordered list algorithm is the same one QEMU uses,
so MREV/SPLIT/REVOKE walks carry over), reserved ids 0..1 and a header record, and the paged format: two dependent
directory reads before every record read, which turns one D-cache access per node touch into three and lengthens the
REVOKE walk and every capability check's liveness probe accordingly. The R-12 reclaimer's generation bits disappear
(the profile recycles only after a software sweep), which also removes the generation compare from the R-35 cache and
the trackers. Cost: the rev-node Anvil unit and its ex_stage glue, roughly the size of the R-12 reclaimer work
(A1..A6, two weeks), plus the paged walk, plus a reflash cycle per iteration. Risk: the unit sits in the standing
loop list's cone (the 2026-08-21 arbiter lesson), so synthesis before board time at every step.

**B. Translation of capability accesses — DIVERGENT, structural.** The LSU's capability check is gated on
`capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M` (`load_store_unit.sv:1304-1306`); the base core enables translation only
when `priv_lvl_o != PRIV_LVL_M` (`csr_regfile.sv:3048-3051`, no capability term; the same `enable_translation_i` gates
instruction fetch in `cva6_mmu.sv:363-380`, and the MMU module has no capability awareness, zero hits for `capmode` or
`CAPSTONE` under `core/cva6_mmu/`). So in C mode the cursor IS the physical address today, and a page fault cannot
happen. `offsetmmu` (CSR 0x803) is stored, restored by the switcher and read by nothing (`csr_regfile.sv:697, 1690,
1976`; no consumer anywhere). The MMU itself is present (`capstone_cv64a6_imafdc_sv39_config_pkg.sv:78`, Sv39). What the
profile needs: a virtual-context flag that turns `en_translation_o`/`en_ld_st_translation_o` on under M privilege with
USER permission checking, for data and fetch; the capability bounds check staying on the virtual cursor (it already
reads `lsu_ctrl.vaddr`, `load_store_unit.sv:338,1339`); the LSU's exception path carrying page faults to commit as
resumable events; the consuming LDC's "same translated granule" rule (read and clear through one translation; the
clear is in `load_unit.sv:216-220`); and an audit of every `PRIV_LVL_M` assumption the bypass rule protects (PMP/CPMP,
MPRV, WFI, CSR privilege). Cost: a design change across csr_regfile, the MMU interface, load/store unit, load_unit,
frontend and commit, with a new fault class in the escape path; the biggest single item after the encoding. Risk: the
translation path adds a TLB lookup to every capability access that today bypasses it; on the FPGA the dcache/LSU paths
already carry the WNS.

**C. Capability word — MATCH, with one open check.** `cap_metadata_t` = {revnode_id 30, perm 3, cap_type 3, bounds 28}
beside a 64-bit cursor (`ariane_pkg.sv:639-659`), tag as bit 64 of the data-path word (`:773-775`) and one bit per
granule in the shadow tag region `CAP_TAG_MEM_BASE` (`:603-604`). The guide states the same numbers for the deployed
RTL and 27/31 for QEMU; the one-bit difference in the bounds and node fields is a format decision to settle before any
freeze, and whether the two compression functions are bit-exact for the shared widths is UNRESOLVED (nobody has
compared `compress_bounds` field by field with QEMU's `cap_compress.c`). This is R-33's class of problem: both sides
round, and the rounding must agree.

**C'. The exact-bounds profile — NOT POSSIBLE inside the current format.** QEMU's physical tag map holds, per 16-byte
granule, a tag bit AND a `capboundsfat_t {cursor, base, end}` (`cap_mem_map.h`: `capboundsfat_t bounds[4*64]` per 4 KiB
page, "we hack this to prevent precision loss for now"; `cap.h:58-62`); commit 24b1e95b1dac makes it authoritative under
`x-capstone-exact-bounds`, advertised in `scapctl` bit 8, and its own message says "this prototype uses extra shadow
metadata and does not specify an RTL encoding." The guide and the runtime docs say ABI v5 and local mallocng require it
and that "a future tag-bit-only implementation must not widen authority during a round trip." The RTL has no place for
24 bytes per granule. The options are the lead's: (i) a wider capability (two granules per capability: 128 bits of
cursor+metadata plus 128 bits of exact bounds; every capability load/store becomes a 32-byte access, the write buffer,
forwarding, the tag granule and the D-cache metadata lane change, and the runtime's pointer size doubles), (ii) a
side store of 24 bytes per capability-bearing granule in DRAM, written on every STC and read on every LDC (3x the
memory traffic of a capability access, a second tag-like shadow region, and a coherence problem the single-bit tag
does not have), or (iii) keep the tag-bit-only format and change the allocator contract to representable bounds (the
physical profile's R-33 rule), which the virtual stack rejected for mallocng's slot geometry. None is an RTL lane
decision.

**D. Entry, events and contexts — same shape, different instruction, two kinds missing.** The RTL has no
`CSRUNV`/`CSMINT`/`CSRETIRE` (zero hits). It has `CSSUPERVISE` (`capstone_dyn_unit.anvil:358-396`), the CSRs
`csupquantum/csupctl/csupstatus/csupcause/csupepc/csuptval` (`csr_regfile.sv:1029-1035`), a SAVE/RESTORE walk of
register ids 3..66 into a 1 KiB area (`capstone_dom_switcher.anvil`), an escape on any committed exception, CSR
violation or PC-capability fault while `sup_active` (`commit_stage.sv:264-270, 678-731`), three event kinds RETURN /
PREEMPT / FAULT (`ariane_pkg.sv:1256-1258`), and resume by re-arming: the saved PCC cursor is the faulting pc, so a
resume re-executes the instruction, which is what a resumable page fault needs. Missing: the page-fault kind (nothing
can raise one today, see B), the node-pressure kind and the collection action (the RTL recycles in hardware instead),
the service-ECALL kind with the reply-slot action 2 (today an ECALL is a FAULT the monitor decodes from `csupcause`),
the frame layout (the RTL parks in the seal plus a private save area; the spec's frame is a plain physical buffer with
initial register slots consumed on first entry), and 32 saved contexts (the RTL holds one armed context; more means
the save area becomes the context, which the spec's "frame format does not by itself specify a future RTL context
storage" leaves open). A `CSRUNV` front end over the existing switcher is a moderate change (new opcode, frame
addressing, two kinds, action codes); the saved-context semantics are already proven on silicon (B0..B3).

**E. Instruction policy — close, two gaps.** Under `sup_active` the decoder rejects SRET/MRET/WFI, CALL, nested
CSSUPERVISE, CAPENTER, and CAPCREATE/CAPTYPE/SPLIT/SEAL/MREV (`decoder.sv:232-289, 1138, 1289, 1314, 1318`); the CSR unit
denies CIH and CPMP0..15 through CCSRRW and every plain CSR with `addr[9:8] != 0` plus {0x800, 0x801, 0x802, 0x804,
0x810, 0x811} (`csr_regfile.sv:2879-2886`). Gaps against the spec: CCSRRW itself is not decode-gated and CTVEC/CEPC/
CSCRATCH stay legal by design (the runtime's context probe uses them), and 0x803 (`offsetmmu`) is absent from the
deny list. Both are one-line policy changes once the profile's list is adopted. In the other direction, the spec keeps
ordinary capability arithmetic and lifetime operations available to virtual code, so the RTL's v1 rejection of SPLIT and
MREV under supervision is STRICTER than the virtual policy and would have to be relaxed for mallocng to run.

**F. Causes — MATCH.** `capstone_unit.anvilh:314-328` enumerates UNEXPECTED_OPERAND 24 .. ILLEGAL_OPERAND_VALUE 29 and
INSUFFICIENT_SYSTEM_RESOURCES 30; the LSU and commit literals agree, and an assertion defends the range
(`csr_regfile.sv:3425`). The virtual chapter's table is the same.

**G. Consuming transfers — MATCH on authority, no PTE half.** LDC's clear needs a tagged linear-family capability and
the authorizing capability's write permission (`load_unit.sv:202-220`), and the load needs read permission
(`load_store_unit.sv:1335`); the "writable PTE / same translated granule" half has no counterpart until B exists.

**H. Namespace collection and tags.** The spec's sweep clears dead tags in every tagged page and in saved contexts
in software, with all contexts stopped; the RTL's tags are physical (shadow region) and clearable by ordinary stores,
so the sweep is implementable by the adapter without RTL help, except that the RTL's R-35 revocation cache and the
CPMP/PC trackers would need an explicit flush at collection (today their invalidation is broadcast-driven by REVOKE).
Dropping the generation bits also drops the (g+1, i) re-adopt guard the trackers rely on (`pmp_data_if.sv:82-107`).

## What the audit cannot decide, and the lead can

1. **The exact-bounds profile** (C'). The software stack requires it; the hardware cannot provide it in the current
   format. Either the format changes (a hardware architecture decision with a cost in every memory path and in
   pointer size) or the allocator contract changes back to representable bounds. Until this is decided, no RTL work
   on the virtual profile can be qualified against the landed runtime, because the runtime refuses to run without the
   feature bit (`CPONVVM5` requires `scapctl` bit 8; the helper lane measured that the launcher runs nothing without it).
2. **Which silicon profile is the target.** The guide and the spec patch both say the physical domain interface
   "remains a separate profile." If the board is to run the virtual profile, B (translation) and A (table) are the two
   multi-week items and both reach into the loop-bearing cones; if the board stays physical, the useful RTL work is
   the small convergences (E's policy list, the `CSRUNV` front end as an alias of the supervised CALL, a `srevroot`
   register in place of the constant) so that one runtime adapter can target both.
3. **The format freeze** (27 vs 28 bounds bits, 31 vs 30 node bits, generation or not). The guide says to reconcile
   before claiming equivalence; the RTL side's numbers are fixed in every bitstream since April, the QEMU side's can
   still move.

## If the lead chooses silicon on the virtual profile: order of work

1. Decide C' (format) first; nothing below is qualifiable without it. If (iii), the runtime's mallocng port changes,
   not the RTL.
2. B, translation in C mode: the virtual-context flag; `en_*translation` under M with user permissions; page faults
   as a resumable escape kind; the consuming-LDC translation rule; the `PRIV_LVL_M` assumption audit. One bitstream
   cycle to prove translation plus escape on silicon with a bare test, before any runtime.
3. A, the lifetime table: `srevroot` register, the 16-byte record layout, reserved ids, the header; the flat format
   first (the spec keeps it valid), paging second. The reclaimer's generation path is removed, the trackers' re-adopt
   guard replaced by the collection flush (H). One cycle each.
4. D, the `CSRUNV` front end: frame addressing, actions 0..3, the two missing kinds, `CSMINT`/`CSRETIRE` as DYN-unit
   ops over the new table. One cycle.
5. E, the policy list, with the SPLIT/MREV relaxation. Free, folded into 4.
Four to five bitstream cycles at this month's rate, each with its own synthesis risk in the LSU/dcache/rev-node cones,
and the format decision before the first.

## Questions for the lead

- Is silicon meant to run the virtual profile, or does the board stay the physical profile with QEMU carrying the
  virtual one?
- On exact bounds: widen the capability format, add a side store, or require representable bounds from the allocator?
- Who owns the model's semantics (which lane), so that the open encoding questions (27/28 bits, 31/30 ids,
  generations) get one answer?
