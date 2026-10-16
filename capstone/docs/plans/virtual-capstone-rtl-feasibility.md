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

---

# Addendum, 2026-10-16: re-baseline after the lead's decision that silicon runs the full virtual profile

*RTL lane. The lead decided on 2026-10-16 (`docs/plans/2026-10-16-virtual-profile-on-silicon.md`, 44d82a9eb3f3) that
silicon runs the FULL virtual profile, one bitstream with both profiles under a per-context flag, the physical profile
kept. This addendum re-reads the audit above against what moved since it was written. Reference pinned here:
capstone-qemu `virtual-capstone` at **d35af47b163c**, verified on 2026-10-16 to be the branch head (`git log -1
origin/virtual-capstone`), which is also llvm-capstone dev's submodule pin. RTL side: caplifive_sublet_d93a8d66b (the
resident bitstream), tree capstone-sublet b7e3287e4. Four read-only strands fed this section (QEMU merges, the
CDERIVE/CREVOKE divergence, the capability data path, the M-privilege assumptions); every claim acted on here was
re-read at the quoted line.*

## 0. What moved on the RTL side (the audit above read 0568f93a9)

- **The capability word is now {revnode_id 30 | perm 4 | cap_type 3 | bounds 27}** (d34c26002), not the {30 | 3 | 3 | 28}
  the audit's item C describes. The fourth permission bit is MANAGE (Sublet). QEMU virtual's word is 27 bounds / 31
  node (guide); the two formats now agree on the bounds width and differ on how the 31st bit is spent: the RTL's
  MANAGE against QEMU's node bit. Re-read on both sides on 2026-10-16: QEMU's word at the pin is bounds [26:0] (bE 0-2, b 3-13, tE 14-16, t 17-25, iE 26),
  type [29:27], perms [33:30] with MANAGE = 8, node [63:34] (`cap_compress.c:30-37`, `cap.h:13-24`), and the RTL's
  packed `cap_metadata_t` is {revnode_id 30 | perm 4 {MANAGE, R, W, X} | cap_type 3 | bounds 27 {iE, T 12, B 14}}
  (`ariane_pkg.sv:638-642`): **the two in-memory words have the same layout bit for bit.** Item C's open check is
  therefore the two compression FUNCTIONS, not the field split, and they are known to differ in one place (Q-16, the
  renormalisation). What the 30-bit node field MEANS still differs: the RTL packs {generation 14 | index 16} into it
  (`capstone_rev_node.anvil:346`, `generation = nid[16+:14]`), QEMU uses it as a flat 2^30 index with no generation
  (`cap_rev_tree.h:29`); that is item A's question, settled by the freeze proposal in the exact-bounds document.
- **The encoder renormalises and has no cursorless scheme** (2a3a87dfc, 4a4261bb4): every capability rounds OUTWARD,
  R-11 cannot occur, R-54 cannot occur. The pinned QEMU does not renormalise (Q-16, filed 2026-10-16); under exact bounds
  the stored bounds bypass the encoder, so that gap bites the physical profile only.
- **CDERIVE and CREVOKE exist on silicon** (b330433fe..d93a8d66b): the node record carries a parent link, CREVOKE's kill
  walk is REVOKE_NODE in kill mode, MREV deepens the subtree. Item A's "the depth-ordered list algorithm is the same one
  QEMU uses" now extends to derive/kill; the differences are in section 2.
- **The exception encoder reads the spec's 24..29** (R-24, since 054cea69b): item F's MATCH is now a silicon fact.

## 1. QEMU and the runtime since 1a6dd20732 (item by item)

Reference: capstone-qemu 1a6dd20732..d35af47b163c (nine merges and acbad0f7ec), the guide at 3fe4541c1299..origin/dev,
and the dev runtime commits b02ddf9043cc, cbbc2ca958af, 6576dca09c37, 601c8d52c260, 7901153f10fa. Agent-sourced
(general-purpose, read-only), the word layout and the CDERIVE/DROP semantics re-read here; file:line at d35af47b163c.

| merge / commit | what changed (ISA-visible) | audit item, direction |
|---|---|---|
| #21 7dbfdd534e Sublet lifetimes | CDERIVE 0x5b/0x51 and CREVOKE 0x52 (rd = x0) in the virtual profile; child = RWX of parent + MANAGE, NONLIN, depth + 1, inserted after the parent; checks 24, 26, 25, 29, compact-profile representability 29 (`op_helper.c:1212-1263`); the word gains the MANAGE bit and the node shrinks to 30 (`cap_compress.c:30-37`); record word1 = depth[29:0] / parent[59:30] / flags[62:60], NODPAGE2, 4:9:9:8 directory, 16 root pointers (`cap_rev_table_abi.h:8-30`); MREV deepens the subtree and records the parent; NONLIN SPLIT allocates no sibling (`:1430-1475`); **virtual-only stale-reload rule**: a reloaded revoked NONLIN keeps its tag, use faults 25 (`:2144-2149`) | A further on format (new record layout, paged directory), closer on algorithm (the RTL has the same tree rules since 464abe2ce); C closer (30-bit node, MANAGE: now identical); C' new requirement; E two new unprivileged ops, the SPLIT/MREV relaxation no longer needed by mallocng |
| #22 9c7218cd52 | PCC checked once per translation block; no ISA effect (blocks end before a PCC-refusing instruction); the guide's "one instruction per block" (isa.md:242-244) is stale | none |
| #23 4497dba0b2 | tag clearing moved to the TLB slow path; one semantic correction: a FAILED SC no longer stores, so the granule keeps its tag (`trans_rva.c.inc`) | G edge; UNRESOLVED whether the RTL's failed SC clears the tag |
| #25 9306a3d182 | untagged CINCOFFSET / CINCOFFSETIMM / SCC compute an untagged integer result in the virtual profile instead of raising 24 (`op_helper.c:817-930`); both profiles: a tagged rs2 raises 24, UNINIT/SEALED rs1 raises 26 | **E new divergence**: the RTL raises 24 (`capstone_flu_unit.anvil`); the virtual SDK's SPEC programs depend on the CHERI rule; the lead decided it 2026-10-14 |
| #26 70e98dd390 | LDC/STC single helpers, clz encoder, lockless tag map; no ISA effect; LDC of a tagged granule takes its bounds from the tag map `:2133-2135` | none; see C' |
| #27 d35af47b16 MOVC | virtual: MOVC nulls rs1 only if tagged non-NONLIN; physical: an untagged source is nulled as the RTL does (`:660-690`) | E: Q-04 option 1; the RTL half Q-04 (b) has no commit (C-32) |
| #28 ed3e1b1de4 | every capability helper writes rd through one path; rd = x0 discards (`capstone_helper.h:85-92`) | fidelity toward the RTL; none |
| #29 b8e73e0986 | own MMU index for virtual C, decoded as single-stage USER (`internals.h:49`); SFENCE.VMA by address | B: confirms USER-permission translation of virtual-C data and fetch (`index8.S`) |
| #30 2b6fbe3eaf | F/D in virtual C: CSRUNV starts FS Initial with zeroed f registers; per-context save carries fpr, frm, fp_status (`capstone_supervisor.c:31-54, 916-930`); FP loads and stores go through the capability path with causes 24-28; CSRUNV refuses V = 1 | **D new storage requirement**: 32 fpr + fcsr per context; the RTL's switcher saves no FP state |
| acbad0f7ec | CDERIVE/CREVOKE decoded on the physical profile too; CREVOKE rd != x0 stays illegal | lands the physical half the Sublet bitstream needed |
| guide (5 files, +186/-35) | isa.md: the Sublet section, MANAGE = 8, the compact alignment rule (L >= 4 KiB needs base and end aligned to 2^(msb(L)-9)), record word1, 4:9:9:8, "27 bounds, 3 type, 4 permission, 30-bit node", the stale-reload rule with Q-11 named; ownership.md: mallocng delinearizes the grant once then CDERIVEs; guarantees.md: disjointness and scrubbing are allocator duties; unchanged: scapctl bit 0 / read-only bit 8, CSRUNV actions 0..3, events 0..4, frame 576/640/656, the policy list, no generations; stale against d35: isa.md:43-45 (physical decode) and :242-244 | C, E, H as above |
| dev 6576dca09c37 | `heap.c`: the grant is `ldc; delin` once; one CDERIVE per group over the mapping; slot parents are SHRINK copies at base + 16 + i*stride (no node); **every public object is a CDERIVE at the REQUESTED size** (`heap.c:167-178`), freed by CREVOKE; nested groups at the parent's key | C': the dependence is on arbitrary (key, size) children reaching hardware unrounded |
| dev b02ddf9043cc | `capability.h:36-64`: `capstone_cap_derive` probes the parent with three LCCs then issues the CDERIVE with cursor = base + offset; `capstone_cap_revoke_child`; `without_manage` (4-bit TIGHTEN) called only by the wrapper tests | E, C' |
| dev 7901153f10fa | the representable-bounds heap deleted (sublet_heap.c, 472 lines): the audit's option (iii) has no living implementation | C' |
| dev 601c8d52c260, cbbc2ca958af | qualification under `x-capstone-exact-bounds=true`; linear operand refused intact | none |

Where the exact-bounds requirement actually sits (file:line at origin/dev): `runtime/virtual/exec.c:697-698` (a v5 image
refuses to start without scapctl bit 8), `module/capstone_vm.c:452, 548, 1089-1091` (grants must be power-of-two
aligned without it; mremap needs it), `heap.c:167-178, 105-111, 62-70` (objects at requested sizes, nested groups at
arbitrary keys, SHRINK slot parents that QEMU never round-trips), every virtual gate's CPU string (`run.py:130`,
`run-malloc.py:79`, `cli.py:319-323`). In QEMU the flag only advertises bit 8 and lifts two refusals; **LDC takes the
shadow bounds unconditionally** (`op_helper.c:2135`), so even QEMU's compact profile is not a tag-bit-only model. The
per-workload count of mallocng objects whose bounds are unrepresentable in 27 bits is UNRESOLVED (Q-16's 43,591 is codec
divergence on random triples, not an allocator census); the guide's alignment rule says every object >= 4 KiB whose key
or size is not aligned to 2^(msb(L)-9) is one.

## 2. Where d93a8d66b's CDERIVE/CREVOKE differ from the virtual profile's

Agent-sourced (rtl-oracle), the two divergences re-read here at the quoted lines. Reference: the virtual profile under the
CPU string the runtime actually uses (`run.py:130`: `x-capstone-u-mode=true,x-capstone-exact-bounds=true`).

| aspect | silicon d93a8d66b | QEMU virtual d35af47b163c | verdict |
|---|---|---|---|
| CDERIVE check order | **24, 26, 29, 25** (`capstone_dyn_unit.anvil:246-259`: the size/window test at :251 runs before the liveness query at :257) while the unit's own comment at :238 claims "checks in QEMU's order ... 25, 29" | 24, 26, **25, 29** (`op_helper.c:1221-1235`) | **DIVERGENT, new (Q-17)**: a dead parent AND a window outside it reads 29 on silicon, 25 on QEMU; reachable from `capstone_cap_derive`, whose three LCC probes check no liveness on either side (RTL `:376` queries only for zimm 0; QEMU `helper_cslcc` has no `cap_rev_live`). Q-15's "fault order: agrees" is wrong for this pair. Fix: move the query before the size test, one Anvil reorder, next bitstream |
| child bounds | compressed, rounded OUTWARD unconditionally (`ariane_pkg.sv` compress_bounds: B truncated, T rounded up), no check that the rounded child fits the parent (`:238` "No representability check") | exact under the runtime's flag (the representability refusal at `:1243-1247` is skipped); refused 29 under the compact profile | **DIVERGENT, the C' requirement**: silicon can hand back a child wider than requested, up to a granule, invisible to the QEMU run the port is qualified against; bounds-model.py `cderive_child` predicts the slack (exact below ~4 KiB at a 16-byte key, tens to hundreds of bytes above) |
| child permissions, type, cursor | RWX of parent + MANAGE (`:264`), NONLIN, cursor = parent cursor = child start (`:265`) | `(perms & RWX) | MANAGE` (`:1238`), NONLIN, cursor = base | MATCH |
| node record | inserted after the parent at depth + 1 (`capstone_rev_node.anvil:346`); generation 14 bits in the id; depth 17 bits (max 131071 -> 30) | inserted after the parent at depth + 1 (`cap_rev_tree.c:143-146`); no generation; depth max 2^30 - 1 | insertion MATCH; resource model differs (A) |
| CREVOKE | 24, 26, 25 (parent), 25 (child), 27 (MANAGE on rs1), then 29 (not a direct non-linear child, or self) / 25 (stale between query and kill); kill walk retires the child's subtree (`capstone_rev_node.anvil:368-407`); rd != x0 illegal (`decoder.sv:1278-1282`) | 24, 26, 25 (either dead), 27, 29 (`op_helper.c:1264-1289`, `cap_rev_tree.c:162-176`); subtree retired; rd != x0 illegal | MATCH, including order |
| stale reload (LDC of a capability on a dead node) | tag kept for every type; the fault is at use, 25 (`load_unit.sv:217-220` clears only the linear family on a consuming load) | virtual + NONLIN: tag kept, use faults 25 (`:2144-2149`); physical or non-NONLIN: tag cleared, use faults 24 | MATCH for the virtual profile's NONLIN children; the physical-profile difference is Q-11 |
| R-45 flush | REVOKE, DROP, CREVOKE flush at commit (`commit_stage.sv:716-720`) | sequential | same observable result |
| DROP of a NONLIN with children | `drop_req` clears the node's `valid` globally without freeing it (`capstone_rev_node.anvil:217-233, 649-651`): every alias dies, the slot leaks, children become unendable (audit F-B) | `helper_csdrop` (`op_helper.c:1127-1163`) nulls the register only, touches no table: other aliases and children live on | **DIVERGENT, new (Q-18)**: not reached by `heap.c` (no DROP emitted; `sublet_clear` is an STC of zero); UNRESOLVED whether the compiler ever emits DROP implicitly |
| CSMINT / CSRETIRE | absent by design (the physical arms debug-mint) | the table bootstrap of the virtual profile | no RTL counterpart; item D |

## 3. Item B re-read: MPRV is not the mechanism

The audit said B needs "a virtual-context flag that turns `en_translation_o`/`en_ld_st_translation_o` on under M
privilege with USER permission checking". The RTL already has a data-side mechanism that does half of this, and it is
wired into the wrong place:
- `csr_regfile.sv:2385,2399`: `ld_st_priv_lvl_o = (mprv) ? mstatus_q.mpp : priv_lvl_o`. MPRV with mpp = U gives
  translated data accesses with USER PTE permissions while the privilege stays M.
- The same signal gates the ENTIRE capability check: `load_store_unit.sv:1304-1305` requires
  `ld_st_priv_lvl_i == PRIV_LVL_M` for the bounds/tag/revocation check (causes 24-28), and `pmp_data_if.sv:299` turns
  the CPMP bypass OFF when it is not M. So MPRV + mpp = U buys data translation and silently disarms the capability
  check on exactly that data.
- Fetch has no MPRV path at all: `cva6_mmu.sv:77` has `flag_mprv_i` commented out, and `csr_regfile.sv:3048-3051`
  forces `en_translation_o = 0` whenever the privilege is M, independent of satp.
Consequence for the design: the virtual-context flag must (a) enable data AND fetch translation under M with user PTE
permissions as its own term in `en_translation_o` / `en_ld_st_translation_o`, and (b) the capability check's gate must
be re-keyed on capability mode (`capmode_i`) and not on the privilege signal, so it stays armed with translation on.
Two hazards that are unreachable today and become live the moment translation turns on:
- **the consuming LDC's deferred clear re-uses the read's physical address** (`load_unit.sv:490-496`, `cap_clear_addr_q`)
  with no re-translation and no writable-PTE check; QEMU's `capstone_ldc_granule` (op_helper.c:2165-2227) re-probes the
  store side, compares physical addresses and fails closed with a store access fault, and checks the PTE's W bit;
- **the escape machinery has no terminal-versus-resumable kind**: `esc_kind` carries RETURN/PREEMPT/FAULT and every
  escape saves and restores; QEMU gives a page fault its own resumable kind (4) distinct from the terminal synchronous
  fault (kind 2). This is item D's gap and B's page-fault escape depends on it.
What the audit got right and stands: the bounds check is on the virtual cursor (`load_store_unit.sv:336-339,1339-1342`),
CPMP compares physical addresses (`pmp_data_if.sv:127-128,149-151`), the rev-node table is physical by construction
(`ex_stage.sv:1375`, D-cache ports 2/3, no MMU), `offsetmmu` 0x803 has no consumer, and nothing in the boot ROM or the
test corpus writes `satp`, so a bare test under the flag needs page tables nothing currently builds (the first B test
builds them). The Anvil units contain no privilege or capmode term; every such decision is in the surrounding RTL.

## 4. The capability data path today (what the exact-bounds costing is done against)

- Two parallel register files: the 64-bit data (cursor) file and a 65-bit {tag, metadata} file
  (`issue_read_operands.sv:1748`, `i_ariane_cap_metadata_regfile_fpga`); every scoreboard entry carries a 65-bit
  capability lane beside the 64-bit result (`cva6.sv:111-134`).
- The LSU checks DECODED bounds: `decompress_cap_tagged` at `load_store_unit.sv:336` feeds the compare at `:1339-1340`.
  Exact bounds remove that decode from the access path.
- The AXI user sideband is OFF (`DataUserEn = 0`, config pkg :42); a capability's metadata travels as a SECOND 64-bit
  data beat (`wt_axi_adapter.sv:217-221`, `axi_wr_data[1] = is_cap_req ? user : 0`, a two-beat burst). The line is one
  granule (`DcacheLineWidth = 128`), the cache is write-through.
- The tag bit is cached in L1 (`cap_tag_q`, `wt_dcache_mem.sv:146`); an L1 hit costs no AXI traffic for the tag. Every
  access that reaches AXI, capability or not, also issues a serialised one-byte TAG_WR/TAG_RD transaction to the shadow
  tag region (`wt_axi_adapter.sv:156-157, 445-480`).
- Traffic today: STC = 2 data beats + 1 tag transaction on EVERY store (write-through); LDC miss = 2 + 1; LDC hit = 0.
- CPMP holds 16 compressed entries (128 bits each, `pmp_data_if.sv:44`) and decompresses all of them every cycle in
  `cpmp_check` (`:125-130`); the switcher's save slot is 129 bits logical (`capstone_dom_switcher.anvil:21-24`).
- compress/decompress sites in core/: 6 compress, 9 decompress (the "three compress_cap instances" of the synthesis
  reports are the three on the worst timing paths, not a count).
- 8 of the 40 standing UNOPTFLAT loops sit in load_store_unit / load_unit / wt_dcache_ctrl / wt_dcache_wbuffer /
  wt_dcache, one of them `rd_ctag` (`wt_dcache.sv:144`), the tag ring itself. Any memory-format change enters those cones.

## 5. The acceptance set (from the runtime lane, 2026-10-16; the plan carries the authoritative copy)

Tier 0, the directed suites to port into the RTL harness, all at d35af47b163c: tests/virtual-capstone-model (model.py,
8 directed traces per storage backend), tests/sublet-lifetimes (119 at acbad0f7ec), tests/virtual-capstone-m1 (60),
tests/virtual-capstone-runtime (69) plus exact-bounds.sh (4), tests/capstone-x0-writes (24), tests/capstone-untagged-arith,
and dev's runtime/tests/sublet-lifetimes. Re-measured on 2026-10-16 on a fresh build of the pin (serial, one QEMU at a time): sublet-lifetimes 119/119,
virtual-capstone-m1 69/69 (development.md says 60), virtual-capstone-runtime 69/69, exact-bounds.sh 4/4,
capstone-x0-writes 24/24, capstone-untagged-arith 6/6, model.py's lines as development.md:30-31 print them, and dev's
runtime/tests/sublet-lifetimes 20/20 (development.md says eight; this one ran against the pre-existing 2026-10-10 SDK
under /tmp/capstone/virt-stack, not a fresh build, so it is a non-canonical pass). No failures. On the exact-bounds
axis only exact-bounds.sh and one sublet-lifetimes case run with `x-capstone-exact-bounds=true`; every other suite
leaves it at QEMU's default false, so Tier 0 does not exercise the board's stated feature set on that axis, and the
directed RTL arms ported from it say per test which setting they model. Tier 1:
the guide's Linux smoke (capstone_vm.ko, capstone-vexec contract.dom reading VIRTUAL_APPLICATION_OK, the stale access
denied 25 / exit 139), feature set `x-capstone-u-mode=true,x-capstone-exact-bounds=true`. Tier 2: run.py with the SQLite
and mruby images. Tier 3: every verdicts-v1 bundle with target capstone-virtual (26 bundles, 382 cells, 304 images at
44d82a9eb3f3; 91 images survive under /mnt/hdd/<user>/capstone/virtual-corpus-images-2026-10-16).

## 6. What this changes in the order of work

1. The two new Q-rows (Q-17 check order, Q-18 DROP) and the E divergences (#25 untagged arithmetic, Q-04 (b)) are
   one-line RTL changes each; they go into the first virtual bitstream's batch, not into separate cycles.
2. B stays first and is now specified: a virtual-context flag as its own term in `en_translation_o` and
   `en_ld_st_translation_o` (fetch and data, USER PTE permissions), the capability check re-keyed on `capmode_i`, the
   LDC clear re-translated and PTE-W-checked, a resumable page-fault escape kind. The first bare test under the flag
   builds its own Sv39 tables.
3. The exact-bounds core widening (the part common to (i) and (ii), see the costing document) can be designed with B
   and synthesised in the same cycle, since both reach the issue/forwarding and LSU cones; the memory-format half waits
   for the lead's choice.
4. A (the srevroot table) follows with the flat 30-bit id: the generation bits leave the id under the flag.
5. D grows: FP state per context (#30) joins the frame, the page-fault and node-pressure kinds, CSMINT/CSRETIRE.
6. The audit's four-to-five cycles become five to six: B (+ the common widening), the memory format, A, D, plus the
   batch above folded into whichever is first. Each is pre-registered before synthesis as the plan requires.
