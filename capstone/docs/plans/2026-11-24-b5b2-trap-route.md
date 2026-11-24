# B5b-2, the trap route on the Sublet format: plan and step list (2026-11-24, nothing built)

**What B5b-2 is.** The lead's option (C), staged (2026-11-22, relayed by the runtime lane; `2026-11-22-b5b1-format.md`).
B5b-1 was the format, the record and the instructions with CSRUNV kept; it is sealed as d0afdb40f, with CSMINT CONFIRMED
ON SILICON. B5b-2 is the trap route, the raw tag moves and the deletions. The scope, and the list of what the
collaborator's kernel needs, is llvm-capstone land/rtl-sublet-p2 `docs/plans/2026-11-17-sublet-b5b-scope.md`, items 3-7.
The runtime lane ranks it first: the default bitstream cannot move to Line B until the monitor runs without CAPENTER,
and nothing else on Line B reaches SPEC before that.

**Base:** capstone-ariane b5b1-step3 (the B5b-1 line; at 8446ce50c today). It already carries R-75 option (d)
(`R75_EARLYCLR` defined, config pkg :23; the lead ruled it on 2026-11-22, `2026-11-22-r75-option-d-on-bitstream-1.md`),
the Sublet format (`CAPSTONE_SUBLET_FMT`), CSMINT in hardware and the R-79 fix.

**Reference:** capstone-qemu address-space-root **9289114e06**, which is the kernel's pin. It is NOT c1985bb529, the
commit the 2a unit was built against. The commits from c1985bb529 to 9289114e06 are B5b-2's specification:
- a3006acfa6: srevroot protects U; sscratch and mscratch keep tags;
- 656a737d7d: zero-mask writes clear the scratch tag;
- 35de150cef: CSRUNV, the supervisor and virtual C deleted;
- 7800b39822: legacy C mode and the 30- and 55-bit formats deleted;
- 1d7995a5b4: tests ported;
- 500905eab2: rule 1 (S accesses through references) and rule 2 (plain S off user pages);
- 9289114e06: the review's gaps.

The interruptible CREVOKE (Line B step 4(b), spec-silicon 24ecb7b4daa4) moves the reference's walk_run later. That is
4(b)'s reference, not B5b-2's, and it is a separate item after B5b-2's trap route.

## The rules, read from the reference

- **Capability contexts** (7800b39822): protected U and S's references. M is plain: under Sublet nothing in M is
  checked.
- **Protected U** (a3006acfa6; translate.c:130-139): U is protected exactly when srevroot is nonzero, against the
  table srevroot names. Every write recomputes it and advances the table generation.
  - **Data accesses only.** "Protected U checks object authority on its data accesses, while its address still goes
    through the ordinary U-mode MMU". The scope note's "load, store and fetch" is wrong on fetch: the reference has no
    fetch or PC-liveness check (the scope note's own R-75 section says the same, op_helper.c:1101 and following).
  - Every data access needs a reference. An untagged base raises 24 (op_helper.c:1448-1462), a stale reference 25,
    and out of bounds 28. The fault address is cursor + imm, or scalar + imm for an untagged base.
  - Vector memory operations and the T-Head forms are illegal in protected U (translate.c:1430-1453). CVA6 has
    neither, so nothing to do.
- **Rule 1** (500905eab2): in S while srevroot is nonzero, a load, store, AMO, LR/SC, LDC or STC whose base register
  holds a reference takes protected U's checks against srevroot's table, translated with S's SUM index. An untagged
  base takes the plain path.
- **Rule 2** (500905eab2): while srevroot is nonzero, a plain S access reaches no user page, whatever SUM says
  (13 or 15).
- **Gaps** (9289114e06): HLV/HSV are illegal while srevroot is nonzero, and Zcmp and the T-Head memory forms are
  refused at startup. Our config has H off (CVA6ConfigHExtEn = 0) and Zcmp off (CVA6ConfigZcmpExtEn = 0), so both
  already hold by configuration. Nothing to build.
- **Tagged scratch registers** (a3006acfa6, 656a737d7d):
  - csrrw on sscratch or mscratch moves the whole register, tag included, both ways, also when rd == rs1;
  - csrr reads the address untagged;
  - every integer write clears the tag, and that includes csrrs and csrrc with rs1 != x0 even when the register holds
    0;
  - csrrs and csrrc with x0, and csrrsi and csrrci with an immediate of 0, are pure reads and keep the tag;
  - scapctl (0x5c0) is illegal;
  - U's access to srevroot and sscratch raises 2.
- **Raw tag moves** (op_helper.c:3514-3585):
  - **CLDRAW** reads a 16-byte granule into rd as {lo, hi} with tag = 0 and live_ok = 0, plus a separate raw_tag bit
    (the memory tag) and the granule's bounds as opaque state. The raw register grants no authority.
  - **CSTRAW** raises 24 on a TAGGED rs2. With rs2's raw_tag set it writes the 16 bytes and republishes the memory tag
    with the opaque bounds; otherwise it writes the bytes and clears the tag. A destination that cannot hold tags
    raises a store access fault before any byte changes.
  - Every other register write clears raw_tag (cap.h:176/181/211; op_helper.c:759, :1748).
- **CPTAG** (cap_mem_map.c, page_summary; op_helper.c:3573):
  - rs1 is a PHYSICAL address; a third operand acknowledges.
  - The result is the page's sticky "may hold tags" bit.
  - Every tag publication sets the bit, a raw copy included. An acknowledge clears it only when no tag in the page is
    set.
  - A page that is not 4 KiB-aligned or lies outside the tagged region raises a load access fault with tval = paddr.
- **Causes 24-30 delegable to S:** already in the RTL (Q-24, csr_regfile medeleg mask; CONFIRMED ON SILICON N=2).
  Nothing to build.

## What the RTL has (b5b1-step3 8446ce50c, re-read at source)

- The LSU has two capability gates:
  - load_store_unit.sv:1443 (cap_violation_detection): plain loads and stores (fu LOAD/STORE), checked only under
    `capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M`;
  - the LDC/STC lifetime verdict (:1604-1616): deliberately its own gate, with no capmode or privilege term, so every
    LDC/STC that reaches the LSU gets a verdict. It is a separate output applied after the DYN unit's causes.
  - (:1798 is a sim-only R43_TRACE display with the same capmode/M condition, not a gate.)
  - `capmode` is set only by CAPENTER (csr_regfile.sv:342), which is illegal under Sublet. **So on d0afdb40f the LSU
    checks no plain load or store (fu LOAD/STORE) at any privilege.** LDC and STC are fu CAPSTONE_DYN and take the DYN
    unit's own checks, and CSMINT has its own probes. Step 2 enumerates which of those are capmode-gated as well.
- srevroot: a CSR (0x5C1), written with flush_o (csr_regfile.sv:2026). `vctx_table_o` selects the table namespace
  for a context, for S outside a context with srevroot set, and for M under csupctl.table (:2939-2942). It does NOT
  cover U.
- sscratch is an integer register (:650, :1462). mscratch holds a capability only as half of CCSR CSCRATCH.
- The custom-0 decode holds only the debug print. CLDRAW, CSTRAW and CPTAG are illegal there, and no per-page summary
  exists.
- Board DRAM: 0x3c2d2000 usable bytes (caplifive.dts :46), so 246,482 4-KiB pages. The shadow-tag region starts at
  CAP_TAG_MEM_BASE 0xBC2D2D2D, one tag byte per 16-byte granule (ariane_pkg.sv:659). A per-page bitmap already exists
  in the tag adapter (step 6).

## Steps (each gets a synthesis-only gate before anything is built on it)

0. **The sweep list for the B5b-2 line, decided before step 1.** The 95-test list is full of CAPENTER, CSRUNV and
   CPMP tests, so after step 1 every sweep would read as a regression and the gate would be lost. Each test goes into
   one of three classes:
   - still valid, compared against b5b1-step3 in values, RVFI and cycles;
   - expected to raise 2, as positive controls for step 1;
   - dropped, with the reason.
   The list is committed with step 1.
1. **Decoder: unreachable, not removed.** The deleted operations decode as illegal:
   - CAPENTER, CALL, RETURN, CJALR and CCSRRW;
   - CSRUNV and the supervisor operations;
   - REVOKE, TIGHTEN, DELIN, SPLIT, SEAL, MREV, INIT, DROP, CSRETIRE, GENCAP and CAPMEM;
   - scapctl.
   The positive control is a port of QEMU's illegal.S, one arm per deleted operation, each raising 2. Physical
   removal (dom_switcher, the CPMP trackers, the supervisor) is step 8, gated alone: the scope calls removal the
   biggest risk, and a failed gate must be able to say which half failed.
2. **The LSU admission gate, by privilege.** Before the edit, every consumer of `capmode` is enumerated (csr_regfile,
   LSU, decoder, ex_stage, pmp_data_if, issue, commit) by name, and each one's fate is recorded. With CAPENTER gone,
   capmode is either dead or silently gating something still needed. The new rule replaces the plain gate (:1443);
   the LDC/STC verdict gate already covers every privilege and only needs its namespace (U) checked:
   - protected U (U, srevroot != 0): every data access is checked, and an untagged base raises 24;
   - S with srevroot != 0: a tagged base is checked, and an untagged one takes the plain path;
   - M, and anything with srevroot == 0: plain.
   The lifetime query resolves in srevroot's table for U as it already does for S: `vctx_table_o` gains protected U.
   Arms: route.S's protected-U cases (25 through a stale reference under T1; T1, T2, 0, T1 in turn; A's verdict under
   T1 does not carry to T2), and its s_ref_access and s_ref_state cases.
3. **Rule 2 in the MMU.** SUM becomes per-access: a plain S access ignores SUM while srevroot != 0, and an S reference
   access translates with SUM. The request needs a "base tagged" bit. The shape to copy is the CSRUNV forced-U
   override, cva6_mmu.sv:142-143. This touches the TLB permission cone, so the change is checked against the standing
   UNOPTFLAT list before a term is added. Arm: route.S s_plain_user_page (load 13, store 15 with SUM set; the load
   passes once srevroot is 0).
4. **Tagged sscratch and mscratch.** The rules above, through the CSR file's existing capability-wide lane (CCSRRW's
   CSCRATCH path), not a new one. Arms: route.S sscratch_keeps_tag and mscratch_keeps_tag, check by check (both
   commits' checks 8-9, 19-26, 36-37), csr_access, and masked_write.
5. **CLDRAW and CSTRAW, and the raw register state.** This is a third register state besides tagged and untagged: 128
   raw bits plus raw_tag, granting no authority. Before any code:
   - choose the representation (a raw_tag bit beside the tag, the bounds carried in the 128 bits as stored);
   - enumerate EVERY register read site: forwarding, the scoreboard, issue_read_operands, each FU's operand decode,
     commit and the CSR lane;
   - show at each site that a raw register reads as untagged: no check accepts it, no capability is built from it,
     and only CSTRAW reads raw_tag;
   - show that every other write clears raw_tag.
   The likely synthesis surprise is the extra bit through every forwarding path. Arms: CLDRAW then CSTRAW of a tagged
   granule republishes the tag; a tagged rs2 to CSTRAW raises 24; a raw register used as a base raises 24 in protected
   U; an integer write between CLDRAW and CSTRAW clears raw_tag.
6. **CPTAG as the OR of the page's live tags** (the runtime lane's ruling, 2026-11-24; no sticky array).
   - **Contract:** dev `docs/design/virtual-capstone/raw-tag-moves-proposal.md:60`: "False positives are allowed;
     false negatives are not." The reference's sticky bit is one allowed over-approximation. The OR is exact.
   - **The kernel's call sites** (dev `runtime/virtual/unpinned/linux-6.1-unpinned.patch`), all correct under OR:
     - copy_highpage copies raw only when the query reads 1;
     - arch_free_page queries, then acknowledges and clears;
     - arch_prepare_to_swap refuses on 1.
     None relies on the sticky history ("published since the last acknowledge").
   - **Divergence from 9289114e06, recorded as a model freedom:** after a scalar store clears a page's last tag, the
     reference reads 1 until an acknowledge, and this RTL reads 0. A test that expects 1 there tests the reference's
     choice, not the contract. The acknowledge stays an encoding and returns the same exact value as the query.
   - **Hardware: the existing page bitmap first, then the OR.** wt_axi_adapter.sv:122-135 (spec-silicon
     `2026-11-12-tag-page-bitmap.md`) already keeps one bit per 4 KiB page (4096 x 64 bits, 8 BRAM36), set by a page's
     first tagged store and cleared only at reset.
     - **Bit 0:** the page has had no tagged store since reset, so every tag in it reads 0. CPTAG answers 0 without
       touching DRAM. It MUST NOT read DRAM there: an unmarked page's tag bytes may be left over from an earlier boot
       (the comment at :125-127), and a stale 1 would be a false positive.
     - **Bit 1:** OR the page's 256 tag BYTES (one byte per granule, ariane_pkg.sv:659), read through the TWC or after
       draining it. The run is not word-aligned (CAP_TAG_MEM_BASE ...2D2D), so it spans 33 words.
     - The acknowledge does NOT clear the bitmap bit in the first cut. The result is exact without it, and clearing it
       would break the TWC's invariant that no page is unmarked except at reset (:183-184; F2's targeted fill
       invalidation relies on it). Clearing it is an optimisation that needs its own TWC invalidation.
   - **Binding condition:** the OR must see tags not yet written back. It reads through the tag cache, or drains the
     tag cache for that page first; a dirty tag missed is a false negative, the one forbidden outcome.
   - Faults as the reference: not 4 KiB-aligned or outside tagged RAM gives a load access fault, tval = rs1.
   - Arms: a page with one tag published reads 1; after a scalar store clears it, 0; a tag still dirty in the tag
     cache reads 1 (the positive control for the binding condition); a raw copy (CSTRAW with raw_tag) sets it; an
     unaligned or untagged-region address faults.
   - The gate runs alone, because it adds a tag-store read path.
7. **Namespace must-pass** (scope note, R-75 section):
   - va-rc-trap and va-rc-trap-zp as regression arms;
   - a record clear that takes effect no later than the namespace change it follows, so that no verdict from the
     previous namespace is consumed in the change's cycle;
   - every check added outside M resolving its id in the namespace of the code that runs under it.
   Under (d), as already built, the LSU's records give no verdict while the namespace change is pending. Steps 2 and 3
   add the U and S consumers, and each gets an arm that switches srevroot (T1, T2, 0) between two accesses in adjacent
   cycles.
8. **Physical removal** of dom_switcher, the CPMP trackers and the supervisor: later, with its own gate, after steps
   1-7 read on silicon.

**Cost to measure in simulation:** srevroot is written at every switch_mm, and each write raises flush_o (a TLB and
namespace flush). The per-switch cost lands on the SPEC baseline later, so step 2 reads it in cycles.

## Steps 0 and 1: what was found, and step 1's first cut (2026-11-24)

**Step 0 and step 1 as written above over-stated the work.**
- B5b-1 already makes most of step 1's list illegal under CAPSTONE_SUBLET_FMT (decoder :1212-1456): REVOKE, TIGHTEN,
  DELIN, SPLIT, SEAL, MREV, INIT, DROP, CSRETIRE, CALL, RETURN, CSSUPERVISE and CAPENTER. funct7 0x23 (supervisor_gc),
  0x40 (gencap), 0x41 (capmem) and 0x42 (clearcmmap) fall into the custom-2 default, which is illegal at decode.
- B5b-1 keeps an ids55 build (`+define+CAPSTONE_IDS55_FMT`; the (K) ruling of 2026-11-23). Its 95-test sweep is the
  regression gate of the shared datapath.
- So the deletions go under the Sublet format only, as B5b-1's did. The ids55 build stays unchanged and is checked by
  `verilator -E -P` token identity, and the 95-test triage is not needed. The semantics owner ruled this (2026-11-24).
- **Step 0's list** is therefore:
  - the ids55 95-sweep, unchanged, carried by preprocess identity;
  - the 15 Sublet directed tests of testlist_b5b1.yaml, compared before and after each deletion;
  - b5b2-illegal (two builds) as the positive control.

**What is left to delete.** Comparing against the reference's deletions (insn32.decode c1985bb529..9289114e06), four
remain legal under Sublet.

| op | encoding | deleted in the reference by | order (the semantics owner's ruling) |
|---|---|---|---|
| CJALR | custom-2 funct3 5 | 7800b39822 | first: step 1's first cut |
| CCSRRW | custom-2 funct3 7 | 7800b39822 | after step 4 gives tagged sscratch/mscratch their own path |
| CSRUNV | custom-2 f3 1 f7 0x24 | 35de150cef | once the monitor no longer needs it |
| capcreate, captype, capnode, capperm, capbound | custom-3 f3 0 f7 0x04-0x08 | CVA6-only (the reference's debug mint gencap went in 7800b39822) | last, after the Sublet tests mint through CSMINT |

- The custom-3 mints are illegal on the Line B Sublet bitstream (ruled). The reason: "an instruction that mints
  authority from an integer has no place in Section 5's ISA".
- CAPPRINT stays, the analogue of csdebugprint, provided it changes no architectural state on silicon.
- Six Sublet tests use the mints and carry a note saying so: csmint, r79-cpmp, sublet-fmt, sublet-life, sublet-ns,
  va-runv-snap.

**Boot-set scan before each deletion** (ruled): every image a Line B kit bakes or loads, by encoding. That means
fw_payload, the kernel Image, every .ko, .dom and userspace binary in the cpio, and any external initrd, taken from
the kit's SHA256SUMS and its cpio listing.
- Today's set is the csmint-bare-2026-11-23 kit. Its five images (sha256 = SHA256SUMS) read 0 CJALR, 0 CCSRRW and
  0 CSRUNV words at any 2-byte offset, and 20-204 custom-3 mint words each.
- Positive controls: a real ELF (an ids55 sweep jalr.o, cjalr = 1 as its source), and a mutant (2 of 2, one at a
  2-byte offset).
- The collaborator's trap-route kernel and firmware join the set before the CCSRRW and CSRUNV deletions.

**Step 1, first cut** (capstone-ariane b5b2-step1 c36facb5a, on b5b1-step3 27e75c38a): CJALR is illegal under Sublet.
- b5b2-illegal.S, the port of the reference's illegal.S, was pre-registered in its header before any run.
  - Base: sites 1-17 read 2, with mepc = the site and mtval = the word. CJALR reads 24. The five mints and CCSRRW
    retire. The trap count is 18. CSRUNV, its own build, retires.
  - c36facb5a: CJALR reads 2 / site / word, and everything else is unchanged.
- The 15 Sublet directed tests are identical before and after the deletion: logs byte for byte, .iss traces once
  wall-clock and port lines are removed. The same holds on the committed tree, dirty=no.
- ids55: the preprocess output is 98,689 lines, byte-identical.
- Lint, Sublet build: PASS at REF (UNUSEDSIGNAL 1567, UNOPTFLAT 39, loop names 39 = 39). ids55 build: equal to B5b-1's
  own ids55 listing on every counter and by name.
- Not yet synthesised. The synthesis-only gate is batched with step 2: step 1 adds one illegal term to the decoder.

### Step 2's prerequisite: every consumer of `capmode` (b5b2-step1 c36facb5a, read 2026-11-24)

How the list was made: `git grep -w -E 'capmode(_[a-z]+)?' -- core`, excluding the generated `*.anvil.sv`. No `.anvil` or
`.anvilh` source mentions capmode; `cap_mode` has no hits. 42 lines in 8 files.

- **State** (csr_regfile.sv): `capmode_q`, reset to 0 (:3365) and set sticky by `capmode_set_i` (:342), which is CAPENTER's
  commit pulse (cva6.sv:2207 `capenter_commit`). CAPENTER is illegal under the Sublet format (decoder :1456), so
  **under Sublet capmode is 0 from reset on**, and every consumer below is already dead there.

| # | site | what it gates | fate |
|---|---|---|---|
| 1 | csr_regfile.sv:2212 | flush after a CCSRRW write | CCSRRW is deleted after step 4. Step 4's tagged sscratch/mscratch must NOT inherit this capmode gate: gate by CSR address and privilege |
| 2 | csr_regfile.sv:2654 | the CCSR write case (CTVEC, CEPC, CSCRATCH, CPMP...) | as 1. Today a Sublet CCSRRW writes nothing (b5b2-illegal site 24 retires with no trap) |
| 3 | decoder.sv:1452 | CSRUNV illegal in M with capmode | goes with CSRUNV's deletion |
| 4 | commit_stage.sv:392 | the PC capability check (M && capmode) | stays dead. The reference has no fetch or PC check (op_helper.c:1101 on); removed in step 8 |
| 5 | commit_stage.sv:487 | `pcc_chk_o`, the PC probe request | as 4 |
| 6 | load_store_unit.sv:1443 | the plain load/store capability check (M && capmode, fu LOAD/STORE) | **replaced in step 2** by the privilege rule (protected U; S with a tagged base) |
| 7 | load_store_unit.sv:1798 | R43_TRACE `$display` (`ifndef SYNTHESIS`) | sim-only; follow 6's new condition or drop |
| 8 | pmp_data_if.sv:331 | the CPMP execute check on S/U fetch | stays dead (no CPMP in the reference; B5b-1 tied the trackers off); step 8 |
| 9 | pmp_data_if.sv:396 | the CPMP check on S/U loads and stores | as 8 |
| - | cva6.sv:513/1495/1881/2154/2236, id_stage.sv:86/341, ex_stage.sv:91/901, load_store_unit.sv:214/734, pmp_data_if.sv:43 | ports and wiring | follow their consumers |

The LDC/STC verdict gate (load_store_unit.sv:1604-1616) has no capmode term; the comment at :1604-1607 says so.

## Silicon acceptance: bare kits

No Linux kit can run until the runtime lane's integer-M monitor exists: the current one executes CAPENTER at init. So
B5b-2's silicon acceptance is a set of bare kits. They are QEMU's tests/trap-route route.S cases ported to the bare
wrapper, with M writing srevroot and setting medeleg bits 24-30 in the prologue:
- protected U: 25 and 0, across T1, T2 and 0;
- CSR access (2);
- the scratch tags;
- s_ref_access;
- s_plain_user_page (13 and 15);
- s_ref_state;
- the raw moves and CPTAG arms of steps 5 and 6.
Each kit is pre-registered from image simulations before any board run.

## Kept on the list (runtime lane, 2026-11-24), not in B5b-2

- **Interruptible CREVOKE** (Line B step 4(b)): spec-silicon 24ecb7b4daa4. Read at source on b5b1-step3:
  - the 2a unit has no interrupt input, and a request runs to its end (capstone_sublet_life.sv:33);
  - Sublet's CREVOKE walks the run twice, a check pass (:495-505) and a commit pass (:528-546).
  The design is the suffix release with the resume record t, a partial status, and a replay at the same pc. K is a
  localparam, first estimated from 2a simulation, then read on silicon on the first 4(b) bitstream. The QEMU revision
  lands before the RTL.
- **The record-cache decision model:** QEMU record-access traces for gcc, mcf and Perl, replayed through the 5.2 cache,
  a generation-only cache and no cache, and calibrated on B5a's mcf reading and events 34-37. The cache is built only
  on a gain on at least one of the three and a loss on none. It runs alongside B5b-2.
