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
   - **Binding condition** (SUPERSEDED, see "Step 6 part 1" below: the TWC never holds a dirty tag; the condition is
     ordering behind every older tag write): the OR must see tags not yet written back. It reads through the tag
     cache, or drains the tag cache for that page first; a dirty tag missed is a false negative, the one forbidden
     outcome.
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

## Step 2: the LSU admission gate by privilege (2026-11-24)

capstone-ariane b5b2-step2 94cbc880e (on c36facb5a; the arm pre-registrations fc2c24817 and 7690e761b were pushed before
their runs). Under CAPSTONE_SUBLET_FMT only. The ids55 preprocess is byte-identical.
- The rule: protected U checks every LOAD/STORE-fu access, and an untagged base raises 24. S with srevroot != 0 checks a
  tagged base (rule 1); an untagged one is plain. M is plain, and so is anything with srevroot == 0. The privilege is the
  hart's: M with MPRV is plain, as in the reference.
- The order is capstone_check_access's: 24, lifetime (25 / replay), 26, 27, 28. A checked access's capability cause
  outranks misalignment and a non-canonical address (the claim-auditor's F1/F2).
- Protected U joins the table namespace (vctx_table_o). The claim-auditor enumerated every consumer, and only the
  reference's behaviour changes: U's lifetime ops and probes resolve in srevroot's table.
- The precondition, confirmed by the semantics owner and read here: the kernel's twin runs with
  x-capstone-trap-route=true (native-kernel-new-qemu, capstone/kernel/tests/{trap-route,syscall-boot}/boot.py:6).
- Readings, as pre-registered:
  - b5b2-uprot reads 25 / 24 / 28 / 25 in protected U, all plain with srevroot 0 and in M, and 25 / 28 on S's tagged
    base; traps 6 (base: every access plain, traps 0).
  - -DUP_MIS: 0x18 0x1c 0x4 0x19. Before the F1 fix, and on the base: 4 4 4 4.
  - b5b2-illegal and the 15 Sublet directed tests are identical to step 1. Lint is at REF, with 0 by-name lines.
- Recorded, not changed: the AMO/LR/SC permission arm (write only, against the reference's LR read / SC write / AMO
  both) is unreachable while Sublet capabilities decode with every right (the semantics owner: re-open if rights
  return). The ea + size bounds sum can wrap near 2^64 (pre-existing).
- The ids55 lint's LOOP-NAMES-RC=1 (39 = 39) is the Sublet REF's long-standing port-name difference (edf925592), the
  same on b5b1-step3. It is not B5b-2's.

### Synthesis-only gate on steps 1 and 2: pre-registration (written before the request)

Hash 94cbc880e. Base: d0afdb40f's own synthesis-only gate (b5b3-d0afdb40f-gate). Its core/ is identical to b5b1-step3
27e75c38a's.

| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within +/-505 MB of 5,446.4 MB, under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 combinational loops (as the base) |
| 3 core LUT | at most the base + 2,500 (145,492 -> 147,992); FF reported |

Reported, not graded (the per-module floor is +/-2,500; the ex_stage + issue + lsu stem is read summed):
- lsu_i, predicted within +/-500 of 24,089. The new gate replaces the capmode term with two flop-derived bits and
  reorders an existing chain; the F1 override is one priority term.
- csr_regfile_i, predicted within +/-200.

Timing is not graded by a synthesis-only run. The auditor's open point: the reorder puts the lifetime lookup and the
same-cycle invalidation ahead of the 64-bit bounds compare inside cap_exception, the LSU's dominant slack term. That is
read at B5b-2's full build, where steps 1-6 are batched into one bitstream.

### The gate on steps 1 and 2: read (board lane, 2026-11-24), PASS 3/3

94cbc880e (the job's sha verified). Base: d0afdb40f's gate. Grade file
`/mnt/hdd/<user>/synth/b5b2s2-94cbc880e-gate-out/grade-b5b2s2.txt`.
- Key 1 (memory): TM peak 5,368.5 MB, against 5,446.4 (-77.9; ±505). The whole-run peak was 5,436.6 MB. PASS.
- Key 2 (loops): 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy 1), LUTLP-1 0 (toy 1); the in-flight
  gate PASS.
- Key 3 (core LUT): 144,583, against 145,492 (-909; limit 147,992). PASS.
- Reported, inside the predictions:
  - lsu_i 23,818 (-271, predicted ±500);
  - csr_regfile_i 7,409 (+9, predicted ±200);
  - ex_stage + issue 103,895 (+122). ex_stage +2,048 against issue -1,926 (scoreboard -1,075, issue_read_operands
    -851) is redistribution across the stem.
- No module row is beyond ±2,500. FF -117. LUTRAM and BRAM are unchanged.
- The synth dcp is kept as twc/gate-synth-b5b2s2-94cbc880e.dcp (sha16 ab9efe547bd4b481).

## Step 3: rule 2 in the MMU (2026-11-24)

capstone-ariane b5b2-step2 99d7ef22c, on step 2. The arm pre-registrations b3329c988 and b60a63efc were pushed before
their runs; the test fixes 882fc699e and 223a6db2a are recorded in the test's header.
- **The rule** (riscv_cpu_mmu_index). While the hart is in S with srevroot != 0, SUM is not consulted:
  - a plain access to a user page faults 13 / 15;
  - an access through a reference reaches user pages (S's SUM index);
  - M with MPRV and MPP = S keeps SUM.
- **How it is built.** `daccess_err`'s S term takes `cap_sref_ctx_i ? cap_ref_q : sum`. `cap_ref_q` is "the base is a
  reference", taken from lsu_ctrl and registered with the request (as misaligned_ex), so it judges the same access as
  dtlb_pte_q.
- **Readings, as pre-registered.**
  - b5b2-rule2, step 3: A 0xd 0xf 0 0x77 | B 0 0x77 0xd | C 0 0x77 0 0x77 | D 0 0x77 | traps 3.
  - The base: A 0 0 0 0x77 | B 0xd 0 0xd | C and D the same as step 3 | traps 2.
  - -DR2_PAIR, the claim-auditor's leak test: step 3 1 0xd | 0 0 0x77 | 1 0xf; base 2 0xd | 1 0xd 0 | 2 0xf.
  - Every other B5b-2 arm and the 15 Sublet tests are byte-identical. Lint is at REF, with 0 by-name lines.
- **The claim-auditor's residuals** (all low, none introduced here):
  - (a) A switcher store translated while a STORE or STC is at the LSU head takes the head's cap_ref, as misaligned_ex
    already does. A refused switcher store would drop the switcher write (store_unit.sv:382, pre-existing).
    Reachability needs cap_sref_ctx (outside a context) together with a switcher write.
  - (b) cap_sref_ctx is read unregistered at judgement time, as sum_i is. The switcher's register-path srevroot and
    privilege writes show no flush in what was read.
  - (c) The hardware keys "a reference" on the decoded type != NOT_CAP, the reference on the tag. They differ only for a
    tagged register of type 0, which no Sublet op produces.
- **Test slips** (memory each-phase-sets-its-own-preconditions):
  - The copied page-table block clobbered the trap counter (t3 = x28).
  - Phase D inherited SUM from phase E.
  - Both are fixed and the record holds the fixed test's readings.

## Step 2b: supervisor references (2026-11-24, the semantics owner's ruling)

capstone-ariane b5b2-step2b 95644ee8b, on step 3. The test commits e68d49a9c, cfff477d5 and f4890ec7f were each pushed
before their runs.
- **The configuration.** The evaluated configuration runs x-capstone-supervisor-refs=true (the kernel boots' cpu
  strings). The reference is trans_capstone.c.inc plain_base_refs / gen_csldc_body.
- **LDC/STC are checked only in a capability context:**
  - protected U;
  - S with srevroot != 0 and a tagged base;
  - inside a supervised context, cap_data_ctx[2] = sup_active. That is CSRUNV's under Sublet, where CALL is illegal;
    the reference has no contexts, and B5b-1's checks stay there while CSRUNV lives.
- **Any other S or M LDC/STC is plain and keeps the tag:** alignment only, tval the address, no lifetime verdict.
  Unprotected U has no LDC/STC (2).
- **LCC selector 8 in S** answers the id, generation << 31 | slot.
- **Readings (b5b2-sref), as pre-registered:**
  - the base reads its vector (traps 10);
  - step 2b reads M 0 0 1 1 0 | S 0 0 1 4 3 0x1c 6 | 6 0 1 | S0 0 | U0 2 | U1 0x18 0 | traps 3;
  - -DSR_GEN (a free record at generation 0xC0000001): LCC 8 = 0x6000000080000006, the record's generation, flag 1.
- **First cut, fixed:**
  - LCC 8 read 0: the override was registered, and LCC answers in its dispatch cycle.
  - The context tests lost their LDC checks (contexts run at M).
  - The test's counter was clobbered (memory each-phase-sets-its-own-preconditions, sharpened with a grep).
- **Regression.** The other B5b-2 arms are byte-identical. Of the 15 Sublet tests, 29 of 30 files are byte-identical;
  sublet-ns-edgem's .iss is not.
  - Its re-executed S LDC after the edge (srevroot 0) is now plain and no longer replays: 45 cycles fewer.
  - R43_TRACE shows the arm still creates its condition: one probe sent and never resolved, i.e. pending across the
    edge, on both trees.
  - The base's other probes came from the re-executed, then-checked LDC.
- **Lint:** at REF, by name 0 lines.
- **claim-auditor:** nothing refuted. The plain path is plain end to end, the partition equals the reference's in every
  reachable combination, and the id is right bit for bit. Residuals:
  - (a) STC's bypass record still reads the base (a tagged UNINIT rs1 moves +16; rd is cleared for a non-NONLIN rs2).
    This is reachable only through the custom-3 debug mints, which are to be deleted.
  - (b) Under +define+CAPSTONE_NO_DYN_BYPASS, the A/B control, the DYN unit would still raise 24 on an integer base.
  - (c) A plain LDC/STC to a device range is untraced (the reference: 5 / 7).
  - (d) R43_TRACE's verdict line is blind to LDC/STC under the bypass.

### Synthesis-only gate on steps 3 and 2b: pre-registration (written before the request)

Hash f4890ec7f (core/ = 95644ee8b). Base: 94cbc880e's gate (b5b2s2-94cbc880e-gate-out).

| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within +/-505 MB of 5,368.5 MB, under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (the toy fires) |
| 3 core LUT | at most the base + 2,500 (144,583 -> 147,083) |

Reported, predicted within the per-module floor:
- lsu_i ±500 of 23,818 (the LDC/STC plain branch);
- csr_regfile_i ±200 of 7,409;
- the MMU (cva6_mmu inside lsu_i: one 2:1 mux on daccess_err plus a flop);
- ex_stage + issue summed: a 64-bit 2:1 mux on the DYN result, plus the LCC 8 record.

### The gate on steps 3 and 2b: read (board lane, 2026-11-24), PASS 3/3

f4890ec7f, graded against the pre-registration above (20d7cda0d031). Base: 94cbc880e's gate. Grade file
`/mnt/hdd/<user>/synth/b5b2s3-f4890ec7f-gate-out/grade-b5b2s3.txt`.
- Key 1 (memory): TM peak 5,065.6 MB, against 5,368.5 (-302.9; ±505). The whole-run peak was 5,206.2 MB. PASS.
- Key 2 (loops): 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy 1), LUTLP-1 0 (toy 1); the in-flight
  loops-only gate against 94cbc880e PASS.
- Key 3 (core LUT): 146,206, against 144,583 (+1,623; limit 147,083). PASS.
- Reported, OUTSIDE my two module predictions (reported rows, not keys; both bands were written tighter than the
  recorded ±2,500 per-module redistribution floor, so a miss there grades nothing):
  - lsu_i 25,365 (+1,547, predicted ±500). Step 3's MMU term and step 2b's plain LDC/STC branch both sit in lsu_i;
    which of the two carries the growth is not attributed.
  - csr_regfile_i 6,900 (-509, predicted ±200), after the ctx[2] widening.
- ex_stage + issue 105,036 (+1,141): ex_stage_i +1,341, issue_stage_i -200 (issue_read_operands +398, scoreboard
  -598); id_stage_i +238, dom_switcher +365. Core FF 55,175 (+175). LUTRAM and BRAM are unchanged. Top LUT 173,504
  (+1,638).
- Three rows beyond ±2,500, as one pair: ex_stage_i/dyn_unit 14,843 -> 10,714 (-4,129; its _spawn_0 -4,183) and
  ex_stage_i/rev_node (capstone_sublet_life) 3,312 -> 7,690 (+4,378, FF +1). Net +249. Neither module's source changed
  in 94cbc880e..f4890ec7f; their parent ex_stage gained the LCC 8 block, which muxes dyn_unit's result cursor and
  metadata. UNATTRIBUTED: no cell-level diff was run. It is recorded as a move between siblings, not explained.
- The synth dcp is kept as twc/gate-synth-b5b2s3-f4890ec7f.dcp (sha16 8baaa42e6fee767e); vivado-guard peak RSS
  22.00 GB, 53m43s.
- This gate's dcp is step 4's base.

### Context arms and the edgem checker (b2f27b3b3, tests only, pushed before the runs)

- sublet-ns -DSL_CTXINT (an LDC inside a CSRUNV context through an integer base) and -DSL_CTXOOB (through a tagged
  base one past its bound): the full vectors are equal on the base core (99d7ef22c) and on step 2b. The LDC reads 0x18
  (24) and 0x1c (28) respectively. Contexts run at M, so this is the reading that cap_data_ctx[2] = sup_active keeps
  them checked.
- edgem-r43-check.py makes the pending-probe condition part of sublet-ns-edgem's verdict. It exits 0 on both trees'
  R43_TRACE runs (a probe dropped while pending), 1 on a hand-made all-resolved trace, and 2 on a run with no R43
  lines.

## Step 4: sscratch and mscratch hold a reference (2026-11-24)

capstone-ariane b5b2-step4 eb03e7c99 = 7c8550cf8 (the change) + its fix. Both tests were pushed with their pre-registrations
before any run.
- **The rule** (capstone-qemu 9289114e06 helper_capstone_scratch_swap / _rmw; route.S cases 2, 3, 4 and 5).
  - csrrw (the register form) moves the whole register both ways, also when rd = rs1.
  - csrr reads the address untagged and keeps the tag.
  - Every other write clears the tag: csrrs / csrrc with a source register (even one holding 0), csrrwi, the nonzero
    immediate forms, and csrw of an integer.
  - csrrs / csrrc with x0 and the zero-immediate forms write nothing.
  - U access and scapctl (0x5C0) were already illegal (the default arms and the privilege check); they are tested,
    not built. So is masked_write (the srevroot write masks; protection keys on srevroot_q).
- **How it is built, with no signal on the standing loop list taking a new term:**
  - issue: under Sublet, the metadata file's read port 0 also addresses rs1 for a CSR_WRITE without use_zimm.
  - ex_stage puts a plain CSR op's rs1 lane on writeback port 0 with cap_result.valid = 0, so neither the rs1 write
    lane nor rs1 forwarding moves.
  - commit's csr_scr_swap (CSR_WRITE, not use_zimm) sends that lane to the CSR file and writes rd's lane from a new
    port, csr_scr_rmetadata, keyed on the CSR buffer's registered address (not ccsr_en, csr_read or conv_csr_addr).
  - sscratch gains a 98-bit lane; mscratch's lane is the existing cscratch triple, so a CCSRRW of CSCRATCH and a csrrw
    of mscratch see one register. The switcher's sscratch write (id 65) and the context init clear the tag.
- **First run VOID (7c8550cf8).** b5b2-scratch read the base vector on both trees, value for value, at the same 39,529
  cycles. The metadata file's read port 0 addressed rs1 only for capability ops and loads/stores, so every csrrw
  carried x0's lane. The run also exposed a test defect: two arms copied a capability with mv, an integer op. Both
  were fixed in eb03e7c99, and the test header records the void run. The pre-registered vector is what made the no-op
  visible: lint, the preprocess identity and every regression arm were clean on it.
- **Readings (eb03e7c99), checked by script against the headers:**
  - b5b2-scratch, step 4: all 53 as pre-registered (route.S checks 1-26, 30-37, 8-9). This includes the end-to-end
    controls: a capability fetched back through sscratch faults 28 on an S load one past its bounds, and one that went
    through mscratch, stc, then ldc does the same.
  - b5b2-scratch, base (b2f27b3b3's core): all 53 as pre-registered. The two trees differ at 18 positions. The grader's
    negative control (base readings against step 4's vector) shows 18 mismatches.
  - b5b2-mwrite: 1 0x19 0x18 1 0 0x77 2 on both trees, as pre-registered.
  - Regression: the other 25 tests (8 B5b-2 arms, 15 Sublet tests, 2 context arms) read the same values at the same
    cycle counts as step 2b's own runs.
- **Static checks on eb03e7c99:**
  - ids55 is token-identical to step 2b: 359,790 tokens, one line break moved.
  - Lint at REF: LATCH 44, MULTIDRIVEN 3, UNOPTFLAT 39 (the 39 reference names), UNDRIVEN 25, UNUSEDSIGNAL 1567.
  - By name against step 2b: 0 lines in both builds, WIDTH included.
- **claim-auditor** (over 7c8550cf8, then eb03e7c99). It found the read-port cause independently. With the fix, no
  defect in eight items: every reader of a valid = 0 cap_result; csrrwi through use_zimm; CCSRRW and the mscratch
  arms; port 0 shared with mult; latches and loops; the read port for the switcher and the hazard logic; forwarded rs1.
  Residuals:
  - (a) Inherited: a csrrw at the commit head with a PC-capability exception still writes the CSR (csr_op_o stays
    CSR_WRITE while commit_csr_o is cleared). Before this change that was the integer only; now it is the lane too.
  - (b) Latent, RVH = 0: the read lane is keyed on the raw address while writes go through conv_csr_addr. Under V = 1 a
    csrrw of sscratch would return HS sscratch's lane.
  - (c) A tagged linear rs1 is copied, not moved, as in the reference.
  - (d) A CSRUNV context's save and restore moves sscratch as an integer, so the tag does not survive it.

### Synthesis-only gate on step 4: pre-registration (written before the request)

Hash eb03e7c99. Base: f4890ec7f's gate (b5b2s3-f4890ec7f-gate-out, dcp sha16 8baaa42e6fee767e). No loop-listed signal takes
a new term, but csr_regfile, commit and issue_read_operands are where the list lives, so step 4 gets its own gate.

| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within +/-505 MB of 5,065.6 MB, under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (the toy fires), LUTLP-1 0 (the toy fires) |
| 3 core LUT | at most the base + 2,500 (146,206 -> 148,706) |

Reported, graded nothing (bands at the recorded ±2,500 per-module floor, not tighter):
- core FF: predicted +98 (sscratch's lane: 64 metadata + 1 tag + 33 slot), within ±300;
- csr_regfile_i (base 6,900 LUT): the lane's write muxes and a 98-bit 2:1 read mux by address, a few hundred LUT;
- commit and the regfile write data: one more 98-bit 2:1 mux on rd's lane;
- issue_read_operands: one more term in port 0's read address; scoreboard: no new flops (port 0's rs1 lane fields
  exist), possibly LUT on the port-0 cap write;
- the dyn_unit / rev_node pair (f4890ec7f: 10,714 / 7,690): if either moves beyond ±2,500 again, a cell-level diff of
  ab9efe547bd4b481, 8baaa42e6fee767e and this gate's dcp is asked for before step 5.

### The gate on step 4: read (board lane, 2026-11-24), PASS 3/3

eb03e7c99, graded against aad164b18412. Base: f4890ec7f's gate. Grade file
`/mnt/hdd/<user>/synth/b5b2s4-eb03e7c99-gate-out/grade-b5b2s4.txt`.
- Key 1 (memory): TM peak 5,066.5 MB, against 5,065.6 (+0.8; ±505). The whole-run peak was 5,134.6 MB. PASS.
- Key 2 (loops): 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy 1), LUTLP-1 0 (toy 1); the in-flight
  loops-only gate against f4890ec7f PASS.
- Key 3 (core LUT): 145,169, against 146,206 (-1,037; limit 148,706). PASS.
- Reported:
  - csr_regfile_i FF +98 exactly (6,468 -> 6,566): sscratch's lane, as predicted. Its LUT is 7,491 (+591).
  - Core FF 54,970 (-205): OUTSIDE my +98 ±300. ex_stage_i is FF -310 (dyn_unit +7, rev_node -2, lsu_i -11), so
    about -304 sits in ex_stage's own logic. Unattributed.
  - i_issue_read_operands 26,611 (-2,342) and i_scoreboard 17,244 (+2,326): issue_stage_i as a whole -16. commit_stage_i
    286 (+8). ex_stage_i 59,863 (-1,302), lsu_i 23,997 (-1,368), id_stage_i +226, dom_switcher -346, i_wt_dcache +350,
    top 172,460 (-1,044).
  - Beyond ±2,500: dyn_unit 10,714 -> 15,670 (+4,956; _spawn_0 +4,971) and rev_node 7,690 -> 3,241 (-4,449). The pair is
    back near the steps 1-2 values (14,843 / 3,312), and neither source changed in any of the three steps. Summed, the pair
    reads 18,155 / 18,404 / 18,911 across the three gates. The pre-registered condition fired: the board lane runs a
    read-only cell census over the three dcps (ab9efe547bd4b481, 8baaa42e6fee767e, b05ca47912c75223), shaped to separate a
    re-homed shared cone (stems moving with an equal and opposite sign, a flat sum) from logic that appears and
    disappears. It does not hold step 5.
  - **The census reads re-homing** (board lane, read-only; twc/pairdiff/analysis.txt on apollo):
    - The pair's sum in LUT primitives is 21,627 / 21,271 / 21,768. That unit counts above the utilization report's
      by 897-2,566 per instance; the direction of every move matches.
    - 265 leaf stems, 47 seen in both modules; 9,927 moved across them in total. About 9,400 of that is seven stems,
      all mem_q[N].sbe (cap_result ra_data / rs1_metadata / rs1_cursor, result, pc_next, pc_next_metadata, base_addr),
      each moving with an equal and opposite sign.
    - These are a scoreboard-entry cone: its inputs are the entry flops (5,272 FDCE, always in i_scoreboard) and the
      writeback data. Its outputs were not followed to their loads, so write-side or read-side is untested.
    - The cone sits in rev_node only in f4890ec7f's dcp. rev_node's own stems never move. In dyn_unit the cone's pins
      include the LCC 8 block's nets.
    - A residue of a few hundred LUT is unattributed (the s4 LUT6/LUT5 mix shift).
    - The control (the 218 stems confined to one module show no cross-module move) cannot fail by construction. The
      informative contrast is 7 stems carrying about 95% of the move.
  - **From step 5 on**, dyn_unit and rev_node are graded as one sum together with i_scoreboard, never per module.
- The synth dcp is kept as twc/gate-synth-b5b2s4-eb03e7c99.dcp (sha16 b05ca47912c75223); vivado-guard peak RSS 24.59 GB,
  51m41s.

## CCSRRW deleted under the Sublet format (2026-11-24)

capstone-ariane b5b2-ccsrrw 87aec16b4, on step 4. One decoder term (capstone-qemu 7800b39822 deleted it). Under Sublet a
CCSRRW already wrote nothing (a ccsr write needs capmode, which only the illegal CAPENTER sets), so the deletion turns an
inert instruction into cause 2. Pre-registered in b5b2-illegal.S's header before the run; checked by script:
- b5b2-illegal: site 24 reads 2 / site / word, and the trap count is 19.
- Every test that executes an inert ccsrrw reads as predicted. sublet-ns.S and its eight arms, and va-runv-snap, each
  read exactly one more cause-2 print with the rest unchanged. r79-cpmp (built without its IDS55 define) reads a trap
  count +4.
- The other 15 tests: identical in value and cycles to step 4.
- sublet-ns-edgem still creates its condition on an R43_TRACE build: edgem-r43-check.py exits 0, a probe pending across
  the edge.
- Lint: Sublet at REF, with 0 by-name lines against step 4 in both builds. ids55 is token-identical.
- **Not handed to synthesis:** the ruled boot-set scan lacks the collaborator's trap-route images (f-keep fw_jump.elf,
  the kernel Image(s), the initramfs). None of them is on our hosts; the runtime lane asked for them, with sha256 and
  the cpio listing, through the lead.
  - By source, the kernel patch series (0001-0051) has no CCSRRW encoding (the same grep finds LDC/STC in 0011, 0014 and
    0017). The shared runtime has CCSRRW only in tests/application/context-probe-asm.S, an old P1 native-context test
    to look for in the initramfs listing.
  - f-keep's source is not in our repositories, and patch 0013 describes it as firmware that keeps register tags across
    M traps, so it is the main unknown. A source check does not replace the binary scan.
  - The deletion's own gate is batched with step 5.
- **The boot set, scanned (2026-11-24).**
  - **Instruments.** Two scanners of my own, each counting raw encodings (opcode 0x5b, funct3 7) over a linear
    RVC-aware sweep of the executable sections:
    - an objdump-based one, used for the kernel;
    - an ELF and ar parser, for the capstone-machine images, which GNU objdump cannot disassemble. Its first pass read
      "no instructions", an error, not a zero.
    - Controls: a CCSRRW after a c.nop reads 2, as an object and as an archive member; LDC and LCC read 0. The parser
      reproduces the objdump scanner's vmlinux census.
  - **Kernel:** the runtime lane's defconfig REBUILD of the trap-route kernel, from 0133b9e21cd5's tree (Linux 6.1,
    apply.sh, qemu 9289114e06 target/riscv). 0 CCSRRW in vmlinux and its 111 modules. vmlinux carries 38 op-0x5b
    words: LDC 15, STC 4, LCC 10, CINCOFFSET 3, CSMINT 1, CDERIVE 1, CREVOKE 3, SHRINK 1. This is not the collaborator's
    binary, whose config is unknown.
  - **Userland** (origin/trap/eval build-trap-smoke.sh, dev's clang): 0 CCSRRW in the 8 smoke .dom images and every
    member of libc-capstone.a and libtrap-runtime.a (1,205 with instructions). The .dom images carry 1,703-2,852
    op-0x5b words each.
  - **The runtime lane RETRACTED** start-musl.S's 30 CCSRRW as evidence: the trap route links runtime/virtual/start.S
    (Application.cmake :44, :49-50, checked here).
  - **The semantics owner's ruling** (the runtime lane, reported to the user; the lead may overrule): f-keep is not
    needed.
    - The deletion is reference-exact: 7800b39822 is an ancestor of 9289114e06.
    - The collaborator's tests ran on qemu-1b = 9289114e06, where CCSRRW is already illegal. That bounds only the
      paths those tests executed.
    - f-keep never boots on the bitstream. The condition: the board's trap-route firmware gets this encoding scan
      before its first boot.
  - **Next:** the deletion is carried onto the B5b-2 tip as a new commit, then lint, its arms and a synthesis-only gate.
    It rides the next bitstream.
- **Carried onto the B5b-2 tip:** capstone-ariane b5b2-ccsrrw2 7997b51a2 (0c0458149 + 87aec16b4's two files,
  cherry-picked clean).
  - ids55 is token-identical (359,790).
  - Lint on the Sublet build is at REF, with 0 lines by name against step 6c in either build.
  - Readings, script-checked against the tip without the deletion: only the pre-registered positions move.
    - b5b2-illegal: site 24 goes 0xff -> 2/1/1, and the trap count 18 -> 19.
    - sublet-ns and its seven arms, and va-runv-snap: one cause-2 print each, inserted.
    - r79-cpmp: trap count +4.
    - Every other test, steps 5-7's included, reads the same.
  - Against the deletion's own run on step 4, all 27 shared tests read the same.
- **Its synthesis-only gate: pre-registration** (written before the request).
  - Hash 7997b51a2. Base: gate 6b (0c9992bc6, dcp sha16 3986f0ae821b9832), whose synthesised RTL equals the full
    build's.

  | key | pass if |
  |---|---|
  | 1 memory | technology-mapping peak within ±505 MB of 5,063.9 MB, under the 40 GB guard |
  | 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (the toy fires), LUTLP-1 0 (the toy fires) |
  | 3 core LUT | at most the base + 2,500 (147,893 -> 150,393) |

  - Reported, graded nothing:
    - id_stage_i / decoder: one OR term into illegal_instr, so tens of LUT at most;
    - core FF: 0 ± 300 (retiming noise only);
    - every other row inside the ±2,500 floor.
  - It enters the next bitstream, which is batched with step 8.
- **A constraint on step 8:** the trap-route kernel executes SHRINK (funct7 0x01) in S (sublet_store_user_ref). SHRINK
  is legal under Sublet (decoder.sv:1260-1265) and must survive the physical removal.

## Step 5: CLDRAW and CSTRAW, the design (2026-11-24, before any code)

The reference: capstone-qemu 9289114e06 op_helper.c helper_cldraw / helper_cstraw / raw_granule_probe,
trans_capstone.c.inc raw_moves_allowed, tests/capstone-raw-moves; dev docs/design/virtual-capstone/raw-tag-moves-proposal.md.
- **Encodings:** custom-0 (0x0b) funct3 4 (I: CLDRAW rd, imm(rs1)) and 5 (S: CSTRAW rs2, imm(rs1)). Today the decoder
  takes custom-0 only for debug.print (funct3 0); funct3 4, 5 and 6 are illegal (decoder.sv:506-520).
- **Privilege:** legal only in S or M, outside a supervised context (sup_active) and a virtual context (vctx). U,
  protected U and contexts take 2. The Sublet build only; ids55 keeps them illegal.
- **The address** is rs1's integer view plus imm, through the current privilege's ordinary translation (MPRV included).
  A raw move is a PLAIN access:
  - no capability check, even through a tagged base in S (rule 1 does not apply);
  - rule 2's S index applies, so with srevroot != 0 it reaches no user page;
  - 16-byte alignment, else 4 / 6 with tval = the address;
  - translation and PMP faults as for a load / store.
- **CLDRAW** writes rd with all 128 stored bits and the memory tag, without decoding: the integer = lo, the lane =
  {32'b0, raw_tag = the memory tag @65, tag 0 @64, hi @63:0}. rd's usable tag is 0, so it grants nothing. rs1 is not
  written.
- **CSTRAW** of a tagged rs2 (lane[64]) raises 24 before any store. Otherwise it stores lo = rs2's integer and hi =
  lane[63:0], with memory tag = lane[65]. It writes no register.

**Representation: raw_tag is lane[65] while lane[64] = 0.** This rests on the read/write enumeration of 87aec16b4 (an
Explore pass; the two load-bearing facts re-read here: ariane_pkg.sv:813-832 and :1007-1017):
- **Readers.** No reader interprets an untagged lane's bits [97:65]:
  - decompress_cap_tagged (ariane_pkg.sv:813-832) maps tag 0 to NOT_CAP with no slot, and every authority path goes
    through it (FLU, DYN, LSU check, commit's CCSRRW path);
  - the reads that skip the tag are gated later (bypass types :1641-1643, r60 :2602-2607, commit :597);
  - load_unit's rs1_perm_write (:240-242) is dead under Sublet (fmt_md_type returns only NONLIN or NOT_CAP).
  So a raw register reads as untagged everywhere: no check accepts it, and no capability is built from it.
- **Writers.** Every ordinary producer writes lane bits [97:65] = 0, so raw_tag is cleared with no new logic. This
  matches the reference's scalar_destroys_tag and movc_clears_raw:
  - ALU, branch, mul/div, loads, AMO, FPU moves and CSR reads: '0;
  - FLU / DYN: cap_lane with slot 0 for NOT_CAP;
  - MOVC of an untagged source: slot zeroed;
  - LDC: fmt_unpack_lane gives {0, 0, hi} for an untagged granule (ariane_pkg.sv:1009-1010);
  - the scratch CSRs keep a lane only with its tag (step 4);
  - the switcher's save (fmt_pack_hi ignores the slot when untagged).
  - Copies that keep the lane unchanged: forwarding, and the bypass STC's rd (= rs2) and rs1 write-back. These are the
    reference's "the register was not written", so raw_tag correctly survives them.
- **The hard part is identity, not the bit.** The untagged LDC / STC data path already is a raw transport minus the tag
  (an untagged LDC yields integer lo and lane hi; an untagged STC stores integer and hi, fmt_pack_lo/hi). So CLDRAW and
  CSTRAW are the bypass LDC / STC with three differences: no unpack even when the memory tag is set, raw_tag in and out,
  and no capability check. They need new op identities: LDRAW and STRAW, appended to fu_op in ariane_pkg.sv AND to
  capstone_unit.anvilh's twin enum (Anvil regenerates; no unit sees them). Every op-identity site the enumeration listed
  must then admit or exclude them explicitly:
  - check_cap_op / is_cap_load / is_cap_store (so the rs2 lane is read);
  - the LSU's capability-context gate and the MMU's cap_ref (both excluded: a plain access);
  - the lifetime verdict and the R43 probe (excluded);
  - load_unit's data_is_cap / data_wuser request (included) and its clear / linear logic (excluded);
  - store_unit's st_is_cap / st_ctag / st_user (included, with ctag = lane[65]);
  - commit's LDC rs1-lane rule (LDRAW writes rd only) and the STC write-back (STRAW writes nothing).
- **The likely synthesis surprise** the scope warned of, an extra bit through every forwarding path, does not arise:
  the bit rides in the existing 98-bit lane.

**Arms, ported from tests/capstone-raw-moves/probe.S** (the arm numbers are the reference's; each to be pre-registered
from the reference's own expectation):
- 1 tagged round trip republishes the tag;
- 3 / 4 U load and store: 2;
- 5 / 6 misaligned: 4 / 6;
- 9 M mode;
- 10 opaque authority: a raw register as a base in protected U raises 24;
- 11 a scalar write destroys raw_tag;
- 13 translated alias;
- 14 / 15 read-only load and store;
- 16 / 17 unmapped load and store;
- 18 x0 faults;
- 20 / 21 MOVC clears raw;
- plus a tagged CSTRAW source: 24, and a context: 2.
- 7 / 8 (MMIO: a tagged CSTRAW outside tagged RAM faults before changing bytes) needs the platform's taggable-memory
  map. The simulation has none (its NI rule is off without CAPSTONE_FPGA_NI); recorded as a silicon arm.

## Step 5: CLDRAW and CSTRAW, built and read (2026-11-24)

capstone-ariane b5b2-step5, on step 4 (eb03e7c99), not on the held CCSRRW deletion:
- fcd169dc5, the change and b5b2-raw.S, pre-registered;
- 39e83f3be, the decode gated on CAPSTONE_DYN_BYPASS as well;
- 220fa584c, fix 2 and b5b2-rawlcc.S, pre-registered;
- 4457a77a7, fix 2's port-1 index (lint).

The design is the section above, with one encoding detail settled while building. The raw ops present rs1 to the LSU as
an untagged lane, so lsu_cap_type is NOT_CAP: neither rule 1 nor the MMU's reference path sees them, with no new terms
there. CSTRAW's "rs2 is tagged" flag rides that lane's bit 65. LDRAW and STRAW are appended to fu_op under the Sublet
define (logic [7:0], width unchanged); Anvil's twin enum is untouched, because no unit receives them.

**Readings** (script-checked against the headers):
- **b5b2-raw.S** (37 readings, ported from tests/capstone-raw-moves/probe.S):
  - the base (step 4's core) reads all 37 as pre-registered;
  - fcd169dc5 read 35/37: readings 4 and 8 untagged where the reference keeps the tag;
  - 4457a77a7's code (220fa584c) reads all 37.
- **The divergence behind 4 and 8.** Every FLU/DYN instruction hands rs1 back at commit through cap_lane, which zeroes
  lane[97:65] of an untagged rs1, raw_tag included. The reference's LCC writes nothing to rs1.
  - Isolated by an uncommitted matched pair on fcd169dc5: an LCC of the raw register between cldraw and cstraw reads
    untagged (7), and the same LCC on another register reads tagged (1).
  - Forwarded and register-file paths, with or without a fence after the source's stc, kept the tag (4 more arms).
  - The semantics owner ruled it a defect to fix: no divergence from the reference.
- **Fix 2 (220fa584c),** the semantics owner's 1-bit variant:
  - rs1_tag_q in issue_read_operands, one bit per scoreboard entry, outside mem_q, written at issue from the issued rs1
    lane's tag;
  - with it clear, commit drops the rs1 write on both ports and the entry is no rs1 forwarding source;
  - the rs1 claim stays, so a reader waits for commit and reads the unchanged register. Retirement does not depend on
    the write.
  - b5b2-rawlcc.S (E: LCC of the raw register; F: E's control; G: MOVC with cstraw back to back, the forwarding gate; H:
    the same after a delay, the commit gate) reads 1 1 1 1 with the fix and 7 1 7 7 on the pre-fix core, both as
    pre-registered.
- **Regression:** the other 27 tests (10 B5b-2 arms, 15 Sublet tests, the 2 context arms) are identical in value and
  cycles to step 4's runs: on fcd169dc5, and again on 220fa584c.
- **Static checks on 4457a77a7:**
  - the Sublet lint gate passes; by name against step 4 it is 0 lines in both builds; ids55 is token-identical (359,790
    tokens);
  - 220fa584c had failed the gate on SELRANGE (32 -> 33, a [1] index in the dead port-1 block), fixed by 4457a77a7.

**claim-auditor** (fcd169dc5): no functional defect in six areas: the request lane's consumers, the LDC-only paths, the
CLDRAW lane's index alignment, CSTRAW's store and its 24, decode and issue, and loop-list terms. Items:
- **Refuted claim,** withdrawn on the lane: "every other build keeps them illegal" (fcd169dc5's message). The decode
  was gated on the raw-lane define alone, which stays on in the CAPSTONE_NO_DYN_BYPASS control, where the raw ops would
  also have reached the DYN unit. 39e83f3be gates it on the bypass too.
- **Timing:** one more gate level on the request lane's tag select in front of cap_exception, the LSU's dominant slack
  term. The full build reads it. If it costs, the zeroing can move into issue_read_operands.
- **Sim-only gap:** the S-07 retire-without-lane $error checks cover LDC/STC only.

**Divergences on the list** (not step 5's to fix):
- A PMP-denied access reads the access fault (5 / 7) where the reference raises 24 for a tagged CSTRAW source and 4 / 6
  for a misaligned raw move. This is pre-existing and shared with plain LDC/STC.
  - Cause: pmp_data_if.sv:374-390 overwrites any pending exception when PMP denies.
  - Sized fix, capability causes only (24-28 before PMP): one condition on that overwrite.
  - Misalignment before PMP for the raw moves and LDC/STC as well: the same line, plus an op scope or the change of
    plain-access priority (the priv spec allows either), which then needs its own sweep.
  - Ruling pending with the semantics owner.
- A CSRUNV context's save and restore moves sscratch as an integer (step 4, residual d).

### Synthesis-only gate on step 5: pre-registration (written before the request)

Hash 4457a77a7 (b5b2-step5: fcd169dc5, 39e83f3be, 220fa584c, 4457a77a7, on eb03e7c99). Base: step 4's gate
(b5b2s4-eb03e7c99-gate-out, dcp sha16 b05ca47912c75223). The CCSRRW deletion is NOT in it: it is held for the boot-set
scan.

| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within +/-505 MB of 5,066.5 MB, under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (the toy fires), LUTLP-1 0 (the toy fires) |
| 3 core LUT | at most the base + 2,500 (145,169 -> 147,669) |

Reported, graded nothing (bands at the ±2,500 floor):
- core FF: predicted about +16 (byp_raw_q and rs1_tag_q, NR_SB_ENTRIES each), ±300;
- lsu_i (the request's lane mux, the STRAW 24 arm and the raw-op decodes in front of cap_exception), load_unit and
  store_unit: a few hundred LUT. The auditor named the cap_exception cone the LSU's dominant slack term; this gate
  cannot read slack, so the full build reads it;
- dyn_unit + rev_node + i_scoreboard graded as one SUM (the census: the scoreboard-entry cone moves between the first
  two), against the base's 15,670 + 3,241 + 17,244;
- issue_read_operands and commit_stage: the rs1_tag gate, tens of LUT.
The semantics owner's branches, fixed before the run:
- PASS: fix 2 stays.
- Loop growth (key 2) or core LUT beyond key 3: fall back to (b). Fix 2 comes out; the divergence (an FLU/DYN
  instruction with a raw rs1 drops raw_tag) is recorded; b5b2-rawlcc stays a committed arm reading 7 1 7 7; (a)
  becomes a later item.

### The gate on step 5: read (board lane, 2026-11-24), PASS 3/3

4457a77a7, graded against a37873a76ed7. Base: step 4's gate. Grade file
`/mnt/hdd/<user>/synth/b5b2s5-4457a77a7-gate-out/grade-b5b2s5.txt`.
- Key 1 (memory): TM peak 5,244.6 MB, against 5,066.5 (+178.2; ±505). PASS.
- Key 2 (loops): 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy 1), LUTLP-1 0 (toy 1); the in-flight
  loops-only gate PASS.
- Key 3 (core LUT): 146,277, against 145,169 (+1,108; limit 147,669). PASS.
- **The semantics owner's branch:** neither fallback condition holds, so fix 2 stays.
- Reported:
  - **Core FF 55,396 (+426): OUTSIDE my +16 ±300.** ex_stage +361, csr +37, commit +15, scoreboard +10. Step 5 added
    only byp_raw_q (8 flops) to ex_stage; the rest is unattributed. A census is asked.
  - **dyn_unit + rev_node + i_scoreboard summed:** 33,165 (-2,990; ±2,500: OUTSIDE), while i_issue_read_operands rose
    3,916 (30,527). With IRO added the four read 63,692 against 62,766 (+926). So the rule written after step 4's
    census (grade the pair as a sum with i_scoreboard) was too narrow: the scoreboard-entry cone can also land in IRO,
    where fix 2's own logic sits too. The census on this dcp separates the two (cone stems by module; fix 2's cells).
    Until it reads, the sum is graded with IRO included.
  - lsu_i 25,495 (+1,498), i_load_unit 716 (+15), i_store_unit 5,349 (+407), commit_stage_i 278 (-8); issue_stage_i
    48,186 (+4,331), id_stage_i -521, csr_regfile_i 7,269 (-222), ex_stage_i 57,706 (-2,157).
  - Guard peak RSS 30.24 GB (step 4: 24.59), under the 40 GB ceiling; 52m43s. The dcp is kept as
    twc/gate-synth-b5b2s5-4457a77a7.dcp (sha16 88f79e9f72d2c792).

## Step 6 part 1: CPTAG's ordering and the page bitmap (2026-11-24)

**Correction to the step 6 design above** (its "binding condition" bullet, written before the adapter was read). The TWC
never holds a dirty tag. It is a read cache whose valid entries show what DRAM will hold (wt_axi_adapter.sv:155-156),
and every tagged store issues its own AXI tag write. A tag can be newer than DRAM in six places:
- the store buffer;
- the dcache wbuffer;
- the adapter's 2-deep request FIFO;
- the latched TAG_FILL / TAG_WR (tag_wr_value_q);
- writes issued with B pending (the twl, tag_wr_pend).
- The TWC is never ahead.
The concrete false-negative hazard: a page's bitmap bit is set only at the END of its TAG_FILL, while the data write's
B returns earlier. So the condition is ORDERING behind every older tag write, not draining a cache.
Side note (not this step's): ariane_pkg.sv:663's comment says tag = base + (paddr >> 4); the RTL subtracts DATA_MEM_BASE
first (wt_axi_adapter.sv:271-273).

**The semantics owner's rulings:**
- 6b, in two gated parts: part 1 is the ordering + bitmap; part 2 is the acknowledge's exact OR, with its own gate and
  a fallback to part 1 alone.
- The AMO port as the carrier.
- The core half as a CSR-class op (no translation, no PMP: rs1 is physical).
- The region: the reference's rule (page-aligned, wholly inside the tagged region, cap_mem_map.c:504-522) on silicon's
  values [0x8000_0000, 0xBC2D_2D2D), so the last page inside is 0xBC2D_1000. A QEMU machine's region is its main RAM
  (cpu.c:1520-1539, -m), recorded as a machine-configuration difference.

**Built:** capstone-ariane b5b2-step6 eff52e152, on step 5 (4457a77a7).
- **Decode:** custom-0 funct3 6, I, mode = the immediate. Others: 2. U, protected U, contexts: 2.
- **Execute:** fu CSR, op CPTAG; the result is rs1 with the mode in bit 0; the fault is 5 with tval = rs1.
- **Commit:** as an AMO (it waits for amo_resp, which carries rd; the CSR buffer is freed on the ack; no CSR is touched).
- **LSU:** CPTAG on the AMO port in place of the store unit's AMO buffer, sent once no store is pending; the missunit
  then waits for an empty wbuffer.
- **Adapter:** at the FIFO head, no AXI transaction and no invalidation. Granted when the tag FSM is idle and
  tag_wr_pend is 0 and the bitmap lookup is valid; the bitmap bit is latched and returned as an ATOMIC_ACK on the
  lowest-priority return slot.
- **Lint on the way:** two false Verilator loops through the missunit's mem_data_o (a standing member), both fixed: a
  select derived from amo_valid_commit, and a whole-struct amo_req assignment.

**Readings** (b5b2-cptag.S, pre-registered in its header; script-checked):
- 23/23 on both trees.
- In the .iss: 7 CPTAG-ANSWER lines with the predicted pages and bits.
- 48 CPTAG-WAIT cycles (tag_wr_pend 1) ending one cycle before pg_tag's first answer: the hazard window was exercised.
- The base logged none.
- 29 regression tests are identical to step 5's run.
- **My slip:** the first count read 0 / 0 from the .log; the sim's displays go to the .iss.
- Lint: the Sublet gate passes; 0 lines by name against step 5 in both builds; ids55 is token-identical.

## Step 6 part 2: CPTAG's acknowledge (2026-11-24)

The semantics owner's ruling (ii), against (i) "OR on every query" (a divergence: after the tags are cleared, a
non-ack reads 0 where the reference reads 1 until an ack) and (iii) a second sticky bitmap: an acknowledge that finds
the page empty clears its bitmap bit and invalidates the page's TWC entries, and a non-ack answers the bit.

capstone-ariane b5b2-step6 8f83093fc, on part 1 (adapter only):
- **The walk.** An ack on a marked page reads the page's tag words one at a time (TAG_QRD / TAG_QWT) on AXI read ID 2
  (icache 0, dcache 1, LR 3). That is 33 words, 32 if the run is word-aligned: the 256 tag bytes start at byte 5 of a
  word since CAP_TAG_MEM_BASE ends in ...2D, as TAG_FILL's comment at :889-892 counts. The first and last words are
  masked exactly as TAG_FILL's byte enables.
- **An empty page:** the last beat clears the page's bit (the bitmap's one write port, read-modify-write of the head's
  word) and bumps the TWC epoch, as an untargeted fill does. That restores the invariant at :183-184.
- **The answer:** TAG_QDONE pops the head and latches the result for part 1's return slot.
- **A tag read of a page whose bit is 0** gets no tag beat (zero without fetching, :1149). After a clear, the next
  tagged store re-marks the page (TAG_FILL) before any later read.
- **Lint:** the Sublet gate passes; 0 lines by name against part 1 in both builds.

**Readings** (b5b2-cptag2.S, the semantics owner's seven arms in order, pre-registered; script-checked):
- 12/12 on part 2: 1 | 1 7 | 0 0 | 1 1 | 0 1 | 1 1 | 0.
- 12/12 on part 1's adapter, where an ack answers the sticky bit.
- In the .iss: exactly 3 walks (results 0, 0, 1) and 2 clears, both pg_a.
- Regression: 29 tests identical in value. b5b2-cptag takes +1,119 cycles: its one acknowledge on a marked page now
  walks (1 walk against 0 on part 1).

**claim-auditor** (over eff52e152 and 8f83093fc):
- **Confirmed:** no false 0 (every older tag write is complete when the CPTAG is answered or walks: store buffer, then
  wbuffer, then the adapter's tag FSM and tag_wr_pend); ID 2's beats match no other decode; the clear's
  read-modify-write and epoch bump are sound; no CSR side effect.
- **Fixed in 0c9992bc6:** the clear is registered (TAG_QDONE, from the walk's result), so no path runs from the R data
  to tbm_mem's write enable or the TWC epoch; and the F2 invariant's text. Same readings and cycle counts after the fix.
- **Inherited AMO hazard, unreachable under Sublet:** a PC-capability fault on an AMO-class head leaves
  amo_valid_commit set while the head is flushed (the missunit would wait in AMO_WAIT forever). pc_cap_ex_valid
  requires priv M and capmode (commit_stage.sv:423), and capmode is never set under the Sublet format. Not changed.
- **A false 1, never a false 0:** an AMO-residual tag byte (S-06 P6, I4), or a software write into a marked page's tag
  bytes, makes the ack answer 1 and keep the bit where the reference answers 0. This is contract-safe, recorded as a
  divergence.

### Synthesis-only gates on step 6: pre-registration (written before the requests)

Two gates, one per part, so each part's cost is attributed on its own (the semantics owner's ruling):
- **Gate 6a:** part 1, eff52e152, against step 5's gate (b5b2s5-4457a77a7-gate-out, dcp sha16 88f79e9f72d2c792).
- **Gate 6b:** part 2, 0c9992bc6 (8f83093fc + the claim-auditor's fixes), against gate 6a.

| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within ±505 MB of the base's (6a: 5,244.6 MB), under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (the toy fires), LUTLP-1 0 (the toy fires) |
| 3 core LUT | at most the base + 2,500 (6a: 146,277 -> 148,777; 6b: gate 6a's + 2,500) |

Reported, graded nothing (bands at the ±2,500 floor; the scoreboard-entry cone is read as one sum of dyn_unit,
rev_node, i_scoreboard and i_issue_read_operands):
- **6a:** i_axi_adapter (wt_axi_adapter) a few hundred LUT and about 40 FF (the CPTAG answer and tid registers);
  commit_stage, lsu_i and ex_stage tens of LUT; the page bitmap still 8 RAMB36 (its write port gains no source in 6a).
- **6b:** i_axi_adapter a few hundred LUT more and about 75 FF (the walk's word address, count, offset and result);
  tbm_mem still BRAM with a third write source (the claim-auditor's point: one write address, one read address),
  RAMB36 unchanged.
- **Timing:** these gates read no slack. The full build reads the walk's beat-to-register path and the LSU's
  cap_exception cone (step 5).
- **The semantics owner's branch for 6b, fixed now:** loop growth (key 2) on the missunit mem_data_o channel (a
  standing member), or core LUT past key 3, ships part 1 alone. The acknowledge's divergence (an ack never clears
  the bit, so a page that once held a tag reads 1 until reset) is then recorded with b5b2-cptag2's arms as its
  evidence, and part 2 becomes the next item.
- Gate 6a has no fallback of its own beyond the keys: a failure there means CPTAG does not enter the bitstream in this
  round.

**Step 5's census** (board lane, 88f79e9f72d2c792 against b05ca47912c75223):
- The scoreboard-entry cone's cap_result part moved dyn_unit -> i_issue_read_operands, field for field. The
  result/imm pair looks like the same cone renamed; that is a reading of names, not a measurement.
- Fix 2, measured directly: 8 FF (rs1_tag_q) and 19 LUT.
- The core FF miss is retiming register replication: FPU _bret/_fret (net +300) and dyn_child_q[base] (+74). The only
  RTL flops are byp_raw_q's 8.
- From step 5 on, the cone is graded as one sum of dyn_unit, rev_node, i_scoreboard and i_issue_read_operands.

### Gate 6a (step 6 part 1): read (board lane, 2026-11-24), PASS 3/3

eff52e152, graded against 03902b250094. Base: step 5's gate.
- Key 1 (memory): TM peak 5,063.9 MB, against 5,244.6 (-180.7; ±505). PASS.
- Key 2 (loops): 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy 1), LUTLP-1 0 (toy 1); in-flight PASS.
- Key 3 (core LUT): 147,798, against 146,277 (+1,521; limit 148,777). PASS.
- Reported:
  - i_axi_adapter +85 LUT / +5 FF, RAMB36 48 (+0); tbm_mem's BRAM row identical (4K x 64). This is under my
    prediction (a few hundred LUT, about 40 FF).
  - The cone sum is 63,428 (-264, inside).
  - ex_stage_i +1,102 (lsu_i +565), i_load_unit +350, i_store_unit +406, commit_stage_i +60: all inside the floor.
    Part 1 touches none of those except lsu_i's amo_req mux, so this is redistribution, not attributed.
  - Core FF +112; no row beyond ±2,500.
  - Guard peak 27.46 GB; dcp sha16 a82acbbe85801608.

### Gate 6b (step 6 part 2): read (board lane, 2026-11-24), PASS 3/3; part 2 is kept

0c9992bc6, graded against gate 6a (dcp sha16 a82acbbe85801608).
- Key 1 (memory): TM peak 5,063.9 MB, against 5,063.9 (-0.0; ±505); whole-run peak 5,132.0 MB. PASS.
- Key 2 (loops): 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy 1), LUTLP-1 0 (toy 1); in-flight PASS.
  No loop growth on the missunit mem_data_o channel or anywhere else.
- Key 3 (core LUT): 147,893, against 147,798 (+95; limit 150,298). PASS.
- **The semantics owner's branch did not fire** (no loop growth, core LUT far inside key 3), so part 2 stays and CPTAG
  enters the bitstream whole.
- Reported:
  - i_axi_adapter +94 LUT / +80 FF: at the prediction (a few hundred LUT, about 75 FF).
  - **RAMB36 +2 (top 115, core 100, all in i_axi_adapter): a MISSED prediction** ("RAMB36 unchanged"). Board lane's
    per-memory count on both dcps: the whole +2 is twc_ep_mem (2 -> 4 blocks); tbm_mem, twc_atag and twc_data are
    identical. The port shapes are identical too (TDP 16K x 2, write A, read B, 14 address pins each); what changed is
    live data bits: on 6a one written epoch bit was live, on 6b all 8 are written and read.
    - **The cause, checked at source:** under CAPSTONE_TWC_FILL_TARGETED (on by default, config_pkg :62-64) a fill
      invalidates its two shared words instead of bumping the epoch (wt_axi_adapter.sv:548-552). So up to 6a nothing
      ever bumped twc_ep_q; 6a's .vds propagates it as a constant (Synth 8-3333). Part 2's clear bumps it
      (:535-543, Sublet only), so it is a live 8-bit counter again, and the memory keeps its bits.
    - **So the miss was in my prediction, not in the RTL:** I counted the epoch memory as already live.
    - **Behaviour re-armed with it:** every 255th clear wraps the epoch and runs the 16K-entry scrub (every lookup
      misses while it runs). A clear arriving during a scrub is dropped, which is safe: the scrub zeroes every entry.
      Clear and fill-enter cannot coincide (TAG_QDONE vs TAG_IDLE).
  - The cone sum 63,428, +0 in each of its four modules; ex_stage_i +1, lsu_i +1, i_store_unit +1, commit_stage_i +0;
    core FF 55,589 (+81); top LUT 174,903 (-116); no row beyond ±2,500; LUTRAM +0.
  - Guard peak 27.46 GB; dcp sha16 3986f0ae821b9832.

## B5b-2's full build: pre-registration (written before the request, 2026-11-24)

- **Hash:** capstone-ariane b5b2-step6 0c0458149: steps 1-6 batched (CPTAG whole), plus step 7's test and its
  sim-only `ifndef NS_SIM_NOFLUSH`, which leaves the synthesised code unchanged with the macro undefined. So the
  synthesised RTL is gate 6b's (0c9992bc6). The CCSRRW deletion (87aec16b4) is NOT in it: it is held for the boot-set
  scan.
- **Simulation evidence on the hash:**
  - every step's arms, as recorded above;
  - the 29-test regression, identical in value to step 5's run;
  - step 7's must-pass, both directions with its mutant control;
  - lint at REF on the Sublet build (LATCH 44, MULTIDRIVEN 3, UNOPTFLAT 39 with the 39 names, UNDRIVEN 25,
    UNUSEDSIGNAL 1567, SELRANGE 100), 0 lines by name against step 6a;
  - the ids55 build token-identical to B5b-1's base at 0c0458149 (359,790 tokens), so B5b-1's ids55 95-test sweep
    carries.
- **The base:** B5b-1's seal on d0afdb40f (sha256 38491cf4...). Readings: placed LUT 170,699, FF 77,561; routed clean;
  placed congestion 4 with no router hotspot; check_timing, LUTLP-1 and TIMING-23 all 0; core clock clk_out1
  WNS +1.851, design-wide +0.138 (a MIG path).
- **The synthesis delta behind the area prediction:** d0afdb40f's gate (core LUT 145,492, FF 55,117) to gate 6b
  (147,893, 55,589) is +2,401 LUT and +472 FF.

| key | prediction | the failure branch |
|---|---|---|
| placed LUT | 173,100 ± 3,000 | outside: per-module placed rows against B5b-1's seal before anything else |
| placed FF | 78,033 ± 2,000 | the same |
| routed | 0 errors, 0 unrouted, 0 partial | a routing failure: placed area, congestion windows by module, the census; NO WNS is recorded |
| congestion | <= 5 (placed and router) | above: the windows' modules, against B5b-1's |
| loops | check_timing 0, LUTLP-1 0, TIMING-23 0, each with its toy control firing | any: members by stem |
| WNS (clk_out1) | inside B5b-1's null band, [-1.89, +5.59] (+1.851 ± 3.74) | outside on the BAD side: worst-path family and per-stage depth before any board use; outside on the good side: reported, not graded |

**Reported, not graded:**
- design-wide WNS, TNS, WHS, failing endpoints, slices;
- RAMB36 against B5b-1's seal (gate 6b's +2 in twc_ep_mem should carry);
- the worst-20 family on clk_out1 (B5b-1's: i_issue_read_operands fu_data_q, 103 levels);
- the three timing points the gates could not read, each as the worst clk_out1 path through it with a -through
  positive control:
  - the LSU's cap_exception cone (step 3's reorder, step 5's request-lane tag select);
  - the CPTAG walk's beat-to-register path (cptag_* in i_axi_adapter);
  - the rs1_tag_q gate into commit's cap_we, and the csr_scr_* lane (step 4);
- the sha256.

Flash is the delegated bitstream rule, agreed with runtime. B5b-1's seal stays the default until B5b-2's bare kits
(below) read. Each kit is pre-registered from image simulations of THIS hash before its boot.

## B5b-2's full build: read (board lane, 2026-11-24), PASS 10/10, and the core clock meets timing

Graded from the board lane's `grade-board/grade-b5b2-full.txt`, read here; the seal was re-hashed here.
- **Seal:** `…-pagedtbl-r76-r75d-b5b2_0c0458149.bit`, 11,443,722 B, sha256
  b44291bacb220b8ae6dd02ff113cb68bcfff231893fae70741261c087bb53b9a. Guard exit 0, 1h09m, peak 27.48 GB.
- **The keys:**
  - placed LUT 172,682 (-418 on 173,100) and FF 78,043 (+10 on 78,033); B5b-1 170,699 / 77,561;
  - routed clean (227,293 / 227,293);
  - placed congestion 4; the router reported no hotspot (Phase 2 ran; R-75 (d)'s log parses to 5 as control);
  - check_timing 0 in the build report and on the routed dcp, TIMING-23 0, LUTLP-1 0 (each toy 1);
  - **clk_out1 WNS +0.286**, in the band (B5b-1 +1.851, so -1.565). The core clock still meets timing.
- **Reported:** design WNS +0.177, TNS 0, 0 of 237,785 endpoints failing, WHS +0.056; slices 48,746 (-289); RAMB36
  115 (+2, gate 6b's twc_ep_mem, as expected).
- **The worst 20** start at i_issue_read_operands fu_data_q[0].cap_metadata_b[1] (one at b[26]_rep) and end in
  fu_data_q[0] cap_metadata_a / b / rd_metadata, at 90-95 levels: B5b-1's family (b[14] -> a[17], 103 levels).
- **The points the gates could not read** (worst clk_out1 path through each; B5b-1 in brackets):
  - the LSU cap_exception cone +11.776 at 33 levels [+13.535];
  - CPTAG's walk +24.281;
  - commit's cap_we +11.445 [+11.561];
  - csr_scr_* +24.903;
  - rs1_tag_q +25.326, and rs1_tag_q through cap_we +28.648.
  None binds.
- Flash: the delegated rule, after the bare kit's pre-registration (below) is pushed.

## Step 7: the namespace must-pass (2026-11-24)

capstone-ariane b5b2-step6 eb8e32f08 (the arm and its mutant control), 0c0458149 (the arm's fix after its void first
mutant run). The semantics owner accepted the design and its four additions.
- **va-rc-trap / -zp are ids55-only arms.** They build P1-format tables (va-fetchsplit.S :736-756: a flat RECYCLE
  header with no PAGED flag; record 2's word 1 = 1 << 60, "valid at generation 0"). The Sublet unit reads word 1 as
  {generation@63:32, prev, next} in a paged table (capstone_sublet_life.sv:10-12). So under Sublet the context entry
  is refused: the unit answers status 1 for node 2, and S's CSRUNV at 0x8000101c traps with mcause 0x1d.
  - The printed vector 0x2 / 0x80003000 / 0x1d is the legacy-C control, a3 (left over from the MINT macro's
    `lla a3, LO`, not a tval), and that mcause.
  - B5b-1's and B5b-2's Sublet tips read it identically (81,216 / 81,323 cycles). The ids55 build of B5b-2 reproduces
    the P1 record exactly (93,870 / 94,094 cycles, two R75-BLK events).
  - Not traced: which switcher arm maps the unit's status 1 to cause 29.
- **b5b2-nsedge.S, the must-pass for the consumers outside M:**
  - A is live in T1 and stale in T2; B is stale in T1 and live in T2. T2 is a repointed copy of T1's paged table.
  - Each srevroot write is immediately followed by the access: S under rule 1 with ld and ldc, both directions, and
    protected U with the switch in S right before the sret.
  - The mutant NS_SIM_NOFLUSH is a sim-only `ifndef` around the LSU's verdict-cache flush; with the macro undefined the
    code is unchanged.
  - Readings, script-checked against the header: the build is 16/16; the mutant is 16/16 against its re-registered
    vector. All six discriminating readings move (live to stale 2, 6, 13: 25 -> 0; stale to live 9, 11, 15: 0 -> 25).
  - **First mutant run VOID for 9, 11, 15:** B's ALLOW, cached at its cderive in the pool, was never flushed, so B
    never took a DEAD verdict. Fixed by ending B in the pool first (CREVOKE); the build's readings did not change.
- **Fetch is not covered, by design on both sides.** Under Sublet no fetch check consults the namespace: both
  PC-capability checks need M and capmode (commit_stage.sv:423 pc_cap_check, :518 pcc_chk_o), and capmode is set only
  by the illegal CAPENTER. The reference checks protected U's data accesses only (translate.c:130-132;
  op_helper.c:1387-1390).

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

**The kit: `capstone/tests/rtl-smoke/b5b2-bare-2026-11-24`, pre-registered from image sims of 0c0458149.**
- **Images:** 16. The Sublet control (sublet-life) runs first, then every B5b-2 test configuration in
  testlist_b5b2.yaml: illegal (and -runv), uprot (and -mis), rule2 (and -pair), sref (and -gen), scratch, mwrite,
  raw, rawlcc, cptag, cptag2, nsedge. They are the in-tree tests, unchanged.
- **Every vector equals its in-tree run except one position:** b5b2-raw reading 15 compares a raw word 0 (which carries
  bounds << 38) with a projected plain ld, so it depends on the link address. That is explained in the PREREG.
- **Two kit defects, found in the first image-sim pass:**
  - the B5b-2 tests end in RVTEST_PASS, a tohost store that prints nothing on the board; the kit maps it to the UART
    report;
  - clang expanded macro names in a test's `#` comments; the kit builds with the tree's gcc and lets each test include
    the board header itself.
- **Checks on every image:**
  - the exact UART character count, 59 + 21·SR (the same check read 25 for every image before the fix, so it fires);
  - a plain-store recorder for b5b2-sref, cross-checked in simulation on b5b2-illegal;
  - a scan for loads of uninitialised DRAM: 0 in every B5b-2 image. It flags sublet-life's pool dump as its control,
    and those positions (25-34) are ungraded, as in csmint-bare's amendment.

## B5b-2 on silicon: CONFIRMED (board 1, N=1, 2026-11-24)

The kit b5b2-bare-2026-11-24 on the seal `…-r75d-b5b2_0c0458149.bit`, board 1 (booking 1488; board/meas-pw
8661f733bdae). **15/15 AS PREDICTED, with the control exact.**
- Steps 1-7 read on silicon as in simulation: the deletions, protected U and S rule 1, rule 2 in the MMU, supervisor
  references, tagged sscratch/mscratch, CLDRAW/CSTRAW and the rs1-tag gate, CPTAG's answer and its acknowledge walk,
  and the namespace edge.
- Graded twice from the raw UART: by the board lane's grade-b5b2kit.py and by a separate script here, which reads the
  PREREG's JSON from git and whose negatives fired before the run.
- The flash checked the seal's local sha256; the resident name was read back after the power cycle and after the loads.
- **Which bitstream is the default stays unchanged** (R-75 (d), fallback bitstream 1). B5b-2's seal runs bare kits
  only: CAPENTER is illegal under the Sublet format, and the current monitor executes it at init. It becomes a
  default candidate with the integer-M monitor.

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
