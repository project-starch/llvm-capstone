# Capstone ISA semantics: what the compiler assumes, against spec, QEMU and RTL

One row per instruction the backend can emit (plus the pseudos that expand to them). The
"compiler" column is what `CapstoneInstrInfo.td` and the selectors actually encode, with the
line that encodes it. The spec, QEMU and RTL columns hold only what has been read at a quoted
line or shown by a test this validation ran; anything else says **not audited** rather than
guessing. A cell that reads "agrees" without a citation is a bug in this document.

Sources: spec `capstone/capstone-spec/parts/cap-man-insn.adoc` (line numbers as of 2026-09-04);
QEMU `capstone/capstone-qemu/target/riscv/op_helper.c`, `cap.h`, `cpu_bits.h`, `insn32.decode`,
`insn_trans/trans_capstone.c.inc`; RTL `capstone/capstone-ariane/core/anvil_build/
capstone_{flu,dyn}_unit.anvil`, `capstone_unit.anvilh`, `core/decoder.sv`,
`core/issue_read_operands.sv`, `core/commit_stage.sv`, `core/load_unit.sv`. The RTL and QEMU
cells were filled from the rtl-oracle pass of 2026-09-04 (quoted lines re-checked where they
drive a compiler change). Tests: `llvm/test/CodeGen/Capstone/`, `llvm/test/MC/Capstone/`,
`capstone-ariane/verif/tests/custom/capstone/`.

**Exception numbering, which applies to every fault cell below.** RTL delivers `spec code + 1`
for all six capability exceptions (`capstone_unit.anvilh:289-311`: `mcause = 24 +
exception_code`, so UNEXPECTED_OPERAND is 25, INVALID_CAPABILITY 26, UNEXPECTED_CAP_TYPE 27,
INSUFFICIENT_PERMISSION 28, OUT_OF_BOUNDS 29, ILLEGAL_OPERAND_VALUE 30; confirmed by
`cincoffset-stale-metadata.S:78-80`). QEMU matches the spec (`cpu_bits.h:692-698`: 24..29).
Every `.S` expectation therefore names the exception, never the number (R-24).

Columns: **consumes LINEAR rs1** = the source register is nulled when it holds a LINEAR
capability; **untagged / wrong type** = what happens when the operand is not a capability, or a
capability of a type the instruction does not accept; **result** = the value type produced.

## Memory

| insn | compiler assumption (file:line) | spec | QEMU | RTL | consumes LINEAR | untagged / wrong type | result | settled by |
|---|---|---|---|---|---|---|---|---|
| `ldc rd, imm(rs1)` | `mayLoad`, no side effects; selected only by `selectLDC_STC` (never a pattern), which picks LDC vs LD/SD/SW/SH/SB by memory VT (`CapstoneInstrInfo.td:2420-2431`). **Assumes a reload does not modify memory**: two reloads of one spill slot on a path are treated as equal. | not audited | **no linear-slot semantics**: `trans_capstone.c.inc:146-169` and `helper_reg_set_cap_compressed` (`op_helper.c:1423-1444`) uncompress the word into rd and write nothing back; the only tag-clearing path (`helper_remove_cap_mem_map`) is called from scalar stores alone | **a load of a LINEAR/REVOKE/UNINIT/SEALED/SEALEDRET value CLEARS the source granule** (`load_unit.sv:222-229` `ldc_clear_needed`, `:499-502`), gated by `check_load_data` (`capstone_unit.anvilh:583-609`, INSUFFICIENT_PERMISSION if rs1 lacks write authority). Untagged (S-06) fixed and sim-validated. | the loaded value's memory copy: RTL yes, QEMU no | rs1 untagged: RTL `capstone_dyn_unit.anvil:327-330` UNEXPECTED_OPERAND; QEMU `op_helper.c:1326-1330` raises UNEXP_OP_TYPE. rs1 not LINEAR/NONLIN/SEALEDRET: RTL `:332-337` UNEXPECTED_CAP_TYPE; QEMU raises UNEXP_CAP_TYPE for a load through UNINIT (`:1356-1360`, QEMU-only patch) | c128 | **DIVERGENT**: under QEMU a second `ldc` of the same slot re-reads a LINEAR value that silicon has already erased. Compiled code is safe only under the Tier 4.1 contract (spilled values are NONLIN); the linearity verifier enforces it. `cap-valid.s`, `cap-instructions.txt` |
| `stc rs2, imm(rs1)` | `mayStore`; same selector (`:2425-2427`). **Assumes rs2 survives the store** (a spill is a copy). | not audited | **rs2 never touched** (`trans_capstone.c.inc:172-193`: compress, store, set mem map; no helper nulls rs2) | **rs2 is NULLED after the store when it holds LINEAR/UNINIT/SEALED/SEALEDRET/REVOKE** (`capstone_dyn_unit.anvil:439-446,458-465`); NOT_CAP and NONLIN left alone | rs2: RTL yes, QEMU no | rs1 untagged: RTL `:392-395` UNEXPECTED_OPERAND; QEMU as for ldc. rs1 not LINEAR/NONLIN/UNINIT/SEALEDRET: RTL `:396-400` UNEXPECTED_CAP_TYPE | -- | **DIVERGENT**, mirror of ldc: a spilled LINEAR register reads null on silicon after the spill and stays live under QEMU. Same contract, same verifier. |

## Cursor and bounds arithmetic

| insn | compiler assumption (file:line) | spec | QEMU | RTL | consumes LINEAR | untagged / wrong type | result | settled by |
|---|---|---|---|---|---|---|---|---|
| `cincoffset rd, rs1, rs2` / `cincoffsetimm` | pure, rs1 **not** consumed (`:2437-2450`): may be CSE'd, hoisted, multi-used. A null rs1 never reaches it from a GEP once C-40 is fixed (`selectCIncOffset` materialises a null-based address as an integer). **rs2 is assumed to be a plain integer**, but a `ptrtoint` result is a bare sub-register read of the capability register (C-31, `c31-cincoffset-rs2-from-extract.ll`: at -O2 `q + (long)p` is `cincoffset a0, a1, a0` with a0 the untouched capability). | consumes: `cap-man-insn.adoc:36-38` "if x[rs1] is not a non-linear capability ... write cnull to x[rs1]" | consumes (`op_helper.c:739-744`); untagged rs1: raises UNEXP_OP_TYPE (`:726-728`, observed on the C-40 images); UNINIT/SEALED: UNEXP_CAP_TYPE (`:731-734`). **Never checks rs2's tag** (`:736-737` uses the cursor if tagged). `cincoffsetimm` has a disabled-by-default escape `CAPSTONE_CINC_UNTAGGED_SURVIVE` (`:760-776`) | consumes (`capstone_flu_unit.anvil:45-49`, in the flashed bitstream); untagged rs1 or **tagged rs2**: UNEXPECTED_OPERAND (`:30-31`); UNINIT/SEALED: UNEXPECTED_CAP_TYPE (`:32-34`). `x0` is hard-zero in the metadata regfile (`ariane_regfile_ff.sv:96-99`), so `cincoffset rd, zero, rs2` always traps | all three yes; compiler assumes no (contract: only NONLIN reaches it) | rs2 tagged: **RTL traps, QEMU proceeds** -- C-31, reachable from ordinary C | c128 | `c40-null-base-cincoffset.ll`, `c31-cincoffset-rs2-from-extract.ll` (XFAIL); `twins/findings/C40-*` |
| `scc rd, rs1, rs2` | pure (`:2540-2543`); used by the gp-free glue and `cap_set_cursor` | consumes (`cap-man-insn.adoc:140-143`) | **`assert(rs1_v->tag && !rs2_v->tag)` (`op_helper.c:798`) and type asserts (`:800-801`): an emulator ABORT, not a guest fault**; consumes (`:805-810`) | consumes (`capstone_flu_unit.anvil:109-113`); untagged rs1 or tagged rs2: UNEXPECTED_OPERAND (`:93-94`); UNINIT/SEALED: UNEXPECTED_CAP_TYPE (`:97-98`) | yes on all three | as cincoffset for rs2 (C-31 applies) | c128 | QEMU fidelity item (assert) |
| `shrink rd, rs1, rs2` (rd tied) | pure, tied (`:2490-2494`); emitted per alloca/global under `-capstone-shrink-*` (default on). rs1/rs2 are integers computed from the object's own address, so C-31 applies if either is a bare sub-register read. **Zero-size objects emit a SHRINK with rs1 == rs2**, which is ILLEGAL on all three | `cap-man-insn.adoc:227-229`: `x[rs1] >= x[rs2]` illegal; `x[rs1] < base` or `x[rs2] > end` illegal (new end may equal old end) | raises UNEXP_OP_TYPE for untagged rd or a tagged rs1/rs2 (`op_helper.c:1022-1027`); type (`:1030-1036`) and bounds (`:1045-1046`) checks raise | UNEXPECTED_OPERAND for untagged rd or tagged rs1/rs2 (`capstone_flu_unit.anvil:182-185`); UNEXPECTED_CAP_TYPE for types other than LINEAR/NONLIN/UNINIT (`:187-189`); ILLEGAL_OPERAND_VALUE for `rs1 >= rs2` or outside the parent (`:192`); cursor clamped (`:196-214`) | tied | MATCH on all three (both implementations raise) | c128 | `flags-*`; **C-34** `sem-shrink-zero-size.ll` pending: size 0 traps everywhere |
| `tighten rd, rs1, imm5` | pure (`:2535-2538`); no Sema range check on the immediate | **imm > 7 clamps perms to 0, no exception** (`cap-man-insn.adoc:349-351`); rs1 moved (nulled unless NONLIN) | clamps imm > 7 to no-permissions (`op_helper.c:1137`), nulls rs1 unless copyable (`:1141-1146`); untagged: `assert` (`:1133`, abort) | **imm > 7 raises ILLEGAL_OPERAND_VALUE** (`capstone_dyn_unit.anvil:231-232`) -- RTL deviates from spec here; **rs1 echoed back unchanged, never nulled** (`:243`) -- deviates from spec's MOVC step; untagged: UNEXPECTED_OPERAND (`:227-228`) | spec/QEMU yes, RTL no (R-21 stays open for TIGHTEN) | DIVERGENT two ways (imm range, consumption) | c128 | Sema range 0..7 is the safe choice on all three; `fatal-tighten-range.ll` |
| `init rd, rs1, rs2` | pure (`:2563-2566`) | consumes (`cap-man-insn.adoc:427-429`, MOVC step); `cursor <= end` illegal (`:421`) | untagged/type/cursor: **asserts** (`op_helper.c:1198-1200`, abort); nulls rs1 (`:1204-1208`) | untagged: UNEXPECTED_OPERAND (`capstone_flu_unit.anvil:131-132`); not UNINIT: UNEXPECTED_CAP_TYPE (`:135-136`); `cursor <= end`: ILLEGAL_OPERAND_VALUE (`:139-140`); **writes the new LINEAR capability to BOTH rs1 and rd when rd != rs1** (`:147` passes rd as the cap_rs1 writeback) -- two live LINEAR capabilities over one region | spec/QEMU yes, **RTL no (duplicates)** -- R-21 | DIVERGENT | c128 | R-21 (ISSUES.md); hardware-side item |

## Field query, copy, type change

| insn | compiler assumption (file:line) | spec | QEMU | RTL | consumes LINEAR | untagged / wrong type | result | settled by |
|---|---|---|---|---|---|---|---|---|
| `lcc rd, rs1, sel` | pure, integer result (`:2453-2465`); selectors 0,1,3,4,5 via `selectLCCField`; **selector 2 never emitted** (the cursor is `PseudoTRUNC_CAP`, `:2470-2529`, because selector 2 traps on untagged and a null pointer is untagged, C-19) | UNEXPECTED_OPERAND unconditionally for a non-capability (`cap-man-insn.adoc:167-168`); RTL documents its per-selector total-query behaviour as a deliberate deviation awaiting a spec amendment (`capstone_dyn_unit.anvil:190-194`) | selector 1 on untagged returns 7 (`op_helper.c:837-840`); **every other selector, including 0, raises UNEXP_OP_TYPE** on untagged (`:837-864`); wrong-type per selector: **asserts** (`:865-869`) | **only selector 1 is total** (`capstone_dyn_unit.anvil:195`, answers `NOT_CAP - 1` = 7 at `:208`); selector 0 and 2 trap UNEXPECTED_OPERAND; wrong type per selector raises UNEXPECTED_CAP_TYPE via `check_LCC_invalid_multiplexing` (`capstone_unit.anvilh:469-473`) | no | MATCH on which selectors trap; DIVERGENT on wrong-type (RTL raises, QEMU aborts) | i64 | **C-33**: `cap_get_tag` (selector 0) traps on the value it exists to test, on both. Lower as selector 1 + `sltiu rd, rd, 7`. `sem-get-tag-total.c` pending |
| `movc rd, rs1` | pure copy preserving the tag (`:2497-2501`); `copyPhysReg` uses it for every GPCR copy; **assumes the source survives** | nulls rs1 unless NONLIN or self; "no exception could be raised" (`cap-man-insn.adoc:31`) | copies, then nulls rs1 only if **tagged** and not copyable (`op_helper.c:580-585`); an untagged source is never touched | **the non-NONLIN, non-self branch nulls rs1 for NOT_CAP too** (`capstone_flu_unit.anvil:13-26`, no NOT_CAP exclusion): `movc rd, rs1` of a plain integer or null ZEROES rs1 on silicon | yes on all three (which is why `PseudoCapGlobalBase` fuses `cincoffset; delin`) | **untagged source: RTL zeroes it, QEMU keeps it** -- C-32, reachable when an integer bridged into c128 (inttoptr) stays live after a copy | c128 | `c14-copy-class-postra.mir`; `sem-movc-untagged-live.c` pending |
| `delin rd` (tied) | `hasSideEffects = 1`, tied (`:2571-2580`); emitted by `PseudoCapGlobalBase` for every gp-derived global base under the default ABI, never under `-capstone-gp-captable` | LINEAR only: UNEXPECTED_CAP_TYPE otherwise (`cap-man-insn.adoc:381-384`) | **NONLIN operand: silent return** (`op_helper.c:1174-1181`); untagged: `assert` (`:1172`) | untagged: UNEXPECTED_OPERAND (`capstone_dyn_unit.anvil:473-475`); **anything but LINEAR, NONLIN included: UNEXPECTED_CAP_TYPE** (`:476-477`) | tied | NONLIN: QEMU no-op, spec and RTL trap | c128 | **C-27**: the default global ABI (`delin` of a NONLIN gp-derived base) is QEMU-only by construction. `gp-table-linear-delin.ll`; the DELIN decision is the project lead's |
| `seal rd, rs1` | pure (`:2609-2612`) | LINEAR, RW perms, size >= 1024, 16-aligned | untagged/type/perms raise (`op_helper.c:1219-1232`); **size/alignment only printed, never raised** (`:1234-1237`; `CAP_SEALED_SIZE_MIN` = 528 in `cap.h:35`, spec says 1024); nulls rs1 (`:1243-1245`) | untagged/type/perms raise (`capstone_flu_unit.anvil:161-166`); **the size/alignment check is dead at the netlist** (Anvil mis-parse, `docs/history/20-08-2026_23-38-29_anvil-relational-misparse-seal-spec-violation.md`); nulls rs1 (`:170-173`) | yes | neither enforces the minimum size (two unrelated bugs) | c128 | `intrinsics.ll`; hardware-side item, already filed |
| `mrev rd, rs1` | `hasSideEffects = 1`, rd is GPCRNoC0 (`:2598-2606`) | LINEAR only; INVALID_CAPABILITY for a dead node (`cap-man-insn.adoc:532-533`) | untagged/type: `assert` (`op_helper.c:965-966`, abort); **no invalid-node check** | untagged: UNEXPECTED_OPERAND (`capstone_dyn_unit.anvil:79-80`); not LINEAR: UNEXPECTED_CAP_TYPE (`:81-82`); dead node: INVALID_CAPABILITY (`:94-95`) | no (creates a REVOKE cap over rs1) | DIVERGENT (asserts; missing node check) | c128 | `intrinsics-unused-result.ll`, `cap-invalid.s` |
| `drop rs1` (tied) | `hasSideEffects = 1`, tied, no memory flags (`:2615-2620`); a load through the same capability may be scheduled across it (Tier 4.3) | any capability; UNEXPECTED_OPERAND for a non-capability (`cap-man-insn.adoc:495-497`) | untagged: raises UNEXP_OP_TYPE (`op_helper.c:946-955`); **register set to null** (`:958`) | untagged: UNEXPECTED_OPERAND (`capstone_dyn_unit.anvil:26-27`); **register bits left unchanged** (`:24-38`), invalidation only via `drop_req` on the revocation node | tied | a use after drop: QEMU faults at the tag check (UNEXPECTED_OPERAND), RTL at node validity (INVALID_CAPABILITY) | c128 (tied) | DIVERGENT on the register state; `sem-drop-orders-loads.ll` pending |
| `revoke rs1` (tied) | side effects + `mayLoad`/`mayStore` + barrier (`:2623-2629`) | REVOKE cap in; LINEAR out if every revoked node was NONLIN, else UNINIT | untagged/type: `assert` (`op_helper.c:904-905`); **no invalid-node check**; UNINIT result has cursor = END (`:909-921`, so a following `init` succeeds) | untagged: UNEXPECTED_OPERAND (`capstone_dyn_unit.anvil:45-46`); not REVOKE: UNEXPECTED_CAP_TYPE (`:47-48`); dead node: INVALID_CAPABILITY (`:55-56`); **UNINIT result has cursor = START** (`:67-68`), which RTL's own `init` rejects (`capstone_flu_unit.anvil:139` needs `cursor > end`) | tied | DIVERGENT on the UNINIT cursor -- an RTL defect independent of QEMU | c128 (tied) | `intrinsics-unused-result.ll`; hardware-side item |

## Domain crossing and CSRs

| insn | compiler assumption (file:line) | spec | QEMU | RTL | consumes LINEAR | untagged / wrong type | result | settled by |
|---|---|---|---|---|---|---|---|---|
| `call rd, rs1` (`CAP_CALL`, funct7 `0100000`) | side effects, load/store, barrier, **not** `isCall` (`:2639-2648`): the callee-saved mask still promises callee-saves survive, which a callee domain does not honour (C-36). Mnemonic collides with `PseudoCALL` (C-38) | not audited | decodes at `0100000` (`insn32.decode:977`) | decodes at `0100000` (`decoder.sv:1267`) | not audited | not audited | c128 | encoding MATCHES; operand model pending |
| `capenter rd, rs1` (funct7 `0100010` in the `.td`) | as above (`:2650-2654`); selected by `selectCapEnter` (`CapstoneISelDAGToDAG.cpp:1866`), so it IS emitted | not audited | **decodes CAPENTER at `0001101`** (`insn32.decode:980`) | **decodes CAPENTER at `0001101`** (`decoder.sv:1285`); the `.td`'s `0100010` falls to `default` and traps ILLEGAL_INSTRUCTION (`:1291`, `capstone_flu_unit.anvil:535`) | n/a | n/a | c128 | **C-36 confirmed**: the compiler's encoding is decoded by neither implementation. `cap-valid.s` pins the wrong bytes today |
| `return rs1, rs2` (`CAP_RETURN`, `0100001`) | terminator, `isReturn` (`:2656-2660`) | not audited | decodes at `0100001` (`insn32.decode:978`) | decodes at `0100001` (`decoder.sv:1278`) | not audited | not audited | -- | encoding MATCHES; operand roles not audited |
| `capexit rs1, rs2` (`0100011`) | terminator (`:2662-2666`) | not audited | **no such instruction** (no entry in `insn32.decode`, no helper) | **no such instruction** (no `fu_op` member, no `.anvil` function); would trap ILLEGAL_INSTRUCTION | n/a | n/a | -- | **C-36: a compiler-only phantom**; no `Capstone::CAPEXIT` reference in any `.cpp` |
| `ccsrrw rd, csr, rs1` | side effects, barrier (`:2670-2672`); no Sema check on the CSR id | not audited | not audited | not audited | not audited | not audited | c128 | `cap-valid.s` |

## Pseudos and hand-encoded forms

| form | expands to | assumption | settled by |
|---|---|---|---|
| `PseudoTRUNC_CAP` | `addi rd, rs, 0` (`:2525-2527`) | an integer write clears the metadata shadow on both implementations: QEMU `gen_set_gpr` (`translate.c:395,419`); RTL `cap_we` gated on `cap_result.valid` (`commit_stage.sv:322-325`), shown by `alu-write-clears-shadow.S` and `cincoffset-stale-metadata.S` (run 2026-08-08). **A bare sub-register read is NOT an integer write** -- that is C-31 | `pseudo-expansion-roundtrip.ll` |
| `PseudoCapGlobalBase` | `cincoffset rd, gp, rs1; delin rd` (`:2585-2596`) | one pseudo so no LINEAR value is ever an SSA value with two uses | `global-base-nonlinear.ll`, `gp-table-linear-delin.ll` |
| integer -> capability bridge (`inttoptr`, BITCAST case `CapstoneISelLowering.cpp:8806-8809`) | `INSERT_SUBREG` into an undefined c128 | the result is untagged **only if a copy is emitted**: `copyPhysReg` (`CapstoneInstrInfo.cpp:680-682`) emits nothing when the integer and the capability share a register, so `ptrtoint -> inttoptr -> load` at -O1 is `lbu a0, 0(a0)` alone and the load carries the source's authority (claim-auditor, 2026-09-04) | pre-existing hole; C-32's twin |
| SPLIT (glue only) | `.insn r 0x5b, 1, 6, ...`; no `.td` definition | the compiler never emits it | `cap-insn-split.s` |

## Forwarding and the zero register (from the rtl-oracle pass, for the compiler's model)

- `check_fwd_rs1` (`ariane_pkg.sv:970-975`) forwards a self-consuming rs1 writeback only for
  SPLIT, MOVC, CJALR, CCSRRW, STC. CINCOFFSET, SCC, SEAL and INIT also null rs1 through the same
  writeback channel (`commit_stage.sv:279`) and are **not** in the list; whether the generic
  stall logic covers a zero-gap read of the nulled rs1 is UNRESOLVED. `cincoffset-linear-clear.S`
  Case A is that exact probe and has no recorded run. Compiled code never reads a consumed
  register under the Tier 4.1 contract; the linearity verifier is what makes that true.
- `x0`/`c0` is hard-zero in the metadata regfile on RTL (`ariane_regfile_ff.sv:96-99`). QEMU has
  no such guarantee: `helper_csmovc` guards reads of x0 but not writes to `rd == 0` (`op_helper.c:574`),
  nor do the cincoffset/scc/tighten/init helpers. The compiler never allocates `c0` as a
  destination except through `GPCRNoC0` on MREV; latent, not live.

## Open rows, in the order the plan settles them

1. C-31 (rs2 from a bare sub-register read), C-32 (movc of an untagged live source), C-36
   (CAPENTER encoding, CAPEXIT phantom): compiler fixes, cycle 2.
2. Tier 4.2: C-33 (`cap_get_tag` via selector 1), C-34 (zero-size SHRINK), the end-bound
   convention (SHRINK: old end is an inclusive valid new end on all three).
3. Tier 4.3: `drop` ordering against loads.
4. Tier 4.4: CAP_CALL clobbers, CAP_RETURN operand roles.
5. Hardware-side items to hand over, one folder each: REVOKE's UNINIT cursor, INIT's duplicate
   LINEAR writeback, TIGHTEN's imm range and non-consumption, MOVC zeroing an untagged source,
   the ldc/stc linear-slot semantics absent from QEMU, the QEMU asserts.
