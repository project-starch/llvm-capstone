# The next RTL revision: four items in ONE bitstream, with every acceptance reading written before the build

*RTL lane, 2026-10-20, at the semantics owner's request and under the sufficiency rule ("a bitstream costs ~90
minutes plus a reflash; audit for SUFFICIENCY, not just correctness — and BATCH"). Nothing here is blocked on a
bitstream: both S-23 faces have module-side workarounds with field evidence (the srevroot-zero module for the
data face, the node floor for the fetch face, dev 794d7c5815f5). **The reflash of the resulting bitstream is the
project lead's own word, as always; this document does not assume it.***

## Why these four together

Each is a small RTL change with a question only silicon or a full-tree simulation can answer, and three of the
four share one subsystem (the revocation-node broadcast and the capability encoder), so splitting them across
builds would pay the ~90-minute cycle three times for changes that will interact. **Discovering the second one
after the first has been built costs another full cycle**, which is the rule this batch exists to respect.

The go/no-go question each item has to pass before the build: **what will it READ if the current best hypothesis
is true, and would that reading be uninformative?** Every item below answers it, and the answers are the
acceptance readings.

---

## (1) The namespace-aware invalidation broadcast — the real fix for both S-23 sites

**What is wrong.** The broadcast carries a BARE index under the table namespace
(`ex_stage.sv:1644-1645`, `revnode_invalidation_id_o = node_wr_req[29:0]`, with the generation hardwired `14'd0`
in the virtual record decode at `:1431-1433`), and the CPMP tracker compares **16 bits only**
(`pmp_data_if.sv:107`). So a VIRTUAL index clears a tracker holding a PHYSICAL one. That is the root cause of
both S-23 sites.

**An open DESIGN question the fix has to answer, found while clearing the dependency, and the semantics
owner's to rule on.** "Carry the namespace in the broadcast and compare it" needs something to compare
AGAINST, and the capability word cannot supply it: `cap_metadata_t` is exactly 64 bits -- `{revnode_id 30,
perm 4, cap_type 3, bounds 27}` (`ariane_pkg.sv:644-650`, and the comment at `:638-641` records that the 28th
bounds bit was already spent widening perm) -- so there is **no spare bit for a namespace tag**, and the adopt
copies only `cpmp_i[i].comp_metadata.revnode_id` (`pmp_data_if.sv:88-94`). The namespace therefore has to be
recorded at ADOPT time from the prevailing flag, which is an approximation with two consequences:

- **It fixes the false-DENY direction, which is both S-23 faces, and without needing R-58.** Entries installed
  at M adopt as physical; a context's broadcast carries virtual; they no longer match. `va-fetchsplit`'s firing
  arm goes from faults 1 to 0 on exactly this.
- **It opens a missed-INVALIDATION direction, which is the worse failure in security terms**: a capability
  whose node is VIRTUAL, installed into a CPMP entry while the prevailing flag reads physical, would record
  physical and then survive a legitimate virtual revocation -- stale authority rather than a false refusal. The
  path I can construct is R-58's own situation: a caller at S inside the table namespace derives a virtual
  node, traps to M, and M installs it with `csupctl.table` clear. Whether that is reachable in any supported
  flow is the ruling I need; if it is, a bit in the capability word (or a refusal to install a cross-namespace
  node into a CPMP entry) is needed instead.
- **A ONE-BIT namespace is sufficient only because of the existing flush**, and this must be stated in whatever
  lands: `revnode_ns_flush_o = (vctx_rev_q != vctx_rev_qq) || (srevroot_rev_q != srevroot_rev_qq) ||
  ns_flush_strobe_i` (`ex_stage.sv:1462`) re-arms every entry when the TABLE BASE changes, so inside a
  flush-free window every virtual broadcast belongs to one table. Two different contexts both read "virtual",
  so a fix that narrows or removes that flush would reintroduce cross-table false clears.

**Why it is not a comparator widening.** `pmp_data_if.sv:96-100` records why the compare is index-only: *"a
30-bit compare against a tracked (g, i) with g >= 1 would never fire and invalidation would silently stop
working at generation >= 1"*. The fix must therefore **carry the namespace in the broadcast** and compare it,
not widen the existing compare.

**WRITTEN 2026-10-20 to the ruling above, and smaller than this document feared.** `pmp_data_if` is
instantiated ONCE, inside `load_store_unit.sv:630`, so the namespace bit goes ex_stage -> LSU -> pmp and needs
**no new top-level wire**. Three changes: `ex_stage.sv` gains `revnode_invalidation_vctx_o = vctx_rev_q` beside
the tap at `:1689` (the HELD flag, which is the same one the tap's own `vp_hdr_wr` qualifier uses at `:1469`,
so the bit cannot disagree with the write it describes); the LSU passes it straight through to `i_pmp_data_if`
and uses it nowhere else; and the clear at `pmp_data_if.sv:96` gains `&& !revnode_invalidation_vctx_i`. The
consumer enumeration behind "only the CPMP trackers" is exhaustive and was checked, not assumed: the whole core
has three consumers of the broadcast -- the CPMP tracker (`pmp_data_if.sv:96-107`), the LSU revocation cache
(`load_store_unit.sv:1301`, `:1416`, `:1437`) and the PC tracker (`commit_stage.sv:408-412`) -- and the latter
two are deliberately untouched. **The synthesis risk is the lowest available for this change**: `vctx_rev_q` is
a FLOP output, so no new combinational term enters the CPMP cone, and no UNOPTFLAT stem names
`cpmp`/`pmp_data_if`/`revnode_valid` (checked against the gate's 40 stems). The standing rule about feeding a
new signal into a cone that already carries a loop therefore does not bite here -- but it is still an edit
crossing two module boundaries into the CPMP path, so it goes to synthesis before it goes anywhere else.

**Acceptance readings, pre-registered.**
1. `cap-fetchdeny` firing arm: **still 1 cause-1 fault** at the following instruction. A same-namespace
   collision is a legitimate revocation and must keep denying — a fix that silences this arm has broken
   invalidation, not fixed it.
2. `cap-fetchdeny` control arm: **still 0 faults**.
3. The cross-namespace arm, which NOW EXISTS and has its baseline measured: `va-fetchsplit-s`'s firing
   variant (capstone-ariane, the `FS_SAMETBL` pair) must read **0 faults with the four children still at
   0x100**, where it reads **faults 1, mtval 0x80001020, mepc - s_code 0x20** on the tree that already carries
   item (2) -- measured, not predicted, and measured on the right base, which is what makes it item (1)'s
   reading alone. **Use the SAMETBL
   pair, not the default pair**: in the default arm the caller's srevroot is `tbl` while the context's is
   `tbl3`, so item (2) restoring id 67 changes srevroot at the escape, raises a `revnode_ns_flush` edge
   (`ex_stage.sv:1462`) and re-arms every tracker -- which removes the refusal too, so a 0 there cannot say
   which item did it. The SAMETBL variant sets the caller's own table to the context's, so the restore writes
   an unchanged value, no flush fires, and only item (1) can produce the 0. The default pair is item (2)'s
   acceptance instead. Its control (one CAPNODE operand apart, faults 0) must stay at 0, and the `-DFS_IDX`
   variant must keep reading the same four indices (physical 3 and 3, physical 10, next-unused 7) -- if the fix
   changes WHICH indices are allocated, the arm has stopped being a one-variable pair.
4. **pb10 WITHOUT the srevroot-zero module must read FRTY 0** (the nop module `2a8f85f7`, or any module
   lacking the zeroing write). The semantics owner parked pb10's CSRUNV+4 anomaly on 2026-10-20 -- unexplained,
   covered by srz on dev -- with item (1) as the fix for the operative defect whatever the anomaly turns out to
   be. **This acceptance costs no extra boot**, and it is the anomaly's own test: a nonzero reading there means
   the anomaly is a SECOND defect and comes back open. The reason it is parked rather than pursued: the
   timer/SBI explanation predicts the WRONG DIRECTION. Between calls the nop and csrrc variants leave
   srevroot = table at S, so `vctx_table_o` is 1 and every S->M trap is a 1 -> 0 edge re-arming all slots,
   while srz leaves srevroot = 0 so its traps are 0 -> 0 and re-arm nothing -- interrupt traps between calls
   should therefore make nop/csrrc refuse LESS than srz, and the field reads the reverse. No board test for it.
5. **The generation case, which is the one most likely to be forgotten**: an arm that revokes a node at a
   RECYCLED index, whose generation has been bumped to ≥ 1 by a physical pop (`capstone_rev_node.anvil:289-307`),
   must **still deny**. That is precisely what the index-only compare was chosen to preserve, so it is the
   regression this fix can most easily cause.

**That dependency is CLEARED** (2026-10-20, capstone-ariane `c7e6f604f`, audited). It needed three fixes, each
recorded in the arm's header: the two coverages split so only the caller's own code sits under the tracked entry
(`va-fetchctx` had one grant over the whole image, so the collision denied the CONTEXT's fetches and the fault
was the context's); the context entering by ecall so its escape is terminal and `sup_active_q` is 0 when the
handler repairs the entry (a CCSRRW to 0x010..0x01F is a privilege violation while it is 1,
`csr_regfile.sv:3018-3021` -- and the clean return now MEASURES that, where the wedge only implied it); and the
repair sourced from a register minted for it and never written again, after two attempts used registers the
scaffolding writes as CSRUNV arguments.

**THE DESIGN IS RULED, and the defect is NOT the one this document first named.** The adversarial audit refuted
the clause "because the compare is index-only": both ids in the arm are generation 0 -- the broadcast is masked
to `#{14'd0, index}` (`capstone_rev_node.anvil:3-6`) and the physical bump mints `#{14'd0, head}` (`:314`) --
so a full 30-bit compare fires identically and the arm cannot see the slice. The operative defect is that **ids
carry no namespace identity**. The semantics owner's ruling, which addresses exactly that and needs no
per-entry state:

- **ISA rule**: a CPMP entry holds only PHYSICAL-namespace capabilities. Every supported producer is the
  monitor's M-mode region path with `csupctl.table` clear; a context's virtual nodes live in its registers, its
  frame and its memory, and the CPMP is INIT-zeroed inside a context anyway.
- **RTL**: the broadcast carries one namespace bit and **the CPMP trackers act only on PHYSICAL broadcasts**.
  The LSU revocation cache and the PC tracker keep consuming both, because they do track a context's virtual
  capabilities. This was chosen over recording a namespace per CPMP entry, which is fragile exactly where it
  matters: around a switch the escape's restore walk rewrites the caller's entries while the flag may still read
  virtual, so a re-adopt in that window would tag the caller's physical entries virtual -- bringing the false
  deny back and missing physical revocations as well.
- **Monitor invariant**, documented beside the rule: never install into a CPMP entry a capability obtained from
  a context, a frame or a table-namespace op. The path this lane constructed (an S caller inside the table
  namespace derives a virtual node, M installs it) is not a supported flow, so under this rule it is a monitor
  bug rather than a hardware obligation. No hardware check is possible without a bit in the capability word --
  `cap_metadata_t` is exactly 64 bits -- and none is needed for correctness inside the invariant.

Acceptance reading 4 below becomes more important under this ruling, not less: it is the ONLY arm in the batch
that exercises the index-only slice at all, since it is the only one with a tracked generation >= 1.

## (2) R-58: the caller's `srevroot` restored at the escape

**What is wrong.** The switcher's SAVE/RESTORE window covers register ids 0..66 and `srevroot` is id 67, so the
caller's value is never saved and never restored; after any entry or resume the caller runs in the CONTEXT's
lifetime namespace (`capstone_dom_switcher.anvil:170-173`, `:481`, `:315`, `:527-541`).

**The change**, scoped in R-58's row: id 67 in the caller's snapshot, which extends SAVE and RESTORE together
through the shared `walk_advance`. Verified at source 2026-10-20 rather than inherited from this document's
first draft, and it comes out SMALLER than that draft said:

- **The offset is +944, confirmed by walking the slot sizes.** Ids 0..2 and 9..56 take 16-byte slots and the
  rest 8 (`capstone_dom_switcher.anvil:669-675`, and the comment there records that the unparenthesised
  `a < 3 || (a > 8 && a < 57)` once gave 16 bytes to id 0 alone — [[anvil-relational-binds-loosest]]), so
  0,1,2 at +0/+16/+32, 3..8 at +48..+88, 9..56 at +96..+864, 57..66 at +864..+936. Id 66 ends at +944 and id 67
  takes +944..+952. The caller's FP block sits at +2048 absolute and the snapshot base is +1024, so
  1024+952 = 1976 < 2048: **nothing is displaced**, which is why this is a one-branch edit.
- **The edit is one branch of `walk_advance`** (`:527-532`): the CSRUNV caller's walk jumps 66 -> 75, and it
  becomes 66 -> 67 -> 75, keeping the save and restore bases as they are. The slot-size selector already
  yields 8 bytes for id 67 (`:701-703`), and `save_io`/`restore_io` already address id 67 — the ESCAPE and
  RESUME sequences do exactly that at `:479-481` and `:313-315` — so no new register port is needed.
- **There is NO software or reference-model definition of the snapshot to move with it.** This document's first
  draft asserted one; it does not exist. Searched: `snapshot` and the bases 1024/2048 across
  `capstone/runtime` and `capstone/capstone-ariane/core`; `csrunv`/`CSRUNV` across
  `capstone-qemu/target/riscv/` (**zero hits** — the `dom_switch` matches there are the physical profile's
  CAPENTER/CAPEXIT path, `op_helper.c:1777`, `:2551`); and the module's own frame use, which stops at
  `frame[81]` = +648 (`capstone_vm.c:339-340`, with satp at `frame[0]`, the table at `frame[1]`, options at
  `frame[7]`, the PCC slot at `frame[8]`/`[12]`). The virtual switcher's frame layout lives in the RTL alone,
  so the divergence this bullet was written to prevent cannot occur. **What this removes is a task, not a
  risk**: if a model is added later it inherits +944 from the RTL.

**WRITTEN AND SIMULATION-VALIDATED 2026-10-20** (capstone-ariane lane `524326f34`, one branch of
`walk_advance`; no synthesis until the batch is complete). Three of its four acceptance arms read exactly as
pre-registered and the fourth is refuted as an arm rather than as a fix -- see below. The fix also removes
`va-fetchsplit`'s CSRUNV+4 refusal (faults 1 -> 0), which is why item (1)'s acceptance moved to the SAMETBL
pair, and the same-table arm reading faults 1 on this tree is the evidence that the two items are now
separable.

**Acceptance readings, pre-registered.**
1. `va-srevroot`: the caller must read **its OWN value back** — `s3 = 0x80006000` and **`s4 = 0x80006000`**, where
   today `s4` reads `0x80007000`, the context's table — with the twenty baseline readings byte-identical.
2. `va-srevroot-zn` (`-DSREV_ZERO -DSREV_CSNODE`): the post-escape CSNODE must be **illegal at decode again
   (mcause 2)**, where today it succeeds with a node id, because with its own srevroot back at 0 the caller is
   outside the table namespace -- so **FOUR leading `0x2` instead of three**, and `s5` 0x4 -> 0x0. (This
   document first said "two instead of three", which is wrong in the other direction: an extra trap adds a
   reading, it cannot remove one. It also attached the reading to `-DSREV_CSNODE` alone, where it does not
   follow -- a non-zero own table is still a table namespace, so the CSNODE stays legal there.)
2b. `va-srevroot-n` (`-DSREV_CSNODE` alone, own table non-zero) is the stronger arm and is new: the CSNODE
   stays legal either way, so the reading is the node id, which names WHICH table the caller allocated from --
   `s5` **0x4 -> 0x3**, `tbl`'s untouched next-unused against the inherited `tbl2`'s 4 (4 because the entry
   consumed index 3, which is what -zn's present 0x4 shows). Legal/illegal cannot say that.
3. On silicon, the DATA face at **0 with the OLD module** on the new bitstream — the reading that shows the fix
   removes the persistence rather than the workaround masking it.
3b. **MEASURED, all three** (128,993 / 128,937 / 129,027 cycles, every arm SUCCESS): `va-srevroot` s4
   0x80007000 -> **0x80006000** with exactly one of the twenty readings moved; `-z` s4 -> **0x0**, the arm that
   separates "restored" from "coincidentally equal" because the unfixed arm read the frame's word verbatim;
   `-zn` **four leading 0x2** and s5 -> **0x0**. A fourth arm, `va-srevroot-n` (`-DSREV_CSNODE` alone), is
   **REFUTED as an arm**: with a non-zero own table BOTH CSNODEs are legal, so its s5 is the second call's
   index from the same table and names no table, and its 0x6 against a pre-registered 0x3 is unexplained
   (UNRESOLVED; tbl's next-unused word would settle whether CSNODE consumes two indices per call). It is not
   part of item (2)'s acceptance and its s3 = s4 agrees with the fix.
4. **NOT an acceptance reading**: the surviving CSRETIRE+4 fetch site. R-58 is not required for it (inside the
   window `select_root` deliberately puts the context's table in `srevroot`), and an acceptance test that cannot
   fail for the right reason is worse than none.

## (3) R-59: the precheck accepts a frame word the id-67 write masks to zero

**What is wrong.** `runv_precheck_step` accepts any **non-zero** word at `dom_base + 8`
(`capstone_dom_switcher.anvil:381-387`) while the id-67 write masks the low 12 bits
(`csr_regfile.sv:2051`), so a word in **1..4095** passes the non-zero check and installs **0**: the context is
admitted and then runs outside the table namespace.

**The change**: make the two agree — either the precheck rejects a word with low bits set, or it tests the
masked value. The semantics owner's call which.

**Acceptance readings, pre-registered.** A frame with `+8 = 0x800`: the CSRUNV must be **REFUSED** (the refusal
path, ids 74 and 0, no entry), where today the context is admitted. Control: the same arm with a real
4 KiB-aligned non-zero base must **admit** exactly as it does now. Both arms in one load; the refusal is the
reading, and the control is what makes it a reading rather than a broken arm.

## (4) R-60: an out-of-window cursor must not alias

**What is wrong.** A load or store through a capability whose cursor lies outside its representable window is
bounds-checked against the bounds **decoded at that cursor** — the true bounds shifted by whole window spans — so
when the shifted decode contains the cursor the access **passes and touches memory a span away, with no trap**.
CINCOFFSET checks only the type: it neither clears the tag nor checks representability
(`capstone_flu_unit.anvil:29-60`, writeback `ex_stage.sv:1739`/`:1753`). Characterised on silicon, stores
measured; QEMU traps 28 in every arm.

**This lane's own earlier finding is the benign half of the same mechanism and must not be read as a reassurance.**
`cap-cursor-window` (capstone-ariane `3c2cdf3f2`) measured that the compressed word SURVIVES an excursion —
`compress_bounds` does not take the cursor — so the bounds READBACK self-heals when the cursor returns. It read
base and top with LCC and **never dereferenced**, which is exactly the case R-60 measures. Self-healing bounds
and an aliasing access are the same cursor-relative decode seen from two sides.

**The change**, as ruled in `isa.md` (`6a7038fb5fe3`): CINCOFFSET and SCC **clear the tag, or refuse**, when the
cursor leaves the representable window, so an alias-load or alias-store traps or reads untagged.

**Acceptance readings, pre-registered — and the strongest of the batch, because an arm already records the
current values.** `cap-cursor-window`'s W0 ladder today reads base and top EXACT at `+0` and `+0x800`, WRONG at
`+0x4000` and `+0x10004000`, and EXACT AGAIN all the way back. After the fix:
1. at `+0x4000` and beyond the capability must read **type 7 (NOT_CAP)** from LCC zimm 1, or the CINCOFFSET must
   refuse — the tag is gone;
2. **the self-heal must be GONE**: the return to `+0` must NOT restore the bounds, because there is no longer a
   tagged capability to restore them to. That is a reading no other change in this batch can produce;
3. W1's allocator transit (`neg`/`cincoffset` to cursor 0 and back) must then **fail where it passes today** —
   its SHRINK currently passes and must now refuse or operate on an untagged capability. **This is a
   COMPATIBILITY reading, not a regression**: it is the runtime's hoisted-cursor pattern, and the compiler lane
   must have landed its fix (lane `runtime/cursor-window`) before this bitstream, or correct code breaks on it;
4. a NEW alias-store arm: an out-of-window cursor whose shifted decode contains it must **trap (28)** instead of
   writing a span away, with an in-window control still passing.

Reading 3 is the item's real cost and the reason it is listed last: **it changes the ABI for code that is in the
field today.** **The runtime side of that sequencing is already done** — the cursor-window fix landed on dev with
tier-m at `232d13b220cc` ("form slot offsets in integers before adding them to a capability, so no intermediate
cursor leaves its own window", `capstone/runtime/virtual/heap.c`), verified here rather than taken on report. So
the remaining sequencing is just **the corpus and kit images rebuilt from dev before the acceptance boots**, and
W1's reading changing from PASSES to FAILS as intended. An image built before that commit still carries the
hoisted-cursor pattern and would fail the acceptance boots for a reason that is not a defect.

## (5) Q-04 (b): MOVC must KEEP an untagged source

**Ruled already, and not by this lane**: the project lead decided Q-04 on 2026-10-14, option 1 -- MOVC keeps an
untagged source and consumes every tagged non-NONLIN one. `ref/ISSUES.md` Q-04 records the decision and that
**"RTL change Q-04 (b) NOT YET APPLIED"**, and `plans/2026-10-14-movc-untagged-source.md` holds the exact diff,
the directed tests, the regression, and the synthesis and board predictions. **This item does not restate any of
that.** It is here because the change touches `core/` and so rides this batch, and because two synthesis
questions the plan leaves open are answered below.

**The defect, confirmed at source** (`capstone_flu_unit.anvil:6-27`, read on `c7e6f604f`). `func MOVC` has three
paths: `rs1 == rd` preserves; `cap_type == CAP_TYPE_NONLIN` keeps (`rs1` unchanged, `rd` = the source); and the
`else` sends `create_cnull()` as the new `rs1`. **NOT_CAP falls into that `else`**, so every live-source integer
MOVC with `rd != rs1` nulls its source. The diff is the plan's: add `|| (cap_type == NOT_CAP)` to the preserve
condition at `:14`, each comparison parenthesised on its own because this language binds `||` tighter than `==`.

**Why it matters more than its size suggests.** The compiler lane attributes the corpus's whole silicon-only
`pc = 0` class to it -- cpython's three stops (`movc s2, a3; jalr a3` with `a3 = 0` in
`_PyCfg_OptimizedCfgToInstructionSequence`) and sql0's cells 03 and 07 -- and reports that QEMU with
`CAPSTONE_MOVC_NULL_SCALAR=1` reproduces all five at the same sites and registers with the predictions written
first, while passing all five without the switch. **That attribution is theirs and is recorded as theirs**;
this lane has not re-run it. The virtual SDK has compiled with `+movc-keeps-integer-source` since the ruling,
which is why the divergence bites on vp2 and not under emulation.

**The two synthesis answers, which are this item's own contribution:**
- **No standing combinational loop touches the FLU.** None of the gate's 40 `UNOPTFLAT` stems names
  `flu`/`movc` (checked against the lint output, not assumed), so the rule about feeding a new signal into a
  loop cone does not bite here.
- **The FLU's result endpoint IS the same shape as the DYN unit's** (`41331b04`'s finding): `_ep_res_0` is ONE
  unregistered assign of 910 bits selected by a **58-way ternary chain** over its send sites. **But this edit
  does not touch it**: it widens an existing branch's CONDITION and adds no send site, so the chain's width is
  unchanged and the whole delta is one 3-bit compare and an OR feeding a condition the branch already
  evaluates. That is the cheapest shape an FLU edit can have, and it is why the plan's "timing and area
  essentially unchanged" prediction is credible rather than hopeful.

**Acceptance.** The plan's step 2 list is the directed-test set and it is RICHER than a four-case arm: NOT_CAP
zero and non-zero with `rd != rs1`; NOT_CAP with `rd == rs1`; LIN, REVOKE, UNINIT, SEALED, SEALEDRET and EXIT
each consumed; NONLIN kept; `rd = x0` and `rs1 = x0` with each source kind; and the next instruction reading
`rs1` then `rd`, which is the forwarding and dual-writeback case. Build to that list, not to a subset. Add the
plan's step 3 regression (`tests/rtl-smoke/live-source-copy/`), and **a NEGATIVE arm on `c7e6f604f`**, which
must read `rs1 = null` for the untagged source -- the field's bug, measured rather than asserted, which is what
makes the fixed reading a pair. The plan's board numbers stand as given there (`movcprobe_kernel.h` 7 -> 49,
the SQLite stage-50 probe 55, the Sublet -O2 image's lookaside slots 0 -> non-zero against the emulator's
25,010); they are the plan's predictions and are not re-derived here.

---

## What the 95-test sweep does and does not certify for item (2)

**The sweep is a BUILD and REGRESSION check for item (2), not a behavioural one, and the behavioural evidence
is the four `va-srevroot` arms plus the `va-fetchsplit` arms.** That is checked rather than argued: item (2)'s
change sits entirely inside `(*commit_req.is_runv == 1'b1)`, a non-runv walk reaches the unchanged
`cur_idx == 66 -> walk_end()` branch exactly as before, and reading the sources of all 95 sweep tests, 92 of
which resolve, **ZERO contain a CSRUNV or funct7 0x24** -- the other three are the random generators already on
`sweep-expected-missing.txt`. So the `is_runv` walk is unreachable for the entire sweep. Said plainly so that
"the sweep identical" claims only what it can.

**This lane has no usable sweep BASELINE, which is a gate gap and not an item-(2) result.** The committed
`verif/sim/sweep-base-7e4dc440f.txt` is from 2026-08-12, **216 commits back**, and `sweep-compare.py` compares
two result DIRECTORIES, of which this lane has none from a prior run. At 25 of 95 tests, eight already disagree
with that file's rc column and **five disagree in the IMPROVING direction** (`cincoffset-stale-metadata`,
`cincoffset-linear-clear`, `cincoffsetimm-linear-clear`, `scc-linear-clear`, `stc-register-clear` all read rc 0
where it records 1/2/2/3). Two disagree in the failing direction -- `capsbi-init` (FAILED at the
2,000,013-cycle limit against SUCCESS at 1203) and `capenter` -- and **neither can be item (2)** by the
argument above. They are attributed by a sweep on `c7e6f604f` compared directory-to-directory, and **not
before**; that run also becomes this lane's baseline, committed in place of the 7e4dc440f file with the commit
hash in its name and the run's date in its header. Read the cycle count beside every verdict: the testharness
prints SUCCESS at the cycle limit.

## A PRECONDITION on every image booted on this bitstream

**Every image booted on the batch bitstream -- Tier 1, the corpus kits, and the pb10 FRTY acceptance image --
must carry a runtime containing `232d13b220cc`.** Item (4) puts R-60's clear-tag rule in silicon, and the
compiler lane reports that under it an image built on a runtime WITHOUT that commit faults at its FIRST
allocation: cause 24 at `__capstone_malloc_group_map+0x210`, on the pre-fix `bind_slots`'s
`group + start - base` intermediate, which the clear untags and the code then uses. **Without this line, a
fault at the first allocation reads like a bitstream regression**, which is the most expensive misreading
available on a freshly flashed bitstream.

The commit is verified here: `232d13b220cc`, 2026-10-20, "virtual runtime (v6): form slot offsets in integers
before adding them to a capability", `capstone/runtime/virtual/heap.c` +8/-2.

**How to check it, and NOT by the kit name.** The manifest `build.py` writes (`ports/common/application/build.py:172-189`)
carries the three fields this needs:
- `runtime_dirty` **must be false FIRST.** It is `bool(git status --porcelain)`, and if the build tree was
  dirty then `runtime_revision` says nothing about what was actually compiled. Check this before the ancestry,
  because a dirty build can pass the ancestry test and still lack the hunk.
- **CONTENT, not ancestry.** `git show <runtime_revision>:capstone/runtime/virtual/heap.c` must contain BOTH
  exact substrings `(char *)group + (start - base)` and
  `(r->key - __builtin_capstone_cap_get_base(r->parent))`.
  **The ancestry form that stood here until 2026-10-20 was WRONG and refused exactly the images that
  qualify**: the lead's rebuilds use runtime `8f132305eea1`, which carries the heap.c change as a PORT
  (`5620badda510` onto `5b3e6b79`) rather than by descent, and `git merge-base --is-ancestor 232d13b220cc
  8f132305eea1` **exits 1** -- verified here, as is the content test on four revisions: `5b3e6b79` false and
  false (must refuse), `232d13b220cc`, `5620badda510` and `8f132305eea1` all true and true (must pass).
  Content also closes the revert case that ancestry cannot: a later revert keeps the ancestry and loses the
  hunk.
  The board lane has this as a script, `r60-runtime-gate.py` on lane `vp2-tier12` (a manifest or `--rev`;
  exit 0 PASS, 1 REFUSE, 2 NO DATA, with controls run both ways). **Prefer the script once it is reachable
  and let this text defer to it** -- but the rule is written out here deliberately, because as of this commit
  the script is NOT on `origin/vp2-tier12` (checked after a fetch; that branch's tip is `30a39b497d39`) and a
  precondition pointing only at a file nobody can open is no precondition.
- `runtime_archive_sha256`, the sha256 of the built `libapplication-runtime.a`. **This is the stronger check
  and is revision-independent**: images that must share a runtime must share this hash. Prefer it when
  comparing two images, and use the ancestry test when judging one image on its own.

**The pb10 FRTY acceptance image is included in this**, so it needs rebuilding on a post-`232d13b2` runtime
before that boot; its own acceptance line says nothing about the runtime and would otherwise be read against an
image that dies before it reaches the measurement.

## Gates before the hash is handed to synthesis

Unchanged from every previous build, and none of them is evidence of synthesizability:

- `bash verif/sim/rtl-lint-gate.sh` **PASS** at baseline by NAME, not only by count, run inside the container
  (it fails to start on the host: `verilator: command not found`, which the gate correctly calls an ERROR);
- the 95-test sweep identical, `testlist_sup.yaml` identical plus the new arms at their recorded readings;
- `claim-auditor` over the diff, asked specifically what could break synthesis — inferred latches, combinational
  loops, undriven struct fields, positional struct literals whose field count changed;
- **`UNOPTFLAT` loop membership by STEM, diffed and not just counted.** Item (1) adds a term to the invalidation
  broadcast, which crosses a module boundary into the CPMP path — the highest-risk edit available, and every
  check we have is blind to it. If the change touches a cone on the standing loop list, **it goes to synthesis
  before it goes anywhere else**;
- the hash handed over carries its lint numbers, and **a hash is ready when synthesis has RUN**, not when the
  checks pass.

## What this batch deliberately does NOT include

- The index-only compare's **alternative** fix (a generation in the virtual record) — deferred by a prior ruling
  and not needed once the broadcast carries the namespace.
- Anything for the **third, unobserved S-23 face** (a context clearing the caller grant it is itself fetching
  through). The node floor prevents it for any id the floor covers, and it has never been observed; item (1)
  removes it as a side effect.
- The **CSRUNV+4 open anomaly** (pb10's 2 → 0 under the srevroot-zero module, which no mechanism predicts). It
  needs a reading, not a fix, and the reading is a one-variable pair isolating the inserted instruction from the
  srevroot value — simulation work, not a bitstream item.
