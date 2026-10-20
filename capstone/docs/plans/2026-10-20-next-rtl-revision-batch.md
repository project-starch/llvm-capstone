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

**ACCEPTANCE MEASURED AND PASSED 2026-10-20** (capstone-ariane `7784b37ab`, six arms in ONE model build;
the witness counts were pushed in `56b14ef04` BEFORE the run, and the per-arm expectations were sourced from
each arm's own header rather than from memory).

| arm | pre-registered | read | |
|---|---|---|---|
| `cap-fetchdeny` | faults 0, suppressed 0 | faults 0, suppressed 0, valid-fell 0 | PASS |
| `cap-fetchdeny-f` | faults **1**, suppressed 0 | faults 1, suppressed 0, **valid-fell 1** | PASS |
| `r12-recl-cpmp` | deltas `0 1 1 0 1 0 1`, count 4 | exactly that, byte-identical to four prior records | PASS |
| `va-fetchsplit-sc` | faults 0, suppressed **0** | faults 0, suppressed 0 | PASS |
| `va-fetchsplit-i` | indices 3, 3, 10, 7 | `Revnode_id 3`, `3`, `10`, next-unused `7` | PASS |
| `va-fetchsplit-s` | faults **0** (was 1), children 0x100 x4, suppressed >= 1 | faults 0, mtval 0, children 0x100 x4, **suppressed 1** | PASS |

The measurement is reading 3: `va-fetchsplit-s` went from `faults 1, mtval 0x80001020, mepc - s_code 0x20`
on the base to **faults 0, mtval 0, mepc - s_code = minus s_code** here, with the four children still at
0x100 -- and it is now reading-for-reading IDENTICAL to its one-operand control `va-fetchsplit-sc` at every
pc. Reading positions were taken from the source's print order (`va-fetchsplit.S:908` is the fault count,
`:910` its mtval, `:914` the mepc delta), not inferred from a diff.

**The fix's action is PRINTED, not inferred**, which is why the trace gained a witness line before the run:

    117605 SUP cpmp[0] CLEAR SUPPRESSED by namespace: vctx=1 idmatch=1 nsflush=0, valid stays 1
           | bcast id=00000003 | tracked=00000003

The index matched (virtual 3 against tracked physical 3) and the clear was withheld by the namespace bit
alone. The witness fired on exactly the two cross-namespace arms (`-s`, `-i`) and stayed silent on their
one-operand control and on every M-mode broadcast, so it carries its own matched pair inside the one run.

**The instrument is proven live in the same run**: `cap-fetchdeny-f`'s physical collision still clears the
tracker (`valid-fell 1`) and still denies (faults 1). Without that arm, `-s`'s zero would be an absence.

**Disclosures.**
1. `r12-recl-cpmp`'s basis is NOT what `56b14ef04` claimed. That message required its readings to be
   byte-identical to its log in `.sup/sweepitem2-d76acdb59`; **no such log exists** -- the arm is in
   `testlist_sup.yaml` but not in the 95-test sweep set, so neither base directory (203 and 219 files) holds
   it. Had the check been implemented as written it would have compared against a missing file. It was never
   implemented, so nothing passed silently. The basis used instead: four prior runs in
   `vp7-wt/.sup/{s23,s23b,s23c,s23d}`, all reading identically, agreeing with the arm header's
   independently-written expectation. Those four directories have no PROVENANCE file, so the RTL behind them
   is not pinned.
2. **RETRACTED and now EXPLAINED (an independent audit of item 1, 2026-10-20, caught what this lane had in
   front of it).** This disclosure first reported as "unexplained" that at the same pc (`0x20c4`, `0x20c8`) the
   `-i` arm reads `s3`/`s4` as `0x80007000` where `-s` and `-sc` read `0x80009000`. The cause is that
   **`va-fetchsplit-i` is built `-DFS_IDX` ONLY** (`testlist_sup.yaml`), with no `-DFS_SAMETBL`, so it is the
   DEFAULT pair: `0x80007000` is `tbl` and `0x80009000` is `tbl3`. Two different configurations, not an
   anomaly. The testlist's own comment calling `-i` "the same firing arm" is wrong and is being corrected
   along with the entry.
   **What this RETRACTS from the record above:** `-i` does not certify that the `-s` PAIR is one-variable. Its
   four indices (3, 3, 10, 7) are real and exactly as pre-registered, but they are the DEFAULT arm's
   allocation. Reading 3's verdict is unaffected and rests on its own evidence -- `-s` reads faults 0 with the
   four children at 0x100, reading-for-reading identical to `-sc` at every pc, with the suppressed-clear
   witness firing on `-s` and silent on `-sc`.
3. Reading 4 (pb10 without the srevroot-zero module) is the board lane's and remains UNMEASURED; reading 5's
   generation case is covered by `r12-recl-cpmp` above.

**THE SECOND DEFECT IS NOW FIXED AND MEASURED TWO-SIDED** (capstone-ariane `31ce73cb6` against its
pre-fix parent `7784b37ab`, one model build each, `s23-cpmp-rearm`). That arm already existed, written
2026-10-19 for S-23's data face, and had recorded the resurrection as a FEATURE -- its Q5 is commented
"THE WORKAROUND" and its Q8 "the S-reachable repair". The audit reframes the same probes as the defect, so
no new arm was needed; what was needed was the two-sided run.

| probe | parent `7784b37ab` | fixed `31ce73cb6` | |
|---|---|---|---|
| Q1, Q2, Q6, Q11a, Q11b, Q13a | ALLOWED | ALLOWED | controls: the instrument still allows when it should |
| Q3, Q4, Q7, Q9, Q11c, Q11d, Q13b, Q13c | DENIED | DENIED | the broadcast clear and the adopt are untouched |
| **Q5** (`csrsi csupctl, 4`, the M strobe) | **ALLOWED** | **DENIED** | the re-arm is gone |
| **Q8** (srevroot write, SPACED) | **ALLOWED** | **DENIED** | the re-arm is gone |
| **Q10** (srevroot, BACK-TO-BACK) | **ALLOWED** | **DENIED** | the re-arm is gone |
| **Q12a, Q12b** (one strobe, two slots) | **ALLOWED** | **DENIED** | the strobe's GLOBAL reach is gone |
| **Q13d** | **ALLOWED** | **DENIED** | the re-arm is gone |
| total denials (the arm's own `s11`) | 8 | 14 | |

**Exactly six probes flipped, every one of them a re-arm path, and none flipped the other way.** The six
controls read ALLOWED on BOTH trees, so the fixed tree is not denying blanket; the eight true-deny probes
read DENIED on both, so the clear and the adopt are unchanged. The arm's own correction note of 2026-10-19
establishes that every deny in it is a TRUE deny -- each clear is a REVOKE of the handle above that slot's
own child -- which is what makes six ALLOWED -> DENIED flips readable as the security fix rather than as a
regression: the denies now PERSIST where they used to be healed, and persisting is correct for a revoked
node.

**Q8's setup is PROVEN LIVE on both trees, which the first run of that arm was not.** Its Q8/Q10 were void
by construction in 2026-10-19 (the arm wrote 0 over 0, so no namespace pulse could be produced, and both
probes read DENIED for the wrong reason). The grading now requires the `SUP ns-flush RISE` line and the
HELD copy's transition it prints, and both runs show `srevroot 0 -> 0x81000000 -> 0` with 7 RISE lines. A
DENIED that is void and a DENIED that is the fix are no longer the same reading.

**Item 1's six arms are UNCHANGED on the fixed tree**, re-run in the same build: `va-fetchsplit-s` still
faults 0 with its four children at 0x100 and the suppressed-clear witness firing, `cap-fetchdeny-f` still
denies, and `r12-recl-cpmp`'s deltas are byte-identical for the third time. So the namespace gate's effect
does not depend on the flush, which is what the removal had to show.

**And `va-fetchsplit-i` now certifies what it is cited for**, with `-DFS_SAMETBL` added: it reads
`0x80009000` where it previously read `0x80007000`, matching `-s` exactly, which confirms by measurement
that the retracted "unexplained" difference was only the missing define.

**The PC tracker's half is UNMEASURED and removed on a source argument**, recorded as such at the semantics
owner's instruction rather than claimed two-sided: no route reaches that resurrection, because the
resurrecting instruction traps first (the check at `commit_stage.sv:389` fires at the commit after the
clear) or the adopt re-arms independently of the flush. Removing it is still right -- it is logic that can
only ever wrongly set valid -- but it has no arm.

**A BIGGER HOLE IS OPEN AND NOT YET MEASURED, found while trying to build that arm.** The adopt at
`commit_stage.sv:400-405` sets valid on ANY change of the PCC's id, justified as "On domain change
(CALL/RETURN/CAPENTER), assume valid -- DYN unit already checked". A plain trap to M and back is none of
those and re-checks nothing, yet the id changes twice, so a REVOKED PCC comes back valid after any trap
round trip, a timer interrupt included. This is R-35's defect in the last consumer that still has it: that
commit removed exactly this pattern from the LSU, recording that adopt-on-sight made the revocation check
"VACUOUS" and that "no tracker-local change can fix this ... the fix has to introduce POSITIVE evidence".
QEMU does not have the hole -- in a virtual-C context it checks the PCC's node at every block entry
(`capstone_supervisor.c:1099-1114`, emitted per block at `translate.c:1374-1380`), so a revoked PCC faults
at the next fetch and after any trap round trip -- so this is a QEMU-vs-silicon divergence as well. The arm
is constructible (revoke S's PCC node, one ecall round trip that touches nothing, probe the next fetch,
against a control that revokes with no round trip and must trap 25) and is pre-registered to read NO TRAP
on the current RTL. It is the next measurement.

**A SECOND DEFECT, found by that same audit, means item 1 is not finished and its arms will be re-run.**
`pmp_data_if.sv` carried, since vp2 item A step 2, `if (revnode_ns_flush_i) begin cpmp_tracked_revnode_id_d[i]
= '0; cpmp_revnode_valid_d[i] = 1'b1; end` -- every CPMP entry re-armed VALID on any change of `vctx_rev_q` or
`srevroot`, i.e. at every context entry and escape. Before item 1 that re-arm is what HEALED S-23's false
denies. With item 1 there are no false denies left to heal, and the clause's only remaining effect is to UNDO
a real revocation: a physical node the monitor revoked, tracked by a CPMP entry, would be trusted again after
the next context switch. Ruled by the semantics owner (isa.md): in this bitstream the CPMP trackers IGNORE the
namespace flush, and a tracked id and its validity change only by adopt and by a physical broadcast. The
clause is removed on the lane branch, and item 1's six arms plus a new two-sided resurrection arm re-run on
the result -- so the table above is item 1's acceptance for the namespace bit, not for the item as a whole.
The same clause exists in the PC tracker (`commit_stage.sv:417-420`), where `:389-391` raises cause 25 when
`!pc_revnode_valid_d`, and the flush fires on any `srevroot` write with no switch and no re-check; that is
referred to the semantics owner rather than fixed on this lane's reading. The LSU revocation cache is SAFE by
contrast and is the pattern the CPMP should have followed: it EMPTIES on a flush, "never permissive", and
gates the ALLOW-setting resolve on `!revnode_ns_flush_i`.

**One defect of this lane's own making, recorded because the check that catches it is one grep.** The first
attempt produced no reading at all: the namespace bit was added as an `ex_stage` OUTPUT PORT, which left
`cva6.sv:1772`'s instantiation one pin short, and under `-Werror-PINMISSING` the model build failed for all
six arms inside a minute. The bit has no consumer outside `ex_stage`, so it is now an internal signal
(`7784b37ab`) and the synthesizable delta is SMALLER than this document first described -- no new top-level
wire. Nothing was misreported: the freshness stamp and the `vctx=` provenance requirement made every arm read
VOID rather than present a stale per-day log as a reading, which is exactly how the earlier retracted "item 1
failed its acceptance" was produced.

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

**WRITTEN 2026-10-20: the op set is THREE and RULED, the predicate's site is the PACK, and its validity there is
MEASURED rather than assumed.**

*The op set, enumerated from source and then ruled* (semantics owner, in `design/virtual-capstone/isa.md` on dev at
`ffe4b088c279`, on this lane's enumeration). Exactly three ops write an arbitrary cursor onto a tagged capability,
and all three funnel through one call, `create_capability(metadata, new_cursor)`: CINCOFFSET
(`capstone_flu_unit.anvil:46`, `rs1.cursor + rs2.cursor`), CINCOFFSETIMM (`:82`, `rs1.cursor + imm`) and SCC
(`:122`, `rs2.cursor`, an ABSOLUTE set, so the one that can leave the window in a single instruction from any
starting point). The earlier wording named CINCOFFSET and SCC; the immediate form is the same instruction's
immediate encoding and the one the compiler emits by preference, so excluding it would have left the defect
reachable through the cheaper opcode. The other five callers of `create_capability` -- INIT `:201`, SHRINK
`:276`/`:282`, CAPBOUND `:448`, CCSRRW `:567` -- are in-window by construction and do NOT get the check, so the
predicate must not go into the shared helper. QEMU's compact profile covers the same three.

*The site.* Nothing in the FLU compresses: `create_capability` (`capstone_unit.anvilh:464-466`) is a bare struct
constructor and the FLU carries fat 64-bit bounds throughout. The predicate needs the COMPRESSED word, so it
belongs at the pack, `compress_cap` (`core/include/ariane_pkg.sv:861-886`), the first point holding the fat bounds
and the cursor together. A cost precedent in the same module, already routed: `ex_stage.sv:917` instantiates
exactly this round trip, `decompress_bounds(compress_bounds(dyn_child_q, 5'd0, 1'b0), dyn_child_q.start)`. The
synthesis lane's trial reading of the FLU capture nets (CINCOFFSETIMM at -4.192 ns / 70 levels) is therefore
probably NOT measuring where this lands, and both forms share one instance of the new cone rather than adding to
each path separately.

*Why the pack form is valid, measured.* The pack must recompress, so `decompress(compress(fat), cursor) != fat` is
a correct test only if `compress(decompress(B, c)) == B`. Against capstone-ariane's own
`verif/tests/custom/capstone/bounds-model.py`, 200,000 random regions, E 0..24, a random in-bounds read cursor
each: idempotence failures **0**, fires with the cursor UNCHANGED **0**, disagreements with the two-cursor form
(`decompress(B, new) != decompress(B, old)`, which `isa.md` now gives as the DEFINITION) **0** across the 106,115
cases where that form fires. The reason is structural rather than statistical: a register's fat bounds are
themselves a decode, hence exact on the `iE == 0` branch (lengths below 4 KiB) and granule-aligned on the other,
and on granule-aligned input the R-54 renormalisation step cannot fire because both residues are zero -- the
argument `repr_exact_mask`'s own docstring makes.

*Two CHEAPER forms were tried and REFUTED, so the round trip is the predicate and has to be priced rather than
designed around.* The synthesis lane's trial reading puts that round trip at **73 logic levels and 31 CARRY4,
register to register** (+0.441 ns on tree B, into `dyn_child_repr_q`) -- nearly a full cycle, so a
shift-and-compare would have been worth a great deal. Neither candidate survives a check against the definition,
over 300,000 regions:
- comparing the cursor bits the decode reads, `(c_old >> (E+11)) != (c_new >> (E+11))`: **142,472 disagreements**,
  every one of them OVER-firing. The cursor's low bits enter `decompress` only through the threshold `A3 >= R`,
  not bit-exactly, so a move that changes `A3` without crossing `R` leaves the decode identical;
- adding that threshold, `(c_old >> (E+14)) != (c_new >> (E+14)) || (A3_old >= R) != (A3_new >= R)`: still
  **24,797 disagreements**, again all over-firing, because a change in the window bits can be exactly
  compensated by the span correction the decode itself applies -- which is the whole point of the wrap-around
  reconstruction.
Controls fired for both (dropping either term, or shifting by E+12 or E+10 instead, disagrees in 126,040 /
29,105 / 91,587 / 151,917 cases). Both failures are in the TAG-CLEARING direction, so neither is a safe
approximation: they would untag correct in-window moves. A cheap form, if one is wanted later, has to be DERIVED
from the representable region's edges and validated against the definition the same way, not guessed.

*The hazard, which must be written AT the predicate in the RTL and not only here* (the semantics owner asked for
the sentence to sit where the next person to touch the line will read it). The predicate is safe ONLY because the
register holds DECODED bounds. Evaluated against EXACT requested bounds the same test clears the tag on
**191,953 of 200,000** capabilities -- every inexact one -- because `compress_bounds` rounds the base down
(`ariane_pkg.sv:840`) and the top up (`:842-843`). Correct at the pack, catastrophic one step upstream: a later
"simplification" to the requested bounds would silently untag every inexact capability in the field. That control
is also what makes the three zeros above readable. The FIRST control tried instead varied the renormalisation
flag, which is provably inert on granule-aligned input and so read 0 for a reason with nothing to do with the
subject; the control that replaced it perturbs the decoded bounds one byte off the granule and changes the word
100,000 of 100,000 times.

*Acceptance readings 5-7, added on the semantics owner's request, all three pre-registered TOGETHER so one arm
cannot pass by trading one behaviour for another:*
5. an out-of-window move **CLEARS**: the next load through the result faults **24** (untagged operand), not a
   bounds cause;
6. a move back into the window does **NOT** restore the tag -- the self-heal is gone, which is reading 2 above
   seen from the register side;
7. an in-window move **KEEPS** the tag and the bounds.
8. **CINCOFFSETIMM explicitly, not only CINCOFFSET**, and at the edge rather than far outside: a NEGATIVE
   immediate that leaves the window by ONE GRANULE below the lower edge must clear, and one landing EXACTLY on
   the edge must keep its tag. This is the pair the measurement says is the dangerous direction -- a fix that
   passes 1-4 and fails here has traded a missed trap for a cleared tag on correct code, which is worse than the
   defect.

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

## (6) Generations in virtual node ids — the lead's item, co-designed with item 1

**The spec is the semantics owner's ruling**, `design/virtual-capstone/isa.md` ("Generations in virtual node
ids", revised 2026-10-20 to HARDWARE recycling): a virtual id becomes `{generation[29:16], index[15:0]}`, the
packing a physical id already uses; a record stores its generation in word 1 bits `[59:46]`, the 14-bit hole
between VALID and the 16-bit parent, which that ruling assigns; recycling mirrors the physical path (the kill
and a retire push the record onto the TABLE's free list, the virtual pop hands out `{generation + 1, index}`,
a record at 16383 is not pushed, the fresh path hands out generation 0); LOOKUPS compare the FULL id while
INVALIDATIONS may stay index-only; and the record layout is a module/model contract, not an image ABI, so no
domain image is rebuilt. Why it is wanted: a virtual table never recycles today, so it exhausts, and cpython
lt1 is the first corpus arm that needs recycling.

**EVERY SITE BELOW WAS READ AT THE LANE TIP `7784b37ab`, not taken from the ruling's prose.** The ruling's
own list is a starting point and one of its three named instances does not say what the source says.

*Store the generation (1 site).* `ex_stage.sv:1673-1675`, `vp_update_b`'s third arm, packs word 1 as
`{1'b0, free, linear, valid, 14'd0, parent[15:0], 13'd0, depth[16:0]}` — widths 1+1+1+1+14+16+13+17 = 64, so
VALID lands at `[60]` (which the read side already reads) and the hole is exactly `[59:46]`. That `14'd0`
becomes `node_wr_req[122:109]`, the record's generation, per the layout comment at `:1666-1667`.

*Resolve the generation (1 site).* `ex_stage.sv:1773-1774`:
`revnode_rdfill_id = vctx_rev_q ? {14'd0, rev_rd_idx_q} : {data_ruser[28:15], rev_rd_idx_q}`. The PHYSICAL arm
already reads a generation out of the record; the VIRTUAL arm hardwires zero. It becomes
`data_ruser[59:46]`, the field the writer above now stores. **Left as it is, this is the ruling's predicted
failure**: the LSU's 30-bit probe resolution would deny every recycled node permanently.

*CORRECTION to the ruling's site list, worth recording so it is not re-derived.* isa.md names "the virtual
record-read tap composes `{14'd0, index}` (ex_stage.sv:1729)". The line at that position is
`revnode_fill_id = {node_wr_req[122:109], node_wr_req[15:0]}` — the WRITE-side tap, which already composes the
full id from the payload, identically at `c7e6f604f` and at the lane tip (it was made payload-composed by
`079dc720a`, R-35). The zero-generation site is the READ-side fill's virtual arm above. Same defect, different
line; the ruling's consequence ("deny every recycled node for good") is correct as stated.

*Recycle (2 sites), and it is SMALLER than the ruling implies, because the virtual free list is already
wired.* `vp_alloc_slot`'s `*vp_free_head != 30'd0` branch (`capstone_rev_node.anvil:247-259`) already pops
from the TABLE's free list, already validates the popped record (`fn.valid == 0 && fn.free == 1`, with a
corruption dprint), already maintains `vp_free_head`/`vp_free_cnt` and already calls `vp_write_hdr1()`. Two
things are missing:
- **the PUSH under the flag.** It is gated off deliberately in two places — the revoke walk at `:77`,
  `if((*vctx_q == 1'b0) && (*revoke_index > 30'd2) && (node_in.generation != 14'd16383))`, and the kill at
  `:528`, the same shape on `kc[30+:16]`. Note the saturation guard the ruling asks for is ALREADY in both
  conditions; only the `*vctx_q == 1'b0` term has to go, and the push must target the table's slot 1 rather
  than the physical header;
- **the generation at the POP.** `:253` is `set node_id := *pop_idx` — a bare index. The record `fn` is
  already in hand one line earlier, so `fn.generation` is available without a second read: compose
  `#{fn.generation + 14'd1, 16'd0} + *pop_idx` and write the record back with the bumped generation, exactly
  as the physical pop does at `:289-303`. The fresh path keeps `#{14'd0, *head}` (`:315`).

*Admission and composition gates (2 sites), i.e. the ruling's (3a).*
- `capstone_rev_node.anvil:602` and `:604` — CSNODE's create writes BOTH new records with
  `generation = 14'd0` instead of the generation of the id the pop returned.
- `capstone_dyn_unit.anvil:618` — CSRETIRE refuses `(id >= 64'd65536) || (id < 64'd2)`, and **every id of
  generation >= 1 exceeds 65536**, so a recycled node could never be retired; its record check compares no
  generation either. CSRETIRE of a stale `{g, i}` must answer 1 and change nothing.

*Lookup consumers that must see all 30 bits (rule 3).* The LSU revocation cache
(`load_store_unit.sv:1301`, `:1416`, `:1437`, with `revnode_fill_id_i` 30 bits wide at `:211`) and the PC
tracker (`commit_stage.sv:408-412`). Both already carry 30-bit ids; what makes them correct is the two
generation sites above, not a change here. **No compare is widened anywhere**, and `pmp_data_if.sv:107` stays
index-only for the reason its own comment records — under item 1 the CPMP trackers act only on physical
broadcasts and are out of this question entirely.

*Explicitly OUT of scope, because `#{14'd0, ...}` looks uniform and is not.* The broadcast mask
(`capstone_rev_node.anvil:6`) stays index-only, which rule 3 permits and item 1's namespace bit already
qualifies. The memory ADDRESS composition (`:180`, "address = index only") must stay generation-free or it
addresses the wrong record. The walk's `next`/`prev` link compositions are table indices whose staleness the
record's own generation check catches.

**Acceptance, pre-registered.** The arms are a stale copy REFUSED and a fresh one PASSING, with three
controls:
1. a capability on a recycled index: after kill, push, re-pop and re-node, a load through the STALE copy is
   refused while a load through the NEW capability at the same index passes. This is the item;
2. control, NO REVOKE: the same sequence without the kill must pass on both, so the refusal is attributable
   to the recycle and not to the sequence;
3. control, PHYSICAL: the existing physical recycle arms (`r12-recl-*`) must read exactly what they read
   today — `r12-recl-cpmp`'s seven deltas `0 1 1 0 1 0 1` and count 4 are already measured on this tip under
   item 1, so any change there is item 6's regression;
4. control, SATURATION: a record at generation 16383 is not pushed, so its index is retired from the pool
   rather than wrapping to 0 and aliasing a live id;
5. CSRETIRE of a stale `{g, i}` answers 1 and changes nothing (the gate at `capstone_dyn_unit.anvil:618`
   must no longer refuse it for being above 65536).

**Order of work**, because the consumers are what make this safe: read every generation-reading gate first
(`capstone_rev_node.anvil:77`, `:327`, `:347`, `:353`, `:374`, `:421`, `:472` — four of which already compare a
generation taken from a LINK field, so some links are already expected to carry one), then the two generation
sites, then the recycle, then the two admission gates.

## What the 95-test sweep does and does not certify for item (2)

**The sweep is a BUILD and REGRESSION check for item (2), not a behavioural one, and the behavioural evidence
is the four `va-srevroot` arms plus the `va-fetchsplit` arms.** That is checked rather than argued: item (2)'s
change sits entirely inside `(*commit_req.is_runv == 1'b1)`, a non-runv walk reaches the unchanged
`cur_idx == 66 -> walk_end()` branch exactly as before, and reading the sources of all 95 sweep tests, 92 of
which resolve, **ZERO contain a CSRUNV or funct7 0x24** -- the other three are the random generators already on
`sweep-expected-missing.txt`. So the `is_runv` walk is unreachable for the entire sweep. Said plainly so that
"the sweep identical" claims only what it can.

**CLOSED 2026-10-20: the baseline exists, the comparison PASSES, and the 29 rc=1 results are attributed.**
A sweep on `c7e6f604f` (95/95) compared directory-to-directory against the preserved item-2 run
(`d76acdb59`) reads **92 identical, 0 differ, 0 missing, 3 excused as expected-missing on both sides**, with
the detector silent on both. Same budget (2,000,000) and same testlist hash on both sides, so the provenance
gate ACCEPTED the comparison and merely NOTEd the differing commits and core trees -- its first real use, and
it behaved as designed. Artefacts: `.sup/sweepbase-c7e6f604f` and `.sup/sweepitem2-d76acdb59`, each with its
PROVENANCE.
  - **Item (2) is sweep-identical.** That is the gate, and it is now met on a comparison that differs only in
    RTL: `testlist_capstone.yaml` untouched, none of the 95 sources changed, core trees
    `6d20dd08905ce6f3` against `fe7cc910e6b0c3e1`.
  - **The 29 rc=1 results, including the 23 that reach the 2,000,013-cycle limit, are PRE-EXISTING on this
    lane and not item (2)'s.** They occur identically on both sides. `capsbi-init` and `capenter`, which I
    refused to attribute earlier, are among them. They remain a separate question about when this lane last
    had a meaningful functional gate -- the stale 400,000-budget baseline hid them -- and nothing in this batch
    depends on resolving it.

**The superseded gap, for the record.** The committed
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

## RULED, not open: CSNODE is admitted at S by design (rule 3b)

Recorded here because it outlives the batch and because the verified half is cheap to state. **It is not a
defect claim**: the implementation matches its own stated intent, and what conflicts is the premise that CSNODE
is reachable only where the model's CSMINT is.

**Verified, three ways.** The decode admission (`decoder.sv`, funct7 0x54, identical to CSRETIRE) is

    if (CVA6Cfg.CAPSTONE_EXT && !(vctx_table_i && !(sup_active_i && !debug_mode_i)
                                  && !(CVA6Cfg.RVU && priv_lvl_i == riscv::PRIV_LVL_U))) illegal_instr = 1'b1;

so CSNODE is legal iff the table-namespace flag is set, the caller is not inside a running supervised context,
and the privilege is not U. **There is no M-mode term**, and `vctx_table_o` is 1 at S whenever
`!sup_active_q && priv_lvl_q == PRIV_LVL_S && srevroot_q != 0` (`csr_regfile.sv:2886`) -- a condition an S
caller can create for itself, since srevroot is 0x5C1. **Measured**: `va-srevroot -DSREV_CSNODE` issues a
CSNODE at S after an escape and it returns a node id (s5 = 0x4, 2026-10-20). And `create_req`
(`capstone_rev_node.anvil:586-606`) receives rs1's node id, uses it only in a `dprint`, and writes two FRESH
records with `valid = 1'd1` -- **rs1's prior node is never read or validated.**

**The shape of the concern, NOT a measured escape.** An S caller could set its own srevroot to a table it
controls, hand CSNODE a capability whose node was revoked, and receive it re-noded onto a fresh valid node that
later revocation checks resolve as live. Two things are unestablished and must not be skipped: whether a
caller-built table passes the allocator's header checks, and whether the supported flows ever leave S with a
non-zero srevroot -- though **R-58 is exactly a defect that leaves the caller holding one it did not set**,
which is how the measured instance above arose.

**RULED 2026-10-20 by the semantics owner (isa.md rule 3b): BY DESIGN, not a defect, and NO exploit arm is
needed.** The threat model is that S is trusted. S already holds CSMINT, a trusted root-creation operation that
mints ANY capability -- any bounds, any rights -- from a descriptor onto fresh nodes, so re-noding one through
CSNODE gives S strictly less than it already has. Revocation here protects against U contexts, never against
the trusted kernel. The arm budget goes to item 6's retire control instead.

**Both legs of that ruling check out, and I verified them rather than accepting them** -- the second because it
is the one that keeps the design safe. Q-23 (CLOSED 2026-10-18) rules that **"trusted S performs NO capability
memory access"**: an integer-based LDC or STC from S reads 24 on the RTL and, after QEMU was tightened to it, on
both; the module gets what it needs through the CSMINT descriptor instead, and that entry names "CSNODE inside
the window (any tagged rs1)" as part of the supported emulation sequence -- so the unvalidated rs1 is
documented and deliberate, not an oversight. Adding the two paths I checked myself: the escape restores the
CALLER's own saved registers, not the context's, and a plain `ld` of a frame slot yields the raw word WITHOUT
its tag, so neither hands S a context's capability.

**The two conditions the rule depends on, so a later change cannot break them silently:**
1. **U and a running supervised context must never decode CSNODE.** The predicate quoted above excludes both,
   and items (1) and (6) must keep that true -- neither touches the decoder, which is the easy way to keep it.
2. **No flow may hand S a context's capability in a register.** Backed by Q-23 plus the two paths above.
If either changes, this reopens. My R-58 observation is consistent with the ruling rather than against it: R-58
leaves S holding a non-zero srevroot, which is a CSR value, not a capability in a register.

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
