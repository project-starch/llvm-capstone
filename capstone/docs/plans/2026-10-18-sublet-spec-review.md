# Sublet lifetime rules: what the spec leaves open (review, 2026-10-18)

The Sublet lifetime operations are specified in
[`docs/design/virtual-capstone/isa.md`](../design/virtual-capstone/isa.md): the instruction table and the
section "Sublet child lifetimes". This review compares that text with the two implementations. It lists:

- the statements that were out of date (corrected in the same change as this document);
- the behaviour that an implementation can observe but the spec does not decide.

The point is ordering. Q-20's walk bound is ruled for the RTL and not yet implemented at the RTL revision
read here. Each open item below should get a rule BEFORE that and any further lifetime-table work, so the
RTL follows the spec rather than a reading of QEMU's helpers.

## Sources and citation key

| Key | What | Revision |
|---|---|---|
| `isa.md` | `capstone/docs/design/virtual-capstone/isa.md` | dev f78620223615 (before this change) |
| QEMU | capstone-qemu, `virtual-capstone-v7`, the reference pin recorded in `2026-10-17-vp2-corpus-on-silicon.md` | b290f5656600 |
| `op_helper.c`, `cap.h`, `cap_compress.c`, `capstone_table.c`, `capstone_supervisor.c`, `cpu.c`, `cap_rev_tree.c`, `cap_rev_table.c` | in `target/riscv/` | QEMU |
| `trans_capstone.c.inc` | in `target/riscv/insn_trans/` | QEMU |
| RTL | capstone-ariane, branch `virtual-profile-2-c` (its head when read, 2026-10-18) | 8aea7a41 |
| `dyn_unit.anvil`, `rev_node.anvil`, `unit.anvilh` | `core/anvil_build/capstone_dyn_unit.anvil`, `capstone_rev_node.anvil`, `capstone_unit.anvilh` | RTL |
| `ariane_pkg.sv`, `ex_stage.sv`, `decoder.sv`, `commit_stage.sv`, `csr_regfile.sv` | `core/`, `core/include/` | RTL |
| `bounds-model.py` | `verif/tests/custom/capstone/bounds-model.py` | RTL |
| `capability.h`, `heap.c`, `mallocng.md` | `capstone/runtime/include/capstone/`, `capstone/runtime/virtual/` | dev f78620223615 |

"Table namespace" below is the RTL's `vctx_table` predicate (`csr_regfile.sv:2876-2878`). With the
extension enabled it is on in three cases:
- in a supervised context with `csupctl.virt`, or under CSRUNV;
- outside a supervised context, in S with `srevroot != 0`;
- outside a supervised context, in M with `csupctl.table`.

"Translation context" is `vctx` (`csr_regfile.sv:2886-2887`): a supervised context with `csupctl.virt` or
under CSRUNV, AND `satp` in an SV mode. The RTL selects several Sublet
behaviours on the table namespace, not on "the profile".

Method: every claim was read from the sources at these revisions. Two adversarial review passes (Codex)
checked the claims and the proposals. No simulation or board run was made for this document.

## Corrected in this change (factual, no ruling needed)

- **CDERIVE/CREVOKE decode in every profile.** QEMU has no profile gate (`trans_capstone.c.inc:109-121`). The
  RTL implements both in both profiles (`decoder.sv:1273-1292`, since d93a8d66b). The isa.md text said they
  decode only in the virtual profile.
- **One supervisor slot by default.** `x-capstone-supervisor-slots` defaults to 1 (`cpu.c:1897`, Q-26). The
  isa.md text said 32.
- **The capability word.** The RTL has QEMU's field split: node 30, perm 4, type 3, bounds 27
  (`ariane_pkg.sv:631-650`, since d34c26002). But the raw type codes differ:
  - RTL: LINEAR 1, NONLIN 2 (`ariane_pkg.sv:662-669`);
  - QEMU: LIN 0, NONLIN 1 (`cap.h:28-34`);
  - both store the raw code in the word (`ariane_pkg.sv:881-885`, `cap_compress.c:72`).

  So the two encodings are not binary-interchangeable. The isa.md text still gave the RTL 28 bounds bits. The
  feasibility addendum's "the same layout bit for bit" holds for field positions only.
- **Bounds renormalisation.** The RTL renormalises bounds (`ariane_pkg.sv:827-835`). The QEMU reference pin
  does not (`cap_compress.c:40-62`, Q-16). The residue-form encoder that an RTL comment pairs with,
  capstone-qemu 4a21b485b01f, sits on a branch that diverged from the pin (27 ahead, 4 behind); it is not in
  b290f5656600.

## P0: decide before the next lifetime-table change in the RTL

**1. A transaction rule for every table-mutating instruction.** This covers CDERIVE, CREVOKE, MREV, linear
SPLIT, REVOKE and CSRETIRE. The rule should state:
- the checks;
- their order and cause;
- that a refused instruction leaves registers, header and records unchanged.

Where an instruction reports a status in rd instead of faulting, that status write is part of its result and
is exempt. CSRETIRE writes 1 to rd on a rejected retirement (`isa.md:41`, `capstone_table.c:207-215`).

The current state:
- QEMU preflights before it writes: `dup_before` at `cap_rev_tree.c:65-88`, and `walk_run` and its callers
  from `cap_rev_tree.c:183`.
- RTL CREVOKE invalidates the child BEFORE its walk (`rev_node.anvil:563-566`).
- Legacy REVOKE's response carries only a linearity bit, so it cannot report a corrupt table
  (`unit.anvilh:579-580`, `rev_node.anvil:646-665`).
- No RTL walk except CSRETIRE's dry pass has a visit bound. The dry pass checks cycle, index and visits
  (`rev_node.anvil:145-168`, `:617-643`).
- RTL MREV writes the deepened depths before it allocates the handle (`dyn_unit.anvil:128-161`,
  `rev_node.anvil:128-135`, `:432-455`). A resource refusal can therefore follow writes. A pipeline flush
  abandons the sequence and replays it (`dyn_unit.anvil:140-144`). That is an interruption, not a refusal,
  and the rule must cover it separately.
- The rule must also say how a failed metadata read propagates. For MREV it must say which direct-parent
  relations change: MREV deliberately reparents its operand's node under the new handle
  (`cap_rev_tree.c:114-115`, `rev_node.anvil:455`), and every other relation must survive.

Q-20's ruled bound goes in as ruled: under the virtual-context flag, the bound is `min(next-unused, 65,536)`,
and physical walks are unchanged. Meeting the rule needs an error-capable response path in the RTL's
rev-node interface, so the rule comes first.

**2. Ordered fault tables for CDERIVE and CREVOKE, every stage included.**

CDERIVE on QEMU (`op_helper.c:1212-1253`, `cap_rev_tree.c:137-145`):

| Step | Fault |
|---|---|
| parent untagged, or size tagged | 24 |
| parent not NONLIN | 26 |
| table open and parent liveness | 25 |
| size 0, or the window is not inside the parent | 29 |
| compact round trip fails, unless exact bounds is on | 29 |
| node pressure (`capstone_require_node`) | — |
| parent liveness again | 25 |
| back-link check | 29 |
| depth or capacity | 30 |

CDERIVE on the RTL:
- the same operand checks and window test (`dyn_unit.anvil:259-276`);
- the round-trip 29 only under the table namespace (`dyn_unit.anvil:277-278`, input `ex_stage.sv:1213`);
- the rev-node unit re-reads the parent (dead → 25), tests depth (30) before allocation, and has no back-link
  check (`rev_node.anvil:471-495`);
- its corrupt-table status maps to 29 under the table namespace and to 25 otherwise
  (`dyn_unit.anvil:284-289`).

CREVOKE:
- the decoder makes a nonzero rd illegal (2) on both sides (`trans_capstone.c.inc:115-118`,
  `decoder.sv:1286-1292`);
- then 24 → 26 → 25 (rs1) → 25 (rs2) → 27 (`op_helper.c:1270-1287`, `dyn_unit.anvil:310-327`);
- then a second liveness check before the relationship test: QEMU rechecks parent and child (25,
  `cap_rev_tree.c:166-168`), the RTL rechecks the child's validity and generation (25,
  `rev_node.anvil:519-522`);
- then 29: not a direct child, self, or linear (`cap_rev_tree.c:169-176`, `rev_node.anvil:523-525`);
- QEMU then preflights the subtree; the RTL walks after it has mutated (item 1).

Malformed metadata has more than one outcome today:
- QEMU: a failed table open or lookup gives 25 (`cap_rev_table.c:85-107`, `:142-150`;
  `op_helper.c:1227-1229`); a bad insertion neighbour gives 29 (`cap_rev_tree.c:140-142`).
- The rule should decide each case: reserved id, FREE+VALID, unissued id, oversized field, malformed header.
  It should also decide their precedence against node pressure. One universal "corrupt = 29" would be a new
  ruling, not a description.

**3. A coherence protocol for collection.** isa.md requires a sweep before an id is reused. It does not say how
trusted software makes the RTL's caches see the changed table.
- The RTL's revocation trackers forget on a `csupctl` bit-2 strobe (`csr_regfile.sv:2013`, cleared at `:1177`,
  wired at `ex_stage.sv:1417`).
- Rewriting the same `srevroot` flushes the pipeline (`csr_regfile.sv:2014`) but does not invalidate those
  caches.
- Specify the required sequence before a resume or a reuse.
- Add the end-to-end test isa.md lists as missing (`isa.md:198-200`): a stale reference reloaded from memory
  after its id was collected and reused.

**4. The capability word's type codes.** See the correction above. Choosing one table is an ABI decision for
the format owner. It has to keep LCC's software-visible numbering; the RTL's LCC subtracts one
(`dyn_unit.anvil:403`). It also has to settle the renormalisation difference (Q-16).

## P1: semantics to state explicitly

**5. CDERIVE outside the table namespace (Q-15, kept as ruled).** The RTL skips the round trip there. It
recompresses the child at writeback, which rounds outward (`dyn_unit.anvil:277`, `ex_stage.sv:1751-1753`), so
an inexact child covers bytes outside `[cursor, cursor + size)`.
- This does not hold at an unrepresentable top, where the decode gives an empty capability
  (`ariane_pkg.sv:818-826`).
- The Q-15 ruling states that a child can come out wider than its parent. No counterexample with a parent the
  RTL can actually hold has been produced. `bounds-model.py:73-92` computes `exceeds_parent` for any given
  interval.
- The rule text should name the outward rounding as a known widening of authority on the physical profile,
  keep QEMU's exact-bounds branch (pinned mallocng relies on it: `mallocng.md:116-125`, `heap.c:174-177`), and
  name the condition for removing the physical behaviour.

**6. DROP of NONLIN outside the table namespace (Q-18, kept as ruled).** RTL DROP clears that node's VALID with
no walk and no unlink (`rev_node.anvil:350-358`).
- A CDERIVE subtree under it stays live, and can no longer be CREVOKEd through the dead parent (25).
- It is reclaimed only by a REVOKE, or a CREVOKE of a live ancestor whose run contains it. Nothing guarantees
  that such an ancestor exists.
- Interaction: take `H = MREV(L)` with H writable and no other live linear descendant. Then `DROP L; REVOKE H`
  returns LIN on the RTL, because the walk skips the dead L (`rev_node.anvil:68-106`). QEMU returns UNINIT,
  because its node stays linear (`op_helper.c:1121-1123`, `:1176`; `cap_rev_tree.c:203`).

**7. A privilege × namespace table for CDERIVE and CREVOKE.** With the extension enabled, neither instruction
has a mode gate on either side. On the RTL, U outside a context uses the physical pool (`decoder.sv:1278-1291`,
`csr_regfile.sv:2876-2878`). QEMU's protected-U experiment can select the guest table outside a context
(`capstone_table.c:98-109`). Separate these predicates:
- feature enabled;
- translation context;
- table namespace;
- protected U;
- M-mode emulation.

**8. Field widths fail closed.** The architectural widths are:
- 30 bits for depth, parent and ids;
- u32 for links (`isa.md:383-386`, `:403`).

The RTL reads 17 bits of depth and 16 of parent, and silently truncates wider stored values
(`ex_stage.sv:1431-1433`, `:1584-1585`). An implementation may support less only if it refuses at admission
and at lookup. Limits should be discoverable through a versioned interface. That interface must not be
`urevavail` or `scapctl` (illegal by default) or a reserved header field.

**9. The node reserve and event kinds.** Inside CSRUNV, the RTL applies the 256-id reserve only under the
translation context (`ex_stage.sv:1409-1410`, `rev_node.anvil:239-246`, `:764-766`).
- The RTL reports every cause 30 as resumable kind 5 (`commit_stage.sv:267`).
- QEMU reports an ordinary cause 30 as terminal kind 2 and uses a separate pressure path for kind 5
  (`capstone_supervisor.c:489-500`, `:879-883`).
- Depth exhaustion cannot be repaired by a collection, so a resumable kind 5 for it can loop. Decide it.

**10. Record states after a kill.** Distinguish three states and say which fields each consumer may read:
- retired;
- linked-dead (walkers traverse it);
- FREE (its free-list links stay meaningful with VALID = 0) (`cap_rev_tree.c:193-208`,
  `cap_rev_table.c:195-200`, `rev_node.anvil:100-104`).

Require unlinking before recycling. Today QEMU, and the RTL under the table namespace, null only the retired
head's links (`cap_rev_tree.c:204-206`, `:273-275`; `rev_node.anvil:540-554`). Outside it the RTL keeps the
links or reuses `next` for its free list (`rev_node.anvil:77-86`, `:528-539`).

**11. Stale capabilities in transport (Q-11, Q-19).** A matrix by type and by machine property for LDC and STC
of a capability whose node is dead, including consuming LINEAR/REV transfers.
- QEMU keeps the tag only for NONLIN under `capstone_u_mode` (`op_helper.c:2144-2158`).
- The RTL checks no liveness of the loaded or stored referent (`unit.anvilh:721-745`, `dyn_unit.anvil:745-753`).
- Include Q-19's rule: a non-consuming load must not require a writable PTE.
- Decide Q-11; until then the matrix marks the difference.

**12. TIGHTEN, the operation that removes MANAGE (Q-13).** QEMU and the RTL differ in five ways
(`op_helper.c:1504-1536`, `dyn_unit.anvil:422-439`):

| Case | QEMU | RTL |
|---|---|---|
| immediate > 15 | clamps | 29 |
| escalation | 26 | 29 |
| untagged operand | 26 | 24 |
| non-copyable source (UNINIT included), `rd != rs1` | moved, i.e. cleared (`op_helper.c:1524-1528`) | not moved |
| W removed from a linear capability | clears the node's linear flag | does not |

Both check permissions before liveness. A fault 29 for an immediate above 15 keeps
`capstone_cap_without_manage` working; it uses 0-7 (`capability.h:82-95`).

**13. SPLIT edge cases** (`op_helper.c:1435-1497`, `dyn_unit.anvil:185-216`):

| Case | QEMU | RTL |
|---|---|---|
| `mid == base` | refused, 26 | accepted |
| other endpoint failures | 26 | 29 |
| `rd == rs1` | skips liveness, allocation and bounds changes after the first checks | no such path |
| tag check | rs1 tagged and rs2 untagged, else 26 (`op_helper.c:1435-1436`) | ANDs the two conditions, so a tagged rs2 with a valid parent passes |

For a linear split, the lower, original node keeps the existing descendants, and the new sibling goes before
it (`cap_rev_tree.c:90-115`, `rev_node.anvil:380-388`). Each difference above needs an explicit decision;
some may be defects on one side rather than behaviour to keep.

**14. Linearity invariant vs a late MREV.** After DELIN or a derivation, a retained LINEAR copy can still take
a late MREV handle and revoke a derived subtree (`docs/ref/ISSUES.md` Q-15, "LINEAR duplicates").
- On MREV, the RTL sets the node's LINEAR flag (`rev_node.anvil:455`); QEMU keeps it
  (`cap_rev_tree.c:114-117`).
- State the invariant, and extend QEMU's existing late-MREV test to cover it.

**15. Ordering as a guarantee, not a pipeline rule.** Specify that no younger committed access uses authority
ended by CREVOKE, REVOKE or DROP, plus the interruption and partial-mutation rules. "CREVOKE flushes younger
instructions" (`commit_stage.sv:794-809`) is how the RTL meets this today; it is not the guarantee itself.

## Open items already in isa.md: proposed decisions

- **CDERIVE rd = 0: keep as it is (it allocates).** `capstone_cap_without_manage` relies on
  `CDERIVE x0, cap, x0` to refuse a wrong operand atomically (`capability.h:66-79`).
- **CSRETIRE rs2:** QEMU ignores it at the pin (`trans_capstone.c.inc:502`, `capstone_table.c:207`). Decide
  between "ignored" and "reserved, must be zero". The latter changes the accepted encoding.
- **NOT_CHILD vs a corrupt table, both 29 with tval 0:** defer to an explicit ABI change. The spec requires
  tval 0 (`isa.md:369-370`), and the RTL's status 3 would have to be split.
- **Representability:** specify the encoder round-trip predicate with its boundary cases (`isa.md:162-169`),
  from `bounds-model.py`'s exactness functions. Note that the RTL renormalises and the pinned QEMU does not.
- **Ending a child independently of the parent reference's bounds:** already normative in the authority model
  (`isa.md:126-155`). Close the open item at `isa.md:186-188` as decided, if the semantics owner agrees.
  Narrowing that dropped MANAGE would break mallocng's narrowed management copies (`heap.c:64-69`, used at
  `:129`, `:141`).
- **Mnemonic:** CREVOKE everywhere, encoding unchanged, `cskill` kept as an alias.
- **LLVM MC support** for CDERIVE, CREVOKE, CSRUNV, CSMINT and CSRETIRE: diagnose reserved fields, keep CREVOKE
  distinct from legacy REVOKE, and keep raw `.insn` encodings valid.

## Process proposal

1. Rewrite the Sublet section as numbered normative rules (MUST, per profile where the profiles differ).
   Implementation notes go apart from them: QEMU properties, and RTL deviations with their Q-ids. Every
   Q-ruling edits a numbered rule.
2. Keep a conformance table from rule to tests: capstone-qemu `tests/sublet-lifetimes` and the RTL's directed
   arms. The same table is the Tier-0 port list for the RTL harness.
3. RTL commits cite rule ids, not QEMU helper names.
4. Version the section, and freeze a version with the next bitstream.
