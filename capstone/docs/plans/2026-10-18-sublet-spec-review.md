# Sublet lifetime rules: what is still open between QEMU and the RTL (review 2026-10-18, reduced 2026-10-24)

The Sublet lifetime operations are specified in
[`docs/design/virtual-capstone/isa.md`](../design/virtual-capstone/isa.md). This list keeps only the
behaviour that an implementation can observe, that the two implementations do not agree on, and that no
ruling, ISSUES entry or plan on dev decides yet. Everything else the first version of this review listed has
since been ruled or is tracked elsewhere (the generation ruling at isa.md "Generations in virtual node ids",
Q-10 to Q-20, R-21, R-28, R-45); the first version is the lane commit dfacacb91618 and its revision
db30acae8675. Each item needs a decision, and some may be a defect on one side rather than behaviour to keep.

## Sources

| Key | What | Revision |
|---|---|---|
| QEMU | capstone-qemu, the superproject's pin; `target/riscv/` | b290f5656600 |
| RTL | capstone-ariane, the bitstream-2 seal on both boards since 2026-10-24 (`2026-10-23-next-bitstream-scope.md`) | dcc6daf6e |
| `dyn_unit.anvil`, `rev_node.anvil` | `core/anvil_build/capstone_dyn_unit.anvil`, `capstone_rev_node.anvil` | RTL |
| `ex_stage.sv`, `commit_stage.sv`, `ariane_pkg.sv` | `core/`, `core/include/` | RTL |

"Table namespace" is the RTL predicate defined at isa.md's CDERIVE paragraph.

## Open items

**1. What a refused or interrupted table operation leaves behind.** Q-20 rules mutation-free refusal only for
a walk under the virtual-context flag that exceeds its visit bound, cycles or meets an unreadable record
(`docs/ref/ISSUES.md`, Q-20's ruling), and R-28 tracks interruption. Two residuals are not covered:
- RTL MREV deepens the operand's subtree before it allocates the handle (`dyn_unit.anvil:128-161`: the check
  pass, then the write pass, then `mrev_req`), so a resource refusal of the allocation (status 2, cause 30)
  follows writes. The comment there accepts this for a FLUSH (the replay deepens again); a refusal is not a
  replay, and the rule has to say whether a one-deeper subtree is an allowed residue.
- The contract for a reader of the records: which fields of a record a walker or a lookup may read while
  another operation is between its writes, and after a replay.

**2. Fault precedence for malformed metadata, and oversized depth.**
- Inside `cap_rev_derive`, QEMU refuses a broken insertion back-link (`next.prev != parent`) with CORRUPT
  before it tests depth and capacity (`cap_rev_tree.c:140-145`); the operation's node-pressure check,
  `capstone_require_node`, runs before that call and can escape or fault first (`op_helper.c:1248-1251`). The RTL's derive splices without that check
  (`rev_node.anvil:549-589`). The rule should decide each malformed case (reserved id, FREE+VALID, unissued id,
  oversized field, malformed header) and its precedence against node pressure.
- The RTL reads 17 bits of a record's depth (`ex_stage.sv:1527-1529`, `data_ruser[16:0]`) against the 30-bit
  architectural field, so a wider stored depth is truncated silently. Refuse it, or state the supported width.

**3. The stored type codes.** The two capability words share the field split but not the raw type codes: RTL
LINEAR 1, NONLIN 2 (`ariane_pkg.sv:662-669`), QEMU LIN 0, NONLIN 1 (`cap.h:28-34`). The feasibility addendum's
"the same layout bit for bit" (`virtual-capstone-rtl-feasibility.md:196-202`) holds for the field positions
only. Decide whether the two encodings must interoperate (one code table, which keeps LCC's software-visible
numbering), or record that they need not. The renormalisation difference is Q-16.

**4. DROP then REVOKE on the physical profile.** Take `H = MREV(L)`, H writable, no other live linear
descendant. After `DROP L; REVOKE H` the RTL returns LIN: its walk tests linearity only on records that are
still VALID (`rev_node.anvil:80-126`). QEMU's walk folds every record's linear flag in, dead or not
(`cap_rev_tree.c:193-205`, `retain_data && !rec.linear`), so it returns UNINIT. Q-18 rules DROP itself; this
interaction is not covered by it.

**5. Depth exhaustion: terminal or resumable.** The RTL reports every cause 30 inside CSRUNV as resumable
kind 5, node pressure (`commit_stage.sv:264-268`). QEMU reports an ordinary fault as terminal kind 2 and
raises kind 5 only on its separate pressure path (`capstone_supervisor.c:489-500`, `:879-883`). Depth exhaustion cannot
be repaired by a collection, so a resumable kind 5 for it can loop.

**6. TIGHTEN (two differences outside Q-13 and R-21).**

| Case | QEMU (`op_helper.c:1500-1536`) | RTL (`dyn_unit.anvil:422-439`) |
|---|---|---|
| untagged operand | 26 | 24 |
| W removed from a linear capability | clears the node's linear flag (`cap_rev_delin`) | does not touch the node |

**7. SPLIT edge cases outside Q-10.**

| Case | QEMU (`op_helper.c:1430-1497`) | RTL (`dyn_unit.anvil:185-216`) |
|---|---|---|
| `rd == rs1` | after the operand checks, does nothing: no liveness check, no allocation, no bounds change | no such path |
| tag check | rs1 tagged and rs2 untagged, else 26 | faults only when rs1 is untagged AND rs2 tagged, so a tagged rs2 with a valid parent passes |
