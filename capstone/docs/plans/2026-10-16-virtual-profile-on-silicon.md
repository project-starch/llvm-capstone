# The full virtual Capstone profile on silicon: the RTL lane's task (2026-10-16)

**The lead's decision, 2026-10-16:** silicon is to run the **full** virtual Capstone profile. The landed virtual stack
is to run on the board as it runs on QEMU's virtual profile: ABI v5, local mallocng and the virtual application ports.
This answers the first of the three questions in the RTL lane's audit,
[`virtual-capstone-rtl-feasibility.md`](virtual-capstone-rtl-feasibility.md) (3fe4541c1299, 2026-10-10). That
audit's verdict is "partly, and not as a port". Its item list (A-H) and its order of work are the starting point here.

Today's bitstream, `caplifive_sublet_d93a8d66b`, implements the physical profile plus CDERIVE/CREVOKE. Every silicon
result so far is a physical-profile result: the bug corpus was rebuilt for the silicon ABI, not run from its virtual
images ([`2026-10-15-bug-corpus-on-fpga.md`](2026-10-15-bug-corpus-on-fpga.md) §1).

## What done means

1. **One bitstream, both profiles.** A virtual-context flag selects the virtual profile per context, as the audit's
   item B proposes. The physical profile stays, because the board campaign
   ([`2026-10-16-board-campaign-all.md`](2026-10-16-board-campaign-all.md)) and every recorded silicon result run on it.
2. **The reference is QEMU.** capstone-qemu `virtual-capstone`, at d35af47b163c or its head when work starts, with the
   guide `docs/design/virtual-capstone/*`. Where the RTL and QEMU disagree, the disagreement goes into an ISSUES
   Q-entry; it is never settled silently.
3. **Acceptance on the board, in this order:**
   1. the landed virtual launcher runs, with `scapctl` bit 8 (exact bounds) set;
   2. the virtual runtime suites run unmodified;
   3. the bug corpus's virtual images (`virtual-malloc`, `virtual-nested-pools`, by sha256) read the verdicts
      recorded for them on QEMU.
   The third is the payoff: the corpus then runs on silicon as built, not rebuilt.
4. **The physical regression holds:** the S-16 bare set, the Sublet directed arms and one corpus set read the same on
   the new bitstream as on d93a8d66b.

## First deliverable: re-baseline the audit

The audit read the virtual spec as of 2026-10-07/08 and QEMU 1a6dd20732. Since then:
- **capstone-qemu `virtual-capstone`:**
  - #21, Sublet lifetimes (CDERIVE/CREVOKE in the virtual profile, and the stale-reload rule: a reloaded stale
    non-linear capability keeps its tag in the virtual profile only);
  - #22, #23, #26 and #29, performance;
  - #25, untagged capability arithmetic;
  - #28, x0 capability writes;
  - #30, floating point in virtual C;
  - acbad0f7ec, CDERIVE/CREVOKE decoded on the physical profile;
  - #27, MOVC with an untagged source (Q-04).
- **dev:**
  - 6576dca09c37: virtual mallocng manages lifetimes with CDERIVE/CREVOKE;
  - 601c8d52c260, cbbc2ca958af and b02ddf9043cc;
  - 7901153f10fa: the superseded allocator backends retired.

Write an addendum to the audit that says, item by item, what changed. Include where d93a8d66b's CDERIVE/CREVOKE
differ from the virtual profile's. The known case is rounding: silicon rounds an inexact CDERIVE's base down.

## The decision that gates qualification: exact bounds (audit item C')

The full profile needs exact bounds. ABI v5 and local mallocng run only under `scapctl` bit 8 (`CPONVVM5`), and the
launcher runs nothing without it. The audit's option (iii), representable bounds enforced by the allocator, changes
the runtime rather than the hardware, so it is not the full profile. That leaves:
- **(i)** a wider capability format: two granules per capability;
- **(ii)** a side store of 24 bytes per capability-bearing granule.

**The choice is the lead's. Second deliverable: what the lead decides it from.** A costed comparison of (i) and (ii):
- what each changes, unit by unit: LSU, write buffer, forwarding, the D-cache metadata lane, the tag region, the
  rev-node unit;
- what each changes in the compiler and runtime (pointer size, for (i));
- memory traffic per capability load and store;
- which cones on the standing UNOPTFLAT list each one touches;
- an estimate in bitstream cycles;
- a recommendation.

With it, a proposal for the format freeze: 27 or 28 bounds bits, 31 or 30 node-id bits, generations or none. The
RTL's numbers have been fixed since April, and QEMU's can still move. Who owns the model's semantics is the lead's to
name. Propose an owner.

## Work that does not wait for the decision

These items follow the audit's order. Each is proven in simulation, against directed tests ported from QEMU's virtual
suites, before it goes to synthesis:
- **B, translation in C mode.** This is the largest item and needs no format decision:
  - the virtual-context flag;
  - translation under M privilege with USER permissions, for data and for fetch;
  - page faults as a resumable escape kind;
  - the consuming LDC's same-granule rule;
  - an audit of every `PRIV_LVL_M` assumption.
  The audit plans one bitstream cycle for it, with a bare test before any runtime.
- **A, the lifetime table:**
  - an `srevroot` register;
  - the 16-byte record, the reserved ids and the header;
  - the flat format first, the paged format second;
  - the generation path removed, and replaced by the collection flush (H).
- **D, the `CSRUNV` front end:**
  - frame addressing and actions 0-3;
  - the page-fault, node-pressure and service-ECALL kinds;
  - `CSMINT`/`CSRETIRE` over the new table.
  The spec leaves context storage for 32 contexts open.
- **E, the policy list,** with the relaxation of SPLIT/MREV under supervision. It folds into D.

The audit estimated four to five bitstream cycles, plus the format decision. Options (i) and (ii) add to that.

## Constraints

- **No reflash without the lead's own word.** The board campaign is running on d93a8d66b and a reflash interrupts it,
  so agree the timing with the board lane.
- **CLAUDE.md "Simulation green is not synthesis clean" applies in full:**
  - the lint gate and claim-auditor on every change;
  - a predicted reading written down before synthesis, and used as the go/no-go;
  - every change a bitstream needs batched into it;
  - no change to the synthesis flow;
  - a hash handed on only with its lint and synthesis numbers.
  A and B both reach the rev-node, LSU and D-cache cones that carry the loops and the WNS. They are the high-risk
  class that rule names.
- **No paper edits.**

## Status

| item | owner | state |
|---|---|---|
| re-baseline addendum | RTL lane | to do |
| exact-bounds options, costed (i) vs (ii), format-freeze proposal | RTL lane | to do; the lead decides from it |
| B translation in C mode | RTL lane | to do (simulation first) |
| A lifetime table | RTL lane | to do |
| D `CSRUNV`, `CSMINT`/`CSRETIRE` | RTL lane | to do |
| E policy list | RTL lane | to do, folded into D |
| board acceptance (done-means 3, 4) | board lane | after the first virtual bitstream |
