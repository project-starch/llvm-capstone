# Next bitstream after the batch: scope for one decision (RTL lane, 2026-10-23, read-only)

*Written at the semantics owner's request while the batch bitstream c721c1a2a awaits its first boot. The goal set by the
lead: the whole corpus readable on silicon. The module and QEMU sides are the semantics owner's and the compiler lane's;
the live-peak and lifetime numbers that size the index are the compiler lane's. Nothing here is an RTL edit.*

## 1. The capacity fix: the flat table's 64,001-record ceiling (cpython lt1's six capacity cells)

### Option A — widen the virtual INDEX inside the same 30-bit id: {gen[29:18], idx[17:0]}, 262,144 records

**The id stays 30 bits, so every tracker and cache keeps its width.** Checked: the LSU revocation cache's tag RAM and
records are 30-bit (`load_store_unit.sv:290, :323-337, :1167-1169`), the fill/invalidation ports are 30-bit (`:201, :211,
:214`), `rvc_ok_match` (`:1414`) and `rvc_resolve_live` (`:1438`) compare the full 30 bits. What changes is where the code
SPLITS the id, and there are two kinds of site.

**(a) Index-only compares and masks that hard-code 16 bits.** These compare `[15:0]` and must become `[17:0]`; each one
is a two-bit-wider equality, nothing else:
- LSU revocation cache: `:1195` (install reuse), `:1245` (born-dead), `:1256` (invalidate), `:1307, :1309` (the same-cycle
  invalidate/fill against the access's id), `:1421-1422` (the ALLOW record's kill), `:1426` (the DEAD record's kill),
  `:1428` (`rvc_probe_id_o`, a bare 16-bit index PORT: `:219`, and `ex_stage.sv:745`), `:1437, :1442-1443` (the pending
  probe's resolve and kill).
- PC tracker: `commit_stage.sv:412`.
- CPMP trackers: consume the namespace bit (`ex_stage.sv:322-330`); their compare width is read next (the file moved;
  listed as a site to confirm, not confirmed).
- ex_stage, the virtual table path: `:1503-1504` (header slots `< 2`), `:1515-1516` (the header's capacity/next views,
  which SATURATE at 65,536 when `[61:16]` is non-zero -- the saturation itself is the ceiling's other half and moves to
  `[61:18]` / 262,144), `:1680-1681` and `:1704-1705` (address = root + idx[15:0] * 16: becomes idx[17:0]), `:1783`
  (`revnode_fill_id = {gen[122:109], idx[15:0]}`), `:1793` (`rev_rd_idx_q`, declared 16 bits at `:739`).
- The Anvil unit `capstone_rev_node.anvil`: every `#{14'd0, x[0+:16]}` mask (`:6, :105, :116, :122, :140, :160, :167,
  :200, :234`) becomes `#{12'd0, x[0+:18]}`; the generation width 14 -> 12 at every `14'd16383` retirement compare
  (`:102, :110, :217`) and the pop's `fn.generation + 14'd1` / `#{g1, 16'd0} + *pop_idx` (`:328-330, :372-383`): the
  saturating retirement then fires at 4,095 generations of one index instead of 16,383.

**(b) The record layout, which is the ABI with the module and QEMU.** `rev_node_t` (`capstone_unit.anvilh:665-673`) is
parent 16 | free 1 | generation 14 | depth 17 | prev 30 | next 30 | valid 1 | linear 1. `prev` and `next` are ALREADY
30-bit in the record's word 0, so the walk links need no layout change. The 16-bit `parent` lives in word 1 at [45:30]
with the generation at [59:46] (`ex_stage.sv:1686-1692, :1703, :1709`): widening parent to 18 bits takes [47:30] and the
generation becomes 12 bits at [59:48]. That is a word-1 bit-layout change on both sides (QEMU's `cap_rev_table_abi.h`,
the module's `capstone_vm.c`) with the record size unchanged at 16 B.

**Nothing compares only idx[15:0] where it should compare the full id**, as far as this read goes: the two full-id
compares (ALLOW match, probe resolve) are 30-bit; the index-only compares are index-only BY DESIGN (an invalidation
broadcast carries a bare index, `capstone_rev_node.anvil:3-6`). The physical pool keeps its own 16-bit head
(`REVNODE_HEAD_BITS`, `capstone_unit.anvilh:624-640`, `capstone_rev_node.anvil:397, :840, :849`) and is untouched: option A
widens the VIRTUAL index only, and the physical index stays 16 bits inside the same 18-bit field (top two bits zero).

**Alignment.** The RTL forms every table address as `srevroot + idx * 16` (`ex_stage.sv:1680-1681`) and bounds-checks
nothing: the only requirement is the CSR's own, `srevroot` 4 KiB aligned (`csr_regfile.sv:385`, R-59's precheck
`root & 4095 == 0`). A 4 MiB table at any 4 KiB-aligned physical address is fine; no root-aligned-to-table-size rule.

**Cost and risk, RTL side.** No new function on any result path: every change is an equality widened by two bits, a
mask, or an adder input widened by two bits. The one cone to name is the LSU revocation cache's index compares, which sit
inside `cap_exception` -- the dominant slack term by the file's own comment (`load_store_unit.sv:279`) -- and gain two
bits at eight compares. Expected inside the per-module floor and the WNS null; graded like the batch, by the change's
OWN footprint (cells carrying its identifiers) and the stem summed over hierarchies, never one module. A synthesis-only
size gate is cheap insurance but not demanded by the shape. Lint: no new UNUSEDSIGNAL expected.

### Option B — the paged walk (a table of record pages, non-contiguous)

The unit already decodes a PAGED bit (`vp_paged`, `capstone_rev_node.anvil:860`) and REFUSES it. Supporting it means a
second dependent memory read on every node access (page-table lookup, then the record), i.e. new states in the unit's
walk and the probe path, a second address adder, and a latency that roughly doubles the walk and the R-43 probe (the
probe's post-accept wait is bounded at 65,535 cycles, `load_store_unit.sv:316-320`; silicon walks already reach 11,811
cycles for 256 nodes). It also puts a page-table read on the probe path that today feeds a replay. Cost: a unit
redesign plus a new timing cone, with the S-12/R-43 class of hazards (a stalled walk is a core hang) to re-prove.
**Recommendation: A**, at 18 bits, because the module side's 4 MiB order-10 block is the practical ceiling anyway and
the RTL cost is two bits per site.

## 2. Owed items for the same bitstream

| item | in/out | cost |
|---|---|---|
| six-E-bit narrowing of item 4's predicate input (`r60_src_meta`[63:27] and `src_word`[25:17,13:3], the +2 UNUSEDSIGNAL) | IN | two files, lint-only effect; the proof and the five readings re-run through the wrapper (~1 h) |
| the SEALEDRET negative arm for item 4 (needs a CALL-produced capability; r51-return-pcc / sup-guards / ldc-perm-check hold one) | IN, a test, no RTL | ~1 h |
| the stale fault-site variants (owed since the audit) | IN, tests | ~2 h |

## 3. Every silicon issue still marked for the next bitstream

| issue | shape of the RTL fix | in/out | cost |
|---|---|---|---|
| S-21 (escape restores the S caller's state from frame+1024) | the caller snapshot out of the frame: kernel-only save area or the CSR file for the one slot (ISSUES.md:384-386) | IN if the module fix (a) is not enough; the lead's call | switcher Anvil + CSR: a day, and a cross-tree QEMU twin |
| S-22 (the paused context's state in its frame) | a CONSUMING resume: the switcher clears the 31 slots and tags after loading them; the hidden words out of the frame (:416-418) | IN with S-21 (same switcher walk) | half a day on top of S-21 |
| S-25 / R-57 (the ghost store-buffer entry: a faulting capability store retires without its fault) | the store-side exception carried to writeback (2114e11bd, the R-56 twin) is IN THE BATCH already; R-57's residual is the parity note at ISSUES.md (synthesis audit 2026-10-18) | the first boot's vp2 r8..r11 rungs decide; nothing more to build until read | 0 until read |
| S-29 (satp installed with no TLB flush) | `flush_all_i` on the switcher's satp write | IN THE BATCH (the cross-tree diff) | 0 |
| R-51 (untagged PC after RETURN) | FIXED ON SILICON in caplifive_r51r52 (ISSUES.md:1701); residue: none open | OUT | 0 |
| R-49 (the old mis-check): a sticky "mis-check reached" flag to separate it from timing (:1563) | observation only | OUT unless R-49 recurs on the batch | 0 |
| the S-23 fetch-face residual (ISSUES.md:583: restore the caller's CPMPs before the pc reaches the frontend, or hold the fetch) | the switcher's return ordering | IN only if the batch boot's s23 vector still shows the fetch refusal; the batch's item 1 is the candidate fix | 0 until read |
| the cause-13-on-a-revoked-capability reading (2 % on vp2; this lane's directed sim `rvk-unmap-cause.S` is running) | unknown until the sim reads | decided by the sim, today | 0 until read |

**One decision for the lead:** A at 18 bits + the three owed items is the small bitstream (two days of RTL, one synthesis
pair). Adding S-21/S-22 makes it the large one (a switcher revision, a week, its own QEMU twin). The first boot's readings
(s23 vector, r8..r11, the 2 % cause) come before either is cut, because each can add or remove a row above.

## 2026-10-23: bitstream 2 cut and handed to synthesis (the lead's go, relayed by the semantics owner; S-21/S-22 out)

**Hash:** capstone-ariane `vp3-index18` at `5d365d9a6`, base `c721c1a2a` (the batch). The synthesizable content is
`d44c9441b`; the tip adds comments and an `ifdef SUP_TRACE` block. **Content:** option A at 18 bits -- ids are
{generation[29:18], index[17:0]} in BOTH namespaces (the semantics owner's ruling: one format, the physical pool's
generation shrinking to 12 bits and retiring at 4,095, no second struct), the record ABI word 1 = parent [47:30],
generation [59:48], record size unchanged, the header views saturating at 262,144, every index-only compare at 18 bits
(LSU cache, PC tracker, CPMP trackers, the unit's masks, the probe channel, the DYN CSRETIRE bound), the pool record's
own encoding re-sliced; plus item 4's predicate narrowed to its six exponent bits (the two UNUSEDSIGNAL names gone).

**Gates, in the order the process names:**
- lint on `5d365d9a6`: PASS, every counter at the batch's baseline, UNUSEDSIGNAL 1426 → 1424, the 40 loop names identical;
- the 95-test sweep on `d44c9441b` (prologue-stamped, clean): 92 identical on exceptions, readings and retired count
  against the batch tip's own sweep and against the c7e6f604f baseline, 0 differ, 3 missing on both sides (the random
  arms);
- the claim-auditor: every pack re-derived bit by bit against the regenerated unit and the read decode (table and pool),
  no missed site in `core/`, no signal into any of the 40 loop cones, the QEMU twin's ABI constants agreeing
  (capstone-qemu `virtual/flat-index-18`, GEN_SHIFT 48, GEN_MASK 0xfff, PARENT_SHIFT 30, FLAT_PARENT_MASK 0x3ffff,
  ID_GEN_SHIFT 18); its findings closed: the physical pool's shared generation (ruled), the stale layout comments (fixed);
- acceptance in simulation (`vp3-idx18.S`, three arms on a 4 MiB table): an allocation past 65,535 round-trips (ancestor
  id 0x10001, a record whose 18-bit parent field reads 0x10001, CSRETIRE accepted with the run pushed, free list
  {65536, 2}, next-unused 0x10002); a record at generation 4,094 pops as 4,095 and its CREVOKE retires it (dead, not
  FREE, the next derive bumps); stale ids refused above 2^16 ({0,70000} against a record at 1) and accepted with the
  current generation, the same pair at 300; the R-12 recycle witnesses, three of them registered for the first time, read
  262147/262150, 524291/524294 and 6/262147/7; `r12-recl-cpmp` reads 262147 with its seven CPMP deltas and count 4
  unchanged; item 4's readings on the narrowed predicate are identical to the batch tip's by the same extraction;
  the SEALEDRET negative (`r60-sealedret.S`, a CALLed domain's parked return capability moved 1 MiB) stays silent while
  its LINEAR control fires once and packs a nulled result (a sim-only `SUP r60 FIRE` trace line records every fire).

**Pre-registered for the synthesis-only size gate** (the synth lane's method, with a control that re-finds item 4's
275 LUT + 40 CARRY4 on the base): mapping in the batch's class; the change's own exclusive cone within a few hundred LUT
of the base's either way; the ex+issue stem (base 87,162 LUT) and the core (base 154,409) inside ±2,500 / ±2,000;
the 40 synthesis-stage loops unchanged by distinct cut; routed WNS inside the null band of −18.529.

**Owed and NOT in this hash:** the "stale fault-site variants" named in the owed list have no specification anywhere in
the docs or the plans beyond that phrase; they were not invented, and they wait for the lead or the semantics owner to
name them. The silicon vectors that print ids move with the split and get bitstream-2 pre-registrations by the
recorder's count once the hash is built (`va-alloc-bare`'s 0x10004/0x20004 become 0x40004/0x80004).
