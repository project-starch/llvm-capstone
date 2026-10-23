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

## 2026-10-24: the size gate PASSED on all four keys; the full build launched 11:20 UTC

The synth lane ran the synthesis-only gate on `5d365d9a6` in a fresh worktree with the batch's submodules (guard 40 GB,
watchdog 30 GB), and stopped it after `synth_design` as planned (whole-run peak 22.44 GB; trial C's was 23.53; the synth
checkpoint's sha256 begins `bd5fbd28c4e3d1c3`). Graded here against the pre-registration above:

| key | pre-registered | read | grade |
|---|---|---|---|
| technology mapping | the batch's class (~5.4 GB) | 5,166.3 MB vs the batch's 5,164.3 at the same marker | PASS |
| own exclusive cone, with control | a few hundred LUT either way of 275 LUT + 40 CARRY4 | 272 LUT + 40 CARRY4 (control re-found 275 + 40 on the base), all homed in `issue_read_operands` | PASS |
| ex+issue stem | 87,162 ± 2,500 | 86,408 (−754) | PASS |
| core | 154,409 ± 2,000 | 154,284 (−125) | PASS |
| loops | "the 40 synthesis-stage loops unchanged by distinct cut" | see below | PASS, on the corrected reading |

**The loop key was mis-stated and is corrected here, not moved after the fact.** The 40 is the lint `UNOPTFLAT` count
(identical on tip and base); the synthesis quantity is the "Found timing loop" header count, 7 on the batch and 7 on
bitstream 2, with the member-stem union 15 against the batch's 17 (0 entered, 2 left: `gen_register.d_o`, a
register-slice pass-through name, and `state_q_reg[n]`, the batch's own CARRY cut name). The synth lane first flagged
the Synth 8-326 cut targets as changed (4 distinct on the batch, 11 here), then retracted it on its own null control:
across twelve builds it holds logs for, all within the LUT floor of each other, the 8-326 total swings 4 / 8 / 17 / 17 /
18 / 40 / 50 / 101 ×4 / 73 and the cut families come and go, so a cut target names where Vivado broke a loop in that
build, not whether the loops changed. The confirming reading is the netlist: `check_timing` on both synth checkpoints
reports exactly one loop still unbroken on each side (19 cells on the base, 18 here), the batch's own TIMING-23 ring
(`lsu_bypass cap_clear_addr_q[55]` → `load_unit` → a CARRY4 compare and five `store_buffer` LUTs → `rd_req_q` → the
D-cache arbiter and read-port controller → `ldc_store_ok_q` → back), with ten cells identical by exact name and only
the synthesis-numbered middle renamed. Every member lives in `lsu_bypass`, `load_unit`, `store_buffer` or `wt_dcache`,
none of which is in the diff; on the RTL side the load-store unit's change is width-only (23 id compares and the probe
id widened to 18 bits) and drives the revocation fault, not that ring.

Ungraded readings, recorded for the routed reading: `csr_regfile_i` 29,307 LUT against the batch's 27,639 (+1,668; the
only change under it is `pmp_data_if.sv`'s two compares widened by two bits on each of the 16 CPMP entries, 32 compare
bits in all, and the module's own trajectory with its source unchanged was A 27,073 → C 27,278 → batch 27,639); the
synth lane tags `cpmp_tracked_revnode_id` and `pmp_data_if` nets in the routed-checkpoint query beside the batch's four
handles, with the same positive and negative controls. `r60_src_E` does not survive synthesis (0 nets, as the base's
`r60_src_meta`): the narrowing folds into the item-4 cone, and the −3 LUT is its whole effect.

**The full build** runs on the same worktree (guard 40 GB, watchdog 34 GB with a 64 GB host floor, loop watchdog against
the batch's member union, collection as its own capped step; the batch took 2 h 27 m). Pre-registered: routed WNS inside
the null band of −18.529; the sha256 and the post-route stem rows travel with the hash.

**The stale fault-site variants are closed**, correcting the paragraph above: the semantics owner specified them on
2026-10-23 and they are on the lane branch at `0f94f2cff` as tests only (`r12-stale-sites.S`, six builds on the table
skeleton with index 70,000 recycled as generation 1: LDC 25, SPLIT 25, CREVOKE 25, CSRETIRE status 1; MREV and DELIN
are type-gated at 26 on a non-linear stale child, and their stale halves are `r12-recl-stale`'s U5/U8 at 25 with ids
262147/262150). `core/` is unchanged since `5d365d9a6`, the hash under synthesis.

**`va-alloc-bare`'s bitstream-2 vector is pre-registered** in its `PREREG.md`, cut from the kit's own frozen source on
the bitstream-2 RTL by the recorder's count: it differs from the batch vector at positions 1, 3, 11 and 15 only.

## 2026-10-24: the full build FAILED in routing; attributed to the csr_regfile/tracer window; candidate 68d0ffaf8 drops the tracer

**The failure.** `route_design` on `5d365d9a6` stopped at initial routing after 8 minutes: "Design is not routable",
global congestion level 6 against the batch's 5 (max by direction North 86.9 → 100.0 %, South 86.0 → 118.2 %, East
85.8 → 95.4 %, West 85.7 → 107.5 %; congested regions 8×8–16×16 → 16×16–32×32). Exit 2 after 1 h 41 m, peak 22.29 GB.
Not memory, not a loop, not a DRC. No bitstream and **no WNS**: the pre-route estimate (−18.003) is not a result. Placed
utilization was NOT larger than the batch's: LUT 178,543 against 178,606, FF 96,511 against 96,614, slices 50,142
(98.41 %) against 50,869 (99.84 %). This is the second routing failure at this size with flat utilization (attribution
build A, 2026-10-18, was the first; a hierarchy re-homing alone tipped it). The gate keys stand: they are synthesis-stage
readings. **What this pre-registration lacked and the next one carries:** the failure branch (if routing fails, the graded
readings are the placed area, the gate keys and the congestion census; WNS is none).

**The attribution** (synth lane, read-only on both PLACED checkpoints, the batch's as control, 0 ERROR; the retry
question settled first: no observed placement variation from identical inputs on that host, and the flow is designed to
reproduce, so an unchanged re-run is a repeat, not a sample):

| reading | batch | 5d365d9a6 |
|---|---|---|
| `report_design_analysis -congestion` | no window above level 5 | North Global level 5 at 103 %, CLBLM X46–77 Y276–310: `csr_regfile_i` 43 %, `i_tracer` 25 %, `i_tracer/i_uart_byte_tx` 10 % |
| hot window INT X32–63 / Y222–349, cells / LUT / pins | 48,894 / 26,363 / 282,179 | 51,323 / 29,898 / 299,876 |
| largest in-window gains (cells) | — | `csr_regfile_i` +3,227 (pins 75k → 95k), scoreboard +1,915, `pmp_data_if` +942 (234 → 1,176), `dyn_unit` +323 |
| largest in-window losses | — | FP regfile −2,610, `uart_byte_tx` −743, `i_tracer` −629, `dom_switcher` −512 |
| `cpmp_tracked_revnode_id*` cells, window / design | 322 / 853 | 730 / 829 (moved in, not grown) |
| `rvc_*` cells, window / design | 0 / 3,189 | 0 / 3,426 |
| `revnode_*_id` cells, window / design | 0 / 203 | 0 / 205 |
| `r60_*`, `cursor_outside_window*` | 0 / 0 (do not survive synthesis) | 0 / 0 |
| POS control `fu_data_q*` in window; NEG control | 6,654; 0 | 5,488; 0 |
| `revnode_invalidation_id` loads in window | 96 | 288 (= 16 CPMP entries × 18 bits: all of `pmp_data_if`'s compare inputs) |
| `csr_regfile_i` combinational cells (synth checkpoints) | 33,928 | 35,704; the `stc_lsu_ex*` cone +1,628 in csr / −431 in ex_stage, a re-homing the diff never touches |

So **the index cells and item 4 are not in the congestion**. The local densification is `csr_regfile_i` plus `pmp_data_if`'s
CPMP invalidation compares pulled wholesale into the csr/tracer window; the RTL reason is that the compare reads the
post-adopt id (R-38 Stage 0, `pmp_data_if.sv:120`), a mux fed straight from `csr_regfile`'s `cpmp_q`, so its cells want to
sit beside `cpmp_q`, and widening it from 16 to 18 bits reads 32 more `cpmp_q` bits. On the RTL side the three rev-node id
buses are 30-bit nets at their source on both trees (the split changed which bits 26 compare sites read), `csr_regfile.sv`
is unchanged and instantiates nothing, and `pmp_data_if` is under `load_store_unit`.

**The lever, and why it is the tracer.** `i_tracer` (`core/tracer.sv`, instantiated unconditionally at `cva6.sv:998`)
is 9,319 LUT and 32,849 FF, 6 % of the core's LUTs and **44 % of its flip-flops**: its 256 × 128-bit ring is built from
registers (written inside the async-reset block, read combinationally with a byte select), with an 8,790-LUT read-out mux
in `i_uart_byte_tx`. It is 35 % of the congested sub-window. The board lane read its drivers and 158 vp2-era driver logs:
no tracer function has been used since vp2 (no dump; the drivers forbid `trace_dump` on a wedged core; the switches are
parked at 0 because sw[0] takes the console TX and sw[1] arms the dump), and every graded reading, the trap log, the mepc
bank, the wedge apertures, the commit-pc taps and every monitor line, goes through the LED aperture mux or the 16550. Its
two conditions hold at the source: CSR 0x800 (the monitor's C_PRINT, executed on every `capstone_report`) stays a legal
write, accepted in `csr_regfile.sv:3154-3159` with nothing consuming the strobe; and `uart_debug_tx_o` is driven by the
debug printer alone. The alternative of registering the CPMP compare one cycle later is a semantics change (an
integer-mode access authorised by a CPMP entry has no other revocation gate); the semantics owner ruled it acceptable only
with a per-op commit-ordering proof and directed arms, and at ~1k cells against the tracer's tens of thousands it is
second-order and deferred.

**Candidate `68d0ffaf8`** (lane branch `vp3-notracer`, from the bitstream-2 branch): `core/` equals `5d365d9a6` plus one
file, `cva6.sv` +19 lines: the tracer instance under `` `ifdef CAPSTONE_TRACER `` (off by default), `tracer_uart_tx` tied
to 1 (idle), `trace_buf_empty` tied to 1 (the board drivers DO read that bit: wedge aperture 225 = bank 3'b111 reg
5'b00001, bit 7; on a tracer-less bitstream it reads a constant 1 and carries no information, and the board lane annotates
its label at the flash; its first boot adds a monitor report site printing as the positive control that `csrw 0x800`
still retires without a trap), the console mux reduced to the debug
printer; the watchpoint compare that fed only the tracer is left for synthesis to prune. Gates on that hash, every run
through the Anvil prologue with `rtl_dirty=no`:

- lint: PASS; every counter at the batch's baseline except UNUSEDSIGNAL 1424 → 1431 (seven named wires in `cva6.sv`, all
  consumed only by the tracer) and LATCH 52 → 44 (eight inferred latches, all inside `tracer.sv`); the baseline is re-based
  to 1431 and tightened to 44; UNOPTFLAT 40 with identical names;
- the 95-test sweep: 92 identical, 0 differ, 3 absent on both sides, against the batch tip's sweep and against
  `c7e6f604f`'s, the same result `5d365d9a6` gave;
- the three acceptance arms, the SEALEDRET pair (control fires once, arm never), `va-alloc-csret` and the frozen kit's
  source: every value identical to the pre-registered readings, so `va-alloc-bare`'s bitstream-2 vector above transfers
  to this hash unchanged;
- the simulated model proven tracer-less by artifact (82 of 86 objects rebuilt after the edit; the generated root carries
  no `cva6__DOT__i_tracer__DOT__*` and no `trace_buf`, while the testbench's `i_rvfi_tracer__DOT__*` state is present as
  the control that an instantiated stateful module does appear).

**Pre-registered for its build:** `report_design_analysis -congestion` with no window above level 5 (the batch's reading);
placed FF down by ~32,000 and LUT by ~9,000 against `5d365d9a6`'s placed checkpoint; routed WNS inside the null band of
−18.529; the one TIMING-23 arc unchanged. **If routing fails:** the graded readings are the placed area, the gate keys and
the window census against `5d365d9a6`'s placed checkpoint; WNS is none. Dropping the tracer is a design decision: the hash
goes to synthesis on the lead's word, and the flow's settings stay as they are.

**Ruling (the project lead, 2026-10-24, given in the runtime lane's session and relayed from there): the tracer is
removed.** Bitstream 2 is `68d0ffaf8`. It goes straight to the full flow, not through a fresh synthesis-only gate: the
change removes one instance and adds no logic, the gate's keys on this RTL already passed on `5d365d9a6`, a separate gate
would read only what the tracer's own utilization figures already give (core LUT ≈ 154,284 − 9,319, FF ≈ 74,290 − 32,849),
and the failure to be tested is routing, which only the full flow exercises; the synthesis-stage rows are read from the
full flow's own reports and graded in passing (core LUT −9,319 ± 500, FF −32,849 ± 200, ex+issue within ±2,500 of 86,408,
loop headers 7 with the member union unchanged), with the placed figures (LUT ≈ 169,000, FF ≈ 63,700) and the routing
pre-registration above. The CPMP-compare option stays closed unless this hash fails to route. The launch is the synth
lane's user's approval in that session; the flash is the lead's own word in the board lane's session.

## 2026-10-24: 68d0ffaf8 blew up in technology mapping; both blowups are one base-logic ring; the ring is cut in RTL and bitstream 2 SEALS from dcc6daf6e

**The second failure.** 68d0ffaf8 went straight to the full flow on the argument that a removal adds no logic and the
gate's keys had passed on its parent. The synth lane's independent watchdog killed it in Technology Mapping at 34 GB
(one synth_design task_worker 10.8 -> 18.3 GB in 20 s; the main process at 5 GB), trial B's phase and shape. **That
argument is refuted**: a change that removes logic or ties nets to constants is not exempt from a synthesis reading (it
cost nothing here only because a synthesis-only gate dies at the same phase).

**The partition reading** (synth lane, read-only, from each worker's job files): both blowups, trial B (4fce4d744) and
68d0ffaf8, are the SAME genome, `ariane_xilinx_GT3__1`, the flattened cva6-top partition, breaking the load-unit side of
the LSU <-> D-cache clear/handshake ring one arc at a time (9 `set_disable_timing` exceptions on load_unit cells on both;
none on any completed build). The ring is base logic, the one TIMING-23 arc every bitstream of this line carried; which
build trips on it depends on how the partitioner cuts the top, so the tracer only re-rolled the cut. It closed through
the load unit's same-cycle clear: SEND_TAG_LDC and WAIT_LDC_RESP raised `clear_o` in the result's cycle with
`clear_addr_o = cap_clear_addr_d` and the Q-19 refusal on `ldc_store_ok_d`, into the store unit's clear mux, the store
buffer's page-offset compare and `page_offset_matches`, and back into the load-unit FSM. vp1's own record had named the
remedy (2026-10-16-virtual-profile-on-silicon.md:492-493).

**The cut** (capstone-ariane vp3-ringcut b8104c7f5, on 68d0ffaf8; ruled by the semantics owner with two conditions, and
given the lead's go in the runtime lane's session): every consuming LDC clears from LDC_CLEAR_WAIT, the state that
already served every clear whose store buffer was busy, so `clear_o`, `clear_addr_o` (`cap_clear_addr_q`) and the
refusal (`ldc_store_ok_q`, scoped to that state) read flops only; one cycle per consuming LDC. Condition (a), no
retirement before the clear, holds in source (valid_o for a consuming LDC is raised only in LDC_CLEAR_WAIT with the
clear); condition (b), refusal, cause, tval and clear address bit-identical, holds on every arm (vb-trans with its
refusal, r29-ldctrl, va-alloc-csret, the acceptance arms, all value-identical to 68d0ffaf8, one cycle moved), with one
DELIBERATE change: the scoping closes R-61 (a stale refusal relabelling later load-unit exceptions STORE_PAGE_FAULT;
`docs/ref/ISSUES.md`, confirmed on silicon on the batch, both boards, by two pre-registered bare kits). A younger plain
load of the cleared source reads 0 (`vb-trans-plainld-vp2`). Lint PASS with UNOPTFLAT 40 -> 39, the name that left
being `page_offset_matches` ("Circular combinational logic"), the ring itself; the 95-sweep 92 / 0 / 3 against both
baselines.

**The gate on dcc6daf6e** (`core/` = b8104c7f5's; tests and lint baselines above), synthesis-only, every key passed:

| key | pre-registered | read |
|---|---|---|
| technology mapping | the batch's class, near 5.2-5.7 GB | 5,004.5 MB (an undershoot of the band's lower edge, recorded); Timing Optimization 22 min against the tracer-less build's 80 |
| loop-breaking on load_unit | none | Synth 8-326 0 in total (4-101 on every earlier build by the same grep) |
| loops | headers <= 7, the clear ring out of the member union | headers 0, union 0; `check_timing` on the checkpoint: "0 combinational loops in the design" (1 on vp3's by the same query); the D-cache handshake loops went with the ring |
| core vs 5d365d9a6 | LUT -9,319 +/- 1,000, FF -32,849 +/- 300 | -9,651 / -32,866 |
| load_unit / ex+issue | +/- 500 / +/- 2,500 | +246 / +244 |
| item 4's cone | 272 LUT + 40 CARRY4 | the same |

**The seal** (full flow, exit 0, 1 h 47 m, peak 21.33 GB): `caplifive_vp2c_dcc6daf6e.bit`, 11,443,722 B, sha256
`fcaec59323eb2c9d6fcb56a6e6709ac797d06c24e4b4a5d5fd6080ba2d21906e` (the board lane's copy equal by hash). Placed 169,076 LUT
(predicted ~169,000) and 63,648 FF (~63,700); slices 94.91 % against 98.41 %. Graded against the pre-registration:

| key | read | grade |
|---|---|---|
| loops | `check_timing` 0 combinational loops; TIMING-23 0 (2 on the batch's report by the same matcher); LUTLP-1 0; 0 routing errors | PASS |
| congestion | placer final: no window above level 5 (as the batch); router initial: 2 short level-5 windows at csr_regfile against the batch's 9, three of them level 6 | PASS |
| WNS | **-9.790** (TNS -51,580, 12,477 of 188,703 failing, WHS +0.055) against the batch's -18.529 and the resident vp2's -19.019 | **NOT MET by its letter** ("inside the null band of -18.529"): 8.74 ns outside, on the GOOD side. Not moved; put to the lead before any flash |

The gain is explained in kind and not in cause: the worst paths are the batch's families (csr_regfile `cpmp_q` and the
dom switcher's `cur_idx_q` into issue_read_operands' `fu_data_q`), each improved by about the same amount at about the
same depth (`cpmp_q`: -18.529 at 126 levels -> -9.613 at 124; `cur_idx_q`: -18.134 at 119 -> -9.790 at 121), so it is delay
per level, consistent both with ~33,000 fewer flops in the placement and with the store buffer's page-offset segment
leaving every one of the worst 500 paths (500 -> 0); one build cannot separate them. Nothing points to lost logic:
post-route area matches the tracer's share (-9,564 / -32,966 against -9,311 / -32,849), item 4 is on none of the worst
500 and its own worst path is +7.918, and every simulation arm is value-identical. Not a reading: the `cpmp_tracked` tag
(its positive control found 0).

**First boot** (pre-registered, the images simulated on dcc6daf6e's RTL and read from their recorder areas): s23-cpmp-rearm
reads the batch vector; va-alloc-csret the bitstream-2 vector (`va-alloc-bare-2026-10-23/PREREG.md`); the two R-61 kits
their fixed-tree vectors. The flash is the lead's word in the board lane's session; the board lane holds it pending the
lead's answer on the WNS key.
