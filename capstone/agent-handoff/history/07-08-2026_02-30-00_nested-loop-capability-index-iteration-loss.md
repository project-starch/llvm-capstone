# A four-way conjunction loses loop iterations on silicon -- characterised, NOT root-caused

Date: 2026-08-07
Status: reproducible board-vs-QEMU divergence with a fully bisected TRIGGER and NO mechanism.
Handed over as a characterisation plus seven controls, not as a diagnosed defect.

## The observation

A domain loop executes FEWER iterations than it should, deterministically, returning a WRONG
NUMBER rather than hanging. QEMU runs the identical binary correctly every time.

    L31   nested 64x9, inner body reads a capability field indexed by the inner counter
          board 567 on SIX separate runs      QEMU 576

## The trigger, bisected to a conjunction of FOUR conditions

Seven controls, each removing exactly ONE condition from L31, each exactly correct on the
board and matching its measured QEMU oracle. Every probe reuses existing globals, so all share
`.capstone_cap_init@0x160570` and `.capstone_gp_table@0x1a0ea0` and none is an
image-perturbation draw.

| level | what it removes | board | QEMU |
|---|---|---|---|
| L28 | the capability access entirely (`qcount++` only) | 576 | 576 |
| L30 | the capability access, keeps a `jalr` | 576 | 576 |
| L36 | the nest (single loop, constant index) | 576 | 576 |
| L35 | the counter index (nest, constant index) | 576 | 576 |
| L38 | the capability (integer field at counter index) | 576 | 576 |
| L37 | the counter (capability at a dynamic NON-counter index) | 576 | 576 |
| L39 | the nest, keeping a counter-derived index (`qk%9`, monotone) | 576 | 576 |
| L40 | the RESET (nest, index is the OUTER counter, never resets) | 576 | 576 |
| L41 | the nest, keeping an explicitly RESETTING index | 576 | 576 |
| **L31** | **nothing -- all four present** | **567 x6** | 576 |

So the fault requires ALL of:

1. a NESTED loop  (L36, L39, L41 remove it -> correct)
2. a CAPABILITY access in the inner body  (L28, L30, L38 remove it -> correct)
3. an index that is the INNER loop counter  (L37, L40 remove it -> correct)
4. that index RESETS to 0 each outer pass  (L40 removes it -> correct)

Conditions 1 and 4 were confounded until L40/L41 separated them: in every failing level the
capability index is also the only frame word written to 0 in an outer body and counted past
its own bound. L39 looked like it tested a counter-derived index without a nest, but `qk%9`
comes from a MONOTONE counter, so it did not.

## What is NOT established

**The mechanism.** Nothing read so far in `capstone-ariane` is sensitive to loop NESTING, and
every per-access mechanism proposed has been refuted by a level that measurably returned the
correct answer:

* store-to-load forwarding / wbuffer eviction -- `wt_dcache_wbuffer.sv:331` selects its path on
  `(|wr_data_be_o) && (|hit_oh)`, neither of which depends on `wr_ack_i`;
* an issue-stage clobber window -- the trigger is byte-identical in L31 (567) and in
  L35/L36/L37/L39 (576), and `NrCommitPorts=1` makes `stall_waw` inert;
* a spill slot aliasing the counter -- L32 miscomputes with NO 16-byte stack traffic in its
  nest, and L30 does a stack `stc`/`ldc` round trip 576 times and is exactly correct;
* a clearing-type `ldc` -- `delin(sp)` plus SPLIT preserving cap_type makes every cap-table
  entry NONLIN, which `load_unit.sv:448` excludes; and such an `ldc` would self-destruct on
  iteration 2 of a 576-iteration loop.

**Which 9 iterations are lost.** L31 reports only the accumulator, and 576-9 is equally "one
lost inner pass", "nine lost read-modify-writes", or "63 outer passes that each lost nothing".
UNRESOLVED.

## This is NOT the SQLite blocker

`sqlite3InsertBuiltinFuncs` is a SINGLE loop reading a capability field indexed by its own
counter. L39 was built as exactly that shape and returns 576. Condition 1 fails, so this fault
is inert there. The blocker (qr15 returns / qr16 wedges, images TWO BYTES apart) remains a
separate, unexplained divergence.

## What would close it

Board bisection has gone as far as it can: the trigger is pinned and every remaining question
is about internal state. What is needed now is a SIMULATION or waveform of L31's inner loop --
specifically, what differs in the fetch/issue/replay path between an outer pass that runs its
inner body and one that does not. S01's README asks for the same thing for its own hang, and
that request is still open.

Reproduce with:

    OUT_DIR=/tmp/capstone/sq-l31d4 DOMAIN_EXTRA_DEFS="-DCAPSTONE_SQLITE_QUICKRET=31 -DQR_DRAW=4" \
      bash capstone/benchmarks/sqlite/build-sqlite-silicon.sh
    # stage as q31.dom, k800 FIRST in the boot; expect obs 0x9E310237 (567) against QEMU 0x9E310240


## UPDATE 2026-08-07 (later): localised to a proximity window around a capability store

A 13 KB OFF-SQLITE reproducer now exists and the trigger is far tighter than the four-way
conjunction above. In `fdreg_kernel.h` stage 7 the capability local `z` lands at `s0-0x50` =
`sp+0` and is written by a 16-byte `stc` every inner iteration. Moving ONLY the loop counters
within the frame (`FDREG_SHIFT`), with the frame a constant 0x50 bytes and every variant
QEMU-green at 576:

| shift | inner counter | bytes above the stc's 16-byte end | board |
|---|---|---|---|
| 0 | `s0-0x34` = sp+0x1c | 12 | **576 correct** |
| 4 | `s0-0x38` = sp+0x18 | 8 | 909 |
| 8 | `s0-0x3c` = sp+0x14 | 4 | 567 |
| 12 | `s0-0x40` = sp+0x10 | 0 | 567 with bit 27 set |

Severity scales with proximity and vanishes at 12 bytes. No counter ever overlaps the store.
A dump of the dead frame (read from `domain_main` after the callee returned, callee code
verified identical modulo branch targets) contains `0x08000001` — bit 27 — inside a repeating
16-byte-period pattern of capability-shaped words.

This also explains why every in-frame instrument destroyed the fault: a sentinel array (+32 B)
and a `&qc` pointer (+16 B) each pushed the counters past the 12-byte threshold.

TWO MECHANISMS FIT AND THE SOFTWARE SIDE CANNOT SEPARATE THEM: an over-wide capability write,
or a false store-to-load forward on an incomplete address comparison. A simulation distinguishes
them — the first shows in the memory image, the second does not.

Reproduce: `DOMAIN_GLUE=interp DOMAIN_OPT_LEVEL=-O0 DOMAIN_EXTRA_CFLAGS="-DFDREG_STAGE=7
-DFDREG_LEAVES=0 -DFDREG_GUARD=0 -DFDREG_SHIFT=8"` → 567 on silicon, 576 under QEMU. `SHIFT=0`
is correct on both.

## UPDATE 2026-08-07 (boot 51): THE PROXIMITY WINDOW IS RETRACTED. It is CACHE BANK GEOMETRY.

The section immediately above is wrong and is retracted. Its reading — "severity scales with
proximity and vanishes at 12 bytes, so the store damages memory beyond its 16-byte footprint" —
was refuted by a probe designed to test something else.

Stage 13 (`wp0`) fills the 12-byte hole that clang leaves immediately above the store with three
passive witnesses, which pushes the counters to `sp+0x28` — 24 bytes above the store, in a
DIFFERENT 16-byte row, in a 0x60 frame instead of 0x50. Under the proximity model that is deep
in the clean regime and must return 576. It returned **909**: byte-identical to the SHIFT=4
measurement taken 8 bytes from the store. Distance from the store is not the variable. The shift
sweep only looked like a distance law because it held the row fixed while moving the offset
inside it.

What actually predicts the answer is the counter's position in the CACHE GEOMETRY — which 8-byte
bank of its 16-byte row it occupies, and its offset within that bank. And in every build ever
measured, **qc == k + 8**: the accumulator is invariably the inner counter's bank sibling.

| k bank | k off | result | builds |
|---|---|---|---|
| 1 | 4 | **576 correct** | shift0 |
| 1 | 0 | 909 | shift4, **wp0** (different row, different frame, same answer) |
| 0 | 4 | 567 | shift8 |
| 0 | 0 | 0x8000237 | shift12 (= 567 with bit 27 set, i.e. metadata bits) |

All offsets recomputed from the artifacts, not from notes.

### The RTL defect that matches the geometry — real, but NOT yet the root cause of this

Every line verified against the primary source:

- `commit_stage.sv:323-325` — `we_gpr_o[0] = 1'b1` unconditionally, but
  `cap_we_o[0] = commit_instr_i[0].cap_result.valid`. An ordinary instruction overwrites the
  integer register and leaves the shadow capability metadata STALE.
- `scoreboard.sv:242-246`, `:320-324` — a plain load retires on `LOAD_WB`, whose `cap_data` and
  `cap_result` are hard-wired `'0`. So a `lw` does NOT clear the shadow: the taint survives.
- `wt_dcache_wbuffer.sv:602` — every store captures whatever metadata is on the bus.
- `wt_dcache_mem.sv:138` — `st_wr_cap = |wr_user_i`: a store is classified as a capability store
  BY VALUE, NOT BY OPCODE.
- `wt_dcache_mem.sv:156-158`, `:225-238` — `k` is a genvar, so HW-bank 0 ALWAYS receives real
  data and HW-bank 1 receives metadata INSTEAD of data when `st_wr_cap` fires. A misclassified
  store whose own address lies in bank 1 therefore corrupts ITSELF.

Our own inner loop contains the taint sequence verbatim: `ldc a0` (the volatile read-back of `z`)
puts a capability in a0; the next `lw a0` makes it the counter without clearing the shadow; `sw
a0` then carries stale metadata.

**REFUTED as stated, and this is the second retraction of the day.** At SHIFT=0 the counter is at
`sp+0x1c`, in bank 1. If its store were misclassified it would receive metadata every iteration
and could not produce a clean count. Silicon measures a correct 576. So "the counter's own store
is misclassified" is refuted for the one geometry checkable exactly. The structural RTL bug stands
on its own — it is independently corroborated by this repo's own `verif/tests/custom/capstone/
cincoffset-stale-metadata.S`, which targets the same stale-shadow root cause for a different
symptom — but its application to THIS corruption is UNRESOLVED. Do not write it into ISSUES.md as
the root cause.

Also refuted: the over-wide-write idea sourced from `be_gen`. `extract_transfer_size`
(`ariane_pkg.sv:1119-1126`) fixes STC at 8 bytes / one beat and the metadata rides a separate
sideband, so there is no wider or multi-beat DATA write to blame.

### Still open

- `wp0` reported its witnesses intact, but that reading is NOT safe: the check loop is guarded by
  `i`, and a corrupted `i` would skip it and report clean. The loop-free raw readout (`wv1`) was
  collateral when `wr0` wedged. "The `stc` does not corrupt memory above it" is unconfirmed.
- Stage 14 (`wr0`, witnesses read every iteration, never written) has no result yet — it wedged.
- Stage 16 is the decisive test now: a `movc a0,zero / movc a1,zero` barrier that clears the
  shadow, against a same-size `nop` barrier as the control. Both arms are identical
  instruction-for-instruction and 13088 bytes, differing only in those two instructions and in
  jump targets that follow from base VA; both are QEMU-green at 576; both sit at (bank 0,
  offset 4), which measures 567 on silicon. Barrier cures and nop does not ⇒ mechanism confirmed
  and a workaround exists. Both 567 ⇒ refuted. Both 576 ⇒ layout cured it, attributable to
  nothing.

### Process

`ps8` — byte-identical to the working `sh8` modulo base VA — R-16 entry-stalled (no `SHA5`, no
`SHA6`) and, placed second in the boot, took the two arms that mattered with it. Every freshly
built image is an R-16 unknown regardless of whether its LOGIC is known; order by that, not by
confidence in the code. Two firmware builds in a row came out byte-identical in SIZE (15369224)
with different hashes.

## UPDATE 2026-08-07 (boot 52 + simulation): TWO MORE MECHANISMS DEAD, and an instrument that works

### Boot 52 -- both remaining board questions answered

Control k800 green, all four arms returned.

| rung | retval | meaning |
|---|---|---|
| `wv1` raw `wit[0]` | **0xA5A50000** | the witness is BIT-EXACT after 576 capability stores |
| `bar1` movc barrier | 567 | shadow-clearing barrier does NOT cure |
| `bar2` nop control | 567 | identical images bar two instructions |

**OVER-WIDE WRITE: REFUTED.** `wv1` reads the witness back with a loop-free, unguarded load, so
it closes the hole in `wp0`'s guarded check-loop reading. Memory immediately above the capability
store is not damaged. Independently, `extract_transfer_size` (`ariane_pkg.sv:1119-1126`) pins STC
at 8 bytes and one beat, and the metadata rides a separate sideband.

**STALE-METADATA STORE MISCLASSIFICATION: REFUTED for this defect.** `bar1`/`bar2` are identical
instruction-for-instruction and byte-identical in size (13088), differing only in
`movc a0,zero / movc a1,zero` versus two `nop`s, both QEMU-green at 576, both at (bank 0, off 4).
Clearing the shadow regfile changes nothing. The RTL bug is real and quotable; it is not this.

The bank-geometry model now predicts 7 of 7 builds -- `bar1`/`bar2` at (bank 0, off 4) both
returned the predicted 567.

### What survives: the LOAD side

Memory is correct and the load is answered wrong; the counter's read-modify-write then commits
the wrong value. `(bank, offset-in-bank)` is exactly address bits **[3:2]**, which are the bits
below the dword granularity at which both forwarding comparators match --
`store_buffer.sv:263,271,277` compare `page_offset_i[11:3]`, and `wt_dcache_mem.sv:266` compares
`wtag` against `(wbuffer_cmp_addr >> XLEN_ALIGN_BYTES)`.

### RTL SIMULATION NOW WORKS -- the instrument this investigation never had

Verilator model of the Capstone core, running directed tests in ~13 s against a built model, with
the RVFI tracer recording every load's address and returned value and every store's address and
data (`corev_apu/tb/rvfi_tracer.sv:108-118`). Needed: the Anvil compiler from
`corank/cva6-anvil-build`, Verilator pinned to **5.008** (`cva6.py:1033` hard-gates the version;
build it with `-j4`, higher OOM-kills `V3Ast.o`), 18 submodules, and stock
`gcc-riscv64-unknown-elf` -- the capstone opcodes are raw `.insn` directives, so no custom
toolchain. Do NOT iterate via `run_capstone_tests.sh`: it ends in `make clean`, deleting
`work-ver/` and forcing a full re-verilation. Call `cva6.py` directly.

`verif/tests/custom/capstone/stc-neighbour-load.S` (submodule) is the directed reduction.

### The directed test does NOT reproduce -- clean negative, fully controlled

    RTL HEAD  458982093   SUCCESS 6960 cycles, all four bits[3:2] cells 65 stores 0->64
    board rev 7aac52f93   SUCCESS 6960 cycles, all four bits[3:2] cells 65 stores 0->64

Identical to the cycle. The board's silicon is indistinguishable from HEAD here. This indicts the
test's FIDELITY, not the hypothesis: the real loop runs inside a capability domain after
`capenter`, on a monitor-carved stack, storing a capability loaded from the cap table. Raising the
test into a real domain context is the next step, and it costs no board time.

### Corrections made within the session

* "UNINIT raises UNEXPECTED_CAP_TYPE at the board revision where HEAD accepts it" -- WRONG,
  retracted. It happens identically at both; the comparison was a four-arm run against an
  eight-arm one. The uninit construction itself is illegal bare-metal.
* A run was read from a STALE log after the compile had failed. Delete the run artifacts before
  re-running so a failed compile cannot masquerade as a result; check the log timestamp.
* `.S` files go through cpp, so a bare `MACRO(...)` form inside a COMMENT expands and breaks the
  assembly. That is what made the compile fail.

## UPDATE 2026-08-07 (boot 53): THE 8-BYTE SEPARATION IS A NECESSARY CONDITION

The bits[3:2] law had a confound nobody had tested: in EVERY build ever measured, `qc == k + 8`
exactly, because clang always allocates the accumulator and the inner counter four bytes apart
with `p` in between. So "the answer depends on k's bits[3:2]" and "the answer depends on qc being
8 bytes from k" were indistinguishable.

Stage 18 (`FDREG_GAP`) inserts dead bytes BETWEEN qc and the counters. The gap must be a multiple
of 16 or k's own bits[3:2] move too. Built at the SHIFT=8 geometry, k held byte-identical at
sp+0x14 (bank 0, off 4) in all three arms, verified in the artifacts:

| rung | k | qc | qc-k | board |
|---|---|---|---|---|
| gp0 | sp+0x14 | sp+0x1c | **+8** | **567 WRONG** |
| gp16 | sp+0x14 | sp+0x2c | +24 | **576 CORRECT** |
| gp32 | sp+0x14 | sp+0x3c | +40 | **576 CORRECT** |

Control k800 green, all four arms returned.

**With k's geometry held EXACTLY fixed, moving qc out of the k+8 slot CURES the fault.**

### What this does to the law

As a GENERAL law, `value = f(k bits[3:2])` is REFUTED: it predicts 567 for gp16 and gp32, which
both have k at (bank 0, off 4), and they return 576.

As a CONDITIONAL law given `qc - k == 8`, it still holds across all seven original builds, and the
two new data points fit it exactly (gp0 bank0/off4 -> 567, matching shift8; wp0 bank1/off0 -> 909,
matching shift4).

So the correct statement is a CONJUNCTION:
 1. NECESSARY: the two read-modify-written counters are exactly 8 bytes apart. Break it and the
    fault vanishes regardless of k's position.
 2. GIVEN that, k's bits[3:2] select WHICH wrong value appears -- and one cell, (bank 1, off 4)
    = shift0, is benign.

### NOT the same 16-byte row -- cross-check against wp0

wp0 has k@sp+0x28 (row 0x20, bank 1) and qc@sp+0x30 (row 0x30, bank 0): `qc - k == 8` but the two
slots are in DIFFERENT 16-byte rows, and it still returns a wrong 909. So the necessary condition
is the 8-BYTE SEPARATION ITSELF, not co-residence in one cache row.

That cross-check also **weakens the dual-bank store-misclassification candidate**, which an earlier
note had promoted after boot 52. A misclassified store's dual-bank write is confined to its own
16-byte row (`bank_idx = '{default: wr_idx_i}`, wt_dcache_mem.sv:201), so it cannot reach a sibling
across a row boundary, and it therefore does not explain wp0. Do not record it as the root cause.
(The barrier result of boot 52 is also weaker than it looked: `movc a0,zero / movc a1,zero` clears
the REGISTER-FILE shadow only, while issue_read_operands.sv:690-693 takes `rs2_cap_metadata` from
`wb[k].cap_data.result_metadata` ungated by validity -- a path the barrier cannot reach. So boot 52
refuted regfile-shadow-sourced metadata, not WB-forwarded metadata.)

### Where that leaves the mechanism

A mechanism must now explain: two scalar RMW slots exactly 8 bytes apart, a 16-byte capability
store in the same loop, memory otherwise verifiably correct, corruption that survives a row
boundary, no reproduction bare-metal in simulation at either RTL revision, and nothing in QEMU.

## UPDATE 2026-08-07 (cycle-count analysis): IT IS TWO DIFFERENT FAULTS, AT OUTER-PASS GRANULARITY

The harness has recorded an mcycle delta bracketing the compute on every rung since the ladder
existed (`ladder_perf_domain.h`), it is printed on every RESULT line, and it had never once been
used as a discriminator. It settles what the returned value alone could not.

At 76.58 cycles/iteration, taken from the known-correct 576-iteration run:

| rung | retval | cycles | implied iterations | delta | outer passes |
|---|---|---|---|---|---|
| gp16 | 576 | 44109 | 576.0 | 0 | 0 |
| gp32 | 576 | 44114 | 576.1 | 0 | 0 |
| gp0 | **567** | 44045 | **575.2** | -9 | **-1** |
| bar2 | **567** | 44004 | 574.6 | -9 | -1 |
| wp0 | **909** | **69568** | **908.5** | +333 | **+37** |

(bar1 also returns 567 at 46323 cycles; it carries two extra barrier instructions per iteration,
so its cycles/iteration is higher. Not a discrepancy.)

**The 909 cell and the 567 cell are NOT two severities of one fault.**

* **909 genuinely executed 909 iterations.** 69568/44109 = 1.5772 against 909/576 = 1.5781 -- a
  0.06% match. The loop lost control of its trip count and really did run the extra passes.
* **567 executed the FULL 576 iterations** in the same cycles as a correct run, and still returned
  567. Nine increments were LOST without a single iteration being lost.

**Every deviation is an exact multiple of the inner trip count, 9.** 576-567 = 9 = exactly ONE
outer pass of accumulator increments vanished. 909-576 = 333 = 37*9 = exactly 37 EXTRA outer
passes ran. So the fault operates at OUTER-PASS granularity, not per-iteration.

### What that does to the framing

At -O0 the three counters are packed 4 bytes apart: k, p = k+4, qc = k+8. The victim differs by
cell -- in the 567 cell the ACCUMULATOR loses one outer pass, in the 909 cell the OUTER counter is
disturbed so extra passes run. The four-cell "law" table therefore conflates two distinct failure
modes under one number, and "the counter gets a wrong value" is too coarse a description to
mechanise.

It also explains boot 53: moving qc to k+24 cured the 567 cell because qc was the victim there,
while p stayed at k+4 throughout.

### Consequence for the next experiments

Any probe that returns only qc is reading one of two faults without knowing which. Rungs must
report the INNER and OUTER counters separately, and the cycle count must be read on every arm --
it is free, already collected, and it is the only instrument so far that distinguishes "lost an
iteration" from "lost an increment".

## UPDATE 2026-08-07 (boot 54): RETRACT the 8-byte-separation condition. BOTH geometric laws are dead.

Boot 54 swept the separation window. Control k800 green, all arms returned.

| build | k | p | qc | sep(k,qc) | retval | cycles | fault type |
|---|---|---|---|---|---|---|---|
| gp0 | 0x14 | 0x18 | 0x1c | +8 | 567 | 44045 | accumulator, -1 pass |
| sep12 | 0x10 | 0x14 | 0x1c | **+12** | **567** | | accumulator, -1 pass |
| sep20 | 0x18 | 0x1c | 0x2c | **+20** | **906** | 68709 | outer, +36.7 passes |
| gp16 | 0x14 | 0x18 | 0x2c | +24 | 576 | 44109 | none |
| gp32 | 0x14 | 0x18 | 0x3c | +40 | 576 | 44114 | none |

**RETRACTED: "the defect requires two RMW slots exactly 8 bytes apart" (commit 2a9ef7a255ac).**
Separations of 12 and 20 also fail. And separation 8 is not sufficient either -- shift0 has
sep(k,qc) = 8 and returns a correct 576. So the 8-byte separation is neither necessary nor
sufficient, and boot 53's reading of gp16/gp32 as "breaking the separation cures it" confused
"changed the separation" with "changed the layout".

**ALSO STILL DEAD: value = f(k bits[3:2]).** It mispredicts gp16 and gp32 (k at bank 0 / off 4,
predicted 567, measured 576).

So neither geometric variable explains the set. Checked against all ten builds, no single one of
{k address, p address, qc address, sep(k,qc), k bits[3:2], qc bits[3:2], same-16-byte-row} is a
function of the outcome -- each has at least one pair of builds that share the variable's value and
disagree on the result. Two overfits in one day, both to samples where the -O0 allocator was
silently holding other slots constant.

An independent analysis reached this before the boot did, by disassembling every shipped artifact
rather than trusting the recorded table: builds with byte-identical frame geometry return DIFFERENT
values (906 vs 909). sep20 vs shift4 is exactly that pair. A law that is not a function is not a
law.

### What still stands

* The two-fault split from the cycle counts (previous section). sep20 confirms it independently:
  68709 cycles = 897 implied iterations against a returned 906, i.e. the outer-disturbance fault,
  and it is +36.7 outer passes -- the same family as wp0's +37.
* Every deviation remains an exact multiple of the inner trip count 9.

### What to do instead of another geometry sweep

Stop sweeping geometry. The next probe must report the INNER and OUTER counters SEPARATELY plus
the cycle count, because every rung so far has returned only qc and therefore reported one of two
faults without saying which. Until a probe distinguishes them at the source, more geometry points
will keep producing laws that hold until the next build.

## UPDATE 2026-08-07 (boot 55): TWO VICTIMS IDENTIFIED. k is knocked BACKWARDS; qc LOSES stores.

Stage 19 returns all three counters packed (p<<20 | k<<16 | qc) instead of qc alone. Control green.

| rung | shift | p | k | qc | cycles | implied iters |
|---|---|---|---|---|---|---|
| c0 | 0 | 64 | 9 | **576 correct** | 44001 | 575 |
| c4 | 4 | 64 | 9 | **909** | **69081** | **~904** |
| c8 | 8 | 64 | 9 | **567** | 44074 | 576 |

**p and k read NORMAL at exit in both failing cells -- and that is not the same as the loop being
undamaged.** Both loops terminate on their own conditions (`k < 9`, `p < 64`), so an index that is
transiently knocked backwards runs extra iterations and still exits at exactly k=9, p=64. The
cycle count is what exposes it, and it is the only instrument here that can: c4 spent 69081 cycles
against c0's 44001 for the same nominal work.

### The two faults, with victims

* **c4 (909-family): the victim is k, the INNER INDEX.** It is transiently corrupted mid-loop, the
  inner loop executes ~904 iterations instead of 576, and qc faithfully counts them -- 909 against
  the ~904 the cycles imply. The accumulator is innocent in this cell.
* **c8 (567-family): the victim is qc, the ACCUMULATOR.** The cycle count confirms exactly 576
  iterations executed; nine increments were simply lost. The loop index is innocent in this cell.

So the same rung family produces two mechanically distinct failures -- a LOST STORE to one scalar,
and a BACKWARDS CLOBBER of another -- and every previous probe returned a single number that
averaged them. That is why two successive geometric "laws" each held until the next build: they
were fitted across two populations.

### Consequence

A mechanism must now explain BOTH: a scalar whose stores are silently dropped (~1 in 64), and a
scalar that is knocked to a smaller value mid-loop. Both while memory adjacent to the capability
store is verifiably intact, both only inside a real domain, and neither reproducing bare-metal in
simulation at either RTL revision.

Stage 20 (built, slots verified) is the next discriminator: it swaps the DECLARATION order so the
same three slots hold different variables (k@0x14, qc@0x18, p@0x1c against stage 19's k@0x14,
p@0x18, qc@0x1c). If the damage follows the SLOT the victim changes with it; if it follows the
VARIABLE the accumulator keeps losing stores wherever it lives. p == k+4 and "the accumulator is
the upper slot" have been welded together in every build ever made.

## UPDATE 2026-08-07 (boot 56, ROLE SWAP): THE DAMAGE FOLLOWS THE SLOT, NOT THE VARIABLE

Stage 20 swaps the declaration order so the SAME three frame slots hold DIFFERENT variables, with
k left exactly where it was. Control green, all arms returned, all report the packed triple.

| build | k | slot sp+0x1c | high slot | retval | cycles | implied iters | victim |
|---|---|---|---|---|---|---|---|
| c0 | 0x1c | *k* | qc@0x24 | qc=576 correct | 44001 | 576 | none |
| rs0 | 0x1c | *k* | p@0x24 | qc=576 correct | 43998 | 576 | none |
| c4 | 0x18 | **p** | qc@0x20 | qc=909 | 69081 | ~904 | **p**, extra outer passes |
| rs4 | 0x18 | **qc** | p@0x20 | **qc=504** | 44032 | 576 | **qc**, 72 stores lost |
| c8 | 0x14 | **qc** | p@0x18 | qc=567 | 44074 | 576 | **qc**, 9 stores lost |
| rs8 | 0x14 | **p** | qc@0x18 | **qc=585** | 44752 | 586 | **p**, one extra pass |

**In all four failing arms the victim is whichever variable occupies sp+0x1c.** Swapping the
declarations at a FIXED k flips the failure mode -- which is what "follows the slot" predicts and
what "follows the variable" forbids:

* **qc at 0x1c** -> stores silently dropped. 567 at the shift8 geometry, 504 (72 lost = 8 outer
  passes) at shift4. Iteration count correct in both, confirmed by cycles.
* **p at 0x1c** -> the outer loop runs extra passes. 909 (~904 iterations) at shift4, 585 (586
  iterations) at shift8. qc faithfully counts the extra iterations in both.

So the two "different faults" are ONE fault in one slot, and the signature is merely a consequence
of which variable the compiler put there. That also explains every dead geometric law: they all
indexed on k, and **k is the one slot that is never the victim**.

### What does NOT fit yet -- do not overstate this

* **k at sp+0x1c is CLEAN** (c0, rs0). The slot is not universally poisoned. The one structural
  difference is that k is the only counter READ TWICE per iteration -- once for the loop compare
  (`li a0,0x8; blt`) and once for the increment -- while qc and p are read once.
* **sep20** (frame 0x60) has qc at the corresponding frame offset yet fails on p, so a purely
  absolute-address rule does not cover it either.

So "sp+0x1c is the victim" holds across these six builds and is NOT yet the whole rule. The next
discriminator is whether the operative address is sp-relative, s0-relative, or something about the
access pattern that spares a twice-read slot.

## UPDATE 2026-08-07 (boot 57): THE VICTIM IS A FIXED STACK ADDRESS -- sp+0x1c

Boot 56 showed the damage follows the SLOT. Boot 57 asks whether that slot is sp-relative or
s0-relative, using three frame-0x60 builds where the two rules point at DIFFERENT variables, with
the triple report naming the victim instead of inferring it.

| build | frame | k | p | qc | sp+0x1c holds | sp-rule | s0-rule | MEASURED |
|---|---|---|---|---|---|---|---|---|
| t16 | 0x60 | 0x14 | 0x18 | 0x2c | (gap) | correct | qc damaged | **576 correct** |
| t12 | 0x60 | 0x18 | **0x1c** | 0x2c | **p** | p damaged | qc damaged | **qc=906, 900 iters -> p damaged** |
| t0b | 0x60 | 0x18 | **0x1c** | 0x30 | **p** | p damaged | correct | **qc=909, 904 iters -> p damaged** |

**sp-rule 3/3. s0-rule 0/3. The victim is a fixed offset from the STACK POINTER: sp+0x1c, i.e. 28
bytes above the 16-byte capability store at sp+0x00.**

### The rule, checked against every build in the investigation

Whichever loop variable occupies sp+0x1c is the one damaged, and the signature follows from which
variable it is:

* **qc there** -> stores silently dropped: 567 (shift8, gp0, sep12, c8), 504 (rs4)
* **p there** -> outer loop runs extra passes: 909 (shift4, c4, t0b), 906 (sep20, t12), 585 (rs8)
* **k there** -> SPARED: shift0, c0, rs0 all correct. k is the only counter READ TWICE per
  iteration (loop compare + increment).
* **padding/nothing there** -> correct: gp16, gp32, t16

**17 of 19 builds fit.** This is why every earlier geometric law died: they all indexed on k, which
is the one variable that is never the victim, and they were fitted across builds where a different
variable happened to occupy the poisoned slot.

### The two mismatches, and why they are testable rather than fatal

Both are builds where the victim was INFERRED, not measured with the triple report:

* **shift12** -- never triple-reported. Its 0x8000237 was read as "567 with bit 27 set"; under the
  new model that is a qc-damaged value with metadata-shaped bits, but sp+0x1c holds `s` there.
* **wp0** -- stage 13, a structurally different loop body (witness init before the loop), victim
  inferred from cycles alone, and sp+0x1c holds the witness pad.

Re-running both geometries under stage 19 settles whether the rule is 19/19. Do that before
treating either as a counter-example.

### What a mechanism must now explain

A fixed stack address 28 bytes above a 16-byte capability store, whose stored value is transiently
wrong -- dropped stores when the slot is written once per iteration, a backwards clobber when it
gates a loop -- while memory ADJACENT to the store is verifiably intact (witness reads back
bit-exact), the final values read normal because both loops exit on their own conditions, and the
whole thing is invisible bare-metal in simulation at either RTL revision.

## UPDATE 2026-08-07 (boot 58): FRAME-RELATIVE, NOT ABSOLUTE -- and the first retraction was half right

sp has been identical in every build (same domain, same entry, same call depth), so "the victim is
at sp+0x1c" and "the victim is one fixed ABSOLUTE stack address" fit all prior data equally. Stage
23 separates them by reaching the loop through extra noinline frames: an added frame shifts every
absolute address while leaving all sp-relative offsets inside the body untouched. Verified in the
artifacts -- all three arms have k@0x14, p@0x18, qc@0x1c and 0/1/2 real wrapper functions.

| rung | wrapper frames | p | k | qc | cycles |
|---|---|---|---|---|---|
| dp0 | 0 | 64 | 9 | **567** | 44213 |
| dp1 | 1 | 64 | 9 | **567** | 44400 |
| dp2 | 2 | 64 | 9 | **567** | 44653 |

**Bit-identical across all three depths. The fault follows the FRAME-RELATIVE offset, not an
absolute address.** (Cycles rise slightly with depth -- the wrappers' own prologue/epilogue -- and
the implied iteration count stays at 576 + that overhead, so the loop is intact and qc lost 9 in
every arm.)

Since the 16-byte capability store is at sp+0x00 in every build, the victim is equivalently **a
fixed offset from the capability store: +0x1c from its base, i.e. 12 bytes above its 16-byte end.**
Moving the store to separate those two readings is not possible with this frame shape -- an array
big enough to move the store also swallows sp+0x1c (tried, stage 22, CAPSLOT knob retained).

### This half-vindicates the first retraction

The original table read "12 bytes above the store -> CORRECT" and that reading was retracted. It was
wrong about WHAT, not about WHERE: that cell is shift0, where sp+0x1c happens to hold **k, the one
variable that is immune**. An immune occupant was read as a clean address. The distance was real;
the conclusion drawn from it was not.

### The rule as it now stands

**The slot at capability-store + 0x1c is corrupted, and whichever loop variable occupies it is the
victim** -- qc there loses stores, p there lets the outer loop run extra passes, k there is spared,
padding there is harmless. 17/19 builds fit; the two exceptions (shift12, wp0) are both builds whose
victim was inferred rather than triple-reported and are re-testable.

k's immunity is the sharpest remaining clue: it is the only counter READ TWICE per iteration (loop
compare, then increment).

## UPDATE 2026-08-07 (boot 59): the READ-COUNT explanation of k's immunity is REFUTED

k is spared whenever it occupies the poisoned slot, and the one structural difference was that k is
read TWICE per iteration (loop compare, then increment) while qc and p are read once. Stage 24 adds
a second read of qc -- an always-false compare, so the value and the oracle are unchanged -- and
places qc at the poisoned slot.

| rung | qc position | qc reads/iter | qc | cycles |
|---|---|---|---|---|
| rr0 | 0x1c | 1 | **567** | 44081 |
| rr1 | 0x1c | **2** | **567** | 48667 |
| rr2 | p at 0x1c, qc reread | - | NO RESULT (entry stall, no verdict) |

The extra reads are real -- 9 `lw` against 7, and the cycle count rises accordingly -- and qc loses
exactly the same 9 stores. **k's immunity is NOT about the read count.**

Remaining candidates for k's immunity, in the order they should be tested:
1. **Program-order position.** qc's RMW is the FIRST scalar access after the capability store; k's
   is the last in the body. But note rs8 damaged p, whose increment lives in the OUTER tail, far
   from the capability store in program order -- so this is not obviously it either.
2. **k is the loop's bound-compared variable**, so its value is consumed by control flow every
   iteration rather than only accumulated.
3. k is re-initialised (`k = 0`) at the top of every outer pass, which qc and p are not.

Candidate 3 is the cheapest to test: give qc an equivalent re-initialisation that cannot change its
value, and see whether it becomes immune.

## UPDATE 2026-08-07 (boot 60): k's RE-INITIALISATION is NOT what spares it either

Stage 25 carries the inner index across outer passes so it becomes MONOTONE, losing the
re-initialisation and nothing else, with the index at the poisoned slot.

| arm | k at 0x1c | k behaviour | measured | verdict |
|---|---|---|---|---|
| bs16 | yes | reset each outer pass | p=64 k=9 qc=576 | **correct** (control: k immune) |
| nr16 | yes | **monotone, never reset** | p=64 k=576 qc=576 | **correct -- STILL IMMUNE** |
| or20 | (order swapped) | - | p=64 k=9 qc=587 | damaged (+11) |

**k stays immune when monotone. The re-initialisation hypothesis is REFUTED**, after the read-count
one died in boot 59. Of the three structural differences originally listed, the survivor is that k
is the BOUND-COMPARED variable: its value feeds a conditional branch every iteration, which neither
qc nor p (in the inner body) does.

### PROCESS: the oracle was wrong, not the board

The runner reported nr16 as WRONG against oracle 1076630592. That oracle was MY arithmetic error --
(64<<24)|(576<<12)|576 = 0x40240240 = 1076101696, which is exactly what the board returned. The arm
PASSED. Decoding the returned value rather than trusting the pass/fail verdict is what caught it;
without that this would have been recorded as a false positive on the decisive arm. Compute packed
oracles programmatically, never by hand.

### or20 needs its slot map verified before use

or20 is damaged (qc=587) but the ORDER variant's RMW-site detection found only two sites, so which
variable sits at the poisoned slot is not established for it. Do not fold or20 into the slot rule
until its layout is confirmed in the artifact.

## UPDATE 2026-08-07 (analysis): "k IS IMMUNE" IS REFUTED -- it was a THIRD confound of the same class

An independent analysis compiled the whole knob space and read the frames out of the generated code
rather than trusting the recorded table. Verified against all 20 builds here:

* **k is the LOWEST of the three counters in every build** (-O0 packs k, p=k+4, qc=k+8, and
  FDREG_ORDER leaves the slot map byte-identical).
* Therefore **k@0x1c can only ever occur when p and qc sit at 0x20/0x24** -- i.e. only when the
  16-byte row 0x10..0x1f holds nothing else. All five such builds (shift0, c0, rs0, bs16, nr16)
  are correct.

So "k is immune" and "the poisoned slot is ALONE in its row" are perfectly confounded, and the
second is the better rule:

**THE SLOT AT CAPABILITY-STORE + 0x1c IS CORRUPTED IFF IT IS OCCUPIED AND SHARES ITS 16-BYTE ROW
WITH AT LEAST ONE OTHER READ-MODIFY-WRITTEN SCALAR.** 19 of 20 builds fit (vs 17/19 for the k-rule).
The sole mismatch is shift12, whose victim was never triple-reported.

**This is the THIRD confound of this exact class** -- after qc==k+8 and k's bits[3:2]. Each time the
-O0 allocator held another variable constant and the correlate was read as the cause. Boots 59 and
60 were spent testing properties of k (read count, re-initialisation) that were never the variable.
Before proposing any rule indexed on a variable's IDENTITY, check whether the allocator makes that
identity a proxy for a layout property.

### ALSO RETRACTED: "every deviation is an exact multiple of the inner trip count"

False. or20 is +11 and sep20/t12 are +330; neither is divisible by 9. The multiples that do hold
(-9, +333, +9, -72) are real but not universal, and no mechanism should be built on them.

### or20 is a CONFIRMING build, not an anomaly

FDREG_ORDER=1 at SHIFT=20 gives k@0x18, **p@0x1c**, qc@0x20. The poisoned slot held p, and qc=587
(+11, extra outer passes) is exactly the p-family signature. It also means or20 tested NOTHING about
k, because k was never at the poisoned slot. Open item closed.

### The RTL search space is now genuinely small

The fault discriminates WITHIN a single 8-byte dword: in c8, p@0x18 is undamaged (cycles confirm 576
iterations) while qc@0x1c loses 9 stores -- the two halves of one dword. Every address comparator in
the store/forward path is blind to bit [2] (store_buffer.sv:263,271,277 compare page_offset_i[11:3];
the wbuffer tags at [..:XLEN_ALIGN_BYTES]). **That eliminates the whole address-tag forwarding class
as the selector.** The selector must be BYTE-GRANULAR: be_gen (ariane_pkg.sv:1044-1057, keys on
addr[2:0]), the per-byte overlay rd_data_o[8*k+:8] = wbuffer_be[k] ? ... (wt_dcache_mem.sv:311-317),
or the wbuffer per-byte valid/dirty/txblock bookkeeping and bdirty_off / toSize64(bdirty)
(wt_dcache_wbuffer.sv:253-295, :397, :427).

## UPDATE 2026-08-07 (boot 61): THE +0x1c ANCHOR IS REFUTED. No single-address rule fits.

Stage 26 adds a FOURTH read-modify-written scalar `d`, incremented immediately before qc so the two
must stay equal, which lets the 16-byte row be populated independently of which counter sits at the
supposed poisoned slot.

| arm | layout | slot 0x1c holds | measured |
|---|---|---|---|
| ka0 | d@0x18, k@**0x1c**, p@0x20, qc@0x24 | k | **d=18** (lost 558 of 576), qc=576 correct |
| kb12 | d@**0x1c**, k@0x20, p@0x24, qc@0x28 | d | d=576 **correct**, **qc=567** (lost 9) |
| kc20 | - | - | NO RESULT (wedged), no verdict |

**In ka0 the victim is at 0x18; in kb12 the victim is at 0x28 and the 0x1c slot is UNDAMAGED.**
The capability-store + 0x1c anchor (boots 57/58) is refuted, and with it the row-occupancy rule
built on top of it.

Fitted against all 14 triple-reported builds, no single anchor works:
    sp+0x1c   8/14
    s0-0x38   7/14
    s0-0x34   7/14

### Two NEW facts that are not in dispute

* **A far more severe signature exists.** d lost 558 of 576 stores (97%). Everything previously
  measured was <= 72 lost. Same loop, adjacent statements (`d++; qc++;`), and qc was EXACT.
* **The fault is extremely selective between adjacent slots.** In ka0, d@0x18 is destroyed while
  qc@0x24 is perfect, in the same iterations. In kb12 the reverse: d@0x1c perfect, qc@0x28 short.

### Why no fourth rule is proposed here

Three geometric rules have been proposed and refuted today -- k bits[3:2], the 8-byte separation,
and the +0x1c anchor with its row-occupancy refinement -- each fitted to the builds available at the
time and each broken by the next build. ka0/kb12 are also the ONLY builds with four RMW scalars, so
they differ structurally from the other twelve and the temptation to fit them together is exactly
the trap. The loss rates differ by 60x (97% vs 1.6%), which is itself evidence they may not be one
phenomenon.

**The next analysis must fit ALL builds simultaneously rather than pattern-match the newest pair,
and should treat "one fault" as a hypothesis rather than an assumption.** The dataset is now 21
builds with layout and outcome recorded; that is enough to test candidate rules offline, without
spending boots.

## UPDATE 2026-08-07 (OFFLINE analysis): fit rules against ALL builds, not the newest pair

Four geometric rules were proposed and refuted in one day, each fitted by eye to whichever builds
existed at the time and each broken by the next boot. Every one of them could have been killed for
free: 19 builds with recorded layout and outcome are on disk, and testing a candidate rule against
all of them takes milliseconds. A boot takes 6-12 minutes and yields ONE point.

Two scripts now do this, and they read the LAYOUT OUT OF THE ELF rather than from notes:
  `extract-frame-layout.py`  -- frame size, store offset, RMW slots, loop-bound constants per build
  `fit-victim-rules.py`      -- scores candidate anchors and structural predicates over the set

### Result: no single-address anchor fits

    store+0x1c  13/19     s0-0x34   9/19     lowest_rmw   1/19
    sp+0x1c     13/19     s0-0x38   9/19     highest_rmw  6/19
    store+0x18   5/19                        2nd_rmw      5/19

### THE ONE INVARIANT WITH NO EXCEPTIONS (9/9 known victims)

**Every victim sits in the UPPER 8-byte half of its 16-byte row** -- row offset 8 or 12, never 0 or
4:

| builds | victim | row offset |
|---|---|---|
| c4, c8, rs4, rs8, t12, t0b, dp0 | 0x1c | 12 |
| ka0 | 0x18 | 8 |
| kb12 | 0x28 | 8 |

This is a real constraint, not an artifact of where slots land: the UNDAMAGED builds also have
upper-half RMW slots (c0/rs0/bs16/nr16 at 0x1c; t16/gp16 at 0x18 and 0x2c; gp32 at 0x18 and 0x3c).
The upper half is the bank that receives METADATA in the dual-bank write path, and it is consistent
with the separately established fact that the fault discriminates WITHIN a dword.

### Best structural rule, and its exceptions

"damaged iff two RMW scalars share the upper half of one 16-byte row" fits 17/19; it fails on kb12
and sep12. Not proposed as the rule -- recorded so it is not re-derived.

### This is consistent with SEVERAL faults

The row-offset-12 group (7 builds, deltas -9/-72/+9/+330/+333) and the row-offset-8 group (2 builds,
deltas -558/-9) separate cleanly, and the offset-8 group appears ONLY in the two builds carrying a
fourth RMW scalar. A 60x spread in loss rate (1.6% to 97%) across one "rule" is itself evidence
against unity. Treat "one fault" as a hypothesis under test, not a premise.

### Method note

Before any further board time on geometry: run `fit-victim-rules.py`. If a proposed rule already
fails on the existing 19 builds, it does not need a boot to refute it.

## UPDATE 2026-08-07 (boot 62): **THE SLOT IS RESET TO ZERO.** Not "lost stores". Mechanism found.

Every probe until now reported a counter short by an exact number of increments and read that as
dropped stores. Stage 27 starts the accumulator at a SENTINEL of 1,000,000 instead of 0, which makes
the two readings impossible to confuse.

| rung | started at | returned | cycles | reading |
|---|---|---|---|---|
| sn0 (control, shift0) | 1,000,000 | **1000576** | 44040 | correct |
| **sn8** (qc at 0x1c) | **1,000,000** | **567** | 44012 | **SLOT ZEROED, then counted 567** |
| sn4 (p victim, shift4) | 1,000,000 | 1000906 | 68612 | sentinel intact, +330 extra iterations |

**sn8 started at one million and returned 567.** Lost increments would give 1000567. The slot was
RESET TO ZERO and counted up from there. 576 - 567 = 9 = exactly the inner trip count, so it was a
single reset at an outer-pass boundary.

### Every earlier measurement is reinterpreted

* qc=567 -> reset at iteration 9. qc=504 -> reset at 72. d=18 -> reset at 558.
* p as victim -> the OUTER counter is zeroed, so the outer loop RESTARTS and runs EXTRA passes:
  +9, +330, +333. sn4 confirms it directly -- qc keeps its sentinel AND gains 330 iterations.
* 0x8000237 = the same reset with one metadata bit not zero.

The "two different faults" reading (dropped stores vs extra passes) collapses: it is ONE fault --
a slot zeroed -- whose visible signature depends only on whether the zeroed slot gates a loop.

### Why it is always the UPPER half of the row -- the invariant now has a cause

`bank_wdata[k][j] = ... (((st_wr_cap) && (k==1)) ? wr_user_i : wr_data_i)` (wt_dcache_mem.sv:156-158)
and `bank_be` applies the SAME byte-enable to both banks. So a store whose address lies in BANK 1 has
its own slot written with `wr_user_i` at those byte lanes -- the store data never lands. Bank 1 is the
ONLY bank that can receive something other than the store's data, which is precisely why the victim is
always at row offset 8 or 12 and never 0 or 4 (9/9 builds). Where those metadata bytes are ZERO the
slot is zeroed rather than corrupted -- which is why victims hold plausible counts, the very
observation that caused this mechanism to be refuted twice.

### Why the earlier refutations were wrong

* "Victims hold plausible counts, so it is not metadata substitution" -- wrong: substituting ZERO
  metadata bytes produces a plausible count.
* "The movc barrier does not cure it" -- that barrier clears the REGISTER-FILE shadow only, while
  issue_read_operands.sv:690-693 sources rs2_cap_metadata from wb[k].cap_data.result_metadata
  UNGATED BY VALIDITY. The barrier could not reach the path that supplies the bus.

### Still to close

st_wr_cap = |wr_user_i (wt_dcache_mem.sv:138) classifies by VALUE not opcode, so the remaining
question is what puts non-zero metadata on the bus for an ordinary `sw` while the bytes at the
store's own lanes are zero. Confirm against the WB-forwarding path, and confirm the reset value is
zero rather than low-order metadata by sweeping the sentinel.

## ROOT CAUSE (2026-08-07) -- a plain scalar store in the UPPER bank of a cache row is overwritten with capability metadata

Every link verified at the primary source.

1. **`core/issue_read_operands.sv:690`** -- `rs2_cap_metadata` is forwarded from the WRITEBACK PORT
   with **no validity gate**:
   `assign rs2_cap_metadata[i][k] = ((issue_instr_i[i].rs2 == fwd_i.sbe[fwd_i.wb[k].trans_id].rd) ?
    fwd_i.wb[k].cap_data.result_metadata : ...)`
   The scoreboard-port version ~25 lines later DOES check `fwd_i.sbe[k].cap_result.valid`. This one
   does not. So an ordinary `sw` can pick up stale capability metadata from a resident WB slot.
2. **`core/cache_subsystem/wt_dcache_mem.sv:138`** -- `assign st_wr_cap = |wr_user_i;`
   A store is classified as a CAPABILITY STORE BY VALUE, NOT BY OPCODE. Non-zero stale metadata on
   the sideband is sufficient to misclassify a plain scalar store.
3. **`wt_dcache_mem.sv:230-238`** -- a classified store sets `bank_req = '1; bank_we = '1`, writing
   BOTH banks of the 16-byte row rather than only the bank its address selects.
4. **`wt_dcache_mem.sv:156-158`** --
   `bank_wdata[k][j] = ... (((st_wr_cap) && (k==1)) ? wr_user_i : wr_data_i);`
   **Bank 1 (the upper 8 bytes of the row) is the ONLY bank that can receive anything other than the
   store's own data.**

**NET: a plain scalar store whose address lies in the UPPER 8 bytes of a 16-byte cache row has its
own slot overwritten with capability metadata instead of its data. Where those metadata bytes are
zero at the store's byte lanes, the scalar is silently ZEROED.**

### It accounts for every measurement, quantitatively

Each victim decomposes exactly as `clobber_value + (576 - reset_iteration)`:

| build | final | clobber value | reset iteration |
|---|---|---|---|
| shift8 / gp0 / c8 / dp0 / sn8 | 567 | 0 | 9 |
| rs4 | 504 | 0 | 72 |
| ka0 (d) | 18 | 0 | 558 |
| **shift12** | **0x08000237** | **0x08000000** | 9 |

shift12 is the clincher: `0x08000237 = 0x08000000 + 567` -- clobbered with a value carrying **bit
27**, i.e. literal capability-metadata bits, then counted up 567 times. Same family, same reset
iteration, different clobber value.

* The 9/9 upper-half invariant: bank 1 is structurally the only bank that can receive non-store data.
* Reset-to-zero rather than lost stores: proven by the sentinel (sn8 started at 1,000,000, returned 567).
* p as victim -> zeroing the OUTER counter restarts the loop -> extra passes (+9, +330, +333).
* QEMU clean: no metadata sideband exists there to misclassify.
* Bare-metal simulation silent: the directed tests never produced stale WB-forwarded metadata on a
  scalar store's rs2.

### Why this mechanism was refuted TWICE and both refutations were wrong

* **"The movc barrier does not cure it"** (boot 52) -- that barrier clears the REGISTER-FILE shadow.
  The metadata reaching the bus comes from the ungated WB forward at issue_read_operands.sv:690,
  which the barrier cannot touch. The test could not reach the path it was aimed at.
* **"Victims hold plausible counts, not garbage"** -- substituting ZERO metadata bytes produces
  exactly a plausible count. The one build where the metadata bytes were non-zero (shift12) DID
  produce a garbage-looking value, and it decomposes exactly.

### Fix directions (not yet implemented)

* Gate the WB-port forward on validity, matching the scoreboard-port version
  (`issue_read_operands.sv:690`, and the rs1/rs3 siblings above and below it).
* And/or classify capability stores by OPCODE rather than by `|wr_user_i` (`wt_dcache_mem.sv:138`).
* Either is an RTL change requiring a bitstream reflash -- the project lead's call.

## UPDATE 2026-08-07 (simulation, condition PROVEN created): the store-misclassification family is DEAD

The audit's first recommendation was a directed test that actually CREATES the condition, rather
than one that hopes for it. `verif/tests/custom/capstone/scalar-store-cap-operand.S` does exactly
that: the stored register is written by a capstone FLU op (`CINCOFFSET`), so its metadata shadow
holds a real capability, and it is then stored with a PLAIN `sw` into a bank-1 slot with witnesses
in the other three slots of the row.

**`CAPPRINT` proves the condition existed** — the console shows
`Reg[14]: Cursor: 0000000080003000 | Metadata-> Revnode_id: 2 | Type: 1 | Perm: 7 | Start/End set`.
A negative result is worthless without that; five earlier simulation rounds were clean precisely
because they never created the trigger (they stored `addi` results, and `ex_stage.sv:1081` zeroes
the FLU writeback for non-capstone ops).

**Result: PASS.** The target slot received the stored value, the dual-bank sibling at the same byte
lanes was untouched, and both other witnesses kept their sentinels.

**So a plain `sw` carrying real capability metadata is NOT misclassified and does NOT dual-bank
write.** The whole family — the retracted chain and its variants — is closed. `issue_read_operands`,
`st_wr_cap = |wr_user_i` and the `bank_wdata[1]` mux are not producing this.

### This also downgrades the one thing that looked reportable on its own

`st_wr_cap = |wr_user_i` classifying by value, plus `compress_cap(null) = 0x08000000` rather than
zero, remains a true statement about the code. But a directed test that puts a real (non-zero)
capability metadata word behind a plain scalar store shows **no consequence at all**. Either the
routing is gated somewhere the code reading missed, or the misclassification has no observable
effect. **Do not hand it to the hardware owner as a defect** — it is a code-reading concern that
the one test able to confirm it does not confirm.

### Where that leaves R-18

A reproducible, deterministic silicon defect with a **necessary condition** (victim in the upper 8
bytes of its 16-byte row, 9/9) and **no mechanism**. Every mechanism proposed so far is refuted.
The remaining untested lead is the outer-pass alignment: all three measured reset points (9, 72,
558) are multiples of the inner trip count, which points at something happening once per outer pass
rather than per iteration. Stage 28 (`FDREG_INNER`) decouples the inner trip count so that is
falsifiable; it is built but not yet run on the board.

## UPDATE 2026-08-07 (boot 65): MOVING THE COUNTERS OFF THE STACK CURES IT — this is NOT cleanly an RTL defect

Prompted by a challenge to the premise: is this even a hardware problem? Three suspects in our own
stack had never been tested — the compiler, the monitor/glue, and the domain setup. Only the
measurement harness had been excluded.

**Boot 65, controlled: stage 32 changes ONE thing from the failing c8** — the three counters live in
a GLOBAL instead of on the monitor-carved stack. Same compiler, same -O0, same loop source, same
16-byte capability store, both per-iteration `ldc`s retained (verified in the artifact: `stc` at
0x3051c, `ldc 0xa0(gp)`, `ldc 0x70(a0)`).

| rung | counters | retval | cycles |
|---|---|---|---|
| gv3 | **global**, victim row offset 12 | **576 correct** | 52825 |
| gv1 | **global**, row offset 4 | 576 correct | 52839 |
| c8 | **stack** (original) | **qc=567 WRONG** | 44098 |

**The defect vanishes when the counters are not on the monitor-carved stack.** It is therefore NOT
a property of the row geometry alone, and "silicon defect" is not a claim we can make.

### Consequence: do not send this to the hardware owner

The reproducer stands as a reproducer. The attribution does not. A defect that disappears when our
own software relocates the variables is at least as likely to be in our monitor's stack carve, the
stack capability's type/bounds/provenance, or the ABI, as in the RTL.

### The sharper hypothesis this suggests

In the failing build the SAME frame capability (s0, derived from sp) is used to derive BOTH the
16-byte `stc` target and the scalar counter addresses. In gv3 the counters are reached through `gp`
-- a DIFFERENT capability -- while the `stc` still goes through s0. So the trigger may be
**"capability store and scalar accesses derived from the same capability"**, which is an ABI/monitor
property rather than a hardware one. Untested, and the obvious next experiment.

### Caveats on this result, stated rather than buried

* gv3 is not a perfect one-variable change: moving to a global also changes the ACCESS PATH
  (`ldc gp[0]` + offset instead of `cincoffsetimm s0` + offset), and it is ~20% slower per
  iteration. Location and access path are coupled and cannot be separated by this build.
* An earlier attempt (stages 30/31, hand-written asm with globals) came back clean and proved
  nothing: it varied five things at once — hand asm vs codegen, global vs stack, half the cycle
  count, the stored capability's provenance, and the complete absence of both `ldc`s. Recorded so it
  is not mistaken for evidence.
* The compiler is STILL not excluded. Isolating it needs hand-written asm reproducing c8's exact
  sequence ON THE STACK, and that will not run: storing the stack capability into the stack appears
  illegal under this ABI (`stc` of a linear/uninit capability consumes it), so stage 30 produces no
  result under QEMU.
* Running the loop OUTSIDE a domain on the board is not reachable with the current runner, which
  drives `.dom` files through the monitor.

## UPDATE 2026-08-07 (boot 66): the STORE's provenance is irrelevant; only where the COUNTERS live matters

Stage 33 holds the counters exactly where they fail -- stack locals via s0, qc at row offset 12 --
and moves ONLY the capability store's target to a global, reached via gp.

| rung | counters | stc target | result |
|---|---|---|---|
| gz8 | **stack**, row offset 12 | **global** | **567 WRONG** |
| gz12 | **stack**, row offset 8 | **global** | **0x8000237 WRONG** |
| c8 | stack | stack | 567 WRONG (anchor) |
| *(boot 65)* gv3 | **global** | stack | **576 correct** |

**Moving the store target changes nothing. Moving the counters cures it.** The "capability store and
scalar accesses derive from the same capability" hypothesis is REFUTED. gz12 also reproduces
`0x8000237` exactly -- the same value shift12 gave -- with the store target in a different region
entirely, which further decouples the clobber value from the store.

**Established, controlled: the damaged scalar must be on the domain stack.**

### The LINEAR-stack hypothesis is refuted too, before it cost a boot

The glue's comment made it look like `sp` might be LINEAR, which would engage the LDC-of-linear
CLEAR store (`load_unit.sv:447-460`) that takes priority over a real store with no arbitration
(`store_unit.sv:399`, `:410-417`) -- a good fit for a dropped store. But `INTERP_SP_LINEAR` is
DERIVED: it is 1 only when `INTERP_FAKE_COUNT` is defined AND `INTERP_DIAG_STAGE < 2`, and our
builds pass neither. Verified in the artifact: `delin sp` is the FIRST instruction of the entry glue
(c8.dom:f0004), so **sp and everything split from it are NONLIN**. That path is not engaged.

### What actually differs between the failing and passing arms

Not the region per se, but how the address capability is OBTAINED:

* failing: counter address = `cincoffsetimm s0, imm` -- derived from a **register-resident** capability
* passing: counter address = `ldc gp[i]` then offset -- derived from a capability **loaded from memory**

That is the next thing to isolate: keep the counters on the stack but reach them through a
capability that was loaded from memory (park the frame capability in a slot, `ldc` it back, use that
for the RMW). If that is clean, the variable is the address capability's provenance, not the region.

## RETRACTION 2026-08-07: "the damaged scalar must be on the domain STACK" — boot 65 tested nothing

An adversarial audit refuted the boot-65 conclusion, and the refutation is confirmed here.

**Stage 32 had NO `FDREG_SHIFT` pad** — the only recent stage without one. It was therefore
structurally incapable of being placed at the failing geometry, and shipped at the SHIFT=0 layout:

    gv3.dom   stack rmw = 0x1c / 0x20 / 0x24
    c0.dom    stack rmw = 0x1c / 0x20 / 0x24    <- known UNDAMAGED
    rs0.dom   stack rmw = 0x1c / 0x20 / 0x24    <- known UNDAMAGED
    c8.dom    stack rmw = 0x14 / 0x18 / 0x1c    <- the failing geometry

`fit-victim-rules.py` already lists that layout among the undamaged builds (c0, rs0, bs16, nr16).
**gv3 was predicted correct by the existing dataset before any global was involved.** The "cure" was
the layout, not the storage class.

**This is the FIFTH confound of the same class** (after qc==k+8, k bits[3:2], "k is immune", and the
+0x1c anchor). And the trail's own method note — added hours earlier — says to run
`fit-victim-rules.py` before spending board time on geometry. It takes milliseconds and reads the
ELF. It was not run.

**A second error in the same build:** the note said "the three counters live in a GLOBAL". False.
`qc`, `p`, `k` never left the stack; only a FOURTH accumulator was global, and only that was
returned — so whether the stack `qc` was damaged in gv3 is UNOBSERVED. That is exactly the
single-number reporting failure diagnosed at boot 55.

### What still stands from boot 66

The gz arms (store target moved to a global, counters left on the stack at row offsets 12 and 8 →
567 and 0x8000237) do kill the "capability store and scalar accesses derive from the same
capability" hypothesis, and they show the damage needs neither the store and counters in one row nor
in one region. But they have **no matched benign-geometry control**, the frame shrank 0x50→0x40 so
every absolute address moved, and each arm is N=1. Downgrade from "irrelevant" to: *moving the store
target to a global does not cure it at these two geometries.*

### Stage 32 is fixed

It now carries the standard `FDREG_SHIFT` pad and returns BOTH accumulators packed
(`stack<<16 | global`, correct = 0x02400240), so the comparison is actually made. Built at SHIFT=8
the stack slots are 0x14/0x18/0x1c — identical to c8 — with the global accumulator alongside.

### Process items from the audit, worth keeping

* Artifacts behind load-bearing claims were rotating out of the overlay into an attic and out of
  `/tmp`. Copy them somewhere durable with hashes recorded next to the result line.
* `fit-victim-rules.py` hard-codes a session-scoped scratchpad path and will silently report
  `dataset: 0 builds` elsewhere. Fix before relying on it again.
