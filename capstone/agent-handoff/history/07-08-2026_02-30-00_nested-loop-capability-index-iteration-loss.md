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
