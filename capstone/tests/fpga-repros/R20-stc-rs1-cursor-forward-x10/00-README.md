# R-20 — after `stc`, a load into **x10** is read by the next instruction as the store's base address

**Wrong symptom? Read this paragraph first.** This package is the **stale-operand-on-x10**
signature: a plain `ld` whose value the *next* instruction reads as the *store's base address*
instead of the loaded data, only when the register is **x10/a0**. Three sibling packages describe
different signatures and are not this issue: `../R18-scalar-store-metadata-clobber/` is a scalar in
the upper half of a 16-byte row being silently **zeroed**; `../R19-movc-zero-metadata-in-slot/` is a
slot coming back holding **`compress_cap(NULL) + n`**; `../R01-lsu-hazard/` is a load through one
capability register missing a store through another. R-20 corrupts **no memory at all** — the
memory is correct and a later reader of the same register sees the correct value. Only the
immediately following instruction is wrong.

## The defect in one paragraph

On this silicon, `stc rX, 0(a0)` immediately followed by `ld a0, 0(a0)` immediately followed by any
consumer of `a0` gives that consumer **the address the store used (`a0` before the load)** rather
than the value the load fetched. The memory is correct; a consumer one instruction later is
correct; the same sequence on any other register is correct. It is silent — no trap, no tag
violation, nothing in any log — and the same byte-identical binary is correct under QEMU. Because
the stale value and the correct value are usually both non-zero, it is normally invisible: it
changes behaviour only where the loaded value is genuinely **zero**, which is exactly the
`if (pointer)` idiom after a function returns NULL.

## It reproduces in RTL SIMULATION — ~14 s, no board, no SQLite

`./run.sh sim`. `sim/r20-stc-ld-x10.S` runs the shape five times and the RVFI trace prints the
value every instruction wrote, so the LOAD's own retired value and its CONSUMER's operand are
both visible side by side — which no board instrument can give.

**Genuine completion in 558 cycles, no exceptions.** Every load retires with the CORRECT value:

```
ld a0, 0(a0)  -> x10 = 0x0000000000000000      <- all five loads return 0, correctly
```

and then the very next instruction reads something else:

| arm | consumer reads | |
|---|---|---|
| A  `a0`, adjacent | `0x80003000` | **DEFECT — the store's base address** |
| B  `a3`, adjacent | `0` | correct — **CONTROL: only the register differs** |
| C  `a0`, 1 nop | `0x80003000` | defect |
| D  `a0`, 2 nops | `0x80003000` | defect |
| E  `a0`, 4 nops | `0` | correct — **CONTROL** |

This is the load-bearing observation: **memory is right, the load is right, and only the
consumer's operand is wrong.** It places the fault in operand delivery, not in the store, the
cache, or the load.

Arms B and E are the instrument validation — the same probe on cases that must read 0, and both
do. `.option norvc` is set so the encodings are 4-byte, as on the board.

**The simulated tree and the bitstream's revision are identical in the relevant logic.**
`git diff e1b3db6ba HEAD -- core/issue_read_operands.sv core/include/ariane_pkg.sv
core/scoreboard.sv core/decoder.sv` is EMPTY; the only `.sv` differences between them are
`wt_dcache_mem.sv` and `store_unit.sv`, which are the R-18 fix and sit in the store *data* path.

**One quantitative difference from the board, stated plainly:** in simulation the window closes
between 2 and 4 nops, on the board it closed at 1 nop (`gap`). The qualitative result — defect on
x10, clean on another register, cured by enough separation — matches. The likely causes are the
known fidelity gaps: bare M-mode versus a real capability domain, a register-resident capability
versus one reached through the cap table, and cache warmth. Do not read the sim threshold as the
silicon threshold.

## Reproducer — 13 KB, one boot, no SQLite

`src/sbx_kernel.h` runs the same seven-instruction shape seven times and returns a bitmask, so one
run reports every arm and nothing can hang. Frozen images: `src/sbx8.dom`, `src/sbx20.dom`,
`src/sbx36.dom` (three draws of identical code at different link offsets; see "Draws", below).

**Measured on `caplifive_65536_r18_fix.bit`, three draws, identical: `retval = 0xD0000001`.**

| bit | arm | result | what it establishes |
|---|---|---|---|
| 0 | `stc t0,0(a0); ld a0,0(a0); beqz a0` | **SET** | **the defect** — the branch sees non-zero where the slot holds 0 |
| 1 | same, **1 nop** before the branch | clear | one instruction of separation cures it |
| 2 | same, 2 nops | clear | stays cured |
| 3 | same, 4 nops | clear | stays cured |
| 4 | same shape on **t1** | clear | **instrument valid**, and the defect is x10-specific |
| 5 | `sd` instead of `stc` | clear | a **capability** store is required; a scalar store does not trigger it |
| 6 | no adjacent store at all | clear | **instrument valid**, and the store is required |

Bits 4 and 6 are the instrument validation: they are the same probe on a case that must come back
clean, and they do. The nop arms are a measurement, not a control.

## Necessary conditions, each established by a one-variable pair

| condition | pair | evidence |
|---|---|---|
| the register is **x10/a0** | `R13` vs base — the whole triple rewritten on `a3`, nothing else changed | RETURN vs WEDGE |
| the store is a **capability** store | sbx bit 5 vs bit 0 — `sd` vs `stc` | clear vs set |
| the **load** is adjacent to the store | `adj` vs base — one nop between `stc` and `ld` | RETURN vs WEDGE |
| the **consumer** is adjacent to the load | `gap` vs base — one nop between `ld` and the branch | RETURN vs WEDGE |
| the branch **target** is irrelevant | `Z` vs base — **one byte**, the branch offset only | WEDGE vs WEDGE |

## The value is measured, not inferred

Arm `V1` computes `(a0 - s0) + 0x50` and branches on zero, so it returns **iff** the value read for
`a0` was exactly `s0-0x50` — the address the `stc` used. It RETURNED. `V0` is the same chain with
the read moved one slot later and the opposite branch polarity; it also RETURNED, so the chain can
produce both answers.

## Where it bites in real code

`sqlite3InsertBuiltinFuncs`, at `-O0`, compiles `pOther = functionSearch(...); if (pOther)` to:

```
13cb64: stc  a1, 0x0(a0)      ; spill the returned pointer (NULL) to a stack slot
13cb68: ld   a0, 0x0(a0)      ; read it back as an integer -> 0
13cb6c: beqz a0, 0x13cbc8     ; NOT TAKEN on silicon
13cb7c: ldc  a3, 0x20(a0)     ; so the if-branch runs and dereferences NULL -> core wedges
```

`functionSearch` provably returns NULL here: the build clamps registration so only
`sqlite3AlterFunctions` runs, with `nDef = 1`, into a zeroed table, so no duplicate name can exist.
There are **736** instances of this exact `stc/ld/consumer` shape on x10 in that one image; the
other 735 are invisible because the stale and correct values are both non-zero there.

## Suggested places to look in the RTL

Read at `capstone-ariane` `e1b3db6ba`. The simulation establishes the SYMPTOM in the pipeline —
the consumer receives the store's base address while the load retires correctly — but it does not
by itself isolate WHICH of the two sites below produces it. See "What this package does NOT
establish".

* `core/issue_read_operands.sv:566-567` marks an in-flight capability op's `rs1` as clobbered, so a
  reader of that register stalls. `check_cap_op` (`core/include/ariane_pkg.sv:902-912`) includes
  `STC`.
* `core/issue_read_operands.sv:568` then **unconditionally overwrites that entry for x10 alone**:
  `gpr_clobber_vld[5'd10][i] = fwd_i.sbe[i].op == ariane_pkg::CAPENTER && fwd_i.still_issued[i];`
  — an `=` where the surrounding intent appears to need `|=`. This is the only register-specific
  line in the block, and x10 is the only register that fails on the board.
* `core/issue_read_operands.sv:674-677`, with `check_fwd_rs1`
  (`core/include/ariane_pkg.sv:929-935`, which also includes `STC`), serves a reader whose `rs1`
  matches an in-flight STC's `rs1` with that STC's **`rs1_cursor`** — which is the store's base
  address, i.e. exactly the value `V1` measured.

Line 568 dates from `4891d379a` (2026-04-24) and is unrelated to the R-18 fix.

## THE FIX — one character, validated in simulation

`capstone-ariane` branch **`r20-fix`** (`2efb3604f`, based on `e1b3db6ba`) changes
`core/issue_read_operands.sv:568` from `=` to `|=`:

```systemverilog
-  gpr_clobber_vld[5'd10][i]  = fwd_i.sbe[i].op == ariane_pkg::CAPENTER && fwd_i.still_issued[i];
+  gpr_clobber_vld[5'd10][i] |= fwd_i.sbe[i].op == ariane_pkg::CAPENTER && fwd_i.still_issued[i];
```

| | A `a0` adj | B `a3` (ctl) | C 1 nop | D 2 nops | E 4 nops (ctl) |
|---|---|---|---|---|---|
| `e1b3db6ba` unpatched | **WRONG** | ok | **WRONG** | **WRONG** | ok |
| with the fix | ok | ok | ok | ok | ok |

Same 558 cycles, so it costs no time. `capenter`, `stc`, `capldc`, `cap-overwrite` and
`cincoffset` all pass on the patched tree. (`break-load-store-forwarding` fails to compile on
the patched AND unpatched trees alike — pre-existing.)

The blast radius is provable from the two statements: for `op == CAPENTER` the line above yields
false because of its own `op != CAPENTER` term, so `[10][i]` is 0 and `0 |= x == x` — CAPENTER is
bit-identical. For `op != CAPENTER` this line's RHS is false, so `=` forced 0 while `|=` leaves
the rs1 claim standing, which differs only when `rs1 == x10`. The change can only ever ADD a
clobber bit, which makes a reader stall — the conservative direction.

## WORKAROUNDS while the RTL is unfixed — and one that does NOT work

**Do not use nop padding.** It is the obvious workaround and it is unsafe. The board cured the
defect with ONE nop; simulation needs FOUR, measured on both sides of the load:

| separation | board | simulation |
|---|---|---|
| 1 nop between `stc` and `ld` | cured (`adj`) | **still defective** |
| 2 nops between `stc` and `ld` | not tested | **still defective** |
| 4 nops between `stc` and `ld` | not tested | cured |
| 1 nop between `ld` and consumer | cured (`gap`) | **still defective** |
| 4 nops between `ld` and consumer | not tested | cured |

The window is context-dependent, so any fixed nop count is a workaround that appears to work in
one setting and silently fails in another. That is worse than no workaround.

**The only cure that holds on both board and simulation, and is not a timing window, is to keep
x10 out of the capability store's base register** — board arm `R13` and sim arm B, both clean.
Two routes, neither yet implemented:

1. **Restrict the STC address operand's register class.** `CapstoneInstrInfo.td:2402-2403` has
   `(ins GPR:$rs2, GPRMem:$rs1, ...)` where `GPRMem = MemOperand<GPR>` is shared by every load
   and store, so this needs a new register class excluding X10 plus a new `MemOperand` used only
   by `STC`. Deterministic, but it is a codegen change and needs a full LLVM rebuild plus lit and
   the QEMU suites before it can be trusted.
2. **Build at `-O1`, which removes the spill pattern entirely.** `build-sqlite-silicon.sh`
   already defaults the string primitives to `-O1` for an unrelated real defect, and its own
   comment records that at `-O1` "the pointer stays in a register and the loop contains no
   ldc/stc at all". It is currently blocked by a separate compiler bug: `cond ? capA : capB`
   reaches ISel as an i128 `CapstoneISD::SELECT_CC`, and `Select_GPRCAP_Using_CC_GPR` is emitted
   only under `!is64Bit()`, so capstone64 aborts with "Cannot select"
   (`CapstoneInstrInfo.td:1741-1747`). Fixing that unblocks this route and removes the pattern
   wholesale rather than register by register.

Scale of the exposure, for whoever picks this up: the current SQLite silicon image contains
**3657** capability stores based on `a0`, **2186** of which are immediately followed by a reader
of `a0` — the vulnerable shape. Almost all are invisible because the stale value and the correct
value are both non-zero there.

## What this package does NOT establish

* **Which of the two RTL sites to change.** The simulation shows the consumer receiving the
  store's base address while the load retires correctly, which is consistent with the
  `check_fwd_rs1(STC)` cursor path being the value source and line 568 being the enabler — but
  nothing here isolates one from the other. That needs a patched-RTL run: change one site, re-run
  `./run.sh sim`, and require arm A to have been WRONG before the patch or the test proved
  nothing.
* **The bitstream-to-revision mapping is an inference.** The runner enforces the resident
  bitstream's *name* (`caplifive_65536_r18_fix.bit`); nothing on our side records which RTL
  revision built it. `e1b3db6ba` is our best match by name and date.
* **Board wedge verdicts are one-per-boot.** The runner stops at the first non-return, so a wedging
  arm is always the last arm of its boot. `Z` was reproduced at two different boot positions (2 and
  4) and `gap` at two (3 and 4) specifically to rule out a position artifact, and `R13`/`V1` were
  re-run in a boot that also contained an in-domain wedge (`Z`) as a positive control.

## Draws

The three `sbx*.dom` images are identical code at different link offsets (`SBX_PAD` emits
never-executed nops ahead of it). This is not redundancy for its own sake: this platform has a
per-image entry stall (`../R16-entry-stall/`) in which a domain never runs at all, and retrying the
same binary is futile. `sbx_compute` is byte-identical across all three; only the whole-image hash
differs. If an image produces no `SQ: G/enter`, use the next draw.

## Files

```
src/sbx_kernel.h        the probe: seven arms, returns a bitmask, cannot hang
src/sbx_fpga_app.c      rung wrapper
src/sbx_host.c          native oracle: correct hardware returns 0xD0000000
src/sbx{8,20,36}.dom    frozen images, three draws
sim/r20-stc-ld-x10.S    the directed RTL test -- five arms, two of them controls
sim/run-sim.sh          builds, runs and parses it; refuses a timeout or an exception
sim/r20-rvfi-trace.log  the RVFI trace from the run quoted above
board/make-arms.py      regenerates the SQLite arms as byte patches of one base image
board/ARM-SHA256SUMS    sha256 of the base image and of every arm actually run
SHA256SUMS              covers everything committed here
run.sh                  see below
```

The SQLite arms are **not** committed: each is 1.5 MB and differs from the base by 1-12 bytes.
`board/make-arms.py` regenerates all eight, and the regenerated files match the binaries that were
run byte-for-byte (verify with `board/ARM-SHA256SUMS`).

## Running it

```bash
./run.sh sim        # RTL simulation, ~14 s                                (no board)  <- start here
./run.sh verify     # check every frozen artifact against SHA256SUMS       (no board)
./run.sh arms <base sqlite_silicon.dom>   # regenerate + verify the SQLite arms (no board)
./run.sh rung       # the 13 KB reproducer on the board: expect retval 0xD0000001
```

`run.sh rung` runs a known-good control first; a boot whose control fails carries no verdict.
