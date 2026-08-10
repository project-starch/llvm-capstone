# R-20 compiler workaround — TEMPORARY. Revert this when the RTL fix ships.

This file exists so the workaround can be removed cleanly, by someone who was not here when it
was added. It is the whole record: what was changed, why, how to check it is still needed, and
exactly how to take it out.

## What the workaround is

The register allocator is prevented from ever choosing **a0/x10** as the **base register of a
capability store**. R-20 needs `stc <val>, imm(a0)` for its trigger; if the base can never be
x10, the pattern is unconstructible and the defect cannot fire through this path.

It does **not** fix R-20. The silicon defect is still there. This only stops our compiler from
emitting the shape that trips it.

## Why not something simpler

**Nop padding was measured and rejected.** It is the obvious workaround and it is unsafe:

| separation | board | RTL simulation |
|---|---|---|
| 1 nop between `stc` and `ld` | cured | **still defective** |
| 2 nops | not tested | **still defective** |
| 4 nops | not tested | cured |
| 1 nop between `ld` and consumer | cured | **still defective** |

The board cured it with one nop; simulation needed four. The window is context-dependent, so any
fixed nop count works in one setting and silently fails in another. That is worse than no
workaround, because it looks like it worked.

Changing the register is the only cure that holds on **both** board and simulation and is not a
timing window (board arm `R13`, sim arm B — both clean).

## The commit to revert

Single commit. Four files in `llvm/lib/Target/Capstone/`, plus eight lit tests:

| file | change |
|---|---|
| `CapstoneRegisterInfo.td` | adds `def GPRNoX10 : GPRRegisterClass<(sub GPR, X10)>` |
| `CapstoneInstrInfo.td` | adds `def GPRMemNoX10 : MemOperand<GPRNoX10>` |
| `CapstoneInstrInfo.td` | `STC`'s base operand: `GPRMem` -> `GPRMemNoX10` |
| `CapstoneRegisterInfo.cpp` | `eliminateFrameIndex` takes STC's base scratch from `GPRNoX10RegClass` |
| `Disassembler/CapstoneDisassembler.cpp` | adds `DecodeGPRNoX10RegisterClass` |
| `llvm/test/CodeGen/Capstone/*.ll` | 15 CHECK lines relaxed; 4 tests marked `XFAIL` |

**The `.td` change alone is NOT sufficient**, and this was measured, not assumed. Frame-index
elimination runs AFTER register allocation, so the operand's register class never reaches it: a
large stack offset was materialised into a fresh GPR the scavenger could make a0, rebuilding the
exact shape (`lui a0; cincoffset a0, s0, a0; stc ..., off(a0)`). With only the `.td` change, 25
vulnerable sites survived in SQLite. The `CapstoneRegisterInfo.cpp` hunk closes that path.

Every added block is tagged `R-20 WORKAROUND` in a comment, so `git grep -n "R-20 WORKAROUND"`
finds all of them.

**The decoder deliberately does NOT reject x10.** The class exists to steer the allocator; an
`stc` with base a0 is a legal encoding the hardware executes, and every already-built image is
full of them. An earlier draft failed the decode, which made `llvm-objdump` print those as
`unknown` and silently broke a static scan of existing binaries. If you touch that function,
keep it permissive.

## How to revert

```bash
git revert <this commit>            # or: git grep -n "R-20 WORKAROUND"  and remove each block
cd llvm/cmake-build-debug && ninja -j90 llc clang lld     # never -j112
```

Then rebuild anything compiled with it -- at minimum
`capstone/benchmarks/sqlite/build-sqlite-silicon.sh` -- and re-run the ladder and QEMU suites.

**The four XFAILed lit tests are self-cancelling.** `aggregate-copy.ll`,
`aggregate-memcpy-align.ll`, `globals.ll` and `mem-intrinsics.ll` have hand-written CHECK chains
pinned to the pre-workaround allocation. Once the workaround is reverted they will XPASS, which
lit reports as a failure -- that is the signal to delete the `XFAIL` block from each. The other
15 relaxed CHECK lines are allocation-agnostic and pass either way.

## How to know it is safe to revert

The workaround is only needed while a bitstream without the RTL fix is resident. The fix is one
character, on `capstone-ariane` branch **`r20-fix`** (`2efb3604f`, based on `e1b3db6ba`):
`issue_read_operands.sv:568` changes `=` to `|=`.

Revert this workaround once **both** hold:

1. a bitstream built from RTL containing that fix is flashed and confirmed resident, and
2. `./run.sh sim` in this folder passes on the RTL revision that bitstream was built from — arm A
   reads 0 rather than the store's base address.

Check 2 matters on its own: the RTL fix has been validated in simulation but has **never been in
a bitstream**, so nothing has yet confirmed it on silicon.

## Validation at the time it was applied

* lit `llvm/test/CodeGen/Capstone/`: **43 passed, 4 expectedly failed** (the XFAILs above).
* Silicon ladder under QEMU, `DOMAIN_OPT_LEVEL=-O1`: `matmult_int`, `beebs_prime`, `beebs_bs`,
  `beebs_cover`, `ctrsanity`, `beebs_aha_mont64` -- **6/6 PASS**.
* CoreMark and 81 of 82 BEEBS runners fail, and were verified to fail **identically on an
  unmodified baseline** (stash, rebuild, re-run, restore). They are pre-existing QEMU
  capability-emulation assertions (`helper_csshrink`, `helper_cssplit`), unrelated to this
  change. Runtime coverage of the workaround therefore rests on the six ladder rungs, not on the
  broader corpus.

## Measured effect

Raw-encoding scan of the SQLite silicon domain (opcode `0x5B`, funct3 `4`, base = bits 19:15;
counted by decoding the bytes, not by reading `llvm-objdump` text, because the count has to be
able to be zero):

| build | `stc` with base `a0` | vulnerable shape |
|---|---|---|
| before | 3657 | **2751** |
| `.td` change only | 33 | **25** |
| `.td` + frame-index hunk | **0** | **0** |

Scan: `sim/scan-r20.py`, shipped with this package. It looks for `stc <v>, off(a0)` -> an
instruction that WRITES a0 -> a reader of a0, and cross-checks its `stc`-with-base-a0 count
against a raw-encoding decode, so a disassembler problem cannot make it silently report zero.

**A CORRECTION, recorded because the wrong version was briefly written down.** An earlier draft
of this file said the 33 residual sites were hand-written entry glue and that 0 were vulnerable.
Both were wrong: they were the frame-index materialisation above, and 25 of them WERE vulnerable.
The first scan only checked whether the instruction immediately after the `stc` read a0, so it
missed the common `stc a1,off(a0); ldc a0,0(a1); <reader>` form -- and it undercounted the
unpatched build as 1998 rather than 2751. If you re-measure, use the scan named above.

## What this does NOT cover

R-20's underlying condition is broader than `stc`. `check_cap_op`
(`ariane_pkg.sv:902-912`) also includes `LDC`, `MOVC`, `CINCOFFSET` and others, so in principle
any capability op with `rs1 == x10` and `rd != x10`, followed by a write to x10 and then a read
of x10, could lose the same clobber claim. Only the `stc` shape has been observed failing, on
the board and in simulation, and only that shape is blocked here. If a new wedge appears with
this workaround in place, widen the scan before assuming the workaround is at fault:

```
any capstone op with rs1 == x10 and rd != x10
  → an instruction writing x10
  → a reader of x10
```
