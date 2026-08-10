# R-20 compiler workaround — REVERTED 2026-08-10. Kept as the record of what was done.

> **STATUS: no longer applied.** `caplifive_r20.bit` carries the RTL fix, this package's own repro
> passes on it (`sbx8`: `0xD0000001` -> `0xD0000000`), and the SQLite-level site `Z.dom` returns
> where it used to wedge. Both conditions below were met, so commit `30c275b5d781` was reverted and
> `llvm/` is byte-identical to its pre-workaround state (`git diff 30c275b5d781^ -- llvm/` is
> empty). Nothing here needs doing again; it is kept so the reasoning, the measurements and the
> rejected alternatives are not lost.

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

**`30c275b5d781`** — "R-20 workaround: keep a0/x10 out of the capability store's base register"
(full sha `30c275b5d781fe14e479919c110671beac85669a`, on branch `capstone-bootstrap`).

```bash
git revert 30c275b5d781
```

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

## Every commit involved, and what to do with each

Audited with `git log 30c275b5d781^..HEAD -- llvm/`, which returns **exactly one commit**. Only
that one carries code; the rest are documentation and diagnostics that merely describe it.

| commit | contains | on revert |
|---|---|---|
| `30c275b5d781` | **THE CODE** — 4 files in `llvm/lib/Target/Capstone/` + 8 lit tests | **revert it** |
| `77c7eeef8cff` | this file's commit-hash pin + the standing TODO in `state/current-next-step.md` | delete the TODO; leave this file as history |
| `13c84d28bb8f` | ISSUES.md: S-03 closed, S-04 opened | keep — a real result, independent of the workaround |
| `8ed12710c631`, `29621b095bb3` | ISSUES.md: S-04 ruled-out list | keep |
| `87280248da7f` | measured MOVC residual + `sim/scan-fwd.py`, `sim/scan-r20-wide.py` | keep the scanners; they stay useful |

**Verify that claim yourself before reverting** — if anything else has since touched compiler
code, this table is stale:

```bash
git log --oneline 30c275b5d781^..HEAD -- llvm/     # must list ONLY 30c275b5d781
```

If it lists more, revert those too, newest first.

## How to revert

```bash
git revert 30c275b5d781                  # or: git grep -n "R-20 WORKAROUND"  and remove each block
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

**This workaround is PARTIAL, and the residual is measured, not hypothetical.**

`issue_read_operands.sv:568` drops x10's clobber claim for ANY non-CAPENTER capability op with
`rs1 == x10` and `rd != x10` -- not just `stc`. The wrong VALUE is then supplied by the
rs1-cursor forwarding mux, gated by `check_fwd_rs1` (`ariane_pkg.sv:929-935`) =
**{SPLIT, MOVC, CJALR, CCSRRW, STC}**. `LDC` and `CINCOFFSET` are NOT in that set, so they can
lose the stall without corrupting a value; `MOVC` very much is.

Measured on the SQLite domain with `sim/scan-fwd.py`, counting only ops in that set, for the
shape `<op with rs1 == x10, rd != x10> -> an instruction writing x10 -> a reader of x10`:

| adjacency window | before the workaround | with the workaround |
|---|---|---|
| 1 (immediately adjacent) | 4481  (MOVC 2914, STC 1567) | **1051 — all MOVC** |
| 4 | 6006  (MOVC 3778, STC 2228) | 2396 — all MOVC |

So every `stc` site is gone and **~1051 `MOVC` sites remain**.

**BUT THE MOVC RESIDUE IS DEMONSTRABLY HARMLESS.** Measured in RTL simulation with a matched
pair (`sim/r20-stc-ld-x10.S`, arms A and M): the identical structure -- a capability op with
`rs1 == a0` and `rd != a0`, then an instruction writing a0, then a reader of a0 -- **CORRUPTS
with `stc` and does NOT corrupt with `movc`**, on the UNPATCHED RTL where the `stc` arms fail.
The positive control fires in the same run, so this is a real negative, not a dead test.

| arm | unpatched RTL | patched |
|---|---|---|
| A `stc` rs1=a0, adjacent | **DEFECT** | correct |
| M `movc` rs1=a0, same structure | **correct** | correct |

Likely reason, offered as a hypothesis rather than a finding: `stc` executes in the
dynamic-latency unit (`capstone_dyn_unit`) and stays in flight for many cycles -- it blocks on a
revocation-node query -- while `movc` is a 1-cycle fixed-latency op (`capstone_flu_unit`) that
retires before a dependent reader can issue alongside it. If that is right, only the LONG-LATENCY
members of `check_fwd_rs1` matter, which are `stc` and `cjalr`; the scan finds **no** `cjalr`,
`split` or `ccsrrw` sites in the SQLite domain at all.

So the `stc`-only workaround may well be COMPLETE in practice. Do not read the raw MOVC count as
outstanding exposure.

**Consequence:** treat this workaround as "removes the `stc` shape", not "removes R-20". If a new
silicon symptom appears with it in place, run `sim/scan-fwd.py` before assuming the workaround is
at fault -- an unexplained NOMEM from `sqlite3_open` (issue S-04) is currently a live suspect for
exactly this residue. `sim/scan-r20-wide.py` reports the broader `check_cap_op` shape as well,
which is the upper bound rather than the value-corrupting subset.
