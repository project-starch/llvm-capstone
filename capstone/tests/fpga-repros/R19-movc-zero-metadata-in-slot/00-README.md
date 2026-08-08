# R-19 — a `movc rd, zero`-sourced store leaves `compress_cap(NULL)` in its OWN bank-1 slot

**Status: trigger established on silicon and reproducible on demand. MECHANISM NOT CONFIRMED — a
directed Verilator test at the same geometry PASSES. NOT yet reported to the board owner.**

## Why this is not R-18

R-18 has already been reported. It is the **zeroing** form: the victim is written with `0` and
counts up from there, and raw full-width readbacks show **no metadata anywhere** (`craw` reads
`0x00000237`, `graw` and `gztr` likewise clean).

R-19 is a **different observable**: the victim comes back holding `compress_cap(NULL) + n`.

|  | R-18 (reported) | R-19 (this) |
|---|---|---|
| arms | `c8`, `rs8`, `dp0`, `sn8` | `fdp0`, `fdpraw`, `fdpO1`, `fdp0fix` |
| victim ends up | written with `0`, counts up | written with `0x08000000`, counts up |
| example value | 567 of an expected 576 | `0x08000A31` = `0x08000000` + 2609 |
| metadata in the slot | **no** — raw readback is clean | **yes** |

They share a trigger class (a store whose data register carries a null-capability metadata shadow)
and the **same workaround clears both**. Whether they are one defect with two manifestations or two
defects is **unknown**, and this package does not assert either. They are tracked apart because the
R-18 report already sent describes the zeroing form, and folding this into it would misinform the
reader.

## What is measured

`k800` control green in every boot; the damaged arm reproduced on **three** boots at two entry VAs.

| image | build | returned |
|---|---|---|
| `fdp0.dom` | accumulator initialised by `movc a0, zero; sw`, `-O0` | **`0x08000A31`** = `0x08000000` + 2609 |
| `fdp0fix.dom` | identical but initialised by `addi a0, x0, 0` | **2609** — clean |
| `fdpraw.dom` | returns the accumulator alone, no second term | `0x08000A31` — **the victim is that slot** |
| `fdpO1.dom` | `-O1`, accumulator kept in a **register** | **2609** — clean |

`0x08000000` is `compress_cap` of a null capability (`ariane_pkg.sv:754-772`) — a hardware encoding
the program has no way to materialise; it only ever writes `0` to that slot. QEMU computes 2609 for
the same binary.

* **Trigger:** a store whose data register carries a null-capability metadata shadow.
* **Immunity condition:** the accumulator's **storage class**. Register-resident (`-O1`) is clean;
  memory-resident at row offset 8 (bank 1, `-O0`) is damaged.
* `fdpraw` matters because the original return was `s + fdreg_gate - 1`, and `fdreg_gate == 0x08000001`
  fitted the same number. Returning `s` alone rules the global out.

## What is NOT established

**The path through the cache.** `sim/movc-zero-self-clobber.S` builds the same geometry bare-metal —
bank-1 slot at row offset 8, `movc`-zero initialiser, 64 increments, an RMW row-mate keeping the row
active, witnesses either side, raw readback — and returns **SUCCESS in 1715 cycles**. The simulated
RTL does not write metadata into the slot.

A chain is readable in the source and fits every board observation:
`issue_read_operands.sv:1140` puts the metadata on the store's write-user sideband ungated by
opcode → `wt_dcache_mem.sv:138` classifies by value (`st_wr_cap = |wr_user_i`) → `:158` gives bank 1
`wr_user_i` instead of `wr_data_i`. **It is not reproduced, so it is not claimed.**

Untested candidates for the divergence: the resident bitstream may not match this RTL revision; the
board runs inside a capability domain after `capenter` on a monitor-carved stack while the directed
test is bare metal; or the test lacks a co-factor — it has no capability traffic in the loop, no
indirect calls and no cap-init.

The passing test is shipped **deliberately**. Six earlier clean directed tests were once read as
"the hardware is innocent" when they simply never created the condition.

## Corrections to the R-18 report already sent

Both were found after that report went out, and both are about the *mechanism*, not the reproducer:

1. **The `R XOR 8` splash rule is withdrawn.** It is arithmetically "the victim is 8 bytes from the
   trigger". The corpus splits into distance-8 builds where it holds (10) and distance-4 builds
   where it fails (`rs4`, `ka0`, `gnt`, `gz0`, `gzn`, `graw`); distance is invariant under base
   alignment, so no alignment argument rescues it.
2. **The dual-bank chain is not confirmed**, per the Verilator result above.

The reproducer, the trigger and the workaround are unaffected by either.

## Rebuilding

    source capstone/tests/capstone-test-env.sh
    cd capstone/tests/runtime-qemu/silicon-ladder
    # the damaged arm
    RUNG=fdp0 DOMAIN_GLUE=interp DOMAIN_BASE_VA=0x30000 \
      HOST_EXTRA_CFLAGS="-DFDREG_STAGE=4 -DFDREG_PAD=0" \
      DOMAIN_EXTRA_CFLAGS="-DFDREG_STAGE=4 -DFDREG_PAD=0" \
      bash verify-and-stage-rung.sh fdreg
    # the one-instruction cure
    ... same, plus  -mllvm -capstone-int-zero-for-zero-copy  in DOMAIN_EXTRA_CFLAGS only
    # the victim-identity discriminator
    ... same, plus  -DFDREG_RAWSUM=1  in BOTH host and domain flags
    # the storage-class control
    ... same, plus  DOMAIN_OPT_LEVEL=-O1

`DOMAIN_OPT_LEVEL` is load-bearing and was the source of an apparent contradiction: a 2026-08-06
board result recorded stage 4 as returning 2609, which was an `-O1` build whose accumulator lived in
a register. Record the opt level with any stage-4 result.

## Workaround

Shared with R-18: `capstone/agent-handoff/design/R18-workaround-movc-zero.md` —
`-capstone-int-zero-for-zero-copy`, default OFF, emits an integer move for a copy from `x0` so no
null-capability shadow reaches the store. Silicon-confirmed here by `fdp0` vs `fdp0fix`.
