# R-19 — a `movc rd, zero`-sourced store leaves `compress_cap(NULL)` in its OWN bank-1 slot

**Status: this signature is established on silicon and reproducible on demand. It does NOT reproduce
in Verilator — unlike R-18's splash form, which does, and which ships here too so the contrast is in
one place. A compiler-side workaround is silicon-confirmed.**

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

They share a trigger class (a store whose data register carries **capability metadata** — simulation
shows a real, valid capability triggers R-18's form too, so it is not specific to the null form)
and the **same workaround clears both** (see `../R18-scalar-store-metadata-clobber/` for that
issue's own package). Whether they are one defect with two manifestations or two
defects is **unknown**, and this package does not assert either. They are tracked apart because the
R-18 report already sent describes the zeroing form, and folding this into it would misinform the
reader.

## What is measured

`k800` control green in every boot. `fdp0` reproduced in **three consecutive boots**, one image linked
at `0x30000`; a sibling build `fdpraw` linked at `0x60000` shows the same value, so it is not tied to
one load address. (A further sibling `fdp1`, also at `0x60000`, shows the same value but is **not
shipped here**, so treat it as uncheckable from this folder.)

**Confirmed again 2026-08-08 at two FRESH LINK ADDRESSES.** A 7-arm single boot reproduced the
signature away from the addresses in the table above: `fdd` (damaged, `0xb0000`) returned
`134220337` = `0x08000A31`, and `fdw` (workaround, `0x90000`) returned the correct `2609`. Control
`k800` green, all seven arms entered and returned. So neither the defect nor the cure is tied to a
particular load address.

**Bitstream, stated precisely because an earlier draft of this paragraph got it wrong.** All board
measurements in this package — the four arms above and this confirmation — were taken with
`caplifive_65536_nodes.bit` resident. A draft claimed this run was a *second* bitstream; it was not,
and that claim is **withdrawn**. Whether the signature also appears on `caplifive_fixed_forward.bit`
is **untested**.

| image | build | returned |
|---|---|---|
| `fdp0.dom` | accumulator initialised by `movc a0, zero; sw`, `-O0` | **`0x08000A31`** = `0x08000000` + 2609 |
| `fdp0fix.dom` | identical but initialised by `addi a0, x0, 0` | **2609** — clean |
| `fdpraw.dom` | returns the accumulator alone, no second term | `0x08000A31` — **the victim is that slot** |
| `fdpO1.dom` | `-O1`, accumulator kept in a **register** | **2609** — clean |

### Why the value is `0x08000000 + n` and not a bare `0x08000000`

Worth stating because it looks like an objection to the obvious mechanism and turns out to be the
opposite. The candidate mechanism (`wt_dcache_mem.sv:156-158`) is a pure **select** —
`(((st_wr_cap)&&(k==1)) ? wr_user_i : wr_data_i)` — so it deposits a *constant*, and a constant
cannot explain a value that tracks the loop count.

It does not have to. The trigger fires **once**, at the accumulator's initialisation
(`movc a0, zero; sw`), and everything after it is ordinary integer `lw`/`addiw`/`sw`. So the slot
starts at `0x08000000` instead of `0`, and the program then accumulates on top of it:

    corrupted init   0x08000000  +  2609 accumulated  =  0x08000A31  = 134220337   <- observed
    clean init                0  +  2609 accumulated  =        2609               <- fdp0fix

Both arms are exact, with no free parameter. A pure select is therefore fully consistent with the
observation, and the arithmetic is a check the hypothesis could have failed and did not.

**What this does NOT establish** is that the mux is where the corruption comes from. That still rests
on the value's provenance — `0x08000000` is a hardware encoding and the immediate appears nowhere in
the image — and the path has been *read*, never traced. Simulation does not reproduce this signature
(see above), so treat the mechanism as consistent-and-unrefuted, not confirmed.

`0x08000000` is the compressed encoding of a null capability — the cursorless-bounds branch at
`ariane_pkg.sv:754-772`, reached from `compress_cap` at `:813`. It is a hardware encoding; the
immediate `0x8000` appears nowhere in the image. The slot is initialised with an integer `0` and then
accumulated into, so the expected final value is 2609.

QEMU computes 2609 for the **same source at the same flags** — not a byte-identical comparison: the
`_fpga` image writes 24 B into our QEMU harness's 8-byte shared region and faults, so the QEMU run
uses the `_app` variant of the same build.

* **Trigger:** a store whose data register carries capability metadata.
* **An observation, NOT an established condition:** the one `-O1` build is clean, but it is not a
  controlled variation — see the rebuild note. We are not claiming storage class.
* The row offset is **derived, not measured**: the accumulator is at `s0-0x28`, which is row offset 8
  only if `s0` is 16-byte aligned on the monitor-carved domain stack. That is unmeasured.
* `fdpraw` matters because the original return was `s + fdreg_gate - 1`, and `fdreg_gate == 0x08000001`
  fitted the same number. Returning `s` alone rules the global out.

## What is NOT established

**This signature, and which slot the board picks.** All four simulation tests ship here so the
contrast is in one place:

| test | verdict | what it shows |
|---|---|---|
| `sim/scalar-store-movc-zero.S` | **FAILS** (tohost 3, 12822 cyc) | a neighbour 8 bytes away is zeroed |
| `sim/scalar-store-realcap-samegeom.S` | **FAILS** (tohost 3, 12807 cyc) | a *real, valid* capability triggers it too |
| `sim/scalar-store-addi-zero.S` | passes (12816 cyc) | matched control, one instruction different |
| `sim/movc-zero-self-clobber.S` | **passes** (1715 cyc) | **inconclusive — see the warning below** |

**The negative result is now CONTROLLED — the test was run at a matched trigger count and the check
is demonstrably able to fire.** This replaces an earlier note here which said the pass discriminated
nothing; that was true of the default build alone and is no longer the state of the evidence.

`movc-zero-self-clobber.S` fires its trigger **once** by default, where the reproducing R-18 test
fires it **64 times**. Rebuilding it with `-DTRIG_IN_LOOP=1` raises the count to 64 (its `EXPECT`
oracle moves with the knob: with the trigger inside the loop the accumulator is reset each
iteration, so the clean value is 1, not `NITER`). Both configurations were run in Verilator:

| configuration | triggers | verdict | |
|---|---|---|---|
| default | 1 | **SUCCESS**, 1715 cyc | no observable effect |
| `-DTRIG_IN_LOOP=1` | 64 | **FAILED** (tohost 3), 1974 cyc | **the splash** — witness A `0x0a0a0a0a` → `0`, witness B → `1`, row-mate reset |

The define was verified to have reached the build rather than assumed: the loop back-edge moves onto
the `MOVC` address, the `EXPECT` arm switches from 64 to 1, `MOVC` retirements go 1 → 64 and
accumulator stores 65 → 128 in the RVFI trace.

**What this establishes.** Trigger count is the differentiator: a single-shot trigger produces
nothing, so the default build's pass carries no information on its own. At 64 triggers the check
**does** fire — which is the positive control this test previously lacked. And with it firing, the
store's **own slot still read back the clean value `0x1`, with no `0x08000000` component**. So the
metadata-in-slot signature does **not** reproduce in simulation even at a matched trigger count,
while the R-18 splash does.

That is a real negative result rather than an untested one. It does not explain the board, and the
mechanism at `wt_dcache_mem.sv:156-158` remains a candidate that simulation has not confirmed.

The readable chain is `issue_read_operands.sv:1140` → `load_store_unit.sv:1013` →
`store_unit.sv:345` → `store_buffer.sv:173` (onto the dcache write-user sideband, ungated by opcode)
→ `wt_dcache_mem.sv:138` (`st_wr_cap = |wr_user_i`, classified by value) → `:234-237` (both banks
asserted) → `:158` (bank 1 takes `wr_user_i`). **Confirmed for the splash; not for this signature.**

Untested candidates for the divergence: the resident bitstream may not match this RTL revision; the
board runs inside a capability domain after `capenter` on a monitor-carved stack while the directed
test is bare metal; or the test lacks a co-factor — it has no capability traffic in the loop, no
indirect calls and no cap-init.

The passing test is shipped **deliberately**. Six earlier clean directed tests were once read as
"the hardware is innocent" when they simply never created the condition.

## Corrections to the R-18 report already sent

Both were found after that report went out, and both are about the *mechanism*, not the reproducer:

1. **`R XOR 8` is WITHDRAWN as a board rule.** It holds in ten builds whose victim lies 8 bytes from
   the trigger and fails in six (`rs4`, `ka0`, `gnt`, `gz0`, `gzn`, `graw`) whose victim lies 4 bytes
   away. Distance is invariant under base alignment, so it is not an alignment artefact. `rs4.dom`
   and `ka0.dom` are **not** shipped in either package, so that corpus is not checkable from here.
   (An earlier version of this line called the rule "sharpened, not withdrawn"; the R-18 package and
   the issue registry both say withdrawn, and they are right.)
2. **The trigger class is wider than the null form** — `scalar-store-realcap-samegeom` shows a real
   capability triggers the splash in simulation.

Neither affects the reproducer, the trigger or the workaround.

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
    # the victim-identity discriminator -- NOTE the different base VA
    RUNG=fdpraw DOMAIN_BASE_VA=0x60000  ... -DFDREG_STAGE=4 -DFDREG_RAWSUM=1  (host AND domain)
    # the -O1 arm -- NOTE the different base VA, and NO -DFDREG_PAD
    RUNG=fdpO1  DOMAIN_BASE_VA=0xf0000 DOMAIN_OPT_LEVEL=-O1  ... -DFDREG_STAGE=4

`fdpO1` is NOT a controlled variation of `fdp0`. It has a different link address, different `-D`
flags, and `fdreg_compute` is inlined away. (**Correction, 2026-08-08:** this paragraph previously
claimed `fdpO1` "has ZERO trigger sites in the whole image". That was **false** — it has **seven**
`movc rd, zero` sites, the same count as the damaged arm `fdp0`. The image with zero is `fdp0fix`,
the workaround build. The paragraph's conclusion is unaffected; one of its four reasons was wrong.) Treat its
clean result as an observation, not as evidence that storage class is the immunity condition.

`DOMAIN_OPT_LEVEL` is load-bearing and was the source of an apparent contradiction: a 2026-08-06
board result recorded stage 4 as returning 2609, which was an `-O1` build whose accumulator lived in
a register. Record the opt level with any stage-4 result.

## Workaround

Shared with R-18: `capstone/agent-handoff/design/R18-workaround-movc-zero.md` —
`-capstone-int-zero-for-zero-copy`, default OFF, emits an integer move for a copy from `x0` so no
null-capability shadow reaches the store. Silicon-confirmed here by `fdp0` vs `fdp0fix`.
