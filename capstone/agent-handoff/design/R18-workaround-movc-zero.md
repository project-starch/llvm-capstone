# The R-18 workaround: `movc rd, zero` → `addi rd, x0, 0`

**Flag:** `-capstone-int-zero-for-zero-copy` · **Default: OFF** ·
**Code:** `llvm/lib/Target/Capstone/CapstoneInstrInfo.cpp`, `copyPhysReg`

This document exists so the workaround can be **removed** without re-deriving why it was added.
It is a mitigation for a hardware defect. When the hardware is fixed, delete it — §"How to cancel".

---

## What it does, in one line

When the compiler copies **from the zero register**, emit an integer move instead of a capability
move.

```
  before:   movc  a0, zero        # capability move; writes 0x08000000 into a0's metadata shadow
  after:    addi  a0, x0, 0       # integer move;    leaves a0's metadata shadow at 0
```

Both put the value **0** in `a0`. They differ only in what they leave in the register's *capability
metadata shadow*, which is invisible to the program and visible to the cache.

## Why that matters — the defect it dodges

Four RTL facts, each verified in `capstone-ariane`:

1. **Every GPR write updates the metadata shadow.** The cap-metadata regfile is written under the
   *integer* write-enable (`issue_read_operands.sv:1663-1665`, `.we_i(we_pack)`), so an ordinary
   integer op writes a **zero** shadow (a non-FLU writeback carries `cap_result = '0'`,
   `scoreboard.sv:246`) while a capability op writes a real one.
2. **`movc rd, zero` produces a non-zero shadow.** `compress_cap` of a null capability is
   **`0x08000000`** (`ariane_pkg.sv:754-772`, the cursorless branch).
3. **The shadow rides along on every store, ungated by opcode.**
   `issue_read_operands.sv:1140` → `load_store_unit.sv:1013` → `store_unit.sv:345` →
   `store_buffer.sv:173` → the dcache write-user sideband.
4. **The dcache classifies a store as a capability store BY VALUE.**
   `wt_dcache_mem.sv:138`: `st_wr_cap = |wr_user_i`. A so-classified store asserts **both banks**
   (`:230-238`) and the **same byte enable is applied to both** (`:152-158`).

Net effect: an ordinary `sw` out of a `movc`-written register **also writes its data into the same
byte lanes of the other bank of the 16-byte row**, silently overwriting an unrelated scalar. No
trap, no tag violation, nothing in any log. QEMU is correct throughout, so it only appears on
hardware.

A copy from `x0` is materialising an **integer zero**, not moving a capability — `x0` is hardwired
and can never carry a tag — so the integer form is the semantically honest encoding *and* it dodges
the defect.

## Evidence it works

| level | evidence |
|---|---|
| RTL simulation | `scalar-store-addi-zero.S` **passes**; `scalar-store-movc-zero.S`, byte-identical except for this one instruction, **fails** with a witness zeroed |
| Silicon | `c8` (movc) qc=**567**, `c8fix` (addi) qc=**576** — same source, same frame geometry (frame 80, rmw `[20,24,28]`), one instruction apart, control green, comparable cycles |
| Non-regression | flag OFF byte-identical on 4/4 rungs; QEMU ladder 6/6 both ways; lit 47/47 |

## What it does NOT do — read this before relying on it

**It removes the common case, not the class.** The trigger is *any* non-zero capability metadata on
a store's data register. This flag only handles the `x0` source. Measured with
`capstone/tests/runtime-qemu/silicon-ladder/scan-r18-trigger.py`:

| binary | trigger sites, flag off | flag on |
|---|---|---|
| `c8` (the reproducer) | 7 | **2** |
| **SQLite** (`sqlite_silicon.dom`) | **5494** | **~3081** |

For SQLite the breakdown by what wrote the store's data register is:
`movc`-from-zero **2413 (44%)** — removed; `movc` register-to-register **1567 (28%)**; `ldc`
**1259 (23%)**; `cincoffsetimm` **222**; `cincoffset` **31**; `lcc` **2** — all still exposed.

So the flag roughly halves SQLite's exposure and leaves ~3000 sites. **It is not a fix for SQLite.**

## Two blockers before it could ever be default-ON

1. **It is keyed on `SrcReg == X0`, not on "this is an integer zero".** Genuine null-*capability*
   materialisations also lower to `movc rd, zero` and are rewritten too —
   `llvm/test/CodeGen/Capstone/select-cap.ll` (`select_cap_null`) and `calling-conv.ll`
   (`test_call_vararg`). The argument that this is harmless (X0 is hardwired, so both are untagged
   zero) is **plausible but unverified**: `0x08000000` is a canonical null encoding and an integer
   op leaves the shadow at 0, and nobody has checked what a later `stc` of such a register writes to
   memory in each case. **Settle that on the ISA semantics before promoting the flag.**
2. Those two lit tests FileCheck the literal `movc` mnemonic and would flip to FAIL. They need
   flag-on RUN lines or updated CHECKs.

## How to cancel it

The workaround is **obsolete the moment the hardware classifies stores by opcode** rather than by
`|wr_user_i` (`wt_dcache_mem.sv:138`), or gates the metadata onto the sideband by opcode at issue
(`issue_read_operands.sv:1140`). When a bitstream carrying either fix is resident:

1. **Confirm the fix is really in the resident bitstream**, the same way R-16 acceptance works —
   run `capstone/tests/fpga-repros/R18-scalar-store-metadata-clobber/run.sh trigger`. With the
   hardware fixed, `c8` must return **576**, not 567. If `c8` still returns 567 the fix is not in
   that bitstream and the workaround must stay.

   **BOTH signatures must clear, not just this one.** This flag is the workaround for **R-19** as
   well (`capstone/tests/fpga-repros/R19-movc-zero-metadata-in-slot/`), whose signature is the
   store's own slot returning `compress_cap(NULL) + n` rather than being zeroed. A bitstream could
   fix the zeroing form and leave the metadata-in-slot form live, and this gate as originally
   written would not have noticed — it would have deleted the workaround with R-19 still biting.
   So the acceptance set is `c8` → **576** *and* `fdp0` → **2609** (not `0x08000A31`). Both arms
   are staged; run them in one boot.
2. Delete the `cl::opt` `CapstoneIntZeroForZeroCopy` and the `SrcReg == Capstone::X0` block in
   `copyPhysReg`. Both are contiguous and commented with `R-18`.
3. Rebuild and confirm every measured rung is byte-identical to its flag-OFF build — since the flag
   defaults to OFF, removing it should change **nothing**. That is the whole regression check.
4. Delete this document and the `sim/scalar-store-*.S` tests, or move them to
   `fpga-repros/ARCHIVED/`.

**Do not cancel it because the defect stopped reproducing in software.** The defect is layout
sensitive: it disappears whenever the `-O0` allocator happens to move a scalar out of the splashed
slot. Only a bitstream change justifies removal, and step 1 is how you check.

## Related

- Issue and full evidence: `capstone/agent-handoff/ref/ISSUES.md`, entry **R-18**
- Handover package incl. simulation tests: `capstone/tests/fpga-repros/R18-scalar-store-metadata-clobber/`
- Trail with every retraction: `capstone/agent-handoff/history/07-08-2026_23-55-00_r18-localized-to-row-mate-traffic.md`
- Exposure scanner: `capstone/tests/runtime-qemu/silicon-ladder/scan-r18-trigger.py`
