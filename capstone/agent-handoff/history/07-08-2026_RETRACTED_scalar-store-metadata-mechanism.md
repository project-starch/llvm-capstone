# RETRACTED — a mechanism for R-18 that did not survive

**This document is kept only as the trail of a refuted hypothesis. It is NOT a defect report
and must NOT be linked to anyone outside the project.** The reproducer that IS suitable for
handover is `capstone/tests/fpga-repros/R18-scalar-store-metadata-clobber/`.

Refuted by an adversarial audit and then independently by directed simulation
(`scalar-store-cap-operand.S`, condition proven created via CAPPRINT, result PASS). See R-18
in `ISSUES.md`.

---

# Silicon defect report: a plain scalar store in the upper half of a cache row is overwritten with capability metadata

**Issue:** R-18. **Status: DO NOT SEND. The causal chain below was RETRACTED on 2026-08-07 by an
adversarial audit, hours after it was written and before handover.** The *observations* (§3) stand;
the *mechanism* (§2) and the *fix* (§5) do not. See the retraction box immediately below, and R-18
in `ISSUES.md`. **Date:** 2026-08-07.

---

> ## RETRACTED — read this before anything else
>
> * §2 link 1 is misread. The scoreboard-port "validity gate" (`issue_read_operands.sv:765`) has
>   `cap_result.result_metadata` in **both** arms of its ternary; it does not sanitise to zero. The
>   asymmetry the chain depends on does not exist, and **the §5 fix would have been a no-op.**
> * No source of stale metadata on an ordinary store has ever been demonstrated.
>   `ex_stage.sv:1081` zeroes the FLU writeback when the op is not a capstone op, so an ordinary
>   `addi` forwards zero. `wr_user_i != 0` on a scalar store has never been measured.
> * Our own `movc` barrier refutes it: `compress_cap(null) = 0x08000000`, not zero, so `bar1`
>   should have pinned the accumulator near zero and instead matched its `nop` control exactly.
> * `c8` fails while `gp16`/`gp32`/`t16` succeed at the **same bank and same byte lanes**, so the
>   geometry cannot be the cause — it is a necessary condition.
> * The §3 decomposition is an arithmetic identity (two free parameters per observation), not a fit.
>
> **What survives and is worth reporting on its own:** `st_wr_cap = |wr_user_i`
> (`wt_dcache_mem.sv:138`) classifies capability stores **by value rather than opcode**, and the
> compressed encoding of a **null** capability is `0x08000000`, not zero — so any store carrying
> null-cap metadata is misclassified and dual-bank written. That is a clean, quotable structural
> defect. It is *not* shown to be the cause of the 567.

**Affected RTL:** `capstone-ariane`. Verified present at HEAD `458982093` and at `7aac52f93`, the
commit the resident bitstream `caplifive_65536_nodes.bit` is built from. `git diff 7aac52f93..HEAD`
does not touch any of the files below, so the defect is identical on both.

---

## 1. Summary

A plain scalar store (`sw`) whose address lies in the **upper 8 bytes of a 16-byte D-cache row** can
have **its own slot overwritten with capability metadata instead of the value being stored**. Where
those metadata bytes happen to be zero at the store's byte lanes, the effect is that the variable is
**silently set to zero** — no trap, no tag violation, nothing in any log.

Consequence for software: at `-O0`, any loop that mixes capability traffic with ordinary scalar
locals can have a counter or accumulator reset mid-loop. We hit it as wrong benchmark results; a
loop-control variable landing in the affected position instead causes extra iterations. It is not
specific to our reproducer — it is a property of where the compiler happens to place a scalar.

---

## 2. The mechanism — four links, each quoted from the source

**(1) Stale capability metadata can ride on an ordinary store's operand.**
`core/issue_read_operands.sv:690` forwards `rs2`'s capability metadata from the **writeback port**
with no validity gate:

```systemverilog
assign rs2_cap_metadata[i][k] = ((issue_instr_i[i].rs2 == fwd_i.sbe[fwd_i.wb[k].trans_id].rd)
    ? fwd_i.wb[k].cap_data.result_metadata : ...);
```

Compare the **scoreboard-port** forward roughly 25 lines later, which *does* check validity:

```systemverilog
... && (fwd_i.sbe[k].rs1 == issue_instr_i[i].rs2) && fwd_i.sbe[k].cap_result.valid)) & ...
```

The WB-port path has no equivalent `cap_result.valid` term.

**(2) A store is classified as a capability store by VALUE, not by opcode.**
`core/cache_subsystem/wt_dcache_mem.sv:138`:

```systemverilog
assign st_wr_cap = |wr_user_i;
```

So a non-zero metadata sideband is *sufficient* to classify a plain `sw` as a capability store.

**(3) A classified store writes BOTH banks of the row.**
`wt_dcache_mem.sv:230-238` — the normal path selects one bank from the address; the `st_wr_cap` path
does not:

```systemverilog
if (!(st_wr_cap)) begin
  bank_req |= dcache_cl_bin2oh(wr_off_i[...]);
  bank_we   = dcache_cl_bin2oh(wr_off_i[...]);
end else begin
  bank_req = '1;
  bank_we  = '1;
end
```

**(4) Bank 1 is the only bank that can receive something other than the store's data.**
`wt_dcache_mem.sv:156-158`:

```systemverilog
assign bank_wdata[k][j] = (wr_cl_we_i[j] & wr_cl_vld_i) ? wr_cl_data_i[...]
                        : (((st_wr_cap) && (k==1)) ? wr_user_i : wr_data_i);
```

`bank_be` applies the **same byte-enable to both banks**, so for a store whose address is in bank 1
the metadata lands on the store's *own* byte lanes and the real data never lands.

---

## 3. What we measured on silicon

21 domain builds, layouts read out of the ELF (not from notes). Nine builds had the victim measured
directly by a probe that returns all loop variables packed in one word.

**The invariant, 9 of 9, no exceptions: the victim is always in the upper 8 bytes of its 16-byte
row** — row offset 8 or 12, never 0 or 4. Undamaged builds also carry upper-half scalars, so this is
a genuine constraint, not an artifact of where the allocator puts things.

**The victim is overwritten, not skipped.** A run with the accumulator initialised to a sentinel of
1,000,000 instead of 0 returned **567**, not 1000567. The slot was written and counted up from
there.

**Every measurement decomposes as `clobber_value + (576 − reset_iteration)`:**

| build | returned | clobber value | reset iteration |
|---|---|---|---|
| shift8 / gp0 / c8 / dp0 / sn8 | 567 | 0 | 9 |
| rs4 | 504 | 0 | 72 |
| ka0 (`d`) | 18 | 0 | 558 |
| **shift12** | **0x08000237** | **0x08000000** | 9 |

`shift12` is the clearest single datum: `0x08000237 = 0x08000000 + 567`. The slot was clobbered with
a value carrying **bit 27** — metadata-shaped, not a plausible integer — and then incremented 567
times. The same build family with zero metadata bytes returns exactly 567.

**A loop-control variable in the affected slot causes extra iterations instead.** Zeroing the outer
counter restarts the loop: we see +9, +330 and +333 iterations, and the **cycle counts confirm the
extra iterations really executed** (69081 cycles against 44001 for the correct run, ≈904 implied
iterations against a returned 909).

**QEMU computes the correct answer for every variant** — it has no metadata sideband to misclassify.

---

## 4. Reproduce it

Environment:

```bash
source capstone/tests/capstone-test-env.sh
```

Build the two arms (identical apart from where the allocator puts the accumulator):

```bash
# ARM A -- accumulator in the UPPER half of its row. Silicon returns 567; correct is 576.
DOMAIN_GLUE=interp DOMAIN_OPT_LEVEL=-O0 DOMAIN_BASE_VA=0x30000 \
DOMAIN_EXTRA_CFLAGS="-DFDREG_STAGE=19 -DFDREG_LEAVES=0 -DFDREG_GUARD=0 -DFDREG_SHIFT=8" \
bash capstone/tests/runtime-qemu/silicon-ladder/build-ladder-domain.sh \
     capstone/tests/runtime-qemu/silicon-ladder/fdreg_fpga_app.c c8.dom

# ARM B (control) -- same source, accumulator elsewhere. Silicon returns the correct value.
DOMAIN_GLUE=interp DOMAIN_OPT_LEVEL=-O0 DOMAIN_BASE_VA=0x60000 \
DOMAIN_EXTRA_CFLAGS="-DFDREG_STAGE=19 -DFDREG_LEAVES=0 -DFDREG_GUARD=0 -DFDREG_SHIFT=0" \
bash capstone/tests/runtime-qemu/silicon-ladder/build-ladder-domain.sh \
     capstone/tests/runtime-qemu/silicon-ladder/fdreg_fpga_app.c c0.dom
```

Stage 19 returns `p<<20 | k<<16 | qc`, so one number names which variable was damaged. Correct is
`0x04090240` (p=64, k=9, qc=576).

The **sentinel** variant is the one that proves overwrite-vs-skip. `FDREG_STAGE=27` starts the
accumulator at `FDREG_SENTINEL` (default 1,000,000); correct is 1000576, and silicon returns 567:

```bash
DOMAIN_EXTRA_CFLAGS="-DFDREG_STAGE=27 -DFDREG_LEAVES=0 -DFDREG_GUARD=0 -DFDREG_SHIFT=8"
```

Confirm the layout in the artifact before trusting any run — the effect depends entirely on where
the accumulator lands:

```bash
python3 capstone/tests/runtime-qemu/silicon-ladder/extract-frame-layout.py c8.dom c0.dom
```

which prints each build's frame size, the capability store's offset, and every scalar
read-modify-write slot. In the failing arm the accumulator sits at row offset 12 (upper half); in
the control it sits at row offset 4 (lower half) and is never damaged:

```
c8.dom: frame=0x50 store=0x00 rmw=['0x14','0x18','0x1c'] upper-half=['0x18','0x1c']
c0.dom: frame=0x50 store=0x00 rmw=['0x1c','0x20','0x24'] upper-half=['0x1c']
```

Everything under discussion is in
`capstone/tests/runtime-qemu/silicon-ladder/fdreg_kernel.h` (stages 19, 25, 26, 27) and the full
measurement trail is in
`capstone/agent-handoff/history/07-08-2026_02-30-00_nested-loop-capability-index-iteration-loss.md`.

---

## 5. Suggested fix

Either link breaks the chain:

* **Gate the WB-port forward on validity** in `core/issue_read_operands.sv:690`, matching the
  scoreboard-port version. The `rs1` and `rs3` siblings immediately above and below have the same
  shape and should be reviewed together.
* **Classify capability stores by opcode** rather than by `|wr_user_i` in
  `core/cache_subsystem/wt_dcache_mem.sv:138`.

The second is the more conservative of the two — classifying a store by the value that happens to be
on a sideband is fragile regardless of where the stale value came from.

We have not implemented either: both need a bitstream reflash, which is your call.

---

## 6. Why this was hard to find, and what would make it easier

Recorded because it bears on how to instrument this core, not as narrative:

* **The damage is invisible in final values.** Both loops in the reproducer terminate on their own
  conditions, so a counter zeroed mid-loop still exits at its normal value. Only the **cycle count**
  revealed extra iterations, and it is printed on every result line — it had gone unused for the
  whole investigation.
* **Any added local moves the frame slots** and can move the victim out of the affected position, so
  most in-frame instruments cure the effect they are trying to measure.
* **A wrong value looks plausible.** Zeroing produces a believable count, which is why the mechanism
  was twice rejected on the grounds that "the victim isn't garbage".

---

## 7. Open, and stated plainly

* **The defect does not reproduce in Verilator.** Directed tests
  (`verif/tests/custom/capstone/stc-neighbour-load.S`, `stc-counter-pair.S`) pass at both RTL
  revisions, cycle-for-cycle identical, across five rounds of added fidelity — a pair of scalars 8
  bytes apart, a nested loop with a resetting index, the faithful `-O0` instruction sequence, and a
  cap-table load. They are bare-metal M-mode; the failing code runs inside a capability domain after
  `capenter` on a monitor-carved stack. We could not construct a directed test that produces stale
  WB-forwarded metadata on a scalar store's `rs2`, so the simulation result neither confirms nor
  refutes the chain — **it means the trigger condition was never created**, and that is the single
  biggest gap in this report.
* **The rate varies by ~60×** — one clobber in 576 iterations in most builds, 558 in one. We have no
  account of why adding a fourth scalar to the loop changes the rate that much, and it is the
  strongest argument that more than one thing may be going on.
* **What would settle it:** observing `st_wr_cap` and `bank_we` at the cycle the victim's dword
  retires. If `st_wr_cap` is ever high for a retirement whose address is not a capability's cursor,
  that is direct confirmation.
