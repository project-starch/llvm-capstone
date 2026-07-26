# gp-captable silicon miscompute: the documented root cause is REFUTED

**Date:** 2026-07-25 · **Lane:** B · **Board-free** (static analysis + QEMU) · Deadline
gate #1 of `plans/b-silicon-track-handoff.md`.

## Bottom line

The open bug note
(`history/23-07-2026_17-30-00_gp-captable-silicon-array-loop-miscompute-OPEN.md`) records
the root cause as an **RTL `shrink`→store forwarding hazard**, with "build with shrink off"
as a proven workaround (~90% confidence, "escalation to the board owner is justified").

**That explanation cannot be right for the 4 failing ladder rungs, and the workaround does
not generalize.** The rungs are built shrink-off *and contain zero `shrink` instructions*,
yet 4 of 6 still miscompute on silicon. **Do not send the board owner the shrink story.**

Four further hypotheses were tested and also refuted (below). The real mechanism is **not
yet identified** — this note is deliberately a set of negative results plus the instrument
built to settle it.

## Evidence

### 1. `shrink` is absent from every rung (so it cannot be the cause)

`build-ladder-domain.sh` passes `-capstone-shrink-stack=false -capstone-shrink-globals=false`,
and disassembly of the exact `.dom` files that ran on the board confirms it:

| rung | silicon | `shrink` | `ldc (gp)` | `stc` | `scc` |
|---|---|---:|---:|---:|---:|
| rv8_primes | PASS | **0** | 2 | 14 | 2 |
| beebs_prime | PASS | **0** | 6 | 27 | 2 |
| matmult_int | FAIL | **0** | 6 | 20 | 2 |
| beebs_crc32 | FAIL | **0** | 3 | 22 | 2 |
| beebs_insertsort | FAIL | **0** | 5 | 19 | 2 |
| beebs_recursion | FAIL | **0** | 2 | 27 | 2 |

The 23-07 result ("`rc_p1` + shrink-off PASSES on silicon") is not disputed — it just does
not explain these four, which fail *with no shrink at all*. Either there are two distinct
bugs, or the shrink localization was a coincidence of that probe set.

### 2. "Array store with a live accumulator" is the wrong characterization

`beebs_recursion` **has no array**. Its two globals are a `volatile int` and an `int`
(4 bytes each). Its distinguishing feature is deep self-recursion (`fib(10)`) plus mutual
recursion. The headline framing in `current-state.md` and the 25-07 sweep note is therefore
too strong; it fits 3 of the 4 failures, not 4.

### 3. Capability-bounds representability is NOT the discriminator

Hypothesis: real HW compresses bounds, QEMU models them precisely, so large/oddly-aligned
objects round wrong. Per-rung global sizes from the `.capstone_gp_table` descriptor:

| rung | verdict | globals |
|---|---|---|
| rv8_primes | **PASS** | one **12,504 B** array — the largest in the whole set |
| beebs_prime | PASS | 8 B, 8 B, 4 B |
| matmult_int | FAIL | 256 B × 3 |
| beebs_crc32 | FAIL | 2048 B, 8 B |
| beebs_insertsort | FAIL | 44 B, 44 B (one **initialized**) |
| beebs_recursion | FAIL | 4 B, 4 B |

The rung with by far the largest object passes; the rung with the smallest objects fails.
Refuted.

### 4. No instruction-level discriminator

Mnemonic sets: **no mnemonic occurs in all 4 FAIL rungs and in no PASS rung.** The PASS
rungs use a strict *superset* (18 mnemonics appear only in PASS rungs). So the fault is not
keyed to a distinctive instruction.

### 5. The corruption is BROAD, not one bad value (checksum inversion)

Each rung folds its results with FNV-1a. Each byte step is invertible, so if all but one
folded word were correct we could **solve exactly** for the bad word. Done in
`/tmp/capstone-b/fnv_solve.py` (meet-in-the-middle; model validated — it reproduces
`beebs_recursion`'s published oracle `1579141629` exactly).

- `beebs_recursion` (only **3** folded words): the single-word solutions are
  `0xB88E7A1C`, `0x46C26864`, `0xBFDAF978` for `In`, `0x3CA1E33D` for `kalle(10)`, and
  **none** for `anka(10)`. All are implausible — not small integers, not the `0x8B5x_xxxx`
  data-region addresses the earlier `rc_*` probes showed. **No plausible single-word
  corruption reproduces the silicon value.**
- `matmult_int` (64 folded words): the fold is **not injective**, so a wide fold admits many
  preimages and the inversion is swamped by collisions — no usable signal.

Conclusion: silicon is not getting one value wrong; it is getting many wrong (or the fold
loop itself runs wrong). That is why a checksum-only harness cannot localize this.

### 6. Ruled out by inspection

The entry glue is correct: `BUILD_GP_CAPTABLE` carves storage downward from `sp.END` and
**ends with `scc(sp, sp, t1)`**, which re-seats the stack cursor at the top of the remaining
region before `cincoffsetimm(sp, sp, -96)`. Earlier concern that the post-`split` cursor
(QEMU's `helper_cssplit` resets `rs1.cursor = base`) would leave the stack out of bounds is
unfounded.

## What was built to settle it: the `gp_diag` rung

A checksum destroys the information needed to localize the fault, so a **diagnostic rung**
now returns **raw per-probe values in separate result slots**. One board run (one
power-cycle) discriminates all the live hypotheses at once.

- `silicon-ladder/gp_diag_kernel.h` — 9 independent probes, each isolating one mechanism.
- `silicon-ladder/gp_diag_fpga_app.c` — writes `res[3+p]` = raw value of probe *p*.
- `silicon-ladder/gp_diag_app.c`, `gp_diag_host.c` — QEMU + native-oracle variants.
- `rtl-smoke/ladder_perf_ctl.c` — prints `DEBUG <name> dbg0=… dbg8=…` when any slot is
  non-zero (perf rungs leave them zero, so this is free for them).
- `fpga_driver/run_ladder_perf_fpga.py` — parses and reports the `DEBUG` line.

| slot | probe | expected |
|---|---|---|
| dbg0 | scalar global write→read via `ldc gp[i]` | 0x5A5A |
| dbg1 | global array store→readback | 1234 |
| dbg2 | global read from inside a **noinline callee** | 0x5A5A |
| dbg3 | deep self-recursion `fib(10)` (no globals) | 89 |
| dbg4 | mutual recursion `anka(10)` (no globals) | 1 |
| dbg5 | array store in a loop with a **live accumulator** | 28 |
| dbg6 | **initialized** global (init-template materialization) | 36 |
| dbg7 | function-local `static`, read-before-write (zero-init) | 3 |
| dbg8 | canary constant (plumbing sanity) | 0xC0FFEE |

**Validated on QEMU:** `run-ladder-qemu.sh gp_diag` → PASS, retval `875368783` (= oracle).
Artifacts built into `$CAPSTONE_TMP_ROOT/ladder-fpga/`. Fits the constraints: `.text` =
0x9b8 (2488 B) inside the 4 KiB PCC window; LOAD span 0x10f0 > 0x1000 so the monitor SPLIT
is not degenerate.

**To run it (one power-cycle):**
`LADDER_RUNGS="gp_diag" python3 capstone/tests/rtl-smoke/fpga_driver/run_ladder_perf_fpga.py`

## Bonus finding: `switch` jump tables break gp-captable domains

Building `gp_diag` exposed a real, general landmine. A dense `switch` lowers to a table of
**code addresses in `.rodata`** plus an indirect `jr`. In a gp-captable domain `.rodata` is
not reachable as plain data (globals come from the gp cap-table), so the table load faults:

```
[CAPSTONE] Cap mem access requires capability: pc = 101560490, rs1 = x10, imm = 0
[CAPSTONE] domain halted by capability fault: cause = 24, pc = 0x101560490, ...
```

— i.e. **it looks exactly like a domain crash, with a wild PC**. Confirmed by disassembly
(`.rodata` held 8 code addresses `0x104ac…`, reached via `jr a0`).

**Fixed** by adding `-fno-jump-tables` to `SILICON_FLAGS` in `build-ladder-domain.sh`.
Regression-checked: `beebs_insertsort` still passes QEMU at `271779359` with the flag.

This matters well beyond the diagnostic rung — **SQLite is switch-heavy** and would have
hit this and looked like a random crash.

It looked like a candidate for `coremark_matrix`'s unexplained `-Os` hang (`-Os` forms jump
tables more readily than `-O0`), but **checked and refuted**: the pre-flag
`coremark_matrix.dom` has **no `.rodata` section and zero indirect jumps**, so
`-fno-jump-tables` changes nothing for it. Its hang remains unexplained, and its real
blocker is still *transfer* (the dom wedges the console shell at every tier), not codegen.

## Status / next

- The published root cause is **withdrawn** for these 4 rungs; the mechanism is **unknown**.
- **Do not** escalate the shrink story to the board owner.
- Next decisive step is a **single board run of `gp_diag`** — everything for it is built and
  QEMU-validated. Its outcome picks the next move: if dbg3/dbg4 fail, it is recursion/stack;
  if dbg6 fails, initialized-global materialization; if dbg0–dbg2 fail, the basic gp path;
  if *everything* is wrong, suspect the domain-entry/data-region setup rather than codegen.
- Still blocked behind this: the silicon perf table, any silicon-compatibility claim, and
  the `capstone-gp-free` merge.
