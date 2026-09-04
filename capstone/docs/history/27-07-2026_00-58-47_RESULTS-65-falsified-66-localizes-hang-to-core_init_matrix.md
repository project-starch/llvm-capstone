# Board results #65/#66: the fragile-exit hypothesis is FALSIFIED; coremark's hang is inside `core_init_matrix`

**Date:** 2026-07-27
**Lane:** B
**Cost:** one board session, two rungs, two attempts each. Board powered off + unlocked.
**Supersedes the forward-looking half of:**
`27-07-2026_00-28-51_loop-exit-condition-splits-hang-from-miscompute.md`

---

## Results

| probe | build | verified before the run | board |
|---|---|---|---|
| **#65** `matmult_int` ordered exits | −O1 `-DLADDER_ORDERED_EXITS` | 0 fragile / 8 ordered branches (2 `bge`, 6 `blt`); QEMU parity leg returned the oracle `774662735` | **HANGS** — no END marker in 120 s, 2 attempts |
| **#66** `coremark_matrix` init-only | −O0 @32 KiB `-DLADDER_CM_INIT_ONLY` | `mulw` loop present; QEMU parity leg returned `9` | **HANGS** — no END marker in 120 s, 2 attempts |

Both binaries are functionally correct under QEMU **through the identical board controller**
(`run-ladder-perf-qemu.sh`), so neither result is a harness artifact.

---

## #65 — the fragile-exit hypothesis is DEAD

**Predicted:** forcing ordered loop exits at −O1 would stop the hang and return a *wrong* value,
unifying the hang and the −O0 miscompute into one fault whose symptom is selected by the emitted
branch.

**Observed:** it still hangs, identically.

So the `bne`-overshoot mechanism is **refuted**. The codegen split reported yesterday is real and
total — `matmult_int` at −O1 emits 8 conditional branches, all `bne`; the same source at −O0 emits
8, all `blt` — but **that split is not what makes −O1 hang and −O0 merely miscompute.** It is a
correlate, not a cause.

**Retract accordingly.** Do not carry "the miscompute and the hang are ONE fault with two symptoms,
selected by the branch kind" forward without new evidence. It was the strongest available
hypothesis for about two hours and it is now dead. This is the second hypothesis in two days to die
this way (the first was "domain-entry fault"), and the pattern is the same: a strong correlation
observed statically, promoted to a mechanism before a board test could speak.

What still stands from that note: the M-extension disassembler trap and its correction; that the
"no discriminating instruction" conclusion survives a proper re-run; and the mode-7 localization
of the hang to inside the compute.

---

## #66 — a real localization for `coremark_matrix`

The probe returns `N` straight out of `core_init_matrix` and skips the entire benchmark. It hangs.

Bisect against the mode-7 result (task #63), **same rung, same −O0 @32 KiB configuration**:

| build | what runs | board |
|---|---|---|
| mode 7 | entry path only; whole compute branched over | **RETURNS** |
| **#66** | entry path **+ `core_init_matrix`**, nothing else | **HANGS** |
| normal | everything | **HANGS** |

⇒ **The fault is inside `core_init_matrix`.** That narrows it from the whole benchmark
(`matrix_add_const`, `mul_const`, `mul_vect`, `mul_matrix`, `mul_matrix_bitextract`, `matrix_sum`,
the crc16 chain) to a single ~40-line function.

**Do not over-read it to "the `while` loop".** `core_init_matrix` contains two candidates:

1. the dimension loop `while (j < blksize) { i++; j = i*i*2*4; }` — `bgeu` at `0x10428` driven by
   `mulw` at `0x10444`. An ordered exit that never fires if the multiply never reaches 666.
2. the N×N seeding double loop, whose bound is that same `N`, and whose body runs
   `seed = ((order * seed) % 65536)` — i.e. a multiply **and** a remainder, per element, writing
   `A[]` and `B[]` through the gp-delivered block capability.

Either would hang, and the pre-registered "HANGS ⇒ it is the `while` loop" reading was too narrow.
Separating them is one more cheap probe: return `N` *before* the seeding loop.

---

## Where this leaves the blocker

Established, cumulative:
- The hang is **inside the compute**, not at domain entry (#63) — stale-icache-at-entry dead, and
  the domain-boundary `fence.i` (#61) remains the wrong layer.
- For `coremark_matrix`, it is inside **`core_init_matrix`** (#66).
- It is **not** the loop-exit condition (#65).
- It is **not** discriminated by instruction mix (incl. M-extension ops, re-checked properly),
  code size, global count, or `.bss` size.

Still unknown: the mechanism, and whether the two hanging rungs share one.

Next, cheapest first:
1. Split `core_init_matrix` — return `N` before the seeding loop. One boot, separates the two
   candidates above.
2. `matmult_int` has no equivalent single suspect left. The remaining lever there is a phase
   bisect: return after the init loops, then after the `mm_cell` loops, then after the FNV fold.
   Two to three boots, and it needs no mechanism guess.
3. Keep the paper on 3 measured rungs with the §5 caveats. With the deadline at end-July this
   blocker must not hold the eval section hostage; a documented hardware limitation is an
   acceptable outcome, an unexplained one is not.
