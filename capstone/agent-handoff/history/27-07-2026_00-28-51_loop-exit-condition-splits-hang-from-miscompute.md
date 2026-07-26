# The loop-exit condition splits the hang from the miscompute (static, no board time)

**Date:** 2026-07-27
**Lane:** B
**Cost:** zero board boots. Everything here is static analysis of already-built domain
images plus one new local build.

Follows `26-07-2026_23-56-07_the-hang-is-in-the-compute-not-at-domain-entry.md`, which
localized the `matmult_int` / `coremark_matrix` hangs to **inside the compute** and
retracted the "domain-entry fault" framing. That note ended with a proposed next step —
shrink the iteration count until the rung returns. This note supersedes that step with
two sharper experiments, and reports the static work that produced them.

---

## 1. A methodological defect in the earlier instruction sweep

The repeated claim "no instruction is present in every hanging build and absent from
every passing one" was produced with the **Capstone-triple** disassembler. That
disassembler renders every RISC-V **M-extension** instruction as `<unknown>`:

```
   103c0: 030787b3     <unknown>          # capstone64 triple
   103c0: 030787b3     mul  a5, a5, a6    # --triple=riscv64 --mattr=+m
```

The domains are built with `-Xclang -target-feature -Xclang +m` (CVA6 has M), so every
multiply and divide in the corpus was invisible to a mnemonic-keyed sweep.

**Re-ran the sweep with `--triple=riscv64 --mattr=+m`. The M-extension hypothesis is
REFUTED** — presence of multiply/divide does not discriminate:

| build | M-ops | outcome |
|---|---|---|
| `matmult_int` −O1 | 1 `mul`, 1 `mulw` | HANG |
| `coremark_matrix` −O0 | 22 `mulw` | HANG |
| `beebs_prime` −O0 | 1 `mul`, 1 `mulw`, 1 `remu` | **PASS** |
| `beebs_recursion` −O1 | 1 `mulw` | **PASS** |

**Scope of the defect, measured:** `<unknown>` is only **2%** of instructions, and the
fraction is identical (2%) across all four binaries. So the earlier sweep was ~98%
valid and its conclusion stands. The blind spot was real but small, and it is now
covered. Recording it so nobody re-derives the alarm.

The remaining non-M `<unknown>`s are Capstone capability opcodes (major opcode `0x5b`)
in the `_start` / `test` glue, which is byte-identical across rungs and therefore
cannot discriminate.

---

## 2. The result: loop-exit condition, not instruction mix

Classifying every conditional branch by **exit condition kind**:

| build | conditional branches | outcome |
|---|---|---|
| `matmult_int` **−O1** | **8 × `bne`** (all) | **HANG** |
| `matmult_int` **−O0** | **8 × `blt`** (all) | **MISCOMPUTE** |
| `coremark_matrix` −O0 | 17 `bgeu`, 1 `bge`, 1 `blt`, 1 `bne` | HANG |
| `beebs_prime` −O0 | 1 `bltu`, 2 `blt` | PASS |

The `matmult_int` pair is **the cleanest control in the corpus**: same source file, same
rung, same build script, one opt level apart — and the branch kind flips 8-for-8 exactly
where the symptom flips.

**Mechanism.** `bne` exits on *exact equality*: a loop-control value that is perturbed
can step past the bound and the test never fires again — an infinite loop. `blt` exits
on *ordering* and cannot be overshot: the same perturbation yields a wrong answer and
the loop still terminates.

⇒ **The known miscompute and the hang are plausibly ONE fault with two symptoms,
selected by which branch the optimizer emitted.** If that holds, task #49 (root-cause
the miscompute) and the hang blocker collapse into a single root cause.

### What this does NOT establish

- **It is not a global discriminator.** `beebs_recursion` −O1 contains `bne` backedges
  and **passes**. Fragility is an *amplifier*, not a cause: it only matters if the
  corruption actually lands on that loop's induction variable or bound. Do not restate
  this as "`bne` causes the hang".
- **It does not explain `coremark_matrix`.** Its exits are overwhelmingly ordered
  (`bgeu`), so overshoot cannot be its mechanism. See §3.
- Nothing here identifies *what* corrupts the value. The underlying fault is still the
  unexplained silicon divergence of task #49.

### Two wrong intermediate results, recorded

1. A first backedge classifier parsed the `+0xc0` inside the disassembler's
   `<crcu8+0xc0>` symbol suffix as the branch target, inverting several
   backward/forward classifications. Fixed by stripping `<...>` before matching.
2. At `-O0` clang emits a *forward* exit test plus an unconditional `j` backedge, so a
   backedge-only metric reports `0` conditional backedges for every `-O0` build. The
   table above therefore counts **all** conditional branches, which is the comparison
   that is actually meaningful across opt levels.

---

## 3. `coremark_matrix` has a different mechanism: a data-dependent bound

`matmult_int` has no data-dependent loop bound at all — every loop is `i < MM_N` with
`MM_N` a literal `8`, materialized as `li a3, 0x8`.

`coremark_matrix` does. `core_init_matrix` derives the matrix dimension at runtime:

```c
while (j < blksize) { i++; j = i * i * 2 * 4; }
N = i - 1;
```

which compiles (−O0) to the exit test `bgeu a0, a1, …` at `0x10428` driven by
**`mulw a0, a0, a0` at `0x10444`**. An ordered exit still never fires if the multiply
never produces a value reaching `blksize` (666). And `N` then feeds **every** downstream
loop bound in the benchmark via `p->N`.

So: two hanging rungs, two distinct amplification paths, one plausible common cause.

---

## 4. Next experiments (supersede "shrink the iteration count")

**Test A — task #65, `matmult_int`.** Rebuild at −O1 with the loop exits forced to
ordered comparisons. **Falsifiable prediction: it stops hanging and returns a wrong
value.** Confirms the mechanism and unifies the two symptoms. If it still hangs, the
fragile-exit mechanism is dead.

**Test B — task #66, `coremark_matrix`.** Return `N` straight out of
`core_init_matrix`. Three-way outcome: hangs ⇒ the fault is that `mulw` loop;
returns `N != 9` ⇒ the miscompute caught red-handed and it explains every downstream
bound; returns `N == 9` ⇒ init is clean, look downstream.

Both are one board boot each and need no firmware change. Task #64 (shrink until it
returns) is **deleted** — it yields only a yes/no threshold, where these yield a
mechanism.
