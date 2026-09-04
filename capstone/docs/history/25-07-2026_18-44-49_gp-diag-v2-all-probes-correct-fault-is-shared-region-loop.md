# gp_diag v2 on the board: ALL 11 probes CORRECT — the fault is narrower than believed

**Date:** 2026-07-25 · **Lane:** B · **Second board run** (gp_diag v2, one power-cycle).
Supersedes the hypothesis in `25-07-2026_17-51-27_gp-diag-board-result-loop-terminates-early.md`.
Board left powered off + unlocked.

## Result

```
rung      retval       oracle       cycles   correct
gp_diag   3536372812   3613869247   36259    NO

raw probes: dbg0=23130 dbg1=1234 dbg2=23130 dbg3=89 dbg4=1 dbg5=28
            dbg6=36 dbg7=3 dbg8=9 dbg9=9 dbg10=12648430
```

**Every single raw probe matches its expected value.**

| slot | mechanism | expected | silicon | |
|---|---|---:|---:|---|
| dbg0 | scalar global write→read via `ldc gp[i]` | 23130 | 23130 | ✅ |
| dbg1 | global array store→readback | 1234 | 1234 | ✅ |
| dbg2 | global read inside a **noinline callee** | 23130 | 23130 | ✅ |
| dbg3 | deep self-recursion `fib(10)` | 89 | 89 | ✅ |
| dbg4 | mutual recursion `anka(10)` | 1 | 1 | ✅ |
| dbg5 | **array store in a loop with a LIVE ACCUMULATOR** | 28 | 28 | ✅ |
| dbg6 | **initialized global** (init-template materialization) | 36 | 36 | ✅ |
| dbg7 | function-local `static`, read-before-write (zero-init) | 3 | 3 | ✅ |
| dbg8 | counted loop, **no call** in body | 9 | 9 | ✅ |
| dbg9 | counted loop, **CALL** in body | 9 | 9 | ✅ |
| dbg10 | canary | 12648430 | 12648430 | ✅ |

## Two hypotheses refuted outright

1. **"Call in a loop body" — REFUTED.** `dbg9` is a loop whose body calls a `noinline`
   function, and it returns 9. This was the leading hypothesis from the v1 run; it is dead.
2. **"Array store with a live accumulator" — REFUTED on silicon.** `dbg5` is exactly that
   pattern (`for i: arr[i]=i; s+=arr[i];`) and returns the correct 28. This is the pattern
   the ENTIRE original bug report was built on
   (`23-07-2026_17-30-00_..._array-loop-miscompute-OPEN.md`, the `rc_const0`/`rc_p1` pair).
   It works.

Also cleared, each in isolation: the gp cap-table path, globals via a callee, deep and
mutual recursion, initialized-global materialization, and function-local static zero-init.
**None of the mechanisms we have been chasing is broken by itself.**

## What is still wrong, and the one thing left

`retval` is wrong (3536372812 vs 3613869247) even though every folded input is correct. The
only computation left in the domain is the checksum fold:

```c
for (int i = 0; i < GPD_NPROBE; i++) {
    unsigned v = (unsigned)res[3 + i];        /* reads the SHARED REGION cap */
    for (int b = 0; b < 4; b++) { h ^= (v >> (8*b)) & 0xff; h *= 16777619u; }
}
```

Checked and **excluded**: it is not truncation (no prefix of the correct values reproduces
it, with or without zero-fill), not an index shift (`res[i]`, `res[2+i]`, `res[4+i]` all
fail to match), not "all reads returned res[3]", not a 10- or 12-iteration count, not a
zeroed canary. Single-word solves (assume 10 of 11 read correctly) yield only implausible
garbage. So, as in the perf rungs, **the corruption is broad, not one bad datum**.

### Leading hypothesis: loops that touch the SHARED REGION

The one feature separating every failing computation from every passing one is **access to
the shared-region capability `res` from inside a loop**:

| computation | touches `res`? | in a loop? | result |
|---|---|---|---|
| v1 outer loop `res[3+p] = v` | yes | yes | **ran 1 iteration** |
| v1 inner byte-fold | no | yes | correct |
| v2 probe stores (unrolled) | yes | **no** | all correct |
| v2 probes P10/P11 (loops) | no | yes | correct |
| v2 checksum fold `res[3+i]` | yes | yes | **wrong** |

Straight-line access to `res` is fine (the 11 stores, plus `res[0]`/`res[1]`/`res[2]`, all
landed — the controller read them back). Looping over `res` is not. At `-O0` each iteration
reloads the region capability from the stack (`ldc`) and `cincoffset`s it by the index, so
the suspect is a **capability reload inside a loop**, not the arithmetic.

This also retro-fits v1: there the loop reloaded the spilled capability `&p` across the
call each iteration, and the loop counter came back wrong (so it exited) — consistent with
a bad capability reload rather than the call itself.

**This is a hypothesis, not a result.** It is consistent with all five rows above but has
not been tested directly. Note it is a *different* claim from the v1 one and must not be
reported as confirmed.

## Why this matters regardless

The domain codegen for every capability mechanism we suspected is **correct on silicon**.
That is a large, genuinely good negative result: it clears the gp cap-table ABI, initialized
globals, recursion, and array-store-with-accumulator — the things blocking the
silicon-compatibility claim and the `capstone-gp-free` merge — *as individual mechanisms*.
The remaining fault looks confined to iterated access through the shared-region capability.

## Next: gp_diag v3 (one board run)

Isolate the shared-region-loop hypothesis minimally, straight-line-driven as in v2:
- probe A: loop summing a **global array** (no `res`) → expect correct (control; ≈ dbg5).
- probe B: loop summing **`res[3..13]`** into a scalar, result stored straight-line →
  the suspected failure.
- probe C: same as B but with the region cap **hoisted into a local** before the loop.
- probe D: loop over `res` with a **constant** index (`res[3]` each iteration).
If B fails while A passes, the reproducer is tiny, `-O0`, QEMU-correct, silicon-wrong, and
involves only a shared-region cap in a loop — a far better artifact for the board owner
than anything so far. C and D then say whether it is the reload or the indexing.

**Do not** re-run the perf rungs for numbers until this is settled; their loops all read or
write through domain memory and would inherit the same fault.
