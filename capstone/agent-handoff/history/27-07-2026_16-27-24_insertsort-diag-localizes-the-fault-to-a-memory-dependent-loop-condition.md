# The fault is localized: a loop whose EXIT CONDITION reads memory. Stores, bounds and straight-line load-after-store are all proven fine.

**Date:** 2026-07-27
**Lane:** C (primary)
**Cost:** four board boots. Board powered off + unlocked after each.

This is the first **hard localization** in this investigation — everything before it was a
static correlation, and eight of those have now died. What follows is what the hardware
actually did, not what a code pattern suggested.

---

## 1. The decisive probe: `insertsort_diag`

`beebs_insertsort` returns a wrong value (957879052 vs 271779359) with only ~560 retired
instructions — far too few for the sort. A checksum cannot say which part is wrong, so this rung
returns the array state RAW in the controller's debug slots. One boot, all of it:

| slot | board | correct | verdict |
|---|---|---|---|
| `dbg0–10` array after `is_init()` | `0 11 10 9 8 7 6 5 4 3 2` | identical | ✅ **stores are correct** |
| `dbg11–21` array after `is_sort()` | `0 10 9 8 7 6 5 4 3 2 11` | `0 2 3 4 5 6 7 8 9 10 11` | ❌ |
| `dbg22` verify | 0 | 1 | ❌ |
| `dbg23` outer trips | **9** | 9 | ✅ **loop bounds are correct** |
| `dbg24` inner trips | **9** | **45** | ❌ exactly one per outer pass |

**The observed array is exactly what "one swap per outer pass" produces**, element for element:
i=2 swaps a[2],a[1]; i=3 swaps a[3],a[2]; … i=10 swaps a[10],a[9] — walking the 11 from index 1
to index 10 and shifting everything else down. It also predicts inner=9. Both match. This is an
arithmetic identity, not an impression.

So the inner loop `while (is_a[j] < is_a[j-1]) { swap; j--; }` **exits after its first
iteration, every time**, when for i>=3 the condition should still be true.

## 2. Two mechanisms proposed and both REFUTED on the board

**(a) Store-to-load forwarding is broken** — `rawhazard2`, loop-free:
`store OLD; store NEW; <0,2,4,6,8,10,12 nops>; load` → **187 (correct) at every distance**,
including zero. Refuted.

**(b) Store-to-load *disambiguation* is broken** (same address, different address computation —
insertsort stores through `is_a[j-1]` and loads through `is_a[j]` after `j--`) — `rawhazard3`,
loop-free, five probes including insertsort's exact shape and two controls:
**187 on all five.** Refuted.

Both probes are QEMU-verified with the same expected values, so a board deviation would have
been real. There was none.

## 3. What the negative results jointly point at

| probe | contains a loop? | loop condition | result |
|---|---|---|---|
| `rawhazard` v1 | **yes** (distance sweep + nop loops) | register | **HUNG** |
| `rawhazard2` v2 | no — straight line | — | **PASS** |
| `rawhazard3` v3 | no — straight line | — | **PASS** |
| insertsort **outer** loop | yes | `i <= 10` — **register only** | **CORRECT (9 trips)** |
| insertsort **inner** loop | yes | `is_a[j] < is_a[j-1]` — **two memory loads** | **WRONG (1 trip)** |

Straight-line code is fine. A loop with a register-only bound is fine. **A loop whose exit
condition is computed from values loaded out of the capability-addressed array is not.** That is
the narrowest statement consistent with every board result to date, and it also fits the other
three failures: `matmult_int` and `coremark_matrix` both drive inner loops over values held in
such arrays, and `beebs_crc32` reads its table back after writing it.

It further explains the pass set without special pleading: `beebs_prime` and `beebs_recursion`
have no memory-dependent loop condition at all, and `rv8_primes`'s inner loop is bounded by a
register counter — its array traffic is inside the body, not in the exit test.

**This is a hypothesis about a class, not yet a mechanism.** It is stated here because it is the
first one that survives all eight refutations, not because it is confirmed.

## 4. The next probe, already specified

Minimal, loop-free elsewhere, one boot:

```c
rh_a[1] = 5;  long n = 0;
while (rh_a[1] > 0) { rh_a[1]--; n++; }   /* memory-dependent exit condition */
res[3+0] = n;                              /* correct = 5 */
```

plus a register-condition control (`long c = 5; while (c > 0) { c--; n++; }`) in the same image.
If the memory-condition loop returns n != 5 while the register control returns 5, the class is
confirmed with a two-loop repro that fits on a slide. If both are correct, the loop framing is
wrong too and the next step is to bisect `is_sort` itself rather than model it.

## 5. Incidental finding

`Cannot select: i128 = sign_extend_inreg` when an `int` index feeds capability address
arithmetic — the same family as the `i128 = and` gap fixed earlier today. Worked around with
`long` indices in the probe; not chased. Worth fixing alongside the remaining
`lowerScalarI128Logical` gap.

## 6. Standing

Paper position is unchanged and safe: **3 measured rungs + the §5 caveats**. Nothing here
changes a number. What it changes is that the blocker now has a *localized* description —
"a loop whose exit condition reads capability-addressed memory" — instead of "unexplained",
which is a materially better thing to hand the board owner.
