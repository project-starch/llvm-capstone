# CULPRIT FOUND: a register-indexed load does not observe pending stores to multiple addresses

**Date:** 2026-07-27
**Lane:** C (primary)
**Cost:** four board boots (`rawhazard4`–`7`), each QEMU-verified first. Board off + unlocked.
**Supersedes:** every prior mechanism hypothesis for the silicon divergence.

---

## The finding

> **A load whose address arrives through a register — either a register-carried capability or a
> register-computed offset — does not observe pending stores to other addresses.** Only a load at
> a compile-time-constant offset from a freshly re-derived cap-table entry (`ldc gp[i]` then
> `lw imm(cap)`) is safe, and only one stored location is safe for the register forms.

QEMU executes every probe below correctly, so this is a **hardware** fault, not a compiler bug.

## How it was isolated

Starting point: `insertsort_diag` showed stores correct, loop bounds correct, and the inner loop
running 1 trip instead of 45 — the array ending exactly as "one swap per outer pass" predicts.

| probe | shape | board | correct |
|---|---|---|---|
| v4 P1 | memory condition, **literal** index, one location stored | 5 | 5 ✅ |
| v4 P2 | register-only condition (control) | 5 | 5 ✅ |
| v4 P3 | condition reads memory the body never writes | 5 | 5 ✅ |
| **v4 P4** | **register index + a store to another location** | **1** | 5 ❌ |
| v5 A | register index, **no** other store | 5 | 5 ✅ |
| v5 B | literal index, **with** other store | 5 | 5 ✅ |
| v5 C | = P4, re-run on a fresh boot (stability control) | **1** | 5 ❌ |
| v5 D | as C, store *after* the decrement | **1** | 5 ❌ |
| v5 E | plain `[j]` instead of `[j-1]` | **1** | 5 ❌ |

**Neither ingredient alone fails.** A (register index) passes; B (extra store) passes; together
they fail. D shows ordering is irrelevant, E shows the index arithmetic is irrelevant — it is
register-indexed addressing as such. C reproducing on a separate boot establishes determinism.

## It is not loop-specific

`rawhazard6` W2 hoisted the condition value into a register: the loop then ran **zero** times,
meaning the single register-indexed load *before* the loop returned 0 where 5 had just been
stored. So the fault does not need a loop at all — the loop merely makes it visible.

## No software mitigation exists (7 tried, all failed)

| mitigation | board |
|---|---|
| `fence rw,rw` before the load | **1** ❌ |
| `fence rw,rw` after every store | **1** ❌ |
| hoist the value into a register | **0** ❌ (worse) |
| make the other store register-indexed too | **1** ❌ |
| separate the two locations by 64 B (cache lines) | **1** ❌ |
| pointer walk, constant-offset load `lw 0(p)` | **1** ❌ |
| both accesses through pointers | **1** ❌ |

**Fences not helping is diagnostic**: this is not a memory-ordering problem, it is address
disambiguation in the load path. That also retires the old `fence.i` / domain-boundary line of
enquiry for good.

**Sixth and seventh candidates also failed** (`rawhazard7`): walking a pointer so the load is a
constant offset `lw 0(p)` still returns stale data (P1 = 1), and doing both accesses through
pointers likewise (P2 = 1). A read-only pointer walk with no stores in the loop is correct
(P3 = 5), confirming reads alone are fine.

**This corrects the earlier reading that "constant offset is safe".** It is safe only when the
base capability is re-derived from the cap-table each time (`ldc gp[i]` + `lw imm(cap)`, the
v5-B shape). Once the capability is *carried in a register* across the access — which is what a
pointer walk does, and what any optimiser would naturally produce — the fault returns. So the
strength-reduction mitigation is dead too: there is no way to express a dynamic array index whose
base is a compile-time constant, which means **no general software workaround exists**.

## Why this explains the whole 3-pass / 4-fail split

For the first time, one mechanism accounts for every rung without special pleading:

| rung | shape | outcome |
|---|---|---|
| `beebs_prime` | scalar, no memory in any loop condition | ✅ passes |
| `beebs_recursion` | recursion, same | ✅ passes |
| `rv8_primes` | inner loop touches exactly ONE location per iteration → no second pending store | ✅ passes |
| `matmult_int` | `C[i*N+j] += A[…]*B[…]` — register-indexed loads + a store elsewhere | ❌ fails |
| `coremark_matrix` | same shape | ❌ fails |
| `beebs_crc32` | table load register-indexed, plus other stores | ❌ fails |
| `beebs_insertsort` | inner loop condition register-indexed, body stores two locations | ❌ fails |

`rv8_primes` passing was the fact that killed several earlier theories; it is now explained
rather than excused.

## Status of the paper

Unchanged in its numbers: **3 measured rungs + the §5 caveats**. What changes is the quality of
the caveat — "an unexplained divergence" becomes "a characterised hardware fault with a five-line
reproducer and seven failed mitigations". That is a materially stronger thing to publish and to
hand to the board owner.

## Retired hypotheses

For the record, all refuted against controls: shrink→store forwarding; domain-entry fault;
fragile `bne` loop exits; extra capability load/store in a loop; block-capability memory
round-tripping; redundant NONLIN→NONLIN `delin` (the `delin` *is* separately fatal, but is not
this); narrow `sh` accesses; plain store→load forwarding; store→load disambiguation with
identical address expressions; "any loop with a memory-dependent exit condition".
