# Lua binary-trees — CHERI vs Capstone, performance & memory

**The temporal-safety cost of running an OFFICIAL benchmark — the Computer Language
Benchmarks Game `binary-trees` — on both platforms, on two axes: time and memory.**
This is the perf/memory companion to `RESULTS.md` (which is the *security* axis: which
platform actually catches the stale accesses). Same benchmark, same size (N=6), each
platform in its native deployment mode.

## One-sentence result

**Capstone reaches *eager-strength* temporal safety — a synchronous revoke on every
`free`, which catches 13/13 of the cross-domain use-after-frees — at ≈1.0002× the
run time; CHERI's eager mode buys the same strength at ≈370× time, so CHERI actually
ships the *async* mode that catches 0/13 at the access.** Capstone pays for the cheap
revoke in space: per-object revocation-node metadata.

## What is run

The unmodified CLBG `binary-trees` workload: build many complete binary trees of Lua
tables, checksum each (count its nodes), discard it. It is allocation- and
GC-intensive by design — exactly the churn that exercises a temporal-safety mechanism.
At N=6 it builds 82 trees (deepest = 255 nodes) via ~7,960 heap allocations, each of
which is later freed (and, under the safe configs, revoked).

| | **Capstone** | **CHERI** |
|---|---|---|
| Vehicle | freestanding **domain**, Buildroot-Linux host controller, Capstone-QEMU | ordinary **CheriBSD purecap process**, CHERI-QEMU |
| Real Lua? | yes — reference Lua 5.4.7, base library, the actual tree build/traverse/GC | yes — reference Lua 5.4.7 |
| Allocator | `revoke_arena_domain.c` on `revoke_on_free_alloc.h` — every allocation an independently revocable capability | CheriBSD `malloc` + kernel revocation |
| Time metric | dynamic instruction count, `rdcycle` under `-icount shift=0` | dynamic instruction count, `rdinstret` |
| Memory metric | rof heap counters + QEMU revocation-node high-water | peak process RSS, `getrusage(RUSAGE_CHILDREN)` |

Both are **functional-model proxies** (deterministic dynamic instruction counts, not
cycle-accurate timing). The ISAs differ, so the comparable quantity is the **ratio**
within a platform (the overhead of turning temporal safety on), not absolute counts.

## The metric that makes them comparable

Each platform has a config with per-object spatial safety but **no** temporal safety,
and configs that add the temporal mechanism on top:

| Capstone | CHERI | temporal safety? |
|---|---|---|
| `norevoke` — per-object revocable caps, revoke suppressed | `spatial` — per-object bounds, revocation off | no (baseline) |
| `revoke` — revoke on every `free` | `eager` — revoke on every `free` | **yes, synchronous** |
| — | `temporal` — async quarantine sweep (**the deployed default**) | partial (async) |

So `revoke − norevoke` (Capstone) and `eager − spatial` / `temporal − spatial` (CHERI)
all isolate *the cost of the temporal layer*.

## Results (binary-trees N=6)

### Capstone

| build | instr (whole benchmark) | check |
|---|---:|---|
| norevoke (baseline) | 464,313,752 | 4398 |
| revoke (full temporal) | 464,393,352 | 4398 |
| **revoke − norevoke** | **+79,600 instr → 1.0002×** | equal ✓ |

The +79,600 over the whole run cross-checks the O(1) micro-cost: ~7,960 frees × ~10
instr/revoke. Memory (revoke build): working set **1,030 peak live objects** / ~90 KB
live; total heap carved 732 KB (rof never reclaims); **17,849 revocation nodes**
(~0.35 MB metadata). The node count is ~2.2× *total* allocations, not ~2× the working
set — the metadata **leaks** because the non-reclaiming allocator never re-uses freed
bytes, so a collected object's child-capabilities linger and keep the node referenced.
A reclaiming allocator would bound it to ~2× the working set (~2 K nodes ≈ ~40 KB).

### CHERI

| config | instr (workload) | time vs spatial | peak RSS | RSS vs spatial |
|---|---:|---:|---:|---:|
| spatial (baseline) | 868 M | 1.00× | 3,377 KB | 1.00× |
| **temporal** (deployed) | 1.01 B | **≈1.2×**† | 4,121 KB | **+744 KB (1.22×)** |
| **eager** | 322 B | **≈370×**† | 3,480 KB | +103 KB (1.03×) |

Async temporal holds freed objects in a quarantine → +744 KB RSS; eager revokes
immediately so nothing accumulates → +103 KB, but ~370× time.

† CHERI instruction counts carry a few-percent **`rdinstret` noise** — the counter
advances for ALL hart activity (kernel, daemons) inside the bracket, and eager is n=1.
Two runs bracketed temporal at 1.17–1.24× and eager at 370–390×; the memory figures are
stable. The Capstone numbers are deterministic (`-icount`), hence exact.

## The comparison

Match on **security** (from `RESULTS.md`): Capstone `revoke` and CHERI `eager` both give
synchronous revoke-on-free and both catch **13/13** cross-domain UAFs at the access;
CHERI's deployed `temporal` (async) catches **0/13** at the access.

| | Capstone `revoke` | CHERI `eager` | CHERI `temporal` (deployed) |
|---|---:|---:|---:|
| CDP UAFs caught at the access | **13/13** | 13/13 | **0/13** |
| time overhead | **1.0002×** | **≈370×** | ≈1.2× |
| temporal memory overhead | ~0.35 MB rev-node metadata¹ | +103 KB RSS | +744 KB RSS |

¹ rof×GC-leaked figure; ~40 KB with a reclaiming allocator.

**Reading:** both `revoke` and `eager` deliver the same strong, synchronous temporal
safety. CHERI pays for it in **time** (a quarantine sweep per free → ~370× → undeployable,
which is why the shipped mode is async and catches nothing at the access). Capstone pays
for it in **space** (per-object revocation-node metadata), but the revoke itself is an
O(1) tag-tree splice, so time is essentially free. It is a **time-for-space trade that
lands on the right side for this security property**.

## Honest caveats

- Functional-model proxy, not silicon timing; compare ratios, not absolute counts across
  ISAs.
- The two memory figures live in **different places** — Capstone's is hardware/model
  revocation-tree metadata; CHERI's is process-heap quarantine RSS. Both are
  "temporal-safety memory," but not the same bytes.
- Capstone's node figure is inflated by the **prototype allocator's non-reclamation**;
  it is an allocator property, not a floor of the Capstone model.
- Capstone-QEMU's revocation-node pool was raised from 10,000 to **65,536** to match the
  deployed silicon bitstream (`caplifive_65536_nodes.bit`); this is pure model metadata
  and does not affect any instruction count.

## Reproduce

```bash
# one command, both platforms, both axes (QEMU is serialized; ~45 min):
CAPSTONE_REPO_ROOT=<repo> N=6 bash xlang/lua-cdp/reproduce-temporal-comparison.sh
```

Constituent scripts, runnable on their own:

- `xlang/lua-cdp/capstone-lua/measure-bintrees-cost.sh` — Capstone perf + memory.
- `xlang/lua-cdp/cheri/bench/reproduce-cheri-lua-bench.sh` — CHERI perf + memory.

Prereqs: a built `capstone-qemu` (65,536-node pool + the `REV-NODES` print), the Capstone
clang/lld and Buildroot host toolchain, and a cheri-baseline-provisioned CheriBSD purecap
image.
