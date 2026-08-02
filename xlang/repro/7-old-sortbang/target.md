# target.md — Row 7 ("mruby #6701 / `mrb_bint_reduce`")

* **Spec description:** "UAF: freed during GCD in `mrb_bint_reduce` (bigint gem),
  then read by the VM"
* **Spec issue number:** mruby #6701
* **Product:** mruby, `mruby-bigint` gem
* **Status:** **NOT REPRODUCED — the row as specified does not appear to exist**
* **Replacement identified 2026-08-02 — see "A real defect for this slot" at the
  end of this file.**
* **Pin used for investigation:** `cda2567c36ca33cd404908ce2fa7bd55ea2a8ed9` (3.4.0-1476)
* **Build:** works (host+ASan and riscv64, with `mruby-bigint` + `mruby-rational`)
* **Probe result:** `trigger.rb` drives the described path under maximum GC
  pressure and completes cleanly, native and under QEMU.

This is a substantive negative result, not an unattempted row. Three independent
problems with the row as specified, each verified against the source.

## 1. The issue number belongs to Row 6

The upstream commit that closes mruby **#6701** is
`e50f15c1c6e131fa7934355eb02b8173b13df415`, whose message is *"mruby-compiler: fix
bytecode corruption in pattern matching optimization"* and which ends with
**"Fixes #6701"**. It changes exactly one thing: the `JMPNOT`→`JMPIF` peephole in
`mrbgems/mruby-compiler/core/codegen.c`. That is **Row 6**, and Row 6 now
reproduces (see `../6/`).

Searching the whole repository history for `6701` returns only that commit. So
#6701 is not a bigint issue, and the spec's pairing of that number with a
`mrb_bint_reduce` GCD bug is a misattribution.

This was confirmed against NVD on 2026-07-27, so it no longer rests on the upstream
commit message alone. The NVD record for **CVE-2026-1979** references issue #6701
and commit `e50f15c1` directly, and names the affected component as the
**"JMPNOT-to-JMPIF Optimization"** in `mrb_vm_exec` / `src/vm.c` — Row 6's defect,
with no mention of bigint, `mrb_bint_reduce`, or the rational path anywhere in the
record. The number is spoken for.

Note this is the opposite of what `../6/target.md` used to claim. The old text said
Rows 6 and 7 were the *same* bug and skipped both on one rationale. In fact the
*identifier* is shared but the *described defects* are different, and only the
pattern-matching one is real.

## 2. The function does not exist in the version the spec assigns

Spec §6 places Row 7 in the **"Tier 1 — mruby, single `3.1.0` build"** cluster, and
§5 lists row 7 among those covered by `git checkout 3.1.0`. But:

| ref | occurrences of `mrb_bint_reduce` |
|---|---|
| `3.1.0` | 0 |
| `3.2.0` | 0 |
| `3.3.0` | 0 |
| `cda2567c` (3.4.0 line) | 1 |

`mrb_bint_reduce` first appears in the 3.4.0 line, and only under
`#ifdef MRB_USE_RATIONAL`. It cannot be built, let alone triggered, from the
3.1.0 checkout the spec prescribes. This row was therefore never buildable as
grouped.

## 3. The plausible GC hazard is closed by the allocation arena

Reading the function, there is an obvious-looking hazard — exactly the shape the
spec describes:

```c
/* mrbgems/mruby-bigint/core/bigint.c:4771 */
struct RBigint *b1 = bint_new(ctx, &a);
struct RBigint *b2 = bint_new(ctx, &b);   /* allocates -> can trigger GC */
*xp = mrb_obj_value(b1);                  /* b1 rooted nowhere?           */
*yp = mrb_obj_value(b2);
```

`b1` is held only in a C local across a second allocating call, and mruby does not
scan the C stack. The same pattern appears in the only caller, `rational_new_b`
(`mrbgems/mruby-rational/src/rational.c:265`), where `n` and `d` are plain locals
across `rat_alloc`.

**It is not exploitable, because mruby roots every fresh allocation in the GC
arena.** `mrb_obj_alloc` (`src/gc.c:468`) ends with:

```c
gc->live++;
gc_protect(mrb, gc, &p->as.basic);   /* pushes the new object onto the arena */
```

so `b1`, `b2`, `n` and `d` are all arena-rooted from birth. Confirmed further:

* **No arena save/restore anywhere in the path** — `grep` for
  `arena_save|arena_restore|ARENA` across `bigint.c` and `rational.c` returns
  nothing, so nothing un-protects them before use.
* **Arena overflow does not silently drop protection** — `gc_arena_keep`
  (`src/gc.c:381`) grows the arena by default; with `MRB_GC_FIXED_ARENA` it raises
  a proper arena-overflow exception.
* **`MRB_GC_STRESS` would not help.** The full GC it forces is invoked from inside
  `mrb_obj_alloc` itself, *before* `gc_protect` for the new object but *after* the
  previous one was protected — so `b1` survives a stress collection during `b2`'s
  allocation by construction.

`trigger.rb` drives `Rational(bignum, bignum)` 3000 times with
`GC.interval_ratio = 1`, `GC.step_ratio = 1`, a churning heap and periodic
`GC.start`. It completes cleanly under ASan and under `qemu-riscv64`.

## What would settle it

The build infrastructure here is working and correct — `build.sh` and
`build_config.rb` produce host+ASan and riscv64 mruby with `mruby-bigint` and
`mruby-rational` enabled (both are required: `mrb_bint_reduce` is compiled only
when `MRB_USE_RATIONAL` is defined, which `mruby-rational`'s mrbgem.rake sets).
So if the row is real, only the trigger is missing.

To resolve, someone needs to read the actual upstream source of this row:

1. Find the real issue number for a bigint/GCD use-after-free, if one exists. The
   spec's own §6 cites `github.com/mruby/mruby/issues/6701` for this row, and that
   URL is the pattern-matching bug — so the citation cannot be taken at face value.
2. Check whether the intended defect is in a *different* bigint entry point
   (`mrb_bint_gcd` immediately below `mrb_bint_reduce` has the same shape and the
   same arena protection), or in an older bigint implementation.
3. If no such issue exists, drop Row 7 from the benchmark. Note that the companion
   note's tally sentence explicitly claims the corpus spans "the bigint gem", so
   dropping the row means amending that claim.

Until then this row should be reported as **unresolved**, not as verified. It is
listed in the benchmark table as a confirmed temporal-borrow defect with a public
ASan trace; nothing found here supports that.

> **Supersedes an earlier SKIPPED filing.** The previous rationale claimed ASan
> cannot boot mruby 3.3.0 because of "GC stack-scanning" failures. That is false —
> the ASan build here boots and runs normally at the 3.4.0 pin, and Row 6's ASan
> build boots at the same commit. The previous artifact also consisted of nothing
> but a `target.md`; there was no build, trigger, or run script to substantiate any
> claim either way.


---

## A real defect for this slot — REPRODUCES, but see the oracle problem


The row as specified does not exist, but the *slot* it was meant to fill — a
Ruby↔C temporal UAF reachable from a script — has a real, recent CVE that fits
better than the original ever did.

**CVE-2025-13120 / mruby #6649 — use-after-realloc in `Array#sort!`.**

| | |
|---|---|
| Vulnerable pin | `ec58dca22f0e` (parent of the fix) |
| Fix | `eb398971bfb4`, 2025-10-27, *"array.c: fix use-after-realloc in Array#sort!"* |
| Site | `sort_cmp` / `heapify` / `insertion_sort`, `src/array.c` |
| Class | CWE-416, temporal |

`sort_cmp` is passed `mrb_value *p`, a **raw pointer into the array's backing
store**, cached by its callers. The comparison block is **Ruby code**, so it can
grow or shrink the array and cause that store to be reallocated. The pre-fix
guard is only:

```c
if (RARRAY_PTR(ary) != p) { ... "array modified during sort" ... }
```

which catches a *move* but not a *length change*. The fix re-reads
`RARRAY_PTR(ary)` inside `sort_cmp` and adds `RARRAY_LEN(ary) != n`.

### Why this is a better fit than the original row 7

- **It is real, and recent.** The original row's hazard is closed by the GC
  arena; this one has an assigned CVE and an upstream fix.
- **It is the corpus's dominant mechanism at a NEW site.** Six rows are already
  "interior pointer cached across a re-entrant Ruby callback, backing store
  moves" — but all six are the *VM register stack*. This is the *Array backing
  store*. It strengthens the claim that the pattern recurs across subsystems
  rather than being one bug counted six times.
- **The boundary is clean.** A Ruby block re-entering the interpreter during a C
  sort is exactly the cross-language shape the corpus is about.

### What it costs to adopt

Not free, and the trigger is the uncertain part. Re-pin to `ec58dca22f0e`,
rebuild with ASan (the current `build_config.rb` enables `mruby-bigint` and
`mruby-rational` for the old defect and would no longer need them), then find a
block that shrinks the array such that the realloc **returns the same address** —
otherwise the old guard catches it and raises `RuntimeError` instead of faulting.
That is the whole subtlety of the bug and it is where the work is.

Then: a shim, an entry in `check_shim_fidelity.py`'s table, and a measurement in
each of the two columns. **Neither column has measured it**, so both `RESULTS.md`
files remain 14 rows until it lands.


---

# UPDATE 2026-08-02 — the replacement REPRODUCES, and both sanitizers are blind to it

Re-pinned to `ec58dca22f0e`, dropped the bigint/rational gems (`Array#sort!` is
core), wrote the upstream reporter's input as `trigger.rb`, and built.

## It reproduces, deterministically

| build | result |
|---|---|
| **plain** (`-g -O1`, no sanitizer) | **SIGSEGV, exit 139, 3 runs of 3** |
| ASan | `RuntimeError: array modified during sort` — no fault |
| valgrind | exit 1, **0 errors from 0 contexts** — no fault |

The plain crash is exactly what the reporter described:

```
#0  mrb_obj_is_kind_of ... src/object.c:558
#1  cmpnum            ... src/numeric.c:2119
#2  num_lt            ... src/numeric.c:2185
#3  mrb_vm_exec       ... src/vm.c:2247
#5  mrb_yield_argv    ... src/vm.c:1288
```

A stale `mrb_value` read through the cached backing-store pointer is passed to
the comparison block, and comparing it dereferences garbage.

## Why BOTH sanitizers mask it — and why that matters

The pre-fix guard is `if (RARRAY_PTR(ary) != p) raise "array modified during
sort"`. It catches a *move*.

**ASan and valgrind both replace `realloc` with an implementation that always
moves** — allocate new, copy, free old — because that is how they detect
use-after-realloc. So under either tool the pointer changes, the guard fires,
and a clean `RuntimeError` is raised instead of the defect occurring.

With the system allocator, shrinking 100 elements to 2 is satisfied **in place**:
the pointer does not change, the guard passes, and heap sort keeps indexing
through a buffer that is now far smaller.

This is a **different blindness from row 3**, and a sharper one. Row 3's ASan
blindness is that the offending access executes in uninstrumented prebuilt code.
Here the sanitizer does not fail to *observe* the defect — it **prevents** it, by
changing the allocator behaviour the defect depends on.

That generalises into the strongest sanitizer-limitation argument the corpus
has: for any defect whose trigger depends on allocator reuse or shrink-in-place,
a sanitizer that reallocates differently cannot see it, and no amount of extra
instrumentation helps. It is an argument for enforcement in the allocator or the
hardware rather than in a debugging tool.

## Why it is NOT adopted into the measured tables yet

Two blockers, both substantive rather than mechanical:

1. **The corpus's native oracle is ASan**, and `check_shim_fidelity.py`
   validates every shim by triggering its defect under ASan. This row cannot be
   validated that way. Its oracle would have to be the plain build's SIGSEGV,
   which is a second oracle kind the gate does not currently model.

2. **The "free" here is a shrink-in-place, so there may be no free at all for
   revocation to observe.** That is materially different from every other
   temporal row. On CHERI the cached capability keeps its original bounds over
   memory that is still mapped, so a MISS in all three configs is the likely
   outcome. On Capstone our `rof_realloc` *always* moves and revokes, so it would
   be caught — but that is a property of **our allocator**, not of the defect,
   and scoring it as a catch would be measuring the harness.

Blocker 2 is the interesting one and should not be papered over: it says the
shim methodology assumes every "free" is a real free, and this defect violates
that assumption. Resolving it means either modelling shrink-in-place faithfully
in the mock on both columns, or documenting the row as out of scope for the
shim approach.

## Files as they now stand

`build.sh` is re-pinned to `ec58dca22f0e` and `build_config.rb` no longer pulls
in bigint/rational. `trigger.rb` is the upstream input, unmodified. The old
non-existent-defect analysis above is retained because it is still the reason
the original row was dropped.

**The RISC-V leg was not built** — `riscv64-linux-gnu-gcc` is absent on this
machine, and its failure aborts rake before the host target finishes. Use a
host-only config to rebuild here.
