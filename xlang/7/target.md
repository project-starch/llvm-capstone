# target.md — Row 7 ("mruby #6701 / `mrb_bint_reduce`")

* **Spec description:** "UAF: freed during GCD in `mrb_bint_reduce` (bigint gem),
  then read by the VM"
* **Spec issue number:** mruby #6701
* **Product:** mruby, `mruby-bigint` gem
* **Status:** **NOT REPRODUCED — the row as specified does not appear to exist**
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
