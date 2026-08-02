# Boundary Violation — Row 7 (`mrb_bint_reduce`)

> **This row is NOT REPRODUCED.** No boundary violation has been demonstrated, so
> there is no free site or stale-use site to annotate. Read `target.md` first: the
> row's issue number belongs to Row 6, the named function does not exist in the
> mruby version the spec assigns, and the plausible GC hazard inside it is closed
> by mruby's allocation arena.

## The boundary the row *would* have exercised

Recorded so a future attempt does not have to re-derive it.

**Object:** a `struct RBigint` — a heap-allocated mruby object holding an
arbitrary-precision integer's digit array. It is created inside the bigint C
extension and returned to the VM as an `mrb_value`.

**Owner vs. borrower:** the mruby GC owns the lifetime; the `mruby-bigint` C
extension is the borrower, holding raw `struct RBigint*` locals (`b1`, `b2`) and
`mpz_t` views over their digit arrays across allocating calls. Its caller,
`mruby-rational`'s `rational_new_b`, likewise holds `mrb_value` locals (`n`, `d`)
across `rat_alloc`. This is the standard C-extension boundary — the same surface
Rows 8, 12, 13 and 15 exercise, and the reason the row was placed in the corpus.

**Where a violation would appear:** if a collection ran while one of those
temporaries was unrooted, the object would be swept and the subsequent
`mrb_obj_value(b1)` / `mrb_obj_ptr(n)` store would publish a freed pointer into a
live `Rational`, to be read later by the VM.

**Why it does not:** `mrb_obj_alloc` (`src/gc.c:468`) calls `gc_protect()` on every
new object, so each temporary is arena-rooted from birth, and nothing in
`bigint.c` or `rational.c` saves/restores the arena to undo that. See `target.md`
§3 for the full argument including why `MRB_GC_STRESS` cannot change the outcome.

## If a real bigint temporal bug is found

The build here is ready — `mruby-bigint` and `mruby-rational` are both enabled
(required: `mrb_bint_reduce` compiles only under `MRB_USE_RATIONAL`), with
host+ASan and riscv64 targets. Only a trigger would be needed, and this file
should then be rewritten to the normal §8 shape with concrete free and stale-use
sites.
