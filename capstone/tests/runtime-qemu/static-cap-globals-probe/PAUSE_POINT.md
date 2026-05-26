# Static capability globals pause point

This note captures the current stopping point for the LLVM-path static/global
capability work so the tree can switch to benchmark bring-up and later return to
finish the broader integration.

## What is already proven

The reduced probe in this directory now demonstrates four distinct states:

1. `direct_use_domain.c`
   - direct capability use works
2. `runtime_materialize_domain.c`
   - runtime-side writable-object reconstruction works
3. `descriptor_materialize_domain.c`
   - the same reduced object works when rebuilt from descriptors + raw templates
4. `static_const_domain.c`
   - loading equivalent values from a file-scope `static const` object still fails

So the narrow blocker is now well isolated:

> direct capability use works, but static/global read-only objects with
> capability-valued fields are not yet supported in the LLVM-generated domain
> path.

## Why this is a safe pause point

The work has advanced past raw diagnosis and now has:

- a stable reproducer
- a stable positive control
- a minimal descriptor contract
- a positive descriptor-driven runtime-side proof of concept

That is enough to pause SQLite-specific follow-up without losing the essential
technical thread.

## External reference captured before pausing

`capstone-c` appears to implement the analogous idea in its code-generation
prologue path:

- `capstone/capstone-c/src/codegen.rs`
- `FnCodeGen::generate_prologue()`
- currently around lines 264-293

That path:

1. takes the initial memory capability from `cscratch`
2. reserves space for a global capability table
3. places that table in `gp`
4. allocates one capability per global object
5. stores each capability into the table

This is a useful reference for the eventual LLVM-path runtime/global-object
access model, even though the current reduced probe intentionally stays much
smaller and more local.

## What remains when returning to SQLite

When this line of work is resumed, the next unresolved layers are:

1. choose the first real runtime policy:
   - eager-at-init, or
   - lazy-on-first-use
2. generalize the one-object descriptor-driven POC into a small live global
   object table/access model
3. connect LLVM-side emission to runtime descriptors/templates
4. re-enable natural static/global initialization in the SQLite-facing slice

## Recommended re-entry point

If benchmark work becomes the temporary priority, return here later and resume
from:

> generalizing `descriptor_materialize_domain.c` into a tiny runtime global
> table/access model, still without touching the loader first.

