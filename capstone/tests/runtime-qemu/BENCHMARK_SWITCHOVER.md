# Benchmark switchover note

This note records the recommended handoff from the current SQLite/static-global
work to the next benchmark-oriented stage.

## Recommended point to pause SQLite

Pause now, after the reduced static-capability line has reached:

- committed reproducer
- committed positive control
- committed descriptor contract
- committed descriptor-driven runtime-side POC

Do **not** wait for full LLVM-wide static/global capability integration before
switching focus.

## What to do next

Prioritize benchmark bring-up in this order:

1. `CoreMark`
2. one representative `BEEBS` benchmark
3. `RV8` viability/build check

For each target, do a fast first pass:

- source acquisition
- first compile attempt through the LLVM Capstone path
- first link/domain build attempt
- first runtime attempt if the build succeeds
- record the first real blocker

## Why this order

- `CoreMark` is the best candidate for a quick first meaningful run
- `BEEBS` is good for exposing multiple smaller compiler/runtime gaps
- `RV8` is likely the heaviest target and is better tackled after the first two

## Source management recommendation

Do not add new benchmark repositories as submodules yet.

Prefer this order instead:

1. collect canonical upstream links
2. record exact commit/tag/version to test
3. do the first manual bring-up outside the main tree or through existing package paths
4. add an in-tree vendor copy/submodule only if we later need:
   - local patches,
   - reproducible pinned sources in-tree, or
   - automated wrappers that depend on a checked-in source layout

## Current exception

`CoreMark` already has packaging presence under:

- `capstone/caplifive-buildroot/buildroot/package/coremark/`

So the first exploration should check whether that existing packaging path can be
reused before introducing any new source-management mechanism.

