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
2. one representative `BEEBS` benchmark, but only after the first `CoreMark` blocker is known
3. `RV8` viability/build check, only after `CoreMark` stops being the speculative front

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

## Current decision

For `CoreMark`, the tree should now use a split layout instead of the sparse
Buildroot package stub:

- in-tree orchestration and notes under `capstone/benchmarks/coremark/`
- fetched upstream sources under `/tmp/capstone/coremark-src` by default
- build outputs under `/tmp/capstone/coremark-build` by default

This keeps the main repository clean while avoiding a misleading dependency on
`capstone/caplifive-buildroot/buildroot/package/coremark/`, which is not the
right primary bring-up location for the current work.

Canonical `CoreMark` references:

- repository: <https://github.com/eembc/coremark>
- site: <https://www.eembc.org/coremark/>

Pinned first-pass upstream target:

- tag: `v1.01`
- commit: `cfa9ab377835911f23d9b0831c7be302ed1f58de`

## SQLite parking status

SQLite should now stay parked at the current reduced-VFS/static-global pause
point until `CoreMark` produces the first concrete compiler/linker/runtime
blocker worth switching back from.

So for now:

- continue with `CoreMark`,
- do not add new SQLite-facing scope,
- do not spend time on `BEEBS`/`RV8` links beyond keeping them on the later list.

