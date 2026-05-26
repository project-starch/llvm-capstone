# CoreMark first-pass bring-up

This directory is the clean in-tree orchestration point for the first `CoreMark`
bring-up.

It intentionally does **not** vendor the upstream benchmark sources into this
repository and does **not** reuse
`capstone/caplifive-buildroot/buildroot/package/coremark/` as the primary work
area.

Instead, the split is:

- in-tree scripts/docs live here,
- upstream sources are fetched into `/tmp/capstone/coremark-src` by default,
- build outputs land under `/tmp/capstone/coremark-build` by default.

## Canonical upstream references

- repository: <https://github.com/eembc/coremark>
- site: <https://www.eembc.org/coremark/>

Default pinned upstream for the first pass:

- ref: `v1.01`
- commit: `cfa9ab377835911f23d9b0831c7be302ed1f58de`

## What is in scope right now

1. fetch a reproducible upstream `CoreMark` tree outside the repo,
2. compile/link a first Capstone-domain smoke against the real upstream sources,
3. record the first real compiler/linker/runtime blocker,
4. keep `BEEBS` and `RV8` deferred until the first `CoreMark` blocker is clear.

## Files

- `fetch-coremark.sh` — fetch or refresh the pinned upstream tree.
- `build-coremark-capstone.sh` — build a first Capstone-domain smoke.
- `coremark_domain.c` — tiny `domain_main()` wrapper for the upstream benchmark entry.
- `port/core_portme.h` — minimal Capstone-specific `CoreMark` port header.
- `port/core_portme.c` — minimal Capstone-specific `CoreMark` port implementation.

## Quick start

```bash
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh
bash capstone/benchmarks/coremark/fetch-coremark.sh
bash capstone/benchmarks/coremark/build-coremark-capstone.sh
```

## Result of the current smoke

The fetch step works and pins upstream correctly.

The current tree has been revalidated after the latest reduced static-capability
work, and the first honest `CoreMark` blocker is still the same one.

The first honest Capstone compile attempt currently fails before link with a
backend crash while compiling upstream `core_list_join.c`:

- function: `core_list_init`
- current symptom: Capstone backend instruction-selection failure on an `i128`
  shift/division path (`Cannot select ... i128 = shl ...`)
- observed at: `-O0`, `-O1`, and `-O2`

So the current helper is best understood as a **reproducible first-blocker
detector**, not a successful domain build yet.

When `clang` crashes, it emits a preprocessed reproducer and run script under
`/tmp/`, for example:

- `/tmp/core_list_join-*.c`
- `/tmp/core_list_join-*.sh`

That is exactly the current handoff value: the tree now has a pinned upstream
source, a stable Capstone invocation, and a concrete backend failure instead of a
speculative benchmark plan.

Do **not** publish scores from this setup.

## Why this layout

- keeps the main repo free of a vendored benchmark copy,
- avoids coupling first bring-up to the sparse Buildroot package metadata,
- keeps fetch/build paths reproducible,
- makes it easy to throw away and refetch `/tmp/capstone/coremark-src`.

## Next step after the first smoke

Once the first real `CoreMark` blocker is identified:

1. fix that blocker,
2. rerun this smoke,
3. only then decide whether `BEEBS` needs to be pulled in,
4. leave `RV8` for after the first benchmark path is less speculative.


