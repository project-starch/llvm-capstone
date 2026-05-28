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

The fetch step still works and pins upstream correctly.

The current tree has been revalidated with `+m` enabled in the canonical helper,
and the earlier compile-time CoreMark blockers have moved again:

- upstream `core_list_join.c` compiles,
- upstream `core_main.c` now compiles at both `-O1` and `-O2` with the current
  local lowering work,
- the full helper now links a real
  `/tmp/capstone/coremark-build/coremark_capstone.dom`.

That means the first end-to-end blocker is no longer a backend selection
failure during compilation. It is now a later runtime failure on the domain path:

- the generic tiny runtime smoke still passes,
- but a QEMU execution of `coremark_capstone.dom` currently aborts in
  `helper_cscincoffsetimm` with `Assertion 'rs1_v->tag' failed` after the domain
  is loaded and created.

So the helper has moved from a **first-blocker detector for compilation** to a
**reproducible compile-and-link smoke plus a runtime reproducer**.

When `clang` crashes, it emits a preprocessed reproducer and run script under
`/tmp/`, for example:

- `/tmp/core_main-*.c`
- `/tmp/core_main-*.sh`

That is exactly the current handoff value: the tree now has a pinned upstream
source, a stable Capstone invocation with `+m`, a successful compile/link path,
and a concrete later-stage runtime failure instead of an earlier backend crash.

Do **not** publish scores from this setup.

## Why this layout

- keeps the main repo free of a vendored benchmark copy,
- avoids coupling first bring-up to the sparse Buildroot package metadata,
- keeps fetch/build paths reproducible,
- makes it easy to throw away and refetch `/tmp/capstone/coremark-src`.

## Next step after the first smoke

Once the current runtime blocker is identified precisely enough:

1. fix that blocker,
2. rerun this smoke,
3. only then decide whether `BEEBS` needs to be pulled in,
4. leave `RV8` for after the first benchmark path is less speculative.


