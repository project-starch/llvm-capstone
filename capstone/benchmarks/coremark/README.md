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
failure during compilation. It is now a later runtime failure on the domain path.

The currently observed runtime problems split into **three distinct classes**:

1. stack capability addresses that must stay capability-preserving during backend
   selection,
2. compiler-generated static tables of capability-valued addresses (for example a
   lowered switch table in `get_seed_32()`),
3. zero-initialized globals that fall into `.bss` and end up outside the current
   `gp`-relative capability bounds used by the domain runtime.

The reduced `.gct` proof of concept already in tree is still valid, but it is
important to keep its scope precise:

- it proves compiler emission plus reduced runtime-side consumption for narrow
  one-slot static objects,
- it does **not** yet provide a generic startup/runtime materializer for arbitrary
  compiler-generated static capability tables inside a real benchmark image,
- and it does **not** yet widen the domain's `gp`-relative capability policy to
  cover all zero-initialized globals automatically.

So the current CoreMark blocker is from the **same broad static-capability area**,
but not from the already-closed reduced `.gct` path.

For the current benchmark smoke, the helper applies two temporary CoreMark-local
workarounds while the generic path is still under construction:

- compile `core_util.c` with `-fno-jump-tables` to avoid emitting a static switch
  table of capability addresses,
- compile `port/core_portme.c` with `-fno-zero-initialized-in-bss` so the volatile
  seed globals stay in `.data` instead of splitting the last seed into `.bss`.

After those two benchmark-local workarounds, any remaining runtime failures are a
much cleaner signal for residual backend capability-selection bugs instead of the
already-known static table / `.bss` issues.

- the generic tiny runtime smoke still passes,
- and CoreMark now serves as a reproducible runtime bring-up target for the
  remaining capability-selection issues after those localized benchmark
  workarounds.

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


