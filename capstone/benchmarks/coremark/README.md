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

The fetch step still works and pins upstream correctly. The current helper builds
and links `/tmp/capstone/coremark-build/coremark_capstone.dom`, runs all three
CoreMark algorithms on the split Capstone PureCap domain path, and validates the
profile-run CRCs. Expected runtime output includes `Correct operation validated.`
and `__COREMARK_PASSED__`.

CoreMark now uses the compiled C `domain_main` wrapper in `coremark_domain.c`;
`coremark_domain_entry.S` is retained only as historical reference and is no
longer compiled or linked by `build-coremark-capstone.sh`. The wrapper is kept at
`-O0` while higher optimization levels still expose the known rd!=rs1 LINEAR-cap
sink issue. The remaining backend workarounds are documented in
`capstone/agent-handoff/plans/backend-compiler-fixes.md`.

When `clang` crashes, it emits a preprocessed reproducer and run script under
`/tmp/`, for example:

- `/tmp/core_main-*.c`
- `/tmp/core_main-*.sh`

The current handoff value is a pinned upstream source, a stable Capstone
invocation with `+m`, and an end-to-end correctness run for compiler/runtime
bring-up.

Do **not** publish scores from this setup.

## Why this layout

- keeps the main repo free of a vendored benchmark copy,
- avoids coupling first bring-up to the sparse Buildroot package metadata,
- keeps fetch/build paths reproducible,
- makes it easy to throw away and refetch `/tmp/capstone/coremark-src`.

## Next step after the validated CoreMark path

Use the CoreMark build pattern to start BEEBS benchmark porting. Start with one
small deterministic benchmark, add a focused run wrapper, and validate it before
expanding the set. Leave RV8 for after the first BEEBS path is stable. If a new
benchmark exposes one of the remaining backend bugs, fix that root cause with a
focused reproducer before adding more benchmarks.


