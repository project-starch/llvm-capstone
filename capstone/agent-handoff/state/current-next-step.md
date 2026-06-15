# Current recommended next step

## Immediate milestone - Finish BEEBS source-adaptation cleanup

**Goal**: remove the remaining embedded C heredocs from BEEBS build scripts before
adding more benchmarks.

**Why this next**: the shared domain/host wrapper cleanup is still the right
direction, but benchmark-specific C embedded inside `.sh` files is not a good
long-term structure. The first cleanup pass moved the Capstone-specific adapted
source for these benchmarks into real files under
`capstone/benchmarks/beebs/adapted/`:

- `bubblesort`
- `prime`
- `strstr`

Continue that pattern before adding another benchmark batch. Shell scripts should
orchestrate fetch/build/link/run only; Capstone-specific C should be reviewable
as C source.

## Cleanup rules

- Keep shared domain/host wrappers for benchmarks with identical marker behavior.
- Do not restore duplicate per-benchmark host/domain C files just to carry
  different benchmark names.
- Move substantial benchmark-specific C out of `.sh` files and into
  `capstone/benchmarks/beebs/adapted/`.
- For scripts that append a small Capstone-specific tail to fetched upstream
  source, keep the tail in `adapted/` and let the script concatenate files
  without embedding C text.
- Keep fetched BEEBS sources under `$CAPSTONE_TMP_ROOT`; do not vendor sources or
  add submodules.
- Success is correctness marker only; do not report or optimize performance.
- If the cleanup exposes a backend/compiler/runtime bug or would need higher
  thinking, stop and report the blocker.

## Remaining embedded-C cleanup candidates

Migrate these next, preferably in batches of 3-4:

- `cnt`
- `duff`
- `fdct`
- `insertsort`
- `janne-complex`
- `jfdctint`
- `levenshtein`
- `recursion`
- `tarai`

After the embedded-C cleanup is complete, resume batched BEEBS bring-up. The next
candidate pool remains:

- `aha-compress`
- `nettle-md5`
- `nettle-cast128`
- `slre`
- `matmult`
- `mergesort`
- `nbody`
- `trio`

## Deferred from the previous probe batch

- `stringsearch1`: backend instruction selection failure in `prep1`.
- `crc32`: builds and runs but returns the wrong correctness marker.
- `sglib-rbtree`, `aha-mont64`, `dijkstra`, `edn`, `ctl-string`, `qrduino`,
  `nettle-arcfour`, `ludcmp`, `nettle-des`, `statemate`: compile-time backend
  crashes or non-trivial source adaptation required.

## Test expectations

For the next cleanup commit:

- each migrated `run-beebs-*.sh`
- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`
- `bash capstone/tests/runtime-qemu/run-coremark.sh`
- focused existing BEEBS regressions: `fac`, `strstr`, `ndes`, and one shared
  simple-helper benchmark, preferably `run-beebs-expint.sh`

## Thinking-level rule

Stay at medium thinking while the work remains mechanical or locally debuggable.
If a benchmark exposes a hard backend/compiler bug, unclear architecture
semantics, or repeated failed runtime debugging where higher thinking looks
necessary, suspend work and tell the user before continuing.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence silently
switches the image to stock OpenSBI and breaks all runtime proofs.
