# APR pools on CheriBSD

Build a direct-link library and [standalone example](../../examples/pools.c)
using the [shared CheriBSD workflow](../../../../common/host/cheribsd/README.md).
From this component directory, after sourcing the repository test environment:

```sh
export CHERI_SDK=/path/to/sdk
export CHERI_SYSROOT=/path/to/rootfs-riscv64-purecap
bash host/cheribsd/build.sh /tmp/capstone/cheribsd/apr
bash host/cheribsd/run.sh /tmp/capstone/cheribsd/apr /tmp/capstone/apr-cheribsd-run-1 \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" --image /path/to/cheribsd.img
```

The CMake library target is `APR::Pools`; `--client /absolute/path/main.c`
links your own `bin/allocator-client` through it. `cmake --preset cheribsd` is
an equivalent configure entry point.

**This is the stock build, and that is its point.** Nodes come from the
platform's own `malloc` and go back through its own `free`, exactly as upstream
APR does (`src/cheribsd/node-malloc.c`); there is no payload region and no
adapter authority. libc revocation is therefore asked the question at the
level where it lives — and APR never calls `free()` on the path where a
destroyed pool's node is reused, so it is never asked. Mode 1 is refused. There
is no PoisonCap build of APR.

`bin/revocation-control` is the positive control that makes a completing stock
arm mean something: it frees a block, sweeps, and reads through the old pointer
at the corpus's own labelled `clbu`. With `--runtime-revocation on` it must
fault there; with `off` it must complete. A stock arm that completes beside a
control that faults says "the mechanism is active and did not fire", which is a
different sentence from "there is no mechanism".

The [httpd/APR corpus](../../../../../bug-corpora/httpd/apr-pool-repros/runners/cheribsd/README.md)
runs its case through this build.
