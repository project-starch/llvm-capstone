# PostgreSQL on CheriBSD

Build a direct-link library, [standalone example](../../examples/contexts.c) and
allocator replay using the [shared CheriBSD workflow](../../../../common/host/cheribsd/README.md).
From this component directory, after sourcing the repository test environment:

```sh
export CHERI_SDK=/path/to/sdk
export CHERI_SYSROOT=/path/to/rootfs-riscv64-purecap
bash host/cheribsd/build.sh /tmp/capstone/cheribsd/postgres
bash host/cheribsd/run.sh /tmp/capstone/cheribsd/postgres /tmp/capstone/postgres-cheribsd-run-1 \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" --image /path/to/cheribsd.img
```

The CMake library target is `PostgreSQL::MemoryContexts`. Add `--client /absolute/path/main.c`
to the build script to link your own `bin/allocator-client` through that target.
The example shows initialization, ownership and the public allocator calls.
`cmake --preset cheribsd` is an equivalent configure entry point.

The manager uses the existing 16-byte capability layout for AllocSet free-list links and chunk headers. The standalone library supplies backend compatibility and printf functions. AllocSet, Generation, Slab and Bump examples run through their real context APIs. The CheriBSD replay entry avoids glibc-only allocation interposition and reports logical operations and payload checks; x86 backing counts are not an ABI-independent oracle.

This is a single-threaded component port. The surrounding libc's revocation
policy is checked separately by the runner; enabling it does not add Sublet's
inner-lifetime semantics. Keep native recording validation and target replay
validation explicit when using these builds for a comparison.
