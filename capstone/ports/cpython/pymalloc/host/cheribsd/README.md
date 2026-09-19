# CPython pymalloc on CheriBSD

Build a direct-link library, [standalone example](../../examples/pymalloc.c) and
allocator replay using the [shared CheriBSD workflow](../../../../common/host/cheribsd/README.md).
From this component directory, after sourcing the repository test environment:

```sh
export CHERI_SDK=/path/to/sdk
export CHERI_SYSROOT=/path/to/rootfs-riscv64-purecap
bash host/cheribsd/build.sh /tmp/capstone/cheribsd/cpython
bash host/cheribsd/run.sh /tmp/capstone/cheribsd/cpython /tmp/capstone/cpython-cheribsd-run-1 \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" --image /path/to/cheribsd.img
```

The CMake library target is `CPython::Pymalloc`. Add `--client /absolute/path/main.c`
to the build script to link your own `bin/allocator-client` through that target.
The example shows initialization, ownership and the public allocator calls.
`cmake --preset cheribsd` is an equivalent configure entry point.

The port uses the capability-preserving arena/pool patch and the real 16-byte pointer size. The unmodified native reference is not compiled for purecap: its pointer-size assumptions are not a valid CHERI reference. The example checks small allocations, zeroed storage, raw fallback and a capability-bearing realloc. Arena and pool frees do not automatically invalidate retained aliases.

This is a single-threaded component port. The surrounding libc's revocation
policy is checked separately by the runner; enabling it does not add Sublet's
inner-lifetime semantics. Keep native recording validation and target replay
validation explicit when using these builds for a comparison.

The experimental [PoisonCap workflow](poisoncap/README.md) uses the published
platform and an explicitly selected inner-lifetime adapter. Its mode 1,
synchronous sweep policy and separate controls do not change this standard
CheriBSD build's protection scope.
