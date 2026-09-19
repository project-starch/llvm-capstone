# Whisper ggml contexts on CheriBSD

Build a direct-link library, [standalone example](../../examples/context.c) and
allocator replay using the [shared CheriBSD workflow](../../../../common/host/cheribsd/README.md).
From this component directory, after sourcing the repository test environment:

```sh
export CHERI_SDK=/path/to/sdk
export CHERI_SYSROOT=/path/to/rootfs-riscv64-purecap
bash host/cheribsd/build.sh /tmp/capstone/cheribsd/whisper
bash host/cheribsd/run.sh /tmp/capstone/cheribsd/whisper /tmp/capstone/whisper-cheribsd-run-1 \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" --image /path/to/cheribsd.img
```

The CMake library target is `Whisper::GgmlContext`. Add `--client /absolute/path/main.c`
to the build script to link your own `bin/allocator-client` through that target.
The example shows initialization, ownership and the public allocator calls.
`cmake --preset cheribsd` is an equivalent configure entry point.

The library contains the extracted context allocator, not inference kernels. The example exercises owned and borrowed backing storage, object allocation, reset and free. Context reset does not revoke old aliases in this CheriBSD integration. Replays account for object-header growth under the capability ABI rather than silently retaining native layouts.

This is a single-threaded component port. The surrounding libc's revocation
policy is checked separately by the runner; enabling it does not add Sublet's
inner-lifetime semantics. Keep native recording validation and target replay
validation explicit when using these builds for a comparison.
