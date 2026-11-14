# The virtual profile's runtime and libc pieces

This directory holds the application side of the virtual profile: the entry
code, the libc's mapping and thread calls, the heap glue, the link script, the
SDK build and the libc contract programs. Protected programs will run as
ordinary Linux tasks through traps
([plan v5](../../docs/plans/2026-11-13-kernel-trap-entry.md)), on the 63-bit
format only.

## Status (2026-11-14)

The Linux adapter is deleted: the loadable module `capstone_vm.ko`, the
trusted launcher `capstone-vexec`, its ioctl ABI and every runner, gate and
result that drove them. The physical route, the 30-bit, 55-bit and flat
formats, CSMINT per mapping and the virtual-C (CSRUNV) route went with them.
The tag `pre-trap-route-2026-11-14` keeps the sources, the qualification
records and the earlier version of this README. The module's rules for R, the
blocks, the kernel-owned flag and the last-page CREVOKE move into the kernel in
plan v5 step 6, which takes them from that tag. The page-tag kernel patch and
the headers the kernel needs are described in
[capstone/kernel/patches](../../kernel/patches/README.md).

Nothing in this directory runs today: the programs build, and they run again
once the kernel of plan v5 loads them. Until plan v5 step 5 rewrites the
libc's system-call path, the entry code and the libc still use the adapter's
service selectors in `vm-abi.h` and the internal interface in `vm.h`.

## Files

| File | Role |
|---|---|
| `start.S` | entry code: takes the grants, initializes capabilities, service stubs |
| `vm-abi.h`, `vm.h` | service selectors and the libc's internal mapping interface (until plan v5 step 5) |
| `mapping.c` | mmap, munmap, mprotect, madvise and mremap of the libc |
| `thread.c`, `pthread.c` | thread creation, exit and futex transport for musl's pthreads |
| `malloc-mapping.c` | musl mallocng over mappings, with a lifetime per object ([representation](mallocng.md)) |
| `fenv.c` | hardware floating-point environment |
| `link.ld` | two load segments: text mapped from the image file, data after it |
| `build-sdk.sh` | builds the libc archive and the SDK (`capstone-cc`) |
| `test-link-layout.sh` | checks the two-segment layout of `link.ld`, with a control that must fail |
| `qemu_lock.py` | the project's QEMU lock (reader-writer with a turnstile) |
| `*-contract.c`, `hosted-noreturn.c` | libc contract programs that plan v5 reuses |

`root-contract.c` is the only test of the kernel-owned flag and of the CREVOKE
that ends a block with its last mapped page. Plan v5 step 6 ports it to the trap
route.

## Build

Use a compiler built from this branch (`clang`, `lld`, `llvm-ar`). Keep all
outputs outside the source tree.

```sh
export CAPSTONE_LLVM_BUILD_DIR=/path/to/llvm-build
source capstone/tests/capstone-test-env.sh
out="$CAPSTONE_TMP_ROOT/virtual-runtime"
export MUSL_CACHE_ROOT="$out/musl-source"
musl=$(bash capstone/ports/musl-capstone/prepare-musl-capstone.sh | tail -1)
bash capstone/runtime/virtual/build-sdk.sh "$out" "$musl"
"$out/sdk/capstone-cc" -O1 -Icapstone/runtime/include \
  capstone/runtime/virtual/contract.c -o "$out/contract.dom"
bash capstone/runtime/virtual/test-link-layout.sh
```

`build-sdk.sh` runs the mallocng policy check of the musl port and accepts the
heap switches `-DCAPSTONE_HEAP_LIFETIMES`, `-DCAPSTONE_HEAP_OBJECT_BOUNDS` and
`-DCAPSTONE_HEAP_SCRUB_FREE`. The host tests of the delegation wire that the
libc links are in `capstone/runtime/tests` (`cmake -S capstone/runtime/tests`).
