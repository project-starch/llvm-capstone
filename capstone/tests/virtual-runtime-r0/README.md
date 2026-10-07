# Virtual C runtime: Linux execution gate

This is the [R0 execution adapter](../../docs/plans/virtual-capstone-runtime-r0.md).
The Linux core and firmware receive no additional patch. A module mints initial
capabilities using the developer prototype's CSMINT, then enters a virtual
C-mode context through the existing QEMU supervision engine's new CSRUNV form.
It is a bounded experimental adapter, not the normal `.dom` launcher.

`probe.c` creates private anonymous mappings in its own Linux address space.
It touches backing pages and remaps a distant page beside the first. The module
checks the actual physical frame numbers; the positive gate requires that the
two adjacent virtual pages have nonadjacent physical frames. The application
accesses both pages, its separate stack and TLS. The test then checks independent
capability and page-table denial of writes and execution, with exact fault PCs.
A fresh context must work after the negative cases.

The application leaves a linear capability in its stack. The adapter discards
the context and clears every registered page's tags while preserving bytes.
The next invocation must reject the saved bytes as untagged. Omitting that
teardown step must fail the gate. Each invocation has a fresh node table;
this cleanup is essential before identities can restart at 1.

Finally, `entry.c` is compiled by the existing Capstone C compiler. Its actual
capability stack spills and pointer arithmetic execute through the same module
and mappings. It is a small freestanding C function, not a libc application.
The gate does not establish malloc/free, delegated syscalls or recoverable
page faults. An ECALL event is the explicit completion convention here.

## Build and run

Use a prepared kernel tree/output matching the **booted** Image. Scratch output
must remain outside the repository. Set paths for the Linux cross compiler and
the existing Capstone toolchain:

```sh
export KERNEL_BUILD=/path/to/prepared/linux
export CROSS_COMPILE=/path/to/riscv64-linux-gnu-
export CAPSTONE_LLVM_BIN=/path/to/capstone-toolchain/bin
export CAPSTONE_CLANG="$CAPSTONE_LLVM_BIN/clang"
export CAPSTONE_LD_LLD="$CAPSTONE_LLVM_BIN/ld.lld"
export CAPSTONE_OBJCOPY="$CAPSTONE_LLVM_BIN/llvm-objcopy"
source capstone/tests/capstone-test-env.sh
out="$CAPSTONE_TMP_ROOT/virtual-runtime-r0"
capstone/tests/virtual-runtime-r0/build.sh "$out"
python3 capstone/tests/virtual-runtime-r0/run.py \
  --qemu capstone/capstone-qemu/build/qemu-system-riscv64 \
  --images /path/to/unchanged/guest/images \
  --module "$out/module/runtime_r0.ko" --probe "$out/probe" \
  --c-entry "$out/c-entry.bin" --record "$out/result.json"
```

Build QEMU from this branch with `--target-list=riscv64-softmmu --disable-docs`.
Networking is unnecessary for this gate; `--enable-slirp` is useful for the
separate physical application regression through `capstone_vm`.

The runner explicitly disables Sv48/Sv57 and uses one hart. It takes the common
QEMU lock, mounts a scratch disk, requires all eight verdicts and module
unloading, and records hashes of every guest input. `--omit-probe` must fail.
Serial output remains in scratch space. The tracked result contains only
verdicts and hashes.

The [recorded run](result.json) passes 8/8 Linux cases. The missing-probe
control and the module mutation that omits teardown tag clearing both fail.
The same QEMU binary also passes the 21-case instruction gate, 60 M1 cases,
69 U-access cases and two existing physical mruby applications.

See the QEMU [instruction gate](../../capstone-qemu/tests/virtual-capstone-runtime/README.md)
for register consumption, revocation, cached PCC checks, quantum resumption and
physical-caller state restoration. These checks are stronger than a successful
Linux boot; neither is a qualification of the final processor or general OS ABI.
