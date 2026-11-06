# Copyable-only Linux prototype

Branch `prototype/copyable-only` runs ordinary Linux and copyable capability
applications with the legacy ownership/control instruction encodings removed.
Use its matching QEMU, module, launcher and SDK. It has no legacy compatibility
switch. The old Capstone firmware executes CAPENTER and is intentionally refused;
use QEMU's bundled **ordinary OpenSBI fw_dynamic.bin** instead.

The prototype keeps tags, bounds, permissions and managed lifetime revocation.
It removes linear grants and DELIN from the virtual mapping path. The new CV_ADD
payload has a different ioctl number: an old adapter fails rather than silently
changing semantics. The old linear remap transaction returns EOPNOTSUPP; public
mremap retains allocate/copy/retire semantics. Runtime mapping permission checks
and lazy admission are independent of capability type.

The model routes device interrupts through ordinary RISC-V MIP, including while
an application runs. MIP is hart state, not a saved context field. This fixes a
real failure with ordinary OpenSBI: legacy CIS routing lost timer wakeups after
CSRUNV, leaving module unload waiting forever. The gate checks sleep/wakeup and
unload, not just successful program output.

## Build

Source `capstone/tests/capstone-test-env.sh` before builds. Keep output beneath
`$CAPSTONE_TMP_ROOT`; use the branch's QEMU source, not another checkout's binary.
Build QEMU for riscv64-softmmu. The adapter discovers `capstone,copyable-v1` through
the device tree; it uses no test-only feature-probing CSR. This prototype uses the
paged 30-bit lifetime table; the module refuses the flat-table build.

For unpinned backing, apply `../../unpinned/linux-6.1-unpinned.patch` to a private
Linux 6.1 tree, start from the platform kernel's configuration, disable KSM and
TRANSPARENT_HUGEPAGE, and build Image and modules. Build the adapter against that
exact prepared kernel. Do not modify the baseline kernel tree.

With `KERNEL_BUILD`, `CROSS_COMPILE`, `CAPSTONE_LLVM_BUILD_DIR` and `MUSL` set:

```sh
source capstone/tests/capstone-test-env.sh
OUT="$CAPSTONE_TMP_ROOT/copyable-build"
bash capstone/runtime/virtual/build-adapter.sh "$OUT/adapter"
bash capstone/runtime/virtual/build-sdk.sh "$OUT/sdk" "$MUSL"
"$OUT/sdk/sdk/capstone-cc" -O2 -fno-builtin -Icapstone/runtime/include \
  capstone/runtime/virtual/contract.c -o "$OUT/contract.dom"
"${CROSS_COMPILE}gcc" -O2 -static -Wall -Wextra -Werror \
  capstone/runtime/virtual/tests/copyable-only/abi-probe.c -o "$OUT/abi-probe"
```

The measurement compiler is the same cached hoist-fast compiler used by the
preceding allocation experiment; its hash is recorded in results. No compiler
change was needed for these applications. This does not remove all legacy
intrinsics from the compiler or every unreachable legacy helper from QEMU source.
The removed instructions remain illegal at every privilege level.

## Run and qualify

`run.sh` takes QEMU, an images directory (Image/rootfs.ext2), the adapter build,
contract.dom, abi-probe, a **new** output directory, and pinned/unpinned:

```sh
bash capstone/runtime/virtual/tests/copyable-only/run.sh \
  "$QEMU" "$IMAGES" "$OUT/adapter" "$OUT/contract.dom" "$OUT/abi-probe" \
  "$OUT/unpinned-check" unpinned
```

It selects ordinary OpenSBI, raw tag support, and disables test CSRs. Set
COPYABLE_FIRMWARE only to an explicitly qualified ordinary firmware. Run both
modes on the same patched kernel. The checker requires eight positive cases,
eight expected-fault cases with exact causes, ABI refusal, timer wakeup and module
unload. It rejects the previous hanging run and wrong fault causes.

Additional gates already exercised:

- QEMU `tests/copyable-only/run.sh`: 45 precise illegal-instruction checks plus
  rejected-linear and successful-copyable mint checks.
- `mm-gate.sh` with the existing unpinned mm-probe module, in a guest containing
  no protected application: tag-preserving copy, refusing live-tag
  acknowledgement, refusing tagged swap, and zero tags at same-frame kernel reuse.
- Existing `unpinned/run-irq-probe.py --icount --expect-lost none
  --expect-retries any --expect-window inside`: 4,000 copies, 2,707 deadlines
  inside copies, 2,501 retries, zero lost tags with ordinary OpenSBI.
- `movement-gate.sh`: build an adapter with CAPSTONE_VM_TORTURE=1 and the existing
  unpinned torture.c and pfns.c fixtures. Stage them as capstone-vexec,
  capstone_vm.ko, torture.dom and pfns. The test-only launcher forks sleeping
  children to force COW. Four live rounds and two stale-pointer rounds pass.
  PFN samples observe 9,902 heap and 1,133 BSS frame changes (observations, not
  distinct pages). Compaction is requested too; this result does not isolate
  migration from COW. This is not protected fork support.

## Instruction comparison

`allocation.c` is byte-for-byte the workload from inspect/musl-differences,
metadata-break/workload.c. Compile with `-O2 -fno-builtin -DMETADATA_PROFILE`.
Stage it as allocation.dom with the production adapter and measure-gate.sh.
Build QEMU's contrib/plugins/capprof.c, then run run-staged.py with `--icount 0`,
`--plugin libcapprof.so` and `--plugin-arg out=NEW_DIRECTORY/capprof`.
Use the same patched kernel, ordinary firmware and CPU properties as run.sh.
The gate runs three alternating pairs, reversing pair 2. Windows exclude startup
and output. Profiling does not include the COW torture hook.

`report.py --out DIRECTORY --platform PLATFORM --program allocation.dom` consumes
DIRECTORY/run/serial.log and DIRECTORY/capprof. PLATFORM contains kernel/System.map,
adapter/{capstone-vexec,capstone-job,module/capstone_vm.ko} and the existing profrun
helper. GNU nm is sufficient. Each window uses that module load's section
addresses; reloading the module cannot silently corrupt attribution.

| Median instruction attempts | Pinned | Unpinned |
|---|---:|---:|
| Program | 21,662,083 | 21,662,083 |
| Linux kernel | 183,294,679 | 161,645,788 |
| Module | 18,742,062 | 13,020,588 |
| Total counted | 226,118,941 | 198,791,518 |
| pin_user_pages_remote entry attempts | 14,515 | 0 |

Total reduction: **12.085%**. Each arm has three passing workload oracles.
Pinned total range: 226,109,804–226,545,441; unpinned:
198,679,903–198,828,522. There are 11,496 ambiguous launcher/job-helper attempts
per run, and one unpinned run has 7,804 other attempts; both remain in the totals.
Rows are independently computed medians and need not sum exactly.

This isolates pinning within the new copyable-only stack. It is not a measured
speedup from deleting linearity alone, and it is not directly comparable to the
older firmware/kernel configuration's totals. These are QEMU instruction attempts,
not hardware cycles, tag-cache costs, or elapsed-time measurements.

## Limits

QEMU prototype only; no RTL implementation. One hart; tagged swap refused;
KSM/THP disabled; supervisor frames still pinned. Raw tag transport and the Linux
hooks remain experimental. General shared-memory authority transfer, protected
fork, DMA ownership and complete legacy-source deletion are not implemented.
A narrowed pointer does not establish exclusive ownership; allocator disjointness
and initialization at isolation boundaries remain separate responsibilities.

Raw serial logs, profiling windows, attribution inputs and key guest binaries are
archived locally at `$CAPSTONE_TMP_ROOT/copyable-only/evidence.tar.gz`, SHA-256
`223049dab82760bef85c8f249a6ea5a77a39194642dd5eb3ab7e63de37763b68`.
The compact [qualification](results/qualification.json) and
[comparison](results/comparison.json) records include hashes and per-run counts.
