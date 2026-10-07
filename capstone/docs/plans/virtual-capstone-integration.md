# Virtual Capstone integration baseline

`virtual-capstone-integration` is the development branch containing the complete
virtual review stack and the documentation guide on current `dev`. It is a
combined test and development base; the individual PRs retain their review
boundaries. Creating this branch does not land those PRs on `dev`.

## Source identity

- Integration merge: `9e07f38a782c63e6920902e014ec6f58fb42e6c4`.
- `dev` base: `6bad3708476812ab12e6f85f1a24f8bf0b08eb82`.
- Application build repairs: `087c140739e16d71845ef5e6ada8911cc4244104`.
- QEMU: `4e271895a88e0e6a963bbf216809b65cf3674cf8`, on
  [`virtual-capstone`](https://github.com/project-starch/capstone-qemu/tree/virtual-capstone).
  This is the merged QEMU PR stack; its source tree equals the reviewed
  `9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9` tree.
- Buildroot: `8fd1ea1249a99a16ffe40886161e9a55831fdbdf` and RTL:
  `776d9d85900e0b7f7fd71b5369a5827af1bb715e`, both retained from `dev`.

The merge includes LLVM PRs
[#182–#193](https://github.com/project-starch/llvm-capstone/pulls?q=is%3Apr+virtual)
through the compiler, delegated authority, adapter, VM, pthread, application and
guide branch heads. The eight applications are SQLite, mruby, Perl, CPython,
PostgreSQL, FFmpeg, tshark and memcached. MicroPython is excluded.

Start with the [system guide](../design/virtual-capstone/README.md) for the
ownership model, processor interface, trusted Linux integration and limits.

## Continue development

Create feature branches from the integration branch until the review stack has
landed on `dev`. Keep processor changes in the QEMU repository, based on its
`virtual-capstone` branch. Push QEMU commits before updating the superproject
gitlink. Do not move the physical QEMU lane or the RTL pin as an incidental
part of virtual development.

```sh
git fetch origin
git -c submodule.recurse=false worktree add -b virtual-next \
  ../llvm-capstone-virtual-next origin/virtual-capstone-integration
cd ../llvm-capstone-virtual-next
git submodule update --init capstone/capstone-qemu capstone/caplifive-buildroot
```

When the PRs land, compare the resulting `dev` tree and pins with this baseline
before moving development there. A merge or squash commit hash alone does not
establish that the tested source is present.
The integration repairs in `087c140739e1` must also land after the original
stack; merging only the original PR heads omits those repairs.

## Build environment

All outputs and fetched application sources belong outside the repository.
Set these paths for the local machine, then source the test environment:

```sh
export CAPSTONE_REPO_ROOT=$PWD
export CAPSTONE_LLVM_BUILD_DIR=/path/to/qualified-llvm-build
export CAPSTONE_LLVM_BIN=$CAPSTONE_LLVM_BUILD_DIR/bin
export CAPSTONE_BUILDROOT_DIR=/path/to/buildroot-with-linux-cross-toolchain
export CAPSTONE_QEMU_BINARY=/path/to/qemu-build/qemu-system-riscv64
export KERNEL_BUILD=/path/to/prepared-kernel-matching-Image
export CROSS_COMPILE=/path/to/riscv64-buildroot-linux-gnu-
export VIRTUAL_INTEGRATION_OUT=/tmp/capstone/virtual-integration
export VIRTUAL_INTEGRATION_IMAGES=/path/to/platform-images
export MUSL_CACHE_ROOT=$VIRTUAL_INTEGRATION_OUT/musl-source
export JOBS=4 CPY_JOBS=4 FFAPP_JOBS=4 MUSL_SURVEY_JOBS=4
source capstone/tests/capstone-test-env.sh
```

`VIRTUAL_INTEGRATION_IMAGES` contains `Image`, `fw_jump.elf` and `rootfs.ext2`.
The module must be built against the kernel that produced that `Image`.
The integration record identifies the reused platform images by hash.
The physical launcher additionally needs the **pinned** Buildroot driver
headers, even when its compiler comes from another Buildroot output directory.

QEMU was freshly configured with `--target-list=riscv64-softmmu --disable-docs
--disable-gtk --disable-sdl --disable-werror --enable-slirp` and built with Ninja.
The full application and virtual qualification first used a build without
slirp. The development binary adds that backend for the SSH-based physical VM
driver; its processor source is identical. All three instruction gates and a
39-check virtual compatibility gate were rerun on the development binary.
The manifest distinguishes the two executable hashes and their test scopes.
The existing
LLVM compiler was reused after verifying equality of the `llvm`, `clang`,
`lld` and `compiler-rt` trees with the reviewed compiler source and rerunning
the seven focused compiler tests. This is not a fresh full LLVM build.
The toolchain freshness script still reports an incomplete discovery set
(`llvm-capstone`); its partial result alone is not the compiler provenance check.

Build a fresh virtual SDK and Linux adapter:

```sh
musl_source=$(bash capstone/ports/musl-capstone/prepare-musl-capstone.sh | tail -1)
bash capstone/runtime/virtual/build-sdk.sh \
  "$VIRTUAL_INTEGRATION_OUT/virtual" "$musl_source"
bash capstone/runtime/virtual/build-adapter.sh "$VIRTUAL_INTEGRATION_OUT/adapter"
sdk=$VIRTUAL_INTEGRATION_OUT/virtual/sdk
"$sdk/capstone-cc" -O1 -Icapstone/runtime/include \
  capstone/runtime/virtual/contract.c -o "$VIRTUAL_INTEGRATION_OUT/contract.dom"
"$sdk/capstone-cc" -O1 -Icapstone/runtime/include \
  capstone/runtime/virtual/pthread-contract.c -o "$VIRTUAL_INTEGRATION_OUT/pthread.dom"
```

Application recipes:

- SQLite: `ports/sqlite/app/prepare-sources.sh`, then `build-domain.sh` with the
  fresh SDK, prepared source directory and output directory.
- mruby: `ports/mruby/app/build-mruby-domain.sh`, with `MRBD_ROOT` in scratch,
  `MRBD_SDK` pointing at the fresh SDK and `MRBD_PIN=4.0.0-rc2`.
- Perl: `ports/perl/musl/build-perl-domain.sh`, with `PERLD_PROFILE=virtual` and
  a fresh `PERLD_ROOT`. This integration uses the recipe's default `-O2`;
  the older virtual Perl corpus used `-O1`.
- CPython, PostgreSQL, FFmpeg and tshark:
  `ports/common/application/build-virtual.sh APP NEW_OUTPUT`. These rebuild
  application objects for virtual execution. For tshark, put Meson on `PATH`
  before building its dependencies.
- memcached: `ports/memcached/app/host/build-virtual.sh NEW_OUTPUT`; this includes
  libevent, native controls, the server, marker and safety variants.

All recipe paths above are relative to `capstone/`. Do not relink old physical
objects to claim a virtual source build. Specialized inner-allocator variants
retain their documented source recipes and separate qualification.

Two build repairs were needed when combining with newer `dev`: SQLite's
source adapter now receives directories and its adapted header is retained;
PostgreSQL publishes `link/backend-objects.txt`, which the common relinker uses
to include every enabled static module. An older prepared PostgreSQL tree must
rerun `build-domain.sh` with `PGSU_FROM=link` before using the common relinker.

## Qualification

The [integration results](virtual-capstone-integration-results.json) record
test outcomes and exact input hashes. Raw compiler and guest logs remain in
scratch storage. Reused native fixtures and platform binaries are distinguished
from freshly built target images.

The 2026-10-07 qualification covers all eight freshly built target ports:

| Gate | Result | Scope |
|---|---|---|
| Compiler | 7/7 | Focused lit tests, reused source-matched compiler |
| QEMU | 60/60, 69/69, 33/33; 3/3 bounds | Foundation, access, virtual context and physical bounds |
| Node model | 8,949 prefixes per backend | Host and guest list versus independent forest; mutation controls detected |
| Host runtime | 23/23 Python, 20/20 heap, 87/87 ASan/UBSan | Includes delegated-message thread isolation and its rejected shared-buffer mutation |
| Virtual runtime | 46/46 | SQLite persistence, mruby, Perl smoke, both 200,000-allocation recycling cases |
| Pthreads | 9/9 | Private epoll arm also passes; shared-buffer mutation fails its isolation assertion |
| Four application workloads | 16/16 | CPython, PostgreSQL, FFmpeg and tshark; native oracles and negative controls |
| Memcached | 73/73 | Four workers and 20 applicable outer-heap fixtures |
| FFmpeg / tshark safety | 13/13 / 15/15 | Selected outer-heap fixtures, including expected surviving defects |
| Perl corpus | 11 replayed; 10 unchanged | One historical outcome difference, detailed below |
| Physical guest regression | Pending | Fresh artifacts prepared; full context/thread suites have not run |

Perl case `254b30e378` now faults with cause 24 at the default `-O2`, matching
its registered protected-allocator oracle. The older `-O1` record returned
status 30 without a capability fault. Repeating the original aliasing trigger
reproduces cause 24; changing its source to an independent copy exits normally
with the expected output. This paired control uses the same current binary.
It does not establish that optimization alone caused the historical difference.
The other corpus outcomes, including non-detections, remain recorded explicitly.

Configured dependency tests retain their upstream exclusions and observations.
In particular, libevent's monotonic fallback timing case passed all six required
isolated reruns after one failure; libgcrypt's lock test is an expected failure
with its dependency's threading disabled. These are not blanket upstream-suite
passes. The result manifest records the remaining dependency scope.

The physical guest regression is **not qualified** by this record. The shared
QEMU test slot was occupied by an independent benchmark before the corrected
gate could run. A first attempt used the virtual stock firmware, which lacks
the physical managed-monitor service; a subsequent attempt with the physical
monitor reached the application but omitted the contract's required stdin.
Neither is a passing application gate. Native runtime tests and the three QEMU
physical-bounds checks passed, but do not substitute for the guest suites.

To complete that regression, use the prepared physical launcher, driver and
contract images with a physical process-monitor firmware. The virtual gate's
stock firmware is not interchangeable with that monitor. Supply a RISC-V
Dropbear multi-call executable through `capstone-vm up --ssh-server` when the
rootfs has no SSH server; this test setup uses a statically linked build of
Dropbear 2022.83. Use the matching kernel, a CMA reserve, and the network-enabled
QEMU build described above. Boot through `capstone-vm`, then run:

```sh
export PYTHONPATH=$CAPSTONE_REPO_ROOT/capstone/runtime/host
physical_state=$VIRTUAL_INTEGRATION_OUT/physical-vm
physical_images=$VIRTUAL_INTEGRATION_OUT/physical-contracts
printf 'input\n' | python3 -m capstone_vm --state "$physical_state" run \
  --cwd /tmp -e 'CAPSTONE_CONTRACT=environment with spaces' \
  --result "$VIRTUAL_INTEGRATION_OUT/physical-contract.json" \
  /mnt/host/application-contract.dom healthy '' 'argument with spaces
and newline'
python3 capstone/runtime/tests/application/run-context.py \
  --state "$physical_state" --host-image "$physical_images/context-probe.dom" \
  --nm "$CAPSTONE_LLVM_BIN/llvm-nm" \
  --report "$VIRTUAL_INTEGRATION_OUT/physical-context.json"
for suite in thread pthread; do
  python3 capstone/runtime/tests/application/run-threads.py \
    --state "$physical_state" --suite "$suite" \
    --host-image "$physical_images/$suite-probe.dom" \
    --nm "$CAPSTONE_LLVM_BIN/llvm-nm" \
    --report "$VIRTUAL_INTEGRATION_OUT/physical-$suite.json"
done
```

Only a completed healthy contract and passing reports qualify these suites.
Record their actual firmware and executable hashes alongside the outcome.

Use the existing gates:

- QEMU: `tests/virtual-capstone-m1/run.sh`,
  `tests/trusted-linux-u-access/run.sh`, `tests/virtual-capstone-runtime/run.sh`
  and `tests/virtual-capstone-model/model.py` in the QEMU checkout. Set
  `VIRTUAL_CAPSTONE_QEMU_BINARY` and `TRUSTED_LINUX_QEMU_BINARY` explicitly.
- Host: runtime host unittests, heap qualification unittests and the CTest
  suite from `runtime/exec`, including delegated-message thread isolation.
- Linux virtual runtime: `runtime/virtual/run.py`, with all three SQLite,
  mruby and Perl inputs, the Perl smoke script, a physical rejection control,
  and recycling enabled. Allow sufficient time for both 200,000-allocation
  cases.
- Pthreads: `runtime/virtual/run-pthreads.py`; the controlled epoll mutation
  uses the same inserted scheduling delay on private and shared buffer arms.
- Application comparison: `runtime/virtual/run-ports.py`, with the fresh four
  target images and the pinned native/fixture inputs.
- Memcached: `ports/memcached/app/host/run-virtual.py`.
- Registered FFmpeg/tshark safety expectations:
  `runtime/virtual/run-safety.py`.
- Physical contexts and threads: `runtime/tests/application/run-context.py`
  and `run-threads.py` against a separately provisioned physical guest.

All guest runners serialize through the shared QEMU lock. Do not work around
that lock to run multiple guests concurrently.

The scope remains one hart, trusted Linux, process-local capability ownership
and the documented virtual mapping policies. Matching a safety fixture's
expected surviving bug is an observed gap, not a successful prevention. This
integration does not establish RTL behavior, multicore safety, unrestricted
POSIX compatibility or protection for every nested allocator lifetime.
