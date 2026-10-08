# Development and source map

[Guide](README.md) · Previous: [ISA](isa.md) · Next: [Guarantees](guarantees.md)

Use the matching processor, compiler, runtime and application profile.
The core virtual stack has landed on `dev`. The node-growth review branch
`review/virtual-node-growth` adds paged lifetime storage on top of `dev` and
the build repairs in `review/virtual-build-repairs`. Source links below retain
the original review boundaries; use the checkout's QEMU pin for new builds.

## Repositories and profiles

| Component | Integration/review location |
|---|---|
| Virtual processor | QEMU [#13](https://github.com/project-starch/capstone-qemu/pull/13) → [#14](https://github.com/project-starch/capstone-qemu/pull/14) → [#15](https://github.com/project-starch/capstone-qemu/pull/15) → [#16](https://github.com/project-starch/capstone-qemu/pull/16), merged into `virtual-capstone` |
| Compiler and buffer authority | LLVM [#182](https://github.com/project-starch/llvm-capstone/pull/182) → [#183](https://github.com/project-starch/llvm-capstone/pull/183) |
| Linux adapter, libc VM, pthreads | LLVM [#184](https://github.com/project-starch/llvm-capstone/pull/184) → [#185](https://github.com/project-starch/llvm-capstone/pull/185) → [#191](https://github.com/project-starch/llvm-capstone/pull/191), merged into `dev` |
| Academic ISA amendment | [Transferable spec patch][spec-patch]; publication requires a spec maintainer |

The runtime adapter requires the complete QEMU stack. Its QEMU submodule
pin must follow the final landed processor commit. The existing
`c128-qemu-merge` branch remains the physical integration target.
The [review stack][stack] describes the dependencies; its original draft
labels are historical, since the implementation PRs were subsequently
marked ready for review. Review readiness does not discharge merge gates.

## Build a small application

Run the following in a worktree containing the runtime through the pthread
change, with the matching QEMU and a built Capstone compiler. The kernel
build must match the Linux Image you will boot. Output belongs outside the
source tree. The `/path/to/` values are installation-specific inputs.

```bash
export CAPSTONE_LLVM_BUILD_DIR=/path/to/llvm-build
source capstone/tests/capstone-test-env.sh
export KERNEL_BUILD=/path/to/prepared-linux-build
export CROSS_COMPILE=/path/to/riscv64-linux-gnu-
virtual_out="$CAPSTONE_TMP_ROOT/virtual-guide"
export MUSL_CACHE_ROOT="$virtual_out/musl-source"
musl_src=$(bash capstone/ports/musl-capstone/prepare-musl-capstone.sh | tail -1)

bash capstone/runtime/virtual/build-sdk.sh "$virtual_out" "$musl_src"
bash capstone/runtime/virtual/build-adapter.sh "$virtual_out/adapter"
"$virtual_out/sdk/capstone-cc" -O1 -Icapstone/runtime/include \
    capstone/runtime/virtual/contract.c -o "$virtual_out/contract.dom"
```

Boot the matching guest with one hart and CPU settings
`rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true`.
Copy the generated module, launcher and contract into the guest. From their
guest directory, as root:

```sh
insmod capstone_vm.ko
CAPSTONE_VM_CONTRACT=environment ./capstone-vexec contract.dom ok argument
```

The module is generated at `adapter/module/capstone_vm.ko` and the launcher
at `adapter/capstone-vexec`. The contract expects the environment and arguments
shown above and prints `VIRTUAL_APPLICATION_OK` on success. The device is
root-only in this prototype.
`CAPSTONE_VM_TRACE=1` enables delegated-call tracing. For automated execution,
use the [runtime build/run recipe][runtime-readme] and its `run.py` gate;
that gate also requires SQLite and mruby images. A successful boot or one
successful program is not a substitute for the negative safety cases.

The software interfaces have separate versions: delegated services use
application ABI v2; virtual VM-service v3 images carry `CPONVVM3`. The new
launcher also accepts `CPONVVM2`; v1 virtual images and physical-profile
images are rejected. Rebuild the SDK and application together when changing
the VM-service profile.

## Application ports

| Application | Virtual change | Scope to keep explicit |
|---|---|---|
| SQLite, mruby | Included in [#184](https://github.com/project-starch/llvm-capstone/pull/184) | Recorded workloads; not full upstream suites |
| Perl | [#186](https://github.com/project-starch/llvm-capstone/pull/186) | `PERLD_PROFILE=virtual`; existing build restrictions remain |
| CPython | [#187](https://github.com/project-starch/llvm-capstone/pull/187) | Inner temporal protection requires `CPY_SUBLET_MODE=1` |
| PostgreSQL | [#188](https://github.com/project-starch/llvm-capstone/pull/188) | Single-user backend profile |
| FFmpeg | [#189](https://github.com/project-starch/llvm-capstone/pull/189) | Configured decoder/pool workloads |
| tshark | [#190](https://github.com/project-starch/llvm-capstone/pull/190) | Offline profile and selected wmem adapters |
| Memcached | [#192](https://github.com/project-starch/llvm-capstone/pull/192) | Full server with four workers; outer heap protection |

The CPython/PostgreSQL/FFmpeg/tshark recipes use
`CAPSTONE_APPLICATION_PROFILE=virtual`; consult the [shared build profile][build-profile]
and each leaf PR before building. Separate output/cache identities prevent
accidental physical-object reuse. MicroPython is outside this migration.
Eight configured ports do not imply eight complete upstream test suites or
closure of every nested-allocator safety gap.

## Where to read the implementation

| Concern | Starting point |
|---|---|
| Entry, saved registers, PCC checks and node sweep | QEMU [capstone_supervisor.c][supervisor] |
| Root minting and arena retirement | QEMU [capstone_table.c][table] |
| Guest table, reserved IDs and free list | QEMU [cap_rev_table.c][records] |
| Common forest/list operations | QEMU [cap_rev_tree.c][tree] |
| Memory authority and consuming transfers | QEMU [op_helper.c][helpers] |
| Per-mm owner, pins, faults and ioctls | [module/capstone_vm.c][module] |
| ELF loader, native services and worker loop | [virtual/exec.c][exec] |
| Object lifetimes and heap growth | [virtual/heap.c][heap] and [Sublet primitives][sublet] |
| Public VM and pthread bridge | [mapping.c][mapping], [pthread.c][pthread], [thread.c][thread] |
| Service numbers versus ioctl structures | [vm-abi.h][vm-abi], [wire.h][wire] |

Before adding an OS service, decide which authority crosses the boundary,
which thread owns its temporary state, and what happens if the call blocks
or faults. Before adding a mapping form, define how its tags, saved pointers
and lifetime IDs are retired or moved. These are adapter contracts, not a
reason to reimplement Linux scheduling or allocation.

[stack]: https://github.com/project-starch/llvm-capstone/blob/e0313149630450c1a906cc3506b317add64266e0/capstone/docs/plans/virtual-capstone-pr-stack.md
[spec-patch]: https://github.com/project-starch/llvm-capstone/blob/e0313149630450c1a906cc3506b317add64266e0/capstone/docs/plans/virtual-capstone-isa.patch
[runtime-readme]: https://github.com/project-starch/llvm-capstone/blob/518a5b805aa70c222ef0496c151b2845b4ab8dae/capstone/runtime/virtual/README.md#build-and-run
[build-profile]: https://github.com/project-starch/llvm-capstone/blob/518a5b805aa70c222ef0496c151b2845b4ab8dae/capstone/ports/common/application/README.md#virtual-source-builds
[supervisor]: https://github.com/project-starch/capstone-qemu/blob/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9/target/riscv/capstone_supervisor.c
[table]: https://github.com/project-starch/capstone-qemu/blob/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9/target/riscv/capstone_table.c
[records]: https://github.com/project-starch/capstone-qemu/blob/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9/target/riscv/cap_rev_table.c
[tree]: https://github.com/project-starch/capstone-qemu/blob/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9/target/riscv/cap_rev_tree.c
[helpers]: https://github.com/project-starch/capstone-qemu/blob/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9/target/riscv/op_helper.c
[module]: https://github.com/project-starch/llvm-capstone/blob/518a5b805aa70c222ef0496c151b2845b4ab8dae/capstone/runtime/virtual/module/capstone_vm.c
[exec]: https://github.com/project-starch/llvm-capstone/blob/518a5b805aa70c222ef0496c151b2845b4ab8dae/capstone/runtime/virtual/exec.c
[heap]: https://github.com/project-starch/llvm-capstone/blob/518a5b805aa70c222ef0496c151b2845b4ab8dae/capstone/runtime/virtual/heap.c
[sublet]: https://github.com/project-starch/llvm-capstone/blob/518a5b805aa70c222ef0496c151b2845b4ab8dae/capstone/runtime/include/sublet/sublet.h
[mapping]: https://github.com/project-starch/llvm-capstone/blob/518a5b805aa70c222ef0496c151b2845b4ab8dae/capstone/runtime/virtual/mapping.c
[pthread]: https://github.com/project-starch/llvm-capstone/blob/518a5b805aa70c222ef0496c151b2845b4ab8dae/capstone/runtime/virtual/pthread.c
[thread]: https://github.com/project-starch/llvm-capstone/blob/518a5b805aa70c222ef0496c151b2845b4ab8dae/capstone/runtime/virtual/thread.c
[vm-abi]: https://github.com/project-starch/llvm-capstone/blob/518a5b805aa70c222ef0496c151b2845b4ab8dae/capstone/runtime/virtual/vm-abi.h
[wire]: https://github.com/project-starch/llvm-capstone/blob/518a5b805aa70c222ef0496c151b2845b4ab8dae/capstone/runtime/virtual/wire.h
