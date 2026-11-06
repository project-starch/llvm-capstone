# Experimental Linux task-return backend

Branch `native/linux-task-integration`. This is a working first implementation,
not completion of the full [acceptance plan](../../../../docs/plans/native-linux-task-integration.md).
The [transition contract](../../../../docs/design/virtual-capstone/native-linux-tasks.md#linux-61-first-backend-transition-contract)
explains context tracking, exec failure and synchronous lifetime rules.

The kernel owns the protected task context and mm namespace. `.dom` exec installs
no scalar trampoline mapping or context FD. Architecture return enters CSRUNV
and loops through Linux return work after each transition. Existing checked-copy,
exchange ABI, confinement and providers remain. No linear ownership instructions
are introduced. The model's 5 ms quantum supplies escape opportunities; this is
not direct hardware timer delivery. Caught handlers, full restart semantics,
clone/exec services, SMP and native audit/seccomp/tracing remain outside this slice.

## Build and activation

Use a disposable Linux 6.1 tree with this branch's
`../../unpinned/linux-6.1-unpinned.patch`, `../../kernel-native-mprotect.patch`
and `../../kernel-native-gcc.patch` already applied, plus its prepared platform
configuration. Matching QEMU is pinned at `dd041bac5ad9c145c3872a50804ca714a26b1c88`.
Do not apply the experimental patch to the measured baseline tree.

```bash
source capstone/tests/capstone-test-env.sh
# KERNEL_BUILD: disposable prepared Linux tree; CROSS_COMPILE: Linux compiler prefix.
bash capstone/runtime/virtual/install-native-task.sh "$KERNEL_BUILD"
make -C "$KERNEL_BUILD" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" olddefconfig
make -C "$KERNEL_BUILD" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" -j12 Image
```

The installer applies the architecture patch and stages matching runtime sources
as a built-in driver. It enables `RISCV_CAPSTONE_TASK`, disables SMP and selects
generic virtual CPU accounting with forced user context tracking. The Kconfig
option defaults off and execution additionally requires `capstone_vm.native_task=1`.
For an external matching QEMU source checkout, set `CAPSTONE_QEMU_SOURCE` before
staging. Kernel test variants use `make CV_TEST_HOOKS=1 CV_RACE_TEST=1 Image`.
These deliberate fault/race hooks are absent from the ordinary build.

Boot arguments (no insmod):

```
capstone_vm.native_task=1 capstone_vm.dom_binfmt=1
capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1
capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1
capstone_vm.supervisor_slots=1
```

Use the shared `../../run-staged.py`, ordinary OpenSBI, 4 GiB guest memory and
matching QEMU CPU properties:

```
x-capstone-supervisor-slots=1,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true
```

Build these fixtures with `build.sh OUTPUT`, setting `CAPSTONE_NATIVE_CC` to the
copyable SDK compiler and `CROSS_COMPILE` to the Linux compiler prefix. Build
current `../native-integration` probes with its build.sh. Stage `gate.sh`,
`compute.dom`, `task-driver`, `probe.dom`, `blocked.dom`, `entry-driver` and the
external GCC artifacts `gcc-test.dom`, `t1.c`, `reference.s` in one directory.
GCC is SPEC 502.gcc_r test/t1.c, `-O3 -finline-limit=50000`; no suite is vendored.
Require `NATIVE_TASK_EXIT:0`, zero fixture failures and the exact-output marker;
run-staged's completion marker alone is not a passing oracle.

For deliberate admission refusal, use the test kernel, write `1` to
`/sys/module/capstone_vm/parameters/task_refuse_once`, then execute `compute.dom`.
It must die with SIGSEGV; subsequent ordinary qualification must still pass.
The invalid action raises an S-origin exception before domain activation,
exercising the early context-tracking recovery and exception-table fixup.

For the existing race suite, use **32 slots** in both kernel and QEMU; its
injected racer requires an additional context. Apply each row of
`../native-race/scenarios.txt`, run `pipe-fill race.dom` with that row's arguments,
and use the unchanged `../native-race/judge.py` oracle. One-slot blocking/task
transport is tested separately by `entry-driver` and `task-driver`.

## Verified evidence

[Qualification record](qualification.json): 51 ABI/authority/protection checks,
10 entry/blocking checks, compute STOP/CONT/KILL, ptrace refusal, absence of
trampoline/control FD, twelve repeated blocked-task kills, admission-exception
recovery, reference-exact GCC, and all 41 copy-race scenarios (17 negative
controls). Compute fixture reports 0.501139 seconds user time and zero rounded
system time. Scalar Linux tools continue to run. The external-module variant also
builds; shared native-core C90 declaration warnings remain.

The initial stale probe binary expected ENOSYS for a now-confined pathname;
the rebuilt current source passes all 51 checks. The initial one-slot race suite
refused its racer setup; this was not counted as passing. Full register/FP/bounds
transport matrices, all raw-platform negatives, exhaustive leak/failed-exec race
injection and the remaining Linux policy integration gates are still open.

## Matched preliminary GCC comparison

`measure-gcc.py` takes paths to QEMU, Image/System.map/rootfs, firmware, capprof,
profrun, GCC, input and reference; see `--help`. It runs fresh guests in
ioctl/task/task/ioctl order. The only mode change is `capstone_vm.native_task`;
both use the same built-in kernel, config, binary, input and output oracle.

[Four-run record](gcc-comparison.json), mean attempted instructions:

| Entry | Total | Kernel including built-in adapter |
| --- | ---: | ---: |
| Existing ioctl | 50,928,947.5 | 26,954,892 |
| Architecture task return | 51,037,165.5 | 27,063,220.5 |

Task return is **0.2125% more total work**, about **0.4019% more kernel work**
in this small test. Other work is effectively unchanged. This first version
returns through Linux work checks after every transition and accounts domain
execution as userspace; it does not yet optimize/batch those transitions.
The result supports the architectural change, not a performance win. Do not
compare these absolute counts directly with older 54M runs: this kernel has
changed SMP/accounting configuration and builds the adapter in-tree.

Raw logs, profiles, kernel images/configuration and Argon review are archived at
`$CAPSTONE_TMP_ROOT/native-task/native-task-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/native-task-evidence.tar.gz`).
SHA-256: `b8286ed6a5252441d0589fa1a9631fc20b468d508a7e064a797cbd19047008fd`.
Astra xhigh reviewed entry/context tracking, task lifetime and ptrace/exec races;
Argon independently reviewed the patch. Review findings were checked against
Linux 6.1 source; neither review substitutes for the runtime gates.
