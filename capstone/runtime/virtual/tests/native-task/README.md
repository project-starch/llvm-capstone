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


## Syscall path and direct-register measurement

[Full factorial record](syscall-comparison.json): 128 valid profile windows across
eight fresh guests, two observations per entry/ABI/case/fast-path setting.
All return-count checks and output-byte checks pass, with no kernel diagnostics.
`build-syscalls.sh OUTPUT` builds the existing budget fixture for ordinary RV64,
Capstone exchange, forced scalar exchange and direct register calls.
`measure-syscalls.py --help` lists external artifacts; run with `--fast 1` and
`--fast 0` to separate specialised dispatch from the direct argument ABI.
The latter still changes parsing/copy/provider paths, so it is not a pure cost
measurement of register moves alone.

Mean system-wide attempted instructions **per call**, new task backend:

| Operation | Ordinary Linux | Exchange | Registers, generic dispatcher | Registers, specialised paths |
| --- | ---: | ---: | ---: | ---: |
| getpid, forced kernel request | 601 | 10,715 | 5,264 | 1,345 |
| read, 1 byte | 1,697 | 15,908 | 9,603 | 4,205 |
| write, 1 byte | 2,667 | 16,929 | 10,560 | 5,194 |
| read, 64 KiB | 39,996 | 215,131 | 86,462 | 69,969 |
| write, 64 KiB | 80,050 | 293,606 | 126,287 | 113,427 |

The existing direct-register specialised path saves 73.56% for 1-byte read,
69.32% for 1-byte write, 67.48% for 64 KiB read and 61.37% for 64 KiB write
against exchange on this backend. Kernel work for 1-byte read falls from
13,845 to 4,175 attempts (about 70%); most of the reduction is kernel work,
with application marshalling also disappearing. Checked copies and bounds/
permission enforcement remain. This is a probe ABI comparison; GCC/musl has
**not** been switched wholesale to that ABI.

Important scalar control: ordinary Capstone musl `getpid()` uses launch metadata,
measuring 255 instructions and **zero kernel instructions** here. Calling
`syscall(SYS_getpid)` would still meet that SDK shortcut. The forced-exchange
probe calls `__capstone_delegate_ints` to bypass the shortcut deliberately.
The 10,715-to-1,345 comparison is therefore a transport comparison, not an
argument to replace cached getpid: preserving that cache is cheaper still.

The task-entry change by itself remains a cost: the direct getpid path takes
609 attempts through the old ioctl service loop versus 1,345 through task return
with userspace accounting. The cheap repeated service loop need not exit its
ioctl on every request. These are separate effects: direct argument dispatch
helps substantially; changing ownership/return integration alone does not.

Small cases use 100 calls per window; 64 KiB cases use one call. Buffer pages
are touched before profiling, stdin is a prefilled pipe, writes target a tmp
file, and setup/diagnostic output are outside the window. Counts include loop
and wrappers, plus system-wide interrupt work. Only two repeats per cell were
run; they are preliminary instruction counts, not latency/cycle estimates.
The ordinary static RV64 reference uses the same patched kernel, not stock
Linux with a separately matched libc/compiler. Preserve those distinctions
when using the table to choose the next optimisation.

Syscall logs, profiles, input hashes and probes:
`$CAPSTONE_TMP_ROOT/native-task/syscall-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/syscall-evidence.tar.gz`), SHA-256
`91b3e92a7fb0eeec926ba9c04fadba93cb34d3e432cf87211565b315e18ac00f`. The kernel/model artifacts are the same as the task GCC record.
