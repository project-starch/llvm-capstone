# Experimental native domain exec

The module parameter `dom_binfmt=1` enables the root-only loader; its default is
zero. This experiment uses the existing SDK v7 images and Linux 6.1 platform.
It has one domain thread, fixed pinned transport and no new seccomp filter.
Inherited filters still apply. It refuses setuid/setgid and mutable remote
filesystem images. Stage binaries on guest tmpfs before executing them.

Supported services: thread-self, anonymous mapping and whole-grant retirement,
node reservation, read/write through the existing exchange ABI, and exit.
Protection changes use a validated 92-byte scalar trampoline because Linux's
mprotect implementation is not exported to modules. The trampoline has no
capability authority and receives only validated arguments from the module.
Other delegated calls return ENOSYS. Default signals, Linux wait and normal
module/context teardown have contract coverage; caught capability signals,
fork, capability ptrace and raw brk authority are unresolved. No production
security qualification is implied.

The scripts use the lane's p12 artifact layout. Export the current runtime and
`capstone/tests/capstone-test-env.sh` under `~/lanes/laneAG/src`, with an empty
`llvm` directory for environment validation. Build the adapter with the matching
prepared kernel, toolchain and revision-table ABI header. Copy platform-dev4
with `cp -rL` into `~/lanes/laneAG/platform-native`, replace its adapter/module,
and refresh the platform manifest hashes. Export this test directory as
`~/lanes/laneAG/contracts`, then run `build-contracts.sh` and `run-contract.sh`.
The latter disables binfmt_misc to prove execution and refusals use the native
loader. It writes results to `contracts-final` and shuts its guest down.

For measurement, compile the unchanged `lqmicro.c` and `lq-hog.c` from branch
`perf/linux-question-micro`, directory
`capstone/ports/spec-cpu2017/tools/linux-micro`, into
`~/lanes/laneAG/micro/{lqmicro-cap,lqmicro-plain,lq-hog}` with the baseline SDK
and plain compiler. `measure.sh` runs three pairs of exec 10/100 per arm. The
parent always uses the plain binary. The plain child is that same binary; the
launcher and native child are the same Capstone binary. Every valid invocation
must have rc=0, bad=0 and exactly one complete profiler window.

`run-micro.sh` derives from that branch's harness. Its additions stage a separate
child and record image/trampoline placement outside the measured windows.
`profile.py` extends the kit profiler with these images and preserves ambiguous
byte matches as an explicit bucket. The native placement is sampled from module
parameters after each window; launcher placement is probed by a separate
unmeasured exec. ASLR is disabled. Per-process cost is `(W100-W10)/90`, with
kernel-idle work reported separately. Historical Capstone-parent numbers use a
different denominator and must not be presented as this experiment's control.
