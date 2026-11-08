# kfast: serving the hottest delegated calls inside the module (measurement prototype, QEMU)

Branch `perf/kernel-fastpath`, based on `perf/board-on-dev` 08d74206e5df (dev c9611344eeb7 + the eight
module-only perf commits). That base, not dev's, is the module kit-unq actually runs: kit-unq's
`capstone_vm.ko` (be4c814f) is `perf/unify/qun`, built from `perf/board-on-dev`; a rebuild of that commit
here gives a module with identical `.text`, `.data` and `.rodata`, and the identical launcher 2488e5d2. The
perl and gcc stages write `step_budget_ns`, `resume_quantum` and `retire_exit_sweep`, which only that module has.

## Question

On the board one delegated-call exit costs ~500 k cycles (~68 k instructions at IPC ~0.13). Of that, how many
instructions per call are the launcher round trip, i.e. how many would a module-side service of the hottest calls
remove, measured on QEMU, before any board time is spent on it?

## Design (`capstone_vm.c`, kfast_serve; `wire.h` CV_KFAST, CV_CAUSE_KFAST_PUBLISH)

Default off (`kfast=0`). With `kfast=1`, a kind-3 `CV_SERVICE_THREAD_DELEGATE` escape of the first context that is
read, write, lseek, fstat or clock_gettime is served inside the same CV_STEP and resumed with the action-2
completion the launcher's reply makes. The authority argument is in the comment above kfast_serve: the module
serves a strict subset of the requests the launcher would serve successfully, and returns every other request to
the launcher unchanged, so every refusal and error is still the launcher's.

## Pre-registered prediction (written before the capmix runs)

* kfast=0: every output hash except the two `.err` files (which carry the launcher's timing line) equal to
  kit-unq's run (`perf/lprof/out-perlnd4`), specdiff 2/2, and the same number of kind-3 exits per command.
* kfast=1: the same hashes and specdiff, `kfast_served` > 0 for read, write, lseek and fstat, kind-3 exits that
  reach the launcher fewer by exactly `kfast_served`.
* Instructions saved per served call (module + launcher + kernel, from capmix): between 5 k and 12 k. QEMU's
  measured ~3 k module and ~5 k launcher instructions per exit, minus kfast_serve's own few hundred, plus the
  ioctl entry/exit and CV_STEP prologue the call no longer makes. A saving under 3 k would mean the launcher
  round trip is not where the per-exit cost lies.
* Coverage: perl test's children write to pipes (backticks), which kfast does not serve, so the served fraction
  of all delegated calls is uncertain; predicted 30-70 %.

## Step 1: which delegated calls perl test makes (kit-unq's module, this branch's launcher with CAPSTONE_VM_NRHIST)

`perf/kfast/out-hist1-perl` (specdiff 2/2 PASS). test.pl: 317 launcher processes and contexts, 112,152 kind-3
exits, of which 105,239 are delegated calls the launcher served (the rest are mapping and thread services):

| nr | call | count | share |
|---:|---|---:|---:|
| 64 | write | 24,823 | 23.6 % |
| 63 | read | 15,800 | 15.0 % |
| 79 | newfstatat | 12,517 | 11.9 % |
| 62 | lseek | 10,216 | 9.7 % |
| r5 | sigaction (runtime request) | 9,875 | 9.4 % |
| r4 | ioctl with a buffer (TCGETS and the like) | 7,254 | 6.9 % |
| 80 | fstat | 6,698 | 6.4 % |
| 57 | close | 5,775 | 5.5 % |
| 56 | openat | 5,733 | 5.4 % |
| 25 | fcntl | 3,197 | 3.0 % |
| 135 | rt_sigprocmask | 1,591 | 1.5 % |
| 113 | clock_gettime | 349 | 0.3 % |

The five served calls are 57,886 of the 105,239 (55 %) before the file-shape restriction. newfstatat (a path
lookup; vfs_statx is not exported), sigaction (the launcher's disposition table and a kernel rt_sigaction) and the
buffer ioctl are the next candidates and are not served.
