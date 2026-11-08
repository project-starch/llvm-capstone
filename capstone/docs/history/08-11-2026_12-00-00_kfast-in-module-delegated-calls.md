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

## RETRACTION (lane only): f5a84c4692a3 claimed a strict subset; O_DIRECT broke it

An adversarial audit of f5a84c4692a3 refuted "serves only a strict subset of what the launcher serves
successfully". The launcher performs read and write on its malloc'd BOUNCE copy, the module on the exchange
region. O_DIRECT's alignment check reads the buffer address (fs/iomap/direct-io.c: `bdev_iter_is_aligned`,
dma_alignment 511), and ext2 is the ext4 driver here, so the same O_DIRECT request could fail on one path and
succeed on the other. Fixed in b38986258226: an O_DIRECT descriptor falls back. The same audit found two
smaller defects, fixed in the same commit:
- the f_pos lock was taken before the file-shape filter;
- a signal that became pending after the stop check was performed rather than retried.

What would have caught it: checking which inputs of the copied syscall body are ADDRESSES, not only values. The
equivalence was argued over offsets and lengths, and O_DIRECT is the one regular-file mode where the address
itself matters. The probe's O_DIRECT case (forged 512-byte reads at five offsets of an 8 KiB file) is the
positive control, run against both f5a84c4692a3 and the final module (perf/kfast/out-probe3-old,
out-probe3-new).

## First measurement (f5a84c4692a3, perl test, capmix shift 4, whole boot)

`perf/kfast/out-k0-perl` (kfast=0) and `out-k1-perl` (kfast=1):
- **Correctness.** Both runs give specdiff 2/2 PASS with 0 FAILED lines in test.out. 558 of 560 output hashes are equal between them, and equal to kit-unq's `perf/lprof/out-perlnd4`; the two that differ are test.err and makerand.err, which carry the launcher's timing line.
- **Served calls.** kfast=1 served 32,376 of the 105,239 delegated calls: read 14,289, lseek 9,028, fstat 6,672, write 2,041, clock_gettime 346. The launcher served the remaining 72,863, so the two add up to the kfast=0 total exactly.
- **Fall-backs.** 23,290 for the file shape (mostly writes to pipes) and 2,732 for a stop condition. 27 performed calls were handed back.
- **Exits.** CV_STEP returns fell from 112,975 to 80,579.

Instruction delta (kfast=1 minus kfast=0) per served call, by class (perf/kfast/classify.py):

| class | per served call |
|---|---:|
| launcher (low static range; natperl's code shares it and cancels) | -3,097 |
| kernel, syscall/seccomp/ioctl/VFS paths | -1,904 |
| module, other than kfast | -143 |
| module, kfast's own code (includes the copied VFS bodies) | +1,888 |
| kernel user-access checks (CONFIG_DEBUG_ATOMIC_SLEEP: __might_fault, __might_sleep) | +1,672 |
| net, idle excluded | **-1,584** |
| kernel idle loop (follows host I/O timing; see the null below) | +1,722 |

The gross cost the round trip carried was therefore about 5,100 instructions per call. About 20 user accesses per
call ate a large share of it, and 29768787287b cuts them to four copies.

## Final measurement (f26fffa25c4c) and the null

The kfast=1 runs below are all on perl test; each is compared with two kfast=0 runs, `out-k0-perl`
(f5a84c4692a3) and `out-k3off-perl` (29768787287b). With kfast=0 those two commits execute the same code. All
runs are whole-boot capmix (shift 4), and every figure is per served call.

**The null (k3off against k0), scaled to 32,390 calls.** The idle class moved by +4,807, so it is noise:
QEMU's virtio completions follow the host's timing, and the guest's idle loop counts them. Kernel-other moved
by +700, the module by +34, the launcher by -1, uaccess by -7 and user by +28. The idle class is excluded
below, and kernel-other carries about ±700.

**Per-class results.** Each cell gives the value against k0 / against k3off.

| commit | served | launcher | kernel-other | uaccess | module (all) | net, idle excluded |
|---|---:|---:|---:|---:|---:|---:|
| f5a84c4692a3 (~20 accesses per call) | 32,376 | -3,097 | -1,904 / -2,604 | +1,672 | +1,745 | -1,584 / -2,311 |
| 29768787287b (small copies) | 32,390 | -3,100 | -2,148 / -2,847 | +1,992 | +864 | -2,391 / -3,118 |
| f26fffa25c4c (word accesses) | 32,366 | -3,096 | -2,160 / -2,860 | -276 | +1,517 | **-4,016 / -4,743** |

**What one served call saves on QEMU: about 4.4 k instructions (±0.7 k).** The round trip carried about
5.3 k instructions (the launcher's 3.1 k plus about 2.2-2.9 k of kernel: the ioctl entry and exit, seccomp,
syscall tracing, and a bounce copy), and the module now spends about 1.5 k to replace it. The module's
figure includes the copied VFS bodies, which were kernel instructions before. Over the whole boot the
saving is 130-154 M instructions of 128 G, and 32,366 launcher round trips are removed: 31 % of
test.pl's delegated calls, 29 % of its kind-3 exits.

**Coverage.** Most of the fall-backs are writes to pipes (fb_file 23,294: test.pl captures its children's
output). The rest are calls kfast does not serve (fb_nr 46,853: newfstatat, sigaction, the buffer ioctl,
close, openat) and stop conditions before the call (fb_stop 2,726). Of the performed calls, 7 were handed back
for publication.

**Qualification.** For every module, kfast=0 and kfast=1 give specdiff 2/2 and 0 FAILED lines in test.out, and
558 of 560 hashes are equal to kit-unq's run; the two `.err` files differ by their timing line. gcc train
gives specdiff 1/1 and 4 of 5 hashes equal (the `.err` differs) on kfast=0, f5a84 kfast=1, 29768 kfast=1 and
f26ff kfast=1. gcc train makes only 156 delegated calls (122 of them writev) and kfast served 8, so it
qualifies the change but cannot measure it.

**Board relevance.** On the board one delegated-call exit costs ~68 k instructions. The launcher round trip
measured here accounts for ~5 k of those on QEMU, and kfast removes ~4.4 k of them. The board's remaining
~60 k per exit lie in the CSRUNV escape and resume, the module's CV_STEP path and the kernel. kfast does not
touch any of those, so on silicon it cannot remove more than a small fraction of the per-exit cost, unless
the launcher's share is far larger there than its instruction count. A board run would have to measure that
share first.

## The O_DIRECT control, run (perf/kfast/out-probe7-old, out-probe7-new)

The file is 8 KiB on the writable ext2 root (the guest's /tmp refuses O_DIRECT and /mnt/vm is read-only, and the
probe's gate caught both). The probe opens it with O_DIRECT and forges 512-byte reads at exchange offsets 4096 +
{0, 16, 496, 512, 4096}:

| module | kfast=0 | kfast=1 |
|---|---|---|
| f5a84c4692a3 | -22 -22 512 -22 -22 | **512** -22 **-22 512 512** |
| f26fffa25c4c | -22 -22 512 -22 -22 | -22 -22 512 -22 -22 |

The old module disagreed with the launcher at four of the five offsets, exactly as the audit predicted from the
two buffer addresses. The final module agrees at all five and counts them as fall-backs. Every other probe line
(out-of-range offsets and lengths -14, private descriptors -9, valid requests served) is equal on both settings
for both modules.
