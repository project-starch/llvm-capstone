# North star: a Capstone domain costs what a plain Linux process costs

Status: TARGET, adopted by the lead on 2026-11-03 ("north star metric is what we want"). It fixes where the virtual
profile is going and how progress is measured. It is not a ruling of the ISA (isa.md stays the reference) and it
does not authorise any Linux or ISA change by itself: every step below that needs one is decided separately, on its
own measurements.

## 1. The metric

For each benchmark, the same program on the same inputs, Capstone virtual profile (B) against plain RISC-V Linux
(A), in QEMU instruction attempts (capprof, SPEC CPU2017 test inputs). From `prof-buckets.py A B` form three sums
per arm:

- P = the program buckets (`program: compiled code`, `malloc`, `libc mem*`, `libc other`, `SDK runtime`);
- H = the excluded buckets (`kernel: idle`, `kernel: net/ssh`, `kernel: fs/9p/virtio`, `kernel: proc/printk`,
  `profrun`, `other` = unattributed);
- K = every other bucket: the remaining kernel buckets, `module`, `launcher`, `firmware`, `job-helper` and
  `ambiguous launcher/job-helper`.

Then **program overhead = (P_B - P_A) / (P_A + K_A)** and **boundary overhead = (K_B - K_A) / (P_A + K_A)**. Report
H_A and H_B beside them. H is excluded by convention because it is often harness-dominated; buckets classify by
function name, not by cause, so this is an accounting proxy, not a causal separation (the 9p bucket includes the
application's own file I/O, and the kept scheduler and lock buckets can carry harness noise).

North stars:
- **Boundary overhead: zero AVOIDABLE overhead.** What remains is the necessary cost listed in section 4 (page volume
  of 16-byte pointers, capability state on entry and exit, tag-preserving page copies, lifetime-namespace cloning and
  teardown, image validation, syscall argument checks, revocation synchronisation), each measured on its own.
- **Program overhead: the cost of checked capability code**, reached by compiler and runtime work.

### 1.1 Where we are (census3: dev 0d29c79b85e7, p12 `~/prof-dev/runs/census3/cap-dev3`, platform-dev3)

| benchmark | program overhead | boundary overhead | H_A / H_B | plain run (A) |
|---|---|---|---|---|
| 502.gcc_r | +34.0 % | +55.7 % | 0.75 M / 1.94 M | census3-rv64check |
| 538.imagick_r | +3.4 % | withheld | 16.4 M / 5.6 M | census1 |
| 519.lbm_r | +2.6 % | +3.8 % | 44.3 M / 27.5 M | census1 |
| 544.nab_r | +6.8 % | +0.1 % | 2.12 M / 2.47 M | census1 |

The plain runs for imagick, lbm and nab are census1's and are PROVISIONAL until matched reruns: for gcc the two plain
runs have identical program counts and differ by -0.36 % in total and +0.31 % in P + K, which supports but does not
establish comparability elsewhere. imagick's boundary figure (-1.4 %) is withheld: its plain run differs from the
Capstone run by several million attempts in scheduler, lock and harness buckets, so the run conditions differ.

**Census 4** (dev 976df58996f3, p12 `~/prof-dev/runs/census4/cap-dev4`; QEMU pin 5f329b5061, which differs from
census3's only in the capprof stop-site plugin; plain arm run at the same time in `census4/rv64`; all eight runs
verdict=PASS). It contains every landing since census3: the sign-extension ABI fix, the cursor rule, the heap record
lookup (lane S) and the slot-record change (lane Y).

| benchmark | program overhead | boundary overhead | H_A / H_B | change since census3 |
|---|---|---|---|---|
| 502.gcc_r | +25.4 % | +49.2 % | 0.62 M / 1.77 M | total attempts -8.1 %; minor faults 2,187 -> 1,955 (plain 1,111) |
| 538.imagick_r | +3.3 % | -2.1 % | 17.5 M / 14.2 M | total +5.75 %, mostly kernel-other and idle noise, cause unresolved |
| 519.lbm_r | +2.9 % | +3.7 % | 39.1 M / 27.2 M | total +0.3 %; compiled code +23 M (the cursor rule) |
| 544.nab_r | +7.1 % | +0.1 % | 2.65 M / 2.60 M | total +0.3 %; compiled code +18 M (the cursor rule) |

imagick's boundary figure is now comparable, because its plain run is from the same batch. The rise of lbm and nab in
compiled code is the cost of the cursor rule (dev efe1d58bc469); the census does not isolate it, so that attribution
is the lane's own estimate (+0.4 % program), not a measurement of this batch.

### 1.2 Per-event costs (busy attempts per unit; p12 lane microbenchmarks `lqmicro`, branch perf/linux-question-micro)

| unit | pinned | tag-aware Linux prototype | plain Linux | source |
|---|---|---|---|---|
| a page touched once | 9,185 | 6,540 | 5,891 | lane N, rf3 platform |
| fork of a 444-resident-page process | 1.94 M | 256 K | 55 K | lane N, rf3 platform |
| a new process (exec) | 6.64 M | 5.20 M | 0.198 M | lane N, rf3 platform |
| a new process (exec), today's dev | 3.50 M | prototype not measured | 0.199 M | lane R, platform-dev3 (dev 1482ee6aa642) |

Units are differences of two measurement windows (lane N: touch = (256 MiB - 4 MiB window) / 64,512 pages; fork
excludes the copy-on-write pages written afterwards, priced separately at 4,206 / 6,419 / 5,743 per page). They are
whole-path costs, not isolated handler prices. Each event's north star is the plain column plus its share of the
necessary cost in section 4.

### 1.3 Cycles

Instruction attempts cannot price a hardware context switch: QEMU counts CSRUNV as one instruction. Decisions that
hinge on it (direct trap delivery, section 3) need a separately specified cycle model of CSRUNV's internal work.
(`-icount shift=0` only makes guest time follow the instruction count; it is not that model.)

## 2. What must never be traded for the metric

The five security invariants of the virtual profile apply, augmented by a kernel-access requirement (5). A change
that improves the metric by weakening any of them is rejected, whatever it buys:

1. No capability is usable after its lifetime ends (immediate revocation).
2. A tag never survives a non-capability write (rule 1, every writer including DMA) or a page's reuse.
3. A domain reaches no memory outside its grants.
4. The kernel and the launcher stay outside the domain's authority, and one process cannot affect another's
   lifetimes.
5. The kernel never writes into memory whose authority was recycled while it worked (no confused deputy). How kernel
   accesses synchronise with revocation must be defined explicitly: holding a mapping prevents reuse of the memory,
   but does not by itself preserve capability liveness for the duration of the access.

Huge pages (THP) are not part of this target: the speedup must be structural.

## 3. The end state (three independent design reviews, 2026-11-03)

All three reviews make the domain a native Linux task. The launcher's step loop, the exchange region and its bounce
copies, the spawner helper and dummy thread, the per-domain seccomp filter and the supervisor quantum go away, and so
do the blanket pins that keep tags in place (ordinary pins for DMA and other Linux operations remain).

| event | end state | needs |
|---|---|---|
| page fault | handled in Linux's own fault path; tags follow pages through COW, migration and compaction and are cleared on free | Linux (tag-aware mm), ISA (raw transfers) |
| syscall | handled by the kernel directly; each capability argument checked for tag, permissions, bounds and liveness, synchronised with revocation for the duration of the copy (section 2.5); mmap returns a capability, and a whole-grant unmap retires it before the pages are released | Linux |
| timer, futex, scheduling | Linux's ordinary paths | Linux |
| signal | delivered by the kernel with the interrupted capability state in a kernel-held frame | Linux, ISA (lossless capability state) |
| fork / exec | a binfmt for domain images; fork clones the lifetime namespace at the same slots and copies tags with the pages | Linux |
| exit | stop the threads, drain outstanding accesses, invalidate the namespace, then free it with the address space; no per-arena retirement or tag-collection sweep (page-reuse tag clearing stays) | Linux |

Signals also close today's known gap: CV_STEP already stops on a pending host signal (capstone_vm.c cv_step_stop),
but no mechanism enters the domain's handler asynchronously in a domain that computes without syscalls
(delegation-signals.md, "The bell, later").

Open, not agreed:
- **Entry.** Two reviews want direct trap delivery from the domain into the kernel with lossless capability state;
  the third keeps the CSRUNV escape and dispatches its events through Linux, adding direct traps last and only if a
  cycle model shows the CSRUNV walk costs at least about 2 % of workload cycles (that review's threshold). This note
  stages it last.
- **Partial unmap, permission changes and replacement mappings** need grant semantics of their own (one review
  proposes mapping-grant identities checked with translation).
- **Swap.** Proposed initial restriction: refuse swap-out of tagged pages (one review; the lane M prototype does
  this). Another review requires eventual tag-preserving swap with protected tag metadata.

ISA changes, each justified by what it removes:

- **Recommended by all three: privileged (S/M-only) raw transfers of a capability granule** (16 bytes plus tag, no
  interpretation, no liveness check, no consumption); a proposed extension, not an existing instruction. Today trusted
  S performs no capability memory access (isa.md Q-23). How integer loads read a tagged granule is unsettled: QEMU
  implements an unruled proposal (isa.md, "Proposed with it and NOT ruled", item 1) under which a load of a tagged
  granule's first doubleword reads the projected address, and another review proposes raw scalar reads. Raw transfers
  must preserve the stored representation whatever that ruling becomes. They are what lets Linux move a tagged page
  or save capability registers without pinning. The lane M prototype, using TEST-ONLY page primitives priced at one
  instruction, measured against its pinned controls: gcc -9.6 %, xz -8.0 %, mcf -2.8 %, perlbench -2.0 % (busy, over
  30 of 32 common passing groups). Those numbers do not establish the gain of real granule transfers; lane AC
  measures that.
- **Granule transfers first, no whole-page instruction for now.** One review explicitly opposes an opaque whole-page
  instruction; the other two use granule loops. Further acceleration is reconsidered only on lane AC's measurements.
- **Direct trap delivery**: see "Entry" above.

## 4. What is necessary (not avoidable), each to be measured on its own

- The extra pages of 16-byte pointers under the retained 128-bit ABI: about 510 of gcc's extra faults on its test
  input (lane W's census: about 480 heap pages plus 27 .bss pages). An ABI floor, not a universal lower bound.
- Capability state saved and restored on every kernel entry and exit (wider registers).
- Tag-preserving page copies and tag clearing on page reuse.
- Cloning the lifetime namespace on fork and invalidating it on exit.
- Validating a domain image at exec.
- Checking capability syscall arguments, and synchronising kernel accesses with revocation.
- Capability code itself (program side). One review estimates the total residual overhead at 5-12 % for pointer- and
  allocator-intensive workloads and 0-2 % for streaming floating-point ones; that is unmeasured, one review's
  estimate, and not directly comparable to section 1's metric.

Everything else in the boundary overhead is design residue and is in scope.

## 5. The path, smallest first

Steps without Linux or ISA changes (module, launcher, SDK, compiler):

1. No thread-identity service exit on every delegated syscall (lane AA).
2. Cheaper process setup (lane T, branch runtime/launcher-fork-server, NOT landed): deferred pthread initialisation.
   The -1.07 % per process first reported for it is within the noise of its four estimates per arm (an independent
   review: Welch t about 1.5); the work it removes sums to about 3 K attempts per process (about 0.1 %), and only
   for processes that never install a signal handler or create a thread (gcc still runs it). A persistent fork
   server is UNRESOLVED: a child that returns into libc instead of exec'ing needs a proven thread, TLS and
   robust-list reset, and a shared helper cannot keep the parent and wait semantics. Process setup is fixed
   structurally by the binfmt of section 3, not by a faster launcher.
3. A lazy or right-sized exchange region; map, unmap and node services handled inside the module.
4. Fewer pages: allocator metadata (lane Y, landed as dev d8e7fc417637: gcc 2,170 -> 1,954 minor faults, -3.07 %
   busy attempts against a same-time control), capability-global initialisation (lane Z).

Experiments that test the principal risks before any production Linux or ISA change (they never land; passing one
establishes only the properties it tests):

- **Confused deputy** (lane AB, branch experiment/in-module-io 1b1ed8682f3b, does not land): serve read and write
  inside the module under a hold, race a free and recycle of the buffer. RESULT: a hold on the buffer's mapping
  prevents the recycled-memory race (the unmap of a range under an in-flight copy is refused with EBUSY, the read
  completes into the still-held mapping, nothing reaches the new arena at the same address; with the hold disabled the
  write lands in the new arena, so the test can see the defect), at about 235 attempts per call. A hold cannot
  protect OBJECT lifetime: when the buffer's capability was revoked but its range is still inside a live, held arena
  (a heap object freed and reused), the kernel's write lands in the new owner and, in one case, in the allocator's own
  metadata. The module validates the arena, not the capability, and the SDK's submission check also accepts a revoked
  capability. Consequence for the end state: range lifetime = a hold, cheap; object lifetime = the kernel needs a
  hardware-checked capability check per copied buffer argument (trusted S performs no capability memory access,
  isa.md Q-23, so this is an ISA-visible requirement), and leases are needed only where the copier is a device. Until
  that exists the temporally safe path is the one that copies back through the caller's capability (it faults), about
  38 % cheaper than today at 64 KiB. Priced (attempts per call): a 1 B pipe read costs 8.8 to 9.9 K today and 3.1 K
  with the in-module direct copy against 1.1 K on plain Linux; a 64 KiB pipe read 154 to 160 K, 43.6 K and 40.5 K.
  The test did not cover interruption by a signal, and a held range makes munmap return EBUSY where Linux would not
  (a deferred unmap is the alternative, unresolved).
- **Tag loss without pins** (lane AC; experimental Linux and QEMU-model code): raw granule transfers in the QEMU
  model, the tag-aware Linux prototype rebuilt on them, a torture test (compaction, COW, two processes) with
  stale-capability, reuse and negative controls.

Still to qualify separately before the end state: namespace independence across processes, partial unmap and
replacement, signals, DMA, and multi-hart revocation.

Then, each decided on its own numbers: tag-aware mm and raw transfers (unpinned domains), in-kernel syscall handling,
kernel signal frames, the binfmt and native task, and last, if a cycle model justifies it, direct trap delivery.

## 6. How to keep score

Rerun the census on each landed baseline, with a same-time plain run, and add a row to 1.1 with the baseline commit.
Each measurement cites its exact run, platform and binary baseline, counting convention and exclusions; each estimate
names its source, assumptions and denominator.
