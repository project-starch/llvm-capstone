# gcc train with the runtime arena cache: pre-registration

Written and pushed before the board call (6 Nov, 01:20).

## The change (llvm-capstone perf/arena-cache bbf3632ddb56, on the SDK's runtime commit 86f638759ecf)

`CAPSTONE_HEAP_ARENA_CACHE`, a runtime build option, default off. The default build is unchanged.

**What it does.** A released mallocng group mapping stays in the context instead of returning to the kernel:
- each mapping gets a holder lifetime, derived from its grant. The delinearised grant has no MANAGE and cannot
  parent a CREVOKE: cause 27 on the twin;
- the group is the holder's child, and every slot and object capability descends from the group;
- release revokes the group and keeps the mapping;
- a request of the same size takes it back zeroed, under a fresh child. There is no exit and no fresh page;
- bounded: 512 mappings and 64 MiB. Past the bound the release goes to the kernel as before.

**Why.**
- gcc train's 55,858 "system-call" exits are the allocator's own map, unmap and remap requests (QEMU launcher trace).
  Only 156 are delegated system calls.
- Most of its 95.6 K faults are pages of freshly mapped arenas.

**SDK and image.**
- The SDK recipe (c1-gate/build-mk.sh, sdk-dev-T3-mk) rebuilt from wt-perf reproduces the production archives
  byte for byte (605bb14b, d060b8b7, 99836c3d).
- A relink of the recompiled gcc objects with the production SDK reproduces f1e3271e.
- The arena-cache SDK changes only libapplication-runtime.a (32bf86f3… → ac2).
- gcc relinked with it, then stripped as make-stage does: 77cb4b6cc65c8ffa.

**Incident while building this.** sdks-dev2mk/sdk-lifetimes is a symlink. My first variant SDK copied the link, and
its sdk.json edit rewrote the production devmk SDK from 00:53 to 01:10.
- It was restored from the reference rebuild: only the four edited fields differed.
- Verified: a control relink reproduces f1e3271e, and a fresh link carries no cache symbols.
- No image was built with the production SDK in that window: the perlbench and x264 stages were made from earlier exes.

**QEMU** (kit-profq: quiet launcher, all2 params).
- **gcc train.** Exits 152,100 → 28,020, service exits 55,858 → 7,144, faults 95,760 → 19,720, CV_STEPs 56,592 → 8,355.
  Instructions +1.2 %, peak pages 48.7 K → 63.6 K, specdiff PASS.
- **gcc test.** Exits 2,150 → 1,211, specdiff PASS.
- **Gate of these exact stages** (perf/ga/q-test, q-train): q-test PASS; q-train gates the driver.
- **Revocation probe** `uaf.c`: fill one size class, free all, refill, read the stale pointer. The reference and the
  cache arm both fault on the stale read (cause 25); both controls end normally.
  - Limit: with heap lifetimes on, an object's own lifetime is already revoked at free. So this shows the end-to-end
    guarantee holds; it does not isolate the group revoke.
  - A claim-auditor review of the safety argument is owed before anything lands.

## The run

**Board 2,** after perf/mx's Plain mcf (pid 1370451) ends with DONE. Kit bkit27-gtq: bkit26-gt with the quiet launcher.
- List: gcc test-ac, then gcc train-ac.
- The train stage is gccx's: resume_quantum, a 10 s step budget, sweep 3.
- x264 Plain (results/2026-11-06-rest) moves behind it on board 2.

**Reference:** out-gt-b2, 5,562.1 s, same board and module.

| # | Prediction | Point estimate |
|---|---|---|
| M0, M1, X2 | as gccx | |
| M3 | train run ≤ 0.85 × 5,562.1 s and CV_STEPs ≤ 20,000 | 4,000-4,500 s: about 1.15-1.28× Plain's 3,515.5 s. Outside 1,752 s falls with exits (155.7 K → about 30 K) |
| — | gcc test: reported (all2 reading 26.1 s on board 1) | 15-20 s |

**What would refute:** M3 failing with CV_STEPs ≤ 20,000 means the per-exit cost was not what removing exits saves. It
is then the cost per remaining exit, or the 30 % more pages.

**Grader controls** (perf/ga/ctl): out-gt-b2 as the run reads rc 1; a mock at 4,300 s and 9,000 steps reads rc 0; empty
reads rc 2. The dry-run chain stub checks the list paths; board-env-check passes.

## Addendum 1 (6 Nov 02:00): the safety probe, and the first silicon reading

**The claim-auditor review** (02:00) holds the claim with conditions, on QEMU's revocation-tree semantics:
- Slot parents are SHRINK copies that carry the group node.
- Objects and nested groups are CDERIVE descendants, and CREVOKE retires the whole run under the group.
- Nothing the allocator hands out descends from the holder.

It also found two things:
- The earlier probe could not test the claim: the object's own lifetime was already revoked at free.
- The munmap hook is reachable from any munmap, not only the allocator's. It must become an allocator-only release
  entry before landing. Its data race with other threads' munmaps is formal only.

**Experiment A, as the auditor specified it** (`uaf2.c`, transcript `uaf2-qemu.txt`).
- SDKs with heap lifetimes OFF: there a freed object's capability carries the group's lifetime, so only the group
  revoke (or the kernel's retirement) can stop a stale read.
- Two stale pointers: one in a 200-byte class (mapped groups) and one in a 16-byte class (nested groups, whose address
  is reused).

| arm | cache counters | stale 200 B | stale 16 B |
|---|---|---|---|
| L0C0: no cache | 0 | fault, cause 25 | fault, cause 25 |
| L0C1NR: cache with the revoke compiled out (positive control) | puts 148, hits 148 | **read returns 0x00** | **read returns 0x34, the new owner's data** |
| L0C1: cache | puts 148, hits 148 | fault, cause 25 | fault, cause 25 |

So the probe can fire, the cache engaged, and the group revoke ends every stale capability into a reused mapping,
nested groups included. **Limit:** QEMU semantics only. A silicon run of `uaf2` is owed.

**First silicon reading (board 2, 01:40-01:50).**
- gcc test with the cache: **20.86 s**, against 26.15 s and 26.41 s for the same configuration without it on board 1
  (out-all2-b1, out-ns-b1).
- Exits 1,042 against 1,981; rc 0, specdiff PASS, control passed.
- gcc train is running.

## Addendum 2 (6 Nov 11:00): the gcc train reading, and the audit's fix B

**gcc train on board 2: 4,656.1 s against 5,562.1 s** (0.837×, M3 holds; M0, M1, X2 hold).
- That is 1.32× Plain's 3,515.5 s, down from 1.58×, and closes 44 % of the gap.
- Exits 155,699 → 34,563, CV_STEPs 77,246 → 12,191.
- Outside 1,752 → 827 s, against Plain's 650 s.
- Inside 3,829 s (1.005× the reference), so the remaining gap is inside: 15.15 G instructions against Plain's
  10.89 G at the same IPC.
- The point estimate (4,000-4,500 s) was too optimistic. Per remaining exit the outside time is 23.9 ms, against 11.3 ms
  before: the per-second background stays while the exits go.

**Fix B (claim-auditor).** The cache is no longer reached from public munmap.
- musl patch 0011 maps mallocng's munmap to `__capstone_malloc_munmap` (heap.c). That is the only caller of the cache,
  and it always runs under the heap lock.
- mapping.c is back to its base version.
- Built on a private prepared musl tree, perf/arena/musl-ac: the shared musl-cache does not carry 0011. Commits
  perf/arena-cache 848df5e23e18 and a932ff62511d. SDK ac4.
- QEMU after the fix (`uaf2c-qemu.txt`): the same table as Addendum 1.
  - Positive control: reads stale data, 0x00 and 0x34.
  - Cache arm: both stale reads fault, cause 25.
  - Counters: puts 148, hits 148.
- gcc train: specdiff PASS, exits 27,887.

**Before landing:**
- the virtual-runtime QEMU suites, since patch 0011 changes every image's libc call;
- the uaf2 probe on silicon.

## Addendum 3 (6 Nov 12:05): gcc train, arena cache + llvm-c2 + the simple slot hash + no exit sweep

Lead: gcc train is the target, test can go; reach 1.20x if possible (−437 s from 4,656.1 s).

**Image `cpugcc_r.c2ac7`** (a0 sha cc02847a10691ef3, stripped as make-stage does). It is built from:
- gcc's objects compiled with llvm-c2, origin/dev 3d2ba7d7176b (the compiler lane's c1-gate/gcc-jt/c2, -fno-jump-tables);
- linked against SDK ac7: wt-arena b3457ffce661, arena cache, slot index of 16 bits, and the index hash the masked
  granule number. The 64-bit multiply cost 8 instructions per call, with about 3 calls per object.

**QEMU gcc train**, all specdiff PASS except where noted:

| variant | instructions |
|---|---|
| arena cache alone (the silicon image's runtime) | 15.150 G |
| + llvm-c2 | 14.973 G |
| + simple hash, 16 bits: **this image** | **14.799 G** (−2.3 %) |
| + a one-record find cache | 14.814 G: no gain, dropped |
| + no zeroing of reused mappings | crashes at 0.9 M instructions: mallocng needs them zero; dropped |

**The run.** Board 2, kit bkit27-gtq, ahead of the THP mcf runs.
- The stage is the ga stage (resume_quantum, 10 s budget) with `retire_exit_sweep=2`, so the exit teardown sweeps
  nothing. That is safe by results/2026-11-05-nosweep N1.
- With the cache, in-run retires are rare: they are spills only, and those take mode 0.

**Reference:** out-ga-b2, 4,656.1 s (same board, kit and stage, except image and sweep).

**Prediction (`ga2-grade.py`):**
- M0/M1/X2 as before, with MX-SET reading sweep 2;
- **M3: run ≤ 0.985 × 4,656.1 s**, point −90…−130 s: inside −2.3 % ≈ −88 s, teardown −30 s.
- Grader control: out-ga-b2 given as the run reads rc 1.

That lands near 1.29×. 1.20× needs the allocator path and the globals' bounds construction (about 1 G instructions
each); neither is ready today.

## Addendum 4 (6 Nov 12:25, before the c2ac7 boot): arena_pin added

The c2ac7 stage also sets `arena_pin=1` for this stage only, and restores it at the end. `arena_pin` is an existing
module parameter, default off, that has never been switched on on silicon. On a fault in a lazy arena it pins the rest
of the arena from the faulting page, not one page.

**QEMU gcc train** (perf/xprof gtrAC-c2ac7-ap against gtrAC-c2ac7):
- fault exits 19,720 → 2,486; entries 27,988 → 11,099;
- peak pinned pages 63,813 → 67,045;
- program instructions unchanged; specdiff PASS.

**M3 tightens:** run ≤ 0.975 × 4,656.1 s, point −130…−180 s. The extra part is about 17 K fewer fault exits at
the ~2.5 ms of non-resolve CV_STEP cost each.

**New risk:** this boot is arena_pin's first on silicon. The control rungs run before it, without it.

## Addendum 5 (6 Nov 12:50, before the boot): the image changes to c2ac11, which adds the slot cache

The stage directory keeps its name (devmk-c2ac7), but its gcc is now **cpugcc_r.c2ac11** (f2a5a63135f1f2e5). It is the
c2ac7 build plus `CAPSTONE_HEAP_SLOT_CACHE`, wt-arena 56bc6010ab17 / a98781264eb8, bound 65,536:
- a freed slot of nominal size ≤ 4 KiB stays allocated in mallocng, after free has validated the pointer and revoked
  the object's lifetime;
- malloc of exactly that size issues a fresh child over it.

**QEMU gcc train**, with arena_pin:

| variant | instructions |
|---|---|
| c2ac7 | 14.799 G |
| slot cache, bound 8,192 | 14.614 G |
| **slot cache, bound 65,536** | **14.422 G** (−4.8 % against the silicon image's 15.150 G) |

- Peak pinned pages 73,792; specdiff PASS.
- **Probe** (uafSC-qemu.txt, lifetimes on): the stale slot is reused at the same address (same-address=1), and the
  stale read faults with cause 25.

**M3 tightens to run ≤ 0.95 × 4,656.1 s.** Point estimate −230…−290 s:
- inside −4.8 % ≈ −184 s;
- teardown −30 s;
- arena_pin −30…−60 s.

That lands around 4,400 s, about 1.25×.
