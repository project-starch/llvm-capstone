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

## Addendum 6 (6 Nov 13:10, before the boot): the image changes to c2ac12, which adds the null-capability memset(0)

The image is **cpugcc_r.c2ac12** (fbea9ed0b8aa7f45): c2ac11 plus `CAPSTONE_MEMSET_CAP_ZERO` (wt-arena 01967cfb14b9).
memset with c = 0 over whole aligned 16-byte granules stores 16-byte null capabilities, which zeroes the bytes and
clears the tag. That is half the stores of 8-byte words.

**QEMU gcc train:** memset 348 → 229 M; program instructions 14.422 → **14.302 G**, −5.6 % against the silicon
image's 15.150 G; specdiff PASS.

**On silicon:**
- R-29 concerns 128-bit loads that follow plain stores. A null stc does not make that shape.
- An stc of x0 has not been read back on silicon outside memcpy's granule path; specdiff and twin equality decide.

**M3 unchanged** (≤ 0.95 × 4,656.1 s). The point estimate is now −260…−320 s, about 4,370 s, 1.24×.

## Addendum 7 (6 Nov 14:40, before the boot): the image changes to c3s1, SHRINKTO for small globals

The image is **cpugcc_r.c3s1** (016dce8d0ad5ee56). It is gcc recompiled from the same replayed compile list with
llvm-c3, which is llvm-c2 plus llvm-capstone perf/shrinkto d06490ba9d30, flags `-capstone-shrinkto-globals
-capstone-shrinkto-end-adjust=1`, and linked with SDK ac12 as c2ac12 was. Workload sources are unchanged.
- **What changes:** a data global up to 2,046 bytes is narrowed with one `shrinkto cap, cap, size+1` instead of
  `li; add; shrink`.
- **Static count:** 78,910 shrinkto and 26,427 shrink, against 135,543 shrink in c2ac12.
- **Why +1:** R-32. The C.5a RTL ends SHRINKTO at cursor + imm - 1. The fix, capstone-ariane a3b7a22c0, is
  cherry-picked onto the next RTL batch, not onto C.5a. With +1 the silicon bounds equal SHRINK's exactly, and on
  QEMU they are one byte wider. **This image is for C.5a only**; the driver flashes C.5a (1fc2cd5aa).
- **The edge case:** with +1, a global ending exactly at the end of gp's region would trap. The image window is
  rounded up to a power of two (runtime/virtual/exec.c `rounded()`), 16 MiB here. The last object
  (.L_MergedGlobals.6) ends at 0xf99930, and the window at 0x1010000.

**QEMU gcc train** (perf/best gtr-c3s1 against xprof gtrAC-c2ac12, kit-profq): program instructions 14.302 →
**13.737 G** (−3.95 %; −9.3 % against the silicon image's 15.150 G); specdiff PASS.

**On silicon:** SHRINKTO has run on silicon only in the R-21/R-42 type probes (it returns), never in compiled code.
If its end differs from R-32's reading, the run faults with rc != 0 or reads one byte wider. It cannot read narrower
and pass silently.

**M3 tightens to run ≤ 0.93 × 4,656.1 s (4,330 s).** Point estimate −400…−470 s, about 4,225 s, **1.20×** Plain's
3,515.5 s. This adds about −140 s to Addendum 6's estimate: −3.95 % of c2ac12's ~3,610 s inside.

Correction to Addendum 7: its "14:40" heading is wrong. It was written and pushed at 14:18 (b71e64845128), before
the boot, as the commit time shows.

## Addendum 8 (6 Nov 15:24, before the boot): the image changes to c3f, the same build on a fixed heap base

**Why.** perlbench-ac failed re/reg_email on silicon today (rest Addendum 3). That is the October realloc exactness
defect: cause 29 at realloc+0x5b4, in the CDERIVE of publish(). This branch's base (86f638759ecf, 26 Oct) lacks
dev's two fixes for it:
- 348d9cf2c29c: a realloc that would keep an unpublishable offset moves the object;
- de7cdacda737: a slot's header is the granule of its whole stride (heap.c and musl patch 0009).

Both are cherry-picked onto perf/arena-cache as e53540bb84de and 0ee4200bfc0d (clean). SDK **ac15** is built from
that head with ac12's switches, and its private musl is re-prepared with the new 0009.

**The image is cpugcc_r.c3f** (5642377943d724c6): c3s1's objects relinked with ac15, nothing recompiled.
- QEMU gcc train (perf/best gtr-c3f): 13.735 G instructions (c3s1: 13.737 G), specdiff PASS.
- The gp-window edge check holds: last object end 0xf99920, window end 0x1010000.
- The exe in the stage is byte-identical to the QEMU-passed one.

**M3 and the point estimate are unchanged** (≤ 0.93 × 4,656.1 s; about 4,225 s).

**Correction to Addendum 4's estimate.** The board's exit counters for gcc train (out-ga-b2) put the outside cost at
about 23 ms per exit: CV_STEP 9.3, between steps 12.9, other ioctls 0.7. The ~17 K fault exits that arena_pin
removes on QEMU may therefore be worth up to −390 s, not −30…−60 s, if fault exits cost as much as the average.
Their split by exit kind was not measured. The pre-registered M3 bound stands; this widens the point estimate
downward, to 3,850…4,225 s.

## Addendum 9 (6 Nov 16:10, before the boot): gcc train moves to board 1

The lead asked to focus on gcc train rather than wait for board 2. Board 2 is still in x264 Plain, whose end cannot be
read from its console.
- **The run.** The same stage (perf/ga2/stages/...-devmk-c2ac7, image c3f 5642377943d724c6, ga2's parameter
  block) now runs on **board 1** as line 3 of perf/rest/b1.list. It runs through that board's running chain: kit
  bkit27-gtq, C.5a, the module and launcher hashes of the ga2 driver.
- **What is preempted.** Board 1's x264-ac boot is stopped mid-run for it. That boot is the old configuration
  without the heap fix (rest Addendum 3).
- **Board 2's ga2 driver** (pid 2024017, waiting, no board call made) is killed. Board 2's sequence now starts from
  x264 Plain's driver.
- **Grading:** `MX_ROOTS=$CAPSTONE_WORK/spec/board/out-rest-b1 ga2-grade.py`. The reference out-ga-b2 is on board 2,
  so M3 becomes a cross-board comparison. Two notes on that:
  - the 5 Nov gcc train pair (gcc train Plain on board 1, Temporal on board 2) took the same step;
  - board-to-board run-time spread has not been measured for gcc train. It is reported beside M3.
- M0-M3 are otherwise unchanged.

## Addendum (6 Nov 16:11, before any board call): the board move is reverted

The lead's correction: x264 Plain has at most about 2 h left on board 2, so gcc train waits for it there.
Arena Addendum 9 and rest Addendum 5 are withdrawn before any board call.
- **Board 2:** the ga2 driver is relaunched unchanged (perf/ga2/gt-run.sh 2) behind x264 Plain's driver, pid
  1655525. b2-seq.sh waits on it again, then R-32, THP mcf and perlbench c3f.
- **Board 1:** as rest Addendum 3. Line 3 of b1.list is x264 c3f again, and x264-preempt.sh is relaunched to stop
  x264-ac once x264 c3f's QEMU gate passes.

## RETRACTION and corrections (6 Nov 17:40), from a claim-auditor pass over today's claims

**RETRACTED: the SHRINKTO edge-trap argument.** Arena Addendum 7 and the dev landing message 0001a099efa3 say the
virtual image window is rounded up to a power of two (runtime/virtual/exec.c `rounded()`), so a global ending at the
end of gp's region cannot occur. That is false for the kits used here:
- kit-profq's and the board kits' launcher is built `-DCAPSTONE_VEXEC_EXACT_IMAGE` and grants the exact image. The
  gtr-c3s1 grant reads `bytes=16302080`, and the board control reads `bytes=405504`.
- Only dev's launcher rounds (kit-devq: `bytes=16777216`). `rounded()` in the old exec.c belongs to the nested spawner.
- The images on the board are safe by MARGIN only, by `shrinkto-edge-check.py` (this folder; negative-tested with a
  stub where the last object ends at the window: TRAPS, rc 1):

  | image | last SHRINKTO-sized object | slack to the window's end |
  |---|---|---|
  | gcc c3f | ends 0xf99920 | 1,760 B |
  | perlbench c3g | ends 0x51fdd0 | 560 B |
  | x264 c3f | ends 0x2dc6a4 | 2,396 B |

- "A QEMU pass rules it out" holds only for the SHRINKTOs that executed.
- Rule from here: every adjust-1 image is checked with shrinkto-edge-check.py before its boot.

**Corrections to the reg_email account** (rest Addenda 3-4):
- The board's own failure cause was NOT observed. The board run has no test.err and no fault line, and its twin of the
  same full suite passes reg_email. What is established is the twin reproduction on reg_email alone: cause 29 in
  realloc's publish derive, 7/8 for the production image and -ac. It is consistent with October's defect, not proven
  to be the board's cause.
- The fix control is not one variable. c3s1 → c3f (same objects) moves the runtime 01967cf → 0ee4200, which carries
  c7d7ca0 besides the two cherry-picks, and its musl differs only in meta.h. AC → c3f also changes compiler and musl.
  1/8 against 0/8 cannot attribute it.
- "HF" is the production devmk image (sdks-dev2mk: wt-perf 86f638759ecf runtime), not the heapfix build. The sweep
  script's docstring is corrected.

**Corrections to the dev comparison:**
- dev's runtime against this branch's is a stack comparison, not one variable. The kits' launchers, modules and
  parameters differ, and so do the SDK configs.
- Its instruction figures count in-context instructions only. Under icount dev's run took 16.41 s against c3f's 14.57 s.
- The one-variable result is PR #306 (dev with and without in-domain reuse, same kit, same objects): entries
  50,991 → 7,698.

**CAPSTONE_DELEGATE_SELF_CACHE:** correct only for contexts created through capstone_virtual_thread_create; a raw
service-3 ecall bypasses the flag. Its threaded path has executed zero times.

## Addendum 10 (6 Nov 18:11, before any board-3 call): gcc train moves to the new board 3

**Board 3** was installed today by the board lane: an RPi 5 host, the same console build and supervision as board 2.
Its resident is None and its store is empty.
- **Tooling.** The driver trees (wt-gate, wt-spec, wt-b2 fpga_console.py; the three board-busy.py) gain board "3":
  its console URL file, the lock /tmp/capstone/.board3.lock, and board 2's shared-board watch. board-env-check.py
  takes the board's URL file by number. Board-3 copies of the drivers live in perf/b3: gt-run-ga2.sh, gt-run-thp.sh,
  board/chain-board.sh and board/spec-board.sh, each identical to the board-2 one except the board-3 case.
- **First flash** (perf/b3/b3-seq.sh):
  - **bitstream 3** (caplifive_vp2c_e7af48457, sha256 d6ed97b6…) non-volatile, then its name read back;
  - this makes the known-good fallback resident and stored before anything new;
  - then perf/b3/gt-run-ga2.sh 3 flashes C.5a (sha256 5b7b1edc…), runs this stage and flashes bitstream 3 back.
- **The run is unchanged:** the c2ac7-named stage with image c3f 5642377943d724c6 (Addendum 8) and the ga2 grader,
  with `MX_ROOTS` set to board 3's out dir.
- **Board-to-board.** The reference is still out-ga-b2. M3 is now a board 3 against board 2 comparison, on the same
  bitstream, kit and design. Their speed difference is unmeasured, so the grader's run-time ratio is reported beside
  it.
- **After gcc** on board 3: the THP mcf runs (thp Addendum 1), moved from board 2.
- **Board 2,** after x264 Plain: R-32, then perlbench c3g. Its ga2 and THP steps are removed.

## Addendum 11 (6 Nov 18:15, before any further board call): board 3's first flash failed; gcc train returns to board 2

**What happened.** perf/b3/b3-seq.sh tried board 3's first flash, bitstream 3:
- the upload was accepted (HTTP 201) and the file was present in the store;
- after power-on and lock, the console's flash_bitstream answered `flash_state: board reported 'error'` at once;
- the first attempt died earlier, in flash-bitstream.py itself: board 3 reports flash_state as a string when nothing
  is resident. The tool now reads that as no name.
- The board is powered off. Nothing ran on it, and its resident and NV state are as before (None).
- This is handed to the board lane.

**gcc train goes back to board 2** behind x264 Plain (driver pid 1655525), unchanged: perf/ga2/gt-run.sh 2.
b2-seq.sh waits on it, then runs R-32, THP mcf and perlbench c3g, as in Addendum 9's revert. Addendum 10's move
applies again once board 3 flashes.

## Addendum 12 (6 Nov 18:23, before any board-3 call): board 3 repaired; Addendum 10's plan is rerun

- **The cause** (board lane): openFPGALoader was never installed on board 3. Its app.log read
  `Flash error: [Errno 2] No such file or directory: 'openFPGALoader'`.
- **The fix:** it is now copied from board 2, sha256 equal, `--detect` rc 0, xc7k325t. The resident is still None.
- **The plan:** board 2's ga2 driver and b2-seq are stopped while waiting, before any board call. Then:
  - perf/b3/b3-seq.sh runs again on board 3: bitstream 3 first, then gcc train c3f, then THP mcf;
  - board 2 after x264 Plain: R-32, then perlbench c3g.
