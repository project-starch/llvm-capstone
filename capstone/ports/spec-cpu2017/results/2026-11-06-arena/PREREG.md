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
