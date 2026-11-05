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
