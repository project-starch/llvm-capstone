# FFmpeg's own pools under Sublet leases, inside the whole program (QEMU)

**Why this exists.** The heap arms (`../2026-09-23-qemu-safety/`) protect `malloc`/`free`.
FFmpeg's own pools — `AVBufferPool` and the refstruct pools — return memory to the pool and
reissue it **without** calling `free`, so no heap arm sees their lifetimes, and those pools are
what the nested-allocators work is about. The buffer-pool port protects them (a Sublet lease per
get, revoked on return), but only in a replay of recorded pool calls. This folder puts that
protection inside the whole running decoder.

**Verdict (QEMU only).**

1. **Correctness holds with the pools leased.** On `pool2`, `AVBufferPool` buffers and **every
   refstruct object, pooled or not** (patch 0002 routes all of `av_refstruct_alloc_ext` through
   the payload allocator) live in payload blocks under a Sublet lease per get, revoked on return,
   over the revoking `sublet` heap. Every pool2 M1–M5 boot that completed (5 of 10; the rest
   stalled in the guest before any result, see the instrument notes) is bit-identical to native
   (30/30) with the flipped control firing. M5's payload allocator took **406 leases** and made
   406 revocations (returns and frees, including leases re-taken at teardown in order to free),
   over 43 payload blocks; the same counts in every completed run.
2. **Every pool fixture came out as pre-registered** (14/14; predictions committed and pushed at
   `456fa82` before the pool arms were first built):
   - `pool0` (bounds only): a pool buffer read after its return reads its old byte; after the
     pool reissues it to a new get, the stale pointer reads the **new owner's** byte; a stale
     refstruct unref is **accepted** and drops the live owner's reference, so a third get
     aliases a live object. The same holds for refstruct pool objects.
   - `pool2`: each of those stale reads **faults** on capstone-qemu (the revoked lease reloads
     untagged and the touch's pointer arithmetic faults), and the stale unref is **refused** by
     the pool code (`ff2_fail 304`) before anything is touched, the program then exiting cleanly
     with that code. The faults are the lease's, not the heap's: pool0 runs the same heap and
     returns; and in pool2 fixture 13's register dump the old lease is untagged while the new
     owner's capability at the SAME address is tagged, which only lease revocation produces.
   - both: one past a pool buffer faults on bounds; so does one byte below a refstruct object,
     which shows the object's capability is bounded exactly to it. That the `RefCount` is out of
     band holds by construction (patch 0002 allocates it separately); the fault alone does not
     show where the header is, and the matching control on a stock build was not run.
3. **The two arms differ only in the pool mode**: the M images differ in two `li` immediates (the
   mode passed to `ff2_set_mode` and the one printed); the fixture images in one, or (fixtures 7
   and 16) in which register holds that constant.
4. **Emulator evidence, as before**: the temporal faults are capstone-qemu untagging a reloaded
   revoked capability; the deployed silicon lets a stale data access retire (ISSUES Q-11,
   measurements §7r). This ABI's `gp` is fabricated by QEMU.

## How the pools are protected (reused, not re-derived)

| piece | from | role |
|---|---|---|
| `patches/ffmpeg-9.0.1-0001`, `-0002` | `../../buffer-pool/` (unchanged) | event hooks, and the payload lifetime hooks: pool buffers and refstruct objects live in payload blocks; refstruct's `RefCount` moves out of band |
| `patches-pool/0001-capstone-pool-payload-only-for-pool-buffers.patch` | this port | the replay routed `av_buffer_alloc`/`av_buffer_default_free` themselves through the payload allocator; in a whole program that also catches every non-pool buffer (demuxed packets, `av_buffer_realloc`), which fails on the first. Only `pool_alloc_buffer` takes a payload block now |
| `shared/pool-allocator.c`, `capstone-domain/payload-capabilities.c`, `allocators/sublet/pool-leases.c` | buffer-pool (unchanged) | the payload allocator and its Capstone/Sublet backend |
| `src/pool/ff2_record.c`, `ff2_observe.c` | this port | stand-ins for the replay's recorder and observer: no-op events and lock; a pool refusal prints `FFAPP-POOL fail <code>` and exits with it |
| `src/capstone-domain/ffapp_pool.h` | this port | sets the mode, hands the payload region (the host's fourth shared region, 4 MiB, transferred linear) to the backend before FFmpeg runs; reports the pool's Sublet counts |

Build: `FFAPP_HEAP=sublet FFAPP_POOL=0|2 host/build-domain.sh` (source from
`host/prepare-source.sh --pool`, its own FFmpeg build directory). The native builds and every
earlier arm are untouched: the level0 run-of-record images still rebuild byte-identical (7/7
against `../2026-09-23-qemu-m1-m5/SHA256SUMS`).

## Fixtures 11–17 (`src/capstone-domain/ffapp_safety.c`), through FFmpeg's real API

| # | fixture | pool0 | pool2 |
|---|---|---|---|
| 11 | a pool buffer's capability length | returns, **len 64** | returns, **len 64** |
| 12 | pool buffer read after `av_buffer_unref` returned it | returns **0xa0** (its old byte) | **temporal fault** at the touch |
| 13 | ... after the pool reissued it (same address) and the new owner wrote 0x5b | returns **0x5b** | **temporal fault** at the touch |
| 14 | one past a 64-byte pool buffer | **OOB fault** | **OOB fault** |
| 15 | refstruct pool object read after `av_refstruct_unref` returned it | returns **0xa0** | **temporal fault** at the touch |
| 16 | one byte below a refstruct object (not a pool object) | **OOB fault**: the capability is bounded exactly to the object | **OOB fault** |
| 17 | stale refstruct unref after the entry went to a new owner | **accepted**: a third get aliases the live object, which reads the aliasing owner's 0x77 | **refused**, `FFAPP-POOL fail 304`, clean exit with 304 |

## M1–M5 per pool arm

| arm | M5 frames vs native | control | pool (M5) | heap (M5) |
|---|---|---|---|---|
| pool0 | **30/30 MATCH** | fires | 43 blocks, 315,072 B payload, no leases | 721 alloc/free, peak 155 |
| pool2 | **30/30 MATCH** | fires | 43 blocks, **406 leases taken, 406 revocations**, mrev 482, split 43 | 721 alloc/free, peak 155 |

Revocation nodes per pool2 decode: heap 1,375 (split 327 + mrev 1,048) + pools 525 (43 + 482)
= **1,900**, 2.9% of the deployed bitstream's per-boot pool (65,532; ISSUES R-12).

## What this does not establish

1. **The board** (verdict 4).
2. **Performance.** QEMU timing means nothing; the counts above are the cost this port can state.
3. **Every pool, and the allocation itself.** The fixtures drive `AVBufferPool` and the refstruct
   pool with 64-byte objects; the decode exercised the real pools (406 leases) but only this
   workload's. FFmpeg's pool bookkeeping runs unchanged, but the payload ALLOCATION is the port's:
   `pool->alloc` is never called, and a pool with a custom allocator is refused (`ff2_fail`
   125/126); none occurs on this decode path.
4. **Repeat counts.** Fixture cells N = 1 in round 2; fixtures 13 and 15 also faulted in round 1
   (N = 2), pool2 fixture 12 only in round 2. M1–M5 matched in every completed run (pool2 5 of 5,
   pool0 4 of 4). The M1–M5 boots write no per-boot hash sidecar; their images are the current
   builds (unchanged since the pool rebuild, and the booted share copy of M5 hashes the same).

## Instrument notes and defects found

- **The exit path crashed: ISSUES C-56, a second consumer.** The first pool run printed the
  refusal, and then the domain jumped to the run-time address of link address 0 (image base −
  0x10000; cause 2), ending the emulator, so fixture 12, queued after it, never ran. This was 3 of
  3 attempts on one image, rebuilt byte-exact by the audit. Cause:
  - `hostcall.c` guards its exit hook with `if (__capstone_at_exit)`, where the hook is an
    UNDEFINED weak function;
  - these domains run position-independent with no load-time relocation, so the address of an
    undefined weak symbol is not NULL.
  This is **C-56**, already recorded (found a day earlier through CPython's `exit()`) and fixed
  on the runtime branches `runtime/c56-weak-at-exit` and `runtime/at-exit-hook`; neither is on
  `dev` yet. This port had searched only `dev`'s registry. The pool arms define the hook
  (`ffapp_pool.h`), and with it the same image exits cleanly with 304.
  - **Merge hazard for whoever lands C-56:** its runner check refuses images that carry an
    undefined weak symbol. This port's level0, shrink and sublet images carry
    `w __capstone_at_exit`, so that check must land together with the `hostcall.c` fix.
  - **Classifier:** it had scored the crashing run's fixture 17 AS PREDICTED from the refusal line
    alone. It now requires a clean exit with the refusal code, and reclassifies that run as
    `POOLFAIL-THEN-HALT`. Round 2 re-ran every pool cell.
- **9p read stalls.** Of 10 pool2 M1–M5 attempts, 5 matched, 2 stalled before login, and 3 stalled
  reading from the 9p share:
  - two stalls came inside the loader's page faults on the image, which it mmaps from the share;
    in one, the previous image (M1) had done no pool work at all;
  - one came in a plain `cp` from the share **before any domain ran**.

  So the stall is neither the pool's nor the domain's. pool0 completed 4 of 4; 0/4 against 3/8 is
  within chance. The runners now copy the images into the guest's `/tmp` first, which moves a
  stall ahead of the experiment instead of into it; 2 runs have completed that way. The stall's
  own cause is UNRESOLVED: QEMU's virtio-9p or the guest.
- **Classifier additions for the pool arms**, made before any pool run and negative-tested on
  doctored logs: a revoked capability held in a register faults as "access on revoked capability"
  (matched by cursor to the printed target); a pool refusal is `FFAPP-POOL fail <code>`.
