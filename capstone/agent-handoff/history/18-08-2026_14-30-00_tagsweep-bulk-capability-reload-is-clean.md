# TAGSWEEP: 2.1M capability reloads through DRAM on silicon, zero tags lost

Date: 2026-08-18. Bitstream `caplifive_s06s08fix_s07tag2_618f4ce.bit`. No RTL change.

## Result

A new standalone rung — **10-39 KB, no SQLite** — stores capabilities into memory, reloads them,
and asks each one's type with `lcc` field 1 (total; returns 7 for NOT_CAP **without raising**), so
tag loss is **counted instead of fatal**. Four domains, one boot, control first:

| rung | N slots | reps | checks | retval |
|---|---|---|---|---|
| `k800` | — | — | — | `4` — **control PASS, boot valid** |
| `ts1` | 8 | 1 | 8 | `0xA5000000` |
| `tsml` | 512 (8 KiB, cache-resident) | 2048 | 1,048,576 | `0xA5000000` |
| `tagsweep` | 4096 (64 KiB, exceeds the 32 KiB cache) | 256 | 1,048,576 | `0xA5000000` |

**Zero unseeded tag losses in 2,097,160 reloads.**

## The instrument is PROVEN, which is the only reason the zero means anything

Each board arm deliberately clobbers `SEED = 3` slots per rep with a scalar store, which drops the
granule's tag. The domain returns `TAGSWEEP_OK | (lost - seeded_lost)` **only if**
`seeded_lost == SEED * REPS` exactly, and `TAGSWEEP_FAULT` (`0xEE000000`) otherwise. A return of
`0xA5000000` therefore certifies that the counter detected **6144** deliberately-untagged granules
in `tsml` and **768** in `tagsweep`, and found no others. A sweep that could not report a loss
would have returned `0xEE000000`.

This is exactly the check `s07evict` lacked — that arm is recorded as a negative but was **VOID**:
it assumed 64-byte cache lines, and the real geometry
(`capstone_cv64a6_imafdc_sv39_config_pkg.sv:48-50`) is 32768 B / 8-way / **128-bit** lines, so its
eviction never happened.

## What this refutes

The D-cache is **write-through, no-write-allocate**, so a `stc` does not allocate: after the store
pass the slots are in DRAM and not in the cache, and each first reload is a genuine **miss
refill** — the path the measured wedge implicates (`src=1, MISS REFILL`). Over 2.1M such reloads,
**the refill path did not lose a single tag**.

So S-07 is **not** a generic property of storing a capability to memory and reloading it, at any
rate above ~1 in 2.1M. Whatever the mechanism is, bulk store→DRAM→reload is not sufficient to
trigger it, and the "the refill path erases tags" reading of H-mem is refuted at that rate.
Measured throughput for the cache-resident arm: **186 cycles/check**.

## Two instrument defects found on the way, both of which cost boots

**1. `DOMAIN_WINDOW` is unusable without a monitor rebuild.** `build-ladder-domain.sh` will happily
relocate a rung's globals to a 32 KiB window, but the monitor hardcodes
`#define GPFREE_GLOBALS_OFFSET 0x1000` (`sbi_capstone.c:189`). The two then disagree about where
the blob starts and the domain hangs before it runs. Build rungs at the default window unless the
monitor is rebuilt to match.

**2. The rung driver's entry-stall verdict is a MISDIAGNOSIS for a slow domain, and its advice is
actively harmful.** On no `SHA6` it prints "ENTRY STALL (R-16), the domain never ran … R-16 is
per-image: REDRAW, **do not retry this binary**". But `SHA6` is only printed when the domain
*returns*, so a domain that is still running looks identical to one that never entered. Three
boots were spent on rebuilds — distinct entry VAs, a different glue, three buffer sizes — before a
deliberately trivial arm (`ts1`, 8 checks) returned instantly and showed the domain had been
entering and running correctly the whole time. It was simply slower than estimated: ~2400
cycles/check against my ~40, because every reload really does go to DRAM.

The board-run skill already says `SHA5`-last cannot distinguish a stall from an immediate wedge
and to discriminate on an entry marker. The rung path has **no** such marker, so that rule cannot
be applied there — the driver should say the discriminator is unavailable rather than assert R-16
and tell the operator not to retry. **The general lesson: when a run produces no result, build one
arm that must finish in microseconds before rebuilding anything.**

## Where this leaves S-07

Unchanged and still open: memory-holds-residue vs load-returns-wrong-data. This narrows *how* a
future probe must look — a bulk sweep is not sensitive enough, so the trigger depends on something
the sweep does not reproduce (a specific address, a specific access pattern, contention with other
traffic, or the much longer-lived capabilities SQLite holds). The next arms are the SQLite
minimization pair `MRO`/`MRR`, which are behavioural rather than statistical.
