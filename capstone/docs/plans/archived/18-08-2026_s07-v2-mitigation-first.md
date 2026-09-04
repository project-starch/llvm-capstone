# S-07 v2: resolve it — mitigation first, structural minimization second

## Context

This replaces the previous plan, whose workstream A is **done and came back clean**. That result
changed the strategy, so the plan changes with it.

`TAGSWEEP` — a 10-39 KB standalone rung, no SQLite — stored capabilities, reloaded them and typed
each one with a non-raising query: **zero tag losses in 2 097 160 reloads**, with the counter
proven (it reported exactly the 6912 granules the arms deliberately untagged). Because the cache
is write-through/no-write-allocate, every first reload was a genuine **miss refill** — the path
our own wedge data implicates. So bulk store→DRAM→reload does **not** lose tags, and "the refill
path erases tags" is refuted at ~1 in 2.1M.

**The pivot.** The constraint that matters was already in the folder and I under-weighted it: for a
given image the fault site is **FIXED**, and only *whether it fires* varies run to run
(`history/14-08-2026_18-30-00_...md:56-62`). A defect that is structural per image will not be
found by sampling — which is exactly why 2.1M samples found nothing while SQLite wedges ~1 run in
3. **Stop probing statistically; act on the failing image itself.**

Constraints unchanged: current bitstream `caplifive_s06s08fix_s07tag2_618f4ce.bit`, **no RTL
change** (the RTL lane is unavailable — their last change caused uncontrolled memory consumption).

## Priority 1 — the mitigation, which is now also a discriminator

**Correction to my own earlier call.** I dropped `CapstoneLdcRetry` saying it "discriminates
nothing". That was reasoned from the empty-list-walk sites, where the loaded field should be NULL
so there is no valid capability to recover. But those are the **minority — 3 of 8 wedges**. The
**dominant site, 5 of 8, is `sqlite3OsRead+0x4c`**, whose `pMethods` is a legitimately assigned,
legitimately non-NULL capability (`sqlite3-capstone.c:110205`, `&MemJournalMethods`). At that site
a retry *does* discriminate:

* retry returns it **tagged** → the value in memory was intact → **load-returned-wrong-data**;
* retry returns it **untagged** → **memory-holds-residue** at that site.

And it is the only instrument that survives the fixed-site property. A per-site probe changes the
image, which changes the site — the documented "instrumenting a site moves the death". A **global**
transform does not care where the site lands, because it covers all of them.

Applicability checked against both observed shapes: at `output_text+0xdc` the faulting
`cincoffset a1, a1, a2` consumes an `ldc`-loaded `a1`, and at `sqlite3OsRead+0x4c` the faulting
`ldc a4, 0x20(a4)` consumes the preceding `ldc`. The pass instruments each `LDC`, so both are
covered. Known limitation, stated rather than discovered: it runs at `addPreRegAlloc` and cannot
see register-allocator spill reloads; every observed site is a program-level load.

**Nothing to build** — `llvm/lib/Target/Capstone/CapstoneLdcRetry.cpp` exists, is lit-tested with a
negative arm, and is off by default. One `-mllvm -capstone-retry-untagged-ldc` build.

**Statistics, because the rate is unstable and one clean run proves nothing.** The unmitigated rate
is roughly 1 in 3, so under "no effect" a mitigated arm passing k times has p ≈ (2/3)^k: k=8 gives
p≈0.04. With ~5 informative slots per boot that is **3 boots** of matched pairs, each boot pairing
mitigated and unmitigated arms so both see the same board state.

## Priority 2 — structural minimization of the failing image

**Reuse `CAPSTONE_EXT_STOP=<n>`** (`sqlite_capstone_domain.c:1062-1090`) — an existing 8-phase
preprocessor ladder that returns `SQLITE_OK` early from `run_sqlite_extended`, producing a full
clean return path plus an `EXTSTOP` marker so a stop is never mistaken for a pass. This is the
behavioural bisection lever; do not write a new one.

Ladder the stop value down and find the smallest workload that still wedges. Two things make this
sharper than it looks:

* **Track the fault site, not just pass/fail.** Every wedge's `mepc` is latched and symbolisable.
  If the site is stable across stop values, the trigger is upstream of everything cut; if it moves,
  the cut changed the layout and that itself bounds the mechanism.
* **Cutting work changes the image, hence possibly the site.** Expected, not a confound — record
  the site per arm rather than assuming continuity.

Plus the two genuinely new arms from the previous plan, which remain unbuilt: `MRO` (open
`:memory:`, close, nothing else) and `MRR` (open, one rolled-back transaction, close). The current
workload contains **no ROLLBACK at all**, so `MRR` is new coverage rather than a subset.

Note for expectations: staged builds are **not** smaller — every one links the whole library at
~1.5 MB. This minimizes *behaviour*, not bytes.

## Priority 3 — one opportunistic rung, only in a spare slot

The sweep covered a 64 KiB buffer written and read in a tight loop. It did **not** cover a **wide
address range with a long dwell** — capabilities spread over megabytes, left alive across a large
amount of unrelated traffic, then read back. That is the one statistical shape still untested and
it is the one that would show tag-store aliasing or capacity effects. It is a variant of the
existing `tagsweep_kernel.h` (bigger `.bss`, a traffic phase between store and read), so it costs
almost nothing — but it is **priority 3**, and it must not displace a mitigation arm.

## The boots

Control first or the boot is VOID; everything expected to return first; at most one
expected-to-wedge arm, last. ~6-run ceiling (`SILICON-BLOCKER.md:5130-5136`).

**Boots 1-3 — mitigation A/B, the same shape three times**

| # | domain | expected |
|---|---|---|
| 1 | `S7T` control | must pass |
| 2-4 | `XR` = `XU` + `-capstone-retry-untagged-ldc`, three reps | the arm under test |
| 5 | `XU` unmitigated | expected to wedge — confirms the board is in a wedging state |

Reps come from listing a domain repeatedly in `SQLITE_STAGE_DOMS`; no driver change needed. A boot
where slot 5 does **not** wedge carries no evidence about the mitigation and is discarded — that is
the control for board-state drift, and it is why the unmitigated arm must be in every boot.

**Boot 4 — minimization**

| # | domain | expected |
|---|---|---|
| 1 | `S7T` control | must pass |
| 2 | `MRO` | may wedge |
| 3 | `MRR` | may wedge |
| 4 | `EXT4` (`CAPSTONE_EXT_STOP=4`) | may wedge |
| 5 | `EXT2` | may wedge |
| 6 | `XU` | expected to wedge |

Whichever arm first fails to return **is** the minimization result; symbolise its `mepc`.

## Files

* **Reuse, no edits:** `llvm/lib/Target/Capstone/CapstoneLdcRetry.cpp` (+ its lit test);
  `CAPSTONE_EXT_STOP` in `benchmarks/sqlite/sqlite_capstone_domain.c`;
  `benchmarks/sqlite/bake-sqlite-doms.sh`; `fpga_driver/run_sqlite_stages_fpga.py`;
  `tests/runtime-qemu/silicon-ladder/tagsweep_kernel.h`
* **New, small:** `CAPSTONE_MIN_OPEN` / `CAPSTONE_MIN_ROLLBACK` blocks in
  `sqlite_capstone_domain.c`, wired in `build-sqlite-silicon.sh` **above** the `_domain_defs` read
  (below it the define reaches nothing and is silently compiled out)
* **Priority 3 only:** a wide-range variant beside `tagsweep_kernel.h`

## Verification

1. **The unmitigated arm must wedge in the same boot**, or that boot says nothing about the
   mitigation. This is the positive control for the whole exercise.
2. **Flag-off codegen must stay byte-identical** — diff the `XU` artifact against the historical
   `f1214600` reproducer; a codegen flag that perturbs the default build invalidates prior numbers.
3. **QEMU + lit before any boot** for the retry build, and the corpus suites serialized (the shared
   `rootfs.ext2` write lock means never two at once).
4. **Verify every domain by content hash inside the cpio**, not on the filesystem.
5. **Do not trust a rate**: the measured rate moved 23% → 0% → back, unexplained. Report k and the
   observed p, not "it works".
6. **When a run yields no result, build one arm that must finish in microseconds before rebuilding
   anything** — that habit is what ended the three-boot detour last session, where a slow domain
   was misdiagnosed as an entry stall.
7. `precommit-scan.sh` before every commit and push.

## Explicitly not doing

No RTL change, no bitstream, no monitor change, no DT change. No cross-boot DRAM recorder (the
harness powers the board off for 8 s every boot, so nothing retains). No further bulk statistical
sweeps beyond the single priority-3 arm — that shape has been tested to 2.1M samples and the
fixed-site property says it is the wrong instrument.
