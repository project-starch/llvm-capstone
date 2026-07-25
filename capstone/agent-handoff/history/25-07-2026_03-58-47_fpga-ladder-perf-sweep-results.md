# FPGA silicon-ladder perf sweep — on-board mcycle results

## ⚠⚠ UPDATE 2 (2026-07-25, fresh-dom re-sweep) — SUPERSEDES the stale-build claim below for matmult/recursion

The "stale build" correction in UPDATE 1 was **over-generalized**. A re-sweep with
**freshly-rebuilt post-fix doms** (runner now rebuilds-by-default) shows the core
silicon divergence is **REAL, not stale**:

| rung (fresh dom) | silicon retval | oracle | mcycle | verdict |
|---|---:|---:|---:|---|
| rv8_primes | 99991 | 99991 | 17,283,292 | ✅ PASS |
| beebs_prime | 582955588 | 582955588 | 47,804 | ✅ PASS |
| **matmult_int** | 1166210317 | 774662735 | 76,498 | ❌ **real miscompile** |
| **beebs_recursion** | 2095861164 | 1579141629 | 30,263 | ❌ **real miscompile** |
| coremark_matrix | — (transfer flaked) | 14343 | — | no verdict |
| beebs_crc32 | — (transfer flaked) | 1703161001 | — | no verdict |
| beebs_insertsort | — (transfer flaked) | 271779359 | — | no verdict |

- `matmult_int` and `beebs_recursion` return the **same wrong value** as the stale
  run on a **freshly-built dom**, while QEMU is correct. The memcpy fix (`d078839`)
  does not change their codegen (recursion has **no** local const array at all), so
  these were never the stale/memcpy artifact — they are genuine QEMU-correct/
  silicon-wrong miscompiles, i.e. the **open gp-captable silicon bug**
  (`23-07-2026_17-30-00_gp-captable-silicon-array-loop-miscompute-OPEN.md`), and
  they persist with **shrink OFF** (so not the 23-07 shrink→store workaround's target).
- Only `beebs_insertsort` fits the stale/memcpy story (its `255001740` = the exact
  pre-fix memcpy signature, and it *does* have a local const array) — but the
  re-sweep couldn't confirm it because its transfer flaked all attempts.
- The 3 "no verdict" rungs failed at **transfer**, not compute: fast_xfer over the
  board UART/websocket non-deterministically drops (empty-file sha `e3b0c442…`,
  unterminated-quote `> ` shell prompt, `DN_` marker timeouts). So `coremark_matrix`'s
  `-Os` hang (Finding #2) is **still unverified** on a fresh dom.

**Bottom line:** UPDATE 1's process-bug finding is correct and the runner fix
stands, but its "these 4 don't corroborate the RTL/gp-captable bug" conclusion is
**retracted for matmult_int + beebs_recursion** — those ARE real silicon
miscompiles. Next: (a) robuster transfer / prompt-sync to clear the 3 no-verdicts;
(b) fold matmult/recursion into the open gp-captable silicon-bug investigation.

---

## ⚠ CORRECTION (2026-07-25, same day, post-sweep) — the 4 "miscompiles" were a STALE BUILD

Finding #1 below ("gp-captable silicon miscompile", 4/4 array-store rungs) is
**RETRACTED**. Root cause of those 4 wrong retvals was **stale domains**, not
silicon/RTL:

- `run_ladder_perf_fpga.py` uses whatever `<rung>.dom` already exists in
  `$OUT_DIR/ladder-fpga` and **never rebuilds** it (it only errors if missing).
  The doms in that dir predated the **24-07 sub-cap memcpy `stc`-packing fix**
  (commit `d078839`, `history/24-07-2026_03-57-54_ladder-rung2-insertsort-...`).
- `beebs_insertsort`'s silicon retval `255001740` is the **exact** pre-fix memcpy
  signature ("array sorted right, `expected[]` read wrong") from that 24-07 note —
  an address-independent FNV checksum, not the `0x8Bxx` address-contamination
  shape of the real shrink hazard. That alone pins it to the compiler bug.
- **Verification (board-free, current compiler):** rebuilt & ran all four under
  QEMU via `run-ladder-qemu.sh` — **all PASS the oracle**: matmult_int 774662735,
  beebs_crc32 1703161001, beebs_insertsort 271779359, beebs_recursion 1579141629.
  The current `-b` clang (built 24-07 18:30) is newer than the fix; the sweep
  (25-07 04:00) simply ran pre-fix `.dom` artifacts.

**Consequence:** these 4 do **not** corroborate the gp-captable array-store /
`shrink`→store RTL bug. That RTL-leaning bug (23-07 note) is real but **separate**
and is worked around here by shrink-off (the ladder builds `-capstone-shrink-*
=false`), so it is not the mechanism behind these four. The mcycle numbers for the
4 miscompiled runs are measurements of wrong executions → **discard**. Only
`rv8_primes` and `beebs_prime` remain valid silicon points.

**Action:** a board **re-sweep with freshly-built doms** (delete `$OUT_DIR/
ladder-fpga/*.dom` first, or have the runner force-rebuild) to get correct
on-silicon retvals + mcycle for 6/7 (coremark still hangs — real, still open).
The runner should be hardened to rebuild-or-fingerprint doms so this can't recur.

---

**Date:** 2026-07-25
**Task:** run the 7 ready silicon-ladder rungs on the Genesys2 CVA6 FPGA, record
`mcycle` + `retval`, gate correctness on `retval == native cc -O0 oracle`.
**Status:** DONE — 6/7 rungs produced real on-board `mcycle`; 2/7 also pass the
correctness gate. Board powered off + unlocked in `finally` (etiquette clean).

## Results (silicon)

| rung | retval | oracle | mcycle | correct |
|---|---|---|---|---|
| matmult_int | 1166210317 | 774662735 | 76,747 | ✗ miscompile |
| coremark_matrix | — (hang) | 14343 | — | ✗ cscall hang (`-Os`) |
| rv8_primes | 99991 | 99991 | 17,286,789 | ✓ |
| beebs_crc32 | 1568735421 | 1703161001 | 311,956 | ✗ miscompile |
| beebs_insertsort | 255001740 | 271779359 | 10,464 | ✗ miscompile |
| beebs_prime | 582955588 | 582955588 | 47,796 | ✓ |
| beebs_recursion | 2095861164 | 1579141629 | 30,280 | ✗ miscompile |

Only `rv8_primes` and `beebs_prime` have BOTH a correct `retval` and a valid
`mcycle`, so those two are the trustworthy silicon perf points. The four
miscompiling rungs ran to completion (`ran=0xD09E` marker present) so their
`mcycle` is a real measurement, but of a miscompiled execution — treat as
lower-confidence. `matmult_int` reproduced identically across three separate
boots (retval 1166210317; mcycle 76747 / 76723 / 76479, ≈0.3% jitter).

## Findings

### 1. The gp-captable silicon miscompile tracks "store-to-global-array + live accumulator"
The correctness split is coherent, not random. The two PASS rungs
(`rv8_primes`, `beebs_prime`) are pure scalar-accumulator loops. All four FAIL
rungs write into a **global array while keeping a live accumulator**
(matmult → result matrix; crc32 → table-driven; insertsort → array in place;
recursion → array/stack), which is exactly the address-contamination shape of
the open, un-root-caused gp-captable codegen bug
(`plans/gp-captable-codegen-plan.md` §Stage-4; earlier `rc_const0` PASS vs
`rc_p1` FAIL on-board). This is the strongest on-silicon corroboration of that
bug so far: 4/4 array-store rungs miscompile, 2/2 scalar rungs are correct.

### 2. coremark_matrix (`-Os`) hangs the cscall even as the FIRST domain
`coremark_matrix` is the only rung built `-Os` (it overflows the 4 KiB PCC code
window at `-O0`). It hangs the `cscall` (silent, no RESULT, no reset banner)
**even as the first domain of a clean boot** — so this is code-specific, not the
multi-domain hang below. The `-Os` codegen emits something the CVA6 domain-entry
path chokes on. Distinct finding; not yet root-caused.

### 3. A second same-VA domain hangs within one boot (icache / missing fence.i)
On a single boot the first domain calls + returns fine, but a **second** domain
(entry reused at VA `0x10000`) silently hangs its `cscall`. This is the
domain-boundary icache-coherence gap (no `fence.i` between placing the new
domain's code and fetching it — the RTL does no icache invalidate on the switch;
see `plans/curried-crunching-gizmo.md` / the domain-CALL diagnostics plan). The
sweep sidesteps it with **one full power-cycle + firmware reload per rung** so
every rung runs as a first domain against a clean icache.

## How the sweep runs (runner mechanics)
`tests/rtl-smoke/fpga_driver/run_ladder_perf_fpga.py`:
- Upload the 15 MB `fw_payload_fpga_up_gpfree.bin` to the console image store
  ONCE (HTTP; persists across power-cycles).
- Per rung: `cold_boot()` = full power-cycle + JTAG `load_image` store→DDR (~2 min,
  the dominant cost) → set pc=0x80000000, a0=0 → continue → confirm root shell →
  **`insmod /capstone.ko`** (the UP image ships but does not auto-load it) →
  `fast_put` (tier-1 `fast_xfer`, ~3× the old per-chunk transfer) the controller
  + that rung's `.dom` → run `ladder_perf_ctl <name> <dom>` → harvest
  `RESULT <name> retval=… cycles=… ran=…`.
- Board etiquette: lock → power-cycle → run → power off + unlock in `finally`;
  resident bitstream verified `working-caplifive-captype-fixed.bit` before
  measuring.

### Two runner bugs fixed this session (why the earlier runs got 0 results)
1. **Missing `insmod`.** The runner assumed `/dev/capstone` at boot; the UP image
   requires an explicit `insmod /capstone.ko`. Without it every `open()` failed.
2. **False-positive device check.** The old check did `"DEVOK" not in out`, which
   matched the *echoed command line* (it literally contains the word `DEVOK`), so
   a `DEVNO` result passed. Now gated on the trailing token.

### Warm reboot does NOT work here
`monitor reset halt` + `continue` from `0x80000000` (no reload) yields zero new
UART — the fw_payload OpenSBI does not re-enter cleanly from a soft reset (its
one-time hart/DDR init is not re-runnable). A real power-cycle + reload is
required between rungs. (Ruled out as the cheap path; power-cycle-per-rung is the
reliable one, ~2.5 min/rung.)

## Delivery-path note (tier-2b / JTAG question)
Independently verified against the driver code (not just the KB): the web
"load image" button (`POST /api/load-image {filename}`, `config.py`) is
filename-only and hardwired to `0x80000000` — a boot-image loader, it cannot
place a domain at an arbitrary address in running Linux. Two real JTAG routes:
- **Live poke** (gdb `monitor load_image dom <addr>`): needs a board-owner-blessed
  **reserved RAM region** (else it stomps the booted kernel; controller must know
  the addr to read it). This is what the "never guess a RAM address" rule guards.
- **Bake into initramfs + reload whole image** ("recompile the image and load
  it"): needs NO reserved address (the fw_payload has a built-in initramfs —
  confirmed), but requires rebuilding the image, which is gated on the
  monitor-regen boot-hang (`plans/monitor-regen-audit-task-B.md`).
For a handful of tiny integer domains, tier-1 `fast_put` in one session is the
right tier (per `ref/HOW-TO-LAUNCH-ON-FPGA.md`); tier-2b is the SQLite-scale
on-ramp.

## Next
- Root-cause the gp-captable array-store miscompile (compiler lane; the 4 FAIL
  rungs are ready reproductions with silicon-vs-oracle deltas).
- coremark_matrix `-Os` cscall hang: separate reproduction; likely needs the
  domain-boundary `fence.i` and/or an `-Os` codegen look.
- If in-boot multi-domain is wanted (to avoid 2.5 min/rung), the `fence.i` monitor
  fix is the unlock, gated on monitor-regen.
