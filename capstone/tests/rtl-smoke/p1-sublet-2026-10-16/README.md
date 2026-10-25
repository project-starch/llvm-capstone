# P1 cells 5 and 6 at -O2 beside the native baseline on the Sublet bitstream: pre-registration (2026-10-16, before the boots)

The board campaign's wave 1 item 2 (`docs/plans/2026-10-16-board-campaign-all.md`): the P1 native baseline in the same
boot as cell 5 and as cell 6 (fixed).
- Both cells enter at 0x10000, so they cannot share one boot without relinking an artifact under measurement.
- So each cell gets its own boot beside the native.
- This is not P1's full protocol (five arms, five runs over three boots): it is one capability/native ratio per cell
  on this bitstream, at size 1.

## The images: one compiler, one tree

Both cells come from lane b0-silicon-runtime (dev's sqlite port and Sublet runtime, which carries the sublet-runtime
fix a07a11194fa0), built with `clang version 22.0.0git (llvm-capstone 612b3ec514c0)` through
`ports/sqlite/run-speedtest1-measure.sh`, P1's recipe (`~/capstone-artifacts/p1-O2-2026-09-30/README.md`): -O2,
lookaside 1200,40.

| cell | image | geometry (CPDOMREQ data / stack) | QEMU `--stats`, 2 MiB | lookasides |
|---|---|---|---|---|
| 5, memsys5 + lookaside, 2 MiB static heap | `21f4b2659074511c` (stack 385,024) | 0x293ab0 / 0x5e000 | 333,794,832 | 25,010 |
| 6, the Sublet port, `--arena 2097152 --tables 1750285` | `4bfe3b0ec8494385` | 0x135de0 / 0x100000 | 344,110,502, sublet 5568/37970/32569/37970/5401 | 25,010 |

- **Cell 6 rebuilt byte-identical** to the Sublet acceptance's "fixed" cell 6, which ran on this bitstream on 2026-10-16
  (c6pair, 1,292,584,693 cycles). Its CPDOMREQ equals P1's own 09-30 cell 6 (df484d98b489aeab).
  - A first rebuild here carried S1's stack override (385,024), which P1's cell 6 never had. It was discarded (796c9a94).
- **Cell 5 is new:** the 09-30 cell 5 (5f4b44257c933477) came from another tree and toolchain (08ff5d0702c3). Today's
  has the same CPDOMREQ, and different code (1,443,880 bytes against 1,378,344).
- Both earned their pass records from the benchmark (`112006 38bb59fd`).
- The native -O2 baseline is the 09-30 one, `9a80c1cd2a576ed2`, QEMU BASELINE-WARM 240,714,705.

## The boots

`drivers/board-b80s.sh`, run from a private copy whose tree root is the main clone (as `../h1-sublet-2026-10-16`):
- the classic line, d04bd83 / df0e9a8 / 93440d5;
- the relinked k800 589ceee3853c6092 at 0x20000 (the cells occupy 0x10000) and host 2c9e82d101b48160;
- `REFUSAL_RECORD=1`, `PREFLIGHT_DTS_NODES=65536`;
- **`MEMINFO=1` (new):** `/proc/meminfo` is read first, after Linux, and last, after the native. This is not a verdict
  arm. It is the board's idle memory figure the virtual-profile plan asked for.

| boot | arms, in order |
|---|---|
| p1s16-c5 | meminfo, k800, cell 5 `--stats`, k800, native `warm --stats`, meminfo |
| p1s16-c6 | meminfo, k800, cell 6 `--arena 2097152 --tables 1750285 --stats` (`CELL_HOSTARGS`, `C5_QEMU_HEAP=1344064`), k800, native `warm --stats`, meminfo |

`board-b80s.sh` gains `CELL_HOSTARGS`, `C5_QEMU_HEAP` and `MEMINFO`, all defaulting to the old behaviour, and keeps each
boot's `fw_payload.bin`.

## Pre-registered readings

**Per boot:**
- k800 returns 4 twice;
- both arms read `112006 38bb59fd`;
- the cell reads HEAP 2097152 (cell 5) or 1344064 (cell 6), with DROPPED 0;
- lookasides 25,010 on the cell and 25,122 on the native.
- The cell 6 boot prints sublet 5568/37970/32569/37970/5401. The cell 5 boot prints no sublet line.

**Cycles, computed from today's QEMU counts and Thursday's silicon CPIs:**

| arm | predicted | from |
|---|---:|---|
| native -O2 warm | 856,160,837 ± 0.1 % | Thursday's a8, where 09-29 R-42 read 856,120,199 |
| cell 5 (new image) | 1,167,686,616 | 333,794,832 × Thursday's cell 5 CPI 3.4982 |
| cell 6 | 1,292,584,693 ± 0.5 % | Thursday's c6pair, the same image |

**Ratios:** cell5/native ≈ 1.364, cell6/native ≈ 1.510, cell6/cell5 ≈ 1.107.
- Cell 5's CPI on the new code is the one estimate. A different CPI moves only the first ratio.
- The driver's own band is CPI 3.2-4.9.

**Not established:** N = 1 per cell; size 1 only; P1's other arms.

## Amended before the rerun (2026-10-16 22:35): the first boot stopped at the meminfo arm

**Boot p1s16-c5 (22:16-22:22) produced no reading.** The driver HARD STOPPED at the first arm, `meminfo|idle`, with
"idle produced no RESULT retval= marker, not a staged marker". `meminfo.sh` printed `/proc/meminfo` and nothing the
stage driver accepts as a host-binary arm's result. It was never run through the driver before this boot. So no
cell, control or native ran. Its record stays under `board-b80s-p1s16-c5`.

**The fix, in `drivers/meminfo.sh` only:**
- it now ends with `BASELINE-PROBE memfree_kb = <MemFree>`, the form `run_sqlite_stages_fpga.py` accepts for a
  host-binary arm (as for boot sw52's probe arms);
- the value is read with a plain `sh` loop, not awk;
- tested under busybox `sh` against the driver's own regex (`BASELINE-PROBE (\S+) = (\d+)`). The old output does not
  match it, the new one does.

The rerun's boots are **p1s16b-c5** and **p1s16b-c6**, with arms, images and readings exactly as pre-registered above.
They run after SLT, because R1's boots were already under way.
