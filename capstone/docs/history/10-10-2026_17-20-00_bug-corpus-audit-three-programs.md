# Bug-corpus audit of the three target programs (FFmpeg, tshark, memcached), 2026-10-10

Every cell of the published per-bug tables (`docs/ref/spatial-vs-temporal-three-programs.md` sections 0,
0b, 0b.1; `tools/catch-tables.py`) traced to the run record that produced it, by four independent
read-only audits, one per program family. The findings, the fixes, and the runs below. Lane branch
`audit-three-programs`.

## Pre-registered runs (written before each run except P3, see Outcome; this file's commit is the timestamp)

Case 12 of `ffmpeg/carved-repros` (moved from `subobject-repros/09`) is pre-registered in its own
`case.json`, commit `26cdee4e4d8d`.

### P1. wmem on stock CheriBSD with the system allocator as libc (`WM_LIBC_SYSTEM=ON`, `WM_CHUNKS=OFF`)

What it replaces: `wireshark/wmem-repros/results/20261007-cheribsd-revocation-control/`, which ran
the hosted harness's bump arena (one 384 MiB `aligned_alloc`; `g_free` a no-op) with the chunk port
compiled in (`WM_CHUNKS=ON`). That arm could not read anything but "completes": no wmem object ever
reached libc `free()`, not even control 90's jumbo.

The run: stock CheriBSD purecap, image `0cb16209…`, libc revocation on, every case under
`supervise` in the native fix-differential invocation (`program 0 N buggy|fixed`), so a temporal
case asserts that the next dissection's first allocation reoccupies the freed storage
(`wm_reoccupy`, `CHECK(next == stale)`) before reading through the stale pointer.

Predictions:
- **All 22 cases, buggy arm: exit 0 with `VERDICT DEFECT-REPRODUCED`.** Temporal (00-12): the stale chunk
  is REISSUED inside the block wmem kept, so the quarantine never sees it -- CHERI column "missed
  (reused)", not "held". Spatial (13-21): the crossing stays inside one `g_malloc`'d block and
  completes.
- **All 22, fixed arm: exit 0 with `VERDICT FIXED`.**
- **Control 90 (a jumbo `wmem_free_all` hands to `g_free`, then read): exit 0 with
  `VERDICT DEFECT-REPRODUCED`, reading the jumbo's own fill** -- the chunk HELD in libc's quarantine
  (freed, not reissued, no sweep before the read). By the board's rule a hold is "caught". If the
  revoker sweeps first it faults instead; either is the mechanism acting on wmem's own free.
- **revocation-control (memcached's, unchanged): SIGPROT si_code 2 at its labelled load.**

If a temporal case exits 75 (`CHECK(next == stale)` failed), the storage was NOT reissued, and that
case's CHERI reading would change from "missed" to "held": recorded as a refuted prediction.

### P2. carved-repros on CheriBSD and PoisonCap, all 13 cases (re-measures 0-11; first run of 12)

Arms `cheribsd-revocation`, `cheribsd-subobject`, `cheribsd-carve-bounds` on the stock image
(`0cb16209…`) and `poisoncap-spatial`, `poisoncap-protected` on the PoisonCap image (`c8df9e17…`),
with the corpus's own `runners/run-cheribsd.sh`. Predictions for 0-11: the readings committed on
2026-10-09 (`results/2026-10-09-*`), unchanged; for 12: its `case.json`. Every run must record its
binaries' sha256 and the bounds control's exit 162 (`summary.json`).

### P3. Capstone arms of carved-repros, all 13 cases

`spatial`, `sublet`, `capstone-subobject`, `capstone-carve-bounds`, `sublet-carve` with
`tools/run-capstone-domain.py` on the platform the 2026-10-09 records name (capstone-qemu
`32e7c9754f9a`, kernel `e58613598c89`, firmware `6f2b082cb677`, module `a88ed2159b43`, capstone-exec
`0752aa7c49c9`). Predictions: as P2 -- the 2026-10-09 readings for 0-11, `case.json` for 12.

### P4. memcached allocator 06 and 07 on the Capstone `spatial` and `sublet` modes

Their arms have carried oracle text only (both "complete"), backed by prose about a 2026-10-05 run with
no result line. Prediction: both modes COMPLETE on both cases (06: the suffix write stays inside the
slab chunk, which mode 0's alias and mode 1's region both bound at the chunk; 07: the 64-byte scan cap
keeps the read inside the chunk). Then `--negative-control`, which must make every oracle fire.

## Outcome (written after the runs)

Every prediction above held.

- **P1** (`wireshark/wmem-repros/results/2026-10-10-cheribsd-libc/`): 48 of 48 programs passed.
  - All 22 buggy arms reproduced. For the 13 temporal cases that means each case's own same-address reuse
    assertion held, so the chunk was reissued and the reading is "missed". All 22 fixed arms printed FIXED.
  - Control 90 was held by the quarantine.
  - The revocation control faulted at its labelled load.
- **P3** (`ffmpeg/carved-repros/results/2026-10-10-capstone-*/`): 65 of 65 rows as predicted.
  - Timing: P3 started at 18:09, after `26cdee4e4d8d` was pushed (18:05) and BEFORE this file was
    (18:14). So what pre-registered it is not this section. It is the readings already on `dev` for
    cases 0-11 and that commit's `case.json` for case 12.
  - Case 12 completes on spatial, sublet and capstone-subobject.
  - It is caught at `ffc_read_probe_u32`, cause 5, by capstone-carve-bounds and sublet-carve.
- **P4** (`memcached/allocator-repros/results/2026-10-10-capstone-0607/`): 4 of 4 arms completed, and the
  negative control fired on 4 of 4. Two boots first read "BOOT PRODUCED NO RESULT". The domain had exited 0;
  the runner's 90 s guest-command timeout cut the session before its end marker. At 400 s every run was
  clean.
- **P2**: (`ffmpeg/carved-repros/results/2026-10-10-{cheribsd-*,poisoncap-mode0,poisoncap-mode1}/`): 130 of 130 case
  programs met the arm's pre-registered expectation, with the platform controls and the carve control in each boot.
  - Case 12 is caught by `cheribsd-carve-bounds`. The fault is SIGPROT si_code 1 at 0x10253e, which `llvm-nm`
    on the run's own binary places inside `ffc_read_probe_u32` [0x10252e, 0x102548).
  - It completes on `cheribsd-revocation`, `cheribsd-subobject` and both PoisonCap modes.
  - The first attempt (B2) was refused with exit 75 by the runner's per-case table, which lacked case 12 -- the
    gate working. The re-run (B3) followed the one-line fix.

What the audit found and changed is section A of `docs/ref/spatial-vs-temporal-three-programs.md`.
