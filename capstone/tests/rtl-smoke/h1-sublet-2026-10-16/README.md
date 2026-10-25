# H1 calibration on the Sublet bitstream: pre-registration (2026-10-16, before the boots)

The board campaign's wave 1 item 3 (`docs/plans/2026-10-16-board-campaign-all.md`, H1 calibration). It repeats
`../paper-r43v2-2026-09-29` "H1" (three boots h1v2r1-r3 on `caplifive_r43_8f6a0af.bit`) on
`caplifive_sublet_d93a8d66b.bit`, unchanged. A calibration repeat across bitstreams is worth most when nothing but the
silicon differs.

## What runs, identical in all three boots

`drivers/board-r1e4.sh` with:
- `R1_IMG` = image **da9c54e578845735**, the R1 harness at defaults, entry 0x410000: the 2026-09-29 image itself,
  not a rebuild;
- `R1_LIST` = `drivers/lists/h1-v2-calibration.txt`, all six lines in each boot. These are the latency chase with
  three timed traversals, `--calib` in three fresh domains, the Sublet nodes regression and the linear series;
- the readback host `sqlite_host_rr.user` = 2c9e82d101b48160, 2026-09-29's;
- the stock k800 control b2d60e525f807ea4 first and last, `REFUSAL_RECORD=1`;
- the classic line: buildroot d04bd83, opensbi df0e9a8, capstone-sbi 93440d5 (carrying 2dcd3a5), the #3 module.

**The driver copy.** A private copy of the lane's `board-r1e4.sh` is used, not committed (it names a local path).
It differs in two lines:
- `R=` is the main clone, which holds the submodules, instead of the script's own tree, as on Thursday's R1 boots;
- after the bake it keeps the boot's `fw_payload.bin` beside the boot's record.
- the runner, preflight and watchdog are the main clone's (as `../repro322-silicon-2026-10-16`, "The runner tree"),
  so the run declares `PREFLIGHT_DTS_NODES=65536`.

**Why the image is not rebuilt.**
- check-repr's process-ABI replay reports this image AT RISK under the full format: 43 inexact global carve points.
- It reports the same for the R1 image 1b7a04fe237e1580, which read as pre-registered on this bitstream on Thursday
  (R-43 a2, on the exact-regions firmware: 60 point lines `ok=1`, no fault). It also reports it for every classic-line image of that vintage. The replay does not model the classic
  monitor's carve, so it cannot tell a good classic image from a bad one here.
- Images of the 09-29 vintage ran on d93a8d66b as pre-registered on Thursday. Among them is the frozen R-35 probe
  35fb3fec3196841b (a10).
- The emulator record is 2026-09-29's (`~/capstone-artifacts/h1-v2/q-latency`, -calib, -nodes, -linear; the pass
  record `qemu-pass/da9c54e5…`).

## Pre-registered readings, per boot

The change detectors are 2026-09-29's silicon readings, identical in its three boots unless a range is given:
- **k800** returns 4 twice, instret 1089. Otherwise the boot is void.
- **Latency** (`R1 lat … per=`): 9, 9, 40, 48, 48 cycles per load at 4, 16, 64, 256 and 1024 KiB, flat across the
  three traversals.
  - A uniform shift is a result, not a failure.
  - New readings would be: 4 KiB ≠ 16 KiB, a non-monotone curve, or traversals disagreeing by more than 0.1 cycles.
- **Timer:** `cyc_cyc=2 cyc_cyc2=2 ret_ret=1` in all three domains.
- **Nodes regression:**
  - nd = 2n, `ok=1 bad=0` at every n = 1, 4, 16, 64, 256;
  - no cause 25;
  - REVOKE (`rv=`) recorded against 2026-09-29's 144-146 / 193 / 722-725 / 2,873-2,897 / 11,106-11,293.
  - **New on this bitstream:** the Sublet node record carries a parent link, and CREVOKE sits in R-45's flush. A
    revoke cost that moves is a finding about that, and it is reported per n.
- **Linear:** arms 0-11 read types 7,1,7,7,7,1,7,1,0,1,0,1, as on v2. Arms 8 and 10 read 0 (`tighten`, `shrinkto`
  copy, R-21); the controls read 1.
  - **New on this bitstream:** the capability word is the full format, with cap_type in 3 bits. A type that reads
    differently is a finding about the new decoder.

**Not established here:** three boots, N = 3 per point, the same as 2026-09-29.
