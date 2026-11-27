# b5b2-ticklog-2026-11-27: results

Runtime lane, 2026-11-27. Pre-registration: PREREG.md (28cdc156a874), Amendment 1 (340a17e4b463). The LB verdict is
rtl's, from the F_k file below by the rule in Amendment 1; it is recorded here when it lands.

## The boot

- Board 2, seal b42c8c644 (readback before and after), session tick2-b2-20261127T0342. First attempt, no rerun.
  Preflight exit 0 GO at 04:45:54Z. RUN END rc=0 at 07:19:41Z, "TRAPTEST: done seen", 2 h 34 min of the 4 h bound.
- Payload fw_payload.bin 181680bd16eec663ee16b88f4a5113887166323c58f66b36e7633a43423ed221, as Amendment 1 records.
- Capture: uart sha256 da46fa1586c4000e203c9a1fa42662488fc8820a6c252cbcebed8026f1992cf9, 11,353,873 bytes (not
  committed; board lane's session directory).
- Firmware and kernel lines: `TICKFW mhpmevent3=2 mcountinhibit=0x1f0 trap=0,0` (bit 3 clear: hpmcounter3 counts;
  the register holds bits for the implemented counters only, so it reads 0x1f0 where the twin read 0xfffffff0) and
  `ticklog: firmware M-path counters on (error 0 value 0x544b)`, `1048576 records of 44 bytes (version 3)`.
- All three stages rc=0. No `cause=` substring in the capture.

## The reader (read-ticklog.py --fw-fixed 251 --k-entry 170)

- Table sha256 0695833879ef604f02de8d4013b9ce86246d8ba326aca48ae9488e5c771024a8 matched the printed one; n = 675,665,
  lost 0, fw_ok 1; the clock calibrates to 25,000,000 Hz.
- **Every control PASS:** table, firmware counters on, clock within 0.1 %, N_s within 1 %, mcycle (≥ 99 % within 1 %
  of 250,000), awk loop (99.01 % within 3 % of its median 78,174), idle resolved (median 0), synthetic log.
  VERDICT-INPUT READY.
- **The window:** mcf rc 0, [105.28, 6752.71], 6,647.43 s, +0.70 % against the row's 6,601.15 s (no flag);
  N_s = 664,521 against 664,743 expected.
- **F_k:** 664,521 lines in [0, 1), sha256 ff2cf54e411c948c4354e27a1c4eead0e1ef56267b4042961994a819ed2f2bb3
  (ticklog/board/Fk.txt), sent to rtl for thin_keep.py.

## Reported, not graded

- **Program instructions** 26,679,585,857 against the twin key 26,644,263,390: +35,322,467, +0.133 %. Per tick that
  residue is 53.2 instructions; on the twin dry run it was 53.9 (+19,438,083 over 360,499 ticks). The same per-tick
  constant on both sides, not attributed and not subtracted.
- **All-mode** 30,658,727,592; kernel share 3,627,595,017; program cycles 149,137,877,260 (program CPI 5.59).
- **c_M:** 650,915 machine-timer interrupts from U, each measured at exactly 81 (sd 0, min = max) plus the fixed 251;
  no interval with two. 112 other M traps from U, 26,208 measured instructions, fixed part not subtracted.
- **hpmcounter3 (L1D miss, any D$ port):** 4,043,074,728 in the window, kernel 150,460,082, program 3,892,614,646.
  nested's model totals for the window are 3.70 G (ic6) and 3.80 G (ic7); the difference includes the PTW's and the
  lifetime unit's misses and is not a bound.
- **scause over the whole log:** S timer 658,448; store page fault 140,897; ecall from U 8,867; load page fault
  6,091; instruction page fault 3,271; S external 64; cause 30, 1. 26,852 window intervals carry a non-timer trap.
- **The placement:** the mean program delta is 40,149 per tick, and 47.6 % of the window's program instructions run
  in ticks at ≥ 1.8x that mean. The distribution is bimodal (bins aggregated from the reader's 0.1 bins by script):

  | delta / mean | ticks | share of instructions | program CPI |
  |---|---|---|---|
  | 0.0-0.3 | 6,622 | 0.05 % | 9.3-15.3 |
  | 0.3-0.4 | 158,868 | 9.13 % | 14.8 |
  | 0.4-0.5 | 38,279 | 2.73 % | 11.6 |
  | 0.5-0.6 | 123,637 | 10.21 % | 10.3 |
  | 0.6-1.8 (12 bins) | 193,145 | 30.26 % | 3.3-8.5 |
  | 1.8-2.0 | 23,847 | 6.84 % | 2.9-3.1 |
  | 2.0-2.5 | 115,598 | 38.88 % | 2.3-2.8 |
  | ≥ 2.5 | 4,524 | 1.90 % | 1.7-2.2 |

  The prediction written before the boot ("the ≥ 1.8x tail near 0 %", under the hypothesis that silicon places ticks
  as uniformly in instructions as the uniform model) is **refuted**: silicon's ticks are far from uniform in
  instructions. What that does to n is rtl's re-placement to say.

## Disclosures

- mcf ran first; in the 165.03 G row it ran after lbm, nab and xz.
- The kernel and firmware differ from the row's by the tick log only (PREREG, "The instrument"; Amendment 1).
- Two earlier payloads were built and withdrawn before any boot (247d7ea4, armed and withdrawn at 02:29:05Z; and
  39f07012, superseded after its twin key-match read +0.786 %).
