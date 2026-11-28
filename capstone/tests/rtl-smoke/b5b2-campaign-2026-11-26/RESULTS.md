# b5b2-campaign-2026-11-26: results

Runtime lane. Pre-registration PREREG.md (fec8d4357796) with Amendments 1-3 (dadd37b994ae, 2e80c57456ed,
9a8152a646ec). Every row below was graded from its own capture by this lane: the time is the sum of up1 − up0 over
the stage's @@SPEC-END lines, and the hashes are the v2 grader's result.json, compared file by file with
`twin-camp.json` (CF and CS) or between boots of one image (CFP). The comparison's positive control, one changed hash,
reads `1 differ`. x264's `log.dec` carries the decoder's timing (PREREG, "Twin reference") and is reported, never
counted as a difference. Board-lane step logs and uarts are not committed; each row cites its uart's sha256 prefix.

**Interim, 2026-11-28.** Phases A, C and C2 and board 2's long rows are complete. Phase B is still running on
board 1 (CS g1b and x264b, two of six read) and board 3 (CF x264b, g1b, x264b; one of four read). The Phase B
section and P3/P4 on the long stages are filled in when those rows land.

## P1: every stage PASS, hashes equal to the twin, no cause-25 fault at an LDC

**RETRACTED 2026-11-28 for x264 (dev 59f229e2579d):** x264's "PASS" is not a correct-output verdict. Every Capstone
x264 writes a malformed H.264 stream (x264's WORD_SIZE = sizeof(void*) = 16 on capstone64; the JM decoder rejects a
1-frame encode from B5TL1, while Plain's decodes), and the twin reference it equals was produced by the same defect.
The reconstructed frames, rc 0 and the encoder's decisions hold; the encoded stream does not. What follows from it
for the x264 rows is the lead's decision.

Holds on every row read so far: 42 boots of the campaign's images (board 1: 21, board 2: 20, board 3: 1; counted by
script over the session directories): v2 and v1 PASS, every hash equal to
its twin reference (CFP: equal between its boots and to CF's twin), no `cause=` substring in any capture, and mcf
completes on every g1b boot. C2's three g3v2 boots read the decoded files Amendment 3 names (.s 58111208509dd871,
.err 0c86179b08efc831, .out e3b0c44298fc1c14), as do C2's three CF-g3 boots.

## Phase A: board 1, S F F S S F, the Full / +Spatial pair on one board

CS = 14f2ef81a, CF = b42c8c644. Times in seconds, three runs per seal, mean and (range).

| stage | CS runs | CS mean (range) | CF runs | CF mean (range) | CS against CF |
|---|---|---|---|---|---|
| imagick | 11.21, 11.18, 11.20 | 11.197 (0.03) | 11.21, 11.21, 11.22 | 11.213 (0.01) | −0.15 % |
| gcc | 5.49, 5.49, 5.49 | 5.490 (0.00) | 5.55, 5.55, 5.54 | 5.547 (0.01) | −1.02 % |
| perlbench | 279.57, 279.95, 278.74 | 279.420 (1.21) | 276.91, 279.17, 276.75 | 277.610 (2.42) | +1.81 s (+0.65 %) |

- Full/+Spatial ratio (CF mean / CS mean) on one board: imagick 1.0015, gcc 1.0103, perlbench 0.9935.
- CS-g3's first boot (step 1) was the size proof for 21,662,216 B on 14f2ef81a: it reached /init.
- Board 1's CF-g1b (step 3): lbm 1,015.96, nab 727.45, xz 4,692.99, mcf 6,368.63 (uart c5ed03b6).

## Phase C and C2: board 2, the ports pairs (b42c8c644)

| image | runs | mean (range) | against CF |
|---|---|---|---|
| CF-B5TL1-g3 (Phase C) | 5.56, 5.55, 5.54 | 5.550 (0.02) | |
| CFP-B5TL1-g3, port v1 | 5.83, 5.83, 5.83 | 5.830 (0.00) | +5.05 % |
| CF-B5TL1-g3 (C2) | 5.56, 5.55, 5.55 | 5.553 (0.01) | |
| CFP-B5TL1-g3v2, port v2 | 5.84, 5.82, 5.82 | 5.827 (0.02) | +4.92 % |
| CF-B5TL1-g4mk | 277.87, 276.92, 277.10 | 277.297 (0.95) | |
| CFP-B5TL1-g4mk | 278.61, 278.55, 280.42 | 279.193 (1.87) | +0.68 % |

- No prediction was made for v2 against v1 (Amendment 2); they read 5.827 and 5.830.
- The two N=1 readings carried into the design read CF-g4mk 275.70 and CFP-g4mk 279.75 (a162a8aefb60, another image);
  CF's sits 1.6 s below this N=3 mean.

## Long stages, CF on b42c8c644, three boards

| board | lbm | nab | xz | mcf | x264 gen | x264 run |
|---|---|---|---|---|---|---|
| 1 (step 3) | 1,015.96 | 727.45 | 4,692.99 | 6,368.63 | | |
| 2 | 1,015.37 | 727.40 | 4,694.02 | 6,367.95 | 2,056.43 | 32,816.41 |
| 3 (Phase B, first) | 1,015.56 | 727.44 | 4,695.80 | 6,366.06 | pending | pending |

Spread over the three boards: lbm 0.058 %, nab 0.007 %, xz 0.060 %, mcf 0.040 %.

## P2: CF no slower than its qualification row on the same seal

Holds on every stage read:
- perlbench 277.61 (board 1) and 277.30 (board 2) against 282.94;
- gcc 5.547 and 5.550 against 5.76;
- imagick 11.213 against 11.63 (the qualification g2b row, B5TL1-g2b 6f8c83c0, a different overlay image);
- lbm, nab, xz, mcf about 3.5 % faster than board 3's qualification B5TL1-g1b (1,052.26 / 752.86 / 4,863.14 /
  6,601.15);
- x264 gen −3.3 % and run −3.1 % against board 3's B5TL1-x264b (2,127.40 / 33,881.05).
The final kernel only removes work, as P2 says; these are not CF effects.

## P3 (rtl's): on one board, CS ≤ CF on every stage

- **Holds** on imagick and gcc.
- **perlbench: neither shown nor refuted.** CS's mean is 1.81 s above CF's, inside the larger of the two ranges
  (2.42 s). The pre-registered hpm row is triggered only by a difference beyond that range, so it does not run.
- Long stages: pending Phase B. The first CS-g1b (board 1, uart d2bf5f0a) reads lbm 1,014.74, nab 723.93, xz
  4,690.35, mcf 6,033.17: below board 1's CF on every stage, mcf by 335.5 s (5.3 %). N = 1 per seal so far.

## P4: board 1's CF within 0.2 % of boards 2 and 3

Holds on every stage read: gcc −0.06 %, perlbench +0.11 % (board 1 against board 2), and the four g1b stages within
0.06 % over all three boards. imagick has no reading (Amendment 1).

**The qualification perl gap** (293.96 s on 14f2ef81a against 282.94 s on b42c8c644): with P3 and P4 as read, board
1's CS and CF lie within each other's noise and board 1's CF equals board 2's, so the gap is **not reproduced** on one
board at N = 3: 1.81 s of the 11.02 s.

## Incidents and disclosures

1. **Board 1, flash to b42c8c644, 2026-11-26 20:28-20:30Z:** the console's power-on timed out before the flash lock;
   no SPI write; the board was left off with 14f2ef81a resident, and the session stopped. Resumed from that flash
   step (camp-b1-r1.plan).
2. **Board 1, CF-g2i failed start, 20:58-21:10Z:** a 60 s upload, then engineio's "packet queue is empty, aborting",
   then the same power-on timeout. Mechanism (engineio client.py:664-665 and 510-512): the write loop ends while the
   state stays "connected", so a power toggle emitted after a long upload is lost. The wrapper's rerun then stopped
   "board busy", because its mv preceded the idle wait and the old holder missed its stop file. Fixed: the console
   client re-verifies and re-opens the socket and reads power state from a fresh event only (board/console-reconnect
   76bfcbb876ea, ceae409cfe8b), and the rerun order is fixed (run-one-h-rt, then run-one-i-rt). The reconnect path
   has not fired on a board since: uploads ran 5-29 s.
3. **Board 2, placement PL3, 02:08Z, and 4. board 1, CS-x264b, 2026-11-27 07:25 local:** the JTAG load timed out in
   GDB ("Timed out after 120s waiting for busy to go low", abstractcs 0x8001102) after a good power cycle and upload,
   with no console byte. Both reran once under the fixed order and read. N = 2 on two boards, not attributed.
- **The dry-run board leak:** a dry run of run-one-f exported the real FPGA_BOARD to its fake runner, so board-busy
  read the real board as busy during dry boots. There was no console contact; real idle waits were 0-10 s. Fixed in
  run-one-g.
- **C10 and its DECLARED checker:** every row above 20 MB passed C10's size rule by this lane's ruling ("this size ran
  on this seal"), never through the DECLARED-bake content check. That checker (fw-declared-payload.py at
  6daffca3824c) read only the first of a payload's two concatenated cpio archives; the board lane found it on
  2026-11-27 and fixed it on preflight-declared-multi aca4580de705 (7/7 tests, rerun by this lane). It gated no row
  here.
