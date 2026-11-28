# B5b-2 seals on the boards, 2026-11-25/26: the 28fcbb646 diagnostic and the a5c45a46d (R-80) sessions

Result lines only. The kits, their pre-registrations and their vectors are the rtl lane's (llvm-capstone
land/rtl-crevoke4b); the SPEC and Linux amendments are the runtime lane's (runtime/b5b2-linux). Sibling records:
`../b5b2-spectest-board-2026-11-25/` (the SPEC chains, the mcf exposure table) and `../b5b2-linux-board-2026-11-25/`
(boots 1-12).

## The seals

| seal | build | grade | sha256 |
|---|---|---|---|
| 28fcbb646 (diagnostic: apertures + the FLU's untagged-x[rs1] rule) | full, 1 h 35 m, peak 24.05 GB | WAF-FULL PASS 10/10: placed LUT 151,405, FF 69,497, clk_out1 +1.516, led_stretch +8.556 | 925607df163bd8c8b96a2ccef84126478251124eb02fe31d301cfee18cc0d32e |
| a5c45a46d (28fcbb646 + R-80: a faulting LDC pops in SEND_TAG_LDC) | gate 3/3 (core LUT +6, load_unit +4), full 1 h 41 m, peak 24.09 GB | R80-FULL PASS 10/10: placed LUT 151,406, FF 69,500, clk_out1 +1.778, led_stretch +9.412 | 9b46b14fb6252873696acc9762d7aecbc53271567447460207737c2366f1e36e |

## Board 2, the 28fcbb646 session (`diag-28fcbb646-board2.result-lines.txt`)

- Flashed 12:33:41-12:35:59Z, rc 0; the name read back after the flash and after every step.
- wapflu cand 1/1 AS PREDICTED. b5b2-untagged reads the candidate column; the same captures graded against the
  88db40027 column MISS at 13 positions. So the flash is shown by data as well as by the readback (crevoke4b cannot show
  it on this seal: its candidate column equals 88db40027's).
- crevoke4b cand 16/16 AS PREDICTED (4(b) at N=2 across two builds of the same core logic); latency (2, 1174) and
  (4, 1130) cycles.
- g2: imagick PASS on all three arms. x264 NO READING on all three at the 3,600 s stage bound: gen alone took about
  2,128 s and run was still going about 1,430 s later, with no fault. On 904cd9c6b x264 took cause 24 at CINCOFFSET
  seconds into run. The g2b images (Amendment 10: per-stage bounds) replace g2 for the later sessions.

## Board 3, the a5c45a46d short session and its reruns (`a5c45a46d-board3-short.result-lines.txt`)

- Flashed, read back; every kit's loads under the seal's name.
- ldcexc2 fixed AS PREDICTED: ctl 0x0 0x0, arm SR 2 = 0x18 0x1 (one trap, no wedge). It discriminates only once the
  28fcbb646 control on board 2 REPRODUCES the wedge (pending at the time of writing).
- wapflu cand 1/1.
- crevoke4b cand 16/16 across the session and its rerun: every image read AS PREDICTED at least once.
- b5b2n 20/21 with the one MISS, 15-b5b2-cptag. That is reading 21's instrument artifact, already classified on dev
  (the diagnostic reads H-fetch; a seventh H-fetch here: warm 4/4, flag 1, miss 43, VALID).
- Three loads FAILED TO START (below). They carry no reading; their reruns are recorded as reruns, not as second readings.

## Board 3, the b42c8c644 session (`b42c8c644-board3.result-lines.txt`, cut while the session runs on)

- The seal's silicon Tier 1: flashed by b5b2n (booking 1544) and read back by name, then b5b2n 21/21 AS PREDICTED (the diag
  15-b5b2-cptag AS PREDICTED), crevoke4b cand 16/16 (1545), wapflu cand 1/1 (1546). They were graded against the simulated
  vectors of fca456c2aad8 (b5b2n) and 1258f0833b60 (crevoke4b, wapflu), which the graders re-check against those commits.
  Two kit loads failed to start and were rerun inside their kit (below).
- boot3 P1-P4 ok.
- SPEC: imagick PASS in all three g2b images, gcc PASS in all three g3 images, no cause-25 fault in any capture. x264
  has no reading on this seal: B5T-g2b's x264 ran past 12,180 s within its 14,400 s bound, and by the runtime lane's ruling
  B5TL0-g2b and B5TL1-g2b were STOPped after imagick's stage end (their notes and the stopper's log are in the file).

## Board 2, the b42c8c644 session (`b42c8c644-board2.result-lines.txt`, cut while the session runs on)

- The same Tier 1 on a second board: flashed by b5b2n (1550) and read back, b5b2n 21/21, crevoke4b 16/16 (1551), wapflu 1/1
  (1552), misscost both images AS PREDICTED (1553).
- Boot 13 (the Full final kernel): 4/4 as predicted.
- nested-cpymem is the nested lane's payload, graded by that lane (nested-workloads 098cae0f858b). The file carries its
  stage lines, every @@UP line and CPython's up0/up1 lines, with the labels that lane asked for.

## Board 1, the +Spatial seal 14f2ef81a (`14f2ef81a-board1.result-lines.txt`)

- Kit session: flashed by b5b2n and read back; b5b2n 20/21 (the one MISS is 15-b5b2-cptag, the instrument artifact; the diag
  reads H-fetch), crevoke4b 16/16, ldcexc2 fixed AS PREDICTED, wapflu 1/1, misscost informational (zero record misses).
  The b5b2n entry is the RE-GRADE of the same run: the session's own grade ERRORed in its diag section through a wrapper
  defect (the environment was set after the diag subprocess); the file says so.
- Boot 14 (the +Spatial final kernel), attempt 1: NO READING (infra), booking 1566. load_image completed, then GDB's
  `set $pc` timed out, so the image never ran; the once-only rerun was refused "board 1 busy" by attempt 1's still-live
  holder. Not a kernel reading. The rerun wrapper now waits for the board to be idle first. Boot 14 is re-armed by hand.

## Infra note: the intermittent JTAG load hang (boards 3 and 1)

- On board 3, 2026-11-25, `monitor load_image … 0x80000000 bin` hung for OpenOCD's full 120 s ("Timed out after 120s
  wait") on 3 of 64 bare-kit loads, on three different images: 16-b5b2-cptag2 (66,048 B), 04-b5b2-crevint-sh and
  15-b5b2-sbdrain-sret (53,184 B each).
- Every one of the other 61 loads took 0-1 s from load_image to continue, 94,144 B images included. Each failing image
  loaded fine at another time.
- Counted over every bare-kit load of 2026-11-25/26: board 3, 5 of 108 (3 of 64 in the a5c45a46d session and its reruns,
  2 of 44 in the b42c8c644 session); board 1, 2 of 50 (the 14f2ef81a kit session); board 2, 0 of 64. Board 1 also lost
  boot 14's first attempt to a JTAG timeout of a neighbouring shape: the 240 s Linux load completed and the next GDB
  command (`set $pc`) timed out.
- So it is image- and size-independent, and not specific to one board, though board 2 shows none. The image never ran: the
  runner had not reached `continue`, and every load is preceded by its own power cycle and `monitor reset halt`. A
  failed start therefore leaves no state, and the loads after it in a kit are valid.
- From the b42c8c644 sessions on, a load with that full signature (the OpenOCD message AND no UART byte AND no SR line)
  is rerun ONCE inside its kit (`run-*-rt.sh`), and a Linux boot with no OpenSBI banner and no UART byte is rerun once
  (`run-one-e-rt.sh`). Both attempts are logged; a second failure stays NO READING (infra). Any UART byte means the image
  ran, and that is never retried (the runtime lane's conditions, 2026-11-26; the branches are dry-tested, see below).

## Dry runs (`dry-runs/`)

The b42c8c644 orchestrator's new branches were run stubbed on recorded data before it was armed (board-run rule 8):
- the wait for the seal's grading;
- an armed, a late-armed, a never-armed and a sha-mismatched nested boot;
- the re-offer after ldcexc2;
- a non-PASS grade after the wait, and a chain that never ends;
- a failed start rerun inside a kit.

The test copy differs from the armed script only in the seal identity (a5c45a46d's in place of b42c8c644's, which did not
exist yet).
