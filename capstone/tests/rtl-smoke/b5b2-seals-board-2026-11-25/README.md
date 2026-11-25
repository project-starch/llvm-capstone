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

## Infra note: board 3's intermittent JTAG load hang

- On board 3, 2026-11-25, `monitor load_image … 0x80000000 bin` hung for OpenOCD's full 120 s ("Timed out after 120s
  wait") on 3 of 64 bare-kit loads, on three different images: 16-b5b2-cptag2 (66,048 B), 04-b5b2-crevint-sh and
  15-b5b2-sbdrain-sret (53,184 B each).
- Every one of the other 61 loads took 0-1 s from load_image to continue, 94,144 B images included. Each failing image
  loaded fine at another time. No such timeout appears in board 2's loads of the same day.
- So it is image- and size-independent, and specific to board 3 (its link or its probe). The image never ran: the
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
